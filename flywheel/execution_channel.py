"""HTTP service for nested block execution from inside containers.

Replaces the former ``BlockBridgeService``.  Routes requests to
the appropriate executor based on the request mode:

- ``mode=record`` → ``RecordExecutor``  (legacy sugar)
- Default (invoke) → ``ContainerExecutor``

Also exposes the **block-execution lifecycle API**:

- ``POST /execution/begin`` opens a logical block execution.
  Resolves declared inputs to their latest registered instances,
  opens a ledger row with ``status="running"``, and returns the
  execution_id and resolved input bindings.
- ``POST /execution/end/{id}`` closes a previously-opened
  execution.  On success, registers output artifacts atomically
  and updates the ledger row to ``"succeeded"``.  On failure,
  records the error and registers nothing.

This API is the runtime surface for tool-triggered logical block
executions (e.g., an MCP tool inside an agent container that
invokes a block via the ``flywheel.tool_block`` decorator).  See
``plans/flywheel-block-execution-refactor.md`` for the full design.

Preserves the exact HTTP protocol of the legacy modes so that
existing MCP servers inside containers need zero changes during
the migration.

Fires ``ExecutionEvent`` callbacks after each execution, replacing
the former ``on_record`` callback with a typed, executor-agnostic
event system.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any

from flywheel.artifact import ArtifactInstance, BlockExecution

if TYPE_CHECKING:
    from flywheel.blocks.manifest import ToolBlockManifest
from flywheel.executor import (
    ContainerExecutor,
    ExecutionEvent,
    RecordExecutor,
)
from flywheel.template import BlockDefinition, Template
from flywheel.workspace import Workspace


def _find_block(
    template: Template, block_name: str,
) -> BlockDefinition | None:
    """Look up a block definition by name in the template."""
    for block in template.blocks:
        if block.name == block_name:
            return block
    return None


def _resolve_inputs_for_begin(
    block_def: BlockDefinition,
    workspace: Workspace,
) -> tuple[dict[str, str], dict[str, str], dict[str, str | None]]:
    """Resolve declared block inputs to latest registered instances.

    For each input slot of the block, picks the most recently
    registered ``ArtifactInstance`` of that name and returns:

    - ``input_bindings``: ``{slot_name: artifact_id}``.
    - ``input_paths``: ``{slot_name: workspace-relative path}`` for
      the artifact directory (callers can resolve to absolute).
    - ``input_hashes``: ``{slot_name: None}`` placeholder, reserved
      for future content-hash freshness checks.

    Optional slots that have no available instance are silently
    skipped.  Required slots that have no instance raise
    ``ValueError``.
    """
    bindings: dict[str, str] = {}
    paths: dict[str, str] = {}
    hashes: dict[str, str | None] = {}

    for slot in block_def.inputs:
        instances = workspace.instances_for(slot.name)
        if not instances:
            if slot.optional:
                continue
            raise ValueError(
                f"Block {block_def.name!r} input slot "
                f"{slot.name!r} has no registered instances "
                f"in workspace"
            )
        latest = instances[-1]
        bindings[slot.name] = latest.id
        if latest.copy_path:
            paths[slot.name] = str(
                workspace.path / "artifacts" / latest.copy_path
            )
        else:
            paths[slot.name] = ""
        hashes[slot.name] = None

    return bindings, paths, hashes


class _ChannelRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler that delegates to executors."""

    # Assigned by ExecutionChannel before the server starts.
    workspace: Workspace
    _template: Template
    _record_executor: RecordExecutor
    _container_executor: ContainerExecutor
    _allowed_blocks: list[str] | None
    _counter: list  # [int]
    _invocation_count: list  # [int]
    _max_invocations: int | None
    _lock: threading.Lock
    _stopping: threading.Event
    _active_container: list  # [str | None]
    _service_id: str
    _on_execution: Callable[[ExecutionEvent], None] | None
    _agent_workspace_dir: str | None
    # Manifest-driven tool→block table.  Empty dict means "no
    # manifest enforcement"; callers are accepted regardless.
    _invocation_table: dict[tuple[str, str], str]

    def do_POST(self):  # noqa: N802
        """Handle a POST request to execute a block."""
        if self._stopping.is_set():
            self._send_json(503, {
                "ok": False,
                "error_type": "stopping",
                "message": "Execution channel is shutting down",
            })
            return

        # Lifecycle API endpoints — separate from the legacy
        # block_name/mode payloads.
        path = self.path.rstrip("/")
        if path == "/execution/begin":
            self._handle_lifecycle_begin()
            return
        if path.startswith("/execution/end/"):
            execution_id = path[len("/execution/end/"):]
            self._handle_lifecycle_end(execution_id)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError) as e:
            self._send_json(400, {
                "ok": False,
                "error_type": "invalid_json",
                "message": str(e),
            })
            return

        block_name = payload.get("block_name", "")
        mode = payload.get("mode", "invoke")

        if mode == "record":
            self._handle_record(payload, block_name)
        else:
            self._handle_invoke(payload, block_name)

    def _has_any_binding_for(self, mcp_server: str) -> bool:
        """Whether the channel's manifest declares this MCP server.

        Used by manifest enforcement: a call from an MCP server
        the channel has *no* manifest for is allowed through
        without binding checks (back-compat during the Phase 3
        rollout).  Calls from a *known* MCP server must match.
        """
        return any(
            srv == mcp_server
            for (srv, _tool) in self._invocation_table)

    def _read_json_body(self) -> tuple[dict | None, str | None]:
        """Read and JSON-parse the request body.

        Returns a (payload, error) tuple where exactly one is set.
        """
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
        except Exception as e:
            return None, f"read failed: {e}"
        if not body:
            return {}, None
        try:
            return json.loads(body), None
        except (json.JSONDecodeError, ValueError) as e:
            return None, str(e)

    def _handle_lifecycle_begin(self) -> None:
        """Handle ``POST /execution/begin``.

        Opens a logical block execution: resolves declared input
        artifacts to their latest registered instances, allocates a
        per-execution scratch directory, opens a ledger row with
        ``status="running"``, and returns the execution_id and
        resolved input bindings.
        """
        payload, err = self._read_json_body()
        if err is not None:
            self._send_json(400, {
                "ok": False,
                "error_type": "invalid_json",
                "message": err,
            })
            return
        assert payload is not None

        block_name = payload.get("block", "")
        if not block_name:
            self._send_json(400, {
                "ok": False,
                "error_type": "missing_field",
                "message": "begin request requires 'block'",
            })
            return

        if (self._allowed_blocks
                and block_name not in self._allowed_blocks):
            self._send_json(200, {
                "ok": False,
                "retryable": False,
                "error_type": "block_not_allowed",
                "message": (
                    f"Block {block_name!r} not in allowed list: "
                    f"{self._allowed_blocks}"),
            })
            return

        block_def = _find_block(self._template, block_name)
        if block_def is None:
            self._send_json(200, {
                "ok": False,
                "retryable": False,
                "error_type": "unknown_block",
                "message": (
                    f"Block {block_name!r} not found in template"),
            })
            return

        params = payload.get("params") or {}
        caller = payload.get("caller")
        parent_execution_id = payload.get("parent_execution_id")
        runner = payload.get("runner")  # caller-declared, optional

        # Manifest enforcement.  When the channel was started with
        # any tool-to-block manifests, every begin call from a
        # known MCP server must match the manifest's binding for
        # that tool.  Calls without a caller (e.g., direct CLI
        # invocations, tests) bypass enforcement.  Calls from
        # unknown MCP servers are also accepted, since not every
        # MCP server has to ship a manifest yet during the Phase 3
        # rollout.
        if self._invocation_table and isinstance(caller, dict):
            mcp_server = caller.get("mcp_server")
            tool_name = caller.get("tool")
            if (mcp_server and tool_name
                    and self._has_any_binding_for(mcp_server)):
                expected = self._invocation_table.get(
                    (mcp_server, tool_name))
                if expected is None:
                    self._send_json(200, {
                        "ok": False,
                        "retryable": False,
                        "error_type": "manifest_violation",
                        "message": (
                            f"Tool {tool_name!r} of MCP server "
                            f"{mcp_server!r} is not declared in "
                            f"the tool-to-block manifest; in-"
                            f"container code may not invent "
                            f"new artifact effects"),
                    })
                    return
                if expected != block_name:
                    self._send_json(200, {
                        "ok": False,
                        "retryable": False,
                        "error_type": "manifest_violation",
                        "message": (
                            f"Tool {tool_name!r} of MCP server "
                            f"{mcp_server!r} is bound to block "
                            f"{expected!r} by the manifest, but "
                            f"the request asked for block "
                            f"{block_name!r}"),
                    })
                    return

        # Resolve input artifact bindings.  Latest-instance for each
        # declared input slot.  Missing required slots are an error.
        try:
            input_bindings, input_paths, input_hashes = (
                _resolve_inputs_for_begin(
                    block_def, self.workspace,
                ))
        except ValueError as e:
            self._send_json(200, {
                "ok": False,
                "retryable": False,
                "error_type": "missing_input",
                "message": str(e),
            })
            return

        # Open the ledger row.  Allocate a scratch dir for this
        # execution under the workspace.
        with self._lock:
            self._counter[0] += 1
            request_id = (
                f"{self._service_id}_{self._counter[0]:04d}")

        execution_id = self.workspace.generate_execution_id()
        scratch_dir = (
            self.workspace.path / "execution_scratch" / execution_id
        )
        scratch_dir.mkdir(parents=True, exist_ok=True)

        execution = BlockExecution(
            id=execution_id,
            block_name=block_name,
            started_at=datetime.now(UTC),
            status="running",
            input_bindings=input_bindings,
            output_bindings={},
            image=block_def.image,
            parent_execution_id=parent_execution_id,
            runner=runner,
            caller=caller,
            params=params or None,
        )
        try:
            self.workspace.add_execution(execution)
            self.workspace.save()
        except Exception as e:
            shutil.rmtree(scratch_dir, ignore_errors=True)
            self._send_json(500, {
                "ok": False,
                "error_type": "internal_error",
                "message": f"failed to record execution: {e}",
            })
            return

        self._send_json(200, {
            "request_id": request_id,
            "ok": True,
            "execution_id": execution_id,
            "input_bindings": input_bindings,
            "input_paths": input_paths,
            "input_hashes": input_hashes,
            "scratch_dir": str(scratch_dir),
            "parent_execution_id": parent_execution_id,
        })

    def _handle_lifecycle_end(self, execution_id: str) -> None:
        """Handle ``POST /execution/end/{id}``.

        Closes a previously-opened execution.  On ``status="ok"``,
        registers any supplied output artifacts atomically and marks
        the ledger row succeeded.  On ``status="failed"``, records
        the error and registers nothing (atomicity).
        """
        if not execution_id:
            self._send_json(400, {
                "ok": False,
                "error_type": "missing_field",
                "message": "end request requires execution_id in path",
            })
            return

        payload, err = self._read_json_body()
        if err is not None:
            self._send_json(400, {
                "ok": False,
                "error_type": "invalid_json",
                "message": err,
            })
            return
        assert payload is not None

        status_in = payload.get("status", "ok")
        outputs_data = payload.get("outputs") or {}
        elapsed_s = payload.get("elapsed_s")
        error = payload.get("error")

        # Look up the existing execution row.  Must be in "running".
        if execution_id not in self.workspace.executions:
            self._send_json(404, {
                "ok": False,
                "error_type": "unknown_execution",
                "message": (
                    f"Execution {execution_id!r} not found "
                    f"(begin may have failed)"),
            })
            return

        existing = self.workspace.executions[execution_id]
        if existing.status != "running":
            self._send_json(400, {
                "ok": False,
                "error_type": "not_running",
                "message": (
                    f"Execution {execution_id!r} has status "
                    f"{existing.status!r}, expected 'running'"),
            })
            return

        block_def = _find_block(
            self._template, existing.block_name)
        if block_def is None:
            # Should be impossible — begin already validated.
            self._send_json(500, {
                "ok": False,
                "error_type": "internal_error",
                "message": (
                    f"Block {existing.block_name!r} disappeared "
                    f"between begin and end"),
            })
            return

        finished_at = datetime.now(UTC)

        # Failure path: record error, no outputs registered.
        if status_in not in ("ok", "succeeded"):
            failed = BlockExecution(
                id=existing.id,
                block_name=existing.block_name,
                started_at=existing.started_at,
                finished_at=finished_at,
                status="failed",
                input_bindings=existing.input_bindings,
                output_bindings={},
                exit_code=existing.exit_code,
                elapsed_s=elapsed_s,
                image=existing.image,
                stop_reason=existing.stop_reason,
                predecessor_id=existing.predecessor_id,
                parent_execution_id=existing.parent_execution_id,
                runner=existing.runner,
                caller=existing.caller,
                params=existing.params,
                error=error or "execution failed",
            )
            self._replace_execution(failed)
            self.workspace.save()
            self._cleanup_scratch(execution_id)

            self._send_json(200, {
                "ok": True,
                "execution_id": execution_id,
                "status": "failed",
                "error": failed.error,
            })

            if self._on_execution is not None:
                with contextlib.suppress(Exception):
                    self._on_execution(ExecutionEvent(
                        executor_type=existing.runner or "lifecycle",
                        block_name=existing.block_name,
                        execution_id=execution_id,
                        status="failed",
                        output_bindings={},
                    ))
            return

        # Success path: atomically register all declared outputs
        # supplied in the payload.  Buffered until we have written
        # everything; on any error, roll back partial writes.
        registered: list[tuple[str, str]] = []  # (slot_name, aid)
        try:
            for slot in block_def.outputs:
                data = outputs_data.get(slot.name)
                if data is None:
                    continue

                aid = self.workspace.generate_artifact_id(slot.name)
                output_dir = (
                    self.workspace.path / "artifacts" / aid)
                output_dir.mkdir(parents=True)

                output_file = output_dir / f"{slot.name}.json"
                output_file.write_text(
                    json.dumps(data, separators=(",", ":")),
                    encoding="utf-8",
                )

                instance = ArtifactInstance(
                    id=aid,
                    name=slot.name,
                    kind="copy",
                    created_at=finished_at,
                    produced_by=execution_id,
                    copy_path=aid,
                )
                self.workspace.add_artifact(instance)
                registered.append((slot.name, aid))
        except Exception as e:
            # Roll back: remove any artifacts written this attempt.
            for _, aid in registered:
                aid_dir = self.workspace.path / "artifacts" / aid
                shutil.rmtree(aid_dir, ignore_errors=True)
                self.workspace.artifacts.pop(aid, None)
            self._send_json(500, {
                "ok": False,
                "error_type": "output_write_failed",
                "message": str(e),
            })
            return

        output_bindings = {name: aid for name, aid in registered}

        succeeded = BlockExecution(
            id=existing.id,
            block_name=existing.block_name,
            started_at=existing.started_at,
            finished_at=finished_at,
            status="succeeded",
            input_bindings=existing.input_bindings,
            output_bindings=output_bindings,
            exit_code=existing.exit_code,
            elapsed_s=elapsed_s,
            image=existing.image,
            stop_reason=existing.stop_reason,
            predecessor_id=existing.predecessor_id,
            parent_execution_id=existing.parent_execution_id,
            runner=existing.runner,
            caller=existing.caller,
            params=existing.params,
        )
        self._replace_execution(succeeded)
        self.workspace.save()
        self._cleanup_scratch(execution_id)

        # Build response with name → artifact_id mapping for
        # convenience, plus instance counts (per-name) so callers
        # can derive sequence position without a separate query.
        instance_counts = {
            f"{name}_instance_count": len(
                self.workspace.instances_for(name))
            for name in output_bindings
        }
        artifact_id_map = {
            f"{name}_artifact_id": aid
            for name, aid in output_bindings.items()
        }
        self._send_json(200, {
            "ok": True,
            "execution_id": execution_id,
            "status": "succeeded",
            "output_bindings": output_bindings,
            **artifact_id_map,
            **instance_counts,
        })

        if self._on_execution is not None:
            with contextlib.suppress(Exception):
                self._on_execution(ExecutionEvent(
                    executor_type=existing.runner or "lifecycle",
                    block_name=existing.block_name,
                    execution_id=execution_id,
                    status="succeeded",
                    output_bindings=output_bindings,
                    outputs_data=outputs_data,
                ))

    def _replace_execution(
        self, execution: BlockExecution,
    ) -> None:
        """Replace an existing execution row in the workspace.

        BlockExecution is frozen, so transitions from ``"running"``
        to terminal states require constructing a new record and
        swapping it in.  Bypasses ``add_execution`` since the ID
        already exists by design.
        """
        with self.workspace._lock:  # noqa: SLF001
            self.workspace.executions[execution.id] = execution

    def _cleanup_scratch(self, execution_id: str) -> None:
        """Remove the scratch directory for a finished execution."""
        scratch_dir = (
            self.workspace.path / "execution_scratch" / execution_id
        )
        shutil.rmtree(scratch_dir, ignore_errors=True)

    def _handle_record(
        self, payload: dict, block_name: str,
    ) -> None:
        """Handle a record-mode request via RecordExecutor."""
        if not block_name:
            self._send_json(400, {
                "ok": False,
                "error_type": "missing_field",
                "message": "Record request requires 'block_name'",
            })
            return

        with self._lock:
            self._counter[0] += 1
            request_id = (
                f"{self._service_id}_{self._counter[0]:04d}")

        try:
            handle = self._record_executor.launch(
                block_name=block_name,
                workspace=self.workspace,
                input_bindings=payload.get("inputs", {}),
                outputs_data=payload.get("outputs", {}),
                elapsed_s=payload.get("elapsed_s"),
                allowed_blocks=self._allowed_blocks,
            )
            result = handle.wait()
        except (ValueError, FileNotFoundError) as e:
            # Map executor errors to the old bridge response format.
            error_type = "internal_error"
            msg = str(e)
            if "not found in template" in msg:
                error_type = "unknown_block"
            elif "not a record block" in msg:
                error_type = "not_record_block"
            elif "not in allowed" in msg:
                error_type = "block_not_allowed"
            elif "not provided" in msg:
                error_type = "missing_input"
            elif "slot expects" in msg:
                error_type = "slot_mismatch"
            elif "not found" in msg:
                error_type = "unknown_artifact"

            self._send_json(200, {
                "request_id": request_id,
                "ok": False,
                "retryable": False,
                "error_type": error_type,
                "message": msg,
            })
            return
        except Exception as e:
            self._send_json(500, {
                "request_id": request_id,
                "ok": False,
                "error_type": "internal_error",
                "message": str(e),
            })
            return

        # Build response in the old bridge format.
        instance_counts = {}
        for name in result.output_bindings:
            instance_counts[f"{name}_instance_count"] = len(
                self.workspace.instances_for(name))

        response = {
            "request_id": request_id,
            "ok": True,
            "execution_id": result.execution_id,
            **{f"{name}_artifact_id": aid
               for name, aid in result.output_bindings.items()},
            **instance_counts,
        }
        self._send_json(200, response)

        # Fire execution event.
        if self._on_execution is not None:
            with contextlib.suppress(Exception):
                event = ExecutionEvent(
                    executor_type="record",
                    block_name=block_name,
                    execution_id=result.execution_id,
                    status=result.status,
                    output_bindings=result.output_bindings,
                    outputs_data=payload.get("outputs", {}),
                )
                self._on_execution(event)

    def _handle_invoke(
        self, payload: dict, block_name: str,
    ) -> None:
        """Handle an invoke-mode request via ContainerExecutor."""
        artifact_path = payload.get("artifact_path", "")
        if not block_name or not artifact_path:
            self._send_json(400, {
                "ok": False,
                "error_type": "missing_field",
                "message": (
                    "Request requires 'block_name' and "
                    "'artifact_path' fields"),
            })
            return

        with self._lock:
            self._counter[0] += 1
            request_id = (
                f"{self._service_id}_{self._counter[0]:04d}")

            if self._max_invocations is not None and (
                    self._invocation_count[0]
                    >= self._max_invocations):
                self._send_json(200, {
                    "request_id": request_id,
                    "ok": False,
                    "retryable": False,
                    "error_type": "invocation_limit_exceeded",
                    "message": "Block invocation limit reached",
                    "max_invocations": self._max_invocations,
                    "invocations_used": self._invocation_count[0],
                    "remaining": 0,
                })
                return

            self._invocation_count[0] += 1
            invocations_used = self._invocation_count[0]

        try:
            handle = self._container_executor.launch(
                block_name=block_name,
                workspace=self.workspace,
                input_bindings={},
                artifact_path=artifact_path,
                allowed_blocks=self._allowed_blocks,
                stopping=self._stopping,
                active_container=self._active_container,
                agent_workspace_dir=self._agent_workspace_dir,
            )
            result = handle.wait()
        except (ValueError, FileNotFoundError) as e:
            error_type = "internal_error"
            msg = str(e)
            if "not found in template" in msg:
                error_type = "unknown_block"
            elif "not in allowed" in msg:
                error_type = "block_not_allowed"
            elif "escapes workspace" in msg:
                error_type = "path_traversal"
            elif "Artifact not found" in msg:
                error_type = "file_not_found"
            elif "no input slots" in msg:
                error_type = "no_inputs"
            elif "shutting down" in msg:
                error_type = "cancelled"

            self._send_json(200 if error_type != "internal_error"
                            else 500, {
                "request_id": request_id,
                "ok": False,
                "retryable": False,
                "error_type": error_type,
                "message": msg,
            })
            return
        except Exception as e:
            self._send_json(500, {
                "request_id": request_id,
                "ok": False,
                "error_type": "internal_error",
                "message": str(e),
            })
            return

        # Build response in the old bridge format.
        response: dict[str, Any] = {
            "request_id": request_id,
            "ok": result.exit_code == 0,
            "exit_code": result.exit_code,
            "elapsed_s": round(result.elapsed_s, 1),
            "block_name": block_name,
            "execution_id": result.execution_id,
        }

        # Read scores.json from output artifacts.
        for aid in result.output_bindings.values():
            scores_file = (
                self.workspace.path / "artifacts" / aid
                / "scores.json")
            if scores_file.exists():
                try:
                    scores = json.loads(
                        scores_file.read_text(encoding="utf-8"))
                    response["score"] = scores.get(
                        "score", scores.get("mean"))
                    response.update(
                        {k: v for k, v in scores.items()
                         if k != "fights_won"})
                except Exception:
                    pass
                break

        if result.exit_code == 2:
            for aid in result.output_bindings.values():
                scores_file = (
                    self.workspace.path / "artifacts" / aid
                    / "scores.json")
                if scores_file.exists():
                    try:
                        scores = json.loads(
                            scores_file.read_text(encoding="utf-8"))
                        response["error_type"] = scores.get(
                            "abort_reason", "aborted")
                        response["message"] = scores.get(
                            "abort_message", "Block aborted")
                    except Exception:
                        pass
                    break
            response["ok"] = False
        elif result.exit_code != 0:
            response["error_type"] = "container_error"
            response["message"] = (
                f"Container exited with code {result.exit_code}")

        if self._max_invocations is not None:
            response["max_invocations"] = self._max_invocations
            response["invocations_used"] = invocations_used
            response["remaining"] = (
                self._max_invocations - invocations_used)

        self._send_json(200, response)

        # Fire execution event.
        if self._on_execution is not None:
            with contextlib.suppress(Exception):
                event = ExecutionEvent(
                    executor_type="container",
                    block_name=block_name,
                    execution_id=result.execution_id,
                    status=result.status,
                    output_bindings=result.output_bindings,
                )
                self._on_execution(event)

    def do_GET(self):  # noqa: N802
        """Handle GET requests to query artifact instances.

        ``GET /<artifact_name>`` returns all instances.
        ``GET /<artifact_name>/latest`` returns only the last.
        ``GET /<artifact_name>/count`` returns ``{"count": N}``.
        """
        path = self.path.strip("/")
        parts = path.split("/", 1)
        artifact_name = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        if not artifact_name:
            self._send_json(400, {
                "ok": False,
                "message": "Artifact name required in path",
            })
            return

        instances = self.workspace.instances_for(artifact_name)

        if suffix == "count":
            self._send_json(200, {
                "ok": True,
                "count": len(instances),
            })
            return

        if suffix == "latest":
            instances = instances[-1:] if instances else []

        result = []
        for inst in instances:
            entry: dict[str, Any] = {
                "id": inst.id,
                "created_at": inst.created_at.isoformat()
                if inst.created_at else None,
            }
            if inst.copy_path:
                artifact_dir = (
                    self.workspace.path / "artifacts"
                    / inst.copy_path
                )
                json_files = sorted(artifact_dir.glob("*.json"))
                if json_files:
                    try:
                        entry["data"] = json.loads(
                            json_files[0].read_text(
                                encoding="utf-8"))
                    except Exception:
                        entry["data"] = None
            result.append(entry)

        self._send_json(200, {"ok": True, "instances": result})

    def _send_json(self, status: int, data: dict) -> None:
        """Send a JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        """Suppress default stderr logging."""


def _kill_container(name: str) -> None:
    """Kill a Docker container by name. Ignores errors."""
    with contextlib.suppress(Exception):
        subprocess.run(
            ["docker", "kill", name],
            capture_output=True, timeout=10)


class ExecutionChannel:
    """HTTP service for nested block execution from containers.

    Replaces ``BlockBridgeService``.  Routes requests to executors
    based on the request mode.  Preserves the same HTTP protocol.

    Args:
        template: Template containing block definitions.
        workspace: The flywheel workspace.
        overrides: CLI flag overrides for container blocks.
        allowed_blocks: If set, only these block names allowed.
        max_invocations: Maximum container invocations (None =
            unlimited).
        host: Host to bind to.
        port: Port to listen on (0 = auto-assign).
        on_execution: Callback fired after each successful
            execution.  Receives an ``ExecutionEvent``.  Runs in
            the HTTP handler thread — must be thread-safe.
        on_record: Deprecated alias for ``on_execution``.  Wraps
            the callback to accept ``(block_name, outputs)``
            signature.
    """

    def __init__(
        self,
        template: Template,
        workspace: Workspace,
        overrides: dict[str, Any] | None = None,
        allowed_blocks: list[str] | None = None,
        max_invocations: int | None = None,
        host: str = "0.0.0.0",
        port: int = 0,
        on_execution: Callable[[ExecutionEvent], None] | None = None,
        on_record: Callable[[str, dict], None] | None = None,
        agent_workspace_dir: str | None = None,
        manifests: "list[ToolBlockManifest] | None" = None,
    ):
        """Initialize the execution channel.

        ``manifests`` is the list of tool-to-block manifests
        (one per MCP server) that this channel is allowed to
        serve.  When supplied, ``/execution/begin`` requests with
        a ``caller`` field are validated against the resulting
        ``(mcp_server, tool) → block`` table; mismatches are
        rejected with ``error_type="manifest_violation"``.

        When ``manifests`` is ``None`` or empty, the channel
        accepts any caller (back-compat with non-manifest tools).
        """
        from flywheel.blocks.manifest import (
            build_invocation_table,
        )

        self.template = template
        self.workspace = workspace
        self.overrides = overrides
        self.allowed_blocks = allowed_blocks
        self.max_invocations = max_invocations
        self.host = host
        self.port = port
        self.agent_workspace_dir = agent_workspace_dir
        self.manifests = list(manifests) if manifests else []
        self._invocation_table = (
            build_invocation_table(self.manifests))

        # on_record backward compat: wrap into on_execution.
        if on_execution is not None:
            self._on_execution = on_execution
        elif on_record is not None:
            def _wrapped(event: ExecutionEvent) -> None:
                on_record(event.block_name,
                          event.outputs_data or {})
            self._on_execution = _wrapped
        else:
            self._on_execution = None

        self._record_executor = RecordExecutor(template)
        self._container_executor = ContainerExecutor(
            template, overrides)

        self._service_id = secrets.token_hex(4)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._active_container: list = [None]
        self._lock = threading.Lock()

    def start(self) -> int:
        """Start the HTTP server in a background thread.

        Returns:
            The port the server is listening on.
        """
        lock = self._lock
        stopping = self._stopping
        active_container = self._active_container
        service_id = self._service_id
        on_execution = self._on_execution
        record_executor = self._record_executor
        container_executor = self._container_executor

        agent_ws_dir = self.agent_workspace_dir
        invocation_table = dict(self._invocation_table)

        class Handler(_ChannelRequestHandler):
            workspace = self.workspace
            _template = self.template
            _record_executor = record_executor
            _container_executor = container_executor
            _allowed_blocks = self.allowed_blocks
            _counter = [0]
            _invocation_count = [0]
            _max_invocations = self.max_invocations
            _lock = lock
            _stopping = stopping
            _active_container = active_container
            _service_id = service_id
            _on_execution = (
                staticmethod(on_execution)
                if on_execution else None
            )
            _agent_workspace_dir = agent_ws_dir
            _invocation_table = invocation_table

        self._server = HTTPServer((self.host, self.port), Handler)
        self.port = self._server.server_address[1]

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()

        print(
            f"  [execution-channel] listening on "
            f"{self.host}:{self.port}")
        return self.port

    def stop(self, timeout: float = 30.0) -> None:
        """Shut down the server and kill any in-flight container."""
        self._stopping.set()

        with self._lock:
            container_name = self._active_container[0]
        if container_name:
            print(
                f"  [execution-channel] killing {container_name}")
            _kill_container(container_name)

        if self._server:
            self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._server = None
        self._thread = None

    @property
    def url(self) -> str:
        """The base URL of the running server."""
        return f"http://{self.host}:{self.port}"
