"""HTTP bridge for nested block executions.

An agent running inside a Docker container may need to trigger
other block executions in the same workspace — for example,
evaluating an artifact it just produced. This module provides an
HTTP service that receives such requests and dispatches them as
real flywheel block executions, reading the block definition from
the template.

Two modes are supported:

- **Invoke mode** (default): launches a Docker container for the
  block, mounts input/output artifact directories, and records
  the execution.
- **Record mode**: creates artifacts and execution records without
  launching a container. Used for provenance tracking of actions
  that already happened (e.g., game steps via a REST API).
  Record-mode blocks use the ``__record__`` sentinel image.

This is a generic flywheel capability. The blocks being invoked
and what they do (evaluation, validation, compilation, etc.) are
project-specific — defined by the template, not by flywheel.

Deployment pattern: one service per agent step. Each agent step
creates its own ``BlockBridgeService`` instance with its own port,
invocation budget, and service ID. Parallel agents get independent
services.

``BlockBridgeService`` accepts an optional ``on_record`` callback,
fired after each successful record-mode invocation with the block
name and output data. This enables the host to react in real-time
to artifacts created by the agent (e.g., killing the agent on a
prediction mismatch).
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
from typing import Any

from flywheel.artifact import ArtifactInstance, BlockExecution
from flywheel.container import ContainerConfig, run_container
from flywheel.template import BlockDefinition, Template
from flywheel.workspace import Workspace


class _BridgeRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for nested block execution requests.

    Each POST triggers a block execution in the workspace using the
    block definition from the template.
    """

    # Assigned by BlockBridgeService before the server starts.
    template: Template
    workspace: Workspace
    overrides: dict[str, Any] | None
    allowed_blocks: list[str] | None
    _counter: list  # [int]
    _invocation_count: list  # [int]
    _max_invocations: int | None
    _lock: threading.Lock
    _stopping: threading.Event
    _active_container: list  # [str | None]
    _service_id: str
    _on_record: Callable[[str, dict], None] | None
    _agent_workspace_dir: str | None

    def do_POST(self):  # noqa: N802
        """Handle a POST request to invoke a block execution."""
        if self._stopping.is_set():
            self._send_json(503, {
                "ok": False,
                "error_type": "stopping",
                "message": "Block bridge is shutting down",
            })
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        print(f"  [block-bridge] received {content_length} bytes: "
              f"{body[:200]!r}", flush=True)

        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [block-bridge] JSON parse error: {e}",
                  flush=True)
            self._send_json(400, {
                "ok": False,
                "error_type": "invalid_json",
                "message": str(e),
            })
            return

        block_name = payload.get("block_name", "")
        mode = payload.get("mode", "invoke")

        if mode == "record":
            # Record mode: create artifacts without launching a
            # container.  Used for provenance tracking of actions
            # that already happened (e.g., game steps).
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
                response = _process_record_invocation(
                    request_id=request_id,
                    block_name=block_name,
                    inputs=payload.get("inputs", {}),
                    outputs=payload.get("outputs", {}),
                    elapsed_s=payload.get("elapsed_s"),
                    template=self.template,
                    workspace=self.workspace,
                    allowed_blocks=self.allowed_blocks,
                )
            except Exception as e:
                self._send_json(500, {
                    "request_id": request_id,
                    "ok": False,
                    "error_type": "internal_error",
                    "message": str(e),
                })
                return

            self._send_json(200, response)

            # Notify caller of the successful record.
            if self._on_record is not None and response.get("ok"):
                with contextlib.suppress(Exception):
                    self._on_record(
                        block_name, payload.get("outputs", {}))

            return

        # Invoke mode (default): launch a container.
        artifact_path = payload.get("artifact_path", "")
        if not block_name or not artifact_path:
            print(f"  [block-bridge] missing fields: "
                  f"block_name={block_name!r} "
                  f"artifact_path={artifact_path!r}", flush=True)
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
            request_id = f"{self._service_id}_{self._counter[0]:04d}"

            if self._max_invocations is not None and (
                    self._invocation_count[0] >= self._max_invocations):
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
            response = _process_block_invocation(
                request_id=request_id,
                block_name=block_name,
                artifact_rel=artifact_path,
                template=self.template,
                workspace=self.workspace,
                overrides=self.overrides,
                allowed_blocks=self.allowed_blocks,
                stopping=self._stopping,
                active_container=self._active_container,
                agent_workspace_dir=self._agent_workspace_dir,
            )
        except Exception as e:
            self._send_json(500, {
                "request_id": request_id,
                "ok": False,
                "error_type": "internal_error",
                "message": str(e),
            })
            return

        if self._max_invocations is not None:
            response["max_invocations"] = self._max_invocations
            response["invocations_used"] = invocations_used
            response["remaining"] = (
                self._max_invocations - invocations_used)

        self._send_json(200, response)

    def do_GET(self):  # noqa: N802
        """Handle GET requests to query artifact instances.

        ``GET /<artifact_name>`` returns all instances of that
        artifact as a JSON array, ordered by creation time.  Each
        element contains the instance metadata and the contents of
        the first JSON file in the artifact directory.

        ``GET /<artifact_name>/latest`` returns only the last
        instance.

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
                    self.workspace.path / "artifacts" / inst.copy_path
                )
                json_files = sorted(artifact_dir.glob("*.json"))
                if json_files:
                    try:
                        entry["data"] = json.loads(
                            json_files[0].read_text(encoding="utf-8"))
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


def _find_block(template: Template, block_name: str) -> BlockDefinition | None:
    """Look up a block definition by name in the template."""
    for block in template.blocks:
        if block.name == block_name:
            return block
    return None


RECORD_SENTINEL = "__record__"


def _process_record_invocation(
    request_id: str,
    block_name: str,
    inputs: dict[str, str],
    outputs: dict[str, Any],
    elapsed_s: float | None,
    template: Template,
    workspace: Workspace,
    allowed_blocks: list[str] | None = None,
) -> dict:
    """Record artifacts and a block execution without launching a container.

    Used for provenance tracking of actions that already happened
    (e.g., game steps executed via a REST API).  Each output value
    is written as a JSON file in a new artifact directory.

    Args:
        request_id: Unique ID for this request.
        block_name: Name of the block (must have ``__record__`` image).
        inputs: Maps input slot names to existing artifact instance IDs.
        outputs: Maps output slot names to JSON-serializable data.
        elapsed_s: Wall-clock time of the recorded action.
        template: The template containing block definitions.
        workspace: The flywheel workspace (modified in place).
        allowed_blocks: If set, only these block names can be invoked.

    Returns:
        Response dict with ``ok``, artifact IDs, and execution ID.
    """
    if allowed_blocks and block_name not in allowed_blocks:
        return {
            "request_id": request_id,
            "ok": False,
            "retryable": False,
            "error_type": "block_not_allowed",
            "message": (
                f"Block {block_name!r} is not in the allowed "
                f"list: {allowed_blocks}"),
        }

    block_def = _find_block(template, block_name)
    if block_def is None:
        return {
            "request_id": request_id,
            "ok": False,
            "retryable": False,
            "error_type": "unknown_block",
            "message": f"Block {block_name!r} not found in template",
        }

    if block_def.image != RECORD_SENTINEL:
        return {
            "request_id": request_id,
            "ok": False,
            "retryable": False,
            "error_type": "not_record_block",
            "message": (
                f"Block {block_name!r} is not a record block "
                f"(image is {block_def.image!r}, expected "
                f"{RECORD_SENTINEL!r})"),
        }

    # Validate input artifact IDs exist in workspace and match the slot.
    input_bindings: dict[str, str] = {}
    for slot in block_def.inputs:
        artifact_id = inputs.get(slot.name, "")
        if artifact_id:
            if artifact_id not in workspace.artifacts:
                return {
                    "request_id": request_id,
                    "ok": False,
                    "retryable": False,
                    "error_type": "unknown_artifact",
                    "message": (
                        f"Input artifact {artifact_id!r} not found "
                        f"in workspace"),
                }
            actual_name = workspace.artifacts[artifact_id].name
            if actual_name != slot.name:
                return {
                    "request_id": request_id,
                    "ok": False,
                    "retryable": False,
                    "error_type": "slot_mismatch",
                    "message": (
                        f"Input artifact {artifact_id!r} has name "
                        f"{actual_name!r} but slot expects "
                        f"{slot.name!r}"),
                }
            input_bindings[slot.name] = artifact_id
        elif not slot.optional:
            # Try latest instance for this slot name.
            instances = workspace.instances_for(slot.name)
            if instances:
                input_bindings[slot.name] = instances[-1].id
            else:
                return {
                    "request_id": request_id,
                    "ok": False,
                    "retryable": False,
                    "error_type": "missing_input",
                    "message": (
                        f"Required input {slot.name!r} not provided "
                        f"and no instances exist in workspace"),
                }

    # Write output data as artifact instances.
    execution_id = workspace.generate_execution_id()
    started_at = datetime.now(UTC)
    output_bindings: dict[str, str] = {}

    for slot in block_def.outputs:
        data = outputs.get(slot.name)
        if data is None:
            continue

        artifact_id = workspace.generate_artifact_id(slot.name)
        output_dir = workspace.path / "artifacts" / artifact_id
        output_dir.mkdir(parents=True)

        output_file = output_dir / f"{slot.name}.json"
        output_file.write_text(
            json.dumps(data, separators=(",", ":")),
            encoding="utf-8",
        )

        instance = ArtifactInstance(
            id=artifact_id,
            name=slot.name,
            kind="copy",
            created_at=started_at,
            produced_by=execution_id,
            copy_path=artifact_id,
        )
        workspace.add_artifact(instance)
        output_bindings[slot.name] = artifact_id

    # Record the execution.
    execution = BlockExecution(
        id=execution_id,
        block_name=block_name,
        started_at=started_at,
        finished_at=started_at,
        status="succeeded",
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        elapsed_s=elapsed_s,
        image=RECORD_SENTINEL,
    )
    workspace.add_execution(execution)
    workspace.save()

    print(f"  [block-bridge] {request_id}: recorded {block_name} "
          f"({len(output_bindings)} outputs)", flush=True)

    # Include instance counts for each output artifact name so
    # callers can derive sequence positions without a separate query.
    instance_counts = {}
    for slot in block_def.outputs:
        if slot.name in output_bindings:
            instance_counts[f"{slot.name}_instance_count"] = len(
                workspace.instances_for(slot.name))

    return {
        "request_id": request_id,
        "ok": True,
        "execution_id": execution_id,
        **{f"{name}_artifact_id": aid
           for name, aid in output_bindings.items()},
        **instance_counts,
    }


def _process_block_invocation(
    request_id: str,
    block_name: str,
    artifact_rel: str,
    template: Template,
    workspace: Workspace,
    overrides: dict[str, Any] | None = None,
    allowed_blocks: list[str] | None = None,
    stopping: threading.Event | None = None,
    active_container: list | None = None,
    agent_workspace_dir: str | None = None,
) -> dict:
    """Process a nested block execution request.

    Imports the provided artifact, looks up the block definition in
    the template, builds the container config from the block's
    declared image and slots, runs it, and records the execution
    and output artifacts in the workspace.

    Args:
        request_id: Unique ID for this request.
        block_name: Name of the block to execute (from the template).
        artifact_rel: Path to the input artifact, relative to the
            agent's workspace directory.
        template: The template containing block definitions.
        workspace: The flywheel workspace (modified in place).
        overrides: CLI flag overrides for the container.
        allowed_blocks: If set, only these block names can be invoked.
        stopping: Event flag for cancellation.
        active_container: Single-element list tracking the running
            container name for kill-on-shutdown.
        agent_workspace_dir: Subdirectory name for the agent
            workspace. Defaults to ``"agent_workspace"``.

    Returns:
        Response dict with "ok" key and results or error details.
    """
    # Validate block name.
    if allowed_blocks and block_name not in allowed_blocks:
        return {
            "request_id": request_id,
            "ok": False,
            "retryable": False,
            "error_type": "block_not_allowed",
            "message": (
                f"Block {block_name!r} is not in the allowed "
                f"list: {allowed_blocks}"),
        }

    block_def = _find_block(template, block_name)
    if block_def is None:
        return {
            "request_id": request_id,
            "ok": False,
            "retryable": False,
            "error_type": "unknown_block",
            "message": f"Block {block_name!r} not found in template",
        }

    agent_ws_name = agent_workspace_dir or "agent_workspace"
    agent_ws = workspace.path / agent_ws_name
    artifact_path = (agent_ws / artifact_rel).resolve()

    if not artifact_path.is_relative_to(agent_ws.resolve()):
        return {
            "request_id": request_id,
            "ok": False,
            "retryable": False,
            "error_type": "path_traversal",
            "message": f"Artifact path escapes workspace: {artifact_rel}",
        }

    # Retry briefly for Windows/WSL2 bind mount delay.
    if not artifact_path.exists():
        for _ in range(3):
            time.sleep(1)
            if artifact_path.exists():
                break
        else:
            return {
                "request_id": request_id,
                "ok": False,
                "retryable": False,
                "error_type": "file_not_found",
                "message": f"Artifact not found: {artifact_rel}",
            }

    if stopping and stopping.is_set():
        return {
            "request_id": request_id,
            "ok": False,
            "retryable": False,
            "error_type": "cancelled",
            "message": "Cancelled — service is shutting down",
        }

    print(f"  [block-bridge] {request_id}: invoking {block_name} "
          f"with {artifact_rel}")

    # ── Import artifact for the first input slot ──────────────────
    if not block_def.inputs:
        return {
            "request_id": request_id,
            "ok": False,
            "retryable": False,
            "error_type": "no_inputs",
            "message": f"Block {block_name!r} has no input slots",
        }

    input_slot = block_def.inputs[0]
    input_instance = workspace.register_artifact(
        input_slot.name,
        artifact_path,
        source=f"agent invocation {request_id}",
    )

    # ── Prepare container from block definition ───────────────────
    execution_id = workspace.generate_execution_id()
    input_bindings = {input_slot.name: input_instance.id}

    # Build mounts from block definition slots.
    mounts: list[tuple[str, str, str]] = []

    # Input: mount the imported artifact directory.
    input_host_path = str(
        (workspace.path / "artifacts" / input_instance.copy_path).resolve()
    ).replace("\\", "/")
    mounts.append((input_host_path, input_slot.container_path, "ro"))

    # Outputs: allocate artifact directories for each output slot.
    output_artifact_ids: dict[str, str] = {}
    output_dirs: dict[str, str] = {}
    for output_slot in block_def.outputs:
        aid = workspace.generate_artifact_id(output_slot.name)
        output_dir = workspace.path / "artifacts" / aid
        output_dir.mkdir(parents=True)
        output_host = str(output_dir.resolve()).replace("\\", "/")
        mounts.append((output_host, output_slot.container_path, "rw"))
        output_artifact_ids[output_slot.name] = aid
        output_dirs[output_slot.name] = str(output_dir)

    container_name = f"flywheel-block-{request_id}"

    cc = ContainerConfig(
        image=block_def.image,
        docker_args=list(block_def.docker_args),
        env=dict(block_def.env),
        mounts=mounts,
    )

    # Build args from overrides.
    container_args: list[str] = []
    if overrides:
        for key, value in overrides.items():
            flag = f"--{key.replace('_', '-')}"
            container_args += [flag, str(value)]

    # ── Run container ─────────────────────────────────────────────
    if active_container is not None:
        active_container[0] = container_name

    started_at = datetime.now(UTC)
    try:
        result = run_container(
            cc, args=container_args or None, name=container_name)
    finally:
        if active_container is not None:
            active_container[0] = None
    finished_at = datetime.now(UTC)

    # ── Record results ────────────────────────────────────────────
    status = "succeeded" if result.exit_code == 0 else "failed"
    output_bindings: dict[str, str] = {}

    for output_slot in block_def.outputs:
        aid = output_artifact_ids[output_slot.name]
        output_dir_path = workspace.path / "artifacts" / aid
        if output_dir_path.exists() and any(output_dir_path.iterdir()):
            output_instance = ArtifactInstance(
                id=aid,
                name=output_slot.name,
                kind="copy",
                created_at=finished_at,
                produced_by=execution_id,
                copy_path=aid,
            )
            workspace.add_artifact(output_instance)
            output_bindings[output_slot.name] = aid
        else:
            shutil.rmtree(output_dir_path, ignore_errors=True)

    execution = BlockExecution(
        id=execution_id,
        block_name=block_name,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        exit_code=result.exit_code,
        elapsed_s=result.elapsed_s,
        image=block_def.image,
    )
    workspace.add_execution(execution)
    workspace.save()

    # Build response — read output for score/result info.
    response: dict[str, Any] = {
        "request_id": request_id,
        "ok": result.exit_code == 0,
        "exit_code": result.exit_code,
        "elapsed_s": round(result.elapsed_s, 1),
        "block_name": block_name,
        "execution_id": execution_id,
    }

    # Try to read scores.json from the first output directory.
    for aid in output_artifact_ids.values():
        scores_file = workspace.path / "artifacts" / aid / "scores.json"
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
        # Convention: exit code 2 means aborted (e.g., speed gate).
        for aid in output_artifact_ids.values():
            scores_file = (
                workspace.path / "artifacts" / aid / "scores.json")
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

    score_str = response.get("score", response.get("mean", "?"))
    print(f"  [block-bridge] {request_id}: {block_name} "
          f"score={score_str}")

    return response


def _kill_container(name: str) -> None:
    """Kill a Docker container by name. Ignores errors."""
    with contextlib.suppress(Exception):
        subprocess.run(
            ["docker", "kill", name],
            capture_output=True, timeout=10)


class BlockBridgeService:
    """HTTP bridge for nested block executions.

    Receives requests from inside an agent container and dispatches
    them as flywheel block executions, using block definitions from
    the template.
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
        on_record: Callable[[str, dict], None] | None = None,
        agent_workspace_dir: str | None = None,
    ):
        """Initialize the block bridge service.

        Args:
            template: Template containing block definitions.
            workspace: The flywheel workspace for artifact tracking.
            overrides: CLI flag overrides for invoked containers.
            allowed_blocks: If set, only these block names can be
                invoked. None means all blocks in the template.
            max_invocations: Maximum block invocations allowed
                (None = unlimited).
            host: Host to bind to.
            port: Port to listen on (0 = auto-assign).
            on_record: Callback fired after a successful record-mode
                invocation. Receives (block_name, outputs). Runs in
                the HTTP handler thread — must be thread-safe.
            agent_workspace_dir: Subdirectory name for the agent
                workspace. Defaults to ``"agent_workspace"``.
        """
        self.template = template
        self.workspace = workspace
        self.overrides = overrides
        self.allowed_blocks = allowed_blocks
        self.max_invocations = max_invocations
        self.host = host
        self.port = port
        self.on_record = on_record
        self.agent_workspace_dir = agent_workspace_dir

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

        agent_ws_dir = self.agent_workspace_dir

        class Handler(_BridgeRequestHandler):
            template = self.template
            workspace = self.workspace
            overrides = self.overrides
            allowed_blocks = self.allowed_blocks
            _counter = [0]
            _invocation_count = [0]
            _max_invocations = self.max_invocations
            _lock = lock
            _stopping = stopping
            _active_container = active_container
            _service_id = service_id
            _on_record = (
                staticmethod(self.on_record)
                if self.on_record else None
            )
            _agent_workspace_dir = agent_ws_dir

        self._server = HTTPServer((self.host, self.port), Handler)
        self.port = self._server.server_address[1]

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()

        print(f"  [block-bridge] listening on {self.host}:{self.port}")
        return self.port

    def stop(self, timeout: float = 30.0) -> None:
        """Shut down the server and kill any in-flight container."""
        self._stopping.set()

        with self._lock:
            container_name = self._active_container[0]
        if container_name:
            print(f"  [block-bridge] killing {container_name}")
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
