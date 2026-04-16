"""HTTP service for nested block execution from inside containers.

Replaces the former ``BlockBridgeService``.  Routes requests to
the appropriate executor based on the request mode:

- ``mode=record`` → ``RecordExecutor``
- Default (invoke) → ``ContainerExecutor``

Preserves the exact HTTP protocol so that MCP servers inside
containers need zero changes.

Fires ``ExecutionEvent`` callbacks after each execution, replacing
the former ``on_record`` callback with a typed, executor-agnostic
event system.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import subprocess
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from flywheel.executor import (
    ContainerExecutor,
    ExecutionEvent,
    RecordExecutor,
)
from flywheel.template import Template
from flywheel.workspace import Workspace


class _ChannelRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler that delegates to executors."""

    # Assigned by ExecutionChannel before the server starts.
    workspace: Workspace
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

    def do_POST(self):  # noqa: N802
        """Handle a POST request to execute a block."""
        if self._stopping.is_set():
            self._send_json(503, {
                "ok": False,
                "error_type": "stopping",
                "message": "Execution channel is shutting down",
            })
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
    ):
        """Initialize the execution channel."""
        self.template = template
        self.workspace = workspace
        self.overrides = overrides
        self.allowed_blocks = allowed_blocks
        self.max_invocations = max_invocations
        self.host = host
        self.port = port
        self.agent_workspace_dir = agent_workspace_dir

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

        class Handler(_ChannelRequestHandler):
            workspace = self.workspace
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
