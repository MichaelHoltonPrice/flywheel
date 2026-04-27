"""Workspace-persistent container runtime management.

The substrate concept is a persistent runtime: a long-lived worker
that accepts per-execution requests prepared by Flywheel and writes
normal proposal bytes for the shared commit path.  This module ships
the first built-in implementation, an HTTP worker in a Docker
container.  The rest of Flywheel should depend on the small runner
surface, not on HTTP-specific details.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from flywheel import runtime
from flywheel.container import ContainerConfig, ContainerResult
from flywheel.state import state_compatibility_identity
from flywheel.template import BlockDefinition
from flywheel.termination import (
    normalize_termination_reason,
    read_termination_sidecar,
)


class PersistentRuntimeError(RuntimeError):
    """Raised when the persistent runtime cannot dispatch a request."""


@dataclass(frozen=True)
class PersistentRuntimeResult:
    """Runtime observation returned by a persistent runner."""

    termination_reason: str
    container_result: ContainerResult | None
    announcement: str | None = None
    error: str | None = None


class PersistentRuntimeRunner(Protocol):
    """Runner surface consumed by block execution orchestration."""

    def run(self, plan: Any, args: list[str] | None = None) -> PersistentRuntimeResult:
        """Dispatch one prepared execution to a persistent worker."""
        ...


@dataclass(frozen=True)
class PersistentContainerInfo:
    """Operator-facing snapshot of a persistent runtime container."""

    name: str
    block_name: str | None
    status: str
    image: str | None
    labels: dict[str, str]


def workspace_runtime_id(workspace_path: Path) -> str:
    """Return a deterministic short ID for Docker labels/names."""
    resolved = str(workspace_path.resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def persistent_exchange_root(workspace_path: Path, block_name: str) -> Path:
    """Return the stable exchange root for one persistent block."""
    return workspace_path / "runtimes" / block_name / "exchange"


def persistent_request_root(
    workspace_path: Path,
    block_name: str,
    execution_id: str,
) -> Path:
    """Return the per-execution request/proposal directory."""
    return (
        persistent_exchange_root(workspace_path, block_name)
        / runtime.REQUEST_TREE_WORKSPACE_RELATIVE
        / execution_id
    )


def persistent_container_name(workspace_path: Path, block_name: str) -> str:
    """Return the deterministic Docker container name."""
    return f"flywheel-{workspace_runtime_id(workspace_path)}-{block_name}"


def _docker_json(cmd: list[str]) -> Any:
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if not completed.stdout.strip():
        return None
    return json.loads(completed.stdout)


def _docker_output(cmd: list[str]) -> str:
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _container_inspect(name: str) -> dict[str, Any] | None:
    try:
        data = _docker_json(["docker", "inspect", name])
    except subprocess.CalledProcessError:
        return None
    if not data:
        return None
    return data[0]


def _image_id(image: str) -> str:
    try:
        return _docker_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"]
        )
    except subprocess.CalledProcessError as exc:
        raise PersistentRuntimeError(
            f"could not inspect Docker image {image!r}"
        ) from exc


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _required_labels(
    *,
    workspace_path: Path,
    block_def: BlockDefinition,
    port: int,
) -> dict[str, str]:
    identity = state_compatibility_identity(block_def)
    return {
        "flywheel.workspace_id": workspace_runtime_id(workspace_path),
        "flywheel.workspace_path": str(workspace_path.resolve()),
        "flywheel.block_name": block_def.name,
        "flywheel.lifecycle": "workspace_persistent",
        "flywheel.scope": "workspace",
        "flywheel.block_template_hash": identity["block_template_hash"],
        "flywheel.image": block_def.image,
        "flywheel.image_id": _image_id(block_def.image),
        "flywheel.protocol": runtime.PERSISTENT_RUNTIME_PROTOCOL_VERSION,
        "flywheel.port": str(port),
    }


def _labels_from_inspect(inspect: dict[str, Any]) -> dict[str, str]:
    config = inspect.get("Config") or {}
    labels = config.get("Labels") or {}
    return {str(key): str(value) for key, value in labels.items()}


def _container_status(inspect: dict[str, Any]) -> str:
    state = inspect.get("State") or {}
    if state.get("Running"):
        return "running"
    return str(state.get("Status") or "unknown")


def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


def _execute_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/execute"


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_s: float | None = 10.0,
) -> Any:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def _wait_for_health(port: int, *, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            _http_json("GET", _health_url(port), timeout_s=2.0)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.25)
    raise PersistentRuntimeError(
        f"persistent container health check failed: {last_error}"
    )


def _normalize_host_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _start_container(
    *,
    name: str,
    config: ContainerConfig,
    labels: dict[str, str],
    port: int,
) -> None:
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"127.0.0.1:{port}:{port}",
    ]
    for key, value in labels.items():
        cmd.extend(["--label", f"{key}={value}"])
    cmd.extend(config.docker_args)
    for key, value in config.env.items():
        cmd.extend(["-e", f"{key}={value}"])
    for host_path, container_path, mode in config.mounts:
        normalized_host = host_path.replace("\\", "/")
        cmd.extend([
            "-v",
            f"{normalized_host}:{container_path}:{mode}",
        ])
    cmd.append(config.image)
    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"
    subprocess.run(cmd, check=True, env=env)


@dataclass(frozen=True)
class DockerHttpPersistentRuntimeRunner:
    """Built-in Docker HTTP persistent-runtime runner."""

    workspace_path: Path
    health_timeout_s: float = 30.0

    def run(self, plan: Any, args: list[str] | None = None) -> PersistentRuntimeResult:
        """Dispatch one prepared execution to the persistent worker."""
        start = time.monotonic()
        name, port = self._ensure_running(plan)
        request_payload = {
            "protocol": runtime.PERSISTENT_RUNTIME_PROTOCOL_VERSION,
            "request_id": plan.execution_id,
            "env": self._request_env(plan),
            "input_root": (
                f"{runtime.FLYWHEEL_EXCHANGE_MOUNT}/"
                f"{runtime.REQUEST_TREE_WORKSPACE_RELATIVE}/"
                f"{plan.execution_id}/input"
            ),
            "output_root": (
                f"{runtime.FLYWHEEL_EXCHANGE_MOUNT}/"
                f"{runtime.REQUEST_TREE_WORKSPACE_RELATIVE}/"
                f"{plan.execution_id}/output"
            ),
            "telemetry_root": (
                f"{runtime.FLYWHEEL_EXCHANGE_MOUNT}/"
                f"{runtime.REQUEST_TREE_WORKSPACE_RELATIVE}/"
                f"{plan.execution_id}/telemetry"
            ),
            "termination_path": (
                f"{runtime.FLYWHEEL_EXCHANGE_MOUNT}/"
                f"{runtime.REQUEST_TREE_WORKSPACE_RELATIVE}/"
                f"{plan.execution_id}/termination"
            ),
            "args": list(args or []),
        }
        (plan.proposals_root / "request.json").write_text(
            json.dumps(request_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        try:
            response = _http_json(
                "POST",
                _execute_url(port),
                request_payload,
                timeout_s=None,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PersistentRuntimeError(
                f"persistent container {name!r} dispatch failed: {exc}"
            ) from exc
        if not isinstance(response, dict):
            raise PersistentRuntimeError(
                f"persistent container {name!r} returned malformed response"
            )
        (plan.proposals_root / "response.json").write_text(
            json.dumps(response, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        elapsed = time.monotonic() - start
        status = response.get("status")
        response_reason = response.get("termination_reason")
        announcement = read_termination_sidecar(plan.termination_file)
        if isinstance(response_reason, str) and announcement is None:
            announcement = response_reason.strip() or None
        if status == "failed":
            error = response.get("error")
            return PersistentRuntimeResult(
                termination_reason=runtime.TERMINATION_REASON_CRASH,
                container_result=ContainerResult(exit_code=1, elapsed_s=elapsed),
                announcement=announcement,
                error=str(error) if error is not None else "worker reported failure",
            )
        if status != "succeeded":
            raise PersistentRuntimeError(
                f"persistent container {name!r} returned status {status!r}"
            )
        reason = normalize_termination_reason(
            announcement=announcement,
            declared_reasons=set(plan.block_def.outputs.keys()),
        )
        return PersistentRuntimeResult(
            termination_reason=reason,
            container_result=ContainerResult(exit_code=0, elapsed_s=elapsed),
            announcement=announcement,
        )

    def _ensure_running(self, plan: Any) -> tuple[str, int]:
        block_def = plan.block_def
        name = persistent_container_name(self.workspace_path, block_def.name)
        inspect = _container_inspect(name)
        if inspect is not None:
            labels = _labels_from_inspect(inspect)
            if _container_status(inspect) != "running":
                raise PersistentRuntimeError(
                    f"persistent container {name!r} exists but is not running"
                )
            port_text = labels.get("flywheel.port")
            if port_text is None:
                raise PersistentRuntimeError(
                    f"persistent container {name!r} is missing port label"
                )
            port = int(port_text)
            expected = _required_labels(
                workspace_path=self.workspace_path,
                block_def=block_def,
                port=port,
            )
            mismatches = [
                key for key, value in expected.items()
                if labels.get(key) != value
            ]
            if mismatches:
                joined = ", ".join(sorted(mismatches))
                raise PersistentRuntimeError(
                    f"persistent container {name!r} was started under "
                    f"different settings ({joined}); stop it before retrying"
                )
            _wait_for_health(port, timeout_s=self.health_timeout_s)
            return name, port

        port = _find_free_port()
        exchange_root = persistent_exchange_root(
            self.workspace_path, block_def.name)
        exchange_root.mkdir(parents=True, exist_ok=True)
        labels = _required_labels(
            workspace_path=self.workspace_path,
            block_def=block_def,
            port=port,
        )
        env = dict(block_def.env)
        env[runtime.CONTROL_PORT_ENV_VAR] = str(port)
        config = ContainerConfig(
            image=block_def.image,
            docker_args=block_def.docker_args,
            env=env,
            mounts=[
                (
                    _normalize_host_path(exchange_root),
                    runtime.FLYWHEEL_EXCHANGE_MOUNT,
                    "rw",
                )
            ],
        )
        _start_container(name=name, config=config, labels=labels, port=port)
        _wait_for_health(port, timeout_s=self.health_timeout_s)
        return name, port

    @staticmethod
    def _request_env(plan: Any) -> dict[str, str]:
        """Return per-execution environment for the request envelope."""
        env = dict(plan.block_def.env)
        if plan.env_overlay:
            env.update(plan.env_overlay)
        return env


def list_persistent_containers(workspace_path: Path) -> list[PersistentContainerInfo]:
    """List persistent containers owned by a workspace."""
    workspace_id = workspace_runtime_id(workspace_path)
    try:
        completed = subprocess.run([
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=flywheel.workspace_id={workspace_id}",
            "--format",
            "json",
        ], check=True, capture_output=True, text=True, encoding="utf-8")
    except subprocess.CalledProcessError as exc:
        raise PersistentRuntimeError(str(exc)) from exc
    rows: list[Any] = []
    for line in completed.stdout.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    infos: list[PersistentContainerInfo] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Names") or row.get("Name") or "")
        inspect = _container_inspect(name) if name else None
        labels = _labels_from_inspect(inspect) if inspect else {}
        infos.append(PersistentContainerInfo(
            name=name,
            block_name=labels.get("flywheel.block_name"),
            status=str(row.get("State") or row.get("Status") or ""),
            image=str(row.get("Image") or ""),
            labels=labels,
        ))
    return infos


def stop_persistent_container(
    workspace_path: Path,
    block_name: str,
    *,
    timeout_s: int,
    reason: str = "cli_stop",
    force: bool = False,
) -> None:
    """Stop and remove the persistent container for one block."""
    name = persistent_container_name(workspace_path, block_name)
    inspect = _container_inspect(name)
    if inspect is None:
        raise PersistentRuntimeError(
            f"no persistent container for block {block_name!r}"
        )
    expected_workspace = workspace_runtime_id(workspace_path)
    labels = _labels_from_inspect(inspect)
    if labels.get("flywheel.workspace_id") != expected_workspace:
        raise PersistentRuntimeError(
            f"container {name!r} does not belong to this workspace"
        )
    exchange_root = persistent_exchange_root(workspace_path, block_name)
    with suppress(OSError, PermissionError):
        exchange_root.mkdir(parents=True, exist_ok=True)
        (exchange_root / runtime.STOP_SENTINEL_WORKSPACE_RELATIVE).write_text(
            f"{reason}\n", encoding="utf-8")
    if force:
        subprocess.run(["docker", "rm", "-f", name], check=True)
    else:
        subprocess.run(
            ["docker", "stop", "-t", str(timeout_s), name],
            check=True,
        )
        subprocess.run(["docker", "rm", name], check=True)
