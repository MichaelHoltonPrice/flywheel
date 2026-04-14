"""Agent block execution orchestration.

Runs an AI agent (Claude Code) inside a Docker container with access
to a block bridge service that lets it trigger nested block executions.
The agent reads source code, writes artifacts, and iteratively
invokes other blocks (e.g., evaluations) through an MCP tool.

The agent container receives:
- A workspace directory (read-write) for writing artifacts
- Source code directories (read-only)
- Previous artifacts as optional inputs (read-only)
- A block bridge endpoint for triggering nested block executions

Each block the agent invokes becomes a tracked block execution
with full artifact provenance in the workspace.

Two APIs are provided:

- ``launch_agent_block()`` returns an ``AgentHandle`` immediately
  for non-blocking control.  The handle supports ``stop()`` to
  request a graceful shutdown (e.g., from a bridge callback) and
  ``wait()`` to block until completion and collect artifacts.
- ``run_agent_block()`` is a blocking convenience wrapper that
  calls ``launch_agent_block()`` then ``handle.wait()``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flywheel.block_bridge import BlockBridgeService
from flywheel.template import Template
from flywheel.workspace import Workspace


@dataclass(frozen=True)
class AgentResult:
    """Result of an agent block execution.

    Attributes:
        exit_code: The agent container's exit code.
        elapsed_s: Wall-clock time in seconds.
        evals_run: Number of evaluations the agent triggered.
    """

    exit_code: int
    elapsed_s: float
    evals_run: int


def _resolve_path(path: Path) -> str:
    """Resolve a path for Docker volume mounts (Windows-safe)."""
    resolved = str(path.resolve())
    if sys.platform == "win32":
        resolved = resolved.replace("\\", "/")
    return resolved


# Default total timeout for an agent step (4 hours).
DEFAULT_TOTAL_TIMEOUT = 14400


def _read_stdout(
    process: subprocess.Popen,
    event_log: Path,
    start_time: float,
    total_timeout: int,
    container_name: str = "",
) -> None:
    """Read agent stdout in a background thread.

    Writes JSON events to the event log, prints summaries via
    ``_log_event``, and requests a graceful stop if the total
    timeout is exceeded.
    """
    with open(event_log, "w") as log_f:
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            log_f.write(line + "\n")
            log_f.flush()

            try:
                event = json.loads(line)
                _log_event(event)
            except json.JSONDecodeError:
                pass

            if time.monotonic() - start_time > total_timeout:
                print(f"  [agent] total timeout "
                      f"({total_timeout}s) exceeded -- stopping")
                if container_name:
                    subprocess.run(
                        ["docker", "exec", container_name,
                         "touch", "/workspace/.agent_stop"],
                        capture_output=True, timeout=10,
                    )
                else:
                    process.kill()
                break


class AgentHandle:
    """Handle to a running agent container.

    Returned by ``launch_agent_block()``.  Provides non-blocking
    control over the container lifecycle.

    The caller **must** call ``wait()`` exactly once to clean up
    resources (join threads, stop bridge, collect artifacts).
    Call ``stop()`` to request a graceful shutdown — the agent
    runner detects the stop file, exports its session, and exits
    on its own.
    """

    def __init__(
        self,
        process: subprocess.Popen,
        bridge: BlockBridgeService,
        workspace: Workspace,
        agent_ws: Path,
        output_names: list[str] | None,
        start_time: float,
        executions_before: int,
        agent_image: str,
        stdout_thread: threading.Thread,
        stderr_thread: threading.Thread,
        container_name: str = "",
    ):
        """Initialize from a launched container and its resources."""
        self._process = process
        self._bridge = bridge
        self._workspace = workspace
        self._agent_ws = agent_ws
        self._output_names = output_names
        self._start_time = start_time
        self._executions_before = executions_before
        self._agent_image = agent_image
        self._stdout_thread = stdout_thread
        self._stderr_thread = stderr_thread
        self._container_name = container_name
        self._waited = False

    def is_alive(self) -> bool:
        """Check if the container process is still running."""
        return self._process.poll() is None

    def stop(self) -> None:
        """Request a graceful shutdown of the agent.

        Creates a ``.agent_stop`` file inside the container via
        ``docker exec``.  The agent runner checks for this file
        between turns, exports the session artifact, and exits
        cleanly.

        Uses ``docker exec`` rather than host-side file writes
        because Docker Desktop bind mounts on Windows don't
        reliably propagate host writes to the container.

        The caller **must** still call ``wait()`` afterward to join
        threads, stop the bridge, and collect artifacts.
        """
        if self._container_name:
            subprocess.run(
                ["docker", "exec", self._container_name,
                 "touch", "/workspace/.agent_stop"],
                capture_output=True, timeout=10,
            )

    def wait(self) -> AgentResult:
        """Block until the container exits and return results.

        Joins background threads, stops the bridge, collects output
        artifacts, and returns ``AgentResult``.  Must be called
        exactly once.

        Raises:
            RuntimeError: If ``wait()`` has already been called.
        """
        if self._waited:
            raise RuntimeError(
                "wait() already called on this AgentHandle")
        self._waited = True

        try:
            self._stdout_thread.join()
            self._process.wait()
        finally:
            self._stderr_thread.join(timeout=5)
            self._bridge.stop()

        elapsed = time.monotonic() - self._start_time

        # Collect output artifacts from agent workspace.
        if self._output_names and self._agent_ws.exists():
            for name in self._output_names:
                for candidate in self._agent_ws.iterdir():
                    if candidate.is_file() and candidate.stem == name:
                        self._workspace.register_artifact(
                            name, candidate,
                            source=f"agent output ({self._agent_image})",
                        )
                        break

        evals_run = (
            len(self._workspace.executions) - self._executions_before
        )
        exit_code = (
            self._process.returncode
            if self._process is not None else -1
        )

        print(
            f"  [agent] completed: exit_code={exit_code}, "
            f"elapsed={elapsed:.1f}s, evals={evals_run}"
        )

        return AgentResult(
            exit_code=exit_code,
            elapsed_s=elapsed,
            evals_run=evals_run,
        )


def launch_agent_block(
    workspace: Workspace,
    template: Template,
    project_root: Path,
    prompt: str,
    agent_image: str = "flywheel-claude:latest",
    auth_volume: str = "claude-auth",
    model: str | None = None,
    max_invocations: int | None = None,
    max_turns: int | None = None,
    total_timeout: int = DEFAULT_TOTAL_TIMEOUT,
    allowed_blocks: list[str] | None = None,
    source_dirs: list[str] | None = None,
    input_artifacts: dict[str, str] | None = None,
    output_names: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
    mcp_servers: str | None = None,
    allowed_tools: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_mounts: list[tuple[str, str, str]] | None = None,
    pre_launch_hook: Callable[[Path], None] | None = None,
    on_record: Callable[[str, dict], None] | None = None,
    isolated_network: bool = False,
) -> AgentHandle:
    """Launch an agent block execution (non-blocking).

    Sets up the workspace, starts the block bridge, launches the
    Docker container, and returns an ``AgentHandle`` immediately.
    The container runs in the background.

    The caller must call ``handle.wait()`` to clean up and get
    the result.  Call ``handle.stop()`` to request a graceful
    shutdown (e.g., from a bridge callback), then ``wait()`` to
    finalize.

    Args:
        workspace: The flywheel workspace for artifact tracking.
        template: The template containing block definitions.
        project_root: Path to the project root.
        prompt: The system prompt for the agent.
        agent_image: Docker image for the agent container.
        auth_volume: Docker named volume with API credentials.
        model: Model name (e.g., "claude-sonnet-4-6").
        max_invocations: Maximum nested block invocations.
        max_turns: Maximum agent conversation turns.
        total_timeout: Maximum wall-clock seconds.
        allowed_blocks: Block names the agent may invoke.
        source_dirs: Source directories to mount read-only.
        input_artifacts: Maps mount names to artifact instance IDs.
        output_names: Artifact names to collect after completion.
        overrides: CLI flag overrides for invoked containers.
        mcp_servers: Comma-separated MCP server names.
        allowed_tools: Comma-separated tool whitelist.
        extra_env: Additional environment variables.
        extra_mounts: Additional volume mounts.
        pre_launch_hook: Callback before container launch.
        on_record: Callback fired after each successful record-mode
            bridge invocation.  Receives ``(block_name, outputs)``.
            Runs in the bridge HTTP handler thread.

    Returns:
        An ``AgentHandle`` for monitoring and controlling the agent.
    """
    # Create agent workspace directory (fresh each step).
    agent_ws = workspace.path / "agent_workspace"
    if agent_ws.exists():
        shutil.rmtree(agent_ws)
    agent_ws.mkdir(parents=True)

    # Seed the agent workspace with the latest artifacts from
    # prior steps so the agent can continue where the previous
    # step left off.
    if output_names:
        for name in output_names:
            instances = workspace.instances_for(name)
            if instances:
                latest = instances[-1]
                if latest.kind == "copy" and latest.copy_path:
                    src_dir = (
                        workspace.path / "artifacts" / latest.copy_path
                    )
                    if src_dir.exists():
                        for f in src_dir.iterdir():
                            if f.is_file():
                                shutil.copy2(f, agent_ws / f.name)

    # Snapshot execution count to compute invocations later.
    executions_before = len(workspace.executions)

    # Start block bridge service.
    bridge = BlockBridgeService(
        template=template,
        workspace=workspace,
        overrides=overrides,
        allowed_blocks=allowed_blocks,
        max_invocations=max_invocations,
        on_record=on_record,
    )
    port = bridge.start()
    bridge_endpoint = f"http://host.docker.internal:{port}"

    # Build Docker command with a named container for reliable stop.
    container_name = f"flywheel-{uuid.uuid4().hex[:12]}"
    cmd = ["docker", "run", "--rm", "-i", "--name", container_name]

    if isolated_network:
        cmd.extend(["--cap-add=NET_ADMIN"])

    cmd.extend(["-v", f"{auth_volume}:/home/claude/.claude"])
    cmd.extend(["-v", f"{_resolve_path(agent_ws)}:/workspace"])

    if source_dirs:
        for i, src in enumerate(source_dirs):
            src_path = project_root / src
            mount_point = "/source" if i == 0 else f"/source{i + 1}"
            cmd.extend([
                "-v", f"{_resolve_path(src_path)}:{mount_point}:ro",
            ])

    if input_artifacts:
        for mount_name, artifact_id in input_artifacts.items():
            if artifact_id not in workspace.artifacts:
                continue
            inst = workspace.artifacts[artifact_id]
            if inst.kind == "copy" and inst.copy_path:
                host_path = (
                    workspace.path / "artifacts" / inst.copy_path
                )
                if host_path.exists():
                    cmd.extend([
                        "-v",
                        f"{_resolve_path(host_path)}"
                        f":/input/{mount_name}:ro",
                    ])

    if extra_mounts:
        for host_path, container_path, mode in extra_mounts:
            cmd.extend([
                "-v",
                f"{_resolve_path(Path(host_path))}"
                f":{container_path}:{mode}",
            ])

    default_block = (
        allowed_blocks[0] if allowed_blocks else "eval_bot"
    )
    env_vars = {
        "EVAL_ENDPOINT": bridge_endpoint,
        "EVAL_BLOCK": default_block,
        "MCP_SERVERS": mcp_servers or "eval",
        "ALLOWED_TOOLS": (
            allowed_tools or "Read,Write,Edit,Glob,Grep"
        ),
    }
    if model:
        env_vars["MODEL"] = model
    if max_turns is not None:
        env_vars["MAX_TURNS"] = str(max_turns)
    env_vars["PYTHONUNBUFFERED"] = "1"
    env_vars["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "128000"
    if isolated_network:
        env_vars["NETWORK_ISOLATION"] = "1"

    if extra_env:
        env_vars.update(extra_env)

    if pre_launch_hook is not None:
        pre_launch_hook(agent_ws)
        if extra_env:
            env_vars.update(extra_env)

    for key, value in env_vars.items():
        cmd.extend(["-e", f"{key}={value}"])

    cmd.append(agent_image)

    # Launch agent container.
    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"

    print(f"  [agent] launching {agent_image}")
    start = time.monotonic()
    event_log = workspace.path / "agent_events.jsonl"
    stderr_log = workspace.path / "agent_stderr.log"

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )

        process.stdin.write(prompt)
        process.stdin.close()
    except Exception:
        bridge.stop()
        raise

    # Background threads for stdout and stderr.
    def _drain_stderr():
        with open(stderr_log, "w", encoding="utf-8") as f:
            for line in process.stderr:
                f.write(line)

    stderr_thread = threading.Thread(
        target=_drain_stderr, daemon=True)
    stderr_thread.start()

    stdout_thread = threading.Thread(
        target=_read_stdout,
        args=(process, event_log, start, total_timeout,
              container_name),
        daemon=True,
    )
    stdout_thread.start()

    return AgentHandle(
        process=process,
        bridge=bridge,
        workspace=workspace,
        agent_ws=agent_ws,
        output_names=output_names,
        start_time=start,
        executions_before=executions_before,
        agent_image=agent_image,
        stdout_thread=stdout_thread,
        stderr_thread=stderr_thread,
        container_name=container_name,
    )


def run_agent_block(
    workspace: Workspace,
    template: Template,
    project_root: Path,
    prompt: str,
    agent_image: str = "flywheel-claude:latest",
    auth_volume: str = "claude-auth",
    model: str | None = None,
    max_invocations: int | None = None,
    max_turns: int | None = None,
    total_timeout: int = DEFAULT_TOTAL_TIMEOUT,
    allowed_blocks: list[str] | None = None,
    source_dirs: list[str] | None = None,
    input_artifacts: dict[str, str] | None = None,
    output_names: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
    mcp_servers: str | None = None,
    allowed_tools: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_mounts: list[tuple[str, str, str]] | None = None,
    pre_launch_hook: Callable[[Path], None] | None = None,
    isolated_network: bool = False,
) -> AgentResult:
    """Run an agent block execution (blocking).

    Convenience wrapper around ``launch_agent_block()`` +
    ``handle.wait()``.  See ``launch_agent_block()`` for full
    argument documentation.

    Returns:
        AgentResult with exit code, elapsed time, and invocation count.
    """
    handle = launch_agent_block(
        workspace=workspace,
        template=template,
        project_root=project_root,
        prompt=prompt,
        agent_image=agent_image,
        auth_volume=auth_volume,
        model=model,
        max_invocations=max_invocations,
        max_turns=max_turns,
        total_timeout=total_timeout,
        allowed_blocks=allowed_blocks,
        source_dirs=source_dirs,
        input_artifacts=input_artifacts,
        output_names=output_names,
        overrides=overrides,
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
        extra_env=extra_env,
        extra_mounts=extra_mounts,
        pre_launch_hook=pre_launch_hook,
        isolated_network=isolated_network,
    )
    try:
        return handle.wait()
    except KeyboardInterrupt:
        print("  [agent] interrupted -- requesting graceful stop")
        handle.stop()
        try:
            handle._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("  [agent] stop timed out -- terminating")
            handle._process.terminate()
            handle._process.wait()
        raise


def _log_event(event: dict) -> None:
    """Print a concise summary of an agent event."""
    event_type = event.get("type", "")

    if event_type == "assistant":
        content = event.get("content", [])
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "")
                    preview = text[:120].replace("\n", " ")
                    if preview:
                        # Replace non-ASCII to avoid cp1252 errors on Windows.
                        safe = preview.encode("ascii", "replace").decode("ascii")
                        print(f"  [agent] {safe}")
                elif btype == "tool_use":
                    tool = block.get("name", "?")
                    if tool.startswith("mcp__"):
                        short = tool.split("__")[-1]
                        print(f"  [agent] [mcp] {short}()")
                    else:
                        print(f"  [agent] [tool] {tool}()")

    elif event_type == "result":
        subtype = event.get("subtype", "")
        print(f"  [agent] result: {subtype}")

    elif event_type == "error":
        msg = event.get("message", "unknown error")
        print(f"  [agent] ERROR: {msg}")
