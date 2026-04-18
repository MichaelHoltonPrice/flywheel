"""Agent block execution orchestration.

Runs an AI agent (Claude Code) inside a Docker container.  The
agent reads source code, writes artifacts, and may trigger nested
block executions via the host-side full-stop handoff path
(:mod:`flywheel.agent_handoff`); see
:mod:`flywheel.local_block` for the recorder side.

The agent container receives:
- A workspace directory (read-write) for writing artifacts
- Source code directories (read-only)
- Previous artifacts as optional inputs (read-only)

Each block the agent invokes becomes a tracked block execution
with full artifact provenance in the workspace.

Two APIs are provided:

- ``launch_agent_block()`` returns an ``AgentHandle`` immediately
  for non-blocking control.  The handle supports ``stop()`` to
  request a graceful shutdown and ``wait()`` to block until
  completion and collect artifacts.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flywheel.artifact import BlockExecution, LifecycleEvent
from flywheel.template import Template
from flywheel.workspace import Workspace


@dataclass(frozen=True)
class AgentResult:
    """Result of an agent block execution.

    Attributes:
        exit_code: The agent container's exit code.
        elapsed_s: Wall-clock time in seconds.
        evals_run: Number of evaluations the agent triggered.
        execution_id: The workspace execution ID recorded for
            this agent run, or None if recording was skipped.
        stop_reason: Why the agent was stopped, if applicable.
        exit_reason: Semantic classification of why the agent
            exited.  One of: ``"completed"``, ``"auth_failure"``,
            ``"rate_limit"``, ``"max_turns"``, ``"stopped"``,
            ``"crashed"``, ``"tool_handoff"``.  ``"tool_handoff"``
            indicates the agent runner intercepted one or more
            handoff tool calls and exited cleanly to let the host
            run the corresponding blocks; the host-side handoff
            loop (``flywheel.agent_handoff``) is responsible for
            splicing results back and re-launching.
        agent_workspace_dir: Workspace-relative path of the
            directory the agent was bind-mounted at (e.g.,
            ``"agent_workspaces/abc12345"``).  Callers that need
            to read post-run files (output collection,
            log scraping) read this rather than guessing the
            path.  ``None`` only when recording was skipped.
    """

    exit_code: int
    elapsed_s: float
    evals_run: int
    execution_id: str | None = None
    stop_reason: str | None = None
    exit_reason: str | None = None
    agent_workspace_dir: str | None = None


def _classify_exit(
    exit_code: int,
    stop_reason: str | None,
    agent_ws: Path,
) -> str:
    """Classify an agent exit into a semantic exit_reason.

    Reads ``.agent_state.json`` from the agent workspace (written
    by the agent_runner) to determine why the agent stopped.

    Args:
        exit_code: The container's exit code.
        stop_reason: If the agent was externally stopped, the
            reason string.
        agent_ws: Path to the agent workspace directory.

    Returns:
        One of: ``"completed"``, ``"auth_failure"``,
        ``"rate_limit"``, ``"max_turns"``, ``"stopped"``,
        ``"crashed"``, ``"tool_handoff"``.
    """
    if stop_reason:
        return "stopped"

    state_file = agent_ws / ".agent_state.json"
    if state_file.exists():
        try:
            state = json.loads(
                state_file.read_text(encoding="utf-8"))
            status = state.get("status", "")
            reason = state.get("reason", "")

            if status == "tool_handoff":
                return "tool_handoff"
            if "auth" in reason:
                return "auth_failure"
            if "rate_limit" in reason:
                return "rate_limit"
            if reason == "max_turns":
                return "max_turns"
            if status == "complete":
                return "completed"
        except Exception:
            pass

    if exit_code == 0:
        return "completed"
    return "crashed"


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
    resources (join threads, collect artifacts).  Call ``stop()``
    to request a graceful shutdown — the agent runner detects the
    stop file, exports its session, and exits on its own.
    """

    def __init__(
        self,
        process: subprocess.Popen,
        workspace: Workspace,
        agent_ws: Path,
        output_names: list[str] | None,
        start_time: float,
        executions_before: int,
        agent_image: str,
        stdout_thread: threading.Thread,
        stderr_thread: threading.Thread,
        container_name: str = "",
        predecessor_id: str | None = None,
        block_name: str = "__agent__",
        agent_workspace_dir: str | None = None,
    ):
        """Initialize from a launched container and its resources."""
        self._process = process
        self._workspace = workspace
        self._agent_ws = agent_ws
        self._output_names = output_names
        self._start_time = start_time
        self._executions_before = executions_before
        self._agent_image = agent_image
        self._stdout_thread = stdout_thread
        self._stderr_thread = stderr_thread
        self._container_name = container_name
        self._predecessor_id = predecessor_id
        self._block_name = block_name
        self._agent_workspace_dir = agent_workspace_dir
        self._stop_reason: str | None = None
        self._waited = False

    def is_alive(self) -> bool:
        """Check if the container process is still running."""
        return self._process.poll() is None

    def stop(self, reason: str = "requested") -> None:
        """Request a graceful shutdown of the agent.

        Creates a ``.agent_stop`` file inside the container via
        ``docker exec``.  The agent runner checks for this file
        between turns, exports the session artifact, and exits
        cleanly.

        Uses ``docker exec`` rather than host-side file writes
        because Docker Desktop bind mounts on Windows don't
        reliably propagate host writes to the container.

        The caller **must** still call ``wait()`` afterward to join
        threads and collect artifacts.

        Args:
            reason: Why the agent is being stopped (e.g.,
                ``"exploration_request"``, ``"prediction_mismatch"``).
                Recorded in the workspace execution record.
        """
        self._stop_reason = reason
        if self._container_name:
            subprocess.run(
                ["docker", "exec", self._container_name,
                 "touch", "/workspace/.agent_stop"],
                capture_output=True, timeout=10,
            )

    def wait(self) -> AgentResult:
        """Block until the container exits and return results.

        Joins background threads, collects output artifacts,
        records the agent execution in the workspace, and returns
        ``AgentResult``.  Must be called exactly once.

        Raises:
            RuntimeError: If ``wait()`` has already been called.
        """
        if self._waited:
            raise RuntimeError(
                "wait() already called on this AgentHandle")
        self._waited = True

        started_at = datetime.fromtimestamp(
            self._start_time - time.monotonic() + time.time(),
            tz=UTC,
        )

        try:
            self._stdout_thread.join()
            self._process.wait()
        finally:
            self._stderr_thread.join(timeout=5)

        elapsed = time.monotonic() - self._start_time
        finished_at = datetime.now(UTC)

        # Collect output artifacts from agent workspace.
        output_bindings: dict[str, str] = {}
        if self._output_names and self._agent_ws.exists():
            for name in self._output_names:
                for candidate in self._agent_ws.iterdir():
                    if candidate.is_file() and candidate.stem == name:
                        inst = self._workspace.register_artifact(
                            name, candidate,
                            source=f"agent output ({self._agent_image})",
                        )
                        output_bindings[name] = inst.id
                        break

        evals_run = (
            len(self._workspace.executions) - self._executions_before
        )
        exit_code = (
            self._process.returncode
            if self._process is not None else -1
        )

        # Record the agent execution in the workspace.
        if self._stop_reason:
            status = "interrupted"
        elif exit_code == 0:
            status = "succeeded"
        else:
            status = "failed"

        execution_id = self._workspace.generate_execution_id()
        execution = BlockExecution(
            id=execution_id,
            block_name=self._block_name,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            output_bindings=output_bindings,
            exit_code=exit_code,
            elapsed_s=elapsed,
            image=self._agent_image,
            stop_reason=self._stop_reason,
            predecessor_id=self._predecessor_id,
            agent_workspace_dir=self._agent_workspace_dir,
        )
        self._workspace.add_execution(execution)

        # Record a lifecycle event if the agent was stopped.
        if self._stop_reason:
            event = LifecycleEvent(
                id=self._workspace.generate_event_id(),
                kind="agent_stopped",
                timestamp=finished_at,
                execution_id=execution_id,
                detail={"reason": self._stop_reason},
            )
            self._workspace.add_event(event)

        self._workspace.save()

        print(
            f"  [agent] completed: exit_code={exit_code}, "
            f"elapsed={elapsed:.1f}s, evals={evals_run}"
        )

        exit_reason = _classify_exit(
            exit_code, self._stop_reason, self._agent_ws)

        return AgentResult(
            exit_code=exit_code,
            elapsed_s=elapsed,
            evals_run=evals_run,
            execution_id=execution_id,
            stop_reason=self._stop_reason,
            exit_reason=exit_reason,
            agent_workspace_dir=self._agent_workspace_dir,
        )


@dataclass
class AgentBlockConfig:
    """Configuration for an agent block launch.

    Groups the parameters for ``launch_agent_block`` into a
    reusable, inspectable object.  Used by ``AgentGroup`` to
    define a base config with per-member overrides.

    Attributes:
        workspace: The flywheel workspace for artifact tracking.
        template: The template containing block definitions.
        project_root: Path to the project root.
        prompt: The system prompt for the agent.
        agent_image: Docker image for the agent container.
        auth_volume: Docker named volume with API credentials.
        model: Model name (e.g., ``"claude-sonnet-4-6"``).
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
        isolated_network: Enable iptables-based network isolation.
        agent_workspace_dir: Subdirectory name for the agent workspace.
        predecessor_id: Execution ID of a previous agent run that
            this launch resumes from.
    """

    workspace: Workspace
    template: Template
    project_root: Path
    prompt: str
    agent_image: str = "flywheel-claude:latest"
    auth_volume: str = "claude-auth"
    model: str | None = None
    max_turns: int | None = None
    total_timeout: int = DEFAULT_TOTAL_TIMEOUT
    allowed_blocks: list[str] | None = None
    source_dirs: list[str] | None = None
    input_artifacts: dict[str, str] | None = None
    output_names: list[str] | None = None
    overrides: dict[str, Any] | None = None
    mcp_servers: str | None = None
    allowed_tools: str | None = None
    extra_env: dict[str, str] | None = None
    extra_mounts: list[tuple[str, str, str]] | None = None
    pre_launch_hook: Callable[[Path], None] | None = None
    isolated_network: bool = False
    agent_workspace_dir: str | None = None
    predecessor_id: str | None = None
    # Pre-resolved post-execution callbacks keyed by block name.
    # See :mod:`flywheel.post_check`.  Reserved for future use:
    # the agent path itself does not run post-checks today; the
    # full-stop handoff loop registers them on its own
    # :class:`flywheel.local_block.LocalBlockRecorder` instead.
    post_checks: dict[str, Any] | None = None
    # Optional ``{{KEY}} -> value`` substitutions applied by
    # :class:`flywheel.pattern_runner.PatternRunner` to each
    # role's prompt text before launch.  The :mod:`launch_agent_block`
    # path does not consult this (the loop already substitutes
    # via ``--key value`` overrides parsed from the CLI tail);
    # patterns surface it as an explicit hook return so the
    # equivalent of ``{{GAME_ID}}`` keeps working without
    # leaking ``--key value`` plumbing into the runner.
    prompt_substitutions: dict[str, str] | None = None


# Subdirectory under the workspace where auto-named agent
# workspaces live; one ``<short-uuid>`` directory per agent
# launch.  Pulled out as a module constant so cyberarc tooling
# (``play_server``, replay scripts) can refer to it without
# duplicating the literal.
AGENT_WORKSPACES_DIR = "agent_workspaces"


@dataclass(frozen=True)
class AgentMount:
    """A bind-mount target prepared for one agent launch.

    Bundles the host directory (rooted under the workspace), the
    container path it'll be mounted at, and whether it's
    writable, so callers don't have to recompute these from the
    name string.  Returned by :func:`prepare_agent_workspace` and
    recorded into :class:`flywheel.artifact.BlockExecution` so an
    operator looking at a ``foundry/workspaces/<ws>/agent_workspaces/<id>``
    directory can find the row that produced it.

    Attributes:
        relative_dir: Path of the mount relative to the workspace
            root (e.g., ``"agent_workspaces/abc12345"``).  This
            is what gets recorded into ``BlockExecution`` for
            traceability.
        host_path: Absolute host path to the prepared directory.
        container_path: Where the directory is mounted inside the
            container.  Always ``/workspace`` today; field exists
            so future patterns can mount auxiliary workspaces.
        mode: ``"rw"`` (the agent writes here) or ``"ro"``.
            Always ``"rw"`` for the primary agent workspace.
    """

    relative_dir: str
    host_path: Path
    container_path: str = "/workspace"
    mode: str = "rw"


def prepare_agent_workspace(
    workspace: Workspace,
    output_names: list[str] | None = None,
    agent_workspace_dir: str | None = None,
    *,
    reuse_workspace: bool = False,
) -> AgentMount:
    """Prepare a fresh agent workspace directory.

    When ``agent_workspace_dir`` is ``None`` the directory is
    auto-named ``agent_workspaces/<short-uuid>`` so parallel
    launches can never collide.  When the caller passes an
    explicit name and the target directory already exists with
    content, this function raises ``FileExistsError`` instead of
    silently rmtree-ing — the legacy footgun the patterns
    campaign set out to fix.  Empty / non-existent explicit
    targets are still accepted (so tests that pre-create an
    empty dir keep working) and existing-but-empty dirs are
    reused in place.

    The returned :class:`AgentMount` records the relative dir
    name so :class:`AgentHandle` can stamp it onto the
    :class:`BlockExecution` row.

    Args:
        workspace: The flywheel workspace.
        output_names: Artifact names whose latest instances should
            be seeded into the workspace before the agent starts.
        agent_workspace_dir: Optional explicit subdirectory name.
            Pass ``None`` (the default) to get a unique
            auto-named mount; pass a string when the caller has
            its own naming convention and is willing to take
            responsibility for collision avoidance.
        reuse_workspace: When ``True``, treat an existing
            non-empty ``agent_workspace_dir`` as intentional reuse
            (no ``FileExistsError``), and skip artifact seeding.
            Used by the host-side handoff loop
            (``flywheel.agent_handoff``) to relaunch into a
            workspace that already contains the spliced
            ``agent_session.jsonl`` from a prior cycle.  Requires
            an explicit ``agent_workspace_dir``; passing both
            ``agent_workspace_dir=None`` and
            ``reuse_workspace=True`` raises ``ValueError`` because
            there is nothing to reuse.

    Returns:
        :class:`AgentMount` describing the prepared directory.

    Raises:
        FileExistsError: if ``agent_workspace_dir`` is explicit,
            the target directory exists with content, and
            ``reuse_workspace`` is ``False``.
        ValueError: if ``reuse_workspace`` is ``True`` but
            ``agent_workspace_dir`` is ``None``.
    """
    if reuse_workspace and agent_workspace_dir is None:
        raise ValueError(
            "reuse_workspace=True requires an explicit "
            "agent_workspace_dir; nothing to reuse otherwise."
        )
    if agent_workspace_dir is None:
        ws_dir_name = (
            f"{AGENT_WORKSPACES_DIR}/{uuid.uuid4().hex[:8]}")
        agent_ws = workspace.path / ws_dir_name
        # In the (cosmically unlikely) event of a uuid collision
        # against an existing dir, redraw rather than rmtree —
        # the whole point of auto-naming is "never blow away
        # someone else's workspace".
        while agent_ws.exists():
            ws_dir_name = (
                f"{AGENT_WORKSPACES_DIR}/{uuid.uuid4().hex[:8]}")
            agent_ws = workspace.path / ws_dir_name
    else:
        ws_dir_name = agent_workspace_dir
        agent_ws = workspace.path / ws_dir_name
        if (agent_ws.exists() and any(agent_ws.iterdir())
                and not reuse_workspace):
            raise FileExistsError(
                f"agent_workspace_dir={agent_workspace_dir!r} "
                f"already exists at {agent_ws} with content. "
                f"Pass agent_workspace_dir=None to auto-name "
                f"this launch under {AGENT_WORKSPACES_DIR}/, or "
                f"choose a unique name to avoid clobbering "
                f"another launch's working directory."
            )

    agent_ws.mkdir(parents=True, exist_ok=True)

    # Skip artifact seeding on reuse: the workspace already
    # carries the prior cycle's state (spliced session JSONL,
    # any intermediate files), and re-seeding could clobber it.
    if reuse_workspace:
        return AgentMount(
            relative_dir=ws_dir_name,
            host_path=agent_ws,
        )

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

    return AgentMount(
        relative_dir=ws_dir_name,
        host_path=agent_ws,
    )


def launch_agent_block(
    workspace: Workspace,
    template: Template,
    project_root: Path,
    prompt: str,
    agent_image: str = "flywheel-claude:latest",
    auth_volume: str = "claude-auth",
    model: str | None = None,
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
    agent_workspace_dir: str | None = None,
    predecessor_id: str | None = None,
    post_checks: dict[str, Any] | None = None,
    reuse_workspace: bool = False,
) -> AgentHandle:
    """Launch an agent block execution (non-blocking).

    Sets up the workspace, launches the Docker container, and
    returns an ``AgentHandle`` immediately.  The container runs
    in the background.

    The caller must call ``handle.wait()`` to clean up and get
    the result.  Call ``handle.stop()`` to request a graceful
    shutdown, then ``wait()`` to finalize.

    Args:
        workspace: The flywheel workspace for artifact tracking.
        template: The template containing block definitions.
        project_root: Path to the project root.
        prompt: The system prompt for the agent.
        agent_image: Docker image for the agent container.
        auth_volume: Docker named volume with API credentials.
        model: Model name (e.g., "claude-sonnet-4-6").
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
        isolated_network: Enable iptables-based network isolation
            inside the container.
        agent_workspace_dir: Subdirectory name for the agent workspace.
            Defaults to ``"agent_workspace"``.  Use distinct names
            when launching multiple agents in parallel against the
            same flywheel workspace.
        predecessor_id: Execution ID of a previous agent run that
            this launch resumes from.  Recorded in the workspace
            execution for resume chain tracking.
        post_checks: Reserved.  The agent path itself runs no
            post-checks; nested-block post-checks live on the
            host-side :class:`flywheel.local_block.LocalBlockRecorder`
            owned by the handoff loop.
        reuse_workspace: When ``True``, the launcher skips the
            "fresh workspace" check and the prior-artifact
            seeding pass.  Used by the host-side handoff loop
            (``flywheel.agent_handoff``) to relaunch an agent
            into the same workspace its prior cycle wrote to,
            so the spliced ``agent_session.jsonl`` is on disk
            ready for ``RESUME_SESSION_FILE``.  Requires an
            explicit ``agent_workspace_dir``.

    Returns:
        An ``AgentHandle`` for monitoring and controlling the agent.
    """
    del overrides, post_checks  # accepted for API compatibility
    mount = prepare_agent_workspace(
        workspace, output_names, agent_workspace_dir,
        reuse_workspace=reuse_workspace,
    )
    agent_ws = mount.host_path
    # Bind the resolved name back so AgentHandle records the
    # directory the run actually used.
    agent_workspace_dir = mount.relative_dir

    executions_before = len(workspace.executions)

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
        workspace=workspace,
        agent_ws=agent_ws,
        output_names=output_names,
        start_time=start,
        executions_before=executions_before,
        agent_image=agent_image,
        stdout_thread=stdout_thread,
        stderr_thread=stderr_thread,
        container_name=container_name,
        predecessor_id=predecessor_id,
        agent_workspace_dir=agent_workspace_dir,
    )


def run_agent_block(
    workspace: Workspace,
    template: Template,
    project_root: Path,
    prompt: str,
    agent_image: str = "flywheel-claude:latest",
    auth_volume: str = "claude-auth",
    model: str | None = None,
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
    agent_workspace_dir: str | None = None,
    predecessor_id: str | None = None,
    post_checks: dict[str, Any] | None = None,
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
        agent_workspace_dir=agent_workspace_dir,
        predecessor_id=predecessor_id,
        post_checks=post_checks,
    )
    try:
        return handle.wait()
    except KeyboardInterrupt:
        print("  [agent] interrupted -- requesting graceful stop")
        handle.stop(reason="keyboard_interrupt")
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
