"""Agent block execution orchestration.

Runs a Claude Code agent as a one-shot container block via
:class:`flywheel.executor.ProcessExitExecutor`.  The agent reads
its prompt from a bind-mounted ``/prompt/prompt.md``, writes
declared outputs to ``/output/<slot>/``, and persists its
conversation session through the ``/state/`` mount.  Cancellation
(operator, watchdog, natural exit) lives in
:class:`flywheel.executor.ContainerExecutionHandle`; this module's
:class:`AgentHandle` is a thin wrapper that translates the
container's :class:`flywheel.executor.ExecutionResult` into an
:class:`AgentResult` carrying agent-specific fields
(``exit_reason``, ``evals_run``).

Two APIs are provided:

- ``launch_agent_block()`` returns an ``AgentHandle`` immediately
  for non-blocking control.  The handle supports ``stop()`` to
  request a graceful shutdown and ``wait()`` to block until
  completion and collect the result.
- ``run_agent_block()`` is a blocking convenience wrapper.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flywheel.artifact import LifecycleEvent
from flywheel.executor import (
    ContainerExecutionHandle,
    ProcessExitExecutor,
)
from flywheel.template import Template
from flywheel.workspace import Workspace

# Default total timeout for an agent launch (4 hours).
DEFAULT_TOTAL_TIMEOUT = 14400


@dataclass(frozen=True)
class AgentResult:
    """Result of an agent block execution.

    Attributes:
        exit_code: The agent container's exit code.
        elapsed_s: Wall-clock time in seconds.
        evals_run: Number of nested block executions triggered
            during this agent run (computed as the delta of
            workspace-wide execution count, minus one for the
            agent's own record).
        execution_id: The workspace execution ID for this run.
        stop_reason: Why the agent was stopped, if applicable.
        exit_reason: Semantic classification of why the agent
            exited.  One of: ``"completed"``, ``"auth_failure"``,
            ``"rate_limit"``, ``"max_turns"``, ``"stopped"``,
            ``"crashed"``, ``"tool_handoff"``.  Derived from the
            ``agent_exit_state`` artifact the agent runner wrote
            on exit.
    """

    exit_code: int
    elapsed_s: float
    evals_run: int
    execution_id: str | None = None
    stop_reason: str | None = None
    exit_reason: str | None = None


@dataclass
class AgentBlockConfig:
    """Configuration for an agent block launch.

    Shared across pattern-driven launches via a base config plus
    per-role overrides; see
    :meth:`flywheel.pattern_runner.PatternRunner._kwargs_for`.

    Attributes:
        workspace: The flywheel workspace for artifact tracking.
        template: The template containing block definitions.
        project_root: Path to the project root.
        prompt: The system prompt for the agent.
        agent_image: Docker image for the agent container (kept
            for compatibility with launchers that predate
            block-YAML-authoritative images; the executor reads
            ``block_def.image`` authoritatively).
        auth_volume: Docker named volume carrying API credentials.
        model: Model name (e.g., ``"claude-sonnet-4-6"``).
        max_turns: Maximum agent conversation turns.
        total_timeout: Maximum wall-clock seconds for the run.
        source_dirs: Project source directories to mount
            read-only into the container.
        input_artifacts: Maps input slot names to artifact
            instance IDs.  Passed straight to the executor.
        overrides: Per-launch CLI flag overrides forwarded to
            the executor.
        mcp_servers: Comma-separated MCP server names enabled
            inside the agent container.
        allowed_tools: Comma-separated tool whitelist for the
            agent runner.
        extra_env: Additional environment variables merged on
            top of the block's declared env.
        extra_mounts: Additional bind mounts appended to the
            substrate's mount list (auth volume, source dirs,
            MCP servers, ...).
        isolated_network: When ``True``, add
            ``--cap-add=NET_ADMIN`` and set
            ``NETWORK_ISOLATION=1`` in the container env so the
            in-container firewall script drops outbound traffic
            by default, allowing only Anthropic API, DNS,
            loopback, and the host ports named in
            ``HOST_WHITELIST_PORTS`` (empty by default — i.e. no
            host access).  Projects that need a specific host-side
            port declare it per-instance via ``extra_env:
            HOST_WHITELIST_PORTS: <port,port,...>`` in the pattern
            YAML.
        agent_workspace_dir: Reserved; no-op under the substrate.
            The /workspace bind is always the executor's per-
            execution scratch tempdir.
        predecessor_id: Execution ID of a previous agent run
            this launch resumes from.  Recorded on the
            :class:`BlockExecution` for chain traversal.
        prompt_substitutions: ``{{KEY}} -> value`` substitutions
            applied by the pattern runner to the role's prompt
            text before launch.
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
    source_dirs: list[str] | None = None
    input_artifacts: dict[str, str] | None = None
    overrides: dict[str, Any] | None = None
    mcp_servers: str | None = None
    allowed_tools: str | None = None
    extra_env: dict[str, str] | None = None
    extra_mounts: list[tuple[str, str, str]] | None = None
    isolated_network: bool = False
    agent_workspace_dir: str | None = None
    predecessor_id: str | None = None
    prompt_substitutions: dict[str, str] | None = None


def _resolve_host_path(path: Path) -> str:
    """Normalize a host path for Docker bind mounts on Windows."""
    resolved = str(path.resolve())
    if sys.platform == "win32":
        resolved = resolved.replace("\\", "/")
    return resolved


def _build_agent_env(
    *,
    model: str | None,
    max_turns: int | None,
    mcp_servers: str | None,
    allowed_tools: str | None,
    isolated_network: bool,
    extra_env: dict[str, str] | None,
) -> dict[str, str]:
    """Assemble the env dict passed to the agent runner.

    The runner consumes ``MODEL``, ``MAX_TURNS``, ``MCP_SERVERS``,
    ``ALLOWED_TOOLS``, and (via HANDOFF_TOOLS etc.) the caller's
    own extras.  ``NETWORK_ISOLATION`` flips the in-container
    firewall script.  Caller-supplied ``extra_env`` wins on key
    collision so a per-role override always takes effect.
    """
    env: dict[str, str] = {
        "MCP_SERVERS": mcp_servers or "eval",
        "ALLOWED_TOOLS": (
            allowed_tools or "Read,Write,Edit,Glob,Grep"),
        "PYTHONUNBUFFERED": "1",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000",
    }
    if model:
        env["MODEL"] = model
    if max_turns is not None:
        env["MAX_TURNS"] = str(max_turns)
    if isolated_network:
        env["NETWORK_ISOLATION"] = "1"
    if extra_env:
        env.update(extra_env)
    return env


def _build_agent_mounts(
    *,
    project_root: Path,
    auth_volume: str,
    source_dirs: list[str] | None,
    prompt_dir: Path,
    extra_mounts: list[tuple[str, str, str]] | None,
    isolated_network: bool,
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Build the agent-specific extra mounts + docker_args.

    Returns ``(mounts, docker_args)``.  ``mounts`` is what gets
    forwarded to :meth:`ProcessExitExecutor.launch` as the
    ``extra_mounts`` kwarg; the executor appends it to the
    substrate-reserved mount list (``/input/*``, ``/output/*``,
    ``/state``, ``/workspace``).  ``docker_args`` extends
    ``block_def.docker_args`` for network-isolation runs.
    """
    mounts: list[tuple[str, str, str]] = [
        (auth_volume, "/home/claude/.claude", "rw"),
        (_resolve_host_path(prompt_dir), "/prompt", "ro"),
    ]
    if source_dirs:
        for i, src in enumerate(source_dirs):
            src_path = project_root / src
            container_path = (
                "/source" if i == 0 else f"/source{i + 1}")
            mounts.append(
                (_resolve_host_path(src_path),
                 container_path, "ro"))
    if extra_mounts:
        mounts.extend(extra_mounts)

    docker_args: list[str] = []
    if isolated_network:
        docker_args.append("--cap-add=NET_ADMIN")

    return mounts, docker_args


def _classify_exit(
    workspace: Workspace,
    execution_id: str | None,
) -> tuple[str, str | None]:
    """Classify an agent exit from the ``agent_exit_state`` artifact.

    Reads the JSON the agent runner wrote into
    ``/output/agent_exit_state/agent_exit_state.json`` (now a
    first-class artifact instance) and maps its
    ``status`` / ``reason`` fields to a semantic ``exit_reason``.
    Returns ``(exit_reason, stop_reason)`` — ``stop_reason`` is
    the :class:`BlockExecution` field populated by the executor
    when a :meth:`ContainerExecutionHandle.stop` call caused the
    exit; it takes precedence over any runner-side status.
    """
    if execution_id is None:
        return "crashed", None
    execution = workspace.executions.get(execution_id)
    if execution is None:
        return "crashed", None
    stop_reason = execution.stop_reason
    if stop_reason:
        return "stopped", stop_reason

    binding = execution.output_bindings.get("agent_exit_state")
    if binding:
        instance = workspace.artifacts.get(binding)
        if instance is not None:
            artifact_dir = (
                workspace.path / "artifacts" / instance.copy_path)
            state_file = (
                artifact_dir / "agent_exit_state.json")
            if state_file.is_file():
                try:
                    state = json.loads(
                        state_file.read_text(encoding="utf-8"))
                    status = state.get("status", "")
                    reason = state.get("reason", "")
                    if status == "tool_handoff":
                        return "tool_handoff", None
                    if "auth" in reason:
                        return "auth_failure", None
                    if "rate_limit" in reason:
                        return "rate_limit", None
                    if reason == "max_turns":
                        return "max_turns", None
                    if status == "complete":
                        return "completed", None
                except (OSError, json.JSONDecodeError):
                    pass

    if execution.exit_code == 0:
        return "completed", None
    return "crashed", None


class AgentHandle:
    """Thin adapter around a :class:`ContainerExecutionHandle`.

    ``is_alive`` / ``stop`` delegate directly to the inner
    handle.  ``wait`` blocks on the inner handle's post-exit
    pipeline, then reads the agent exit-state artifact to
    classify ``exit_reason`` and counts the workspace-execution
    delta for ``evals_run``.

    The caller **must** call ``wait()`` exactly once.  Failing to
    call it leaks the prompt tempdir and the inner handle's
    thread resources.
    """

    def __init__(
        self,
        *,
        inner: ContainerExecutionHandle,
        workspace: Workspace,
        executions_before: int,
        prompt_tempdir: Path,
    ) -> None:
        """Capture references the adapter needs to build AgentResult."""
        self._inner = inner
        self._workspace = workspace
        self._executions_before = executions_before
        self._prompt_tempdir = prompt_tempdir
        self._waited = False

    def is_alive(self) -> bool:
        """Whether the inner container is still running."""
        return self._inner.is_alive()

    def stop(self, reason: str = "requested") -> None:
        """Forward to the inner handle's two-phase cancellation."""
        self._inner.stop(reason=reason)

    def wait(self) -> AgentResult:
        """Block for container exit and return :class:`AgentResult`.

        Cleans up the prompt tempdir in a ``finally`` so a raise
        during the inner wait doesn't leak it.

        Raises:
            RuntimeError: If called more than once.
        """
        if self._waited:
            raise RuntimeError(
                "wait() already called on this AgentHandle")
        self._waited = True
        try:
            result = self._inner.wait()
        finally:
            shutil.rmtree(
                self._prompt_tempdir, ignore_errors=True)

        # +1 for the agent's own execution record.
        evals_run = max(
            0,
            len(self._workspace.executions)
            - self._executions_before - 1,
        )

        exit_reason, stop_reason = _classify_exit(
            self._workspace, result.execution_id)

        if stop_reason:
            execution = self._workspace.executions.get(
                result.execution_id)
            if execution is not None:
                event = LifecycleEvent(
                    id=self._workspace.generate_event_id(),
                    kind="agent_stopped",
                    timestamp=execution.finished_at,
                    execution_id=result.execution_id,
                    detail={"reason": stop_reason},
                )
                self._workspace.add_event(event)
                self._workspace.save()

        return AgentResult(
            exit_code=result.exit_code,
            elapsed_s=result.elapsed_s,
            evals_run=evals_run,
            execution_id=result.execution_id,
            stop_reason=stop_reason,
            exit_reason=exit_reason,
        )


def launch_agent_block(
    workspace: Workspace,
    template: Template,
    project_root: Path,
    prompt: str,
    block_name: str,
    agent_image: str = "flywheel-claude:latest",
    auth_volume: str = "claude-auth",
    model: str | None = None,
    max_turns: int | None = None,
    total_timeout: int = DEFAULT_TOTAL_TIMEOUT,
    source_dirs: list[str] | None = None,
    input_artifacts: dict[str, str] | None = None,
    overrides: dict[str, Any] | None = None,
    mcp_servers: str | None = None,
    allowed_tools: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_mounts: list[tuple[str, str, str]] | None = None,
    isolated_network: bool = False,
    agent_workspace_dir: str | None = None,
    predecessor_id: str | None = None,
) -> AgentHandle:
    """Launch an agent block execution (non-blocking).

    Translates agent-specific launch parameters into a
    :meth:`ProcessExitExecutor.launch` call:

    * ``prompt`` is written to a tempdir ``prompt.md`` and passed
      as an ``extra_mount`` at ``/prompt`` (the agent runner
      reads it at startup).
    * ``auth_volume`` + ``source_dirs`` + any caller
      ``extra_mounts`` append to the substrate's own mounts.
    * ``model``, ``max_turns``, ``mcp_servers``, ``allowed_tools``
      become environment variables via ``extra_env``.
    * ``block_name`` picks the block definition the executor runs;
      ``block_def.image`` / ``inputs`` / ``outputs`` are
      authoritative.

    Args:
        workspace: The flywheel workspace.
        template: The template containing block definitions.
        project_root: Project root (for resolving ``source_dirs``).
        prompt: The system prompt.  Mounted at ``/prompt/prompt.md``.
        block_name: Name of the block to execute.  Must exist in
            the template and declare ``runner: container``.
        agent_image: Reserved; the executor reads
            ``block_def.image`` authoritatively.
        auth_volume: Docker named volume carrying agent credentials.
        model: Optional model override.
        max_turns: Optional turn budget.
        total_timeout: Wall-clock cap.  Enforced by the
            substrate's watchdog thread.
        source_dirs: Project source dirs to mount read-only.
        input_artifacts: Block input slot bindings.
        overrides: CLI flag overrides forwarded to the executor.
        mcp_servers: Comma-separated MCP server names.
        allowed_tools: Comma-separated tool whitelist.
        extra_env: Additional env merged into the container's env.
        extra_mounts: Additional bind mounts appended to the
            substrate mount list.
        isolated_network: When ``True``, adds
            ``--cap-add=NET_ADMIN`` and sets
            ``NETWORK_ISOLATION=1``.  See :class:`AgentBlockConfig`
            for the whitelist contract (``HOST_WHITELIST_PORTS``).
        agent_workspace_dir: Reserved; no-op.
        predecessor_id: Not passed to the executor today (the
            substrate does not accept a predecessor on launch).
            Callers that chain executions — the handoff loop —
            are responsible for writing ``predecessor_id`` onto
            the :class:`BlockExecution` record after wait().

    Returns:
        An :class:`AgentHandle` for monitoring and controlling
        the agent.
    """
    del (overrides, agent_workspace_dir,
         agent_image, predecessor_id)

    # Pattern runners pass ``total_timeout=None`` to mean "no
    # wall-clock cap"; the executor treats that the same way.
    timeout_s = (
        float(total_timeout) if total_timeout else None)

    prompt_tempdir = Path(tempfile.mkdtemp(
        prefix=f"flywheel-prompt-{block_name}-"))
    (prompt_tempdir / "prompt.md").write_text(
        prompt, encoding="utf-8")

    mounts, docker_args = _build_agent_mounts(
        project_root=project_root,
        auth_volume=auth_volume,
        source_dirs=source_dirs,
        prompt_dir=prompt_tempdir,
        extra_mounts=extra_mounts,
        isolated_network=isolated_network,
    )

    env = _build_agent_env(
        model=model,
        max_turns=max_turns,
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
        isolated_network=isolated_network,
        extra_env=extra_env,
    )

    log_dir = workspace.path / "logs" / block_name
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessExitExecutor(template)
    executions_before = len(workspace.executions)

    try:
        inner = executor.launch(
            block_name=block_name,
            workspace=workspace,
            input_bindings=input_artifacts or {},
            extra_env=env,
            extra_mounts=mounts,
            extra_docker_args=docker_args or None,
            log_dir=log_dir,
            total_timeout_s=timeout_s,
        )
    except BaseException:
        shutil.rmtree(prompt_tempdir, ignore_errors=True)
        raise

    if not isinstance(inner, ContainerExecutionHandle):
        # Pre-container failure: the executor returned a
        # :class:`SyncExecutionHandle`.  Still wrap it so the
        # caller's wait/stop/is_alive protocol works uniformly;
        # wait() will get back a pre-completed ExecutionResult
        # and classify exit_reason as "crashed".
        return AgentHandle(
            inner=inner,  # type: ignore[arg-type]
            workspace=workspace,
            executions_before=executions_before,
            prompt_tempdir=prompt_tempdir,
        )

    return AgentHandle(
        inner=inner,
        workspace=workspace,
        executions_before=executions_before,
        prompt_tempdir=prompt_tempdir,
    )


def run_agent_block(
    workspace: Workspace,
    template: Template,
    project_root: Path,
    prompt: str,
    block_name: str,
    **kwargs: Any,
) -> AgentResult:
    """Run an agent block to completion (blocking).

    Convenience wrapper: launch, wait, return the result.  On
    KeyboardInterrupt, requests a graceful stop before
    propagating the exception so the container's teardown path
    runs.
    """
    handle = launch_agent_block(
        workspace=workspace,
        template=template,
        project_root=project_root,
        prompt=prompt,
        block_name=block_name,
        **kwargs,
    )
    try:
        return handle.wait()
    except KeyboardInterrupt:
        # The inner handle's wait() raised after setting the
        # single-wait sentinel on the adapter, so a second
        # ``handle.wait()`` would error.  Forward the operator
        # signal as a stop() so the substrate's two-phase
        # cancellation has a chance to terminate the container
        # if it is still running, then propagate the interrupt.
        print("  [agent] interrupted -- requesting graceful stop")
        handle.stop(reason="keyboard_interrupt")
        raise
