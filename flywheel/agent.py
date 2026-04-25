"""Agent battery block execution types."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flywheel.artifact import LifecycleEvent
from flywheel.artifact_validator import ArtifactValidatorRegistry
from flywheel.container import ContainerConfig, run_container
from flywheel.execution import (
    ExecutionPlan,
    RuntimeResult,
    observe_one_shot_container_exit,
)
from flywheel.executor import ExecutionHandle
from flywheel.state_validator import StateValidatorRegistry
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
            ``"crashed"``, ``"tool_handoff"``.  Derived from
            ``exit_state`` below.
        exit_state: The parsed ``agent_exit_state.json`` the agent
            runner wrote under ``/flywheel/control/`` before exit.
            ``None`` when the container never produced one (pre-
            container failure, runner crash before it could
            write).  When set, carries ``session_id``, ``status``,
            ``reason``, ``timestamp``.
        pending_tool_calls: The parsed ``pending_tool_calls.json``
            the agent runner wrote when a handoff fired.  Empty
            list on a clean non-handoff exit; ``None`` when the
            container never produced the file at all (pre-
            container failure or handoff that bypassed the runner).
    """

    exit_code: int
    elapsed_s: float
    evals_run: int
    execution_id: str | None = None
    stop_reason: str | None = None
    exit_reason: str | None = None
    exit_state: dict[str, Any] | None = None
    pending_tool_calls: list[dict[str, Any]] | None = None


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
            The /scratch bind is always the executor's per-
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
    run_id: str | None = None
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

    Returns ``(mounts, docker_args)``. ``docker_args`` extends
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


def _preserve_malformed(
    path: Path, preserve_dir: Path | None,
) -> None:
    """Copy a malformed control file into ``preserve_dir`` for post-mortem.

    Control files live in a per-execution tempdir that the
    launcher cleans up in ``AgentHandle.wait``'s finally, so a
    malformed payload would otherwise vanish before an operator
    could inspect it.  Copy the raw bytes out before the cleanup
    erases them.  Best-effort: if the copy itself fails we
    silently skip — the cleanup is more important than the
    forensic trail.
    """
    if preserve_dir is None:
        return
    try:
        preserve_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            path, preserve_dir / f"{path.name}.malformed")
    except OSError:
        pass


def _read_control_json(
    path: Path, *, preserve_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Read a JSON object from a control file or return ``None``.

    Used to pick up ``agent_exit_state.json`` after the container
    exits.  Returns ``None`` when the file is absent (container
    crashed before writing) or malformed (treat as "no signal"
    rather than raising during post-exit cleanup).  Malformed
    files are copied into ``preserve_dir`` before the caller's
    tempdir cleanup erases them.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _preserve_malformed(path, preserve_dir)
        return None
    if not isinstance(data, dict):
        _preserve_malformed(path, preserve_dir)
        return None
    return data


def _read_pending_list(
    path: Path, *, preserve_dir: Path | None = None,
) -> list[dict[str, Any]] | None:
    """Read the ``pending_tool_calls.json`` payload.

    The on-disk shape is ``{"pending": [...]}`` (with a schema
    version wrapper).  Returns the inner list, or ``None`` when
    the file is absent or malformed.  Malformed files are copied
    into ``preserve_dir`` before the caller's tempdir cleanup
    erases them.
    """
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _preserve_malformed(path, preserve_dir)
        return None
    if not isinstance(payload, dict):
        _preserve_malformed(path, preserve_dir)
        return None
    pending = payload.get("pending", [])
    if not isinstance(pending, list):
        _preserve_malformed(path, preserve_dir)
        return None
    return [p for p in pending if isinstance(p, dict)]


@dataclass
class _AgentRuntimeObservation:
    """Agent-specific files observed after the container exits."""

    exit_state: dict[str, Any] | None = None
    pending_tool_calls: list[dict[str, Any]] | None = None


class _CanonicalAgentRunner:
    """One-shot runner that adds agent mounts/env, not commit logic."""

    def __init__(
        self,
        *,
        prompt: str,
        agent_image: str,
        auth_volume: str,
        model: str | None,
        max_turns: int | None,
        source_dirs: list[str] | None,
        mcp_servers: str | None,
        allowed_tools: str | None,
        extra_env: dict[str, str] | None,
        extra_mounts: list[tuple[str, str, str]] | None,
        isolated_network: bool,
        project_root: Path,
    ) -> None:
        """Capture agent-specific runtime configuration."""
        self._prompt = prompt
        self._agent_image = agent_image
        self._auth_volume = auth_volume
        self._model = model
        self._max_turns = max_turns
        self._source_dirs = source_dirs
        self._mcp_servers = mcp_servers
        self._allowed_tools = allowed_tools
        self._extra_env = extra_env
        self._extra_mounts = extra_mounts
        self._isolated_network = isolated_network
        self._project_root = project_root
        self.observation = _AgentRuntimeObservation()

    def run(
        self, plan: ExecutionPlan, args: list[str] | None = None,
    ) -> RuntimeResult:
        """Run the prepared plan with agent-specific runtime wiring."""
        prompt_dir = plan.proposals_root / "_agent_prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.md").write_text(
            self._prompt, encoding="utf-8")

        control_dir = plan.termination_file.parent / "control"
        control_dir.mkdir(parents=True, exist_ok=True)

        agent_mounts, agent_docker_args = _build_agent_mounts(
            project_root=self._project_root,
            auth_volume=self._auth_volume,
            source_dirs=self._source_dirs,
            prompt_dir=prompt_dir,
            extra_mounts=self._extra_mounts,
            isolated_network=self._isolated_network,
        )
        mounts = [*plan.mounts, *agent_mounts]
        env = dict(plan.block_def.env)
        env.update(_build_agent_env(
            model=self._model,
            max_turns=self._max_turns,
            mcp_servers=self._mcp_servers,
            allowed_tools=self._allowed_tools,
            isolated_network=self._isolated_network,
            extra_env=self._extra_env,
        ))
        image = plan.block_def.image or self._agent_image
        config = ContainerConfig(
            image=image,
            docker_args=[*plan.block_def.docker_args, *agent_docker_args],
            env=env,
            mounts=mounts,
        )
        try:
            result = run_container(config, args)
        finally:
            self.observation = _AgentRuntimeObservation(
                exit_state=_read_control_json(
                    control_dir / "agent_exit_state.json",
                    preserve_dir=None,
                ),
                pending_tool_calls=_read_pending_list(
                    control_dir / "pending_tool_calls.json",
                    preserve_dir=None,
                ),
            )
        return observe_one_shot_container_exit(plan, result)


def _classify_exit(
    workspace: Workspace,
    execution_id: str | None,
    exit_state: dict[str, Any] | None,
) -> tuple[str, str | None]:
    """Classify an agent exit from the parsed exit-state dict.

    ``exit_state`` is the parsed ``agent_exit_state.json`` the
    agent runner wrote under ``/flywheel/control/`` before exit;
    ``None`` when the container never produced one (pre-container
    failure or crash before the finally block ran).  The dict
    carries ``status``, ``reason``, and is mapped here to a
    semantic ``exit_reason``.

    Returns ``(exit_reason, stop_reason)`` — ``stop_reason`` is
    the :class:`BlockExecution` field populated by the executor
    when a :meth:`ExecutionHandle.stop` call caused the
    exit; it takes precedence over any runner-side status.
    """
    if execution_id is None:
        return "crashed", None
    execution = workspace.executions.get(execution_id)
    if execution is None:
        return "crashed", None
    stop_reason = getattr(execution, "stop_reason", None)
    if stop_reason:
        return "stopped", stop_reason

    if exit_state is not None:
        status = exit_state.get("status", "")
        reason = exit_state.get("reason", "")
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

    if execution.exit_code == 0:
        return "completed", None
    return "crashed", None


class AgentHandle:
    """Thin adapter around an :class:`ExecutionHandle`.

    ``is_alive`` / ``stop`` delegate directly to the inner
    handle.  ``wait`` blocks on the inner handle's post-exit
    pipeline, reads the control files the agent runner wrote
    under ``/flywheel/control/`` (parsed into
    :class:`AgentResult`'s ``exit_state`` and
    ``pending_tool_calls`` fields), classifies ``exit_reason``
    from the parsed exit state, and counts the workspace-
    execution delta for ``evals_run``.

    The caller **must** call ``wait()`` exactly once.  Failing to
    call it leaks the prompt tempdir, the control tempdir, and
    the inner handle's thread resources.
    """

    def __init__(
        self,
        *,
        inner: ExecutionHandle,
        workspace: Workspace,
        executions_before: int,
        prompt_tempdir: Path,
        control_tempdir: Path | None,
        log_dir: Path | None = None,
    ) -> None:
        """Capture references the adapter needs to build AgentResult.

        ``control_tempdir`` is the host-side directory visible to
        the container at ``/flywheel/control``. ``None`` when the
        launcher skipped the mount (pre-container failure path
        where the inner handle is synchronous). :meth:`wait` reads
        ``pending_tool_calls.json`` and ``agent_exit_state.json``
        from it before cleanup.

        ``log_dir`` is the per-block log directory.  If a control
        file is present but malformed, :meth:`wait` copies it here
        before erasing the control tempdir so an operator can
        still post-mortem a truncated / garbage payload.
        """
        self._inner = inner
        self._workspace = workspace
        self._executions_before = executions_before
        self._prompt_tempdir = prompt_tempdir
        self._control_tempdir = control_tempdir
        self._log_dir = log_dir
        self._waited = False

    def is_alive(self) -> bool:
        """Whether the inner container is still running."""
        return self._inner.is_alive()

    def stop(self, reason: str = "requested") -> None:
        """Forward to the inner handle's two-phase cancellation."""
        self._inner.stop(reason=reason)

    def wait(self) -> AgentResult:
        """Block for container exit and return :class:`AgentResult`.

        Reads the control files before cleaning up the tempdirs.
        Both tempdirs are removed in a ``finally`` so a raise
        during the inner wait doesn't leak them.

        Raises:
            RuntimeError: If called more than once.
        """
        if self._waited:
            raise RuntimeError(
                "wait() already called on this AgentHandle")
        self._waited = True
        exit_state: dict[str, Any] | None = None
        pending_tool_calls: list[dict[str, Any]] | None = None
        try:
            result = self._inner.wait()
            if self._control_tempdir is not None:
                preserve_dir: Path | None = None
                if self._log_dir is not None:
                    exec_id = result.execution_id or "unknown"
                    preserve_dir = (
                        self._log_dir / f"{exec_id}-control")
                exit_state = _read_control_json(
                    self._control_tempdir
                    / "agent_exit_state.json",
                    preserve_dir=preserve_dir,
                )
                pending_tool_calls = _read_pending_list(
                    self._control_tempdir
                    / "pending_tool_calls.json",
                    preserve_dir=preserve_dir,
                )
        finally:
            shutil.rmtree(
                self._prompt_tempdir, ignore_errors=True)
            if self._control_tempdir is not None:
                shutil.rmtree(
                    self._control_tempdir, ignore_errors=True)

        # +1 for the agent's own execution record.
        evals_run = max(
            0,
            len(self._workspace.executions)
            - self._executions_before - 1,
        )

        exit_reason, stop_reason = _classify_exit(
            self._workspace, result.execution_id, exit_state)

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
            exit_state=exit_state,
            pending_tool_calls=pending_tool_calls,
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
    run_id: str | None = None,
) -> AgentHandle:
    """Launch an agent block execution.

    Agent execution is a Flywheel batteries-included feature. The
    handle-based container executor it used to depend on has been
    removed; until the agent battery is rebuilt on the canonical block
    execution pipeline, this entry point stays importable and fails
    explicitly instead of breaking ``flywheel run block`` at import time.
    """
    del (
        workspace, template, project_root, prompt, block_name,
        agent_image, auth_volume, model, max_turns, total_timeout,
        source_dirs, input_artifacts, overrides, mcp_servers,
        allowed_tools, extra_env, extra_mounts, isolated_network,
        agent_workspace_dir, predecessor_id, run_id,
    )
    raise NotImplementedError(
        "the agent battery has not yet been rebuilt on the canonical "
        "block execution pipeline"
    )


def run_agent_block(
    workspace: Workspace,
    template: Template,
    project_root: Path,
    prompt: str,
    block_name: str,
    agent_image: str = "flywheel-claude:latest",
    auth_volume: str = "claude-auth",
    model: str | None = None,
    max_turns: int | None = None,
    source_dirs: list[str] | None = None,
    input_artifacts: dict[str, str] | None = None,
    mcp_servers: str | None = None,
    allowed_tools: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_mounts: list[tuple[str, str, str]] | None = None,
    isolated_network: bool = False,
    state_lineage_key: str | None = None,
    validator_registry: ArtifactValidatorRegistry | None = None,
    state_validator_registry: StateValidatorRegistry | None = None,
) -> AgentResult:
    """Deprecated compatibility entry point.

    Claude batteries are invoked as ordinary blocks through
    ``flywheel run block`` using battery-provided images and block
    declarations.
    """
    del (
        workspace, template, project_root, prompt, block_name,
        agent_image, auth_volume, model, max_turns, source_dirs,
        input_artifacts, mcp_servers, allowed_tools, extra_env,
        extra_mounts, isolated_network, state_lineage_key,
        validator_registry, state_validator_registry,
    )
    raise NotImplementedError(
        "agent blocks are invoked through flywheel run block with "
        "a battery image and block declaration"
    )
