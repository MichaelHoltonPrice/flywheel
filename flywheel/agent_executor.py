"""Agent battery wrapped as a :class:`BlockExecutor`.

The agent block is a "battery" — a block whose body is performed
by a prepackaged implementation (Claude Code in a container,
prompt mounting, auth volume, MCP server wiring, allowed-tools
whitelist, isolated network, host-side handoff pause/resume
loop).  Today's callers reach that machinery directly through
:func:`flywheel.agent.launch_agent_block` and
:func:`flywheel.agent_handoff.launch_agent_with_handoffs`.

This module wraps that machinery as a
:class:`flywheel.executor.BlockExecutor` so runners can launch
agent blocks through the same protocol they use for any other
block.  From the runner's vantage point an :class:`AgentExecutor`
and a :class:`flywheel.executor.ProcessExitExecutor` are
indistinguishable: both implement ``launch()`` and return an
:class:`flywheel.executor.ExecutionHandle`.

Constructor takes battery-level defaults the project supplies
once (image, auth volume, model, max_turns, ...).  Per-instance
knobs flow through ``launch(overrides=...)`` and are layered on
top of the constructor defaults at launch time.

Slice-2 scope: purely additive.  No existing caller is changed.
The pattern runner and CLI continue to invoke
``launch_agent_block`` / ``launch_agent_with_handoffs``
directly.  Slice 3 will route the pattern runner through
:class:`AgentExecutor`; slice 5 will retire ``flywheel run agent``
in favour of ``flywheel run block`` against a battery-declared
agent block.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flywheel.agent import (
    DEFAULT_TOTAL_TIMEOUT,
    AgentResult,
    launch_agent_block,
)
from flywheel.agent_handoff import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_RESUME_PROMPT,
    BlockRunner,
    HandoffContext,
    launch_agent_with_handoffs,
)
from flywheel.executor import ExecutionHandle, ExecutionResult
from flywheel.template import Template
from flywheel.workspace import Workspace


# --------------------------------------------------------------------
# Overrides keys consumed by AgentExecutor.launch().
#
# Centralized so callers (project hooks, future ``--override`` CLI
# parsing) and tests can reference them without literal-string
# drift.  Unknown keys in the ``overrides`` dict are ignored
# silently; this matches the protocol contract that validation is
# the executor's concern.
# --------------------------------------------------------------------

OVERRIDE_PROMPT = "prompt"
OVERRIDE_MODEL = "model"
OVERRIDE_MAX_TURNS = "max_turns"
OVERRIDE_TOTAL_TIMEOUT = "total_timeout"
OVERRIDE_MCP_SERVERS = "mcp_servers"
OVERRIDE_ALLOWED_TOOLS = "allowed_tools"
OVERRIDE_SOURCE_DIRS = "source_dirs"
OVERRIDE_AGENT_IMAGE = "agent_image"
OVERRIDE_AUTH_VOLUME = "auth_volume"
OVERRIDE_EXTRA_ENV = "extra_env"
OVERRIDE_EXTRA_MOUNTS = "extra_mounts"
OVERRIDE_ISOLATED_NETWORK = "isolated_network"
OVERRIDE_PREDECESSOR_ID = "predecessor_id"
OVERRIDE_PROMPT_SUBSTITUTIONS = "prompt_substitutions"


@dataclass(frozen=True)
class _LaunchPlan:
    """Effective values for one ``AgentExecutor.launch()`` call.

    The result of layering a per-launch ``overrides`` dict on top
    of the constructor's battery-level defaults.  Frozen so a
    misbehaving caller cannot reach back through the handle and
    mutate it after the launch is in flight.
    """

    prompt: str
    model: str | None
    max_turns: int | None
    total_timeout: int
    mcp_servers: str | None
    allowed_tools: str | None
    source_dirs: list[str] | None
    agent_image: str
    auth_volume: str
    extra_env: dict[str, str]
    extra_mounts: list[tuple[str, str, str]]
    isolated_network: bool
    predecessor_id: str | None


class AgentExecutor:
    """Run an agent block through the :class:`BlockExecutor` protocol.

    Constructor captures the project-level defaults the agent
    battery needs (image, auth volume, model, MCP wiring, the
    handoff loop's tool router and halt source).  ``launch()``
    consumes per-instance knobs from ``overrides`` and delegates
    to :func:`launch_agent_with_handoffs` (or, when no
    ``block_runner`` is configured, the leaner
    :func:`launch_agent_block`).

    The returned handle adapts the agent's
    :class:`flywheel.agent.AgentResult` into an
    :class:`flywheel.executor.ExecutionResult`, looking up the
    actual :class:`flywheel.artifact.BlockExecution` record by
    execution_id to surface ``output_bindings`` and the durable
    ``status``.  Callers that need the raw ``AgentResult`` (for
    ``exit_reason``, ``evals_run``, handoff state) call
    :meth:`AgentExecutionHandle.agent_result` after ``wait()``.

    Args:
        template: Template the executor resolves blocks against
            and forwards to the underlying launcher.
        project_root: Project root the agent's ``source_dirs``
            paths are resolved relative to.
        agent_image: Default Docker image for the agent
            container.  Per-launch override via
            ``overrides["agent_image"]``.
        auth_volume: Default named volume carrying the agent's
            credentials.  Per-launch override via
            ``overrides["auth_volume"]``.
        model: Default model name (e.g. ``"claude-sonnet-4-6"``).
            Per-launch override via ``overrides["model"]``.
        max_turns: Default max conversation turns.  Per-launch
            override via ``overrides["max_turns"]``.
        total_timeout: Default wall-clock cap.  Per-launch
            override via ``overrides["total_timeout"]``.
        source_dirs: Default project source directories to
            mount read-only.  Per-launch override via
            ``overrides["source_dirs"]``.
        mcp_servers: Default comma-separated MCP server names.
            Per-launch override via ``overrides["mcp_servers"]``.
        allowed_tools: Default comma-separated tool whitelist.
            Per-launch override via ``overrides["allowed_tools"]``.
        extra_env: Default env merged into every container's
            startup env.  Per-launch ``overrides["extra_env"]``
            *merges* on top (per-launch wins on key collision).
        extra_mounts: Default bind mounts appended to every
            launch's mount list.  Per-launch
            ``overrides["extra_mounts"]`` is *appended* after
            the constructor's mounts.
        isolated_network: Default network-isolation flag.
            Per-launch override via
            ``overrides["isolated_network"]``.
        block_runner: Tool router consulted by the host-side
            handoff loop when the agent intercepts a tool call.
            ``None`` (the default) skips the handoff loop and
            launches the agent directly via
            :func:`launch_agent_block`; the agent will exit
            cleanly only on a non-handoff exit.
        halt_source: Optional zero-arg callable the handoff
            loop polls between cycles to detect halt directives.
            Ignored when ``block_runner`` is ``None``.
        post_dispatch_fn: Optional callback fired after each
            handoff dispatch (used by the pattern runner for
            event-driven cadence).  Ignored when
            ``block_runner`` is ``None``.
        resume_prompt: Stdin payload passed to the agent on
            relaunch cycles after a tool handoff.  Ignored
            when ``block_runner`` is ``None``.
        max_iterations: Cap on handoff cycles per launch.
            Ignored when ``block_runner`` is ``None``.
    """

    def __init__(
        self,
        template: Template,
        project_root: Path,
        *,
        agent_image: str = "flywheel-claude:latest",
        auth_volume: str = "claude-auth",
        model: str | None = None,
        max_turns: int | None = None,
        total_timeout: int = DEFAULT_TOTAL_TIMEOUT,
        source_dirs: list[str] | None = None,
        mcp_servers: str | None = None,
        allowed_tools: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_mounts: list[tuple[str, str, str]] | None = None,
        isolated_network: bool = False,
        block_runner: BlockRunner | None = None,
        halt_source: Callable[[], list[Any]] | None = None,
        post_dispatch_fn: (
            Callable[[HandoffContext], None] | None) = None,
        resume_prompt: str = DEFAULT_RESUME_PROMPT,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        """Capture defaults; do not start anything yet."""
        self._template = template
        self._project_root = project_root
        self._agent_image = agent_image
        self._auth_volume = auth_volume
        self._model = model
        self._max_turns = max_turns
        self._total_timeout = total_timeout
        self._source_dirs = (
            list(source_dirs) if source_dirs else None)
        self._mcp_servers = mcp_servers
        self._allowed_tools = allowed_tools
        self._extra_env = dict(extra_env or {})
        self._extra_mounts = list(extra_mounts or [])
        self._isolated_network = isolated_network
        self._block_runner = block_runner
        self._halt_source = halt_source
        self._post_dispatch_fn = post_dispatch_fn
        self._resume_prompt = resume_prompt
        self._max_iterations = max_iterations

    def launch(
        self,
        block_name: str,
        workspace: Workspace,
        input_bindings: dict[str, str],
        *,
        execution_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        allowed_blocks: list[str] | None = None,
        state_lineage_id: str | None = None,
        run_id: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_mounts: list[tuple[str, str, str]] | None = None,
    ) -> ExecutionHandle:
        """Launch an agent block; return an :class:`ExecutionHandle`.

        ``overrides`` carries per-instance knobs; see the module
        docstring for the keys consumed.  ``extra_env`` and
        ``extra_mounts`` follow the substrate-wide
        container-extras convention (see ``executors.md``):
        ``extra_env`` merges over the constructor's defaults
        with per-launch winning on key collision;
        ``extra_mounts`` appends after the constructor's mounts.
        Callers that still pass these through ``overrides``
        (legacy battery-shape) are honoured for compatibility,
        with explicit kwargs taking precedence.
        ``execution_id``, ``allowed_blocks``,
        ``state_lineage_id`` are accepted for protocol parity
        but not consumed by today's agent machinery (the
        underlying launcher mints its own ids and the agent's
        container does not honour an external block whitelist).
        ``run_id`` is forwarded so the resulting
        :class:`flywheel.artifact.BlockExecution` is stamped
        with the caller's run grouping.
        """
        del execution_id, allowed_blocks, state_lineage_id

        plan = self._resolve_plan(
            overrides,
            kwarg_extra_env=extra_env,
            kwarg_extra_mounts=extra_mounts,
        )

        launch_kwargs: dict[str, Any] = {
            "workspace": workspace,
            "template": self._template,
            "project_root": self._project_root,
            "prompt": plan.prompt,
            "block_name": block_name,
            "agent_image": plan.agent_image,
            "auth_volume": plan.auth_volume,
            "model": plan.model,
            "max_turns": plan.max_turns,
            "total_timeout": plan.total_timeout,
            "source_dirs": plan.source_dirs,
            "input_artifacts": dict(input_bindings or {}),
            "mcp_servers": plan.mcp_servers,
            "allowed_tools": plan.allowed_tools,
            "extra_env": (
                dict(plan.extra_env) if plan.extra_env else None),
            "extra_mounts": (
                list(plan.extra_mounts)
                if plan.extra_mounts else None),
            "isolated_network": plan.isolated_network,
            "predecessor_id": plan.predecessor_id,
            "run_id": run_id,
        }

        if self._block_runner is None:
            inner = launch_agent_block(**launch_kwargs)
        else:
            inner = launch_agent_with_handoffs(
                block_runner=self._block_runner,
                halt_source=self._halt_source,
                post_dispatch_fn=self._post_dispatch_fn,
                resume_prompt=self._resume_prompt,
                max_iterations=self._max_iterations,
                **launch_kwargs,
            )

        return AgentExecutionHandle(
            inner=inner, workspace=workspace,
        )

    def _resolve_plan(
        self,
        overrides: dict[str, Any] | None,
        *,
        kwarg_extra_env: dict[str, str] | None = None,
        kwarg_extra_mounts: (
            list[tuple[str, str, str]] | None) = None,
    ) -> _LaunchPlan:
        """Layer per-launch overrides on top of constructor defaults.

        The single place where the merge policy lives.  Scalar
        keys take per-launch when present.  ``extra_env`` merges
        with per-launch winning on collision; ``extra_mounts``
        appends per-launch after constructor mounts.  When the
        caller passes ``extra_env`` / ``extra_mounts`` both as
        explicit kwargs (the substrate-wide container-extras
        seam) and via ``overrides`` (legacy battery-shape), the
        kwargs take precedence — overrides are merged in first,
        then the kwargs win on the same merge rule.  Unknown
        keys in ``overrides`` are ignored silently per the
        protocol contract.

        Raises:
            ValueError: When ``overrides`` carries no ``prompt``.
                The agent battery cannot run without a prompt;
                failing fast at launch time produces a clearer
                error than letting the agent container start
                with an empty ``/prompt/prompt.md``.
        """
        ov = dict(overrides or {})
        prompt = ov.get(OVERRIDE_PROMPT)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                "AgentExecutor.launch requires a non-empty "
                "``prompt`` in overrides; got "
                f"{prompt!r}"
            )

        substitutions = ov.get(OVERRIDE_PROMPT_SUBSTITUTIONS)
        if substitutions:
            for key, value in substitutions.items():
                prompt = prompt.replace(
                    "{{" + str(key) + "}}", str(value))

        merged_env = dict(self._extra_env)
        merged_env.update(ov.get(OVERRIDE_EXTRA_ENV) or {})
        if kwarg_extra_env:
            merged_env.update(kwarg_extra_env)

        merged_mounts = list(self._extra_mounts)
        merged_mounts.extend(ov.get(OVERRIDE_EXTRA_MOUNTS) or [])
        if kwarg_extra_mounts:
            merged_mounts.extend(kwarg_extra_mounts)

        return _LaunchPlan(
            prompt=prompt,
            model=ov.get(OVERRIDE_MODEL, self._model),
            max_turns=ov.get(
                OVERRIDE_MAX_TURNS, self._max_turns),
            total_timeout=ov.get(
                OVERRIDE_TOTAL_TIMEOUT, self._total_timeout),
            mcp_servers=ov.get(
                OVERRIDE_MCP_SERVERS, self._mcp_servers),
            allowed_tools=ov.get(
                OVERRIDE_ALLOWED_TOOLS, self._allowed_tools),
            source_dirs=ov.get(
                OVERRIDE_SOURCE_DIRS, self._source_dirs),
            agent_image=ov.get(
                OVERRIDE_AGENT_IMAGE, self._agent_image),
            auth_volume=ov.get(
                OVERRIDE_AUTH_VOLUME, self._auth_volume),
            extra_env=merged_env,
            extra_mounts=merged_mounts,
            isolated_network=ov.get(
                OVERRIDE_ISOLATED_NETWORK,
                self._isolated_network),
            predecessor_id=ov.get(OVERRIDE_PREDECESSOR_ID),
        )

    def for_pattern(
        self,
        *,
        pattern_router: Any,
        post_dispatch_fn: Callable[[str], None],
    ) -> "AgentExecutor":
        """Return a sibling executor wired into a pattern's on_tool router.

        The pattern runner builds one router from a pattern's
        ``on_tool`` instances and asks each agent executor it
        intends to launch to integrate that router.  This
        method does the integration: it merges the pattern's
        router over this executor's existing
        ``block_runner`` and adapts the runner's
        ``post_dispatch_fn(tool_name)`` cadence hook into the
        :class:`HandoffContext`-shaped callback the underlying
        handoff loop expects.  Every other constructor field
        (image, auth, model, source_dirs, mcp_servers,
        ``halt_source``, ...) is carried over verbatim so the
        sibling is a drop-in replacement for this executor on
        the pattern's launch path.

        The merge runs through
        :func:`flywheel.pattern_handoff.merge_block_runners`,
        so the same collision discipline that gates the
        runner's launch_fn wrapping applies here: if this
        executor was constructed with an opaque
        :class:`BlockRunner` (anything other than ``None`` or a
        :class:`ToolRouter`) the merge raises immediately.

        Returns a new :class:`AgentExecutor`; ``self`` is not
        mutated.  Callers that intend to dispatch the same
        block multiple times under one pattern should cache
        the sibling rather than calling this on every launch.

        Args:
            pattern_router: The pattern's on_tool router.  Tools
                it declares will be handled by it; anything else
                falls through to this executor's existing
                ``block_runner``.
            post_dispatch_fn: Cadence hook fired after each
                handoff dispatch with the dispatched tool's
                name.  Used by the pattern runner to drive
                ``every_n_executions`` cohort firing without
                holding a reference to the handoff loop's own
                callback shape.
        """
        from flywheel.pattern_handoff import (
            adapt_post_dispatch_for_handoff,
            merge_block_runners,
        )
        merged = merge_block_runners(
            pattern_router=pattern_router,
            fallback=self._block_runner,
            context_label=(
                f"AgentExecutor.for_pattern: this executor"),
            collision_label="AgentExecutor.for_pattern",
        )
        return AgentExecutor(
            template=self._template,
            project_root=self._project_root,
            agent_image=self._agent_image,
            auth_volume=self._auth_volume,
            model=self._model,
            max_turns=self._max_turns,
            total_timeout=self._total_timeout,
            source_dirs=self._source_dirs,
            mcp_servers=self._mcp_servers,
            allowed_tools=self._allowed_tools,
            extra_env=self._extra_env,
            extra_mounts=self._extra_mounts,
            isolated_network=self._isolated_network,
            block_runner=merged,
            halt_source=self._halt_source,
            post_dispatch_fn=adapt_post_dispatch_for_handoff(
                post_dispatch_fn),
            resume_prompt=self._resume_prompt,
            max_iterations=self._max_iterations,
        )


class AgentExecutionHandle(ExecutionHandle):
    """Adapt an agent handle to the :class:`ExecutionHandle` protocol.

    The agent battery's underlying handles
    (:class:`flywheel.agent.AgentHandle`,
    :class:`flywheel.agent_handoff.HandoffAgentHandle`) return
    :class:`flywheel.agent.AgentResult` from ``wait()``; the
    :class:`BlockExecutor` protocol promises an
    :class:`ExecutionResult`.  This wrapper:

    * forwards ``is_alive`` / ``stop`` to the inner handle;
    * on ``wait()``, runs the inner handle to completion, looks
      up the recorded :class:`flywheel.artifact.BlockExecution`
      by ``execution_id``, and synthesises an
      :class:`ExecutionResult` from its durable
      ``output_bindings`` / ``status`` (falling back to the
      :class:`AgentResult` when no record exists, the
      pre-container failure path);
    * exposes the raw :class:`AgentResult` via
      :meth:`agent_result` for callers that need agent-specific
      fields (``exit_reason``, ``evals_run``, ``exit_state``,
      ``pending_tool_calls``).

    The wrapper does not own any extra resources; cleanup of
    the prompt tempdir, control tempdir, and (for handoff
    handles) the background loop thread is the inner handle's
    job.  Calling ``wait()`` more than once returns the cached
    :class:`ExecutionResult` from the first call rather than
    raising — the inner handle's idempotency is preserved
    because we only call its ``wait()`` once.
    """

    def __init__(
        self,
        *,
        inner: Any,
        workspace: Workspace,
    ) -> None:
        """Wrap ``inner``; defer everything else to ``wait``."""
        self._inner = inner
        self._workspace = workspace
        self._waited = False
        self._result: ExecutionResult | None = None
        self._agent_result: AgentResult | None = None

    def is_alive(self) -> bool:
        """True while the underlying agent / handoff loop is running."""
        return self._inner.is_alive()

    def stop(self, reason: str = "requested") -> None:
        """Forward to the inner handle's two-phase cancellation."""
        self._inner.stop(reason=reason)

    def wait(self) -> ExecutionResult:
        """Block for completion; return an :class:`ExecutionResult`.

        Idempotent: subsequent calls return the cached result
        instead of re-driving the inner handle (which would
        raise on double-wait).
        """
        if self._waited:
            assert self._result is not None
            return self._result

        self._waited = True
        agent_result = self._inner.wait()
        self._agent_result = agent_result

        execution_id = agent_result.execution_id
        execution = (
            self._workspace.executions.get(execution_id)
            if execution_id is not None else None
        )

        if execution is not None:
            status = execution.status
            output_bindings = dict(execution.output_bindings)
            exit_code = (
                execution.exit_code
                if execution.exit_code is not None
                else agent_result.exit_code
            )
        else:
            # Pre-container failure: the agent never produced
            # a durable :class:`BlockExecution`.  Fall back to
            # whatever ``AgentResult`` carries; mark the
            # outcome failed so callers don't mistake a missing
            # record for success.
            status = "failed"
            output_bindings = {}
            exit_code = agent_result.exit_code

        self._result = ExecutionResult(
            exit_code=exit_code,
            elapsed_s=agent_result.elapsed_s,
            output_bindings=output_bindings,
            execution_id=execution_id or "",
            status=status,
        )
        return self._result

    def agent_result(self) -> AgentResult:
        """Return the raw :class:`AgentResult` after :meth:`wait`.

        Provides access to agent-specific fields the protocol
        :class:`ExecutionResult` does not carry: ``exit_reason``,
        ``evals_run``, ``exit_state``, ``pending_tool_calls``.

        Raises:
            RuntimeError: when called before ``wait()``.
        """
        if self._agent_result is None:
            raise RuntimeError(
                "agent_result() called before wait()")
        return self._agent_result
