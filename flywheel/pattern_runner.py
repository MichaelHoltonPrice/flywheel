"""Pattern runner — drives a :class:`flywheel.pattern.Pattern` end-to-end.

Where :mod:`flywheel.pattern` defines what a pattern *is*, this
module defines how one is *executed*: continuous roles fire at
run start
and persist for the run's lifetime; ledger-driven triggers
(``every_n_executions``) fire as the workspace accumulates
matching block executions; reactive triggers (``on_request``,
``on_event``) are recognized but currently raise — the runner
documents them as a known gap so the failure mode is loud rather
than silent.

The runner is intentionally narrow.  It does not own session
resume, prompt construction, or circuit-breaking; patterns make
decisions declaratively, so the runner only needs to translate
that declaration into ``launch_agent_block`` / ``BlockGroup``
calls.

Termination
-----------

A pattern run ends when **all continuous-role agents have
finished**.  Patterns without any continuous roles are rejected
at runner start: a pattern with only ``every_n_executions`` roles
would have no driver to make the workspace grow, and would loop
forever.  Reactive-only patterns will get their own driver later
(probably an ``on_event`` trigger that fires from a workspace
file watcher).

Tests inject ``launch_fn`` and a small fake workspace; production
uses :func:`flywheel.agent.launch_agent_block` and the real
:class:`flywheel.workspace.Workspace`.  The runner does not
depend on Docker.
"""

from __future__ import annotations

import functools
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from flywheel.agent import (
    AgentBlockConfig,
    AgentResult,
    launch_agent_block,
)
from flywheel.agent_handoff import (
    BlockRunner,
    HandoffContext,
    HandoffResult,
    ToolRouter,
)
from flywheel.executor import BlockExecutor
from flywheel.pattern import (
    BlockInstance,
    ContinuousTrigger,
    EveryNExecutionsTrigger,
    OnEventTrigger,
    OnRequestTrigger,
    OnToolTrigger,
    Pattern,
    Role,
)
from flywheel.template import BlockDefinition


class _Handle(Protocol):
    """Minimal handle protocol the runner relies on.

    Both :class:`flywheel.agent.AgentHandle` and the test fakes
    satisfy this.  Kept separate from ``AgentHandle`` so the
    runner does not bind to the full Docker-backed implementation.
    """

    def is_alive(self) -> bool: ...

    def wait(self) -> Any: ...


@dataclass
class _RoleState:
    """Mutable per-role bookkeeping kept by the runner."""

    role: Role
    handles: list[_Handle] = field(default_factory=list)
    cohorts_fired: int = 0


@dataclass(frozen=True)
class InstanceRuntimeConfig:
    """Runtime knobs the pattern runner applies to one instance.

    Supplied by the project layer (e.g. cyberarc's
    ``ProjectHooks``) and forwarded to the executor on launch.
    Keyed by *instance* name, not block name, because two
    instances of the same block may eventually need different
    config (e.g. different game ids, different mount directories).

    Attributes:
        extra_env: Environment variables merged into the
            container's env on startup.
        extra_mounts: Bind mounts appended to the executor's
            standard mount list.  Each entry is
            ``(host_path, container_path, mode)``.
    """

    extra_env: dict[str, str] = field(default_factory=dict)
    extra_mounts: list[tuple[str, str, str]] = field(
        default_factory=list)


# Type alias for the caller-supplied executor factory.  Given a
# block definition (the resolved target of an ``on_tool``
# instance), the factory returns the executor that should
# dispatch calls to that block.  Today's MVP shape is "always
# return the shared RequestResponseExecutor," but the factory
# signature keeps lifecycle-per-block routing available without
# an API change when the need arrives.
ExecutorFactory = Callable[[BlockDefinition], BlockExecutor]


@dataclass
class PatternRunResult:
    """Summary of a completed pattern run.

    Attributes:
        pattern_name: The pattern that was run.
        cohorts_by_role: How many cohorts fired per role.  For
            ``continuous`` roles the value is always ``1`` (one
            cohort at run start); for ``every_n_executions``
            roles it grows over the run.
        agents_launched: Total number of individual agent
            launches across all roles, summed over cohorts and
            cardinality.
        results_by_role: Per-role list of agent results, in the
            order :class:`AgentResult` instances were collected.
            Continuous roles populate this on termination; cohort
            roles populate this as each cohort drains.
    """

    pattern_name: str
    cohorts_by_role: dict[str, int]
    agents_launched: int
    results_by_role: dict[str, list[AgentResult]]


class PatternRunner:
    """Execute a :class:`Pattern` against a workspace.

    Parameters:
        pattern: The pattern definition.
        base_config: Shared agent-launch configuration.  Per-role
            overrides (prompt, model, mcp_servers, etc.) are
            layered on top before each ``launch_fn`` call.
        launch_fn: Callable returning a handle for one agent.
            Defaults to :func:`launch_agent_block`; tests inject
            a fake.
        poll_interval_s: How often to re-scan the workspace
            ledger for trigger evaluation.  Lower values feel
            snappier but burn CPU; the default matches the
            cadence of the tools that emit ``game_step`` executions
            (sub-second is overkill).
        max_total_runtime_s: Hard wall-clock cap.  ``None`` means
            wait until all continuous handles finish naturally
            (the common case — the agent's own ``total_timeout``
            is the real ceiling).

    The runner is single-shot: call :meth:`run` exactly once.
    """

    def __init__(
        self,
        pattern: Pattern,
        *,
        base_config: AgentBlockConfig,
        launch_fn: Callable[..., _Handle] = launch_agent_block,
        poll_interval_s: float = 1.0,
        max_total_runtime_s: float | None = None,
        executor_factory: ExecutorFactory | None = None,
        per_instance_runtime_config: (
            dict[str, InstanceRuntimeConfig] | None) = None,
    ):
        """Wire a pattern to a launch path.

        ``executor_factory`` and ``per_instance_runtime_config``
        are consumed only when the pattern declares ``on_tool``
        instances.  If present, the runner composes a tool
        router from those instances and merges it into the
        block runner the handoff loop consults.  Omitting them
        when the pattern has no ``on_tool`` instances is fine;
        omitting them when it does is an error (see
        :meth:`_build_pattern_tool_router`).
        """
        self._pattern = pattern
        self._base = base_config
        self._launch_fn = launch_fn
        self._poll = poll_interval_s
        self._max_runtime = max_total_runtime_s
        self._executor_factory = executor_factory
        self._per_instance_runtime_config: dict[
            str, InstanceRuntimeConfig
        ] = dict(per_instance_runtime_config or {})

        # ``Pattern.instances`` is the canonical topology list
        # — populated for both grammars.  The runner drives
        # everything off of it; ``pattern.roles`` is preserved
        # only for legacy callers.  Each launchable instance
        # gets a synthesized :class:`Role` so the existing
        # agent-launch plumbing keeps working.  ``on_tool``
        # instances are tracked separately and dispatched via
        # the pattern's own tool router (built below).
        synthesized_roles: list[Role] = []
        self._on_tool_instances: list[BlockInstance] = []
        for inst in pattern.iter_instances():
            if isinstance(inst.trigger, OnToolTrigger):
                self._on_tool_instances.append(inst)
                continue
            synthesized_roles.append(
                _role_from_instance(inst))

        # Build the pattern-owned tool router.  When there are
        # no on_tool instances it's ``None``; the launch path
        # then leaves the caller's existing block_runner in
        # place unchanged.
        self._pattern_tool_router: ToolRouter | None = (
            self._build_pattern_tool_router())

        # Reject reactive triggers we don't yet implement.
        for role in synthesized_roles:
            if isinstance(
                    role.trigger,
                    (OnRequestTrigger, OnEventTrigger)):
                raise NotImplementedError(
                    f"Pattern {pattern.name!r} role "
                    f"{role.name!r}: trigger kind "
                    f"{role.trigger.kind!r} is declared by the "
                    f"pattern grammar but not yet implemented "
                    f"by the runner.  Track in the patterns "
                    f"campaign before authoring patterns that "
                    f"depend on it."
                )

        if not any(isinstance(r.trigger, ContinuousTrigger)
                   for r in synthesized_roles):
            raise ValueError(
                f"Pattern {pattern.name!r} has no continuous "
                f"instance; nothing would drive the run.  Add "
                f"at least one instance with "
                f"`trigger: {{kind: continuous}}`."
            )

        self._state: dict[str, _RoleState] = {
            r.name: _RoleState(role=r)
            for r in synthesized_roles
        }
        self._results: dict[str, list[AgentResult]] = {
            r.name: [] for r in synthesized_roles
        }

        # If the pattern owns tool dispatch, wrap ``launch_fn``
        # so every call forwards a merged block_runner.  The
        # merge raises at construction time on any collision
        # with a ``ToolRouter`` already pre-bound by partial.
        if self._pattern_tool_router is not None:
            self._launch_fn = self._wrap_launch_fn_with_router(
                launch_fn, self._pattern_tool_router)

    def run(self) -> PatternRunResult:
        """Drive the pattern to completion and return a summary."""
        start = time.monotonic()

        for state in self._state.values():
            if isinstance(
                    state.role.trigger, ContinuousTrigger):
                self._fire_role(state.role)

        try:
            while True:
                if self._all_continuous_done():
                    break
                if (self._max_runtime is not None
                        and time.monotonic() - start
                        >= self._max_runtime):
                    print(
                        f"  [pattern-runner] max runtime "
                        f"{self._max_runtime:.0f}s reached; "
                        f"draining handles"
                    )
                    break
                self._evaluate_ledger_triggers()
                time.sleep(self._poll)
        finally:
            self._drain_all_handles()

        return PatternRunResult(
            pattern_name=self._pattern.name,
            cohorts_by_role={
                name: st.cohorts_fired
                for name, st in self._state.items()
            },
            agents_launched=sum(
                st.cohorts_fired * st.role.cardinality
                for st in self._state.values()
            ),
            results_by_role=self._results,
        )

    def _all_continuous_done(self) -> bool:
        """Return True once every continuous-role handle is finished."""
        for state in self._state.values():
            if not isinstance(
                    state.role.trigger, ContinuousTrigger):
                continue
            for handle in state.handles:
                if handle.is_alive():
                    return False
        return True

    def _evaluate_ledger_triggers(self) -> None:
        """Fire ``every_n_executions`` cohorts that are due.

        Counts only ``status=="succeeded"`` executions that are not
        synthetic.  Synthetic-failed executions already trigger
        post-check halts via
        :class:`flywheel.local_block.LocalBlockRecorder`; counting
        them here would double-count infrastructure failures as
        "real" progress.
        """
        for state in self._state.values():
            trigger = state.role.trigger
            if not isinstance(trigger, EveryNExecutionsTrigger):
                continue

            succeeded = self._count_succeeded(trigger.of_block)
            cohorts_due = succeeded // trigger.n
            while state.cohorts_fired < cohorts_due:
                self._fire_role(state.role)

    def _count_succeeded(self, block_name: str) -> int:
        """Count non-synthetic succeeded executions of ``block_name``."""
        ws = self._base.workspace
        return sum(
            1 for ex in ws.executions.values()
            if ex.block_name == block_name
            and ex.status == "succeeded"
            and not ex.synthetic
        )

    def _fire_role(self, role: Role) -> None:
        """Launch ``role.cardinality`` agents for one trigger firing."""
        state = self._state[role.name]
        cohort_index = state.cohorts_fired

        for member_index in range(role.cardinality):
            kwargs = self._kwargs_for(
                role, cohort_index=cohort_index,
                member_index=member_index,
            )
            print(
                f"  [pattern-runner] firing role "
                f"{role.name!r} cohort {cohort_index} member "
                f"{member_index + 1}/{role.cardinality}"
            )
            handle = self._launch_fn(**kwargs)
            state.handles.append(handle)

        state.cohorts_fired += 1

    def _kwargs_for(
        self,
        role: Role,
        *,
        cohort_index: int,
        member_index: int,
    ) -> dict[str, Any]:
        """Build kwargs for one agent launch.

        Layers role overrides on top of ``base_config`` and reads
        the role's prompt file from ``project_root``.  Per-member
        differentiation comes from the launcher's auto-named
        ``agent_workspaces/<execution_id>/`` mounts.
        """
        prompt_path = self._base.project_root / role.prompt
        prompt = prompt_path.read_text(encoding="utf-8")
        for key, value in (
                self._base.prompt_substitutions or {}).items():
            prompt = prompt.replace("{{" + key + "}}", value)

        # A role's name is the block name this launch represents.
        # A role's ``block_name`` is either explicitly set (when
        # the role was synthesized from an ``instances:`` spec
        # whose ``block:`` differs from its instance name) or
        # falls back to the role's own name (legacy ``roles:``
        # grammar where role-name and block-name are the same).
        kwargs: dict[str, Any] = {
            "workspace": self._base.workspace,
            "template": self._base.template,
            "project_root": self._base.project_root,
            "prompt": prompt,
            "block_name": role.block_name or role.name,
            "agent_image": self._base.agent_image,
            "auth_volume": self._base.auth_volume,
            "model": role.model or self._base.model,
            "max_turns": (
                role.max_turns
                if role.max_turns is not None
                else self._base.max_turns
            ),
            "total_timeout": (
                role.total_timeout
                if role.total_timeout is not None
                else self._base.total_timeout
            ),
            "source_dirs": self._base.source_dirs,
            "input_artifacts": self._collect_inputs(role),
            "overrides": self._base.overrides,
            "mcp_servers": (
                role.mcp_servers or self._base.mcp_servers),
            "allowed_tools": (
                role.allowed_tools or self._base.allowed_tools),
            "extra_env": self._merge_env(role),
            "extra_mounts": self._base.extra_mounts,
            "isolated_network": self._base.isolated_network,
            "agent_workspace_dir": self._workspace_dir_for(
                role,
                cohort_index=cohort_index,
                member_index=member_index,
            ),
        }
        return kwargs

    def _merge_env(self, role: Role) -> dict[str, str] | None:
        """Layer ``role.extra_env`` on top of ``base_config.extra_env``.

        Role wins on key collision — that's the whole point of
        per-role overrides; otherwise the role couldn't, e.g.,
        scope ``PREDICTOR_PATH`` to itself.  Returning ``None``
        when both are empty matches the launcher's "no extra
        env" sentinel.
        """
        base = dict(self._base.extra_env or {})
        if role.extra_env:
            base.update(role.extra_env)
        return base or None

    def _collect_inputs(self, role: Role) -> dict[str, str] | None:
        """Resolve role.inputs to the latest registered instance IDs.

        Roles list inputs by *artifact name*; the runner picks
        the latest registered instance of each.  Roles whose
        inputs aren't registered yet pass ``None`` (the launcher
        treats this as "no inputs"), which is the right behavior
        for the play role on a fresh workspace.
        """
        if not role.inputs:
            return self._base.input_artifacts

        ws = self._base.workspace
        bindings: dict[str, str] = dict(
            self._base.input_artifacts or {})
        for name in role.inputs:
            instances = ws.instances_for(name)
            if instances:
                bindings[name] = instances[-1].id
        return bindings or None

    def _workspace_dir_for(
        self,
        role: Role,
        *,
        cohort_index: int,
        member_index: int,
    ) -> str | None:
        """Always defer to the launcher's auto-naming.

        Auto-naming is the default: every launch lands in
        ``agent_workspaces/<short-uuid>/`` so two parallel
        agents in the same workspace can never clobber each
        other.  Roles cannot share a workspace dir by accident —
        which used to happen when a project hand-set
        ``agent_workspace_dir`` to a fixed name and reused it
        for sibling agents.

        We honor an explicit ``base_config.agent_workspace_dir``
        only when a single-cardinality continuous role asks for
        it, because a few callers still want a stable path.
        Even then, the launcher refuses to clobber an existing
        directory with content (raising :class:`FileExistsError`),
        so the worst case is a loud failure rather than silent
        cross-contamination.
        """
        if (self._base.agent_workspace_dir is not None
                and isinstance(role.trigger, ContinuousTrigger)
                and role.cardinality == 1):
            return self._base.agent_workspace_dir
        return None

    # ─── Pattern-owned tool dispatch ──────────────────────────

    def _build_pattern_tool_router(
        self,
    ) -> ToolRouter | None:
        """Turn ``on_tool`` instances into a :class:`ToolRouter`.

        Returns ``None`` when the pattern declares no ``on_tool``
        instances (the runner then leaves the caller's existing
        block_runner unchanged).  Raises at construction time
        on any declared-but-unserviceable instance:

        * Missing ``executor_factory``: the caller must provide
          one whenever the pattern has at least one ``on_tool``
          trigger.
        * Target block not in the template: the pattern
          references a block the project's registry doesn't know.
        * Target block has an ``inputs:`` list with count != 1:
          the tool-input serialisation convention (JSON into the
          single declared input slot) only works for 1-input
          blocks.  Multi-input blocks must wait for an explicit
          ``input_slot`` or ``input_map`` field on the trigger.
        """
        if not self._on_tool_instances:
            return None
        if self._executor_factory is None:
            raise ValueError(
                f"Pattern {self._pattern.name!r} declares "
                f"{len(self._on_tool_instances)} on_tool "
                f"instance(s) but no ``executor_factory`` was "
                f"supplied to PatternRunner; cannot dispatch."
            )
        routes: dict[str, BlockRunner] = {}
        for inst in self._on_tool_instances:
            trigger = inst.trigger
            assert isinstance(trigger, OnToolTrigger)
            block_def = self._find_block(inst.block)
            if block_def is None:
                raise ValueError(
                    f"Pattern {self._pattern.name!r} on_tool "
                    f"instance {inst.name!r} references "
                    f"unknown block {inst.block!r}"
                )
            if len(block_def.inputs) != 1:
                raise ValueError(
                    f"Pattern {self._pattern.name!r} on_tool "
                    f"dispatch to block {block_def.name!r} "
                    f"requires exactly one declared input slot "
                    f"(tool_input is serialised as JSON into "
                    f"that slot); block has "
                    f"{len(block_def.inputs)} inputs.  "
                    f"Declare a wrapper block or extend the "
                    f"trigger with an explicit input mapping."
                )
            # Tool-input serialisation writes a file and calls
            # ``workspace.register_artifact``, which only
            # works for copy artifacts.  Reject incremental or
            # other kinds at construction so the author sees a
            # clear error instead of a runtime failure the
            # first time the tool fires.
            slot_name = block_def.inputs[0].name
            declared_kind = (
                self._base.workspace.artifact_declarations
                .get(slot_name)
            )
            if declared_kind not in (None, "copy"):
                raise ValueError(
                    f"Pattern {self._pattern.name!r} on_tool "
                    f"dispatch to block {block_def.name!r}: "
                    f"input slot {slot_name!r} is declared as "
                    f"{declared_kind!r} but the tool-input "
                    f"bridge only supports ``copy`` artifacts "
                    f"(it serialises tool_input to a single "
                    f"JSON file and registers it as a fresh "
                    f"instance).  Declare the artifact as "
                    f"``kind: copy`` or extend the trigger "
                    f"with an explicit input mapping."
                )
            runtime_cfg = (
                self._per_instance_runtime_config.get(
                    inst.name, InstanceRuntimeConfig())
            )
            runner = self._make_on_tool_runner(
                block_def=block_def,
                runtime_cfg=runtime_cfg,
            )
            if trigger.tool in routes:
                raise ValueError(
                    f"Pattern {self._pattern.name!r}: tool "
                    f"{trigger.tool!r} is declared by multiple "
                    f"on_tool instances"
                )
            routes[trigger.tool] = runner
        return ToolRouter(routes)

    def _make_on_tool_runner(
        self,
        *,
        block_def: BlockDefinition,
        runtime_cfg: InstanceRuntimeConfig,
    ) -> BlockRunner:
        """Build the callable that dispatches one tool → block.

        Called once per ``on_tool`` instance at construction
        time.  The resulting callable writes ``ctx.tool_input``
        as JSON into the block's single declared input slot,
        registers it as a fresh artifact instance, and invokes
        the executor with the per-instance runtime config
        forwarded as ``extra_env`` / ``extra_mounts``.
        """
        assert self._executor_factory is not None
        workspace = self._base.workspace
        executor = self._executor_factory(block_def)
        slot_name = block_def.inputs[0].name

        def _runner(ctx: HandoffContext) -> HandoffResult:
            # Every failure from here to the end is mapped to a
            # ``HandoffResult(is_error=True)`` so no exception
            # can escape into the handoff loop's uncaught path.
            # Earlier versions wrapped only the ``executor.launch``
            # call, letting ``json.dumps`` or
            # ``register_artifact`` failures blow up the loop.
            try:
                try:
                    payload = json.dumps(ctx.tool_input)
                except (TypeError, ValueError) as exc:
                    return HandoffResult(
                        content=(
                            f"ERROR: on_tool dispatch to "
                            f"{block_def.name!r} cannot "
                            f"serialize tool_input: {exc}"),
                        is_error=True,
                    )
                tmp = Path(
                    tempfile.mkdtemp(prefix="on-tool-"))
                try:
                    (tmp / f"{slot_name}.json").write_text(
                        payload, encoding="utf-8")
                    instance = workspace.register_artifact(
                        slot_name, tmp,
                        source=(
                            f"on_tool {ctx.tool_name!r} -> "
                            f"{block_def.name}"),
                    )
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                handle = executor.launch(
                    block_name=block_def.name,
                    workspace=workspace,
                    input_bindings={
                        slot_name: instance.id},
                    extra_env=(
                        dict(runtime_cfg.extra_env)
                        if runtime_cfg.extra_env else None),
                    extra_mounts=(
                        list(runtime_cfg.extra_mounts)
                        if runtime_cfg.extra_mounts else None),
                )
                result = handle.wait()
            except Exception as exc:  # noqa: BLE001
                return HandoffResult(
                    content=(
                        f"ERROR: on_tool dispatch to "
                        f"{block_def.name!r} failed: {exc}"
                    ),
                    is_error=True,
                )
            if result.status == "succeeded":
                return HandoffResult(content="OK")
            execution = workspace.executions.get(
                result.execution_id)
            err = (
                execution.error if execution is not None
                else result.status
            )
            return HandoffResult(
                content=f"ERROR: {err}", is_error=True,
            )

        return _runner

    def _find_block(
        self, block_name: str,
    ) -> BlockDefinition | None:
        """Return the block definition with this name, or ``None``."""
        for b in self._base.template.blocks:
            if b.name == block_name:
                return b
        return None

    def _wrap_launch_fn_with_router(
        self,
        launch_fn: Callable[..., _Handle],
        pattern_router: ToolRouter,
    ) -> Callable[..., _Handle]:
        """Produce a launch_fn that forwards a merged block_runner.

        Construction-time invariant: when the pattern declares
        ``on_tool`` triggers, any pre-bound ``block_runner`` on
        ``launch_fn`` must be a :class:`ToolRouter` (or absent).
        Opaque callables are rejected — the whole point of the
        "pattern is authoritative" guarantee is that we can see
        every tool both sides will dispatch; an opaque fallback
        could silently handle a tool the pattern also claims,
        producing exactly the split-brain dispatch this check
        exists to prevent.

        Once both sides are :class:`ToolRouter`s the merge
        unions the routes; any tool name declared by both
        raises immediately.
        """
        existing_router: Any = None
        if isinstance(launch_fn, functools.partial):
            existing_router = launch_fn.keywords.get(
                "block_runner")
        if (existing_router is not None
                and not isinstance(existing_router, ToolRouter)):
            raise ValueError(
                f"Pattern {self._pattern.name!r}: declares "
                f"{len(self._on_tool_instances)} on_tool "
                f"instance(s) but the caller's ``launch_fn`` "
                f"has a pre-bound ``block_runner`` of type "
                f"{type(existing_router).__name__!r} that is "
                f"not a ``ToolRouter``.  Wrap the caller's "
                f"routing in ``make_tool_router(...)`` so the "
                f"pattern runner can statically detect tool-"
                f"name collisions and honour the 'pattern is "
                f"authoritative' guarantee."
            )
        if isinstance(existing_router, ToolRouter):
            overlap = (
                pattern_router.tools()
                & existing_router.tools()
            )
            if overlap:
                raise ValueError(
                    f"Pattern {self._pattern.name!r}: tool(s) "
                    f"{sorted(overlap)} are declared by both "
                    f"the pattern's on_tool instances and the "
                    f"caller-supplied block_runner; refusing "
                    f"to dispatch one silently over the other"
                )
        merged = _MergedRouter(
            pattern_router=pattern_router,
            fallback=existing_router,
        )

        def _wrapped(**kwargs: Any) -> _Handle:
            kwargs["block_runner"] = merged
            return launch_fn(**kwargs)

        return _wrapped

    def _drain_all_handles(self) -> None:
        """Wait on every launched handle, recording results.

        Sequential drain serializes workspace writes (mirrors
        :class:`flywheel.block_group.BlockGroup`).  Errors from
        individual ``wait()`` calls are caught and logged so one
        crashed agent cannot strand sibling cohorts.
        """
        for state in self._state.values():
            for handle in state.handles:
                try:
                    result = handle.wait()
                except Exception as exc:
                    print(
                        f"  [pattern-runner] role "
                        f"{state.role.name!r} agent wait() "
                        f"raised: {exc!r}"
                    )
                    continue
                self._results[state.role.name].append(result)

    @staticmethod
    def from_pattern_file(
        pattern_path: Path,
        *,
        base_config: AgentBlockConfig,
        block_registry: Any | None = None,
        **runner_kwargs: Any,
    ) -> PatternRunner:
        """Convenience: load a pattern from disk and wrap it.

        Re-exports :meth:`Pattern.from_yaml` so callers who only
        have a path on hand do not need a second import.  Kept
        out of :mod:`flywheel.pattern` so the loader stays free
        of any runner dependencies.
        """
        pattern = Pattern.from_yaml(
            pattern_path, block_registry=block_registry)
        return PatternRunner(
            pattern, base_config=base_config, **runner_kwargs,
        )


class _MergedRouter:
    """Layer a pattern-owned :class:`ToolRouter` over a fallback.

    The pattern router handles every tool it declares; anything
    else is forwarded to the ``fallback`` block runner (usually
    a project-hooks-provided router for the handoff tools the
    pattern has not yet migrated to ``on_tool``).  Overlap is
    detected and rejected at construction time in
    :meth:`PatternRunner._wrap_launch_fn_with_router`; this
    class assumes the merge is disjoint and trusts its builder.
    """

    def __init__(
        self,
        *,
        pattern_router: ToolRouter,
        fallback: BlockRunner | None,
    ) -> None:
        """Capture both routers."""
        self._pattern = pattern_router
        self._fallback = fallback

    def __call__(
        self, ctx: HandoffContext,
    ) -> HandoffResult:
        """Dispatch to the pattern router's tools first."""
        if ctx.tool_name in self._pattern.tools():
            return self._pattern(ctx)
        if self._fallback is not None:
            return self._fallback(ctx)
        return HandoffResult(
            content=(
                f"ERROR: no runner for tool "
                f"{ctx.tool_name!r}"
            ),
            is_error=True,
        )


def _role_from_instance(inst: BlockInstance) -> Role:
    """Synthesise a :class:`Role` equivalent of an instance.

    The runner's launch / trigger plumbing is driver-by-role
    today.  For patterns that declare an ``instances:`` topology
    we build a ``Role`` carrying the instance's agent-specific
    config alongside an explicit ``block_name`` so the launcher
    runs the right block even when its name differs from the
    instance name.

    ``prompt`` is required on the generated role (the agent
    launcher needs it); instances without a prompt are only
    meaningful with triggers this helper never sees
    (``on_tool``), so an empty prompt here indicates a bug
    upstream.
    """
    return Role(
        name=inst.name,
        prompt=inst.prompt or "",
        trigger=inst.trigger,
        model=inst.model,
        cardinality=inst.cardinality,
        inputs=list(inst.inputs),
        outputs=list(inst.outputs),
        mcp_servers=inst.mcp_servers,
        allowed_tools=inst.allowed_tools,
        max_turns=inst.max_turns,
        total_timeout=inst.total_timeout,
        extra_env=dict(inst.extra_env),
        block_name=inst.block,
    )
