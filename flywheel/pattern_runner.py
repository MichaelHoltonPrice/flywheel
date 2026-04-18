"""Pattern runner — drives a :class:`flywheel.pattern.Pattern` end-to-end.

The runner is the second half of the patterns campaign.  Where
:mod:`flywheel.pattern` defines what a pattern *is*, this module
defines how one is *executed*: continuous roles fire at run start
and persist for the run's lifetime; ledger-driven triggers
(``every_n_executions``) fire as the workspace accumulates
matching block executions; reactive triggers (``on_request``,
``on_event``) are recognized but currently raise — the runner
documents them as a known gap so the failure mode is loud rather
than silent.

The runner is intentionally narrow.  It does not own session
resume, prompt construction, or circuit-breaking — those concerns
belonged to the legacy :class:`flywheel.agent_loop.AgentLoop`
because that loop also did the hooks-driven *decide* step.
Patterns make decisions declaratively, so the runner only needs
to translate that declaration into ``launch_agent_block`` /
``BlockGroup`` calls.

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
from flywheel.pattern import (
    ContinuousTrigger,
    EveryNExecutionsTrigger,
    OnEventTrigger,
    OnRequestTrigger,
    Pattern,
    Role,
)


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
            cadence of the tools that emit ``game_step`` rows
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
    ):
        self._pattern = pattern
        self._base = base_config
        self._launch_fn = launch_fn
        self._poll = poll_interval_s
        self._max_runtime = max_total_runtime_s

        # Reject reactive triggers up front; we will add support
        # role-by-role rather than silently dropping them.
        for role in pattern.roles:
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
                   for r in pattern.roles):
            raise ValueError(
                f"Pattern {pattern.name!r} has no continuous "
                f"role; nothing would drive the run.  Add at "
                f"least one role with `trigger: {{kind: "
                f"continuous}}`."
            )

        self._state: dict[str, _RoleState] = {
            r.name: _RoleState(role=r) for r in pattern.roles
        }
        self._results: dict[str, list[AgentResult]] = {
            r.name: [] for r in pattern.roles
        }

    def run(self) -> PatternRunResult:
        """Drive the pattern to completion and return a summary."""
        start = time.monotonic()

        for role in self._pattern.roles:
            if isinstance(role.trigger, ContinuousTrigger):
                self._fire_role(role)

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

        Counts only ``status=="succeeded"`` rows that are not
        synthetic.  Synthetic-failed rows already trigger
        post-check halts via :class:`flywheel.execution_channel.ExecutionChannel`;
        counting them here would double-count infrastructure
        failures as "real" progress.
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

        self._materialize_for(role)

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
        differentiation lives in the ``agent_workspace_dir`` name
        for now; P6 of the campaign replaces this with auto-named
        ``agent_workspaces/<execution_id>/`` mounts.
        """
        prompt_path = self._base.project_root / role.prompt
        prompt = prompt_path.read_text(encoding="utf-8")
        for key, value in (
                self._base.prompt_substitutions or {}).items():
            prompt = prompt.replace("{{" + key + "}}", value)

        kwargs: dict[str, Any] = {
            "workspace": self._base.workspace,
            "template": self._base.template,
            "project_root": self._base.project_root,
            "prompt": prompt,
            "agent_image": self._base.agent_image,
            "auth_volume": self._base.auth_volume,
            "model": role.model or self._base.model,
            "max_invocations": self._base.max_invocations,
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
            "allowed_blocks": self._base.allowed_blocks,
            "source_dirs": self._base.source_dirs,
            "input_artifacts": self._collect_inputs(role),
            "output_names": (
                role.outputs or self._base.output_names),
            "overrides": self._base.overrides,
            "mcp_servers": (
                role.mcp_servers or self._base.mcp_servers),
            "allowed_tools": (
                role.allowed_tools or self._base.allowed_tools),
            "extra_env": self._merge_env(role),
            "extra_mounts": self._base.extra_mounts,
            "pre_launch_hook": self._base.pre_launch_hook,
            "isolated_network": self._base.isolated_network,
            "agent_workspace_dir": self._workspace_dir_for(
                role,
                cohort_index=cohort_index,
                member_index=member_index,
            ),
            "post_checks": self._base.post_checks,
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

    def _materialize_for(self, role: Role) -> None:
        """Roll declared sequences into a single artifact pre-firing.

        Empty for roles that do not declare ``materialize``.  When
        the source block has no rows yet (e.g., the first cohort
        fires before any ``take_action`` rows exist), the call is
        skipped: ``materialize_sequence`` would otherwise create
        an empty artifact and confuse downstream readers expecting
        at least one record.
        """
        if not role.materialize:
            return
        ws = self._base.workspace
        for target, source in role.materialize.items():
            if not ws.instances_for(source):
                print(
                    f"  [pattern-runner] role {role.name!r}: "
                    f"skipping materialize {target!r}<-{source!r}"
                    f" (no source instances yet)"
                )
                continue
            ws.materialize_sequence(source, target)

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
        """Pick a deterministic agent_workspace subdir name.

        For continuous roles with cardinality 1 we pass ``None``
        and let the launcher use its default — that matches
        today's just-play behavior, so a workspace recorded with
        a pattern is byte-equivalent to one recorded with the
        legacy loop.

        Cohort roles (or continuous roles with cardinality > 1)
        get ``<role>_<cohort>_<member>``.  P6 of the campaign
        replaces this hand-rolled naming with the auto
        ``agent_workspaces/<execution_id>/`` scheme.
        """
        if (isinstance(role.trigger, ContinuousTrigger)
                and role.cardinality == 1):
            return self._base.agent_workspace_dir
        return f"{role.name}_{cohort_index}_{member_index}"

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
    ) -> "PatternRunner":
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
