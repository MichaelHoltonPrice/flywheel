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

import queue
import threading
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
from flywheel.executor import ExecutionResult
from flywheel.instance_runtime import (
    ExecutorFactory,
    InstanceRuntimeConfig,
)
from flywheel.pattern import (
    AutorestartTrigger,
    BlockInstance,
    ContinuousTrigger,
    EveryNExecutionsTrigger,
    OnEventTrigger,
    OnRequestTrigger,
    OnToolTrigger,
    Pattern,
    Role,
)
from flywheel.pattern_handoff import (
    build_pattern_tool_router,
    wrap_launch_fn_with_router,
)
from flywheel.template import BlockDefinition

# ``InstanceRuntimeConfig`` and ``ExecutorFactory`` are defined
# in :mod:`flywheel.instance_runtime` so the runner and the
# agent-battery wiring can both depend on them without forming
# an import cycle.  They remain importable from this module for
# back-compat with project code (cyberarc, tests).
__all__ = [
    "ExecutorFactory",
    "InstanceRuntimeConfig",
    "PatternRunResult",
    "PatternRunner",
]


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
class _PostDispatchEvent:
    """One post-dispatch cadence check enqueued by the handoff loop.

    Pushed by :meth:`PatternRunner._post_dispatch_hook` from the
    HANDOFF thread; drained by the main loop which fires any due
    cohorts synchronously and then sets :attr:`done` so the
    handoff loop can relaunch the calling agent.

    Attributes:
        block_name: The block whose dispatch just completed.  Only
            triggers with ``of_block == block_name`` are evaluated
            — avoids scanning irrelevant roles.
        done: Set by the main loop once cohorts have been fired
            and drained.  The handoff loop blocks on this event,
            making it the effective pause mechanism for
            ``pause:[]`` cohorts.
    """

    block_name: str
    done: threading.Event


@dataclass
class PatternRunResult:
    """Summary of a completed pattern run.

    Attributes:
        pattern_name: The pattern that was run.
        run_id: The :class:`flywheel.artifact.RunRecord` id the
            runner opened for this invocation.  Stored on every
            :class:`BlockExecution` the run produced, so callers
            can correlate the summary with the workspace ledger.
        cohorts_by_role: How many cohorts fired per role.  For
            ``continuous`` roles the value is always ``1`` (one
            cohort at run start); for ``every_n_executions``
            roles it grows over the run.
        agents_launched: Total number of individual agent
            launches across all roles, summed over cohorts and
            cardinality.
        results_by_role: Per-role list of result objects, in the
            order they were collected.  Agent-dispatched roles
            yield :class:`AgentResult`; executor-dispatched
            (prompt-less) roles yield
            :class:`flywheel.executor.ExecutionResult`.
            Continuous roles populate this on termination; cohort
            roles populate this as each cohort drains.
    """

    pattern_name: str
    run_id: str
    cohorts_by_role: dict[str, int]
    agents_launched: int
    results_by_role: dict[str, list[AgentResult | ExecutionResult]]


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
        :func:`flywheel.pattern_handoff.build_pattern_tool_router`).
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
        # place unchanged.  The router itself is an agent-
        # battery shape (:class:`flywheel.agent_handoff.ToolRouter`)
        # owned by :mod:`flywheel.pattern_handoff` — the runner
        # treats it as opaque and only checks for ``None``.
        self._pattern_tool_router: Any | None = (
            build_pattern_tool_router(
                pattern_name=self._pattern.name,
                on_tool_instances=self._on_tool_instances,
                workspace=self._base.workspace,
                template=self._base.template,
                executor_factory=self._executor_factory,
                per_instance_runtime_config=(
                    self._per_instance_runtime_config),
                run_id_provider=lambda: self._run_id,
            ))

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

        driver_triggers = (
            ContinuousTrigger, AutorestartTrigger)
        if not any(isinstance(r.trigger, driver_triggers)
                   for r in synthesized_roles):
            raise ValueError(
                f"Pattern {pattern.name!r} has no continuous or "
                f"autorestart instance; nothing would drive the "
                f"run.  Add at least one instance with "
                f"``trigger: {{kind: continuous}}`` or "
                f"``trigger: {{kind: autorestart}}``."
            )

        for r in synthesized_roles:
            if (isinstance(r.trigger, AutorestartTrigger)
                    and r.cardinality != 1):
                raise ValueError(
                    f"Pattern {pattern.name!r} role {r.name!r}: "
                    f"autorestart trigger requires "
                    f"cardinality=1 (got {r.cardinality}).  The "
                    f"role represents a single long-running "
                    f"instance that the runner keeps relaunching."
                )

        self._state: dict[str, _RoleState] = {
            r.name: _RoleState(role=r)
            for r in synthesized_roles
        }
        self._results: dict[
            str, list[AgentResult | ExecutionResult]
        ] = {r.name: [] for r in synthesized_roles}

        # Event-driven cadence:  ``_post_dispatch_hook`` pushes to
        # this queue from the HANDOFF thread after a block_runner
        # dispatch completes; the main loop drains the queue,
        # fires any due every_n_executions cohorts synchronously,
        # and signals completion so the caller's handoff loop can
        # resume.  The queue doubles as the main-loop wake source:
        # ``queue.Queue.get(timeout=poll_interval_s)`` replaces a
        # plain ``time.sleep`` so an incoming event wakes the main
        # loop immediately instead of up to one poll interval late.
        self._event_queue: queue.Queue[_PostDispatchEvent] = (
            queue.Queue())

        # If the pattern owns tool dispatch, wrap ``launch_fn``
        # so every call forwards a merged block_runner and a
        # post_dispatch_fn.  The merge raises at construction time
        # on any collision with a ``ToolRouter`` already pre-bound
        # by partial.  The wiring lives in
        # :mod:`flywheel.pattern_handoff` — the runner only
        # supplies the generic cadence callback (a ``str -> None``
        # function) and lets the wiring adapt it to the handoff
        # loop's :class:`HandoffContext`-shaped protocol.
        if self._pattern_tool_router is not None:
            self._launch_fn = wrap_launch_fn_with_router(
                launch_fn,
                pattern_name=self._pattern.name,
                on_tool_instance_count=len(
                    self._on_tool_instances),
                pattern_router=self._pattern_tool_router,
                post_dispatch_fn=self._post_dispatch_hook,
            )

        # Map block name → whether any every_n_executions trigger
        # references it.  Used by ``_post_dispatch_hook`` to
        # skip the event round-trip for dispatches that can't
        # cause a cohort to fire.
        self._blocks_watched_for_cadence: set[str] = {
            state.role.trigger.of_block
            for state in self._state.values()
            if isinstance(state.role.trigger, EveryNExecutionsTrigger)
        }

        # Assigned in :meth:`run` — a run is opened exactly when
        # execution begins, not at construction.  Construction can
        # raise on trigger / router validation; opening a durable
        # run record before any of that finishes would leave a
        # stuck ``running`` run behind.
        self._run_id: str | None = None

    def run(self) -> PatternRunResult:
        """Drive the pattern to completion and return a summary.

        Opens a :class:`flywheel.artifact.RunRecord` at start so
        every execution the runner drives can be tagged with the
        run id (enabling run-scoped cadence counters and
        durable per-run grouping on the workspace ledger).
        Closes the record in the outer ``finally`` regardless of
        how the body exits; the status written there reflects
        whether the run finished naturally, hit a hard deadline,
        or raised.
        """
        start = time.monotonic()
        run_record = self._base.workspace.begin_run(
            kind=f"pattern:{self._pattern.name}",
        )
        self._run_id = run_record.id
        # Persist the open run immediately.  If the host dies
        # before any execution finishes and triggers a save, the
        # run is still durable — a later operator can see the
        # ``running`` record and decide how to handle it.
        self._base.workspace.save()
        status = "failed"
        timed_out = False

        try:
            for state in self._state.values():
                if isinstance(
                        state.role.trigger,
                        (ContinuousTrigger, AutorestartTrigger),
                ):
                    self._fire_role(state.role)

            try:
                while True:
                    if self._all_drivers_done():
                        break
                    if (self._max_runtime is not None
                            and time.monotonic() - start
                            >= self._max_runtime):
                        print(
                            f"  [pattern-runner] max runtime "
                            f"{self._max_runtime:.0f}s reached; "
                            f"draining handles"
                        )
                        timed_out = True
                        break
                    self._evaluate_ledger_triggers()
                    self._refire_autorestart_if_ready()
                    # Honour ``scope="run"`` post_check halts as a
                    # strong stop: the scope promises "the run is
                    # over," so drive every alive driving-role
                    # handle to exit rather than waiting for its
                    # natural termination.  Symmetric with the
                    # pause mechanism: stop → wait → done.
                    self._stop_driving_handles_if_halted()
                    # Block on the event queue instead of a bare
                    # sleep so post-dispatch hooks from HANDOFF
                    # threads wake the main loop immediately.  The
                    # timeout keeps the poll fallback intact for
                    # triggers that aren't event-driven
                    # (autorestart on death, max_runtime).
                    try:
                        event = self._event_queue.get(
                            timeout=self._poll)
                    except queue.Empty:
                        continue
                    self._handle_post_dispatch_event(event)
            finally:
                self._drain_all_handles()
            status = "stopped" if timed_out else "succeeded"
        except BaseException:
            status = "failed"
            raise
        finally:
            self._base.workspace.end_run(
                run_record.id, status=status)
            self._base.workspace.save()

        return PatternRunResult(
            pattern_name=self._pattern.name,
            run_id=run_record.id,
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

    def _run_halted(self) -> bool:
        """Return True if any execution in this run queued a run halt.

        Scans the workspace's persisted ``BlockExecution``
        records for ``halt_directive={"scope": "run", ...}``
        stamped with the current ``run_id``.  Used to short-
        circuit the autorestart trigger — once a post-check
        asks the run to stop, we do not relaunch the role.
        """
        ws = self._base.workspace
        for ex in ws.executions.values():
            if ex.run_id != self._run_id:
                continue
            hd = getattr(ex, "halt_directive", None)
            if isinstance(hd, dict) and hd.get("scope") == "run":
                return True
        return False

    def _all_drivers_done(self) -> bool:
        """Return True once every driving-role handle is finished.

        "Driving" roles are the ones whose liveness keeps the
        run alive: continuous and autorestart.  An autorestart
        role's current handle being finished is not enough — the
        runner will relaunch it next tick unless the run has
        been halted.  So autorestart roles count as "done" only
        when (a) the current handle is finished AND (b)
        :meth:`_run_halted` is True.
        """
        run_halted = self._run_halted()
        for state in self._state.values():
            trigger = state.role.trigger
            if isinstance(trigger, ContinuousTrigger):
                for handle in state.handles:
                    if handle.is_alive():
                        return False
                continue
            if isinstance(trigger, AutorestartTrigger):
                # Still-running handle keeps the run alive.
                for handle in state.handles:
                    if handle.is_alive():
                        return False
                # All handles done; only "finished" if halted.
                if not run_halted:
                    return False
        return True

    def _refire_autorestart_if_ready(self) -> None:
        """Relaunch autorestart roles whose handle just finished.

        Called once per main-loop tick.  For each autorestart
        role, if the most recent handle is done and no
        run-scoped halt has been queued, fire the role again.
        The re-fire reuses :meth:`_fire_role`, which appends a
        new handle to ``state.handles`` and bumps
        ``cohorts_fired``.
        """
        if self._run_halted():
            return
        for state in self._state.values():
            if not isinstance(
                    state.role.trigger, AutorestartTrigger):
                continue
            if not state.handles:
                continue
            latest = state.handles[-1]
            if latest.is_alive():
                continue
            print(
                f"  [pattern-runner] autorestart role "
                f"{state.role.name!r} handle finished; "
                f"relaunching (restart "
                f"#{state.cohorts_fired})"
            )
            self._fire_role(state.role)

    def _evaluate_ledger_triggers(self) -> None:
        """Fire ``every_n_executions`` cohorts that are due.

        Counts only ``status=="succeeded"`` executions that are not
        synthetic.  Synthetic-failed executions already trigger
        post-check halts via
        :class:`flywheel.local_block.LocalBlockRecorder`; counting
        them here would double-count infrastructure failures as
        "real" progress.

        This is a poll-driven backstop; the event-driven path in
        :meth:`_handle_post_dispatch_event` handles the common case
        where a cohort becomes due because an on_tool dispatch just
        completed.
        """
        for state in self._state.values():
            trigger = state.role.trigger
            if not isinstance(trigger, EveryNExecutionsTrigger):
                continue

            succeeded = self._count_succeeded(trigger.of_block)
            cohorts_due = succeeded // trigger.n
            while state.cohorts_fired < cohorts_due:
                self._fire_role(state.role)

    def _handle_post_dispatch_event(
        self, event: _PostDispatchEvent,
    ) -> None:
        """Fire any cohort now due for ``event.block_name``, then signal.

        Runs on the main thread after a HANDOFF thread's
        post-dispatch hook enqueued the event.  The handoff loop
        is blocked on ``event.done`` — that block IS the pause for
        ``pause:[]`` cohorts, so this method deliberately skips
        :meth:`_pause_and_drain` / :meth:`_relaunch_paused`.  The
        calling agent resumes when we set ``event.done``; it
        relaunches via its handoff loop's normal
        ``predecessor_id`` chaining, not via
        :meth:`_relaunch_paused`.

        If a post_check has already queued a run-scoped halt (the
        dispatch that just returned, or an earlier one whose
        ledger write lands first), skip cohort firing.  Belt-
        and-suspenders against a torn-down run: the halted-handle
        stop sequence in the main loop should already have
        cancelled the handoff loop before this hook ran, but a
        defensive check is cheap and keeps the contract local.

        The ``try/finally`` guarantees ``event.done`` is always
        set — a raise here must not strand the handoff thread.
        """
        try:
            if self._run_halted():
                return
            for state in self._state.values():
                trigger = state.role.trigger
                if not isinstance(
                        trigger, EveryNExecutionsTrigger):
                    continue
                if trigger.of_block != event.block_name:
                    continue
                succeeded = self._count_succeeded(
                    trigger.of_block)
                cohorts_due = succeeded // trigger.n
                while state.cohorts_fired < cohorts_due:
                    self._fire_cohort_inline(state.role)
        finally:
            event.done.set()

    def _stop_driving_handles_if_halted(self) -> None:
        """Actively stop driving-role handles once the run is halted.

        "Driving" roles are ``continuous`` and ``autorestart`` —
        their liveness is what keeps the main loop running.  Once
        a post_check has queued a run-scoped halt, the
        ``scope="run"`` promise is "stop the run," not "don't
        relaunch after the current agent finishes naturally."
        Without this step the current handoff loop can iterate
        many more times before exiting — burning budget and
        potentially corrupting measurement (e.g. post-GAME_OVER
        actions inflating per-run step counts).

        Calls ``handle.stop(reason=)`` on every alive driving
        handle exactly once per halt (no-op for handles already
        stopping / dead).  The handle's own ``wait()`` is driven
        by the outer ``_drain_all_handles`` in ``run()``'s
        ``finally``.
        """
        if not self._run_halted():
            return
        for state in self._state.values():
            trigger = state.role.trigger
            if not isinstance(
                    trigger,
                    (ContinuousTrigger, AutorestartTrigger),
            ):
                continue
            for handle in state.handles:
                if not handle.is_alive():
                    continue
                try:
                    handle.stop(reason="run_halted")
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  [pattern-runner] stop() on "
                        f"{state.role.name!r} handle after halt "
                        f"raised: {exc!r}"
                    )

    def _fire_cohort_inline(self, role: Role) -> None:
        """Launch a cohort, drain it, return.  No pause/relaunch.

        The event-driven variant of :meth:`_fire_role`: used when
        the calling agent is already blocked on a
        :class:`_PostDispatchEvent` (so it's effectively paused
        without us stopping its handle), and will resume itself
        via its own handoff loop's ``predecessor_id`` chaining
        once we return.  Calling :meth:`_pause_and_drain` here
        would deadlock — the caller's handle is alive (its thread
        is running, just blocked in the hook); stopping it would
        forward to an ``is_alive() is None`` inner handle (we are
        between container launches), and ``handle.wait()`` would
        block forever because we ARE that thread.
        """
        state = self._state[role.name]
        cohort_index = state.cohorts_fired
        cohort_handles: list[_Handle] = []
        for member_index in range(role.cardinality):
            print(
                f"  [pattern-runner] firing role "
                f"{role.name!r} cohort {cohort_index} member "
                f"{member_index + 1}/{role.cardinality} "
                f"(event-driven)"
            )
            handle = self._launch_role_member(
                role,
                cohort_index=cohort_index,
                member_index=member_index,
            )
            state.handles.append(handle)
            cohort_handles.append(handle)
        state.cohorts_fired += 1
        self._drain_cohort(role.name, cohort_handles)

    def _post_dispatch_hook(self, tool_name: str) -> None:
        """Called from HANDOFF thread after each successful dispatch.

        Receives only the dispatched tool name — the agent-
        battery's :class:`HandoffContext` is adapted in
        :func:`flywheel.pattern_handoff.wrap_launch_fn_with_router`
        so the runner stays free of any handoff-loop types.

        Resolves the block name behind the tool, skips if no
        ``every_n_executions`` trigger watches that block,
        otherwise enqueues an event and blocks until the main
        loop has evaluated triggers and signalled completion.

        Blocking here is the pause mechanism for ``pause:[]``
        cohorts — the agent's handoff loop doesn't relaunch the
        next container until this hook returns.
        """
        block_name = self._block_name_for_tool(tool_name)
        if block_name is None:
            return
        if block_name not in self._blocks_watched_for_cadence:
            return
        event = _PostDispatchEvent(
            block_name=block_name,
            done=threading.Event(),
        )
        self._event_queue.put(event)
        event.done.wait()

    def _block_name_for_tool(self, tool_name: str) -> str | None:
        """Return the block dispatched by ``tool_name``, or None.

        Scans the pattern's ``on_tool`` instances; returns ``None``
        for tools handled by a project-side fallback runner (those
        dispatches don't flow through this pattern runner's
        cohort bookkeeping, so the post-dispatch hook is a no-op).
        """
        for inst in self._on_tool_instances:
            trigger = inst.trigger
            if (isinstance(trigger, OnToolTrigger)
                    and trigger.tool == tool_name):
                return inst.block
        return None

    def _count_succeeded(self, block_name: str) -> int:
        """Count non-synthetic succeeded executions of ``block_name``.

        Scoped to this runner's own ``run_id``: executions from
        earlier runs in the same workspace (another invocation
        of the same pattern, a manual ``flywheel run block``,
        etc.) do not contribute to the cadence count.  This is
        the fix for "running play-brainstorm twice in a
        workspace fires a backlog of cohorts on the second
        invocation."
        """
        ws = self._base.workspace
        return sum(
            1 for ex in ws.executions.values()
            if ex.block_name == block_name
            and ex.status == "succeeded"
            and not ex.synthetic
            and ex.run_id == self._run_id
        )

    def _fire_role(self, role: Role) -> None:
        """Launch ``role.cardinality`` agents for one trigger firing.

        When the triggering role's ``pause`` list is non-empty the
        firing is serialised against the named instances: each is
        stopped and drained (so ``/state/`` is captured), the
        cohort runs to completion, and the paused instances are
        relaunched with ``predecessor_id`` pointing at their
        stopped execution so the session chain continues.
        """
        state = self._state[role.name]
        cohort_index = state.cohorts_fired

        pause_names: tuple[str, ...] = ()
        if isinstance(role.trigger, EveryNExecutionsTrigger):
            pause_names = role.trigger.pause

        stopped_prev_id = self._pause_and_drain(
            pause_names=pause_names,
            reason=f"pause-for-cohort:{role.name}",
        )

        cohort_handles: list[_Handle] = []
        for member_index in range(role.cardinality):
            print(
                f"  [pattern-runner] firing role "
                f"{role.name!r} cohort {cohort_index} member "
                f"{member_index + 1}/{role.cardinality}"
            )
            handle = self._launch_role_member(
                role,
                cohort_index=cohort_index,
                member_index=member_index,
            )
            state.handles.append(handle)
            cohort_handles.append(handle)

        state.cohorts_fired += 1

        if not pause_names:
            return

        # Pause mode: the firing blocks the outer loop until the
        # cohort finishes and the paused instances are back up.
        # This is deliberate — while a paused instance is down
        # the ledger cannot grow (no new ``of_block`` executions
        # happen), so there is nothing meaningful to poll.
        self._drain_cohort(role.name, cohort_handles)
        self._relaunch_paused(
            pause_names=pause_names,
            stopped_prev_id=stopped_prev_id,
        )

    def _pause_and_drain(
        self,
        *,
        pause_names: tuple[str, ...],
        reason: str,
    ) -> dict[str, str | None]:
        """Stop every live handle for the named instances, wait.

        Returns a map from instance name to the execution id of
        the handle that was actually stopped, so the caller can
        set ``predecessor_id`` on the relaunch and keep the
        session chain intact.  Instances without a live handle
        (already finished naturally) are simply skipped; the
        returned map only contains names the caller should
        relaunch.
        """
        stopped_prev_id: dict[str, str | None] = {}
        for name in pause_names:
            paused = self._state.get(name)
            if paused is None:
                continue
            alive = [
                h for h in paused.handles if h.is_alive()]
            if not alive:
                continue
            for handle in alive:
                try:
                    handle.stop(reason=reason)
                except Exception as exc:
                    print(
                        f"  [pattern-runner] pause stop() for "
                        f"{name!r} raised: {exc!r}"
                    )
            last_execution_id: str | None = None
            for handle in list(paused.handles):
                try:
                    result = handle.wait()
                except Exception as exc:
                    print(
                        f"  [pattern-runner] pause wait() for "
                        f"{name!r} raised: {exc!r}"
                    )
                    continue
                self._results[name].append(result)
                last_execution_id = getattr(
                    result, "execution_id", None)
            paused.handles.clear()
            stopped_prev_id[name] = last_execution_id
        return stopped_prev_id

    def _drain_cohort(
        self,
        role_name: str,
        cohort_handles: list[_Handle],
    ) -> None:
        """Wait for each just-launched cohort handle; record results."""
        state = self._state[role_name]
        for handle in cohort_handles:
            try:
                result = handle.wait()
            except Exception as exc:
                print(
                    f"  [pattern-runner] cohort wait() for "
                    f"{role_name!r} raised: {exc!r}"
                )
                continue
            self._results[role_name].append(result)
        for handle in cohort_handles:
            if handle in state.handles:
                state.handles.remove(handle)

    def _relaunch_paused(
        self,
        *,
        pause_names: tuple[str, ...],
        stopped_prev_id: dict[str, str | None],
    ) -> None:
        """Relaunch each instance stopped by the pause phase.

        Only instances present in ``stopped_prev_id`` are
        relaunched — the dict was built from whatever actually
        had a live handle to stop, so an instance that had
        already finished naturally stays finished.
        """
        for name in pause_names:
            if name not in stopped_prev_id:
                continue
            paused = self._state.get(name)
            if paused is None:
                continue
            prev_id = stopped_prev_id[name]
            for member_index in range(paused.role.cardinality):
                print(
                    f"  [pattern-runner] relaunching paused "
                    f"{name!r} member "
                    f"{member_index + 1}/"
                    f"{paused.role.cardinality} "
                    f"(predecessor={prev_id})"
                )
                handle = self._launch_role_member(
                    paused.role,
                    cohort_index=0,
                    member_index=member_index,
                    predecessor_id=prev_id,
                )
                paused.handles.append(handle)

    def _launch_role_member(
        self,
        role: Role,
        *,
        cohort_index: int,
        member_index: int,
        predecessor_id: str | None = None,
    ) -> _Handle:
        """Launch one cohort member; pick agent vs. executor dispatch.

        A role with no declared prompt is dispatched directly via
        ``executor_factory(block_def).launch(...)`` — the agent
        battery is not in the picture.  This is how pure-container
        patterns (continuous or autorestart roles whose target
        block is, e.g., a one-shot training container) drive a
        run without inheriting any agent surface.

        A role with a prompt is dispatched through the legacy
        agent launch path (``launch_fn`` + ``_kwargs_for``).  The
        agent path stays unchanged in this slice; a follow-up
        slice replaces it with an :class:`AgentExecutor` invoked
        through the same factory.

        Predecessor chaining (``predecessor_id``) is honoured for
        the agent path (session restore).  The executor path
        silently drops it: the
        :class:`flywheel.executor.BlockExecutor` protocol does not
        declare a chaining concept, and probing executor-specific
        kwarg tolerance is too magical for a protocol boundary.
        See :meth:`_dispatch_role_via_executor` for the rationale
        and the migration path.
        """
        if self._role_uses_executor_dispatch(role):
            return self._dispatch_role_via_executor(
                role,
                cohort_index=cohort_index,
                member_index=member_index,
                predecessor_id=predecessor_id,
            )
        kwargs = self._kwargs_for(
            role,
            cohort_index=cohort_index,
            member_index=member_index,
        )
        if predecessor_id is not None:
            kwargs["predecessor_id"] = predecessor_id
        return self._launch_fn(**kwargs)

    def _role_uses_executor_dispatch(self, role: Role) -> bool:
        """Return True iff ``role`` is dispatched via the executor seam.

        Today the discriminator is "no prompt declared on the
        instance" — agents always declare a prompt; pure-container
        instances never do.  A future slice introduces an explicit
        ``battery:`` field on :class:`BlockDefinition` so the
        discriminator becomes a property of the block, not the
        instance.

        The runner-synthesized ``Role`` carries the empty string
        when an instance had no prompt (see
        :func:`_role_from_instance`); the dataclass default for
        the legacy ``roles:`` grammar is also ``""``-ish at
        construction, but the YAML loader requires a non-empty
        value there, so an empty prompt only ever reaches this
        method for instance-grammar pure-container roles.
        """
        return role.prompt == ""

    def _dispatch_role_via_executor(
        self,
        role: Role,
        *,
        cohort_index: int,
        member_index: int,
        predecessor_id: str | None = None,
    ) -> _Handle:
        """Launch one cohort member via the executor factory.

        Builds the protocol-level launch arguments
        (``block_name``, ``workspace``, ``input_bindings``,
        ``run_id``) plus the well-known executor extras
        (``extra_env``, ``extra_mounts``) and calls
        ``executor_factory(block_def).launch(...)``.  The returned
        handle satisfies the runner's :class:`_Handle` protocol —
        :class:`flywheel.executor.ExecutionHandle` instances do —
        so the rest of the runner (drain, stop-on-halt, result
        recording) is handle-shape agnostic.

        ``cohort_index`` / ``member_index`` are accepted for
        signature parity with :meth:`_kwargs_for` but unused
        today: container blocks differentiate parallel members
        through the executor's own per-launch state (e.g.,
        unique scratch dirs), not via runner-driven naming.

        ``predecessor_id`` is intentionally **dropped** on this
        path.  The :class:`flywheel.executor.BlockExecutor`
        protocol does not declare a chaining concept — that
        belongs to the agent battery, where it threads session
        restore between handoff iterations.  Container
        executors have no equivalent semantics, so silently
        suppressing it here keeps the runner inside the
        protocol contract instead of probing executor-specific
        kwarg tolerance.  An explicit chaining concept on the
        protocol is the right home for "yes, this block can
        chain" in the future; until then,
        :meth:`_relaunch_paused` is a no-op for the chain part
        when the paused role is executor-dispatched.

        Per-instance runtime knobs are layered on top of
        ``role.extra_env`` from
        ``self._per_instance_runtime_config[role.name]`` (same
        source the on_tool dispatch path consumes), so a
        prompt-less continuous role can declare
        ``extra_env`` / ``extra_mounts`` from the project hook
        without going through the agent surface.  Role-declared
        env wins over the runtime config on key collision —
        symmetric with :meth:`_merge_env`.

        Raises:
            ValueError: When no ``executor_factory`` was supplied
                to the runner, or when the role's target block
                cannot be resolved against the template.
        """
        del cohort_index, member_index, predecessor_id
        if self._executor_factory is None:
            raise ValueError(
                f"Pattern {self._pattern.name!r} role "
                f"{role.name!r}: instance has no prompt and is "
                f"dispatched via the executor seam, but no "
                f"``executor_factory`` was supplied to "
                f"PatternRunner.  Provide an ``executor_factory`` "
                f"that returns a BlockExecutor for block "
                f"{role.block_name or role.name!r}."
            )
        block_name = role.block_name or role.name
        block_def = self._find_block(block_name)
        if block_def is None:
            raise ValueError(
                f"Pattern {self._pattern.name!r} role "
                f"{role.name!r}: target block {block_name!r} is "
                f"not declared in the template."
            )
        bindings = self._collect_inputs(role) or {}

        runtime_cfg = self._per_instance_runtime_config.get(
            role.name, InstanceRuntimeConfig())
        merged_env: dict[str, str] = dict(
            runtime_cfg.extra_env or {})
        if role.extra_env:
            merged_env.update(role.extra_env)

        extras: dict[str, Any] = {}
        if merged_env:
            extras["extra_env"] = merged_env
        if runtime_cfg.extra_mounts:
            extras["extra_mounts"] = list(
                runtime_cfg.extra_mounts)

        executor = self._executor_factory(block_def)
        return executor.launch(
            block_name=block_name,
            workspace=self._base.workspace,
            input_bindings=bindings,
            run_id=self._run_id,
            **extras,
        )

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
            "run_id": self._run_id,
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
                and isinstance(
                    role.trigger,
                    (ContinuousTrigger, AutorestartTrigger),
                )
                and role.cardinality == 1):
            return self._base.agent_workspace_dir
        return None

    def _find_block(
        self, block_name: str,
    ) -> BlockDefinition | None:
        """Return the block definition with this name, or ``None``."""
        for b in self._base.template.blocks:
            if b.name == block_name:
                return b
        return None

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


def _role_from_instance(inst: BlockInstance) -> Role:
    """Synthesise a :class:`Role` equivalent of an instance.

    The runner's launch / trigger plumbing is driver-by-role
    today.  For patterns that declare an ``instances:`` topology
    we build a ``Role`` carrying the instance's agent-specific
    config alongside an explicit ``block_name`` so the launcher
    runs the right block even when its name differs from the
    instance name.

    ``prompt`` is the discriminator for the runner's two
    dispatch paths: a non-empty prompt routes to the agent
    launcher (the legacy path); an empty prompt — produced
    here when the instance declared no ``prompt:`` field — is
    the bridge signal for executor-seam dispatch (continuous
    or autorestart roles whose target block is a one-shot
    container, a workspace-persistent runtime, etc.).  A
    future ``BlockDefinition.battery`` field replaces this
    heuristic.  ``on_tool`` instances skip this helper
    entirely; their dispatch is built directly from the
    pattern's tool router.
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
