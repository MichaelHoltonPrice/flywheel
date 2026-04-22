"""Pattern runner — drives a :class:`flywheel.pattern.Pattern` end-to-end.

Where :mod:`flywheel.pattern` defines what a pattern *is*, this
module defines how one is *executed*: continuous and autorestart
roles fire at run start and persist for the run's lifetime;
ledger-driven triggers (``every_n_executions``) fire as the
workspace accumulates matching block executions; reactive
triggers (``on_request``, ``on_event``) are recognized but
currently raise — the runner documents them as a known gap so
the failure mode is loud rather than silent.

The runner is intentionally agnostic to what kind of block any
role launches.  It builds protocol-level launch arguments
(``block_name``, ``workspace``, ``input_bindings``, ``run_id``)
plus a free-form ``overrides`` dict and hands them to an
:class:`flywheel.executor.BlockExecutor` returned by the
caller-supplied :class:`ExecutorFactory`.  The executor is
free to be a generic container executor, a battery
(:class:`flywheel.agent_executor.AgentExecutor`), a
workspace-persistent runtime
(:class:`flywheel.executor.RequestResponseExecutor`), or any
future implementation of the :class:`BlockExecutor` protocol.

Termination
-----------

A pattern run ends when **all driving-role handles have
finished**.  Driving roles are those whose liveness keeps the
loop alive — ``continuous`` and ``autorestart``.  Patterns
without any driving role are rejected at runner start: a
pattern with only ``every_n_executions`` roles would have no
driver to make the workspace grow, and would loop forever.
Reactive-only patterns will get their own driver later
(probably an ``on_event`` trigger that fires from a workspace
file watcher).

``on_tool`` instances are not launched directly by the
runner: their dispatch is wired through the pattern's tool
router (built once at construction) and integrated into any
agent executor the factory hands out via
:meth:`flywheel.agent_executor.AgentExecutor.for_pattern`.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from flywheel.agent import AgentResult
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
    make_pattern_executor_factory,
)
from flywheel.run_defaults import RunDefaults
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

    Both :class:`flywheel.executor.ExecutionHandle` and the test
    fakes satisfy this.  Kept intentionally narrow so the runner
    does not bind to any executor-specific surface beyond the
    pieces it actually exercises.
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
            order they were collected.  The runner records
            whatever the executor's handle returns from
            ``wait()``: agent batteries surface
            :class:`flywheel.executor.ExecutionResult` (via
            :class:`flywheel.agent_executor.AgentExecutionHandle`),
            generic container executors return
            :class:`ExecutionResult` directly; tests using fake
            handles may return :class:`AgentResult`.  Continuous
            roles populate this on termination; cohort roles
            populate this as each cohort drains.
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
        defaults: Per-run inputs the runner needs (workspace,
            template, project root) plus a free-form ``defaults``
            bag executors may consult.  See
            :class:`flywheel.run_defaults.RunDefaults`.
        executor_factory: Returns a
            :class:`flywheel.executor.BlockExecutor` for a given
            :class:`flywheel.template.BlockDefinition`.  Called
            once per launch (executors that want to amortise
            setup should memoise internally; the
            :func:`flywheel.pattern_handoff.make_pattern_executor_factory`
            wrapper memoises by block name when it is in play).
        poll_interval_s: How often to re-scan the workspace
            ledger for trigger evaluation.  Lower values feel
            snappier but burn CPU; the default matches the
            cadence of the tools that emit ``game_step`` executions
            (sub-second is overkill).
        max_total_runtime_s: Hard wall-clock cap.  ``None`` means
            wait until all driving handles finish naturally.
        per_instance_runtime_config: Per-instance ``extra_env`` /
            ``extra_mounts`` knobs.  Forwarded into per-launch
            ``overrides`` so the executor can see them; the
            ``on_tool`` dispatch path consumes the same map for
            its tool-input bridge launches.

    The runner is single-shot: call :meth:`run` exactly once.
    """

    def __init__(
        self,
        pattern: Pattern,
        *,
        defaults: RunDefaults,
        executor_factory: ExecutorFactory,
        poll_interval_s: float = 1.0,
        max_total_runtime_s: float | None = None,
        per_instance_runtime_config: (
            dict[str, InstanceRuntimeConfig] | None) = None,
    ):
        """Wire a pattern to its execution surface.

        Construction-time validation:

        * Reactive triggers (``on_request``, ``on_event``)
          raise ``NotImplementedError`` — declared in the
          grammar but not yet driven by the runner.
        * Patterns without any driving role are rejected.
        * Autorestart roles must have ``cardinality == 1``.
        * Patterns with ``on_tool`` instances must declare
          their target blocks in the template; the on_tool
          bridge writes JSON into the block's single declared
          input slot, so missing blocks or multi-input blocks
          are rejected.  See
          :func:`flywheel.pattern_handoff.build_pattern_tool_router`.
        """
        self._pattern = pattern
        self._defaults = defaults
        self._poll = poll_interval_s
        self._max_runtime = max_total_runtime_s
        self._per_instance_runtime_config: dict[
            str, InstanceRuntimeConfig
        ] = dict(per_instance_runtime_config or {})

        # ``Pattern.instances`` is the canonical topology list
        # — populated for both grammars.  The runner drives
        # everything off of it; ``pattern.roles`` is preserved
        # only for legacy callers.  Each launchable instance
        # gets a synthesized :class:`Role` so the launch
        # plumbing keeps working.  ``on_tool`` instances are
        # tracked separately and dispatched via the pattern's
        # own tool router (built below).
        synthesized_roles: list[Role] = []
        self._on_tool_instances: list[BlockInstance] = []
        for inst in pattern.iter_instances():
            if isinstance(inst.trigger, OnToolTrigger):
                self._on_tool_instances.append(inst)
                continue
            synthesized_roles.append(
                _role_from_instance(inst))

        # Build the pattern-owned tool router.  When there are
        # no on_tool instances it's ``None``; the executor
        # factory then passes through unchanged.  The router
        # itself is an agent-battery shape
        # (:class:`flywheel.agent_handoff.ToolRouter`) owned by
        # :mod:`flywheel.pattern_handoff` — the runner treats
        # it as opaque and only checks for ``None``.
        self._pattern_tool_router: Any | None = (
            build_pattern_tool_router(
                pattern_name=self._pattern.name,
                on_tool_instances=self._on_tool_instances,
                workspace=self._defaults.workspace,
                template=self._defaults.template,
                executor_factory=executor_factory,
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

        # When the pattern owns tool dispatch, wrap the
        # executor factory so any executor that knows how to
        # integrate a pattern router (today: AgentExecutor via
        # ``for_pattern``) gets the pattern's router merged in
        # at launch time.  Other executors pass through
        # unchanged.  Wrapping is a no-op when the pattern has
        # no on_tool instances.
        if self._pattern_tool_router is not None:
            self._executor_factory: ExecutorFactory = (
                make_pattern_executor_factory(
                    executor_factory,
                    pattern_router=self._pattern_tool_router,
                    post_dispatch_fn=self._post_dispatch_hook,
                ))
        else:
            self._executor_factory = executor_factory

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
        run_record = self._defaults.workspace.begin_run(
            kind=f"pattern:{self._pattern.name}",
        )
        self._run_id = run_record.id
        # Persist the open run immediately.  If the host dies
        # before any execution finishes and triggers a save, the
        # run is still durable — a later operator can see the
        # ``running`` record and decide how to handle it.
        self._defaults.workspace.save()
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
            self._defaults.workspace.end_run(
                run_record.id, status=status)
            self._defaults.workspace.save()

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

    # ── trigger evaluation ───────────────────────────────────────

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
        ws = self._defaults.workspace
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

    # ── cohort firing ────────────────────────────────────────────

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
        :func:`flywheel.pattern_handoff.adapt_post_dispatch_for_handoff`
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
        ws = self._defaults.workspace
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

    # ── single launch path ───────────────────────────────────────

    def _launch_role_member(
        self,
        role: Role,
        *,
        cohort_index: int,
        member_index: int,
        predecessor_id: str | None = None,
    ) -> _Handle:
        """Launch one cohort member through the executor seam.

        Resolves the role's target block, builds a per-launch
        ``overrides`` dict (prompt body, role-level overrides,
        predecessor chaining) and calls
        ``executor_factory(block_def).launch(...)``.

        ``cohort_index`` / ``member_index`` are accepted for
        future use (e.g., per-member workspace dirs) but are
        not consumed today: executors differentiate parallel
        members through their own per-launch state (e.g., the
        agent battery's auto-named ``agent_workspaces/<id>/``
        mounts).

        Per-launch container runtime knobs (``extra_env`` /
        ``extra_mounts``) flow as explicit launch kwargs per the
        substrate-wide container-extras convention (see
        ``executors.md``); executors that don't speak that
        convention (today: none in tree) would simply not
        accept the kwargs and the runner-side call would
        fail loudly, which is the right outcome.

        ``predecessor_id`` is battery-specific (only the agent
        battery uses it for session restore) and flows through
        ``overrides`` so generic container executors can ignore
        the unknown key per the protocol's "unknown keys are
        ignored silently" contract.
        """
        del cohort_index, member_index

        block_name = role.block_name or role.name
        block_def = self._block_def_for(block_name)
        executor = self._executor_factory(block_def)

        overrides = self._build_overrides(
            role,
            predecessor_id=predecessor_id,
        )
        extra_env, extra_mounts = (
            self._collect_runtime_extras(role))
        bindings = self._collect_inputs(role) or {}

        launch_kwargs: dict[str, Any] = {
            "block_name": block_name,
            "workspace": self._defaults.workspace,
            "input_bindings": bindings,
            "overrides": overrides,
            "run_id": self._run_id,
        }
        # Container-extras are off-protocol kwargs; only attach
        # them when there's something to send so executors that
        # don't accept them keep working for roles with no
        # per-launch runtime knobs.
        if extra_env:
            launch_kwargs["extra_env"] = extra_env
        if extra_mounts:
            launch_kwargs["extra_mounts"] = extra_mounts

        return executor.launch(**launch_kwargs)

    def _build_overrides(
        self,
        role: Role,
        *,
        predecessor_id: str | None,
    ) -> dict[str, Any]:
        """Assemble the per-launch ``overrides`` dict for one role.

        Layers the role's own protocol-level knobs (prompt
        body, model, max_turns, ...) on top of the run-level
        ``defaults`` bag.  Container runtime knobs
        (``extra_env`` / ``extra_mounts``) are *not* placed
        here; they flow as explicit launch kwargs per the
        substrate's container-extras convention — see
        :meth:`_collect_runtime_extras`.

        The dict is intentionally free-form: each executor
        reads the keys it cares about and ignores the rest
        (per the :class:`flywheel.executor.BlockExecutor`
        protocol contract).  Empty entries (e.g. an unset
        ``role.model``) are omitted so executors can fall back
        to their constructor defaults rather than seeing
        ``None`` and overwriting them.
        """
        defaults_bag = dict(self._defaults.defaults or {})
        overrides: dict[str, Any] = dict(defaults_bag)

        if role.prompt:
            prompt_path = (
                self._defaults.project_root / role.prompt)
            prompt_body = prompt_path.read_text(
                encoding="utf-8")
            overrides["prompt"] = prompt_body
            substitutions = (
                defaults_bag.get("prompt_substitutions"))
            if substitutions:
                overrides["prompt_substitutions"] = (
                    substitutions)

        if role.model is not None:
            overrides["model"] = role.model
        if role.max_turns is not None:
            overrides["max_turns"] = role.max_turns
        if role.total_timeout is not None:
            overrides["total_timeout"] = role.total_timeout
        if role.mcp_servers is not None:
            overrides["mcp_servers"] = role.mcp_servers
        if role.allowed_tools is not None:
            overrides["allowed_tools"] = role.allowed_tools

        if predecessor_id is not None:
            overrides["predecessor_id"] = predecessor_id

        return overrides

    def _collect_runtime_extras(
        self, role: Role,
    ) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
        """Resolve the role's container-extras kwargs.

        ``extra_env`` merges :class:`InstanceRuntimeConfig` env
        with the role's own ``extra_env`` (role wins on key
        collision); ``extra_mounts`` is taken from the
        per-instance runtime config alone today (roles do not
        carry a mount field).  Both are passed to the executor
        as explicit kwargs per the substrate's container-extras
        convention so generic container executors receive them
        on their declared seam, not as opaque overrides.

        The agent battery accepts the same kwargs (with the
        same merge semantics) so a single dispatch call handles
        every executor in tree without the runner branching on
        executor type.
        """
        runtime_cfg = self._per_instance_runtime_config.get(
            role.name, InstanceRuntimeConfig())
        merged_env: dict[str, str] = dict(
            runtime_cfg.extra_env or {})
        if role.extra_env:
            merged_env.update(role.extra_env)
        extra_mounts = list(runtime_cfg.extra_mounts or [])
        return merged_env, extra_mounts

    def _collect_inputs(self, role: Role) -> dict[str, str] | None:
        """Resolve role.inputs to the latest registered instance IDs.

        Roles list inputs by *artifact name*; the runner picks
        the latest registered instance of each.  Roles whose
        inputs aren't registered yet pass an empty mapping (the
        executor treats it as "no inputs"), which is the right
        behaviour for the first launch on a fresh workspace.
        """
        ws = self._defaults.workspace
        bindings: dict[str, str] = {}
        for name in role.inputs:
            instances = ws.instances_for(name)
            if instances:
                bindings[name] = instances[-1].id
        return bindings or None

    def _block_def_for(
        self, block_name: str,
    ) -> BlockDefinition:
        """Look up ``block_name`` in the template; raise if absent.

        The executor factory branches on real
        :class:`BlockDefinition` fields (today: ``lifecycle``
        for the workspace-persistent vs one-shot split).  A
        stub default would silently route a launch through the
        wrong executor, so the runner now requires every block
        a pattern references to be declared in the template
        passed via :class:`RunDefaults`.

        Raises:
            ValueError: When ``block_name`` does not match any
                block in the template.  The message surfaces
                the pattern's intent and the template's known
                blocks so the operator can either fix the
                pattern or extend the template.
        """
        for b in self._defaults.template.blocks:
            if b.name == block_name:
                return b
        known = sorted(
            b.name for b in self._defaults.template.blocks)
        raise ValueError(
            f"PatternRunner: pattern references block "
            f"{block_name!r} but the template declares no "
            f"such block.  Known blocks: "
            f"{', '.join(known) or '<none>'}.  Add the block "
            f"to the template or correct the pattern's "
            f"``block:`` reference."
        )

    def _drain_all_handles(self) -> None:
        """Wait on every launched handle, recording results.

        Sequential drain serializes workspace writes (mirrors
        :class:`flywheel.block_group.BlockGroup`).  Errors from
        individual ``wait()`` calls are caught and logged so one
        crashed handle cannot strand siblings.
        """
        for state in self._state.values():
            for handle in state.handles:
                try:
                    result = handle.wait()
                except Exception as exc:
                    print(
                        f"  [pattern-runner] role "
                        f"{state.role.name!r} handle wait() "
                        f"raised: {exc!r}"
                    )
                    continue
                self._results[state.role.name].append(result)


def _role_from_instance(inst: BlockInstance) -> Role:
    """Synthesise a :class:`Role` equivalent of an instance.

    The runner's launch / trigger plumbing is driver-by-role
    today.  For patterns that declare an ``instances:`` topology
    we build a ``Role`` carrying the instance's per-instance
    overrides alongside an explicit ``block_name`` so the
    executor receives the right block even when its name differs
    from the instance name.

    ``on_tool`` instances skip this helper entirely; their
    dispatch is built directly from the pattern's tool router.
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
