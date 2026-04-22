"""Tests for :class:`flywheel.pattern_runner.PatternRunner`.

Exercises the runner against an in-memory fake workspace and a
fake ``launch_fn``; no Docker, no ``launch_agent_block``.  The
fakes are kept tight on purpose: each test focuses on one
contract (continuous fires once, every-N fires at the right
cadence, cohort cardinality, termination, reactive triggers
raise) so failures point at exactly which behavior regressed.
"""

from __future__ import annotations

import functools
import json
import shutil
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flywheel.agent import AgentBlockConfig, AgentResult
from flywheel.agent_handoff import HandoffContext, HandoffResult, ToolRouter
from flywheel.artifact import (
    ArtifactInstance,
    BlockExecution,
    RunRecord,
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
from flywheel.pattern_runner import (
    InstanceRuntimeConfig,
    PatternRunner,
)
from flywheel.template import (
    ArtifactDeclaration,
    BlockDefinition,
    InputSlot,
    Template,
)

# ── Fakes ────────────────────────────────────────────────────────


class _FakeWorkspace:
    """Just enough of :class:`flywheel.workspace.Workspace` for the runner.

    The runner reads ``executions`` (for trigger evaluation) and
    ``instances_for`` (for input-artifact lookup).  Everything
    else is wired through ``base_config`` and never touched.
    """

    def __init__(self, project_root: Path):
        self.path = project_root
        self.executions: dict[str, BlockExecution] = {}
        self.artifacts: dict[str, ArtifactInstance] = {}
        self.artifact_declarations: dict[str, str] = {}
        self.runs: dict[str, RunRecord] = {}
        self.save_calls: int = 0
        self.save_calls_by_run_state: list[
            tuple[str, tuple[str, ...]]] = []

    def instances_for(self, _name: str) -> list:  # noqa: D401
        return []

    def begin_run(
        self,
        kind: str,
        config_snapshot: dict | None = None,
    ) -> RunRecord:
        rid = f"run_{uuid.uuid4().hex[:8]}"
        record = RunRecord(
            id=rid,
            kind=kind,
            started_at=datetime.now(UTC),
            status="running",
            config_snapshot=dict(
                config_snapshot) if config_snapshot else None,
        )
        self.runs[rid] = record
        return record

    def end_run(self, run_id: str, status: str) -> RunRecord:
        current = self.runs[run_id]
        if current.status != "running":
            raise ValueError(
                f"end_run: run {run_id!r} is already terminal "
                f"(status={current.status!r}); "
                f"double-close rejected"
            )
        from dataclasses import replace
        updated = replace(
            current,
            finished_at=datetime.now(UTC),
            status=status,
        )
        self.runs[run_id] = updated
        return updated

    def save(self) -> None:  # noqa: D401
        self.save_calls += 1
        snapshot = tuple(
            (rid, r.status) for rid, r in self.runs.items())
        self.save_calls_by_run_state.append(
            (f"save#{self.save_calls}", snapshot))

    def _active_run_id(self) -> str | None:
        for rid, r in self.runs.items():
            if r.status == "running":
                return rid
        return None

    def register_artifact(
        self,
        name: str,
        tempdir: Path,
        *,
        source: str | None = None,
    ) -> ArtifactInstance:
        """Copy ``tempdir`` contents into ``<ws>/artifacts/<id>/``.

        Matches the shape of
        :meth:`flywheel.workspace.Workspace.register_artifact` just
        enough for the on_tool bridge tests: the returned instance has
        a stable ``copy_path`` and the files land where tests expect.
        """
        artifact_id = f"{name}@{uuid.uuid4().hex[:8]}"
        dest = self.path / "artifacts" / artifact_id
        dest.mkdir(parents=True, exist_ok=True)
        for child in Path(tempdir).iterdir():
            target = dest / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
        instance = ArtifactInstance(
            id=artifact_id,
            name=name,
            kind="copy",
            created_at=datetime.now(UTC),
            source=source,
            copy_path=artifact_id,
        )
        self.artifacts[artifact_id] = instance
        return instance

    def add_succeeded(
        self,
        block_name: str,
        *,
        run_id: str | None = None,
        halt_directive: dict | None = None,
    ) -> None:
        """Append a succeeded execution tagged with the active run.

        Falls back to the workspace's single currently-running
        :class:`RunRecord` when the caller doesn't supply a
        ``run_id`` — mirrors the real executor's behaviour of
        tagging every recorded execution with the caller's
        ``run_id`` and means cadence tests don't have to know
        the runner's own id ahead of time.  ``halt_directive``
        lets tests simulate a post-check halt landing on a
        specific record without going through a real executor.
        """
        idx = len(self.executions)
        ex = BlockExecution(
            id=f"ex_{idx:04d}",
            block_name=block_name,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            status="succeeded",
            run_id=(
                run_id if run_id is not None
                else self._active_run_id()),
            halt_directive=halt_directive,
        )
        self.executions[ex.id] = ex

    def add_synthetic_failed(
        self, block_name: str, *, run_id: str | None = None,
    ) -> None:
        idx = len(self.executions)
        ex = BlockExecution(
            id=f"ex_{idx:04d}",
            block_name=block_name,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            status="failed",
            synthetic=True,
            run_id=(
                run_id if run_id is not None
                else self._active_run_id()),
        )
        self.executions[ex.id] = ex

    def add_run_halt(
        self,
        block_name: str = "ExecuteAction",
        *,
        reason: str = "terminal",
        run_id: str | None = None,
    ) -> None:
        """Append a succeeded execution carrying a scope='run' halt.

        Lets autorestart-trigger tests simulate a post-check
        that asked the pattern runner to stop — the same thing
        :class:`cyberarc.checks.execute_action` does on
        ``GAME_OVER``/``WIN``.
        """
        idx = len(self.executions)
        ex = BlockExecution(
            id=f"ex_{idx:04d}",
            block_name=block_name,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            status="succeeded",
            run_id=(
                run_id if run_id is not None
                else self._active_run_id()),
            halt_directive={"scope": "run", "reason": reason},
        )
        self.executions[ex.id] = ex


class _FakeHandle:
    """Programmable handle: stays alive until ``finish()`` is called."""

    def __init__(
        self,
        kwargs: dict,
        name: str = "",
        *,
        execution_id: str | None = None,
    ):
        self.kwargs = kwargs
        self.name = name
        self._alive = True
        self._waited = False
        self._stop_calls: list[str] = []
        self._result = AgentResult(
            exit_code=0, elapsed_s=0.1, evals_run=0,
            exit_reason="completed",
            execution_id=execution_id,
        )

    def is_alive(self) -> bool:
        return self._alive

    def finish(
        self,
        *,
        exit_reason: str = "completed",
        execution_id: str | None = None,
    ) -> None:
        self._alive = False
        self._result = AgentResult(
            exit_code=0, elapsed_s=0.1, evals_run=0,
            exit_reason=exit_reason,
            execution_id=(
                execution_id if execution_id is not None
                else self._result.execution_id),
        )

    def stop(self, reason: str = "requested") -> None:
        """Record the stop request; caller flips ``_alive`` via wait()."""
        self._stop_calls.append(reason)
        self._alive = False

    def wait(self):  # noqa: ANN201
        self._waited = True
        self._alive = False
        return self._result


def _base_config(
    tmp_path: Path,
    workspace: _FakeWorkspace,
    *,
    template: Template | None = None,
) -> AgentBlockConfig:
    """Build an AgentBlockConfig that points at the fake workspace.

    The runner only reads ``project_root``, ``workspace`` and
    ``template`` from this; everything else is forwarded to the
    fake launch_fn unchanged.  Tests that need the template to
    declare specific blocks (e.g. for ``on_tool`` instances to
    resolve against) can pass ``template=`` explicitly.
    """
    if template is None:
        template = Template(name="t", artifacts=[], blocks=[])
    return AgentBlockConfig(
        workspace=workspace,  # type: ignore[arg-type]
        template=template,
        project_root=tmp_path,
        prompt="",
    )


def _write_prompt(tmp_path: Path, name: str, body: str = "hi") -> str:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return name


# ── Tests ────────────────────────────────────────────────────────


class TestContinuous:
    def test_fires_once_at_start(self, tmp_path: Path):
        ws = _FakeWorkspace(tmp_path)
        prompt = _write_prompt(tmp_path, "play.md", "play prompt")
        pattern = Pattern(
            name="just-play",
            roles=[Role(
                name="play",
                prompt=prompt,
                trigger=ContinuousTrigger(),
                cardinality=1,
            )],
        )

        launches: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs, name="play")
            launches.append(handle)
            # Finish almost immediately so the runner exits.
            threading.Thread(
                target=lambda: (
                    time.sleep(0.05), handle.finish())).start()
            return handle

        runner = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.02,
        )
        result = runner.run()

        assert len(launches) == 1
        assert result.cohorts_by_role == {"play": 1}
        assert result.agents_launched == 1
        assert launches[0].kwargs["prompt"] == "play prompt"

    def test_passes_role_overrides(self, tmp_path: Path):
        ws = _FakeWorkspace(tmp_path)
        prompt = _write_prompt(tmp_path, "p.md")
        pattern = Pattern(
            name="cfg",
            roles=[Role(
                name="play",
                prompt=prompt,
                trigger=ContinuousTrigger(),
                model="claude-sonnet-4-6",
                mcp_servers="arc",
                allowed_tools="Read,Write",
                max_turns=5,
                total_timeout=120,
            )],
        )

        captured: dict[str, dict] = {}

        def launch_fn(**kwargs):
            captured["k"] = kwargs
            handle = _FakeHandle(kwargs)
            handle.finish()
            return handle

        PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.01,
        ).run()

        k = captured["k"]
        assert k["model"] == "claude-sonnet-4-6"
        assert k["mcp_servers"] == "arc"
        assert k["allowed_tools"] == "Read,Write"
        assert k["max_turns"] == 5
        assert k["total_timeout"] == 120


class TestAutorestart:
    """``AutorestartTrigger`` drives the role until a run halt.

    First-launch firing mirrors ``ContinuousTrigger``.  When the
    handle exits, the runner relaunches unless a ``scope="run"``
    halt has been queued for the current run via a post-check.
    The trigger is the right choice for roles whose pattern
    relies on an out-of-role signal (e.g., terminal engine state)
    to end the run rather than on the agent's own judgment.
    """

    def test_relaunches_after_handle_finishes(
            self, tmp_path: Path):
        ws = _FakeWorkspace(tmp_path)
        prompt = _write_prompt(tmp_path, "p.md", "play prompt")
        pattern = Pattern(
            name="p",
            roles=[Role(
                name="play",
                prompt=prompt,
                trigger=AutorestartTrigger(),
                cardinality=1,
            )],
        )

        launches: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs, name="play")
            launches.append(handle)
            # First two launches exit quickly; the third one
            # gets paired with a run halt so the runner stops.
            my_index = len(launches)

            def _finisher():
                time.sleep(0.05)
                if my_index >= 3:
                    ws.add_run_halt(reason="terminal in test")
                handle.finish()
            threading.Thread(target=_finisher).start()
            return handle

        runner = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.02,
        )
        result = runner.run()

        # At least 3 launches: 2 re-fires + 1 final.  The halt
        # queued on the third launch's finish stops further
        # refires on the next tick.
        assert len(launches) >= 3
        assert result.cohorts_by_role["play"] == len(launches)

    def test_run_halt_prevents_relaunch(
            self, tmp_path: Path):
        ws = _FakeWorkspace(tmp_path)
        prompt = _write_prompt(tmp_path, "p.md")
        pattern = Pattern(
            name="p",
            roles=[Role(
                name="play",
                prompt=prompt,
                trigger=AutorestartTrigger(),
                cardinality=1,
            )],
        )

        launches: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs, name="play")
            launches.append(handle)
            # Halt lands synchronously with the first launch.
            ws.add_run_halt(reason="immediate terminal")
            threading.Thread(
                target=lambda: (
                    time.sleep(0.05), handle.finish())).start()
            return handle

        runner = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.02,
        )
        result = runner.run()

        # Only the initial firing; the halt stops refiring.
        assert len(launches) == 1
        assert result.cohorts_by_role["play"] == 1

    def test_cardinality_greater_than_one_rejected(
            self, tmp_path: Path):
        prompt = _write_prompt(tmp_path, "p.md")
        pattern = Pattern(
            name="p",
            roles=[Role(
                name="play",
                prompt=prompt,
                trigger=AutorestartTrigger(),
                cardinality=2,
            )],
        )
        with pytest.raises(
            ValueError, match="cardinality=1",
        ):
            PatternRunner(
                pattern,
                base_config=_base_config(
                    tmp_path, _FakeWorkspace(tmp_path)),
            )

    def test_autorestart_alone_drives_run(
            self, tmp_path: Path):
        """A pattern with only autorestart drivers is accepted.

        No continuous role is required; autorestart counts as a
        driver on its own.
        """
        ws = _FakeWorkspace(tmp_path)
        prompt = _write_prompt(tmp_path, "p.md")
        pattern = Pattern(
            name="p",
            roles=[Role(
                name="play",
                prompt=prompt,
                trigger=AutorestartTrigger(),
                cardinality=1,
            )],
        )

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs, name="play")
            ws.add_run_halt(reason="stop immediately")

            def _finisher():
                time.sleep(0.05)
                handle.finish()
            threading.Thread(target=_finisher).start()
            return handle

        # Construction doesn't raise on "no continuous role"
        # and run returns after the halt propagates.
        runner = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.02,
        )
        runner.run()


class TestEveryNExecutions:
    def test_fires_at_correct_cadence(self, tmp_path: Path):
        ws = _FakeWorkspace(tmp_path)
        play_prompt = _write_prompt(tmp_path, "play.md", "play body")
        bs_prompt = _write_prompt(tmp_path, "bs.md", "brainstorm body")
        pattern = Pattern(
            name="play-bs",
            roles=[
                Role(
                    name="play",
                    prompt=play_prompt,
                    trigger=ContinuousTrigger(),
                ),
                Role(
                    name="brainstorm",
                    prompt=bs_prompt,
                    trigger=EveryNExecutionsTrigger(
                        of_block="take_action", n=3),
                    cardinality=2,
                ),
            ],
        )

        launches_by_role: dict[str, int] = {
            "play": 0, "brainstorm": 0}
        play_handle: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs)
            # Identify the role via the prompt path each role
            # carries (the runner reads the prompt from disk and
            # passes it through verbatim).
            prompt = kwargs.get("prompt") or ""
            role = "brainstorm" if "brainstorm" in prompt else "play"
            launches_by_role[role] += 1
            if role == "play":
                play_handle.append(handle)
            else:
                handle.finish()
            return handle

        # Driver thread: simulate ledger growth over time.
        def driver():
            for _ in range(7):
                time.sleep(0.05)
                ws.add_succeeded("take_action")
            time.sleep(0.05)
            play_handle[0].finish()

        threading.Thread(target=driver).start()

        runner = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.02,
        )
        result = runner.run()

        # 7 take_action executions / n=3 = 2 cohorts; cardinality 2 → 4 agents.
        assert result.cohorts_by_role["brainstorm"] == 2
        assert launches_by_role["brainstorm"] == 4
        assert launches_by_role["play"] == 1

    def test_synthetic_failed_executions_do_not_count(
            self, tmp_path: Path):
        ws = _FakeWorkspace(tmp_path)
        play_prompt = _write_prompt(tmp_path, "play.md", "play")
        bs_prompt = _write_prompt(tmp_path, "bs.md", "bs")
        pattern = Pattern(
            name="p",
            roles=[
                Role(
                    name="play",
                    prompt=play_prompt,
                    trigger=ContinuousTrigger(),
                ),
                Role(
                    name="bs",
                    prompt=bs_prompt,
                    trigger=EveryNExecutionsTrigger(
                        of_block="take_action", n=2),
                ),
            ],
        )

        play_handle: list[_FakeHandle] = []
        bs_count = [0]

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs)
            # Roles are distinguished by their prompt body.
            if (kwargs.get("prompt") or "").startswith("bs"):
                bs_count[0] += 1
                handle.finish()
            else:
                play_handle.append(handle)
            return handle

        def driver():
            for _ in range(5):
                ws.add_synthetic_failed("take_action")
                time.sleep(0.02)
            time.sleep(0.05)
            play_handle[0].finish()

        threading.Thread(target=driver).start()

        PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.02,
        ).run()

        assert bs_count[0] == 0

    def test_every_n_counter_is_run_scoped(self, tmp_path: Path):
        """Cadence counter ignores executions from prior runs.

        The play-brainstorm bug: running the pattern twice in a
        single workspace left the second run's every-N trigger
        counting prior-run executions too, firing the nested
        cohort immediately instead of after N executions of the
        current run.  Runs scope the counter: only executions
        tagged with the current run's ``run_id`` count.
        """
        ws = _FakeWorkspace(tmp_path)
        play_prompt = _write_prompt(tmp_path, "play.md", "play body")
        bs_prompt = _write_prompt(tmp_path, "bs.md", "brainstorm body")

        # Pre-seed the ledger with 5 succeeded executions from an
        # earlier run — enough to fire N=3 twice if the counter
        # were workspace-wide.
        prior = ws.begin_run(kind="pattern:play-bs")
        for _ in range(5):
            ws.add_succeeded("take_action", run_id=prior.id)
        ws.end_run(prior.id, status="succeeded")

        pattern = Pattern(
            name="play-bs",
            roles=[
                Role(
                    name="play",
                    prompt=play_prompt,
                    trigger=ContinuousTrigger(),
                ),
                Role(
                    name="brainstorm",
                    prompt=bs_prompt,
                    trigger=EveryNExecutionsTrigger(
                        of_block="take_action", n=3),
                ),
            ],
        )

        launches_by_role: dict[str, int] = {
            "play": 0, "brainstorm": 0}
        play_handle: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs)
            prompt = kwargs.get("prompt") or ""
            role = (
                "brainstorm" if "brainstorm" in prompt else "play")
            launches_by_role[role] += 1
            if role == "play":
                play_handle.append(handle)
            else:
                handle.finish()
            return handle

        def driver():
            # Only 2 executions in the *current* run — below N=3.
            for _ in range(2):
                time.sleep(0.05)
                ws.add_succeeded("take_action")
            time.sleep(0.1)
            play_handle[0].finish()

        threading.Thread(target=driver).start()

        result = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.02,
        ).run()

        # Prior run's 5 executions must not count; current run's
        # 2 < N=3 so brainstorm never fires.
        assert launches_by_role["brainstorm"] == 0
        assert result.cohorts_by_role.get("brainstorm", 0) == 0

    def test_begin_run_is_persisted_before_first_execution(
            self, tmp_path: Path):
        """Opening the run saves the workspace immediately.

        Without this, a host crash after ``begin_run`` but before
        the first execution-write-triggered save would leave the
        run invisible on disk — defeating the "durable grouping"
        contract.
        """
        ws = _FakeWorkspace(tmp_path)
        prompt = _write_prompt(tmp_path, "play.md", "play prompt")
        pattern = Pattern(
            name="just-play",
            roles=[Role(
                name="play",
                prompt=prompt,
                trigger=ContinuousTrigger(),
            )],
        )

        # Capture the save-call snapshots available at launch_fn
        # time.  Every launch_fn call happens *after* at least one
        # save, so the open run must already be visible in
        # ``ws.runs`` at that point.
        snapshots_at_launch: list[tuple[int, tuple]] = []

        def launch_fn(**kwargs):
            snapshots_at_launch.append(
                (ws.save_calls,
                 tuple(
                     (rid, r.status)
                     for rid, r in ws.runs.items())))
            handle = _FakeHandle(kwargs)
            threading.Thread(
                target=lambda: (
                    time.sleep(0.05), handle.finish())).start()
            return handle

        runner = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.02,
        )
        runner.run()

        # At launch time the runner has already invoked save() at
        # least once and the run is visible as ``running``.
        assert len(snapshots_at_launch) == 1
        save_calls, runs_snapshot = snapshots_at_launch[0]
        assert save_calls >= 1
        assert any(
            state == "running" for _, state in runs_snapshot)

    def test_max_runtime_records_stopped_status(
            self, tmp_path: Path):
        """Hitting ``max_total_runtime_s`` closes the run as ``stopped``.

        Without this, a timeout-drained pattern records
        ``succeeded`` — making ``stopped`` dead vocabulary and
        hiding incomplete runs from downstream analysis.
        """
        ws = _FakeWorkspace(tmp_path)
        prompt = _write_prompt(tmp_path, "play.md", "play prompt")
        pattern = Pattern(
            name="just-play",
            roles=[Role(
                name="play",
                prompt=prompt,
                trigger=ContinuousTrigger(),
            )],
        )

        play_handles: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs)
            play_handles.append(handle)
            # Auto-finish shortly so the outer drain completes
            # after the max-runtime break.  The timeout still
            # triggers first because the 0.05s max_total_runtime
            # elapses before this timer fires.
            threading.Thread(
                target=lambda: (
                    time.sleep(0.2), handle.finish())).start()
            return handle

        result = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.02,
            max_total_runtime_s=0.05,
        ).run()

        assert result.run_id in ws.runs
        assert ws.runs[result.run_id].status == "stopped"

    def test_run_id_propagated_to_launches(self, tmp_path: Path):
        """Every launch_fn call receives the current run's ``run_id``.

        The runner's own run_id is visible to launch_fn kwargs so
        downstream nested executions can inherit it.
        """
        ws = _FakeWorkspace(tmp_path)
        prompt = _write_prompt(tmp_path, "p.md", "play")
        pattern = Pattern(
            name="just-play",
            roles=[Role(
                name="play",
                prompt=prompt,
                trigger=ContinuousTrigger(),
            )],
        )

        captured: list[str | None] = []

        def launch_fn(**kwargs):
            captured.append(kwargs.get("run_id"))
            handle = _FakeHandle(kwargs)
            threading.Thread(
                target=lambda: (
                    time.sleep(0.05), handle.finish())).start()
            return handle

        result = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.02,
        ).run()

        assert len(captured) == 1
        assert captured[0] is not None
        assert captured[0] == result.run_id
        assert result.run_id in ws.runs
        assert ws.runs[result.run_id].status == "succeeded"


class TestRejectsBadPatterns:
    def test_no_continuous_role_raises(self, tmp_path: Path):
        prompt = _write_prompt(tmp_path, "p.md")
        pattern = Pattern(
            name="p",
            roles=[Role(
                name="bs",
                prompt=prompt,
                trigger=EveryNExecutionsTrigger(
                    of_block="take_action", n=5),
            )],
        )
        with pytest.raises(
            ValueError,
            match="continuous or autorestart instance",
        ):
            PatternRunner(
                pattern,
                base_config=_base_config(
                    tmp_path, _FakeWorkspace(tmp_path)),
            )

    def test_on_request_trigger_raises(self, tmp_path: Path):
        prompt = _write_prompt(tmp_path, "p.md")
        pattern = Pattern(
            name="p",
            roles=[
                Role(
                    name="play", prompt=prompt,
                    trigger=ContinuousTrigger()),
                Role(
                    name="bs", prompt=prompt,
                    trigger=OnRequestTrigger(tool="x")),
            ],
        )
        with pytest.raises(NotImplementedError, match="on_request"):
            PatternRunner(
                pattern,
                base_config=_base_config(
                    tmp_path, _FakeWorkspace(tmp_path)),
            )

    def test_on_event_trigger_raises(self, tmp_path: Path):
        prompt = _write_prompt(tmp_path, "p.md")
        pattern = Pattern(
            name="p",
            roles=[
                Role(
                    name="play", prompt=prompt,
                    trigger=ContinuousTrigger()),
                Role(
                    name="r", prompt=prompt,
                    trigger=OnEventTrigger(event="surprise")),
            ],
        )
        with pytest.raises(NotImplementedError, match="on_event"):
            PatternRunner(
                pattern,
                base_config=_base_config(
                    tmp_path, _FakeWorkspace(tmp_path)),
            )


class TestTermination:
    def test_max_runtime_drains(self, tmp_path: Path):
        # A continuous role that never finishes; max_runtime
        # should bail out and drain (wait collects the result).
        ws = _FakeWorkspace(tmp_path)
        prompt = _write_prompt(tmp_path, "p.md")
        pattern = Pattern(
            name="p",
            roles=[Role(
                name="play",
                prompt=prompt,
                trigger=ContinuousTrigger(),
            )],
        )

        launched: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs)
            launched.append(handle)
            return handle

        result = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.01,
            max_total_runtime_s=0.05,
        ).run()

        assert len(launched) == 1
        assert result.agents_launched == 1
        assert launched[0]._waited


class TestInstancesGrammar:
    """PatternRunner drives ``instances:`` patterns.

    Regression guard: the runner previously iterated
    ``pattern.roles`` to seed the continuous triggers at
    ``run()`` start, leaving ``instances:``-only patterns with
    no launches at all.  These tests also pin the new
    ``on_tool`` router plumbing — continuous launches happen,
    ``on_tool`` instances are not launched by the runner, and
    the pattern-derived block_runner lands in ``launch_fn``'s
    kwargs.
    """

    def test_continuous_instance_fires(
        self, tmp_path: Path,
    ):
        prompt = _write_prompt(tmp_path, "p.md", "play body")
        pattern = Pattern(
            name="just-play-instances",
            roles=[],
            instances=[
                BlockInstance(
                    name="play",
                    block="play",
                    trigger=ContinuousTrigger(),
                    prompt=prompt,
                ),
                BlockInstance(
                    name="execute_action",
                    block="ExecuteAction",
                    trigger=OnToolTrigger(
                        instance="play",
                        tool="mcp__arc__take_action",
                    ),
                ),
            ],
        )

        # ExecuteAction block needed in the template so the
        # pattern-router builder can look up its input slot.
        tmpl = Template(
            name="t",
            artifacts=[],
            blocks=[
                BlockDefinition(
                    name="ExecuteAction",
                    image="dummy:latest",
                    runner="container",
                    lifecycle="workspace_persistent",
                    inputs=[InputSlot(
                        name="action",
                        container_path="/input/action",
                    )],
                ),
            ],
        )

        launches: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs)
            launches.append(handle)
            handle.finish()
            return handle

        class _StubExecutor:
            def launch(self, **_):
                raise AssertionError(
                    "executor.launch must not run in this test")

        def executor_factory(block_def):
            return _StubExecutor()

        ws = _FakeWorkspace(tmp_path)
        base = _base_config(tmp_path, ws)
        # Swap in a template that carries ExecuteAction so the
        # router builder can resolve it.
        base = AgentBlockConfig(
            workspace=base.workspace,
            template=tmpl,
            project_root=base.project_root,
            prompt=base.prompt,
        )
        result = PatternRunner(
            pattern,
            base_config=base,
            launch_fn=launch_fn,
            poll_interval_s=0.01,
            max_total_runtime_s=1.0,
            executor_factory=executor_factory,
            per_instance_runtime_config={
                "execute_action": InstanceRuntimeConfig(),
            },
        ).run()

        # Exactly one launch: the continuous ``play`` instance.
        # The on_tool ``execute_action`` instance does not get
        # launched by the runner.
        assert len(launches) == 1
        assert result.agents_launched == 1
        assert launches[0].kwargs["block_name"] == "play"
        # The pattern runner wrapped launch_fn so the call
        # carries a merged block_runner.
        assert (
            "block_runner" in launches[0].kwargs
        ), "pattern runner must inject block_runner"

    def test_on_tool_without_executor_factory_raises(
        self, tmp_path: Path,
    ):
        prompt = _write_prompt(tmp_path, "p.md", "play body")
        pattern = Pattern(
            name="needs-factory",
            roles=[],
            instances=[
                BlockInstance(
                    name="play",
                    block="play",
                    trigger=ContinuousTrigger(),
                    prompt=prompt,
                ),
                BlockInstance(
                    name="ea",
                    block="ExecuteAction",
                    trigger=OnToolTrigger(
                        instance="play",
                        tool="mcp__arc__take_action",
                    ),
                ),
            ],
        )
        ws = _FakeWorkspace(tmp_path)
        with pytest.raises(
            ValueError, match="no ``executor_factory``",
        ):
            PatternRunner(
                pattern,
                base_config=_base_config(tmp_path, ws),
                launch_fn=lambda **k: _FakeHandle(k),
                poll_interval_s=0.01,
                max_total_runtime_s=0.1,
            )

    def test_on_tool_collides_with_existing_router_raises(
        self, tmp_path: Path,
    ):
        prompt = _write_prompt(tmp_path, "p.md", "play body")
        pattern = Pattern(
            name="collide",
            roles=[],
            instances=[
                BlockInstance(
                    name="play",
                    block="play",
                    trigger=ContinuousTrigger(),
                    prompt=prompt,
                ),
                BlockInstance(
                    name="ea",
                    block="ExecuteAction",
                    trigger=OnToolTrigger(
                        instance="play",
                        tool="mcp__arc__take_action",
                    ),
                ),
            ],
        )
        tmpl = Template(
            name="t",
            artifacts=[],
            blocks=[BlockDefinition(
                name="ExecuteAction",
                image="dummy:latest",
                runner="container",
                lifecycle="workspace_persistent",
                inputs=[InputSlot(
                    name="action",
                    container_path="/input/action",
                )],
            )],
        )
        # Pre-bind a router that ALSO claims take_action.
        existing = ToolRouter({
            "mcp__arc__take_action":
                lambda ctx: None,
        })
        base_launch = lambda **k: None  # noqa: E731
        pre_bound = functools.partial(
            base_launch, block_runner=existing)

        def executor_factory(block_def):
            class _E:
                def launch(self, **_):
                    raise AssertionError("unreached")
            return _E()

        ws = _FakeWorkspace(tmp_path)
        base = AgentBlockConfig(
            workspace=ws,
            template=tmpl,
            project_root=tmp_path,
            prompt="",
        )
        with pytest.raises(
            ValueError,
            match=(
                "declared by both the pattern's on_tool "
                "instances and the caller-supplied "
                "block_runner"),
        ):
            PatternRunner(
                pattern,
                base_config=base,
                launch_fn=pre_bound,
                executor_factory=executor_factory,
            )

    def test_instance_launches_correct_block_when_names_differ(
        self, tmp_path: Path,
    ):
        prompt = _write_prompt(tmp_path, "p.md", "body")
        pattern = Pattern(
            name="rename",
            roles=[],
            instances=[
                BlockInstance(
                    name="main",
                    block="play",
                    trigger=ContinuousTrigger(),
                    prompt=prompt,
                ),
            ],
        )

        captured: list[dict] = []

        def launch_fn(**kwargs):
            captured.append(kwargs)
            handle = _FakeHandle(kwargs)
            handle.finish()
            return handle

        ws = _FakeWorkspace(tmp_path)
        PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.01,
            max_total_runtime_s=1.0,
        ).run()

        # Block name must come from BlockInstance.block, not
        # BlockInstance.name.
        assert len(captured) == 1
        assert captured[0]["block_name"] == "play"


class TestOnToolBridge:
    """Direct coverage for the pattern's generic on_tool bridge.

    The docker-gated integration tests exercise the whole path
    end-to-end; these tests target the specific invariants the
    generic bridge adds on top of ``BlockExecutor.launch``:
    construction-time shape checks, JSON serialisation
    convention, per-instance runtime config forwarding, and
    error handling for inputs that the bridge cannot handle.
    """

    def _pattern(
        self,
        *,
        prompt: Path,
        trigger_tool: str = "mcp__arc__take_action",
    ) -> Pattern:
        return Pattern(
            name="on-tool-bridge",
            roles=[],
            instances=[
                BlockInstance(
                    name="play",
                    block="play",
                    trigger=ContinuousTrigger(),
                    prompt=str(prompt),
                ),
                BlockInstance(
                    name="execute_action",
                    block="TARGET",
                    trigger=OnToolTrigger(
                        instance="play",
                        tool=trigger_tool,
                    ),
                ),
            ],
        )

    def _template(
        self, *, input_count: int = 1,
        input_kind: str = "copy",
    ):
        inputs = [
            InputSlot(
                name=f"slot{i}",
                container_path=f"/input/slot{i}",
            )
            for i in range(input_count)
        ]
        artifacts = [
            ArtifactDeclaration(
                name=f"slot{i}", kind=input_kind)
            for i in range(input_count)
        ]
        return Template(
            name="t",
            artifacts=artifacts,
            blocks=[BlockDefinition(
                name="TARGET",
                image="dummy:latest",
                runner="container",
                lifecycle="workspace_persistent",
                inputs=inputs,
            )],
        )

    def _make_runner(
        self,
        *,
        tmp_path: Path,
        project_setup=None,
        template=None,
        launch_fn=None,
        executor=None,
        runtime_config=None,
    ):
        prompt = _write_prompt(tmp_path, "p.md", "body")
        pattern = self._pattern(prompt=prompt)
        tmpl = template or self._template()
        ws = _FakeWorkspace(tmp_path)
        # Mirror the workspace's normal contract: declarations are
        # lifted from the template at construction.  The on_tool
        # bridge reads this map to verify the target slot is a copy
        # artifact before trying to serialise tool_input to JSON.
        for art in tmpl.artifacts:
            ws.artifact_declarations[art.name] = art.kind
        base = AgentBlockConfig(
            workspace=ws,  # type: ignore[arg-type]
            template=tmpl,
            project_root=tmp_path,
            prompt="",
        )

        def _default_launch(**kw):
            handle = _FakeHandle(kw)
            handle.finish()
            return handle

        # Preserve the exact callable the test supplied so
        # ``PatternRunner`` can introspect ``functools.partial``
        # to find any pre-bound ``block_runner``.
        final_launch = (
            launch_fn if launch_fn is not None else _default_launch)

        class _CapturingExecutor:
            def __init__(self):
                self.calls: list[dict] = []

            def launch(self, **kwargs):
                self.calls.append(kwargs)
                result_ns = type("R", (), {})()
                result_ns.status = "succeeded"
                result_ns.execution_id = "exec-test"
                result_ns.exit_code = 0
                result_ns.elapsed_s = 0.01
                result_ns.output_bindings = {}
                handle = type("H", (), {})()
                handle.wait = lambda: result_ns
                return handle

        exe = executor or _CapturingExecutor()
        rt = runtime_config or {
            "execute_action": InstanceRuntimeConfig(),
        }
        runner = PatternRunner(
            pattern,
            base_config=base,
            launch_fn=final_launch,
            executor_factory=lambda block_def: exe,
            per_instance_runtime_config=rt,
            poll_interval_s=0.01,
            max_total_runtime_s=0.1,
        )
        return runner, exe, ws

    def test_construction_rejects_block_with_multiple_inputs(
        self, tmp_path: Path,
    ):
        tmpl = self._template(input_count=2)
        with pytest.raises(
            ValueError, match="exactly one declared input",
        ):
            self._make_runner(
                tmp_path=tmp_path, template=tmpl)

    def test_construction_rejects_block_with_zero_inputs(
        self, tmp_path: Path,
    ):
        tmpl = self._template(input_count=0)
        with pytest.raises(
            ValueError, match="exactly one declared input",
        ):
            self._make_runner(
                tmp_path=tmp_path, template=tmpl)

    def test_construction_rejects_incremental_input_slot(
        self, tmp_path: Path,
    ):
        tmpl = self._template(
            input_count=1, input_kind="incremental")
        with pytest.raises(
            ValueError,
            match="tool-input bridge only supports ``copy``",
        ):
            self._make_runner(
                tmp_path=tmp_path, template=tmpl)

    def test_bridge_writes_tool_input_as_slot_json(
        self, tmp_path: Path,
    ):
        runner, executor, ws = self._make_runner(
            tmp_path=tmp_path,
            runtime_config={
                "execute_action": InstanceRuntimeConfig(
                    extra_env={"A": "1"},
                    extra_mounts=[("host", "/ctr", "ro")],
                ),
            },
        )
        # Fire the pattern router's tool-input bridge
        # directly.
        router = runner._pattern_tool_router
        assert router is not None

        ctx = HandoffContext(
            tool_name="mcp__arc__take_action",
            tool_input={"action": 6, "x": 1, "y": 2},
            session_id="s",
            tool_use_id="t",
            iteration=0,
        )
        out = router(ctx)
        assert out.content == "OK"

        # Exactly one executor.launch call with the declared
        # runtime config forwarded and an action-binding for
        # slot0.
        assert len(executor.calls) == 1
        kw = executor.calls[0]
        assert kw["block_name"] == "TARGET"
        assert kw["extra_env"] == {"A": "1"}
        assert kw["extra_mounts"] == [
            ("host", "/ctr", "ro")]
        aid = kw["input_bindings"]["slot0"]

        # Artifact was registered with the slot's name and the
        # file content on disk is the JSON-serialised
        # tool_input.
        instance = ws.artifacts[aid]
        artifact_dir = (
            ws.path / "artifacts" / instance.copy_path)
        payload = json.loads(
            (artifact_dir / "slot0.json").read_text(
                encoding="utf-8"))
        assert payload == {
            "action": 6, "x": 1, "y": 2}

    def test_bridge_converts_non_serialisable_input_to_error(
        self, tmp_path: Path,
    ):
        runner, _executor, _ws = self._make_runner(
            tmp_path=tmp_path)
        router = runner._pattern_tool_router
        assert router is not None
        ctx = HandoffContext(
            tool_name="mcp__arc__take_action",
            # ``set`` is not JSON-serialisable.
            tool_input={"nope": {1, 2, 3}},
            session_id="s",
            tool_use_id="t",
            iteration=0,
        )
        out = router(ctx)
        assert out.is_error
        assert (
            "cannot serialize tool_input" in out.content)

    def test_opaque_block_runner_rejected_when_on_tool_present(
        self, tmp_path: Path,
    ):
        def opaque_runner(_ctx):
            return HandoffResult(content="OK")

        opaque_launch = functools.partial(
            lambda **_: None, block_runner=opaque_runner)

        with pytest.raises(
            ValueError,
            match="not a ``ToolRouter``",
        ):
            self._make_runner(
                tmp_path=tmp_path,
                launch_fn=opaque_launch,
            )


class TestPauseAndRelaunch:
    """``every_n_executions`` with ``pause`` stops + relaunches.

    Drives a tiny pattern in a helper thread: play role continuous
    (and already launched), brainstorm cohort fires every 2
    take_action executions with ``pause: [play]``.  The test
    asserts the dance in order — play stopped, its wait()
    drained, cohort launched and drained, play relaunched with
    ``predecessor_id`` matching the stopped execution.
    """

    def _pattern(self, *, play_prompt: str, bs_prompt: str) -> Pattern:
        return Pattern(
            name="pause-demo",
            roles=[],
            instances=[
                BlockInstance(
                    name="play",
                    block="play",
                    trigger=ContinuousTrigger(),
                    prompt=play_prompt,
                ),
                BlockInstance(
                    name="brainstorm",
                    block="brainstorm",
                    trigger=EveryNExecutionsTrigger(
                        of_block="take_action",
                        n=2,
                        pause=("play",),
                    ),
                    prompt=bs_prompt,
                    cardinality=2,
                ),
            ],
        )

    def test_stops_drains_cohort_relaunches_with_chain(
        self, tmp_path: Path,
    ):
        ws = _FakeWorkspace(tmp_path)
        play_prompt = _write_prompt(tmp_path, "play.md", "play")
        bs_prompt = _write_prompt(tmp_path, "bs.md", "bs")
        pattern = self._pattern(
            play_prompt=play_prompt, bs_prompt=bs_prompt)

        # One continuous play handle the runner launches at start;
        # we keep a reference so we can finish() it after the
        # pause dance completes and the runner drains on exit.
        launches: list[_FakeHandle] = []
        play_exec_id = "play-exec-1"

        def launch_fn(**kwargs):
            prompt = kwargs.get("prompt", "")
            if prompt.startswith("play"):
                exec_id = (
                    play_exec_id
                    if kwargs.get("predecessor_id") is None
                    else "play-exec-2"
                )
                handle = _FakeHandle(
                    kwargs, name="play",
                    execution_id=exec_id,
                )
                # Second play launch (post-pause) finishes
                # immediately so the overall run terminates.
                if kwargs.get("predecessor_id") is not None:
                    handle.finish(execution_id="play-exec-2")
            else:
                handle = _FakeHandle(
                    kwargs, name="brainstorm",
                    execution_id="bs-exec",
                )
                handle.finish(execution_id="bs-exec")
            launches.append(handle)
            return handle

        # Driver: grow the ledger enough to trigger the cohort,
        # then wait for the pause dance and let the relaunch
        # finish naturally (the second play handle above).
        def driver():
            for _ in range(2):
                time.sleep(0.03)
                ws.add_succeeded("take_action")

        threading.Thread(target=driver, daemon=True).start()

        runner = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.01,
            max_total_runtime_s=2.0,
        )
        result = runner.run()

        # Expected launches in order:
        #   [0] initial play
        #   [1,2] brainstorm cohort (cardinality=2)
        #   [3] play relaunch (predecessor_id set)
        assert [h.name for h in launches] == [
            "play", "brainstorm", "brainstorm", "play",
        ]

        initial_play = launches[0]
        assert initial_play._stop_calls == [
            "pause-for-cohort:brainstorm"]
        assert initial_play._waited  # drained before cohort fired

        relaunched_play = launches[3]
        assert (
            relaunched_play.kwargs["predecessor_id"]
            == play_exec_id
        )
        assert (
            relaunched_play.kwargs["block_name"] == "play")
        # Cohort drained: both brainstorm handles waited.
        assert launches[1]._waited
        assert launches[2]._waited

        # Summary: cohorts_fired for brainstorm is 1; play's
        # cohorts_fired stays at 1 (the relaunch is not a new
        # cohort, just a resume).
        assert result.cohorts_by_role["play"] == 1
        assert result.cohorts_by_role["brainstorm"] == 1

    def test_no_pause_preserves_parallel_behaviour(
        self, tmp_path: Path,
    ):
        """Without ``pause``, the cohort runs in parallel as before."""
        ws = _FakeWorkspace(tmp_path)
        play_prompt = _write_prompt(tmp_path, "play.md", "play")
        bs_prompt = _write_prompt(tmp_path, "bs.md", "bs")
        pattern = Pattern(
            name="no-pause",
            roles=[],
            instances=[
                BlockInstance(
                    name="play",
                    block="play",
                    trigger=ContinuousTrigger(),
                    prompt=play_prompt,
                ),
                BlockInstance(
                    name="brainstorm",
                    block="brainstorm",
                    trigger=EveryNExecutionsTrigger(
                        of_block="take_action", n=2),
                    prompt=bs_prompt,
                ),
            ],
        )
        play_handle_ref: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs)
            if kwargs.get("prompt", "").startswith("play"):
                play_handle_ref.append(handle)
            else:
                handle.finish()
            return handle

        def driver():
            for _ in range(2):
                time.sleep(0.02)
                ws.add_succeeded("take_action")
            time.sleep(0.05)
            play_handle_ref[0].finish()

        threading.Thread(target=driver, daemon=True).start()

        runner = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.01,
            max_total_runtime_s=2.0,
        )
        runner.run()

        # Play was never stopped — no pause field.
        assert play_handle_ref[0]._stop_calls == []


class TestEventDrivenCohortFiring:
    """``post_dispatch_fn`` fires ``every_n_executions`` at exactly N.

    The polling-based path fires up to ``poll_interval_s`` late —
    on a busy handoff loop that means the Nth cohort can fire
    after the (N+1)th ``ExecuteAction`` has already completed.  The
    event-driven hook closes that race: the handoff loop blocks
    on the hook until the runner has fired any due cohorts.

    These tests exercise the hook at two granularities:

    - :meth:`test_hook_returns_after_cohort_fires_in_main_loop`
      is the integration path — a fake ``launch_fn`` that
      simulates the handoff loop by calling the runner's
      ``post_dispatch_fn`` after each ``ExecuteAction``.
    - :meth:`test_hook_skips_unwatched_blocks` confirms the hook
      short-circuits when no ``every_n_executions`` trigger
      watches the dispatched block (the common no-cadence path
      shouldn't pay for a queue round-trip).
    """

    def _pattern(self, play_prompt: str, bs_prompt: str) -> Pattern:
        """Pattern with play, execute_action on_tool, brainstorm every-2.

        ``of_block`` matches the block dispatched by the on_tool
        instance (``ExecuteAction``) — the hook resolves the tool
        name back to this block via
        :meth:`PatternRunner._block_name_for_tool`.
        """
        return Pattern(
            name="event-cohort-demo",
            roles=[],
            instances=[
                BlockInstance(
                    name="play",
                    block="play",
                    trigger=ContinuousTrigger(),
                    prompt=play_prompt,
                ),
                BlockInstance(
                    name="execute_action",
                    block="ExecuteAction",
                    trigger=OnToolTrigger(
                        instance="play",
                        tool="mcp__arc__take_action",
                    ),
                ),
                BlockInstance(
                    name="brainstorm",
                    block="brainstorm",
                    trigger=EveryNExecutionsTrigger(
                        of_block="ExecuteAction",
                        n=2,
                        pause=("play",),
                    ),
                    prompt=bs_prompt,
                    cardinality=1,
                ),
            ],
        )

    def _template(self) -> Template:
        return Template(
            name="t",
            artifacts=[ArtifactDeclaration(
                name="action", kind="copy")],
            blocks=[BlockDefinition(
                name="ExecuteAction",
                image="dummy:latest",
                runner="container",
                lifecycle="workspace_persistent",
                inputs=[InputSlot(
                    name="action",
                    container_path="/input/action",
                )],
            )],
        )

    def test_hook_returns_after_cohort_fires_in_main_loop(
        self, tmp_path: Path,
    ):
        ws = _FakeWorkspace(tmp_path)
        play_prompt = _write_prompt(tmp_path, "play.md", "play")
        bs_prompt = _write_prompt(tmp_path, "bs.md", "bs")
        pattern = self._pattern(play_prompt, bs_prompt)
        template = self._template()

        # Capture the hook the runner injects into play's launch
        # so the test-side "handoff loop" can call it.
        captured_hook: list = []
        launches: list[_FakeHandle] = []
        cohort_fired_before_action_3 = threading.Event()

        def launch_fn(**kwargs):
            prompt = kwargs.get("prompt", "")
            if prompt == "play":
                hook = kwargs.get("post_dispatch_fn")
                if hook is not None and not captured_hook:
                    captured_hook.append(hook)
                handle = _FakeHandle(
                    kwargs, name="play",
                    execution_id="play-exec-1",
                )
            else:
                handle = _FakeHandle(
                    kwargs, name="brainstorm",
                    execution_id="bs-exec",
                )
                handle.finish(execution_id="bs-exec")
            launches.append(handle)
            return handle

        def driver():
            # Wait until the runner has captured the hook.
            for _ in range(100):
                if captured_hook:
                    break
                time.sleep(0.01)
            hook = captured_hook[0]

            # Action 1: record, dispatch hook — no cohort due.
            ws.add_succeeded("ExecuteAction")
            ctx1 = HandoffContext(
                tool_name="mcp__arc__take_action",
                tool_input={}, session_id="", tool_use_id="t1",
                iteration=0,
            )
            hook(ctx1)
            ea_count = sum(
                1 for ex in ws.executions.values()
                if ex.block_name == "ExecuteAction"
                and ex.status == "succeeded"
            )
            assert ea_count == 1

            # Action 2: record, dispatch hook — cohort MUST fire
            # and drain BEFORE the hook returns.  Counting
            # brainstorm launches before hook returns proves the
            # event-driven path is synchronous.
            ws.add_succeeded("ExecuteAction")
            bs_before = sum(
                1 for h in launches if h.name == "brainstorm")
            ctx2 = HandoffContext(
                tool_name="mcp__arc__take_action",
                tool_input={}, session_id="", tool_use_id="t2",
                iteration=0,
            )
            hook(ctx2)
            bs_after = sum(
                1 for h in launches if h.name == "brainstorm")
            assert bs_after - bs_before == 1, (
                f"cohort did not fire synchronously in hook "
                f"(before={bs_before}, after={bs_after})"
            )
            cohort_fired_before_action_3.set()

            # Let the play handle finish so the run terminates.
            launches[0].finish(execution_id="play-exec-1")

        threading.Thread(target=driver, daemon=True).start()

        class _StubExecutor:
            def launch(self, **_):
                raise AssertionError(
                    "executor.launch should not run in this test")

        def executor_factory(block_def):
            return _StubExecutor()

        runner = PatternRunner(
            pattern,
            base_config=_base_config(
                tmp_path, ws, template=template),
            launch_fn=launch_fn,
            poll_interval_s=0.01,
            max_total_runtime_s=5.0,
            executor_factory=executor_factory,
            per_instance_runtime_config={
                "execute_action": InstanceRuntimeConfig(),
            },
        )
        runner.run()

        # Final check: exactly 1 cohort (brainstorm cardinality 1,
        # so 1 brainstorm launch).  If the hook had NOT fired the
        # cohort synchronously, the poll backstop would still fire
        # it eventually — but the driver's mid-flight assertion
        # above would have failed first.
        assert cohort_fired_before_action_3.is_set()
        bs_launches = [
            h for h in launches if h.name == "brainstorm"]
        assert len(bs_launches) == 1

    def test_hook_skips_unwatched_blocks(
        self, tmp_path: Path,
    ):
        """Dispatches for blocks no trigger watches cost no event."""
        ws = _FakeWorkspace(tmp_path)
        play_prompt = _write_prompt(tmp_path, "play.md", "play")
        bs_prompt = _write_prompt(tmp_path, "bs.md", "bs")
        pattern = self._pattern(play_prompt, bs_prompt)
        template = self._template()

        captured_hook: list = []
        launches: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            prompt = kwargs.get("prompt", "")
            if prompt == "play":
                hook = kwargs.get("post_dispatch_fn")
                if hook is not None and not captured_hook:
                    captured_hook.append(hook)
                handle = _FakeHandle(
                    kwargs, name="play",
                    execution_id="play-exec-1",
                )
            else:
                handle = _FakeHandle(
                    kwargs, name="brainstorm",
                    execution_id="bs-exec",
                )
                handle.finish(execution_id="bs-exec")
            launches.append(handle)
            return handle

        def driver():
            for _ in range(100):
                if captured_hook:
                    break
                time.sleep(0.01)
            hook = captured_hook[0]

            # Dispatch a tool the pattern doesn't route via
            # every_n_executions — should return immediately and
            # not enqueue anything.
            ctx = HandoffContext(
                tool_name="mcp__some_unrelated__tool",
                tool_input={}, session_id="",
                tool_use_id="unrelated",
                iteration=0,
            )
            hook(ctx)
            launches[0].finish(execution_id="play-exec-1")

        threading.Thread(target=driver, daemon=True).start()

        class _StubExecutor:
            def launch(self, **_):
                raise AssertionError("unused")

        def executor_factory(block_def):
            return _StubExecutor()

        runner = PatternRunner(
            pattern,
            base_config=_base_config(
                tmp_path, ws, template=template),
            launch_fn=launch_fn,
            poll_interval_s=0.01,
            max_total_runtime_s=2.0,
            executor_factory=executor_factory,
            per_instance_runtime_config={
                "execute_action": InstanceRuntimeConfig(),
            },
        )
        runner.run()

        # No brainstorm launches from the unrelated dispatch.
        bs_launches = [
            h for h in launches if h.name == "brainstorm"]
        assert bs_launches == []


class TestActiveStopOnHalt:
    """``scope="run"`` halt actively stops driving-role handles.

    The pattern contract for ``scope="run"`` is "the run is over" —
    weaker than pause (which relaunches) but stronger than "refuse
    to relaunch after natural exit."  This test confirms the main
    loop calls ``stop()`` on live driving handles as soon as a
    halt lands in the ledger, rather than waiting for the handle
    to finish on its own schedule.
    """

    def test_halt_directive_triggers_handle_stop(
        self, tmp_path: Path,
    ):
        ws = _FakeWorkspace(tmp_path)
        play_prompt = _write_prompt(tmp_path, "play.md", "play")
        pattern = Pattern(
            name="halt-stop-demo",
            roles=[
                Role(
                    name="play",
                    prompt=play_prompt,
                    trigger=AutorestartTrigger(),
                ),
            ],
        )

        launches: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            handle = _FakeHandle(
                kwargs, name="play",
                execution_id=f"play-exec-{len(launches)}",
            )
            launches.append(handle)
            return handle

        # Driver: inject a halt-carrying execution record shortly
        # after the play role launches, then watch for the main
        # loop to call stop() on play's handle.
        def driver():
            # Wait for the initial play launch.
            for _ in range(200):
                if launches:
                    break
                time.sleep(0.005)
            # Simulate a post_check firing by writing a halt
            # directive onto a fresh execution record.
            ws.add_succeeded(
                "SomeBlock",
                halt_directive={
                    "scope": "run",
                    "reason": "test-halt",
                },
            )

        threading.Thread(target=driver, daemon=True).start()

        runner = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.01,
            max_total_runtime_s=2.0,
        )
        runner.run()

        # Exactly one play launch (autorestart did not refire
        # because the halt landed before the handle finished
        # naturally), and that launch received a stop() call.
        assert len(launches) == 1
        assert launches[0]._stop_calls == ["run_halted"]
