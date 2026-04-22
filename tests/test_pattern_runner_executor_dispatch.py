"""Tests for :class:`PatternRunner` driving pure-container blocks.

The runner is agnostic to what kind of block any role launches:
every dispatch goes through ``executor_factory(block_def).launch(...)``.
This module exercises that path against a fake
:class:`flywheel.executor.BlockExecutor` with no agent machinery
in the picture.

The path is what unblocks pure-container patterns: a continuous
or autorestart role can drive a one-shot training container, a
workspace-persistent runtime, or any other block-shaped executor
without inheriting any of the agent battery's surface (no
``prompt``, no ``model``, no ``mcp_servers``, no ``HANDOFF_TOOLS``).
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flywheel.artifact import (
    ArtifactInstance,
    BlockExecution,
    RunRecord,
)
from flywheel.executor import ExecutionResult
from flywheel.pattern import (
    BlockInstance,
    ContinuousTrigger,
    Pattern,
)
from flywheel.pattern_runner import (
    InstanceRuntimeConfig,
    PatternRunner,
)
from flywheel.run_defaults import RunDefaults
from flywheel.template import (
    BlockDefinition,
    Template,
)


class _FakeWorkspace:
    """Minimal :class:`Workspace` stub.  Mirrors ``test_pattern_runner``."""

    def __init__(self, project_root: Path):
        self.path = project_root
        self.executions: dict[str, BlockExecution] = {}
        self.artifacts: dict[str, ArtifactInstance] = {}
        self.artifact_declarations: dict[str, str] = {}
        self.runs: dict[str, RunRecord] = {}

    def instances_for(self, _name: str) -> list:
        return []

    def begin_run(
        self, kind: str, config_snapshot: dict | None = None,
    ) -> RunRecord:
        rid = f"run_{uuid.uuid4().hex[:8]}"
        record = RunRecord(
            id=rid,
            kind=kind,
            started_at=datetime.now(UTC),
            status="running",
            config_snapshot=(
                dict(config_snapshot)
                if config_snapshot else None),
        )
        self.runs[rid] = record
        return record

    def end_run(self, run_id: str, status: str) -> RunRecord:
        from dataclasses import replace
        current = self.runs[run_id]
        updated = replace(
            current,
            finished_at=datetime.now(UTC),
            status=status,
        )
        self.runs[run_id] = updated
        return updated

    def save(self) -> None:
        pass


class _FakeExecutionHandle:
    """Programmable :class:`ExecutionHandle` stand-in."""

    def __init__(
        self,
        *,
        block_name: str,
        execution_id: str,
        recorded_kwargs: dict,
    ):
        self.block_name = block_name
        self.execution_id = execution_id
        self.recorded_kwargs = recorded_kwargs
        self._alive = True
        self._result = ExecutionResult(
            exit_code=0,
            elapsed_s=0.0,
            output_bindings={},
            execution_id=execution_id,
            status="succeeded",
        )
        self.stop_calls: list[str] = []

    def is_alive(self) -> bool:
        return self._alive

    def stop(self, reason: str = "requested") -> None:
        self.stop_calls.append(reason)
        self._alive = False

    def wait(self) -> ExecutionResult:
        self._alive = False
        return self._result

    def finish(self) -> None:
        """Test affordance: simulate natural container exit."""
        self._alive = False


class _RecordingExecutor:
    """Captures every ``launch`` call; returns programmable handles.

    Models the :class:`flywheel.executor.ProcessExitExecutor` /
    :class:`flywheel.executor.RequestResponseExecutor` seam
    rather than the agent battery's: container-extras
    (``extra_env`` / ``extra_mounts``) arrive as explicit
    launch kwargs, ``overrides`` is for free-form
    per-launch knobs (e.g. CLI flag substitutions or the
    battery-specific ``predecessor_id``).  Tests inspect
    ``calls`` to verify the handoff is correct; the
    container-extras kwargs and ``overrides`` keys are flattened
    onto the recorded dict so assertions stay terse — but the
    fake itself enforces the seam by accepting them only on
    their declared channel.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.handles: list[_FakeExecutionHandle] = []

    def launch(
        self,
        *,
        block_name: str,
        workspace,  # noqa: ANN001
        input_bindings: dict[str, str],
        overrides: dict | None = None,
        run_id: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_mounts: list[tuple[str, str, str]] | None = None,
    ) -> _FakeExecutionHandle:
        recorded = dict(overrides or {})
        recorded.update({
            "block_name": block_name,
            "workspace": workspace,
            "input_bindings": dict(input_bindings),
            "run_id": run_id,
        })
        if extra_env is not None:
            recorded["extra_env"] = dict(extra_env)
        if extra_mounts is not None:
            recorded["extra_mounts"] = list(extra_mounts)
        self.calls.append(recorded)
        handle = _FakeExecutionHandle(
            block_name=block_name,
            execution_id=f"ex_{len(self.handles):04d}",
            recorded_kwargs=recorded,
        )
        self.handles.append(handle)
        return handle


def _container_template() -> Template:
    """Template carrying one container block.

    The block definition is intentionally minimal: the runner only
    looks up ``block_def.name`` and forwards the resolved object
    to ``executor_factory``.  The fake executor never inspects
    other fields.
    """
    block = BlockDefinition(
        name="train",
        runner="container",
        image="example/train:latest",
        lifecycle="one_shot",
        inputs=[],
        outputs=[],
    )
    return Template(name="trainer", artifacts=[], blocks=[block])


def _defaults(
    tmp_path: Path, workspace: _FakeWorkspace, template: Template,
) -> RunDefaults:
    return RunDefaults(
        workspace=workspace,  # type: ignore[arg-type]
        template=template,
        project_root=tmp_path,
    )


def _continuous_pattern(role_name: str = "trainer") -> Pattern:
    """One continuous role, no prompt → executor-dispatch path.

    The :func:`_role_from_instance` helper inside the pattern
    runner sets ``role.prompt = inst.prompt or ""``; passing
    ``prompt=None`` here means the runner never reads a prompt
    body off disk and the executor sees no ``prompt`` override.
    """
    return Pattern(
        name="trainer-loop",
        roles=[],
        instances=[
            BlockInstance(
                name=role_name,
                block="train",
                trigger=ContinuousTrigger(),
                cardinality=1,
                prompt=None,
            ),
        ],
    )


def _drain_first_handle(executor: _RecordingExecutor) -> threading.Thread:
    """Spawn a daemon that finishes the first launched handle.

    The continuous handle is alive after the runner fires the
    role; finishing it on a worker lets ``run()`` terminate
    naturally instead of relying on ``max_total_runtime_s``.
    """

    def _finisher() -> None:
        while not executor.handles:
            pass
        executor.handles[0].finish()

    t = threading.Thread(target=_finisher, daemon=True)
    t.start()
    return t


# ── Tests ────────────────────────────────────────────────────────


def test_continuous_pure_container_runs_via_executor_factory(
    tmp_path: Path,
) -> None:
    """A prompt-less continuous role dispatches via the executor seam.

    Verifies the protocol-level arguments land correctly and the
    run terminates when the handle finishes naturally.
    """
    ws = _FakeWorkspace(tmp_path)
    template = _container_template()
    pattern = _continuous_pattern()
    executor = _RecordingExecutor()

    runner = PatternRunner(
        pattern,
        defaults=_defaults(tmp_path, ws, template),
        executor_factory=lambda _block_def: executor,
        poll_interval_s=0.01,
    )

    t = _drain_first_handle(executor)
    result = runner.run()
    t.join(timeout=2.0)

    assert len(executor.calls) == 1, (
        f"expected exactly one executor.launch call, got "
        f"{executor.calls!r}")
    call = executor.calls[0]
    assert call["block_name"] == "train"
    assert call["workspace"] is ws
    assert call["input_bindings"] == {}
    # The runner stamps the active run_id on every launch so the
    # executor can tag executions for cadence accounting.
    assert call["run_id"] == result.run_id
    # No prompt key: the role has no prompt, so the runner does
    # not synthesise one.
    assert "prompt" not in call
    assert result.cohorts_by_role == {"trainer": 1}
    assert result.agents_launched == 1


def test_executor_dispatch_forwards_role_extra_env(
    tmp_path: Path,
) -> None:
    """``role.extra_env`` reaches the executor as the ``extra_env`` override.

    Per-instance env knobs declared on the YAML instance flow
    through to the executor's ``overrides["extra_env"]`` so
    container blocks can be parameterised without baking the
    value into the image.
    """
    ws = _FakeWorkspace(tmp_path)
    template = _container_template()
    pattern = Pattern(
        name="trainer-loop",
        roles=[],
        instances=[
            BlockInstance(
                name="trainer",
                block="train",
                trigger=ContinuousTrigger(),
                cardinality=1,
                prompt=None,
                extra_env={"SEED": "42", "RUN_TAG": "step1"},
            ),
        ],
    )
    executor = _RecordingExecutor()

    runner = PatternRunner(
        pattern,
        defaults=_defaults(tmp_path, ws, template),
        executor_factory=lambda _bd: executor,
        poll_interval_s=0.01,
    )

    _drain_first_handle(executor)
    runner.run()

    assert executor.calls[0]["extra_env"] == {
        "SEED": "42", "RUN_TAG": "step1"}


def test_executor_dispatch_unknown_block_raises(
    tmp_path: Path,
) -> None:
    """Pattern referencing an undeclared block fails fast.

    The executor factory branches on real
    :class:`BlockDefinition` metadata (today: ``lifecycle``).
    A stub default for an undeclared block could silently
    misroute the launch to the wrong executor, so the runner
    requires every referenced block to be declared in the
    template up front.
    """
    ws = _FakeWorkspace(tmp_path)
    template = _container_template()
    pattern = Pattern(
        name="bad-loop",
        roles=[],
        instances=[
            BlockInstance(
                name="ghost",
                block="not_in_template",
                trigger=ContinuousTrigger(),
                cardinality=1,
                prompt=None,
            ),
        ],
    )
    executor = _RecordingExecutor()

    runner = PatternRunner(
        pattern,
        defaults=_defaults(tmp_path, ws, template),
        executor_factory=lambda _bd: executor,
        poll_interval_s=0.01,
    )

    with pytest.raises(ValueError, match="not_in_template"):
        runner.run()


def test_per_instance_runtime_config_extra_env_merges(
    tmp_path: Path,
) -> None:
    """``per_instance_runtime_config`` env layers under ``role.extra_env``.

    The project hook supplies a per-instance runtime config keyed
    by instance name; the executor receives the merge of those
    env vars with the role's own ``extra_env`` (role wins on
    collision).
    """
    ws = _FakeWorkspace(tmp_path)
    template = _container_template()
    pattern = Pattern(
        name="trainer-loop",
        roles=[],
        instances=[
            BlockInstance(
                name="trainer",
                block="train",
                trigger=ContinuousTrigger(),
                cardinality=1,
                prompt=None,
                extra_env={
                    "RUN_TAG": "role-wins",
                    "ROLE_KEY": "role-only",
                },
            ),
        ],
    )
    executor = _RecordingExecutor()
    runtime_cfg = {
        "trainer": InstanceRuntimeConfig(
            extra_env={
                "RUN_TAG": "runtime-loses",
                "RUNTIME_KEY": "runtime-only",
            },
            extra_mounts=[
                ("/host/data", "/opt/data", "ro"),
            ],
        ),
    }

    runner = PatternRunner(
        pattern,
        defaults=_defaults(tmp_path, ws, template),
        executor_factory=lambda _bd: executor,
        per_instance_runtime_config=runtime_cfg,
        poll_interval_s=0.01,
    )

    _drain_first_handle(executor)
    runner.run()

    call = executor.calls[0]
    assert call["extra_env"] == {
        "RUN_TAG": "role-wins",
        "ROLE_KEY": "role-only",
        "RUNTIME_KEY": "runtime-only",
    }
    assert call["extra_mounts"] == [
        ("/host/data", "/opt/data", "ro"),
    ]


def test_predecessor_id_flows_through_overrides(
    tmp_path: Path,
) -> None:
    """``predecessor_id`` is forwarded to the executor via overrides.

    The runner does not assume any executor honours it — agent
    batteries do, generic container executors don't and ignore
    the unknown override per the protocol's "unknown keys are
    silent" contract.  Verifying it lands in the dict is enough;
    each executor's behaviour is tested in its own suite.
    """
    ws = _FakeWorkspace(tmp_path)
    template = _container_template()
    pattern = _continuous_pattern()
    executor = _RecordingExecutor()

    runner = PatternRunner(
        pattern,
        defaults=_defaults(tmp_path, ws, template),
        executor_factory=lambda _bd: executor,
        poll_interval_s=0.01,
    )
    runner._run_id = "run_test"  # noqa: SLF001
    role = next(iter(runner._state.values())).role  # noqa: SLF001

    handle = runner._launch_role_member(  # noqa: SLF001
        role,
        cohort_index=0,
        member_index=0,
        predecessor_id="prev_ex_0001",
    )
    handle.finish()

    assert len(executor.calls) == 1
    assert executor.calls[0]["predecessor_id"] == "prev_ex_0001"


def test_executor_dispatch_runs_alongside_agent_role(
    tmp_path: Path,
) -> None:
    """Mixed pattern: a prompt-bearing role and a prompt-less role coexist.

    Both go through the executor seam in the unified dispatch
    model; the runner just builds different ``overrides`` dicts
    based on role declarations.  The order of fires is start-time
    deterministic (the runner iterates ``self._state.values()``
    in insertion order).
    """
    ws = _FakeWorkspace(tmp_path)
    template = _container_template()
    template.blocks.append(BlockDefinition(
        name="play", runner="container",
        image="agent:latest", lifecycle="one_shot",
        inputs=[], outputs=[],
    ))

    prompt_path = tmp_path / "play.md"
    prompt_path.write_text("hi", encoding="utf-8")

    pattern = Pattern(
        name="mixed",
        roles=[],
        instances=[
            BlockInstance(
                name="play", block="play",
                trigger=ContinuousTrigger(),
                cardinality=1,
                prompt="play.md",
            ),
            BlockInstance(
                name="trainer", block="train",
                trigger=ContinuousTrigger(),
                cardinality=1,
                prompt=None,
            ),
        ],
    )

    executor = _RecordingExecutor()
    runner = PatternRunner(
        pattern,
        defaults=_defaults(tmp_path, ws, template),
        executor_factory=lambda _bd: executor,
        poll_interval_s=0.01,
    )

    def _finisher() -> None:
        while len(executor.handles) < 2:
            pass
        executor.handles[0].finish()
        executor.handles[1].finish()

    threading.Thread(target=_finisher, daemon=True).start()
    result = runner.run()

    assert len(executor.calls) == 2
    block_names = {c["block_name"] for c in executor.calls}
    assert block_names == {"play", "train"}
    play_call = next(
        c for c in executor.calls if c["block_name"] == "play")
    train_call = next(
        c for c in executor.calls if c["block_name"] == "train")
    # The play role carries a prompt; trainer doesn't.
    assert play_call["prompt"] == "hi"
    assert "prompt" not in train_call
    assert result.cohorts_by_role == {"play": 1, "trainer": 1}
