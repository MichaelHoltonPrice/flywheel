"""Tests for the pure-container dispatch path in :class:`PatternRunner`.

Slice-3 step 1 added a branch in
:meth:`PatternRunner._launch_role_member` that bypasses the agent
launch path entirely when a pattern instance declares no
``prompt:``.  This module exercises that branch end-to-end against
a fake :class:`flywheel.executor.BlockExecutor`, with no agent
machinery in the picture.

The new path is what unblocks pure-container patterns: a continuous
or autorestart role can drive a one-shot training container, a
workspace-persistent runtime, or any other block-shaped executor
without inheriting any of the agent battery's surface (no
``prompt``, no ``model``, no ``mcp_servers``, no ``HANDOFF_TOOLS``).
"""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flywheel.agent import AgentBlockConfig
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
    """Programmable :class:`ExecutionHandle` stand-in.

    Stays alive until :meth:`finish` is called or :meth:`stop` is
    invoked, at which point the next :meth:`wait` returns the
    pre-built :class:`ExecutionResult`.  The runner only relies
    on ``is_alive`` / ``wait`` / ``stop``, so mirroring those
    three is enough.
    """

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

    The runner is expected to call ``launch`` with the protocol
    arguments (``block_name``, ``workspace``, ``input_bindings``)
    plus protocol-level extras (``run_id``) and the well-known
    container extras (``extra_env``, ``extra_mounts``).  Tests
    inspect ``calls`` to verify the handoff is correct.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.handles: list[_FakeExecutionHandle] = []

    def launch(
        self,
        block_name: str,
        workspace,  # noqa: ANN001
        input_bindings: dict[str, str],
        **extras,
    ) -> _FakeExecutionHandle:
        recorded = {
            "block_name": block_name,
            "workspace": workspace,
            "input_bindings": dict(input_bindings),
            **extras,
        }
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


def _base_config(
    tmp_path: Path, workspace: _FakeWorkspace, template: Template,
) -> AgentBlockConfig:
    """Build the (still-required) ``base_config`` for the runner.

    Slice-3 step 2 replaces this with :class:`RunDefaults`; for
    step 1 the runner's __init__ still demands ``base_config``.
    """
    return AgentBlockConfig(
        workspace=workspace,  # type: ignore[arg-type]
        template=template,
        project_root=tmp_path,
        prompt="",
    )


def _continuous_pattern(role_name: str = "trainer") -> Pattern:
    """One continuous role, no prompt → executor-dispatch path.

    The :func:`_role_from_instance` helper inside the pattern
    runner sets ``role.prompt = inst.prompt or ""``; passing
    ``prompt=None`` here is what makes
    ``_role_uses_executor_dispatch`` return True.
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


# ── Tests ────────────────────────────────────────────────────────


def test_continuous_pure_container_runs_via_executor_factory(
    tmp_path: Path,
) -> None:
    """A prompt-less continuous role dispatches via the executor seam.

    The runner must call ``executor_factory(block_def).launch(...)``
    instead of any agent launcher.  Verifies the protocol-level
    arguments land correctly and the run terminates when the
    handle finishes naturally.
    """
    ws = _FakeWorkspace(tmp_path)
    template = _container_template()
    pattern = _continuous_pattern()
    executor = _RecordingExecutor()

    runner = PatternRunner(
        pattern,
        base_config=_base_config(tmp_path, ws, template),
        launch_fn=lambda **_: pytest.fail(
            "agent launch_fn must not be invoked for "
            "prompt-less roles"),
        executor_factory=lambda _block_def: executor,
        poll_interval_s=0.01,
    )

    # The continuous handle is alive after the runner fires the
    # role; finish it on a worker thread so ``run()`` terminates.
    import threading

    def _finisher() -> None:
        # Spin until at least one launch has happened.
        while not executor.handles:
            pass
        executor.handles[0].finish()

    t = threading.Thread(target=_finisher, daemon=True)
    t.start()
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
    assert result.cohorts_by_role == {"trainer": 1}
    assert result.agents_launched == 1


def test_executor_dispatch_forwards_role_extra_env(
    tmp_path: Path,
) -> None:
    """``role.extra_env`` reaches the executor as the ``extra_env`` extra.

    Mirrors the on_tool dispatch path: per-instance env knobs
    declared on the YAML instance flow through to the executor's
    ``launch(extra_env=...)`` extra so container blocks can be
    parameterised without baking the value into the image.
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
        base_config=_base_config(tmp_path, ws, template),
        launch_fn=lambda **_: pytest.fail(
            "agent launch_fn must not be invoked"),
        executor_factory=lambda _bd: executor,
        poll_interval_s=0.01,
    )

    import threading

    def _finisher() -> None:
        while not executor.handles:
            pass
        executor.handles[0].finish()

    threading.Thread(target=_finisher, daemon=True).start()
    runner.run()

    assert executor.calls[0]["extra_env"] == {
        "SEED": "42", "RUN_TAG": "step1"}


def test_executor_dispatch_without_factory_raises(
    tmp_path: Path,
) -> None:
    """Prompt-less role + no executor_factory must raise loudly.

    The runner cannot guess which executor to use; failing early
    surfaces the misconfiguration at the first fire instead of
    silently falling through to the agent launcher.
    """
    ws = _FakeWorkspace(tmp_path)
    template = _container_template()
    pattern = _continuous_pattern()

    runner = PatternRunner(
        pattern,
        base_config=_base_config(tmp_path, ws, template),
        launch_fn=lambda **_: pytest.fail(
            "agent launch_fn must not be invoked"),
        executor_factory=None,
        poll_interval_s=0.01,
    )

    with pytest.raises(ValueError, match="no ``executor_factory``"):
        runner.run()


def test_executor_dispatch_unknown_block_raises(
    tmp_path: Path,
) -> None:
    """Pattern referencing a block not in the template raises.

    The check lives in the executor-dispatch branch (the agent
    path's equivalent is implicit through ``launch_agent_block``);
    surfacing it here keeps the failure attributable to the
    pattern, not to a downstream NoneType deref.
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
        base_config=_base_config(tmp_path, ws, template),
        launch_fn=lambda **_: None,
        executor_factory=lambda _bd: executor,
        poll_interval_s=0.01,
    )

    with pytest.raises(ValueError, match="not declared in the template"):
        runner.run()


def test_per_instance_runtime_config_extra_env_merges(
    tmp_path: Path,
) -> None:
    """``per_instance_runtime_config`` env layers under ``role.extra_env``.

    Mirrors the on_tool dispatch path semantics: the project
    hook supplies a per-instance runtime config keyed by
    instance name; the executor receives the merge of those env
    vars with the role's own ``extra_env`` (role wins on
    collision, matching :meth:`PatternRunner._merge_env`).
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
        base_config=_base_config(tmp_path, ws, template),
        launch_fn=lambda **_: pytest.fail(
            "agent launch_fn must not be invoked"),
        executor_factory=lambda _bd: executor,
        per_instance_runtime_config=runtime_cfg,
        poll_interval_s=0.01,
    )

    import threading

    def _finisher() -> None:
        while not executor.handles:
            pass
        executor.handles[0].finish()

    threading.Thread(target=_finisher, daemon=True).start()
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


def test_predecessor_id_is_dropped_on_executor_path(
    tmp_path: Path,
) -> None:
    """Executor dispatch silently drops ``predecessor_id``.

    The :class:`flywheel.executor.BlockExecutor` protocol does
    not declare a chaining concept; forwarding the kwarg as an
    "extra" would probe executor-specific tolerance for unknown
    arguments.  Instead, the runner suppresses it on this path.
    Tested by calling the dispatch helper directly with a
    predecessor — the recorded launch call must contain neither
    a ``predecessor_id`` key nor any other proxy for chaining.
    """
    ws = _FakeWorkspace(tmp_path)
    template = _container_template()
    pattern = _continuous_pattern()
    executor = _RecordingExecutor()

    runner = PatternRunner(
        pattern,
        base_config=_base_config(tmp_path, ws, template),
        launch_fn=lambda **_: pytest.fail(
            "agent launch_fn must not be invoked"),
        executor_factory=lambda _bd: executor,
        poll_interval_s=0.01,
    )
    # The dispatch helper needs an active run id (normally set
    # at the top of ``run()``); fake one for the unit test.
    runner._run_id = "run_test"  # noqa: SLF001
    role = next(iter(runner._state.values())).role  # noqa: SLF001

    runner._dispatch_role_via_executor(  # noqa: SLF001
        role,
        cohort_index=0,
        member_index=0,
        predecessor_id="prev_ex_0001",
    )

    assert len(executor.calls) == 1
    assert "predecessor_id" not in executor.calls[0]


def test_executor_dispatch_runs_alongside_agent_role(
    tmp_path: Path,
) -> None:
    """Mixed pattern: an agent role and a container role coexist.

    The runner must route each role through its respective path
    (agent → ``launch_fn``; container → ``executor_factory``)
    and terminate when both drivers finish.  The order of fires
    is start-time deterministic (the runner iterates
    ``self._state.values()`` in insertion order).
    """
    from flywheel.agent import AgentResult

    ws = _FakeWorkspace(tmp_path)
    template = _container_template()
    # Add a second block so the agent role has something to point at.
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

    agent_handles: list = []

    class _AgentHandle:
        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def stop(self, reason: str = "") -> None:
            self._alive = False

        def wait(self) -> AgentResult:
            self._alive = False
            return AgentResult(
                exit_code=0, elapsed_s=0.0, evals_run=0,
                exit_reason="completed",
                execution_id="agent_ex",
            )

        def finish(self) -> None:
            self._alive = False

    def _agent_launch_fn(**kwargs):
        h = _AgentHandle()
        agent_handles.append(h)
        return h

    executor = _RecordingExecutor()
    runner = PatternRunner(
        pattern,
        base_config=_base_config(tmp_path, ws, template),
        launch_fn=_agent_launch_fn,
        executor_factory=lambda _bd: executor,
        poll_interval_s=0.01,
    )

    import threading

    def _finisher() -> None:
        while not (agent_handles and executor.handles):
            pass
        agent_handles[0].finish()
        executor.handles[0].finish()

    threading.Thread(target=_finisher, daemon=True).start()
    result = runner.run()

    # Container role hit the executor seam; agent role hit launch_fn.
    assert len(executor.calls) == 1
    assert executor.calls[0]["block_name"] == "train"
    assert len(agent_handles) == 1
    assert result.cohorts_by_role == {"play": 1, "trainer": 1}
