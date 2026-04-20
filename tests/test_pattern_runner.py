"""Tests for :class:`flywheel.pattern_runner.PatternRunner`.

Exercises the runner against an in-memory fake workspace and a
fake ``launch_fn``; no Docker, no ``launch_agent_block``.  The
fakes are kept tight on purpose: each test focuses on one
contract (continuous fires once, every-N fires at the right
cadence, cohort cardinality, termination, reactive triggers
raise) so failures point at exactly which behavior regressed.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flywheel.agent import AgentBlockConfig, AgentResult
from flywheel.artifact import BlockExecution
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
from flywheel.pattern_runner import PatternRunner
from flywheel.template import Template

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

    def instances_for(self, _name: str) -> list:  # noqa: D401
        return []

    def add_succeeded(self, block_name: str) -> None:
        idx = len(self.executions)
        ex = BlockExecution(
            id=f"ex_{idx:04d}",
            block_name=block_name,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            status="succeeded",
        )
        self.executions[ex.id] = ex

    def add_synthetic_failed(self, block_name: str) -> None:
        idx = len(self.executions)
        ex = BlockExecution(
            id=f"ex_{idx:04d}",
            block_name=block_name,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            status="failed",
            synthetic=True,
        )
        self.executions[ex.id] = ex


class _FakeHandle:
    """Programmable handle: stays alive until ``finish()`` is called."""

    def __init__(self, kwargs: dict, name: str = ""):
        self.kwargs = kwargs
        self.name = name
        self._alive = True
        self._waited = False
        self._result = AgentResult(
            exit_code=0, elapsed_s=0.1, evals_run=0,
            exit_reason="completed",
        )

    def is_alive(self) -> bool:
        return self._alive

    def finish(self, *, exit_reason: str = "completed") -> None:
        self._alive = False
        self._result = AgentResult(
            exit_code=0, elapsed_s=0.1, evals_run=0,
            exit_reason=exit_reason,
        )

    def wait(self):  # noqa: ANN201
        self._waited = True
        self._alive = False
        return self._result


def _base_config(
    tmp_path: Path,
    workspace: _FakeWorkspace,
) -> AgentBlockConfig:
    """Build an AgentBlockConfig that points at the fake workspace.

    The runner only reads ``project_root``, ``workspace`` and
    ``template`` from this; everything else is forwarded to the
    fake launch_fn unchanged.
    """
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
            ValueError, match="continuous instance",
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
    no launches at all.  These tests fix the shape by asserting
    the continuous instance actually fires and the on_tool
    instance does not get launched by the runner.
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

        launches: list[_FakeHandle] = []

        def launch_fn(**kwargs):
            handle = _FakeHandle(kwargs)
            launches.append(handle)
            handle.finish()
            return handle

        ws = _FakeWorkspace(tmp_path)
        result = PatternRunner(
            pattern,
            base_config=_base_config(tmp_path, ws),
            launch_fn=launch_fn,
            poll_interval_s=0.01,
            max_total_runtime_s=1.0,
        ).run()

        # Exactly one launch: the continuous ``play`` instance.
        # The on_tool ``execute_action`` instance does not get
        # launched by the runner — dispatch is host-side.
        assert len(launches) == 1
        assert result.agents_launched == 1
        # The launch carried the instance's block name
        # (``play``), not the instance's own name (also ``play``
        # here, but the mechanism would differ if they did).
        assert launches[0].kwargs["block_name"] == "play"

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
