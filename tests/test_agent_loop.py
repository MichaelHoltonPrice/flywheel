"""Tests for AgentLoop orchestration primitive."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flywheel.agent import AgentBlockConfig, AgentResult
from flywheel.agent_group import AgentGroupMember
from flywheel.agent_loop import (
    AgentLoop,
    Continue,
    Finished,
    LoopState,
    SpawnGroup,
    Stop,
    load_hooks_class,
)


def _mock_result(
    exit_code: int = 0,
    exit_reason: str = "completed",
    execution_id: str = "exec_mock",
) -> AgentResult:
    return AgentResult(
        exit_code=exit_code,
        elapsed_s=10.0,
        evals_run=0,
        execution_id=execution_id,
        stop_reason=None,
        exit_reason=exit_reason,
    )


def _mock_config(tmp_path: Path) -> AgentBlockConfig:
    ws = MagicMock()
    ws.path = tmp_path
    ws.executions = {}
    ws.events = {}
    ws.instances_for = MagicMock(return_value=[])
    ws.generate_event_id.return_value = "evt_mock"

    return AgentBlockConfig(
        workspace=ws,
        template=MagicMock(),
        project_root=tmp_path,
        prompt="ignored",
    )


class SimpleHooks:
    """Hooks that finish after N rounds."""

    def __init__(self, finish_after: int = 1):
        self._finish_after = finish_after
        self._decide_calls = 0
        self._prompts: list[str] = []

    def decide(self, state: LoopState) -> Continue | Finished:
        self._decide_calls += 1
        if self._decide_calls >= self._finish_after:
            return Finished(summary={"done": True})
        return Continue()

    def build_prompt(self, action, state: LoopState) -> str:
        prompt = f"Round {state.round_number}"
        self._prompts.append(prompt)
        return prompt


class TestAgentLoopBasics:
    @patch("flywheel.agent_loop.launch_agent_block")
    def test_single_round_finish(self, mock_launch, tmp_path):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result()))

        config = _mock_config(tmp_path)
        hooks = SimpleHooks(finish_after=1)
        loop = AgentLoop(hooks, config, max_rounds=10)
        result = loop.run()

        assert result["is_finished"] is True
        assert result["rounds_completed"] == 1
        assert result["done"] is True
        mock_launch.assert_called_once()

    @patch("flywheel.agent_loop.launch_agent_block")
    def test_multi_round(self, mock_launch, tmp_path):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result()))

        config = _mock_config(tmp_path)
        hooks = SimpleHooks(finish_after=3)
        loop = AgentLoop(hooks, config, max_rounds=10)
        result = loop.run()

        assert result["rounds_completed"] == 3
        assert mock_launch.call_count == 3

    @patch("flywheel.agent_loop.launch_agent_block")
    def test_max_rounds_budget(self, mock_launch, tmp_path):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result()))

        config = _mock_config(tmp_path)
        # Never finishes — relies on max_rounds.
        hooks = SimpleHooks(finish_after=999)
        loop = AgentLoop(hooks, config, max_rounds=3)
        result = loop.run()

        assert result["rounds_completed"] == 3
        assert result["is_finished"] is False

    @patch("flywheel.agent_loop.launch_agent_block")
    def test_prompt_passed_to_launch(self, mock_launch, tmp_path):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result()))

        config = _mock_config(tmp_path)
        hooks = SimpleHooks(finish_after=1)
        loop = AgentLoop(hooks, config, max_rounds=5)
        loop.run()

        call_kwargs = mock_launch.call_args.kwargs
        assert call_kwargs["prompt"] == "Round 1"


class TestAgentLoopActions:
    @patch("flywheel.agent_loop.launch_agent_block")
    def test_stop_action(self, mock_launch, tmp_path):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result()))

        class StopHooks:
            def decide(self, state):
                return Stop(reason="manual")
            def build_prompt(self, action, state):
                return "go"

        config = _mock_config(tmp_path)
        loop = AgentLoop(StopHooks(), config, max_rounds=10)
        result = loop.run()

        # First round runs (no decide before first round).
        # Second iteration: decide returns Stop.
        assert result["stop_reason"] == "manual"
        assert result["is_finished"] is False

    @patch("flywheel.agent_loop.launch_agent_block")
    def test_finished_with_summary(self, mock_launch, tmp_path):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result()))

        class FinishHooks:
            def decide(self, state):
                return Finished(summary={"score": 100})
            def build_prompt(self, action, state):
                return "go"

        config = _mock_config(tmp_path)
        loop = AgentLoop(FinishHooks(), config, max_rounds=10)
        result = loop.run()

        assert result["is_finished"] is True
        assert result["score"] == 100

    @patch("flywheel.agent_loop.AgentGroup")
    @patch("flywheel.agent_loop.launch_agent_block")
    def test_spawn_group(self, mock_launch, mock_group_cls, tmp_path):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result()))
        mock_group = MagicMock()
        mock_group_cls.return_value = mock_group

        call_count = [0]

        class GroupHooks:
            def decide(self, state):
                call_count[0] += 1
                if call_count[0] == 1:
                    return SpawnGroup(
                        members=[
                            AgentGroupMember(
                                prompt="explore",
                                agent_workspace_dir="ws_0",
                            ),
                        ],
                    )
                return Finished()

            def build_prompt(self, action, state):
                return "go"

        config = _mock_config(tmp_path)
        loop = AgentLoop(GroupHooks(), config, max_rounds=10)
        result = loop.run()

        # Group was created and run.
        mock_group.add.assert_called_once()
        mock_group.run.assert_called_once()
        assert result["groups_run"] == 1


class TestAgentLoopCircuitBreaker:
    @patch("flywheel.agent_loop.launch_agent_block")
    def test_auth_failure_circuit_breaker(
        self, mock_launch, tmp_path,
    ):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result(
                exit_reason="auth_failure")))

        class NeverFinish:
            def decide(self, state):
                return Continue()
            def build_prompt(self, action, state):
                return "go"

        config = _mock_config(tmp_path)
        loop = AgentLoop(
            NeverFinish(), config,
            max_rounds=20,
            max_consecutive_failures=3,
        )
        result = loop.run()

        # Should stop after 3 consecutive auth failures.
        assert result["rounds_completed"] == 3
        assert "circuit_breaker" in result.get("stop_reason", "")

        # Lifecycle event recorded.
        ws = config.workspace
        ws.add_event.assert_called()
        event = ws.add_event.call_args[0][0]
        assert event.kind == "circuit_breaker"

    @patch("flywheel.agent_loop.launch_agent_block")
    def test_success_resets_failure_counter(
        self, mock_launch, tmp_path,
    ):
        results = [
            _mock_result(exit_reason="auth_failure"),
            _mock_result(exit_reason="auth_failure"),
            _mock_result(exit_reason="completed"),  # Resets counter
            _mock_result(exit_reason="auth_failure"),
            _mock_result(exit_reason="auth_failure"),
            _mock_result(exit_reason="completed"),
        ]
        call_idx = [0]

        def make_handle():
            h = MagicMock()
            idx = call_idx[0]
            call_idx[0] += 1
            if idx < len(results):
                h.wait.return_value = results[idx]
            else:
                h.wait.return_value = _mock_result()
            return h

        mock_launch.side_effect = lambda **kw: make_handle()

        class ContinueForever:
            _count = 0
            def decide(self, state):
                self._count += 1
                if self._count >= 6:
                    return Finished()
                return Continue()
            def build_prompt(self, action, state):
                return "go"

        config = _mock_config(tmp_path)
        loop = AgentLoop(
            ContinueForever(), config,
            max_rounds=20,
            max_consecutive_failures=3,
        )
        result = loop.run()

        # Should complete — success resets the counter.
        assert result["is_finished"] is True
        assert result["rounds_completed"] == 6

    @patch("flywheel.agent_loop.launch_agent_block")
    def test_rate_limit_triggers_breaker(
        self, mock_launch, tmp_path,
    ):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result(
                exit_reason="rate_limit")))

        class NeverFinish:
            def decide(self, state):
                return Continue()
            def build_prompt(self, action, state):
                return "go"

        config = _mock_config(tmp_path)
        loop = AgentLoop(
            NeverFinish(), config,
            max_rounds=20,
            max_consecutive_failures=2,
        )
        result = loop.run()

        assert result["rounds_completed"] == 2
        assert "rate_limit" in result.get("stop_reason", "")


class TestAgentLoopResume:
    @patch("flywheel.agent_loop.launch_agent_block")
    def test_resume_from_existing_executions(
        self, mock_launch, tmp_path,
    ):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result()))

        config = _mock_config(tmp_path)

        # Simulate 2 prior agent executions in workspace.
        prior_exec = MagicMock()
        prior_exec.block_name = "__agent__"
        prior_exec.id = "exec_prior"
        config.workspace.executions = {
            "exec_prior": prior_exec,
            "exec_prior2": prior_exec,
        }

        prompts_seen = []

        class ResumeHooks:
            def decide(self, state):
                if state.is_resumed:
                    return Continue()
                return Finished()

            def build_prompt(self, action, state):
                prompts_seen.append(state.round_number)
                return f"Round {state.round_number}"

        loop = AgentLoop(ResumeHooks(), config, max_rounds=5)
        result = loop.run()

        assert result["is_finished"] is True
        assert result["rounds_completed"] == 1
        assert prompts_seen[0] == 3

    @patch("flywheel.agent_loop.launch_agent_block")
    def test_predecessor_id_set_on_resume(
        self, mock_launch, tmp_path,
    ):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result(
                execution_id="exec_new")))

        config = _mock_config(tmp_path)
        prior = MagicMock()
        prior.block_name = "__agent__"
        prior.id = "exec_prev"
        config.workspace.executions = {"exec_prev": prior}

        call_count = [0]

        class ContinueThenFinish:
            def decide(self, state):
                call_count[0] += 1
                if call_count[0] == 1:
                    return Continue()  # Resume round
                return Finished()

            def build_prompt(self, action, state):
                return "go"

        loop = AgentLoop(ContinueThenFinish(), config, max_rounds=5)
        loop.run()

        # First launch should have predecessor_id from prior exec.
        call_kwargs = mock_launch.call_args.kwargs
        assert call_kwargs["predecessor_id"] == "exec_prev"


class TestAgentLoopOnExecution:
    @patch("flywheel.agent_loop.launch_agent_block")
    def test_on_execution_hook_wired(self, mock_launch, tmp_path):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result()))

        received_events = []

        class HooksWithOnExecution:
            def decide(self, state):
                return Finished()
            def build_prompt(self, action, state):
                return "go"
            def on_execution(self, event, handle):
                received_events.append(event)

        config = _mock_config(tmp_path)
        loop = AgentLoop(HooksWithOnExecution(), config)
        loop.run()

        call_kwargs = mock_launch.call_args.kwargs
        on_record = call_kwargs["on_record"]
        assert on_record is not None

        on_record("test_block", {"test": "data"})
        assert len(received_events) == 1
        assert received_events[0].block_name == "test_block"

    @patch("flywheel.agent_loop.launch_agent_block")
    def test_on_execution_not_wired_without_hook(
        self, mock_launch, tmp_path,
    ):
        mock_launch.return_value = MagicMock(
            wait=MagicMock(return_value=_mock_result()))

        class HooksWithoutOnExecution:
            def decide(self, state):
                return Finished()
            def build_prompt(self, action, state):
                return "go"

        config = _mock_config(tmp_path)
        loop = AgentLoop(HooksWithoutOnExecution(), config)
        loop.run()

        call_kwargs = mock_launch.call_args.kwargs
        assert call_kwargs["on_record"] is None


class TestLoadHooksClass:
    def test_loads_builtin_class(self):
        cls = load_hooks_class(
            "flywheel.agent_loop:AgentLoop")
        assert cls is AgentLoop

    def test_bad_format_raises(self):
        with pytest.raises(ValueError, match="module.path:ClassName"):
            load_hooks_class("no_colon_here")

    def test_missing_module_raises(self):
        with pytest.raises(ImportError):
            load_hooks_class("nonexistent.module:Cls")

    def test_missing_class_raises(self):
        with pytest.raises(AttributeError):
            load_hooks_class(
                "flywheel.agent_loop:NonexistentClass")
