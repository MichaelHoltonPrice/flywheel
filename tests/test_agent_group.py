"""Tests for AgentGroup parallel agent orchestration.

Tests the launch-all-then-wait-sequentially pattern using mocked
launch_agent_block calls (no actual Docker containers).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from flywheel.agent import AgentResult
from flywheel.agent_group import AgentGroup, AgentGroupMember


def _mock_workspace(tmp_path: Path) -> MagicMock:
    """Create a mock workspace rooted at tmp_path."""
    ws = MagicMock()
    ws.path = tmp_path
    ws.executions = {}
    ws.events = {}
    ws.generate_event_id.return_value = "evt_grp1"

    # register_artifact returns a mock with an id.
    def _register(name, path, source=None):
        inst = MagicMock()
        inst.id = f"{name}@mock"
        return inst

    ws.register_artifact = MagicMock(side_effect=_register)
    return ws


def _mock_handle(exit_code: int = 0, elapsed: float = 10.0):
    """Create a mock AgentHandle."""
    handle = MagicMock()
    handle.wait.return_value = AgentResult(
        exit_code=exit_code,
        elapsed_s=elapsed,
        evals_run=0,
        execution_id="exec_mock",
        stop_reason=None,
    )
    return handle


class TestAgentGroupBasics:
    @patch("flywheel.agent_group.launch_agent_block")
    def test_empty_group_returns_empty(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        group = AgentGroup(ws, MagicMock(), tmp_path)
        results = group.run()
        assert results == []
        mock_launch.assert_not_called()

    @patch("flywheel.agent_group.launch_agent_block")
    def test_single_member(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        group = AgentGroup(ws, MagicMock(), tmp_path)
        group.add(AgentGroupMember(
            prompt="test", agent_workspace_dir="ws_0"))
        results = group.run()

        assert len(results) == 1
        assert results[0].index == 0
        assert results[0].agent_workspace_dir == "ws_0"
        assert results[0].agent_result.exit_code == 0
        mock_launch.assert_called_once()

    @patch("flywheel.agent_group.launch_agent_block")
    def test_multiple_members_all_launched(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        group = AgentGroup(ws, MagicMock(), tmp_path)
        for i in range(3):
            group.add(AgentGroupMember(
                prompt=f"task {i}", agent_workspace_dir=f"ws_{i}"))
        results = group.run()

        assert len(results) == 3
        assert mock_launch.call_count == 3

    @patch("flywheel.agent_group.launch_agent_block")
    def test_distinct_workspace_dirs(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        group = AgentGroup(ws, MagicMock(), tmp_path)
        group.add(AgentGroupMember(
            prompt="a", agent_workspace_dir="explore_0"))
        group.add(AgentGroupMember(
            prompt="b", agent_workspace_dir="explore_1"))
        group.run()

        calls = mock_launch.call_args_list
        dirs = [c.kwargs["agent_workspace_dir"] for c in calls]
        assert dirs == ["explore_0", "explore_1"]

    @patch("flywheel.agent_group.launch_agent_block")
    def test_prompts_passed_correctly(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        group = AgentGroup(ws, MagicMock(), tmp_path)
        group.add(AgentGroupMember(
            prompt="prompt A", agent_workspace_dir="ws_0"))
        group.add(AgentGroupMember(
            prompt="prompt B", agent_workspace_dir="ws_1"))
        group.run()

        calls = mock_launch.call_args_list
        prompts = [c.kwargs["prompt"] for c in calls]
        assert prompts == ["prompt A", "prompt B"]


class TestAgentGroupConfig:
    @patch("flywheel.agent_group.launch_agent_block")
    def test_base_kwargs_passed_through(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        group = AgentGroup(
            ws, MagicMock(), tmp_path,
            base_kwargs={
                "agent_image": "custom:v1",
                "model": "claude-sonnet-4-6",
                "isolated_network": True,
            },
        )
        group.add(AgentGroupMember(
            prompt="test", agent_workspace_dir="ws_0"))
        group.run()

        kwargs = mock_launch.call_args.kwargs
        assert kwargs["agent_image"] == "custom:v1"
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["isolated_network"] is True

    @patch("flywheel.agent_group.launch_agent_block")
    def test_member_input_artifacts_override(
        self, mock_launch, tmp_path,
    ):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        group = AgentGroup(
            ws, MagicMock(), tmp_path,
            base_kwargs={"input_artifacts": {"base": "base@1"}},
        )
        group.add(AgentGroupMember(
            prompt="test", agent_workspace_dir="ws_0",
            input_artifacts={"history": "hist@1"},
        ))
        group.run()

        kwargs = mock_launch.call_args.kwargs
        assert kwargs["input_artifacts"] == {"history": "hist@1"}

    @patch("flywheel.agent_group.launch_agent_block")
    def test_member_extra_env_merged(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        group = AgentGroup(
            ws, MagicMock(), tmp_path,
            base_kwargs={"extra_env": {"A": "1", "B": "2"}},
        )
        group.add(AgentGroupMember(
            prompt="test", agent_workspace_dir="ws_0",
            extra_env={"B": "override", "C": "3"},
        ))
        group.run()

        kwargs = mock_launch.call_args.kwargs
        assert kwargs["extra_env"] == {
            "A": "1", "B": "override", "C": "3"}


class TestAgentGroupArtifactCollection:
    @patch("flywheel.agent_group.launch_agent_block")
    def test_collects_matching_files(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        # Create output file in expected workspace dir.
        agent_dir = tmp_path / "ws_0"
        agent_dir.mkdir()
        (agent_dir / "exploration_result.json").write_text('{"ok":true}')

        group = AgentGroup(ws, MagicMock(), tmp_path)
        group.add(AgentGroupMember(
            prompt="test", agent_workspace_dir="ws_0"))
        results = group.run(
            collect_artifacts=[
                ("exploration_result", "exploration_result"),
            ],
        )

        assert len(results[0].artifacts_collected) == 1
        ws.register_artifact.assert_called_once()
        call_args = ws.register_artifact.call_args
        assert call_args[0][0] == "exploration_result"

    @patch("flywheel.agent_group.launch_agent_block")
    def test_missing_file_skipped_without_fallback(
        self, mock_launch, tmp_path,
    ):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        # No output file exists.
        agent_dir = tmp_path / "ws_0"
        agent_dir.mkdir()

        group = AgentGroup(ws, MagicMock(), tmp_path)
        group.add(AgentGroupMember(
            prompt="test", agent_workspace_dir="ws_0"))
        results = group.run(
            collect_artifacts=[("result", "result")],
        )

        assert results[0].artifacts_collected == []
        ws.register_artifact.assert_not_called()

    @patch("flywheel.agent_group.launch_agent_block")
    def test_fallback_fn_called_on_missing_output(
        self, mock_launch, tmp_path,
    ):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        agent_dir = tmp_path / "ws_0"
        agent_dir.mkdir()

        def fallback(index, member):
            return {"prompt": member.prompt, "fallback": True}

        group = AgentGroup(
            ws, MagicMock(), tmp_path, fallback_fn=fallback)
        group.add(AgentGroupMember(
            prompt="test prompt", agent_workspace_dir="ws_0"))
        results = group.run(
            collect_artifacts=[("result", "result")],
        )

        assert len(results[0].artifacts_collected) == 1
        # Verify fallback file was written.
        fallback_file = agent_dir / "result.json"
        assert fallback_file.exists()

    @patch("flywheel.agent_group.launch_agent_block")
    def test_multiple_collect_artifacts(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        agent_dir = tmp_path / "ws_0"
        agent_dir.mkdir()
        (agent_dir / "result.json").write_text("{}")
        (agent_dir / "session.jsonl").write_text("{}")

        group = AgentGroup(ws, MagicMock(), tmp_path)
        group.add(AgentGroupMember(
            prompt="test", agent_workspace_dir="ws_0"))
        results = group.run(
            collect_artifacts=[
                ("result", "exploration_result"),
                ("session", "agent_session"),
            ],
        )

        assert len(results[0].artifacts_collected) == 2
        assert ws.register_artifact.call_count == 2


class TestAgentGroupLifecycleEvent:
    @patch("flywheel.agent_group.launch_agent_block")
    def test_records_group_completed_event(
        self, mock_launch, tmp_path,
    ):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        group = AgentGroup(ws, MagicMock(), tmp_path)
        group.add(AgentGroupMember(
            prompt="test", agent_workspace_dir="ws_0"))
        group.run()

        ws.add_event.assert_called_once()
        event = ws.add_event.call_args[0][0]
        assert event.kind == "group_completed"
        assert event.detail["members"] == "1"
        assert event.detail["succeeded"] == "1"

    @patch("flywheel.agent_group.launch_agent_block")
    def test_event_counts_failures(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)

        # First member succeeds, second fails.
        handles = [_mock_handle(exit_code=0), _mock_handle(exit_code=1)]
        mock_launch.side_effect = handles

        group = AgentGroup(ws, MagicMock(), tmp_path)
        group.add(AgentGroupMember(
            prompt="a", agent_workspace_dir="ws_0"))
        group.add(AgentGroupMember(
            prompt="b", agent_workspace_dir="ws_1"))
        group.run()

        event = ws.add_event.call_args[0][0]
        assert event.detail["members"] == "2"
        assert event.detail["succeeded"] == "1"

    @patch("flywheel.agent_group.launch_agent_block")
    def test_workspace_saved_after_group(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        group = AgentGroup(ws, MagicMock(), tmp_path)
        group.add(AgentGroupMember(
            prompt="test", agent_workspace_dir="ws_0"))
        group.run()

        ws.save.assert_called()
