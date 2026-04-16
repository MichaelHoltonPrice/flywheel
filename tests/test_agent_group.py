"""Tests for AgentGroup backward-compatibility wrapper.

Verifies that the legacy AgentGroup/AgentGroupMember API still
works correctly by delegating to BlockGroup.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from flywheel.agent import AgentResult
from flywheel.agent_group import AgentGroup, AgentGroupMember


def _mock_workspace(tmp_path: Path) -> MagicMock:
    ws = MagicMock()
    ws.path = tmp_path
    ws.executions = {}
    ws.events = {}
    ws.generate_event_id.return_value = "evt_grp1"

    def _register(name, path, source=None):
        inst = MagicMock()
        inst.id = f"{name}@mock"
        return inst

    ws.register_artifact = MagicMock(side_effect=_register)
    return ws


def _mock_handle(exit_code: int = 0, elapsed: float = 10.0):
    handle = MagicMock()
    handle.wait.return_value = AgentResult(
        exit_code=exit_code,
        elapsed_s=elapsed,
        evals_run=0,
        execution_id="exec_mock",
        stop_reason=None,
    )
    return handle


class TestAgentGroupLegacy:
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
    def test_multiple_members(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        group = AgentGroup(ws, MagicMock(), tmp_path)
        for i in range(3):
            group.add(AgentGroupMember(
                prompt=f"task {i}",
                agent_workspace_dir=f"ws_{i}"))
        results = group.run()

        assert len(results) == 3
        assert mock_launch.call_count == 3

    @patch("flywheel.agent_group.launch_agent_block")
    def test_base_kwargs_and_overrides(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        group = AgentGroup(
            ws, MagicMock(), tmp_path,
            base_kwargs={
                "agent_image": "custom:v1",
                "extra_env": {"A": "1", "B": "2"},
            },
        )
        group.add(AgentGroupMember(
            prompt="test", agent_workspace_dir="ws_0",
            extra_env={"B": "override", "C": "3"},
        ))
        group.run()

        kwargs = mock_launch.call_args.kwargs
        assert kwargs["agent_image"] == "custom:v1"
        assert kwargs["extra_env"] == {
            "A": "1", "B": "override", "C": "3"}

    @patch("flywheel.agent_group.launch_agent_block")
    def test_artifact_collection(self, mock_launch, tmp_path):
        ws = _mock_workspace(tmp_path)
        mock_launch.return_value = _mock_handle()

        agent_dir = tmp_path / "ws_0"
        agent_dir.mkdir()
        (agent_dir / "result.json").write_text('{"ok":true}')

        group = AgentGroup(ws, MagicMock(), tmp_path)
        group.add(AgentGroupMember(
            prompt="test", agent_workspace_dir="ws_0"))
        results = group.run(
            collect_artifacts=[("result", "exploration_result")])

        assert len(results[0].artifacts_collected) == 1

    @patch("flywheel.agent_group.launch_agent_block")
    def test_fallback_fn(self, mock_launch, tmp_path):
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
            collect_artifacts=[("result", "result")])

        assert len(results[0].artifacts_collected) == 1
        fallback_file = agent_dir / "result.json"
        assert fallback_file.exists()
