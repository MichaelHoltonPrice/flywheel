"""Tests for AgentHandle and launch_agent_block.

Tests the non-blocking agent handle API using mock subprocesses
(no actual Docker containers).
"""

from __future__ import annotations

import threading
import time
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest

from flywheel.agent import (
    AgentHandle,
    AgentResult,
    launch_agent_block,
    prepare_agent_workspace,
)
from flywheel.execution_channel import ExecutionChannel


def _make_handle(
    *,
    exit_code: int = 0,
    stdout_lines: list[str] | None = None,
    stderr_lines: list[str] | None = None,
    workspace: MagicMock | None = None,
    agent_ws: Path | None = None,
    output_names: list[str] | None = None,
    bridge: MagicMock | None = None,
    predecessor_id: str | None = None,
) -> AgentHandle:
    """Create an AgentHandle with a mock process."""
    process = MagicMock()
    process.returncode = exit_code
    process.poll.return_value = exit_code
    process.wait.return_value = exit_code

    # Mock stdout as a line iterator that finishes immediately.
    stdout_data = "\n".join(stdout_lines or []) + "\n"
    process.stdout = StringIO(stdout_data)

    stderr_data = "\n".join(stderr_lines or [])
    process.stderr = StringIO(stderr_data)

    if bridge is None:
        bridge = MagicMock(spec=ExecutionChannel)

    if workspace is None:
        workspace = MagicMock()
        workspace.executions = {}
        workspace.artifacts = {}

    # Stdout and stderr threads that finish immediately.
    stdout_thread = MagicMock(spec=threading.Thread)
    stderr_thread = MagicMock(spec=threading.Thread)

    return AgentHandle(
        process=process,
        bridge=bridge,
        workspace=workspace,
        agent_ws=agent_ws or Path("/tmp/agent_ws"),
        output_names=output_names,
        start_time=time.monotonic(),
        executions_before=0,
        agent_image="test-image:latest",
        stdout_thread=stdout_thread,
        stderr_thread=stderr_thread,
        predecessor_id=predecessor_id,
    )


class TestAgentHandleBasics:
    def test_wait_returns_agent_result(self):
        handle = _make_handle(exit_code=0)
        result = handle.wait()
        assert isinstance(result, AgentResult)
        assert result.exit_code == 0
        assert result.elapsed_s >= 0

    def test_wait_joins_threads(self):
        handle = _make_handle()
        handle.wait()
        handle._stdout_thread.join.assert_called_once()
        handle._stderr_thread.join.assert_called_once_with(timeout=5)

    def test_wait_stops_bridge(self):
        bridge = MagicMock(spec=ExecutionChannel)
        handle = _make_handle(bridge=bridge)
        handle.wait()
        bridge.stop.assert_called_once()

    def test_bridge_stopped_even_on_error(self):
        bridge = MagicMock(spec=ExecutionChannel)
        handle = _make_handle(bridge=bridge)
        # Make process.wait() raise to simulate an error.
        handle._process.wait.side_effect = OSError("boom")
        with pytest.raises(OSError):
            handle.wait()
        bridge.stop.assert_called_once()


class TestAgentHandleStop:
    @mock_patch("flywheel.agent.subprocess.run")
    def test_stop_calls_docker_exec(self, mock_run):
        handle = _make_handle()
        handle._container_name = "flywheel-test123"
        handle.stop()
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[:3] == ["docker", "exec", "flywheel-test123"]
        assert "/workspace/.agent_stop" in args

    @mock_patch("flywheel.agent.subprocess.run")
    def test_stop_then_wait(self, mock_run):
        handle = _make_handle(exit_code=0)
        handle._container_name = "flywheel-test123"
        handle.stop()
        result = handle.wait()
        assert result.exit_code == 0


class TestAgentHandleIsAlive:
    def test_alive_while_running(self):
        handle = _make_handle()
        handle._process.poll.return_value = None
        assert handle.is_alive() is True

    def test_not_alive_after_exit(self):
        handle = _make_handle()
        handle._process.poll.return_value = 0
        assert handle.is_alive() is False


class TestAgentHandleDoubleWait:
    def test_double_wait_raises(self):
        handle = _make_handle()
        handle.wait()
        with pytest.raises(RuntimeError, match="already called"):
            handle.wait()


class TestAgentHandleArtifacts:
    def test_collects_output_artifacts(self, tmp_path: Path):
        """Output files matching output_names are registered."""
        agent_ws = tmp_path / "agent_ws"
        agent_ws.mkdir()
        (agent_ws / "game_log.txt").write_text("log data")
        (agent_ws / "unrelated.txt").write_text("other")

        ws = MagicMock()
        ws.executions = {}
        ws.path = tmp_path

        handle = _make_handle(
            workspace=ws,
            agent_ws=agent_ws,
            output_names=["game_log"],
        )
        handle.wait()

        ws.register_artifact.assert_called_once()
        call_args = ws.register_artifact.call_args
        assert call_args[0][0] == "game_log"
        assert call_args[0][1].name == "game_log.txt"

    def test_no_output_names_skips_collection(self, tmp_path: Path):
        agent_ws = tmp_path / "agent_ws"
        agent_ws.mkdir()
        (agent_ws / "something.txt").write_text("data")

        ws = MagicMock()
        ws.executions = {}

        handle = _make_handle(
            workspace=ws,
            agent_ws=agent_ws,
            output_names=None,
        )
        handle.wait()
        ws.register_artifact.assert_not_called()


class TestAgentHandleEvals:
    def test_evals_counted_from_executions(self):
        ws = MagicMock()
        ws.executions = {"e1": None, "e2": None}

        handle = _make_handle(workspace=ws)
        # executions_before was 0, now there are 2.
        result = handle.wait()
        assert result.evals_run == 2


class TestLaunchFailure:
    def test_bridge_stopped_on_popen_failure(self, tmp_path: Path):
        """If Popen raises, launch_agent_block stops the bridge."""
        ws = MagicMock()
        ws.path = tmp_path
        ws.executions = {}
        ws.artifacts = {}
        ws.instances_for = MagicMock(return_value=[])

        template = MagicMock()
        bridge_mock = MagicMock()
        bridge_mock.start.return_value = 9999

        with (
            mock_patch("flywheel.agent.ExecutionChannel",
                       return_value=bridge_mock),
            mock_patch("flywheel.agent.subprocess.Popen",
                       side_effect=OSError("Docker not found")),
            pytest.raises(OSError, match="Docker not found"),
        ):
            launch_agent_block(
                workspace=ws,
                template=template,
                project_root=tmp_path,
                prompt="test",
            )

        # Bridge must be stopped even though Popen failed.
        bridge_mock.stop.assert_called_once()


class TestStopReason:
    @mock_patch("flywheel.agent.subprocess.run")
    def test_stop_stores_reason(self, mock_run):
        handle = _make_handle()
        handle._container_name = "flywheel-test123"
        handle.stop(reason="exploration_request")
        assert handle._stop_reason == "exploration_request"

    @mock_patch("flywheel.agent.subprocess.run")
    def test_stop_default_reason(self, mock_run):
        handle = _make_handle()
        handle._container_name = "flywheel-test123"
        handle.stop()
        assert handle._stop_reason == "requested"

    def test_wait_returns_stop_reason(self):
        handle = _make_handle()
        handle._stop_reason = "prediction_mismatch"
        result = handle.wait()
        assert result.stop_reason == "prediction_mismatch"

    def test_wait_returns_none_when_not_stopped(self):
        handle = _make_handle()
        result = handle.wait()
        assert result.stop_reason is None


class TestExecutionRecording:
    def test_wait_records_execution(self):
        ws = MagicMock()
        ws.executions = {}
        ws.generate_execution_id.return_value = "exec_test1"
        ws.generate_event_id.return_value = "evt_test1"

        handle = _make_handle(workspace=ws, exit_code=0)
        result = handle.wait()

        ws.add_execution.assert_called_once()
        execution = ws.add_execution.call_args[0][0]
        assert execution.id == "exec_test1"
        assert execution.block_name == "__agent__"
        assert execution.status == "succeeded"
        assert execution.image == "test-image:latest"
        assert result.execution_id == "exec_test1"

    def test_failed_execution_recorded(self):
        ws = MagicMock()
        ws.executions = {}
        ws.generate_execution_id.return_value = "exec_fail"

        handle = _make_handle(workspace=ws, exit_code=1)
        handle.wait()

        execution = ws.add_execution.call_args[0][0]
        assert execution.status == "failed"
        assert execution.exit_code == 1

    def test_stopped_execution_recorded_as_interrupted(self):
        ws = MagicMock()
        ws.executions = {}
        ws.generate_execution_id.return_value = "exec_stop"
        ws.generate_event_id.return_value = "evt_stop"

        handle = _make_handle(workspace=ws, exit_code=0)
        handle._stop_reason = "exploration_request"
        handle.wait()

        execution = ws.add_execution.call_args[0][0]
        assert execution.status == "interrupted"
        assert execution.stop_reason == "exploration_request"

    def test_predecessor_id_recorded(self):
        ws = MagicMock()
        ws.executions = {}
        ws.generate_execution_id.return_value = "exec_resume"

        handle = _make_handle(
            workspace=ws, predecessor_id="exec_prev")
        handle.wait()

        execution = ws.add_execution.call_args[0][0]
        assert execution.predecessor_id == "exec_prev"

    def test_lifecycle_event_on_stop(self):
        ws = MagicMock()
        ws.executions = {}
        ws.generate_execution_id.return_value = "exec_ev"
        ws.generate_event_id.return_value = "evt_ev"

        handle = _make_handle(workspace=ws)
        handle._stop_reason = "timeout"
        handle.wait()

        ws.add_event.assert_called_once()
        event = ws.add_event.call_args[0][0]
        assert event.kind == "agent_stopped"
        assert event.execution_id == "exec_ev"
        assert event.detail == {"reason": "timeout"}

    def test_no_lifecycle_event_on_normal_exit(self):
        ws = MagicMock()
        ws.executions = {}
        ws.generate_execution_id.return_value = "exec_ok"

        handle = _make_handle(workspace=ws)
        handle.wait()

        ws.add_event.assert_not_called()

    def test_wait_saves_workspace(self):
        ws = MagicMock()
        ws.executions = {}
        ws.generate_execution_id.return_value = "exec_s"

        handle = _make_handle(workspace=ws)
        handle.wait()

        ws.save.assert_called_once()


class TestPrepareAgentWorkspace:
    def test_creates_fresh_directory(self, tmp_path: Path):
        ws = MagicMock()
        ws.path = tmp_path
        ws.instances_for = MagicMock(return_value=[])

        agent_ws = prepare_agent_workspace(ws)
        assert agent_ws.exists()
        assert agent_ws.is_dir()
        assert agent_ws.name == "agent_workspace"

    def test_custom_dir_name(self, tmp_path: Path):
        ws = MagicMock()
        ws.path = tmp_path
        ws.instances_for = MagicMock(return_value=[])

        agent_ws = prepare_agent_workspace(
            ws, agent_workspace_dir="explore_0")
        assert agent_ws.name == "explore_0"
        assert agent_ws.exists()

    def test_removes_existing_directory(self, tmp_path: Path):
        ws = MagicMock()
        ws.path = tmp_path
        ws.instances_for = MagicMock(return_value=[])

        # Pre-create with a file.
        existing = tmp_path / "agent_workspace"
        existing.mkdir()
        (existing / "old_file.txt").write_text("old")

        agent_ws = prepare_agent_workspace(ws)
        assert agent_ws.exists()
        assert not (agent_ws / "old_file.txt").exists()

    def test_seeds_latest_artifacts(self, tmp_path: Path):
        ws = MagicMock()
        ws.path = tmp_path

        # Create a mock artifact instance.
        art_dir = tmp_path / "artifacts" / "game_log@abc"
        art_dir.mkdir(parents=True)
        (art_dir / "game_log.txt").write_text("log data")

        inst = MagicMock()
        inst.kind = "copy"
        inst.copy_path = "game_log@abc"
        ws.instances_for = MagicMock(return_value=[inst])

        agent_ws = prepare_agent_workspace(
            ws, output_names=["game_log"])
        assert (agent_ws / "game_log.txt").read_text() == "log data"

    def test_no_output_names_skips_seeding(self, tmp_path: Path):
        ws = MagicMock()
        ws.path = tmp_path

        agent_ws = prepare_agent_workspace(ws, output_names=None)
        assert list(agent_ws.iterdir()) == []
