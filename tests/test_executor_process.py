"""Tests for ProcessExecutor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from flywheel.executor import ProcessExecutionHandle, ProcessExecutor
from flywheel.workspace import Workspace


def _mock_workspace() -> MagicMock:
    ws = MagicMock(spec=Workspace)
    ws.executions = {}
    ws.generate_execution_id.return_value = "exec_proc1"
    return ws


class TestProcessExecutorLaunch:
    @patch("flywheel.executor.subprocess.Popen")
    def test_launches_subprocess_with_command_string(
        self, mock_popen,
    ):
        ws = _mock_workspace()
        executor = ProcessExecutor()

        mock_popen.return_value = MagicMock()
        mock_popen.return_value.poll.return_value = None

        handle = executor.launch(
            "game_server", ws, {},
            command="python server.py",
        )

        mock_popen.assert_called_once()
        assert mock_popen.call_args[0][0] == "python server.py"
        assert mock_popen.call_args[1]["shell"] is True
        assert isinstance(handle, ProcessExecutionHandle)

    @patch("flywheel.executor.subprocess.Popen")
    def test_launches_subprocess_with_command_list(
        self, mock_popen,
    ):
        ws = _mock_workspace()
        executor = ProcessExecutor()

        mock_popen.return_value = MagicMock()
        executor.launch(
            "game_server", ws, {},
            command=["python", "server.py"],
        )

        mock_popen.assert_called_once()
        assert mock_popen.call_args[0][0] == ["python", "server.py"]

    def test_requires_command(self):
        ws = _mock_workspace()
        executor = ProcessExecutor()

        with pytest.raises(ValueError, match="requires a command"):
            executor.launch("test", ws, {})

    def test_block_not_allowed_raises(self):
        ws = _mock_workspace()
        executor = ProcessExecutor()

        with pytest.raises(ValueError, match="not in allowed"):
            executor.launch(
                "test", ws, {},
                command="echo hi",
                allowed_blocks=["other"],
            )


class TestProcessExecutionHandle:
    def test_is_alive(self):
        process = MagicMock()
        process.poll.return_value = None

        handle = ProcessExecutionHandle(
            process=process,
            workspace=_mock_workspace(),
            block_name="test",
            execution_id="exec_1",
            started_at=MagicMock(),
            start_monotonic=0.0,
        )

        assert handle.is_alive() is True
        process.poll.return_value = 0
        assert handle.is_alive() is False

    def test_wait_records_execution(self):
        process = MagicMock()
        process.returncode = 0
        ws = _mock_workspace()

        handle = ProcessExecutionHandle(
            process=process,
            workspace=ws,
            block_name="test",
            execution_id="exec_1",
            started_at=MagicMock(),
            start_monotonic=0.0,
        )

        result = handle.wait()
        assert result.exit_code == 0
        assert result.status == "succeeded"
        assert result.execution_id == "exec_1"
        ws.add_execution.assert_called_once()
        ws.save.assert_called_once()

    def test_wait_failed_exit(self):
        process = MagicMock()
        process.returncode = 1
        ws = _mock_workspace()

        handle = ProcessExecutionHandle(
            process=process,
            workspace=ws,
            block_name="test",
            execution_id="exec_1",
            started_at=MagicMock(),
            start_monotonic=0.0,
        )

        result = handle.wait()
        assert result.exit_code == 1
        assert result.status == "failed"

    def test_stop_terminates_process(self):
        process = MagicMock()
        process.returncode = 0
        ws = _mock_workspace()

        handle = ProcessExecutionHandle(
            process=process,
            workspace=ws,
            block_name="test",
            execution_id="exec_1",
            started_at=MagicMock(),
            start_monotonic=0.0,
        )

        handle.stop(reason="timeout")
        process.terminate.assert_called_once()

        result = handle.wait()
        assert result.status == "interrupted"

        ex = ws.add_execution.call_args[0][0]
        assert ex.stop_reason == "timeout"

    def test_double_wait_raises(self):
        process = MagicMock()
        process.returncode = 0
        ws = _mock_workspace()

        handle = ProcessExecutionHandle(
            process=process,
            workspace=ws,
            block_name="test",
            execution_id="exec_1",
            started_at=MagicMock(),
            start_monotonic=0.0,
        )

        handle.wait()
        with pytest.raises(RuntimeError, match="already called"):
            handle.wait()

    @patch("flywheel.executor.subprocess.Popen")
    def test_end_to_end_short_process(self, mock_popen):
        """Launch a real-ish process and wait for it."""
        ws = _mock_workspace()
        executor = ProcessExecutor()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        handle = executor.launch(
            "test_block", ws, {},
            command="echo hello",
        )

        result = handle.wait()
        assert result.exit_code == 0
        assert result.status == "succeeded"
        assert result.output_bindings == {}
