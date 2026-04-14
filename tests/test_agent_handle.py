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

import pytest

from flywheel.agent import AgentHandle, AgentResult
from flywheel.block_bridge import BlockBridgeService


def _make_handle(
    *,
    exit_code: int = 0,
    stdout_lines: list[str] | None = None,
    stderr_lines: list[str] | None = None,
    workspace: MagicMock | None = None,
    agent_ws: Path | None = None,
    output_names: list[str] | None = None,
    bridge: MagicMock | None = None,
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
        bridge = MagicMock(spec=BlockBridgeService)

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
        bridge = MagicMock(spec=BlockBridgeService)
        handle = _make_handle(bridge=bridge)
        handle.wait()
        bridge.stop.assert_called_once()

    def test_bridge_stopped_even_on_error(self):
        bridge = MagicMock(spec=BlockBridgeService)
        handle = _make_handle(bridge=bridge)
        # Make process.wait() raise to simulate an error.
        handle._process.wait.side_effect = OSError("boom")
        with pytest.raises(OSError):
            handle.wait()
        bridge.stop.assert_called_once()


class TestAgentHandleKill:
    def test_kill_calls_process_kill(self):
        handle = _make_handle()
        handle._process.poll.return_value = None  # Still alive.
        handle.kill()
        handle._process.kill.assert_called_once()

    def test_kill_then_wait(self):
        handle = _make_handle(exit_code=-9)
        handle.kill()
        result = handle.wait()
        assert result.exit_code == -9

    def test_kill_already_exited_is_safe(self):
        handle = _make_handle()
        handle._process.kill.side_effect = OSError("already exited")
        handle.kill()  # Should not raise.


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
