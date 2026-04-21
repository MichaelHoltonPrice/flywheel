"""Tests for :class:`flywheel.agent.AgentHandle`.

AgentHandle is a thin adapter around
:class:`flywheel.executor.ContainerExecutionHandle`: it delegates
lifecycle (``is_alive`` / ``stop`` / ``wait``) to the inner
handle and translates the returned
:class:`flywheel.executor.ExecutionResult` into an
:class:`flywheel.agent.AgentResult` with agent-specific fields
(``exit_reason`` from the agent exit-state artifact,
``evals_run`` from the workspace-execution delta).

These tests exercise that translation layer with mocked inner
handles; the underlying container / executor pipeline has its
own coverage in :mod:`tests.test_process_exit_executor`.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flywheel.agent import AgentHandle, AgentResult
from flywheel.artifact import BlockExecution
from flywheel.executor import ExecutionResult


class _FakeInner:
    """Minimal stand-in for a :class:`ContainerExecutionHandle`.

    Satisfies the adapter's ``is_alive`` / ``stop`` / ``wait``
    interface and records how ``stop`` was called so the
    delegation test can assert on it.
    """

    def __init__(
        self,
        *,
        result: ExecutionResult,
        alive: bool = False,
    ) -> None:
        self._result = result
        self._alive = alive
        self.stop_calls: list[str] = []

    def is_alive(self) -> bool:
        return self._alive

    def stop(self, reason: str = "requested") -> None:
        self.stop_calls.append(reason)
        self._alive = False

    def wait(self) -> ExecutionResult:
        self._alive = False
        return self._result


def _make_workspace(
    tmp_path: Path,
    *,
    execution_id: str = "exec-1",
    exit_code: int = 0,
    stop_reason: str | None = None,
    executions_before: int = 0,
) -> tuple[MagicMock, ExecutionResult]:
    """Build a MagicMock workspace holding the right execution record.

    The agent-exit-state JSON that used to live as a copy artifact
    now lives in the launcher-owned control tempdir instead;
    tests write it via the ``exit_state`` / ``pending`` args to
    :func:`_make_handle`.
    """
    execution = BlockExecution(
        id=execution_id,
        block_name="play",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        status=(
            "interrupted" if stop_reason
            else "succeeded" if exit_code == 0 else "failed"),
        exit_code=exit_code,
        elapsed_s=1.0,
        output_bindings={},
        stop_reason=stop_reason,
    )

    workspace = MagicMock()
    workspace.path = tmp_path
    workspace.executions = {execution_id: execution}
    workspace.artifacts = {}
    workspace.generate_event_id = MagicMock(return_value="ev-1")

    result = ExecutionResult(
        exit_code=exit_code,
        elapsed_s=1.0,
        output_bindings={},
        execution_id=execution_id,
        status=execution.status,
    )
    return workspace, result


def _make_handle(
    *,
    workspace: MagicMock,
    result: ExecutionResult,
    tmp_path: Path,
    executions_before: int = 0,
    exit_state: dict | None = None,
    pending: list[dict] | None = None,
    control_tempdir: Path | None = None,
) -> tuple[AgentHandle, _FakeInner]:
    """Build an AgentHandle over a _FakeInner and a prompt tempdir.

    ``exit_state`` / ``pending`` populate the launcher's control
    tempdir the same way the real agent runner does — the handle
    reads them back in ``wait()``.  Pass ``control_tempdir=None``
    (different from omitting) to simulate the pre-container
    failure path where no control mount was set up.
    """
    inner = _FakeInner(result=result)
    prompt_tempdir = Path(tempfile.mkdtemp(
        prefix="flywheel-test-prompt-", dir=tmp_path))
    if control_tempdir is None and (
            exit_state is not None or pending is not None):
        control_tempdir = Path(tempfile.mkdtemp(
            prefix="flywheel-test-control-", dir=tmp_path))
    if control_tempdir is not None:
        if exit_state is not None:
            (control_tempdir
                / "agent_exit_state.json").write_text(
                json.dumps(exit_state), encoding="utf-8")
        if pending is not None:
            (control_tempdir
                / "pending_tool_calls.json").write_text(
                json.dumps({
                    "schema_version": 2,
                    "session_id": exit_state.get(
                        "session_id", "") if exit_state else "",
                    "pending": pending,
                }),
                encoding="utf-8",
            )
    handle = AgentHandle(
        inner=inner,
        workspace=workspace,
        executions_before=executions_before,
        prompt_tempdir=prompt_tempdir,
        control_tempdir=control_tempdir,
    )
    return handle, inner


class TestAgentHandleBasics:
    def test_wait_returns_agent_result(self, tmp_path: Path):
        ws, result = _make_workspace(tmp_path)
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path)
        out = handle.wait()
        assert isinstance(out, AgentResult)
        assert out.exit_code == 0
        assert out.execution_id == "exec-1"

    def test_wait_called_twice_raises(self, tmp_path: Path):
        ws, result = _make_workspace(tmp_path)
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path)
        handle.wait()
        with pytest.raises(RuntimeError, match="already called"):
            handle.wait()

    def test_is_alive_delegates_to_inner(self, tmp_path: Path):
        ws, result = _make_workspace(tmp_path)
        inner = _FakeInner(result=result, alive=True)
        handle = AgentHandle(
            inner=inner,
            workspace=ws,
            executions_before=0,
            prompt_tempdir=Path(tempfile.mkdtemp(
                prefix="t-", dir=tmp_path)),
            control_tempdir=None,
        )
        assert handle.is_alive() is True

    def test_stop_forwards_to_inner(self, tmp_path: Path):
        ws, result = _make_workspace(
            tmp_path, stop_reason="test_reason")
        handle, inner = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path)
        handle.stop(reason="test_reason")
        assert inner.stop_calls == ["test_reason"]

    def test_prompt_tempdir_cleaned_up_on_wait(
        self, tmp_path: Path,
    ):
        ws, result = _make_workspace(tmp_path)
        inner = _FakeInner(result=result)
        prompt_dir = Path(tempfile.mkdtemp(
            prefix="flywheel-test-prompt-", dir=tmp_path))
        handle = AgentHandle(
            inner=inner,
            workspace=ws,
            executions_before=0,
            prompt_tempdir=prompt_dir,
            control_tempdir=None,
        )
        assert prompt_dir.is_dir()
        handle.wait()
        assert not prompt_dir.exists()


class TestExitReasonClassification:
    """Adapter reads the ``agent_exit_state`` artifact to classify."""

    def test_handoff_status_maps_to_tool_handoff(
        self, tmp_path: Path,
    ):
        ws, result = _make_workspace(tmp_path)
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path,
            exit_state={"status": "tool_handoff",
                        "reason": "take_action"},
        )
        out = handle.wait()
        assert out.exit_reason == "tool_handoff"

    def test_complete_status_maps_to_completed(
        self, tmp_path: Path,
    ):
        ws, result = _make_workspace(tmp_path)
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path,
            exit_state={"status": "complete", "reason": ""},
        )
        out = handle.wait()
        assert out.exit_reason == "completed"

    def test_auth_reason_maps_to_auth_failure(
        self, tmp_path: Path,
    ):
        ws, result = _make_workspace(tmp_path)
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path,
            exit_state={"status": "paused",
                        "reason": "auth_error"},
        )
        out = handle.wait()
        assert out.exit_reason == "auth_failure"

    def test_rate_limit_reason_maps_to_rate_limit(
        self, tmp_path: Path,
    ):
        ws, result = _make_workspace(tmp_path)
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path,
            exit_state={"status": "paused",
                        "reason": "rate_limit"},
        )
        out = handle.wait()
        assert out.exit_reason == "rate_limit"

    def test_max_turns_reason_maps_to_max_turns(
        self, tmp_path: Path,
    ):
        ws, result = _make_workspace(tmp_path)
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path,
            exit_state={"status": "complete",
                        "reason": "max_turns"},
        )
        out = handle.wait()
        assert out.exit_reason == "max_turns"

    def test_stop_reason_takes_precedence_over_exit_state(
        self, tmp_path: Path,
    ):
        """When the execution record has ``stop_reason`` set, the
        classifier returns ``stopped`` regardless of what the
        agent-written exit state says."""
        ws, result = _make_workspace(
            tmp_path,
            stop_reason="operator_requested",
        )
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path,
            exit_state={"status": "complete", "reason": ""},
        )
        out = handle.wait()
        assert out.exit_reason == "stopped"
        assert out.stop_reason == "operator_requested"

    def test_missing_exit_state_defaults_to_exit_code(
        self, tmp_path: Path,
    ):
        """No exit-state file + zero exit_code means natural
        completion; non-zero means crashed."""
        ws, result = _make_workspace(tmp_path)
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path)
        out = handle.wait()
        assert out.exit_reason == "completed"

    def test_missing_exit_state_nonzero_exit_is_crashed(
        self, tmp_path: Path,
    ):
        ws, result = _make_workspace(
            tmp_path, exit_code=1)
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path)
        out = handle.wait()
        assert out.exit_reason == "crashed"


class TestMalformedControlFiles:
    """Malformed control JSON collapses to ``None`` but is preserved.

    ``agent_runner`` normally writes well-formed JSON, but a
    truncated write (container killed mid-write, disk full, etc.)
    can leave a partial file.  The launcher's helpers swallow
    parse errors and return ``None`` / empty so the runner's
    generic "no handoff payload" error fires; the raw bytes are
    copied to the execution's log dir so an operator can
    post-mortem after the control tempdir is cleaned up.
    """

    def _preserve_dir(
        self, tmp_path: Path, execution_id: str = "exec-1",
    ) -> Path:
        return (
            tmp_path / "logs" / "play"
            / f"{execution_id}-control")

    def test_truncated_exit_state_becomes_none(
        self, tmp_path: Path,
    ):
        ws, result = _make_workspace(tmp_path)
        control_dir = Path(tempfile.mkdtemp(
            prefix="ctrl-", dir=tmp_path))
        (control_dir / "agent_exit_state.json").write_text(
            '{"status": "compl', encoding="utf-8")
        log_dir = tmp_path / "logs" / "play"
        inner = _FakeInner(result=result)
        handle = AgentHandle(
            inner=inner,
            workspace=ws,
            executions_before=0,
            prompt_tempdir=Path(tempfile.mkdtemp(
                prefix="p-", dir=tmp_path)),
            control_tempdir=control_dir,
            log_dir=log_dir,
        )
        out = handle.wait()
        assert out.exit_state is None
        # Falls back to exit_code-based classification.
        assert out.exit_reason == "completed"
        preserved = (
            self._preserve_dir(tmp_path)
            / "agent_exit_state.json.malformed")
        assert preserved.is_file()
        assert preserved.read_text(encoding="utf-8") == (
            '{"status": "compl')

    def test_garbage_pending_becomes_none(
        self, tmp_path: Path,
    ):
        ws, result = _make_workspace(tmp_path)
        control_dir = Path(tempfile.mkdtemp(
            prefix="ctrl-", dir=tmp_path))
        (control_dir / "pending_tool_calls.json").write_text(
            "not json at all", encoding="utf-8")
        log_dir = tmp_path / "logs" / "play"
        inner = _FakeInner(result=result)
        handle = AgentHandle(
            inner=inner,
            workspace=ws,
            executions_before=0,
            prompt_tempdir=Path(tempfile.mkdtemp(
                prefix="p-", dir=tmp_path)),
            control_tempdir=control_dir,
            log_dir=log_dir,
        )
        out = handle.wait()
        assert out.pending_tool_calls is None
        preserved = (
            self._preserve_dir(tmp_path)
            / "pending_tool_calls.json.malformed")
        assert preserved.is_file()
        assert preserved.read_text(encoding="utf-8") == (
            "not json at all")

    def test_wrong_top_level_type_preserved(
        self, tmp_path: Path,
    ):
        """A valid-JSON non-dict top level still preserves the file."""
        ws, result = _make_workspace(tmp_path)
        control_dir = Path(tempfile.mkdtemp(
            prefix="ctrl-", dir=tmp_path))
        (control_dir / "agent_exit_state.json").write_text(
            '["not", "a", "dict"]', encoding="utf-8")
        log_dir = tmp_path / "logs" / "play"
        inner = _FakeInner(result=result)
        handle = AgentHandle(
            inner=inner,
            workspace=ws,
            executions_before=0,
            prompt_tempdir=Path(tempfile.mkdtemp(
                prefix="p-", dir=tmp_path)),
            control_tempdir=control_dir,
            log_dir=log_dir,
        )
        handle.wait()
        preserved = (
            self._preserve_dir(tmp_path)
            / "agent_exit_state.json.malformed")
        assert preserved.is_file()


class TestEvalsRunCount:
    def test_evals_run_counts_additional_executions(
        self, tmp_path: Path,
    ):
        """``evals_run`` is the delta of workspace executions
        minus one for the agent's own record."""
        ws, result = _make_workspace(tmp_path)
        # Add two extra executions to the workspace (nested
        # blocks triggered during the agent run).
        for i in range(2):
            ws.executions[f"nested-{i}"] = BlockExecution(
                id=f"nested-{i}",
                block_name="game_step",
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
                status="succeeded",
                exit_code=0,
                elapsed_s=0.1,
            )
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path,
            executions_before=0,
        )
        out = handle.wait()
        # Total executions: 3 (agent + 2 nested).  Before: 0.
        # evals_run = 3 - 0 - 1 = 2.
        assert out.evals_run == 2

    def test_evals_run_zero_when_no_nested(self, tmp_path: Path):
        ws, result = _make_workspace(tmp_path)
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path,
            executions_before=0,
        )
        out = handle.wait()
        # Total 1 (agent), before 0.  evals_run = 1 - 0 - 1 = 0.
        assert out.evals_run == 0

    def test_evals_run_respects_executions_before(
        self, tmp_path: Path,
    ):
        """When ``executions_before`` is nonzero, only executions
        recorded *after* launch count as evals.  Pre-existing
        workspace history must not inflate the delta."""
        ws, result = _make_workspace(
            tmp_path, execution_id="exec-new")
        # Seed two pre-existing executions that predate the
        # agent launch.
        for i in range(2):
            ws.executions[f"pre-{i}"] = BlockExecution(
                id=f"pre-{i}",
                block_name="game_step",
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
                status="succeeded",
                exit_code=0,
                elapsed_s=0.1,
            )
        # Add one nested execution triggered during the run.
        ws.executions["nested-0"] = BlockExecution(
            id="nested-0",
            block_name="game_step",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            status="succeeded",
            exit_code=0,
            elapsed_s=0.1,
        )
        handle, _ = _make_handle(
            workspace=ws, result=result, tmp_path=tmp_path,
            executions_before=2,
        )
        out = handle.wait()
        # Total 4 (2 pre + agent + 1 nested), before 2.
        # evals_run = 4 - 2 - 1 = 1.
        assert out.evals_run == 1


class _RaisingInner:
    """Inner stand-in whose ``wait()`` raises mid-flight.

    Used to verify the adapter's cleanup path when the substrate
    wait fails rather than returning a result.
    """

    def __init__(self, *, exc: Exception) -> None:
        self._exc = exc
        self.stop_calls: list[str] = []

    def is_alive(self) -> bool:
        return True

    def stop(self, reason: str = "requested") -> None:
        self.stop_calls.append(reason)

    def wait(self):
        raise self._exc


class _KeyboardInterruptingInner:
    """Inner whose first ``wait()`` raises ``KeyboardInterrupt``.

    A second ``wait()`` returns a real ``ExecutionResult`` so
    ``run_agent_block`` can finish its shutdown after forwarding
    the stop.
    """

    def __init__(self, *, final: ExecutionResult) -> None:
        self._final = final
        self._called = 0
        self.stop_calls: list[str] = []

    def is_alive(self) -> bool:
        return self._called == 0

    def stop(self, reason: str = "requested") -> None:
        self.stop_calls.append(reason)

    def wait(self) -> ExecutionResult:
        self._called += 1
        if self._called == 1:
            raise KeyboardInterrupt
        return self._final


class TestWaitFailurePaths:
    """The adapter must clean up the prompt tempdir even when the
    inner wait raises, and must surface the original exception.
    """

    def test_prompt_tempdir_cleaned_when_inner_wait_raises(
        self, tmp_path: Path,
    ):
        workspace = MagicMock()
        workspace.path = tmp_path
        workspace.executions = {}
        workspace.artifacts = {}

        inner = _RaisingInner(exc=RuntimeError("boom"))
        prompt_dir = Path(tempfile.mkdtemp(
            prefix="flywheel-test-prompt-", dir=tmp_path))
        handle = AgentHandle(
            inner=inner,
            workspace=workspace,
            executions_before=0,
            prompt_tempdir=prompt_dir,
            control_tempdir=None,
        )
        assert prompt_dir.is_dir()
        with pytest.raises(RuntimeError, match="boom"):
            handle.wait()
        assert not prompt_dir.exists(), (
            "prompt tempdir must be cleaned even when inner.wait()"
            " raises"
        )


class TestPreContainerSyncHandle:
    """``launch_agent_block`` may return a pre-completed handle
    when the executor's pre-container pipeline fails.  The adapter
    must still route that through ``wait()`` without leaking."""

    def test_sync_handle_result_classified_as_crashed(
        self, tmp_path: Path,
    ):
        """A :class:`SyncExecutionHandle` returned from a pre-
        container failure reports a non-zero exit_code and no
        ``agent_exit_state`` artifact; the adapter maps that to
        ``exit_reason='crashed'``."""
        from flywheel.executor import (
            INVOKE_FAILURE_EXIT_CODE,
            SyncExecutionHandle,
        )

        workspace = MagicMock()
        workspace.path = tmp_path
        workspace.executions = {
            "exec-fail": BlockExecution(
                id="exec-fail",
                block_name="play",
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
                status="failed",
                exit_code=INVOKE_FAILURE_EXIT_CODE,
                elapsed_s=0.0,
                failure_phase="stage_in",
            ),
        }
        workspace.artifacts = {}
        workspace.generate_event_id = MagicMock(return_value="ev")

        sync = SyncExecutionHandle(ExecutionResult(
            exit_code=INVOKE_FAILURE_EXIT_CODE,
            elapsed_s=0.0,
            output_bindings={},
            execution_id="exec-fail",
            status="failed",
        ))
        prompt_dir = Path(tempfile.mkdtemp(
            prefix="flywheel-test-prompt-", dir=tmp_path))
        handle = AgentHandle(
            inner=sync,
            workspace=workspace,
            executions_before=0,
            prompt_tempdir=prompt_dir,
            control_tempdir=None,
        )
        out = handle.wait()
        assert out.exit_reason == "crashed"
        assert not prompt_dir.exists()


class TestRunAgentBlockKeyboardInterrupt:
    """``run_agent_block`` must forward a KeyboardInterrupt to the
    inner handle as a graceful stop, finalize the run, and re-
    raise the interrupt.
    """

    def test_keyboard_interrupt_triggers_stop_and_reraises(
        self, tmp_path: Path, monkeypatch,
    ):
        workspace, base_result = _make_workspace(
            tmp_path, execution_id="exec-interrupted")

        inner = _KeyboardInterruptingInner(final=base_result)

        prompt_dir = Path(tempfile.mkdtemp(
            prefix="flywheel-test-prompt-", dir=tmp_path))
        handle = AgentHandle(
            inner=inner,
            workspace=workspace,
            executions_before=0,
            prompt_tempdir=prompt_dir,
            control_tempdir=None,
        )

        def _fake_launch(**_):
            return handle

        # Patch launch_agent_block via the module the wrapper
        # imports it from.
        monkeypatch.setattr(
            "flywheel.agent.launch_agent_block", _fake_launch)

        from flywheel.agent import run_agent_block
        with pytest.raises(KeyboardInterrupt):
            run_agent_block(
                workspace=workspace,
                template=MagicMock(),
                project_root=tmp_path,
                prompt="hi",
                block_name="play",
            )
        assert inner.stop_calls == ["keyboard_interrupt"]
