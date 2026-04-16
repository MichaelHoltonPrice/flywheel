"""Tests for executor protocol and core abstractions."""

from __future__ import annotations

import pytest

from flywheel.executor import (
    RECORD_SENTINEL,
    ExecutionEvent,
    ExecutionHandle,
    ExecutionResult,
    SyncExecutionHandle,
)


class TestExecutionResult:
    def test_construction(self):
        r = ExecutionResult(
            exit_code=0,
            elapsed_s=1.5,
            output_bindings={"score": "score@abc"},
            execution_id="exec_001",
            status="succeeded",
        )
        assert r.exit_code == 0
        assert r.elapsed_s == 1.5
        assert r.output_bindings == {"score": "score@abc"}
        assert r.execution_id == "exec_001"
        assert r.status == "succeeded"

    def test_frozen(self):
        r = ExecutionResult(
            exit_code=0, elapsed_s=0, output_bindings={},
            execution_id="e", status="succeeded",
        )
        with pytest.raises(AttributeError):
            r.exit_code = 1  # type: ignore[misc]


class TestExecutionEvent:
    def test_construction(self):
        e = ExecutionEvent(
            executor_type="record",
            block_name="game_step",
            execution_id="exec_001",
            status="succeeded",
            output_bindings={"game_step": "game_step@abc"},
            outputs_data={"game_step": {"score": 10}},
        )
        assert e.executor_type == "record"
        assert e.block_name == "game_step"
        assert e.outputs_data == {"game_step": {"score": 10}}

    def test_defaults(self):
        e = ExecutionEvent(
            executor_type="container",
            block_name="eval",
            execution_id="exec_002",
            status="failed",
        )
        assert e.output_bindings == {}
        assert e.outputs_data is None


class TestSyncExecutionHandle:
    def test_wait_returns_result(self):
        result = ExecutionResult(
            exit_code=0, elapsed_s=0.1, output_bindings={},
            execution_id="e", status="succeeded",
        )
        handle = SyncExecutionHandle(result)
        assert handle.wait() is result

    def test_not_alive(self):
        result = ExecutionResult(
            exit_code=0, elapsed_s=0, output_bindings={},
            execution_id="e", status="succeeded",
        )
        handle = SyncExecutionHandle(result)
        assert handle.is_alive() is False

    def test_stop_is_noop(self):
        result = ExecutionResult(
            exit_code=0, elapsed_s=0, output_bindings={},
            execution_id="e", status="succeeded",
        )
        handle = SyncExecutionHandle(result)
        handle.stop()  # Should not raise.

    def test_double_wait_raises(self):
        result = ExecutionResult(
            exit_code=0, elapsed_s=0, output_bindings={},
            execution_id="e", status="succeeded",
        )
        handle = SyncExecutionHandle(result)
        handle.wait()
        with pytest.raises(RuntimeError, match="already called"):
            handle.wait()


class TestExecutionHandleBase:
    def test_not_implemented(self):
        handle = ExecutionHandle()
        with pytest.raises(NotImplementedError):
            handle.is_alive()
        with pytest.raises(NotImplementedError):
            handle.stop()
        with pytest.raises(NotImplementedError):
            handle.wait()


class TestRecordSentinel:
    def test_value(self):
        assert RECORD_SENTINEL == "__record__"
