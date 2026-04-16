"""Tests for BlockGroup generic parallel execution."""

from __future__ import annotations

from unittest.mock import MagicMock

from flywheel.block_group import BlockGroup, BlockGroupMember
from flywheel.executor import ExecutionResult, SyncExecutionHandle


def _mock_workspace() -> MagicMock:
    ws = MagicMock()
    ws.executions = {}
    ws.events = {}
    ws.generate_event_id.return_value = "evt_bg"
    return ws


def _mock_executor(exit_code: int = 0) -> MagicMock:
    executor = MagicMock()

    def _make_handle(**kwargs):
        result = ExecutionResult(
            exit_code=exit_code,
            elapsed_s=5.0,
            output_bindings={},
            execution_id="exec_bg",
            status="succeeded" if exit_code == 0 else "failed",
        )
        return SyncExecutionHandle(result)

    executor.launch.side_effect = _make_handle
    return executor


class TestBlockGroupBasics:
    def test_empty_group(self):
        ws = _mock_workspace()
        group = BlockGroup(ws, _mock_executor())
        assert group.run() == []

    def test_single_member(self):
        ws = _mock_workspace()
        executor = _mock_executor()

        group = BlockGroup(ws, executor)
        group.add(BlockGroupMember(block_name="eval"))
        results = group.run()

        assert len(results) == 1
        assert results[0].index == 0
        assert results[0].block_name == "eval"
        assert results[0].result.exit_code == 0
        executor.launch.assert_called_once()

    def test_multiple_members(self):
        ws = _mock_workspace()
        executor = _mock_executor()

        group = BlockGroup(ws, executor)
        for i in range(3):
            group.add(BlockGroupMember(block_name=f"eval_{i}"))
        results = group.run()

        assert len(results) == 3
        assert executor.launch.call_count == 3

    def test_input_bindings_passed(self):
        ws = _mock_workspace()
        executor = _mock_executor()

        group = BlockGroup(ws, executor)
        group.add(BlockGroupMember(
            block_name="eval",
            input_bindings={"checkpoint": "ckpt@abc"},
        ))
        group.run()

        call_kwargs = executor.launch.call_args
        assert call_kwargs.kwargs["input_bindings"] == {
            "checkpoint": "ckpt@abc"}

    def test_extra_kwargs_passed(self):
        ws = _mock_workspace()
        executor = _mock_executor()

        group = BlockGroup(ws, executor)
        group.add(BlockGroupMember(
            block_name="eval",
            kwargs={"outputs_data": {"score": {"val": 10}}},
        ))
        group.run()

        call_kwargs = executor.launch.call_args
        assert call_kwargs.kwargs["outputs_data"] == {
            "score": {"val": 10}}


class TestBlockGroupLifecycleEvent:
    def test_records_completion_event(self):
        ws = _mock_workspace()
        group = BlockGroup(ws, _mock_executor())
        group.add(BlockGroupMember(block_name="eval"))
        group.run()

        ws.add_event.assert_called_once()
        event = ws.add_event.call_args[0][0]
        assert event.kind == "block_group_completed"
        assert event.detail["members"] == "1"
        assert event.detail["succeeded"] == "1"
        ws.save.assert_called()

    def test_counts_failures(self):
        ws = _mock_workspace()

        results_iter = iter([
            ExecutionResult(0, 1.0, {}, "e1", "succeeded"),
            ExecutionResult(1, 1.0, {}, "e2", "failed"),
        ])

        executor = MagicMock()
        executor.launch.side_effect = lambda **kw: SyncExecutionHandle(
            next(results_iter))

        group = BlockGroup(ws, executor)
        group.add(BlockGroupMember(block_name="a"))
        group.add(BlockGroupMember(block_name="b"))
        group.run()

        event = ws.add_event.call_args[0][0]
        assert event.detail["members"] == "2"
        assert event.detail["succeeded"] == "1"
