"""Tests for BlockGroup parallel execution.

Tests the launch-all-then-wait-sequentially pattern with various
launch functions, artifact collection, fallback, and env merging.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flywheel.block_group import BlockGroup, BlockGroupMember
from flywheel.executor import ExecutionResult, SyncExecutionHandle
from flywheel.template import ArtifactDeclaration, Template
from flywheel.workspace import Workspace


def _mock_workspace(tmp_path: Path | None = None) -> MagicMock:
    ws = MagicMock()
    ws.path = tmp_path or Path("/fake")
    ws.executions = {}
    ws.events = {}
    ws.generate_event_id.return_value = "evt_bg"

    def _register(name, path, source=None):
        # Mirror the real ``Workspace.register_artifact``
        # contract: artifact sources are always
        # directory-shaped, never single files.  Asserting
        # here means every existing BlockGroup test catches
        # any future regression where a caller hands a file in
        # without wrapping it first.
        assert isinstance(path, Path), (
            f"register_artifact path must be a Path, got "
            f"{type(path).__name__}"
        )
        assert path.is_dir(), (
            f"register_artifact requires a directory source; "
            f"got {path} (is_dir={path.is_dir()}, "
            f"is_file={path.is_file()})"
        )
        inst = MagicMock()
        inst.id = f"{name}@mock"
        return inst

    ws.register_artifact = MagicMock(side_effect=_register)
    return ws


def _mock_executor(exit_code: int = 0) -> MagicMock:
    """Create a mock BlockExecutor."""
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


def _mock_agent_handle(exit_code: int = 0, elapsed: float = 10.0):
    """Create a mock agent handle (for agent launch_fn tests)."""
    handle = MagicMock()
    result = MagicMock()
    result.exit_code = exit_code
    result.elapsed_s = elapsed
    handle.wait.return_value = result
    return handle


# ------------------------------------------------------------------
# Basic launch/wait
# ------------------------------------------------------------------


class TestBlockGroupBasics:
    def test_empty_group(self):
        ws = _mock_workspace()
        executor = _mock_executor()
        group = BlockGroup(ws, launch_fn=executor.launch)
        assert group.run() == []

    def test_single_member(self):
        ws = _mock_workspace()
        executor = _mock_executor()

        group = BlockGroup(ws, launch_fn=executor.launch)
        group.add(BlockGroupMember(
            overrides={"block_name": "eval"}))
        results = group.run()

        assert len(results) == 1
        assert results[0].index == 0
        assert results[0].result.exit_code == 0
        executor.launch.assert_called_once()

    def test_multiple_members(self):
        ws = _mock_workspace()
        executor = _mock_executor()

        group = BlockGroup(ws, launch_fn=executor.launch)
        for i in range(3):
            group.add(BlockGroupMember(
                overrides={"block_name": f"eval_{i}"}))
        results = group.run()

        assert len(results) == 3
        assert executor.launch.call_count == 3

    def test_overrides_passed_to_launch(self):
        ws = _mock_workspace()
        executor = _mock_executor()

        group = BlockGroup(ws, launch_fn=executor.launch)
        group.add(BlockGroupMember(overrides={
            "block_name": "eval",
            "input_bindings": {"checkpoint": "ckpt@abc"},
        }))
        group.run()

        call_kwargs = executor.launch.call_args.kwargs
        assert call_kwargs["input_bindings"] == {
            "checkpoint": "ckpt@abc"}


# ------------------------------------------------------------------
# base_kwargs and overrides
# ------------------------------------------------------------------


class TestBlockGroupConfig:
    def test_base_kwargs_passed_through(self):
        ws = _mock_workspace()
        executor = _mock_executor()

        group = BlockGroup(
            ws, launch_fn=executor.launch,
            base_kwargs={"block_name": "eval", "timeout": 30},
        )
        group.add(BlockGroupMember())
        group.run()

        kwargs = executor.launch.call_args.kwargs
        assert kwargs["block_name"] == "eval"
        assert kwargs["timeout"] == 30

    def test_member_overrides_base(self):
        ws = _mock_workspace()
        executor = _mock_executor()

        group = BlockGroup(
            ws, launch_fn=executor.launch,
            base_kwargs={"block_name": "eval", "timeout": 30},
        )
        group.add(BlockGroupMember(
            overrides={"timeout": 60}))
        group.run()

        kwargs = executor.launch.call_args.kwargs
        assert kwargs["timeout"] == 60
        assert kwargs["block_name"] == "eval"

    def test_merge_env(self):
        ws = _mock_workspace()
        executor = _mock_executor()

        group = BlockGroup(
            ws, launch_fn=executor.launch,
            base_kwargs={
                "extra_env": {"A": "1", "B": "2"},
            },
        )
        group.add(BlockGroupMember(
            merge_env={"B": "override", "C": "3"}))
        group.run()

        kwargs = executor.launch.call_args.kwargs
        assert kwargs["extra_env"] == {
            "A": "1", "B": "override", "C": "3"}

    def test_merge_env_without_base(self):
        ws = _mock_workspace()
        executor = _mock_executor()

        group = BlockGroup(ws, launch_fn=executor.launch)
        group.add(BlockGroupMember(merge_env={"X": "1"}))
        group.run()

        kwargs = executor.launch.call_args.kwargs
        assert kwargs["extra_env"] == {"X": "1"}


# ------------------------------------------------------------------
# Agent-style launch function
# ------------------------------------------------------------------


class TestBlockGroupWithAgentLaunch:
    def test_agent_launch_fn(self):
        ws = _mock_workspace()
        mock_launch = MagicMock(
            return_value=_mock_agent_handle())

        group = BlockGroup(
            ws, launch_fn=mock_launch,
            base_kwargs={
                "workspace": ws,
                "agent_image": "test:latest",
            },
        )
        group.add(BlockGroupMember(overrides={
            "prompt": "test prompt",
            "agent_workspace_dir": "ws_0",
        }, output_dir="ws_0"))
        results = group.run()

        assert len(results) == 1
        kwargs = mock_launch.call_args.kwargs
        assert kwargs["prompt"] == "test prompt"
        assert kwargs["agent_workspace_dir"] == "ws_0"

    def test_distinct_workspace_dirs(self):
        ws = _mock_workspace()
        mock_launch = MagicMock(
            return_value=_mock_agent_handle())

        group = BlockGroup(ws, launch_fn=mock_launch)
        group.add(BlockGroupMember(
            overrides={
                "prompt": "a",
                "agent_workspace_dir": "explore_0",
            },
            output_dir="explore_0",
        ))
        group.add(BlockGroupMember(
            overrides={
                "prompt": "b",
                "agent_workspace_dir": "explore_1",
            },
            output_dir="explore_1",
        ))
        group.run()

        calls = mock_launch.call_args_list
        dirs = [c.kwargs["agent_workspace_dir"] for c in calls]
        assert dirs == ["explore_0", "explore_1"]


# ------------------------------------------------------------------
# Artifact collection
# ------------------------------------------------------------------


class TestBlockGroupArtifactCollection:
    def test_collects_matching_files(self, tmp_path: Path):
        ws = _mock_workspace(tmp_path)
        mock_launch = MagicMock(
            return_value=_mock_agent_handle())

        agent_dir = tmp_path / "ws_0"
        agent_dir.mkdir()
        (agent_dir / "result.json").write_text('{"ok":true}')

        group = BlockGroup(ws, launch_fn=mock_launch)
        group.add(BlockGroupMember(
            overrides={"prompt": "test"},
            output_dir="ws_0",
        ))
        results = group.run(
            collect_artifacts=[("result", "exploration_result")])

        assert len(results[0].artifacts_collected) == 1
        ws.register_artifact.assert_called_once()
        assert ws.register_artifact.call_args[0][0] == (
            "exploration_result")

    def test_missing_file_skipped_without_fallback(
        self, tmp_path: Path,
    ):
        ws = _mock_workspace(tmp_path)
        mock_launch = MagicMock(
            return_value=_mock_agent_handle())

        agent_dir = tmp_path / "ws_0"
        agent_dir.mkdir()

        group = BlockGroup(ws, launch_fn=mock_launch)
        group.add(BlockGroupMember(
            overrides={"prompt": "test"},
            output_dir="ws_0",
        ))
        results = group.run(
            collect_artifacts=[("result", "result")])

        assert results[0].artifacts_collected == []
        ws.register_artifact.assert_not_called()

    def test_fallback_fn_called_on_missing_output(
        self, tmp_path: Path,
    ):
        ws = _mock_workspace(tmp_path)
        mock_launch = MagicMock(
            return_value=_mock_agent_handle())

        agent_dir = tmp_path / "ws_0"
        agent_dir.mkdir()

        def fallback(index, member):
            return {"fallback": True, "index": index}

        group = BlockGroup(
            ws, launch_fn=mock_launch,
            fallback_fn=fallback,
        )
        group.add(BlockGroupMember(
            overrides={"prompt": "test"},
            output_dir="ws_0",
        ))
        results = group.run(
            collect_artifacts=[("result", "result")])

        assert len(results[0].artifacts_collected) == 1
        fallback_file = agent_dir / "result.json"
        assert fallback_file.exists()
        data = json.loads(fallback_file.read_text())
        assert data["fallback"] is True

    def test_multiple_collect_artifacts(self, tmp_path: Path):
        ws = _mock_workspace(tmp_path)
        mock_launch = MagicMock(
            return_value=_mock_agent_handle())

        agent_dir = tmp_path / "ws_0"
        agent_dir.mkdir()
        (agent_dir / "result.json").write_text("{}")
        (agent_dir / "session.jsonl").write_text("{}")

        group = BlockGroup(ws, launch_fn=mock_launch)
        group.add(BlockGroupMember(
            overrides={"prompt": "test"},
            output_dir="ws_0",
        ))
        results = group.run(collect_artifacts=[
            ("result", "exploration_result"),
            ("session", "agent_session"),
        ])

        assert len(results[0].artifacts_collected) == 2
        assert ws.register_artifact.call_count == 2

    def test_no_collection_without_output_dir(self, tmp_path: Path):
        ws = _mock_workspace(tmp_path)
        mock_launch = MagicMock(
            return_value=_mock_agent_handle())

        group = BlockGroup(ws, launch_fn=mock_launch)
        group.add(BlockGroupMember(
            overrides={"prompt": "test"}))
        results = group.run(
            collect_artifacts=[("result", "result")])

        assert results[0].artifacts_collected == []


# ------------------------------------------------------------------
# Integration: real Workspace, real register_artifact
# ------------------------------------------------------------------


class TestBlockGroupArtifactCollectionIntegration:
    """End-to-end coverage of the artifact-collection path
    against a real :class:`Workspace`.

    The mocked ``register_artifact`` in the unit tests above
    asserts the ``is_dir()`` contract directly.  These tests go
    one step further: they let
    :meth:`BlockGroup._collect` call into the real
    :meth:`Workspace.register_artifact`, which enforces
    directory-only inputs and stages the bytes into the
    workspace.  If the wrap-into-tempdir step in
    ``_collect`` ever regresses to passing the file directly,
    the real ``register_artifact`` will reject it with the
    ``must be a directory`` error and these tests will fail.
    """

    @staticmethod
    def _build_workspace(tmp_path: Path) -> Workspace:
        # Minimal copy-only template; no git artifact, so we
        # don't need to initialize a git repo.
        foundry_dir = tmp_path / "foundry"
        foundry_dir.mkdir()
        template = Template(
            name="bg_int",
            artifacts=[
                ArtifactDeclaration(
                    name="exploration_result", kind="copy",
                ),
            ],
            blocks=[],
        )
        return Workspace.create(
            "ws_int", template, foundry_dir,
        )

    def test_collected_file_lands_under_artifact_id(
        self, tmp_path: Path,
    ):
        ws = self._build_workspace(tmp_path)
        mock_launch = MagicMock(
            return_value=_mock_agent_handle())

        agent_dir = ws.path / "ws_0"
        agent_dir.mkdir()
        (agent_dir / "result.json").write_text(
            '{"ok": true}')

        group = BlockGroup(ws, launch_fn=mock_launch)
        group.add(BlockGroupMember(
            overrides={"prompt": "test"},
            output_dir="ws_0",
        ))
        results = group.run(
            collect_artifacts=[
                ("result", "exploration_result"),
            ],
        )

        # One artifact id was returned and the bytes landed at
        # the canonical location with the original filename.
        ids = results[0].artifacts_collected
        assert len(ids) == 1
        artifact_path = ws.path / "artifacts" / ids[0]
        assert artifact_path.is_dir()
        assert (artifact_path / "result.json").read_text() == (
            '{"ok": true}')
        # No staging directories should be left behind under
        # artifacts/ once collection is done.
        leftover = [
            p for p in (ws.path / "artifacts").iterdir()
            if p.name.startswith("_staging-")
        ]
        assert leftover == []

    def test_undeclared_artifact_name_is_rejected_end_to_end(
        self, tmp_path: Path,
    ):
        # Catches the broader contract: ``_collect`` happily
        # passes whatever ``artifact_name`` the caller asked
        # for through to ``register_artifact``, and
        # ``register_artifact`` is the gatekeeper for "is this
        # name declared in the workspace's template?".  A
        # ``BlockGroup`` collecting under an undeclared name
        # should surface the rejection rather than silently
        # succeeding.
        ws = self._build_workspace(tmp_path)
        mock_launch = MagicMock(
            return_value=_mock_agent_handle())

        agent_dir = ws.path / "ws_0"
        agent_dir.mkdir()
        (agent_dir / "result.json").write_text("{}")

        group = BlockGroup(ws, launch_fn=mock_launch)
        group.add(BlockGroupMember(
            overrides={"prompt": "test"},
            output_dir="ws_0",
        ))
        with pytest.raises(ValueError, match="not declared"):
            group.run(collect_artifacts=[
                ("result", "not_in_template"),
            ])


# ------------------------------------------------------------------
# Lifecycle events
# ------------------------------------------------------------------


class TestBlockGroupLifecycleEvent:
    def test_records_completion_event(self):
        ws = _mock_workspace()
        executor = _mock_executor()
        group = BlockGroup(ws, launch_fn=executor.launch)
        group.add(BlockGroupMember(
            overrides={"block_name": "eval"}))
        group.run()

        ws.add_event.assert_called_once()
        event = ws.add_event.call_args[0][0]
        assert event.kind == "group_completed"
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
        executor.launch.side_effect = lambda **kw: (
            SyncExecutionHandle(next(results_iter)))

        group = BlockGroup(ws, launch_fn=executor.launch)
        group.add(BlockGroupMember(
            overrides={"block_name": "a"}))
        group.add(BlockGroupMember(
            overrides={"block_name": "b"}))
        group.run()

        event = ws.add_event.call_args[0][0]
        assert event.detail["members"] == "2"
        assert event.detail["succeeded"] == "1"
