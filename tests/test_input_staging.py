"""Unit tests for :mod:`flywheel.input_staging`.

The invariant is that canonical artifacts are never exposed
to a downstream consumer.  These tests cover the staging
primitive and its batch wrapper directly so a regression in
either the copy semantics or the cleanup contract surfaces
here, before it reaches an agent or local recorder
integration test.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flywheel.artifact import ArtifactInstance
from flywheel.input_staging import (
    StagingError,
    cleanup_staged_inputs,
    stage_artifact_instance,
    stage_artifact_instances,
)
from flywheel.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Build a workspace with declared copy/incremental/git artifacts."""
    ws_path = tmp_path / "ws"
    ws_path.mkdir()
    (ws_path / "artifacts").mkdir()
    ws = Workspace(
        name="stg",
        path=ws_path,
        template_name="stg",
        created_at=datetime.now(UTC),
        artifact_declarations={
            "predictor": "copy",
            "history": "incremental",
            "corpus": "git",
        },
        artifacts={},
    )
    return ws


def _register_copy(ws: Workspace, name: str,
                   files: dict[str, str]) -> ArtifactInstance:
    aid = ws.generate_artifact_id(name)
    target = ws.path / "artifacts" / aid
    target.mkdir(parents=True)
    for fname, content in files.items():
        (target / fname).write_text(content)
    inst = ArtifactInstance(
        id=aid, name=name, kind="copy",
        created_at=datetime.now(UTC),
        produced_by=None, copy_path=aid,
    )
    ws.add_artifact(inst)
    return inst


class TestStageArtifactInstance:
    """Single-instance staging shape and isolation guarantees."""

    def test_copy_artifact_contents_appear_in_tempdir(
            self, workspace):
        inst = _register_copy(
            workspace, "predictor",
            {"predictor.json": json.dumps({"v": 1})},
        )
        staged = stage_artifact_instance(workspace, inst)
        try:
            assert staged.exists()
            assert staged.is_dir()
            assert (staged / "predictor.json").is_file()
            assert json.loads(
                (staged / "predictor.json").read_text()) == {"v": 1}
        finally:
            cleanup_staged_inputs({"slot": staged})

    def test_staging_dir_is_outside_workspace(
            self, workspace):
        inst = _register_copy(
            workspace, "predictor", {"a.txt": "hi"})
        staged = stage_artifact_instance(workspace, inst)
        try:
            assert workspace.path not in staged.parents
        finally:
            cleanup_staged_inputs({"slot": staged})

    def test_writes_to_staging_do_not_mutate_canonical(
            self, workspace):
        inst = _register_copy(
            workspace, "predictor", {"a.txt": "hi"})
        canonical = workspace.path / "artifacts" / inst.copy_path
        staged = stage_artifact_instance(workspace, inst)
        try:
            (staged / "a.txt").write_text("tampered")
            (staged / "new.txt").write_text("added")
            assert (canonical / "a.txt").read_text() == "hi"
            assert not (canonical / "new.txt").exists()
        finally:
            cleanup_staged_inputs({"slot": staged})

    def test_each_call_returns_a_fresh_directory(
            self, workspace):
        inst = _register_copy(
            workspace, "predictor", {"a.txt": "hi"})
        first = stage_artifact_instance(workspace, inst)
        second = stage_artifact_instance(workspace, inst)
        try:
            assert first != second
            assert first.exists()
            assert second.exists()
        finally:
            cleanup_staged_inputs({"a": first, "b": second})

    def test_incremental_artifact_entries_file_is_copied(
            self, workspace):
        inst = workspace.register_incremental_artifact("history")
        workspace.append_to_incremental(
            inst.id, [{"step": 0}, {"step": 1}])
        staged = stage_artifact_instance(workspace, inst)
        try:
            entries = staged / "entries.jsonl"
            assert entries.is_file()
            lines = entries.read_text().splitlines()
            assert [json.loads(line) for line in lines] == [
                {"step": 0}, {"step": 1}]
        finally:
            cleanup_staged_inputs({"slot": staged})

    def test_subdirectories_are_copied_recursively(
            self, workspace):
        aid = workspace.generate_artifact_id("predictor")
        target = workspace.path / "artifacts" / aid
        target.mkdir(parents=True)
        nested = target / "subdir"
        nested.mkdir()
        (nested / "leaf.txt").write_text("deep")
        inst = ArtifactInstance(
            id=aid, name="predictor", kind="copy",
            created_at=datetime.now(UTC),
            produced_by=None, copy_path=aid,
        )
        workspace.add_artifact(inst)

        staged = stage_artifact_instance(workspace, inst)
        try:
            assert (staged / "subdir" / "leaf.txt").read_text() == "deep"
        finally:
            cleanup_staged_inputs({"slot": staged})

    def test_git_artifact_raises(self, workspace):
        inst = ArtifactInstance(
            id="corpus@baseline", name="corpus", kind="git",
            created_at=datetime.now(UTC), produced_by=None,
            repo="/dev/null", commit="abc", git_path=".",
        )
        workspace.artifacts[inst.id] = inst
        with pytest.raises(StagingError) as excinfo:
            stage_artifact_instance(workspace, inst)
        assert excinfo.value.instance_id == inst.id

    def test_missing_canonical_directory_raises(
            self, workspace):
        inst = _register_copy(
            workspace, "predictor", {"a.txt": "hi"})
        canonical = workspace.path / "artifacts" / inst.copy_path
        for child in canonical.iterdir():
            child.unlink()
        canonical.rmdir()
        with pytest.raises(StagingError) as excinfo:
            stage_artifact_instance(workspace, inst)
        assert excinfo.value.instance_id == inst.id


class TestStageArtifactInstances:
    """Batch wrapper composes per-slot staging correctly."""

    def test_returns_one_path_per_slot(self, workspace):
        a = _register_copy(workspace, "predictor", {"a.txt": "1"})
        b = workspace.register_incremental_artifact("history")
        workspace.append_to_incremental(b.id, [{"step": 0}])
        staged = stage_artifact_instances(
            workspace, {"predictor": a.id, "history": b.id})
        try:
            assert set(staged) == {"predictor", "history"}
            assert (staged["predictor"] / "a.txt").read_text() == "1"
            assert (
                staged["history"] / "entries.jsonl"
            ).read_text().strip() == json.dumps(
                {"step": 0}, separators=(",", ":"))
        finally:
            cleanup_staged_inputs(staged)

    def test_unknown_artifact_id_is_skipped(self, workspace):
        staged = stage_artifact_instances(
            workspace, {"missing": "nope@absent"})
        assert staged == {}

    def test_git_artifact_is_skipped_silently(self, workspace):
        inst = ArtifactInstance(
            id="corpus@baseline", name="corpus", kind="git",
            created_at=datetime.now(UTC), produced_by=None,
            repo="/dev/null", commit="abc", git_path=".",
        )
        workspace.artifacts[inst.id] = inst
        staged = stage_artifact_instances(
            workspace, {"corpus": inst.id})
        assert staged == {}

    def test_partial_failure_cleans_up_already_staged(
            self, workspace, monkeypatch):
        good = _register_copy(workspace, "predictor", {"a.txt": "1"})
        bad = _register_copy(workspace, "predictor", {"b.txt": "2"})

        original = stage_artifact_instance
        seen_dirs: list[Path] = []

        def _staging_spy(ws, inst, **kwargs):
            if inst.id == bad.id:
                raise StagingError(
                    "boom", instance_id=inst.id)
            path = original(ws, inst, **kwargs)
            seen_dirs.append(path)
            return path

        monkeypatch.setattr(
            "flywheel.input_staging.stage_artifact_instance",
            _staging_spy,
        )

        with pytest.raises(StagingError):
            stage_artifact_instances(
                workspace, {"good": good.id, "bad": bad.id})

        for path in seen_dirs:
            assert not path.exists()


class TestCleanupStagedInputs:
    """Cleanup contract is best-effort and idempotent."""

    def test_removes_each_path(self, workspace):
        inst = _register_copy(workspace, "predictor", {"a.txt": "1"})
        staged = stage_artifact_instances(
            workspace, {"predictor": inst.id})
        path = staged["predictor"]
        cleanup_staged_inputs(staged)
        assert not path.exists()

    def test_missing_path_does_not_raise(self, workspace, tmp_path):
        ghost = tmp_path / "no-such-dir"
        cleanup_staged_inputs({"ghost": ghost})
