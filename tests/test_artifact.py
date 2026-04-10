from __future__ import annotations

from datetime import UTC, datetime

import pytest

from flywheel.artifact import ArtifactInstance, BlockExecution


class TestArtifactInstance:
    def test_copy_instance(self):
        inst = ArtifactInstance(
            id="checkpoint@1",
            name="checkpoint",
            kind="copy",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            produced_by="exec1",
            copy_path="checkpoint@1",
        )
        assert inst.id == "checkpoint@1"
        assert inst.name == "checkpoint"
        assert inst.kind == "copy"
        assert inst.produced_by == "exec1"
        assert inst.copy_path == "checkpoint@1"

    def test_git_instance(self):
        inst = ArtifactInstance(
            id="engine@baseline",
            name="engine",
            kind="git",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            produced_by=None,
            repo="/repo",
            commit="abc123",
            git_path="crates/engine",
        )
        assert inst.id == "engine@baseline"
        assert inst.kind == "git"
        assert inst.produced_by is None
        assert inst.repo == "/repo"
        assert inst.commit == "abc123"
        assert inst.git_path == "crates/engine"

    def test_frozen(self):
        inst = ArtifactInstance(
            id="x@1", name="x", kind="copy",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(AttributeError):
            inst.id = "x@2"

    def test_baseline_has_no_producer(self):
        inst = ArtifactInstance(
            id="engine@baseline",
            name="engine",
            kind="git",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert inst.produced_by is None

    def test_defaults(self):
        inst = ArtifactInstance(
            id="x@1", name="x", kind="copy",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert inst.produced_by is None
        assert inst.copy_path is None
        assert inst.repo is None
        assert inst.commit is None
        assert inst.git_path is None


class TestBlockExecution:
    def test_succeeded_execution(self):
        ex = BlockExecution(
            id="exec1",
            block_name="train",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            finished_at=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
            status="succeeded",
            input_bindings={"engine": "engine@1"},
            output_bindings={"checkpoint": "checkpoint@1"},
            exit_code=0,
            elapsed_s=300.0,
            image="train:latest",
        )
        assert ex.status == "succeeded"
        assert ex.input_bindings["engine"] == "engine@1"
        assert ex.output_bindings["checkpoint"] == "checkpoint@1"

    def test_failed_execution(self):
        ex = BlockExecution(
            id="exec2",
            block_name="train",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            status="failed",
            exit_code=1,
        )
        assert ex.status == "failed"
        assert ex.output_bindings == {}

    def test_frozen(self):
        ex = BlockExecution(
            id="exec1", block_name="train",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(AttributeError):
            ex.id = "exec2"

    def test_defaults(self):
        ex = BlockExecution(
            id="exec1", block_name="train",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert ex.finished_at is None
        assert ex.status == "failed"
        assert ex.input_bindings == {}
        assert ex.output_bindings == {}
        assert ex.exit_code is None
        assert ex.elapsed_s is None
        assert ex.image is None
