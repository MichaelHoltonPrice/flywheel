from __future__ import annotations

from datetime import UTC, datetime

import pytest

from flywheel.artifact import (
    ArtifactInstance,
    BlockExecution,
    RejectedOutput,
    RejectionRef,
    SupersedesRef,
)


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
        assert ex.rejected_outputs == {}


class TestRejectionRef:
    """Stable ledger handle for a rejected output slot."""

    def test_fields(self):
        ref = RejectionRef(execution_id="exec_a3f", slot="checkpoint")
        assert ref.execution_id == "exec_a3f"
        assert ref.slot == "checkpoint"

    def test_frozen(self):
        ref = RejectionRef(execution_id="e", slot="s")
        with pytest.raises(AttributeError):
            ref.execution_id = "other"

    def test_equal_when_pair_matches(self):
        # Two refs to the same (exec, slot) pair compare equal so
        # downstream code can use them as dict keys / set members
        # for "is this the same rejected slot?".
        a = RejectionRef(execution_id="e1", slot="checkpoint")
        b = RejectionRef(execution_id="e1", slot="checkpoint")
        assert a == b
        assert hash(a) == hash(b)


class TestSupersedesRef:
    """One-of artifact_id / rejection; predecessor flavours
    captured at construction."""

    def test_artifact_id_flavour(self):
        ref = SupersedesRef(artifact_id="checkpoint@2026-01-abc")
        assert ref.artifact_id == "checkpoint@2026-01-abc"
        assert ref.rejection is None

    def test_rejection_flavour(self):
        rej = RejectionRef(execution_id="exec_a3f", slot="checkpoint")
        ref = SupersedesRef(rejection=rej)
        assert ref.rejection == rej
        assert ref.artifact_id is None

    def test_both_set_raises(self):
        # Setting both predecessors makes "what does this supersede?"
        # ambiguous and is a programmer error rather than something
        # the substrate should silently disambiguate.
        with pytest.raises(ValueError, match="exactly one"):
            SupersedesRef(
                artifact_id="x",
                rejection=RejectionRef(execution_id="e", slot="s"),
            )

    def test_neither_set_raises(self):
        # Empty SupersedesRef has no predecessor and no meaning.
        with pytest.raises(ValueError, match="exactly one"):
            SupersedesRef()

    def test_frozen(self):
        ref = SupersedesRef(artifact_id="x")
        with pytest.raises(AttributeError):
            ref.artifact_id = "other"


class TestRejectedOutput:
    """Per-slot record on BlockExecution.rejected_outputs."""

    def test_fields(self):
        rec = RejectedOutput(
            reason="model.pt missing",
            quarantine_path="quarantine/exec_a3f/checkpoint",
        )
        assert rec.reason == "model.pt missing"
        assert rec.quarantine_path == (
            "quarantine/exec_a3f/checkpoint")

    def test_quarantine_path_optional(self):
        # Quarantine I/O may fail; the validation reason is the
        # primary signal, so quarantine_path defaults to None.
        rec = RejectedOutput(reason="bad")
        assert rec.quarantine_path is None

    def test_frozen(self):
        rec = RejectedOutput(reason="r")
        with pytest.raises(AttributeError):
            rec.reason = "other"


class TestSupersedesFieldsOnArtifactInstance:
    """Smoke coverage for the ``supersedes`` /
    ``supersedes_reason`` fields on :class:`ArtifactInstance`."""

    def test_defaults_to_none(self):
        inst = ArtifactInstance(
            id="x@1", name="x", kind="copy",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        assert inst.supersedes is None
        assert inst.supersedes_reason is None

    def test_carries_supersedes_artifact(self):
        inst = ArtifactInstance(
            id="x@2", name="x", kind="copy",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            supersedes=SupersedesRef(artifact_id="x@1"),
            supersedes_reason="fix gradient explosion",
        )
        assert inst.supersedes == SupersedesRef(artifact_id="x@1")
        assert inst.supersedes_reason == "fix gradient explosion"

    def test_carries_supersedes_rejection(self):
        rej = RejectionRef(execution_id="exec_a3f", slot="x")
        inst = ArtifactInstance(
            id="x@2", name="x", kind="copy",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            supersedes=SupersedesRef(rejection=rej),
            supersedes_reason="corrected validator-rejected output",
        )
        assert inst.supersedes is not None
        assert inst.supersedes.rejection == rej
