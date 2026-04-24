from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from flywheel.artifact import (
    ArtifactInstance,
    BlockExecution,
    LifecycleEvent,
    RejectedOutput,
    RejectionRef,
    SupersedesRef,
)
from flywheel.artifact_validator import (
    ArtifactValidationError,
    ArtifactValidatorRegistry,
)
from flywheel.run_record import RunMemberRecord, RunStepRecord
from flywheel.template import Template
from flywheel.validation import validate_name
from flywheel.workspace import Workspace
from tests._inline_blocks import from_yaml_with_inline_blocks
from tests.conftest import _init_git_repo

TEMPLATE_YAML = """\
artifacts:
  - name: engine
    kind: git
    repo: "."
    path: src
  - name: checkpoint
    kind: copy
  - name: score
    kind: copy

blocks:
  - name: train
    image: train:latest
    inputs: [engine]
    outputs: [checkpoint]
"""


def _setup_project(tmp_path: Path) -> tuple[Path, Path, Template]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _init_git_repo(project_root)
    (project_root / "src").mkdir()
    (project_root / "src" / "main.rs").write_text("// engine")
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", "add src"],
        check=True, capture_output=True,
    )
    foundry_dir = project_root / "foundry"
    foundry_dir.mkdir()
    templates_dir = foundry_dir / "templates"
    templates_dir.mkdir()
    template_path = templates_dir / "test.yaml"
    template_path.write_text(TEMPLATE_YAML)
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", "add foundry"],
        check=True, capture_output=True,
    )
    template = from_yaml_with_inline_blocks(template_path)
    return project_root, foundry_dir, template


class TestWorkspaceCreate:
    def test_creates_directory(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        assert ws.path.exists()
        assert (ws.path / "artifacts").exists()
        assert (ws.path / "workspace.yaml").exists()

    def test_name_and_template(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        assert ws.name == "test_ws"
        assert ws.template_name == "test"

    def test_artifact_declarations(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        assert ws.artifact_declarations == {
            "engine": "git", "checkpoint": "copy", "score": "copy",
        }

    def test_git_baseline_recorded(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        assert "engine@baseline" in ws.artifacts
        inst = ws.artifacts["engine@baseline"]
        assert inst.kind == "git"
        assert inst.name == "engine"
        assert inst.produced_by is None
        assert inst.commit is not None

    def test_copy_artifacts_not_created(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        copy_instances = [
            a for a in ws.artifacts.values() if a.kind == "copy"
        ]
        assert len(copy_instances) == 0

    def test_duplicate_name_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        Workspace.create("test_ws", template, foundry_dir)
        with pytest.raises(FileExistsError):
            Workspace.create("test_ws", template, foundry_dir)

    def test_invalid_name_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        with pytest.raises(ValueError, match="invalid"):
            Workspace.create("bad/name", template, foundry_dir)

    def test_dirty_repo_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_project(tmp_path)
        (project_root / "src" / "dirty.rs").write_text("// dirty")
        with pytest.raises(RuntimeError, match="uncommitted"):
            Workspace.create("test_ws", template, foundry_dir)

    def test_cleanup_on_failure(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_project(tmp_path)
        (project_root / "src" / "dirty.rs").write_text("// dirty")
        ws_path = foundry_dir / "workspaces" / "test_ws"
        with pytest.raises(RuntimeError):
            Workspace.create("test_ws", template, foundry_dir)
        assert not ws_path.exists()


class TestWorkspaceLoadSave:
    def test_round_trip(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        loaded = Workspace.load(ws.path)
        assert loaded.name == ws.name
        assert loaded.template_name == ws.template_name
        assert loaded.artifact_declarations == ws.artifact_declarations
        assert len(loaded.artifacts) == len(ws.artifacts)

    def test_round_trip_with_execution(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        now = datetime.now(UTC)
        inst = ArtifactInstance(
            id="checkpoint@1", name="checkpoint", kind="copy",
            created_at=now, produced_by="exec1", copy_path="checkpoint@1",
        )
        ws._add_artifact(inst)

        ex = BlockExecution(
            id="exec1", block_name="train", started_at=now,
            finished_at=now, status="succeeded",
            input_bindings={"engine": "engine@baseline"},
            output_bindings={"checkpoint": "checkpoint@1"},
            exit_code=0, elapsed_s=10.0, image="train:latest",
        )
        ws._add_execution(ex)
        ws.save()

        loaded = Workspace.load(ws.path)
        assert "checkpoint@1" in loaded.artifacts
        assert "exec1" in loaded.executions
        assert loaded.executions["exec1"].status == "succeeded"
        assert loaded.executions["exec1"].input_bindings == {
            "engine": "engine@baseline",
        }


class TestAddArtifact:
    def test_add_valid_instance(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ArtifactInstance(
            id="checkpoint@1", name="checkpoint", kind="copy",
            created_at=datetime.now(UTC), copy_path="checkpoint@1",
        )
        ws._add_artifact(inst)
        assert "checkpoint@1" in ws.artifacts

    def test_duplicate_id_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ArtifactInstance(
            id="checkpoint@1", name="checkpoint", kind="copy",
            created_at=datetime.now(UTC), copy_path="checkpoint@1",
        )
        ws._add_artifact(inst)
        with pytest.raises(ValueError, match="already exists"):
            ws._add_artifact(inst)

    def test_undeclared_slot_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ArtifactInstance(
            id="unknown@1", name="unknown", kind="copy",
            created_at=datetime.now(UTC),
        )
        with pytest.raises(ValueError, match="not declared"):
            ws._add_artifact(inst)

    def test_kind_mismatch_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ArtifactInstance(
            id="checkpoint@1", name="checkpoint", kind="git",
            created_at=datetime.now(UTC),
        )
        with pytest.raises(ValueError, match="expects"):
            ws._add_artifact(inst)

    def test_copy_without_copy_path_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ArtifactInstance(
            id="checkpoint@1", name="checkpoint", kind="copy",
            created_at=datetime.now(UTC),
            copy_path=None,
        )
        with pytest.raises(ValueError, match="missing copy_path"):
            ws._add_artifact(inst)

    def test_git_without_required_fields_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ArtifactInstance(
            id="engine@1", name="engine", kind="git",
            created_at=datetime.now(UTC),
            repo="/repo", commit=None, git_path=None,
        )
        with pytest.raises(ValueError, match="missing required fields"):
            ws._add_artifact(inst)


class TestGenerateArtifactId:
    def test_has_name_prefix(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        aid = ws.generate_artifact_id("checkpoint")
        assert aid.startswith("checkpoint@")

    def test_unique_ids(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        ids = {ws.generate_artifact_id("checkpoint") for _ in range(10)}
        assert len(ids) == 10


class TestGenerateExecutionId:
    def test_has_exec_prefix(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        eid = ws.generate_execution_id()
        assert eid.startswith("exec_")

    def test_unique_ids(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        ids = {ws.generate_execution_id() for _ in range(10)}
        assert len(ids) == 10


class TestInstancesFor:
    def test_returns_matching(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        t1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
        ws._add_artifact(ArtifactInstance(
            id="checkpoint@aaa", name="checkpoint", kind="copy",
            created_at=t1, copy_path="checkpoint@aaa",
        ))
        ws._add_artifact(ArtifactInstance(
            id="checkpoint@bbb", name="checkpoint", kind="copy",
            created_at=t2, copy_path="checkpoint@bbb",
        ))
        instances = ws.instances_for("checkpoint")
        assert len(instances) == 2
        assert instances[0].id == "checkpoint@aaa"
        assert instances[1].id == "checkpoint@bbb"

    def test_empty_for_no_instances(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        assert ws.instances_for("checkpoint") == []

    def test_sorted_by_creation_time(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        t1 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
        t3 = datetime(2026, 1, 1, 0, 2, tzinfo=UTC)
        # Add out of order
        ws._add_artifact(ArtifactInstance(
            id="checkpoint@ccc", name="checkpoint", kind="copy",
            created_at=t3, copy_path="checkpoint@ccc",
        ))
        ws._add_artifact(ArtifactInstance(
            id="checkpoint@aaa", name="checkpoint", kind="copy",
            created_at=t1, copy_path="checkpoint@aaa",
        ))
        ws._add_artifact(ArtifactInstance(
            id="checkpoint@bbb", name="checkpoint", kind="copy",
            created_at=t2, copy_path="checkpoint@bbb",
        ))
        instances = ws.instances_for("checkpoint")
        assert instances[0].created_at == t1
        assert instances[1].created_at == t2
        assert instances[2].created_at == t3

    def test_baseline_sorts_first(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        baseline = ws.instances_for("engine")
        assert len(baseline) == 1
        assert baseline[0].id == "engine@baseline"
        later = datetime(2099, 1, 1, tzinfo=UTC)
        ws._add_artifact(ArtifactInstance(
            id="engine@abc123", name="engine", kind="git",
            created_at=later, repo="/r", commit="def", git_path="src",
        ))
        instances = ws.instances_for("engine")
        assert instances[0].id == "engine@baseline"
        assert instances[1].id == "engine@abc123"


class TestGenerateExecutionIdWorkspace:
    def test_has_prefix(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        eid = ws.generate_execution_id()
        assert eid.startswith("exec_")

    def test_unique(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        ids = {ws.generate_execution_id() for _ in range(10)}
        assert len(ids) == 10


class TestAddExecution:
    def test_add_valid_execution(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        now = datetime.now(UTC)
        ex = BlockExecution(
            id="exec1", block_name="train", started_at=now,
        )
        ws._add_execution(ex)
        assert "exec1" in ws.executions

    def test_duplicate_execution_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        now = datetime.now(UTC)
        ex = BlockExecution(
            id="exec1", block_name="train", started_at=now,
        )
        ws._add_execution(ex)
        with pytest.raises(ValueError, match="already exists"):
            ws._add_execution(ex)


class TestRegisterArtifact:
    def test_file_source_rejected(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        bot_file = tmp_path / "bot.py"
        bot_file.write_text("def player_fn(): pass")

        with pytest.raises(ValueError, match="must be a directory"):
            ws.register_artifact("checkpoint", bot_file)

    def test_register_directory(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        src_dir = tmp_path / "bot_dir"
        src_dir.mkdir()
        (src_dir / "bot.py").write_text("def player_fn(): pass")
        (src_dir / "helpers.py").write_text("# helpers")

        inst = ws.register_artifact("checkpoint", src_dir)

        target = ws.path / "artifacts" / inst.id
        assert (target / "bot.py").exists()
        assert (target / "helpers.py").exists()

    def test_register_single_file_via_wrapper_directory(
        self, tmp_path: Path,
    ):
        # Canonical pattern for callers that have a single file:
        # wrap it in a directory and register that.  This makes
        # the artifact's structure explicit at the call site.
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        wrapper = tmp_path / "wrap"
        wrapper.mkdir()
        (wrapper / "bot.py").write_text("def player_fn(): pass")

        inst = ws.register_artifact("checkpoint", wrapper)

        target = ws.path / "artifacts" / inst.id / "bot.py"
        assert target.exists()
        assert target.read_text() == "def player_fn(): pass"

    def test_undeclared_name_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        f = tmp_path / "file.txt"
        f.write_text("data")

        with pytest.raises(ValueError, match="not declared"):
            ws.register_artifact("nonexistent", f)

    def test_git_artifact_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        f = tmp_path / "file.txt"
        f.write_text("data")

        with pytest.raises(ValueError, match="copy"):
            ws.register_artifact("engine", f)

    def test_nonexistent_source_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        with pytest.raises(FileNotFoundError):
            ws.register_artifact("checkpoint", tmp_path / "nope.txt")

    def test_artifact_instance_fields(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        src = tmp_path / "src"
        src.mkdir()
        (src / "data.bin").write_bytes(b"\x00")

        inst = ws.register_artifact(
            "checkpoint", src, source="test import",
        )

        assert inst.name == "checkpoint"
        assert inst.kind == "copy"
        assert inst.produced_by is None
        assert inst.source == "test import"
        assert inst.copy_path == inst.id
        assert inst.id in ws.artifacts

    def test_default_source(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        src = tmp_path / "src"
        src.mkdir()
        (src / "bot.py").write_text("pass")

        inst = ws.register_artifact("checkpoint", src)

        assert inst.source is not None
        assert "imported from" in inst.source

    def test_round_trip_with_source(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        src = tmp_path / "src"
        src.mkdir()
        (src / "bot.py").write_text("pass")

        inst = ws.register_artifact(
            "checkpoint", src, source="from agent",
        )
        loaded = Workspace.load(ws.path)

        assert inst.id in loaded.artifacts
        loaded_inst = loaded.artifacts[inst.id]
        assert loaded_inst.source == "from agent"

    def test_validator_passes_through(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        src = tmp_path / "src"
        src.mkdir()
        (src / "bot.py").write_text("pass")

        seen: list[Path] = []

        def validator(name, decl, path):
            seen.append(path)
            # The validator sees a flywheel-owned staging
            # directory, not the operator's source path, and
            # the directory contains the file shaped exactly
            # as it will appear in the committed artifact.
            assert path.is_dir()
            assert path != src
            assert (path / "bot.py").read_text() == "pass"

        registry = ArtifactValidatorRegistry(
            {"checkpoint": validator},
        )
        inst = ws.register_artifact(
            "checkpoint", src, validator_registry=registry,
        )
        assert inst.id in ws.artifacts
        assert len(seen) == 1
        # On success the staging directory is renamed into the
        # canonical artifact directory; the staging path the
        # validator saw is gone, and the bytes now live at
        # artifacts/<inst.id>/.
        assert not seen[0].exists()
        canonical = ws.path / "artifacts" / inst.id
        assert (canonical / "bot.py").read_text() == "pass"

    def test_validator_rejection_aborts_import(
        self, tmp_path: Path,
    ):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        src = tmp_path / "src"
        src.mkdir()
        (src / "bot.py").write_text("pass")

        def reject(name, decl, path):
            raise ArtifactValidationError("nope")

        registry = ArtifactValidatorRegistry({"checkpoint": reject})
        with pytest.raises(ArtifactValidationError, match="nope"):
            ws.register_artifact(
                "checkpoint", src, validator_registry=registry,
            )

        # Workspace state is untouched: no checkpoint
        # artifact, no canonical checkpoint dir, and no
        # leftover staging dir under artifacts/.  (The git
        # baseline created on workspace creation is still
        # present — it predates this call.)
        assert ws.instances_for("checkpoint") == []
        artifacts_dir = ws.path / "artifacts"
        leftover = [
            d for d in artifacts_dir.iterdir()
            if d.name.startswith("checkpoint@")
            or d.name.startswith("_staging-checkpoint-")
        ]
        assert leftover == []

        # The operator's source path is also untouched.
        assert (src / "bot.py").read_text() == "pass"

    def test_validator_skipped_when_no_registration(
        self, tmp_path: Path,
    ):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        src = tmp_path / "src"
        src.mkdir()
        (src / "bot.py").write_text("pass")

        # Registry exists but has no validator for this name —
        # import should still succeed.
        registry = ArtifactValidatorRegistry()
        inst = ws.register_artifact(
            "checkpoint", src, validator_registry=registry,
        )
        assert inst.id in ws.artifacts


class TestRegisterArtifactSupersedes:
    """``supersedes`` / ``supersedes_reason`` substrate on
    :meth:`Workspace.register_artifact`.

    Covers both predecessor flavours (accepted artifact id and
    rejected slot) and the predecessor-existence checks that
    fire before staging so a bad lineage reference fails cheaply.
    """

    @staticmethod
    def _src(tmp_path: Path, name: str = "src") -> Path:
        src = tmp_path / name
        src.mkdir()
        (src / "bot.py").write_text("pass")
        return src

    def test_supersede_accepted_artifact(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        first = ws.register_artifact(
            "checkpoint", self._src(tmp_path, "v1"),
        )
        second = ws.register_artifact(
            "checkpoint", self._src(tmp_path, "v2"),
            supersedes=SupersedesRef(artifact_id=first.id),
            supersedes_reason="hand-edited weights",
        )

        assert second.supersedes == SupersedesRef(artifact_id=first.id)
        assert second.supersedes_reason == "hand-edited weights"
        assert first.id in ws.artifacts
        assert second.id in ws.artifacts

    def test_supersede_rejected_slot(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        now = datetime.now(UTC)
        ws._add_execution(BlockExecution(
            id="exec_bad", block_name="train",
            started_at=now, finished_at=now,
            status="failed", failure_phase="output_validate",
            rejected_outputs={
                "checkpoint": RejectedOutput(
                    reason="schema mismatch",
                    quarantine_path=(
                        "quarantine/exec_bad/checkpoint"
                    ),
                ),
            },
        ))

        inst = ws.register_artifact(
            "checkpoint", self._src(tmp_path),
            supersedes=SupersedesRef(rejection=RejectionRef(
                execution_id="exec_bad", slot="checkpoint",
            )),
            supersedes_reason="re-derived after fixing schema",
        )

        assert inst.supersedes == SupersedesRef(
            rejection=RejectionRef(
                execution_id="exec_bad", slot="checkpoint",
            ),
        )
        assert (
            inst.supersedes_reason == "re-derived after fixing schema"
        )

    def test_reason_without_supersedes_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        with pytest.raises(
            ValueError, match="reason text is only meaningful",
        ):
            ws.register_artifact(
                "checkpoint", self._src(tmp_path),
                supersedes_reason="orphan reason",
            )

    def test_missing_artifact_id_predecessor_raises(
        self, tmp_path: Path,
    ):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        with pytest.raises(ValueError, match="does not exist"):
            ws.register_artifact(
                "checkpoint", self._src(tmp_path),
                supersedes=SupersedesRef(
                    artifact_id="checkpoint@deadbeef",
                ),
            )

    def test_name_mismatch_predecessor_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        # The 'engine' git baseline was created at workspace
        # creation time and lives under a different declaration
        # name than 'checkpoint'.
        engine_id = next(
            inst.id for inst in ws.artifacts.values()
            if inst.name == "engine"
        )

        with pytest.raises(
            ValueError, match="cannot supersede across artifact names",
        ):
            ws.register_artifact(
                "checkpoint", self._src(tmp_path),
                supersedes=SupersedesRef(artifact_id=engine_id),
            )

    def test_missing_execution_for_rejection_raises(
        self, tmp_path: Path,
    ):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        with pytest.raises(
            ValueError, match="execution_id 'exec_ghost'",
        ):
            ws.register_artifact(
                "checkpoint", self._src(tmp_path),
                supersedes=SupersedesRef(rejection=RejectionRef(
                    execution_id="exec_ghost", slot="checkpoint",
                )),
            )

    def test_slot_not_rejected_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        now = datetime.now(UTC)
        ws._add_execution(BlockExecution(
            id="exec_ok", block_name="train",
            started_at=now, finished_at=now,
            status="succeeded",
            rejected_outputs={},
        ))

        with pytest.raises(
            ValueError, match="no rejected output for slot",
        ):
            ws.register_artifact(
                "checkpoint", self._src(tmp_path),
                supersedes=SupersedesRef(rejection=RejectionRef(
                    execution_id="exec_ok", slot="checkpoint",
                )),
            )

    def test_predecessor_check_runs_before_staging(
        self, tmp_path: Path,
    ):
        # A bad lineage reference must fail before any bytes
        # are copied into the workspace's artifacts dir, so
        # bad calls leave no debris.
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        with pytest.raises(ValueError, match="does not exist"):
            ws.register_artifact(
                "checkpoint", self._src(tmp_path),
                supersedes=SupersedesRef(
                    artifact_id="checkpoint@nope",
                ),
            )

        artifacts_dir = ws.path / "artifacts"
        leftover = [
            d for d in artifacts_dir.iterdir()
            if d.name.startswith("checkpoint@")
            or d.name.startswith("_staging-checkpoint-")
        ]
        assert leftover == []


class TestNameValidation:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_name("", "Test")

    def test_slash_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            validate_name("a/b", "Test")

    def test_valid_name(self):
        validate_name("my-workspace_01", "Test")


class TestBlockExecutionFailurePhase:
    """Tests for the runtime-contract fields on BlockExecution.

    Stripped down after the block-execution schema purge: the
    legacy ``stop_reason`` / ``predecessor_id`` / ``state_dir`` /
    ``state_lineage_id`` round-trip cases were dropped along with
    their fields.  See ``flywheel/docs/specs/block-execution.md``.
    """

    def test_round_trip_with_failure_phase(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        now = datetime.now(UTC)
        ex = BlockExecution(
            id="exec1", block_name="train", started_at=now,
            finished_at=now, status="failed",
            failure_phase="output_validate",
            error="validator rejected slot",
        )
        ws._add_execution(ex)
        ws.save()

        loaded = Workspace.load(ws.path)
        assert (
            loaded.executions["exec1"].failure_phase
            == "output_validate"
        )
        assert loaded.executions["exec1"].status == "failed"

    def test_none_fields_not_serialized(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        now = datetime.now(UTC)
        ex = BlockExecution(
            id="exec1", block_name="train", started_at=now,
            status="succeeded",
        )
        ws._add_execution(ex)
        ws.save()

        with open(ws.path / "workspace.yaml") as f:
            raw = yaml.safe_load(f)
        exec_data = raw["executions"]["exec1"]
        assert "failure_phase" not in exec_data


class TestRoundTripSupersedesAndQuarantine:
    """YAML round-trip for the supersedes + quarantine schema
    additions on :class:`ArtifactInstance` and
    :class:`BlockExecution`."""

    def _make_instance(
        self, *, supersedes: SupersedesRef | None = None,
        supersedes_reason: str | None = None,
        inst_id: str = "checkpoint@2",
    ) -> ArtifactInstance:
        return ArtifactInstance(
            id=inst_id, name="checkpoint", kind="copy",
            created_at=datetime.now(UTC),
            copy_path=inst_id,
            supersedes=supersedes,
            supersedes_reason=supersedes_reason,
        )

    def test_round_trip_supersedes_artifact_id(
        self, tmp_path: Path,
    ):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        ws._add_artifact(self._make_instance(
            inst_id="checkpoint@1"))
        ws._add_artifact(self._make_instance(
            inst_id="checkpoint@2",
            supersedes=SupersedesRef(artifact_id="checkpoint@1"),
            supersedes_reason="fix gradient explosion",
        ))
        ws.save()

        loaded = Workspace.load(ws.path)
        successor = loaded.artifacts["checkpoint@2"]
        assert successor.supersedes == SupersedesRef(
            artifact_id="checkpoint@1")
        assert successor.supersedes_reason == "fix gradient explosion"

    def test_round_trip_supersedes_rejection(
        self, tmp_path: Path,
    ):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        rej = RejectionRef(execution_id="exec_a3f", slot="checkpoint")
        ws._add_artifact(self._make_instance(
            inst_id="checkpoint@2",
            supersedes=SupersedesRef(rejection=rej),
            supersedes_reason="corrected validator-rejected output",
        ))
        ws.save()

        loaded = Workspace.load(ws.path)
        successor = loaded.artifacts["checkpoint@2"]
        assert successor.supersedes is not None
        assert successor.supersedes.artifact_id is None
        assert successor.supersedes.rejection == rej
        assert successor.supersedes_reason == (
            "corrected validator-rejected output")

    def test_round_trip_rejected_outputs(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        now = datetime.now(UTC)
        ex = BlockExecution(
            id="exec_rej", block_name="train",
            started_at=now, finished_at=now,
            status="failed", failure_phase="output_validate",
            error="checkpoint: model.pt missing",
            rejected_outputs={
                "checkpoint": RejectedOutput(
                    reason="model.pt missing",
                    quarantine_path=(
                        "quarantine/exec_rej/checkpoint"),
                ),
                "score": RejectedOutput(
                    reason="score must be a float",
                    # quarantine_path omitted to verify
                    # round-trip when quarantine I/O failed.
                ),
            },
        )
        ws._add_execution(ex)
        ws.save()

        loaded = Workspace.load(ws.path)
        loaded_ex = loaded.executions["exec_rej"]
        assert set(loaded_ex.rejected_outputs) == {
            "checkpoint", "score"}
        ckpt = loaded_ex.rejected_outputs["checkpoint"]
        assert ckpt.reason == "model.pt missing"
        assert ckpt.quarantine_path == (
            "quarantine/exec_rej/checkpoint")
        score = loaded_ex.rejected_outputs["score"]
        assert score.reason == "score must be a float"
        assert score.quarantine_path is None

    def test_absent_fields_not_serialized(self, tmp_path: Path):
        # New fields should be absent from the on-disk YAML when
        # not set, rather than littering ``null`` placeholders
        # across every existing workspace.
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        ws._add_artifact(self._make_instance(inst_id="checkpoint@plain"))
        now = datetime.now(UTC)
        ws._add_execution(BlockExecution(
            id="exec_plain", block_name="train",
            started_at=now, status="succeeded",
        ))
        ws.save()

        with open(ws.path / "workspace.yaml") as f:
            raw = yaml.safe_load(f)
        artifact_yaml = raw["artifacts"]["checkpoint@plain"]
        assert "supersedes" not in artifact_yaml
        assert "supersedes_reason" not in artifact_yaml
        exec_yaml = raw["executions"]["exec_plain"]
        assert "rejected_outputs" not in exec_yaml

    def test_load_old_format_has_safe_defaults(
        self, tmp_path: Path,
    ):
        # An older workspace.yaml predating this schema must
        # still load: unknown fields default to None / empty.
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        ws._add_artifact(self._make_instance(
            inst_id="checkpoint@old"))
        now = datetime.now(UTC)
        ws._add_execution(BlockExecution(
            id="exec_old", block_name="train",
            started_at=now, status="failed",
        ))
        ws.save()

        loaded = Workspace.load(ws.path)
        inst = loaded.artifacts["checkpoint@old"]
        assert inst.supersedes is None
        assert inst.supersedes_reason is None
        ex = loaded.executions["exec_old"]
        assert ex.rejected_outputs == {}

    def test_supersedes_yaml_shape(self, tmp_path: Path):
        # Lock in the on-disk YAML shape so external tooling can
        # rely on it.  Both flavours use a discriminated map.
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        ws._add_artifact(self._make_instance(
            inst_id="a@id",
            supersedes=SupersedesRef(artifact_id="a@parent"),
        ))
        ws._add_artifact(self._make_instance(
            inst_id="a@rej",
            supersedes=SupersedesRef(rejection=RejectionRef(
                execution_id="exec_z", slot="checkpoint")),
        ))
        ws.save()

        with open(ws.path / "workspace.yaml") as f:
            raw = yaml.safe_load(f)
        assert raw["artifacts"]["a@id"]["supersedes"] == {
            "artifact": "a@parent"}
        assert raw["artifacts"]["a@rej"]["supersedes"] == {
            "rejected": {"execution": "exec_z",
                         "slot": "checkpoint"}}


class TestLifecycleEvents:
    """Tests for the LifecycleEvent entity on Workspace."""

    def test_add_and_retrieve_event(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        now = datetime.now(UTC)
        event = LifecycleEvent(
            id="evt_abc1", kind="agent_stopped",
            timestamp=now, execution_id="exec1",
            detail={"reason": "timeout"},
        )
        ws.add_event(event)
        assert "evt_abc1" in ws.events
        assert ws.events["evt_abc1"].kind == "agent_stopped"

    def test_duplicate_event_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        now = datetime.now(UTC)
        event = LifecycleEvent(
            id="evt_dup", kind="test", timestamp=now)
        ws.add_event(event)
        with pytest.raises(ValueError, match="already exists"):
            ws.add_event(event)

    def test_events_for_filters_by_kind(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        now = datetime.now(UTC)
        ws.add_event(LifecycleEvent(
            id="evt_1", kind="agent_stopped", timestamp=now))
        ws.add_event(LifecycleEvent(
            id="evt_2", kind="group_completed", timestamp=now))
        ws.add_event(LifecycleEvent(
            id="evt_3", kind="agent_stopped", timestamp=now))

        stopped = ws.events_for("agent_stopped")
        assert len(stopped) == 2
        assert all(e.kind == "agent_stopped" for e in stopped)

    def test_generate_event_id_unique(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        ids = {ws.generate_event_id() for _ in range(50)}
        assert len(ids) == 50
        assert all(eid.startswith("evt_") for eid in ids)

    def test_round_trip_with_events(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        now = datetime.now(UTC)
        ws.add_event(LifecycleEvent(
            id="evt_rt", kind="agent_stopped", timestamp=now,
            execution_id="exec_x",
            detail={"reason": "exploration_request"},
        ))
        ws.save()

        loaded = Workspace.load(ws.path)
        assert "evt_rt" in loaded.events
        evt = loaded.events["evt_rt"]
        assert evt.kind == "agent_stopped"
        assert evt.execution_id == "exec_x"
        assert evt.detail == {"reason": "exploration_request"}

    def test_load_without_events_key(self, tmp_path: Path):
        """Old workspace.yaml without events key loads correctly."""
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        ws.save()

        loaded = Workspace.load(ws.path)
        assert loaded.events == {}

    def test_empty_events_not_serialized(self, tmp_path: Path):
        """Events key omitted from YAML when empty."""
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        ws.save()

        with open(ws.path / "workspace.yaml") as f:
            raw = yaml.safe_load(f)
        assert "events" not in raw


class TestRuns:
    """Tests for :class:`flywheel.artifact.RunRecord` persistence."""

    def test_begin_and_end_run_round_trip(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        record = ws.begin_run(
            kind="pattern:play-bs",
            config_snapshot={"n": 3},
        )
        assert record.id.startswith("run_")
        assert record.status == "running"
        assert record.finished_at is None

        ws.end_run(record.id, status="succeeded")
        assert ws.runs[record.id].status == "succeeded"
        assert ws.runs[record.id].finished_at is not None

        ws.save()
        loaded = Workspace.load(ws.path)
        assert record.id in loaded.runs
        assert loaded.runs[record.id].kind == "pattern:play-bs"
        assert loaded.runs[record.id].config_snapshot == {"n": 3}
        assert loaded.runs[record.id].status == "succeeded"

    def test_record_run_step_round_trip(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        record = ws.begin_run(kind="pattern:train-eval")

        ws.record_run_step(
            record.id,
            RunStepRecord(
                name="train",
                min_successes="all",
                status="succeeded",
                members=[
                    RunMemberRecord(
                        name="train_dueling",
                        block_name="train",
                        status="succeeded",
                        execution_id="exec_1",
                        output_bindings={"checkpoint": "checkpoint@1"},
                    )
                ],
            ),
        )
        ws.end_run(record.id, status="succeeded")

        loaded = Workspace.load(ws.path)
        step = loaded.runs[record.id].steps[0]
        assert step.name == "train"
        assert step.min_successes == "all"
        assert step.members[0].execution_id == "exec_1"
        assert step.members[0].output_bindings == {
            "checkpoint": "checkpoint@1"
        }

    def test_load_without_runs_key(self, tmp_path: Path):
        """Old workspace.yaml without runs key loads to empty dict."""
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        ws.save()

        with open(ws.path / "workspace.yaml") as f:
            raw = yaml.safe_load(f)
        assert "runs" not in raw

        loaded = Workspace.load(ws.path)
        assert loaded.runs == {}

    def test_end_run_rejects_non_terminal_status(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        record = ws.begin_run(kind="pattern:x")

        with pytest.raises(ValueError):
            ws.end_run(record.id, status="running")

    def test_end_run_rejects_double_close(self, tmp_path: Path):
        """Second ``end_run`` on a terminal run raises.

        Silent overwrite would hide orchestration bugs where two
        layers race to close the same run.
        """
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        record = ws.begin_run(kind="pattern:x")
        ws.end_run(record.id, status="succeeded")

        with pytest.raises(ValueError, match="double-close"):
            ws.end_run(record.id, status="failed")



INCREMENTAL_TEMPLATE_YAML = """\
artifacts:
  - name: engine
    kind: git
    repo: "."
    path: src
  - name: history
    kind: incremental
  - name: notes
    kind: copy

blocks:
  - name: train
    image: train:latest
    inputs: [engine]
    outputs: [history]
"""


def _setup_incremental_project(
    tmp_path: Path,
) -> tuple[Path, Path, Template]:
    """Variant of _setup_project that declares an incremental artifact."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    _init_git_repo(project_root)
    (project_root / "src").mkdir()
    (project_root / "src" / "main.rs").write_text("// engine")
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", "add src"],
        check=True, capture_output=True,
    )
    foundry_dir = project_root / "foundry"
    foundry_dir.mkdir()
    templates_dir = foundry_dir / "templates"
    templates_dir.mkdir()
    template_path = templates_dir / "test.yaml"
    template_path.write_text(INCREMENTAL_TEMPLATE_YAML)
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", "add foundry"],
        check=True, capture_output=True,
    )
    template = from_yaml_with_inline_blocks(template_path)
    return project_root, foundry_dir, template


@pytest.mark.skip(
    reason=(
        "Incremental artifact kind has been excised; will be "
        "replaced by tagging system"
    )
)
class TestIncrementalArtifacts:
    """Tests for the incremental artifact kind."""

    def test_declaration_round_trip(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        assert ws.artifact_declarations["history"] == "incremental"
        loaded = Workspace.load(ws.path)
        assert loaded.artifact_declarations["history"] == "incremental"

    def test_no_baseline_at_create(self, tmp_path: Path):
        """Workspace.create does not auto-create incremental instances."""
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        incremental = [
            a for a in ws.artifacts.values() if a.kind == "incremental"
        ]
        assert incremental == []

    def test_register_incremental_artifact(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ws.register_incremental_artifact("history")
        assert inst.kind == "incremental"
        assert inst.name == "history"
        assert inst.copy_path == inst.id
        artifact_dir = ws.path / "artifacts" / inst.id
        assert artifact_dir.exists()
        entries_path = artifact_dir / "entries.jsonl"
        assert entries_path.exists()
        assert entries_path.read_text() == ""

    def test_register_incremental_with_provenance(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ws.register_incremental_artifact(
            "history",
            produced_by="exec_first",
            source="bootstrapped from initial frame",
        )
        assert inst.produced_by == "exec_first"
        assert inst.source == "bootstrapped from initial frame"

    def test_register_rejects_non_incremental(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        with pytest.raises(ValueError, match="incremental"):
            ws.register_incremental_artifact("notes")

    def test_register_rejects_undeclared(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        with pytest.raises(ValueError, match="not declared"):
            ws.register_incremental_artifact("unknown")

    def test_append_to_incremental(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ws.register_incremental_artifact("history")
        ws.append_to_incremental(inst.id, [{"step": 0, "action": "noop"}])
        ws.append_to_incremental(
            inst.id,
            [{"step": 1, "action": "left"}, {"step": 2, "action": "right"}],
        )
        entries = ws.read_incremental_entries(inst.id)
        assert entries == [
            {"step": 0, "action": "noop"},
            {"step": 1, "action": "left"},
            {"step": 2, "action": "right"},
        ]

    def test_append_empty_list_noop(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ws.register_incremental_artifact("history")
        ws.append_to_incremental(inst.id, [])
        assert ws.read_incremental_entries(inst.id) == []

    def test_append_to_unknown_id(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        with pytest.raises(KeyError):
            ws.append_to_incremental("history@nonexistent", [{"x": 1}])

    def test_append_rejects_copy_artifact(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        copy_inst = ArtifactInstance(
            id="notes@1", name="notes", kind="copy",
            created_at=datetime.now(UTC), copy_path="notes@1",
        )
        ws._add_artifact(copy_inst)
        with pytest.raises(ValueError, match="incremental"):
            ws.append_to_incremental("notes@1", [{"x": 1}])

    def test_round_trip_with_appends(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ws.register_incremental_artifact("history")
        ws.append_to_incremental(inst.id, [{"a": 1}, {"b": 2}])

        loaded = Workspace.load(ws.path)
        assert inst.id in loaded.artifacts
        loaded_inst = loaded.artifacts[inst.id]
        assert loaded_inst.kind == "incremental"
        assert loaded_inst.copy_path == inst.id
        assert loaded.read_incremental_entries(inst.id) == [
            {"a": 1}, {"b": 2},
        ]

    def test_latest_incremental_instance(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        assert ws.latest_incremental_instance("history") is None
        first = ws.register_incremental_artifact("history")
        latest = ws.latest_incremental_instance("history")
        assert latest is not None
        assert latest.id == first.id

    def test_instance_path_for_incremental(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ws.register_incremental_artifact("history")
        path = ws.instance_path(inst.id)
        assert path == ws.path / "artifacts" / inst.id
        assert (path / "entries.jsonl").exists()

    def test_instance_path_rejects_git(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        with pytest.raises(ValueError, match="git"):
            ws.instance_path("engine@baseline")

    def test_add_artifact_validates_incremental_copy_path(
        self, tmp_path: Path,
    ):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        bad = ArtifactInstance(
            id="history@bad", name="history", kind="incremental",
            created_at=datetime.now(UTC), copy_path=None,
        )
        with pytest.raises(ValueError, match="copy_path"):
            ws._add_artifact(bad)

    def test_template_rejects_unknown_kind(self, tmp_path: Path):
        bad_yaml = """\
artifacts:
  - name: history
    kind: bogus

blocks: []
"""
        path = tmp_path / "bad.yaml"
        path.write_text(bad_yaml)
        with pytest.raises(ValueError, match="unknown kind"):
            Template.from_yaml(path)

    def test_template_accepts_incremental_kind(self, tmp_path: Path):
        good_yaml = """\
artifacts:
  - name: history
    kind: incremental

blocks: []
"""
        path = tmp_path / "good.yaml"
        path.write_text(good_yaml)
        tpl = Template.from_yaml(path)
        assert tpl.artifacts[0].kind == "incremental"

    def test_entries_jsonl_format_is_one_per_line(self, tmp_path: Path):
        _, foundry_dir, template = _setup_incremental_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ws.register_incremental_artifact("history")
        ws.append_to_incremental(
            inst.id,
            [{"x": 1, "y": 2}, "scalar", [1, 2, 3]],
        )
        raw = (ws.path / "artifacts" / inst.id / "entries.jsonl").read_text()
        lines = [line for line in raw.split("\n") if line]
        assert len(lines) == 3
        assert json.loads(lines[0]) == {"x": 1, "y": 2}
        assert json.loads(lines[1]) == "scalar"
        assert json.loads(lines[2]) == [1, 2, 3]
