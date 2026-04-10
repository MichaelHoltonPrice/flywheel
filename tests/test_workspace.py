from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flywheel.artifact import ArtifactInstance, BlockExecution
from flywheel.template import Template
from flywheel.validation import validate_name
from flywheel.workspace import Workspace
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
    template = Template.from_yaml(template_path)
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
        ws.add_artifact(inst)

        ex = BlockExecution(
            id="exec1", block_name="train", started_at=now,
            finished_at=now, status="succeeded",
            input_bindings={"engine": "engine@baseline"},
            output_bindings={"checkpoint": "checkpoint@1"},
            exit_code=0, elapsed_s=10.0, image="train:latest",
        )
        ws.add_execution(ex)
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
        ws.add_artifact(inst)
        assert "checkpoint@1" in ws.artifacts

    def test_duplicate_id_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ArtifactInstance(
            id="checkpoint@1", name="checkpoint", kind="copy",
            created_at=datetime.now(UTC), copy_path="checkpoint@1",
        )
        ws.add_artifact(inst)
        with pytest.raises(ValueError, match="already exists"):
            ws.add_artifact(inst)

    def test_undeclared_slot_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ArtifactInstance(
            id="unknown@1", name="unknown", kind="copy",
            created_at=datetime.now(UTC),
        )
        with pytest.raises(ValueError, match="not declared"):
            ws.add_artifact(inst)

    def test_kind_mismatch_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ArtifactInstance(
            id="checkpoint@1", name="checkpoint", kind="git",
            created_at=datetime.now(UTC),
        )
        with pytest.raises(ValueError, match="expects"):
            ws.add_artifact(inst)

    def test_copy_without_copy_path_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ArtifactInstance(
            id="checkpoint@1", name="checkpoint", kind="copy",
            created_at=datetime.now(UTC),
            copy_path=None,
        )
        with pytest.raises(ValueError, match="missing copy_path"):
            ws.add_artifact(inst)

    def test_git_without_required_fields_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        inst = ArtifactInstance(
            id="engine@1", name="engine", kind="git",
            created_at=datetime.now(UTC),
            repo="/repo", commit=None, git_path=None,
        )
        with pytest.raises(ValueError, match="missing required fields"):
            ws.add_artifact(inst)


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
        ws.add_artifact(ArtifactInstance(
            id="checkpoint@aaa", name="checkpoint", kind="copy",
            created_at=t1, copy_path="checkpoint@aaa",
        ))
        ws.add_artifact(ArtifactInstance(
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
        ws.add_artifact(ArtifactInstance(
            id="checkpoint@ccc", name="checkpoint", kind="copy",
            created_at=t3, copy_path="checkpoint@ccc",
        ))
        ws.add_artifact(ArtifactInstance(
            id="checkpoint@aaa", name="checkpoint", kind="copy",
            created_at=t1, copy_path="checkpoint@aaa",
        ))
        ws.add_artifact(ArtifactInstance(
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
        ws.add_artifact(ArtifactInstance(
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
        ws.add_execution(ex)
        assert "exec1" in ws.executions

    def test_duplicate_execution_raises(self, tmp_path: Path):
        _, foundry_dir, template = _setup_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        now = datetime.now(UTC)
        ex = BlockExecution(
            id="exec1", block_name="train", started_at=now,
        )
        ws.add_execution(ex)
        with pytest.raises(ValueError, match="already exists"):
            ws.add_execution(ex)


class TestNameValidation:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_name("", "Test")

    def test_slash_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            validate_name("a/b", "Test")

    def test_valid_name(self):
        validate_name("my-workspace_01", "Test")
