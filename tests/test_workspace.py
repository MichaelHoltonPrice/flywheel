from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from flywheel.artifact import CopyArtifact, GitArtifact, GitRef
from flywheel.template import Template
from flywheel.workspace import Workspace
from tests.conftest import _init_git_repo


def write_template_yaml(path: Path, *, git_repo: str = ".") -> Path:
    """Write a template YAML with one git artifact and two copy artifacts."""
    content = f"""\
artifacts:
  - name: engine
    kind: git
    repo: "{git_repo}"
    path: src
  - name: checkpoint
    kind: copy
  - name: score
    kind: copy

blocks:
  - name: train
    image: train:latest
    inputs: [checkpoint]
    outputs: [checkpoint]
  - name: eval
    image: eval:latest
    inputs: [checkpoint]
    outputs: [score]
"""
    path.write_text(content)
    return path


def write_copy_only_template(path: Path) -> Path:
    """Write a template with only copy artifacts (no git needed)."""
    content = """\
artifacts:
  - name: data
    kind: copy
  - name: output
    kind: copy

blocks:
  - name: process
    image: proc:latest
    inputs: [data]
    outputs: [output]
"""
    path.write_text(content)
    return path


@pytest.fixture()
def project_with_git(tmp_path: Path) -> Path:
    """Set up a project directory that is itself a git repo with a template."""
    project_root = tmp_path / "project"
    project_root.mkdir()

    # Make the project root a git repo
    _init_git_repo(project_root)

    # Create a src/ subdirectory so the path in the template exists
    (project_root / "src").mkdir()
    (project_root / "src" / "main.py").write_text("# engine")
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", "add src"],
        check=True,
        capture_output=True,
    )

    # Create workforce structure
    workforce_dir = project_root / "workforce"
    workforce_dir.mkdir()
    templates_dir = workforce_dir / "templates"
    templates_dir.mkdir()
    write_template_yaml(templates_dir / "my_template.yaml")

    # Commit everything so the tree is clean
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", "add workforce"],
        check=True,
        capture_output=True,
    )

    return project_root


@pytest.fixture()
def project_copy_only(tmp_path: Path) -> Path:
    """Set up a project directory with only copy artifacts (no git needed)."""
    project_root = tmp_path / "project"
    project_root.mkdir()

    workforce_dir = project_root / "workforce"
    workforce_dir.mkdir()
    templates_dir = workforce_dir / "templates"
    templates_dir.mkdir()
    write_copy_only_template(templates_dir / "simple.yaml")

    return project_root


class TestWorkspaceCreate:
    def test_creates_workspace_directory(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        assert ws.path.exists()
        assert ws.path == workforce_dir / "workspaces" / "test_ws"

    def test_creates_artifacts_subdir(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        assert (ws.path / "artifacts").is_dir()

    def test_writes_workspace_yaml(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        assert (ws.path / "workspace.yaml").is_file()

    def test_raises_if_already_exists(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        Workspace.create("test_ws", template, workforce_dir)
        with pytest.raises(FileExistsError, match="already exists"):
            Workspace.create("test_ws", template, workforce_dir)

    def test_workspace_name(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        assert ws.name == "test_ws"

    def test_template_name(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        assert ws.template_name == "my_template"

    def test_has_created_at(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        assert ws.created_at is not None


class TestCopyArtifactsStartNone:
    def test_copy_artifacts_are_none(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        assert ws.artifacts["checkpoint"] is None
        assert ws.artifacts["score"] is None

    def test_copy_only_all_none(self, project_copy_only: Path):
        workforce_dir = project_copy_only / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "simple.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        assert ws.artifacts["data"] is None
        assert ws.artifacts["output"] is None


class TestGitArtifactResolution:
    def test_git_artifact_resolved(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        engine = ws.artifacts["engine"]
        assert isinstance(engine, GitArtifact)

    def test_git_artifact_has_commit_sha(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        engine = ws.artifacts["engine"]
        assert isinstance(engine, GitArtifact)
        # SHA is a 40-char hex string
        assert len(engine.ref.commit) == 40
        assert all(c in "0123456789abcdef" for c in engine.ref.commit)

    def test_git_artifact_has_repo_path(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        engine = ws.artifacts["engine"]
        assert isinstance(engine, GitArtifact)
        assert engine.ref.repo == str(project_with_git.resolve())

    def test_git_artifact_has_path(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)
        engine = ws.artifacts["engine"]
        assert isinstance(engine, GitArtifact)
        assert engine.ref.path == "src"


class TestGitDirtyTreeRefused:
    def test_dirty_tree_raises(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")

        # Make the repo dirty by adding an untracked file
        (project_with_git / "dirty.txt").write_text("uncommitted")
        subprocess.run(
            ["git", "-C", str(project_with_git), "add", "dirty.txt"],
            check=True,
            capture_output=True,
        )

        with pytest.raises(RuntimeError, match="uncommitted changes"):
            Workspace.create("test_ws", template, workforce_dir)

    def test_untracked_file_makes_dirty(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")

        # An untracked file also shows in --porcelain
        (project_with_git / "untracked.txt").write_text("untracked")

        with pytest.raises(RuntimeError, match="uncommitted changes"):
            Workspace.create("test_ws", template, workforce_dir)

    def test_clean_after_commit_succeeds(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")

        # Add and commit a new file -- tree is clean again
        (project_with_git / "extra.txt").write_text("committed")
        subprocess.run(
            ["git", "-C", str(project_with_git), "add", "."],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(project_with_git), "commit", "-m", "add extra"],
            check=True,
            capture_output=True,
        )

        ws = Workspace.create("test_ws", template, workforce_dir)
        assert isinstance(ws.artifacts["engine"], GitArtifact)


class TestSaveLoadRoundTrip:
    def test_round_trip_fields(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        original = Workspace.create("test_ws", template, workforce_dir)

        loaded = Workspace.load(original.path)

        assert loaded.name == original.name
        assert loaded.template_name == original.template_name
        assert loaded.created_at == original.created_at

    def test_round_trip_git_artifact(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        original = Workspace.create("test_ws", template, workforce_dir)

        loaded = Workspace.load(original.path)
        engine_orig = original.artifacts["engine"]
        engine_loaded = loaded.artifacts["engine"]

        assert isinstance(engine_orig, GitArtifact)
        assert isinstance(engine_loaded, GitArtifact)
        assert engine_loaded.ref.repo == engine_orig.ref.repo
        assert engine_loaded.ref.commit == engine_orig.ref.commit
        assert engine_loaded.ref.path == engine_orig.ref.path

    def test_round_trip_none_artifacts(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        original = Workspace.create("test_ws", template, workforce_dir)

        loaded = Workspace.load(original.path)
        assert loaded.artifacts["checkpoint"] is None
        assert loaded.artifacts["score"] is None

    def test_round_trip_artifact_keys(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        original = Workspace.create("test_ws", template, workforce_dir)

        loaded = Workspace.load(original.path)
        assert set(loaded.artifacts.keys()) == set(original.artifacts.keys())

    def test_loaded_path(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        original = Workspace.create("test_ws", template, workforce_dir)

        loaded = Workspace.load(original.path)
        assert loaded.path == original.path

    def test_record_and_reload(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)

        ws.record_artifact(
            "checkpoint", CopyArtifact(name="checkpoint", path=Path("checkpoints/v1"))
        )
        ws.save()

        loaded = Workspace.load(ws.path)
        cp = loaded.artifacts["checkpoint"]
        assert isinstance(cp, CopyArtifact)
        assert cp.path == Path("checkpoints/v1")


class TestRecordArtifact:
    def test_record_undeclared_raises(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)

        with pytest.raises(KeyError, match="not declared"):
            ws.record_artifact(
                "nonexistent", CopyArtifact(name="nonexistent", path=Path("x"))
            )

    def test_record_already_recorded_raises(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)

        ws.record_artifact(
            "checkpoint", CopyArtifact(name="checkpoint", path=Path("v1"))
        )
        with pytest.raises(ValueError, match="immutable"):
            ws.record_artifact(
                "checkpoint", CopyArtifact(name="checkpoint", path=Path("v2"))
            )

    def test_record_sets_artifact(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)

        artifact = CopyArtifact(name="checkpoint", path=Path("checkpoints/v1"))
        ws.record_artifact("checkpoint", artifact)
        assert ws.artifacts["checkpoint"] is artifact

    def test_record_mismatched_name_raises(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)

        with pytest.raises(ValueError, match="does not match slot"):
            ws.record_artifact(
                "checkpoint", CopyArtifact(name="wrong_name", path=Path("x"))
            )

    def test_record_wrong_kind_raises(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")
        ws = Workspace.create("test_ws", template, workforce_dir)

        git_artifact = GitArtifact(
            name="checkpoint",
            ref=GitRef(repo="/fake", commit="a" * 40, path="src"),
        )
        with pytest.raises(TypeError, match="declared as 'copy'.*received 'git'"):
            ws.record_artifact("checkpoint", git_artifact)


class TestFailedCreateCleanup:
    def test_dirty_tree_leaves_no_directory(self, project_with_git: Path):
        workforce_dir = project_with_git / "workforce"
        template = Template.from_yaml(workforce_dir / "templates" / "my_template.yaml")

        # Make the repo dirty
        (project_with_git / "dirty.txt").write_text("uncommitted")

        ws_path = workforce_dir / "workspaces" / "test_ws"
        with pytest.raises(RuntimeError):
            Workspace.create("test_ws", template, workforce_dir)

        assert not ws_path.exists()


class TestGitArtifactPathValidation:
    def test_nonexistent_path_raises(self, tmp_path: Path):
        """Git artifact referencing a path that doesn't exist in the repo."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        _init_git_repo(project_root)

        workforce_dir = project_root / "workforce"
        workforce_dir.mkdir()
        templates_dir = workforce_dir / "templates"
        templates_dir.mkdir()

        # Template references "nonexistent/" which doesn't exist
        template_yaml = templates_dir / "bad_path.yaml"
        template_yaml.write_text("""\
artifacts:
  - name: engine
    kind: git
    repo: "."
    path: nonexistent
blocks: []
""")
        # Commit so tree is clean
        subprocess.run(
            ["git", "-C", str(project_root), "add", "."],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(project_root), "commit", "-m", "add workforce"],
            check=True, capture_output=True,
        )

        template = Template.from_yaml(template_yaml)
        with pytest.raises(FileNotFoundError, match="does not exist in repo"):
            Workspace.create("test_ws", template, workforce_dir)

        # Should also clean up
        assert not (workforce_dir / "workspaces" / "test_ws").exists()
