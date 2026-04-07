from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from flywheel.artifact import CopyArtifact, GitArtifact
from flywheel.container import ContainerResult
from flywheel.execution import run_block
from flywheel.template import Template
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
    docker_args: ["--gpus", "all", "--shm-size", "8g"]
    inputs:
      - name: engine
        container_path: /input/engine
      - name: checkpoint
        container_path: /input/checkpoint
        optional: true
    outputs:
      - name: checkpoint
        container_path: /output
  - name: eval
    image: eval:latest
    inputs:
      - name: checkpoint
        container_path: /input/checkpoint
    outputs:
      - name: score
        container_path: /output
"""


def _setup_git_project(tmp_path: Path) -> tuple[Path, Path, Template]:
    """Set up a project with a git repo, template, and workspace dirs."""
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

    workforce_dir = project_root / "workforce"
    workforce_dir.mkdir()
    templates_dir = workforce_dir / "templates"
    templates_dir.mkdir()
    template_path = templates_dir / "test_template.yaml"
    template_path.write_text(TEMPLATE_YAML)

    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", "add workforce"],
        check=True, capture_output=True,
    )

    template = Template.from_yaml(template_path)
    return project_root, workforce_dir, template


def _commit_all(project_root: Path, message: str = "auto") -> None:
    """Stage and commit all changes in the project git repo."""
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", message],
        check=True, capture_output=True,
    )


def _fake_run_with_output(output_files: dict[str, str]):
    """Create a fake run_container that writes files to output mounts."""
    def fake_run(config, args=None):
        for host, _container, mode in config.mounts:
            if mode == "rw":
                for name, content in output_files.items():
                    file_path = Path(host) / name
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content)
        return ContainerResult(exit_code=0, elapsed_s=1.0)
    return fake_run


class TestBlockLookup:
    def test_nonexistent_block_raises_key_error(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)

        with pytest.raises(KeyError, match="nonexistent"):
            run_block(ws, "nonexistent", template, project_root)

    def test_valid_block_found(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            result = run_block(ws, "train", template, project_root)

        assert result.exit_code == 0


class TestInputResolution:
    def test_git_input_mounts_repo_path(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        captured_config = {}

        def fake_run(config, args=None):
            captured_config["config"] = config
            for host, _container, mode in config.mounts:
                if mode == "rw":
                    (Path(host) / "model.pt").write_text("weights")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        config = captured_config["config"]
        input_mounts = [
            (h, c, m) for h, c, m in config.mounts if c == "/input/engine"
        ]
        assert len(input_mounts) == 1
        host_path, _, mode = input_mounts[0]
        assert mode == "ro"
        assert host_path.endswith("src")

    def test_optional_input_skipped_when_missing(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        captured_config = {}

        def fake_run(config, args=None):
            captured_config["config"] = config
            for host, _container, mode in config.mounts:
                if mode == "rw":
                    (Path(host) / "model.pt").write_text("weights")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        config = captured_config["config"]
        checkpoint_mounts = [
            (h, c, m) for h, c, m in config.mounts
            if c == "/input/checkpoint"
        ]
        assert len(checkpoint_mounts) == 0

    def test_required_input_raises_when_missing(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)

        with pytest.raises(ValueError, match="not available"):
            run_block(ws, "eval", template, project_root)

    def test_copy_input_mounts_when_available(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)

        cp_dir = ws.path / "artifacts" / "checkpoint"
        cp_dir.mkdir(parents=True)
        (cp_dir / "model.pt").write_text("weights")
        ws.record_artifact(
            "checkpoint",
            CopyArtifact(name="checkpoint", path=Path(".")),
        )
        ws.save()

        captured_config = {}

        def fake_run(config, args=None):
            captured_config["config"] = config
            for host, _container, mode in config.mounts:
                if mode == "rw":
                    (Path(host) / "scores.json").write_text("{}")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "eval", template, project_root)

        config = captured_config["config"]
        input_mounts = [
            (h, c, m) for h, c, m in config.mounts
            if c == "/input/checkpoint"
        ]
        assert len(input_mounts) == 1
        assert input_mounts[0][2] == "ro"


class TestConventionBasedOutput:
    def test_records_artifact_when_output_written(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        assert ws.artifacts["checkpoint"] is not None
        assert isinstance(ws.artifacts["checkpoint"], CopyArtifact)

    def test_output_file_exists_in_artifact_dir(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        artifact_file = ws.path / "artifacts" / "checkpoint" / "model.pt"
        assert artifact_file.exists()
        assert artifact_file.read_text() == "weights"

    def test_empty_output_dir_not_recorded(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        def fake_run(config, args=None):
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        assert ws.artifacts["checkpoint"] is None


class TestResourceConfig:
    def test_gpu_and_shm_passed_to_container(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        captured_config = {}

        def fake_run(config, args=None):
            captured_config["config"] = config
            for host, _container, mode in config.mounts:
                if mode == "rw":
                    (Path(host) / "model.pt").write_text("weights")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        config = captured_config["config"]
        assert "--gpus" in config.docker_args
        assert "--shm-size" in config.docker_args


class TestErrorCases:
    def test_nonzero_exit_raises_runtime_error(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        with patch("flywheel.execution.run_container") as mock_run:
            mock_run.return_value = ContainerResult(exit_code=1, elapsed_s=5.0)
            with pytest.raises(RuntimeError, match="exited with code 1"):
                run_block(ws, "train", template, project_root)

    def test_already_recorded_output_fails_fast(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        # Record checkpoint so it's already filled
        cp_dir = ws.path / "artifacts" / "checkpoint"
        cp_dir.mkdir(parents=True)
        (cp_dir / "model.pt").write_text("old weights")
        ws.record_artifact(
            "checkpoint",
            CopyArtifact(name="checkpoint", path=Path(".")),
        )

        # Should fail before launching container
        with patch("flywheel.execution.run_container") as mock_run:
            with pytest.raises(ValueError, match="already recorded"):
                run_block(ws, "train", template, project_root)
            mock_run.assert_not_called()

    def test_dirty_git_repo_raises(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        # Make the repo dirty
        (project_root / "src" / "dirty.rs").write_text("// dirty")

        with pytest.raises(RuntimeError, match="uncommitted changes"):
            run_block(ws, "train", template, project_root)


class TestWorkspaceSaved:
    def test_workspace_saved_after_block(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        reloaded = Workspace.load(ws.path)
        assert reloaded.artifacts["checkpoint"] is not None
        assert isinstance(reloaded.artifacts["checkpoint"], CopyArtifact)

    def test_extra_args_passed_through(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        captured_args = {}

        def fake_run(config, args=None):
            captured_args["args"] = args
            for host, _container, mode in config.mounts:
                if mode == "rw":
                    (Path(host) / "model.pt").write_text("weights")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(
                ws, "train", template, project_root,
                args=["--subclass", "dueling", "--combat-only"],
            )

        assert captured_args["args"] == [
            "--subclass", "dueling", "--combat-only",
        ]


class TestGitBaselinePreserved:
    def test_baseline_not_overwritten_by_execution(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        # Record the baseline commit from workspace creation
        baseline = ws.artifacts["engine"]
        assert isinstance(baseline, GitArtifact)
        baseline_commit = baseline.ref.commit

        # Make a new commit so HEAD differs from baseline
        (project_root / "src" / "new.rs").write_text("// new code")
        _commit_all(project_root, "add new code")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        # Baseline should be preserved in workspace
        after = ws.artifacts["engine"]
        assert isinstance(after, GitArtifact)
        assert after.ref.commit == baseline_commit


class TestTemplateMismatch:
    def test_wrong_template_raises(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)

        # Create a different template
        other_path = workforce_dir / "templates" / "other.yaml"
        other_path.write_text(TEMPLATE_YAML)
        other_template = Template.from_yaml(other_path)

        with pytest.raises(ValueError, match="does not match"):
            run_block(ws, "train", other_template, project_root)


class TestStaleOutputCleanup:
    def test_stale_files_removed_before_run(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        # Plant a stale file in the output directory
        stale_dir = ws.path / "artifacts" / "checkpoint"
        stale_dir.mkdir(parents=True)
        (stale_dir / "stale.pt").write_text("old garbage")
        _commit_all(project_root, "stale file")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        # Stale file should be gone, only new file present
        assert not (stale_dir / "stale.pt").exists()
        assert (stale_dir / "model.pt").exists()


class TestGitPathExistence:
    def test_deleted_git_path_raises(self, tmp_path: Path):
        project_root, workforce_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, workforce_dir)
        _commit_all(project_root, "add workspace")

        # Remove the src directory and commit
        shutil.rmtree(project_root / "src")
        _commit_all(project_root, "remove src")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            run_block(ws, "train", template, project_root)
