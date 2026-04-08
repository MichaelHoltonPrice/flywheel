from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

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
    template_path = templates_dir / "test_template.yaml"
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


def _commit_all(project_root: Path, message: str = "auto") -> None:
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", message],
        check=True, capture_output=True,
    )


def _fake_run_with_output(output_files: dict[str, str]):
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
    def test_nonexistent_block_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        with pytest.raises(KeyError, match="nonexistent"):
            run_block(ws, "nonexistent", template, project_root)

    def test_valid_block_found(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            result = run_block(ws, "train", template, project_root)
        assert result.exit_code == 0


class TestInputResolution:
    def test_git_input_creates_new_instance(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        engine_instances = ws.instances_for("engine")
        assert len(engine_instances) == 2
        assert engine_instances[0].id == "engine@baseline"
        assert engine_instances[1].id.startswith("engine@")
        assert engine_instances[1].id != "engine@baseline"

    def test_git_baseline_preserved(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        baseline_commit = ws.artifacts["engine@baseline"].commit
        _commit_all(project_root, "add workspace")

        (project_root / "src" / "new.rs").write_text("// new")
        _commit_all(project_root, "new code")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        assert ws.artifacts["engine@baseline"].commit == baseline_commit
        new_engine = [
            a for a in ws.instances_for("engine")
            if a.id != "engine@baseline"
        ][0]
        assert new_engine.commit != baseline_commit

    def test_optional_input_skipped_when_missing(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
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
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        with pytest.raises(ValueError, match="not available"):
            run_block(ws, "eval", template, project_root)


class TestRepeatedExecution:
    def test_second_execution_produces_new_artifact(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)
            _commit_all(project_root, "after first train")
            run_block(ws, "train", template, project_root)

        cp_instances = [
            a for a in ws.artifacts.values()
            if a.name == "checkpoint" and a.kind == "copy"
        ]
        assert len(cp_instances) == 2
        assert cp_instances[0].id != cp_instances[1].id

    def test_second_execution_uses_first_output_as_input(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)
            _commit_all(project_root, "after first train")
            run_block(ws, "train", template, project_root)

        # Find the two executions
        execs = sorted(ws.executions.values(), key=lambda e: e.started_at)
        assert len(execs) == 2
        first_output = execs[0].output_bindings["checkpoint"]
        second_input = execs[1].input_bindings["checkpoint"]
        second_output = execs[1].output_bindings["checkpoint"]
        assert second_input == first_output
        assert second_output != first_output


class TestEmptyOutput:
    def test_empty_output_not_recorded_as_artifact(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        # Container succeeds but writes nothing
        def empty_run(config, args=None):
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=empty_run):
            run_block(ws, "train", template, project_root)

        # Execution should succeed but with no output bindings
        ex = next(iter(ws.executions.values()))
        assert ex.status == "succeeded"
        assert ex.output_bindings == {}
        assert len(ws.instances_for("checkpoint")) == 0


class TestExecutionRecords:
    def test_successful_execution_recorded(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        assert len(ws.executions) == 1
        ex = next(iter(ws.executions.values()))
        assert ex.block_name == "train"
        assert ex.status == "succeeded"
        assert "engine" in ex.input_bindings
        assert "checkpoint" in ex.output_bindings

    def test_failed_execution_recorded(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        with patch("flywheel.execution.run_container") as mock_run:
            mock_run.return_value = ContainerResult(exit_code=1, elapsed_s=5.0)
            with pytest.raises(RuntimeError, match="exited with code 1"):
                run_block(ws, "train", template, project_root)

        assert len(ws.executions) == 1
        ex = next(iter(ws.executions.values()))
        assert ex.status == "failed"

    def test_execution_persisted(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        reloaded = Workspace.load(ws.path)
        assert len(reloaded.executions) == 1
        cp_instances = reloaded.instances_for("checkpoint")
        assert len(cp_instances) == 1


class TestResourceConfig:
    def test_docker_args_passed_to_container(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
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
    def test_dirty_git_repo_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")
        (project_root / "src" / "dirty.rs").write_text("// dirty")

        with pytest.raises(RuntimeError, match="uncommitted changes"):
            run_block(ws, "train", template, project_root)

    def test_template_mismatch_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        other_path = foundry_dir / "templates" / "other.yaml"
        other_path.write_text(TEMPLATE_YAML)
        other_template = Template.from_yaml(other_path)

        with pytest.raises(ValueError, match="does not match"):
            run_block(ws, "train", other_template, project_root)

    def test_deleted_git_path_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        shutil.rmtree(project_root / "src")
        _commit_all(project_root, "remove src")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            run_block(ws, "train", template, project_root)

    def test_extra_args_passed_through(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
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
                args=["--subclass", "dueling"],
            )

        assert captured_args["args"] == ["--subclass", "dueling"]


class TestInputBindings:
    def test_explicit_binding_used(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        # Run train twice to get two checkpoints
        fake_run = _fake_run_with_output({"model.pt": "weights_v1"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)
            _commit_all(project_root, "after first train")

        first_cp = ws.instances_for("checkpoint")[0].id

        fake_run2 = _fake_run_with_output({"model.pt": "weights_v2"})
        with patch("flywheel.execution.run_container", side_effect=fake_run2):
            run_block(ws, "train", template, project_root)
            _commit_all(project_root, "after second train")

        # Run eval with explicit binding to the first checkpoint
        def capturing_run(config, args=None):
            for host, _container, mode in config.mounts:
                if mode == "rw":
                    (Path(host) / "scores.json").write_text("{}")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=capturing_run):
            run_block(
                ws, "eval", template, project_root,
                input_bindings={"checkpoint": first_cp},
            )

        eval_exec = [
            e for e in ws.executions.values() if e.block_name == "eval"
        ][0]
        assert eval_exec.input_bindings["checkpoint"] == first_cp

    def test_invalid_binding_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        with pytest.raises(ValueError, match="not found in workspace"):
            run_block(
                ws, "eval", template, project_root,
                input_bindings={"checkpoint": "checkpoint@nonexistent"},
            )


class TestFailureCleanup:
    def test_failed_execution_cleans_output_dirs(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        with patch("flywheel.execution.run_container") as mock_run:
            mock_run.return_value = ContainerResult(exit_code=1, elapsed_s=1.0)
            with pytest.raises(RuntimeError):
                run_block(ws, "train", template, project_root)

        # No checkpoint directories should remain
        checkpoint_dirs = [
            d for d in (ws.path / "artifacts").iterdir()
            if d.name.startswith("checkpoint@")
        ]
        assert len(checkpoint_dirs) == 0

    def test_failed_execution_still_records_git_inputs(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        with patch("flywheel.execution.run_container") as mock_run:
            mock_run.return_value = ContainerResult(exit_code=1, elapsed_s=1.0)
            with pytest.raises(RuntimeError):
                run_block(ws, "train", template, project_root)

        # Git instances are inputs — committed even on failure
        engine_instances = ws.instances_for("engine")
        assert len(engine_instances) == 2
        assert engine_instances[0].id == "engine@baseline"
        assert engine_instances[1].id.startswith("engine@")

    def test_wrong_slot_binding_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)
            _commit_all(project_root, "after train")

        # Try to bind engine@baseline to the checkpoint slot
        with pytest.raises(ValueError, match="belongs to"):
            run_block(
                ws, "eval", template, project_root,
                input_bindings={"checkpoint": "engine@baseline"},
            )

    def test_interrupted_execution_recorded(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        with patch("flywheel.execution.run_container") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()
            with pytest.raises(KeyboardInterrupt):
                run_block(ws, "train", template, project_root)

        assert len(ws.executions) == 1
        ex = next(iter(ws.executions.values()))
        assert ex.status == "interrupted"

        # No checkpoint directories should remain
        checkpoint_dirs = [
            d for d in (ws.path / "artifacts").iterdir()
            if d.name.startswith("checkpoint@")
        ]
        assert len(checkpoint_dirs) == 0

        # Git instances are inputs — committed before container runs
        engine_instances = ws.instances_for("engine")
        assert len(engine_instances) == 2
        engine_id = ex.input_bindings["engine"]
        assert engine_id in ws.artifacts
