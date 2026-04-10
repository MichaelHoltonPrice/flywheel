from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from flywheel.cli import _parse_bindings, create_workspace, main
from flywheel.template import Template
from flywheel.workspace import Workspace
from tests.conftest import _init_git_repo


def make_project(tmp_path: Path) -> Path:
    """Create a full project layout with flywheel.yaml, template, and git repo.

    Returns the project root path.
    """
    project_root = tmp_path / "project"
    project_root.mkdir()

    # Make it a git repo
    _init_git_repo(project_root)

    # Create the engine source dir referenced by the template
    (project_root / "crates" / "engine").mkdir(parents=True)
    (project_root / "crates" / "engine" / "lib.rs").write_text("// engine")
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", "add engine"],
        check=True,
        capture_output=True,
    )

    # Create flywheel.yaml
    flywheel_yaml = project_root / "flywheel.yaml"
    flywheel_yaml.write_text("foundry_dir: foundry\n")

    # Create template directory and file
    templates_dir = project_root / "foundry" / "templates"
    templates_dir.mkdir(parents=True)

    template_yaml = templates_dir / "my_template.yaml"
    template_yaml.write_text("""\
artifacts:
  - name: game_engine
    kind: git
    repo: "."
    path: crates/engine
  - name: checkpoint
    kind: copy
  - name: score
    kind: copy

blocks:
  - name: train
    image: cyberloop-train:latest
    inputs: [checkpoint]
    outputs: [checkpoint]
  - name: eval
    image: cyberloop-eval:latest
    inputs: [checkpoint]
    outputs: [score]
""")

    # Commit the project config files so the tree is clean
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", "add project config"],
        check=True,
        capture_output=True,
    )

    return project_root


def make_copy_only_project(tmp_path: Path) -> Path:
    """Create a project with only copy artifacts (no git artifact)."""
    project_root = tmp_path / "project"
    project_root.mkdir()

    flywheel_yaml = project_root / "flywheel.yaml"
    flywheel_yaml.write_text("foundry_dir: foundry\n")

    templates_dir = project_root / "foundry" / "templates"
    templates_dir.mkdir(parents=True)

    template_yaml = templates_dir / "simple.yaml"
    template_yaml.write_text("""\
artifacts:
  - name: data
    kind: copy

blocks: []
""")

    return project_root


class TestMainCreateWorkspace:
    def test_creates_workspace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)

        main(["create", "workspace", "--name", "test_ws", "--template", "my_template"])

        ws_path = project_root / "foundry" / "workspaces" / "test_ws"
        assert ws_path.is_dir()

    def test_creates_artifacts_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)

        main(["create", "workspace", "--name", "test_ws", "--template", "my_template"])

        ws_path = project_root / "foundry" / "workspaces" / "test_ws"
        assert (ws_path / "artifacts").is_dir()

    def test_creates_workspace_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)

        main(["create", "workspace", "--name", "test_ws", "--template", "my_template"])

        ws_path = project_root / "foundry" / "workspaces" / "test_ws"
        assert (ws_path / "workspace.yaml").is_file()


class TestMissingFlywheelYaml:
    def test_raises_on_missing_flywheel_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Project dir with no flywheel.yaml
        project_root = tmp_path / "empty_project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)

        with pytest.raises(FileNotFoundError):
            create_workspace("test_ws", "my_template")


class TestMissingTemplate:
    def test_raises_on_missing_template(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create flywheel.yaml but no template
        flywheel_yaml = project_root / "flywheel.yaml"
        flywheel_yaml.write_text("foundry_dir: foundry\n")
        (project_root / "foundry" / "templates").mkdir(parents=True)

        monkeypatch.chdir(project_root)

        with pytest.raises(FileNotFoundError):
            create_workspace("test_ws", "nonexistent_template")


class TestCreateWorkspaceFunction:
    def test_creates_via_function(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)

        create_workspace("my_ws", "my_template")

        ws_path = project_root / "foundry" / "workspaces" / "my_ws"
        assert ws_path.is_dir()
        assert (ws_path / "workspace.yaml").is_file()

    def test_copy_only_template(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        project_root = make_copy_only_project(tmp_path)
        monkeypatch.chdir(project_root)

        create_workspace("my_ws", "simple")

        ws_path = project_root / "foundry" / "workspaces" / "my_ws"
        assert ws_path.is_dir()


class TestMainErrorPaths:
    def test_no_args_exits(self):
        with pytest.raises(SystemExit):
            main([])

    def test_invalid_command_exits(self):
        with pytest.raises(SystemExit):
            main(["list"])

    def test_create_without_subcommand_exits(self):
        with pytest.raises(SystemExit):
            main(["create"])

    def test_run_without_subcommand_exits(self):
        with pytest.raises(SystemExit):
            main(["run"])


class TestParseBindings:
    def test_single_binding(self):
        result = _parse_bindings(["checkpoint=checkpoint@abc123"])
        assert result == {"checkpoint": "checkpoint@abc123"}

    def test_multiple_bindings(self):
        result = _parse_bindings([
            "checkpoint=checkpoint@abc",
            "engine=engine@baseline",
        ])
        assert result == {
            "checkpoint": "checkpoint@abc",
            "engine": "engine@baseline",
        }

    def test_empty_list(self):
        result = _parse_bindings([])
        assert result == {}

    def test_malformed_raises(self):
        with pytest.raises(ValueError, match="Invalid --bind format"):
            _parse_bindings(["no_equals_sign"])

    def test_value_with_at_sign(self):
        result = _parse_bindings(["checkpoint=checkpoint@abc123"])
        assert result["checkpoint"] == "checkpoint@abc123"


class TestMainImportArtifact:
    def test_import_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        project_root = make_copy_only_project(tmp_path)
        monkeypatch.chdir(project_root)

        # Create workspace (copy-only template, no git needed)
        template = Template.from_yaml(
            project_root / "foundry" / "templates" / "simple.yaml")
        foundry_dir = project_root / "foundry"
        Workspace.create("test_ws", template, foundry_dir)

        # Write a file to import
        src = tmp_path / "myfile.txt"
        src.write_text("hello")

        ws_path = str(project_root / "foundry" / "workspaces" / "test_ws")
        main(["import", "artifact",
              "--workspace", ws_path,
              "--name", "data",
              "--from", str(src)])

        # Verify the artifact was registered
        loaded = Workspace.load(
            project_root / "foundry" / "workspaces" / "test_ws")
        data_instances = loaded.instances_for("data")
        assert len(data_instances) == 1
        assert data_instances[0].source is not None
        assert "imported from" in data_instances[0].source

    def test_import_with_custom_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        project_root = make_copy_only_project(tmp_path)
        monkeypatch.chdir(project_root)

        template = Template.from_yaml(
            project_root / "foundry" / "templates" / "simple.yaml")
        foundry_dir = project_root / "foundry"
        Workspace.create("test_ws", template, foundry_dir)

        src = tmp_path / "myfile.txt"
        src.write_text("hello")

        ws_path = str(project_root / "foundry" / "workspaces" / "test_ws")
        main(["import", "artifact",
              "--workspace", ws_path,
              "--name", "data",
              "--from", str(src),
              "--source", "written by agent"])

        loaded = Workspace.load(
            project_root / "foundry" / "workspaces" / "test_ws")
        data_instances = loaded.instances_for("data")
        assert len(data_instances) == 1
        assert data_instances[0].source == "written by agent"


class TestMainRunBlock:
    def test_argument_parsing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify that main() parses run block args and calls the right path."""
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)

        # Create a workspace first
        main(["create", "workspace", "--name", "test_ws",
              "--template", "my_template"])

        # run block should fail at container launch (Docker not available
        # in tests), but it should get past argument parsing
        with pytest.raises((RuntimeError, FileNotFoundError, OSError, ValueError)):
            main([
                "run", "block",
                "--workspace",
                str(project_root / "foundry" / "workspaces" / "test_ws"),
                "--block", "train",
                "--template", "my_template",
                "--", "--subclass", "dueling",
            ])
