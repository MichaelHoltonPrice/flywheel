from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flywheel.cli import _parse_bindings, create_workspace, main
from flywheel.workspace import Workspace
from tests._inline_blocks import from_yaml_with_inline_blocks
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
  - train
  - eval
""")

    blocks_dir = project_root / "workforce" / "blocks"
    blocks_dir.mkdir(parents=True)
    (blocks_dir / "train.yaml").write_text("""\
name: train
image: cyberloop-train:latest
inputs: [checkpoint]
outputs: [checkpoint]
""")
    (blocks_dir / "eval.yaml").write_text("""\
name: eval
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
        template = from_yaml_with_inline_blocks(
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

        template = from_yaml_with_inline_blocks(
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
        """Verify that main() parses run block args and calls run_block."""
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)

        main(["create", "workspace", "--name", "test_ws",
              "--template", "my_template"])

        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.elapsed_s = 1.0

        with patch("flywheel.cli.run_block", return_value=mock_result) as mock_rb:
            main([
                "run", "block",
                "--workspace",
                str(project_root / "foundry" / "workspaces" / "test_ws"),
                "--block", "train",
                "--template", "my_template",
                "--", "--subclass", "dueling",
            ])
            mock_rb.assert_called_once()
            call_args = mock_rb.call_args
            assert call_args[0][1] == "train"


# ── flywheel run pattern ────────────────────────────────────────


def _write_pattern(project_root: Path, name: str, body: str) -> Path:
    patterns_dir = project_root / "patterns"
    patterns_dir.mkdir(exist_ok=True)
    path = patterns_dir / f"{name}.yaml"
    path.write_text(body)
    return path


class TestRunPattern:
    def test_unknown_pattern_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)
        main(["create", "workspace", "--name", "ws",
              "--template", "my_template"])

        with pytest.raises(SystemExit):
            main([
                "run", "pattern", "nope",
                "--workspace",
                str(project_root / "foundry"
                    / "workspaces" / "ws"),
                "--template", "my_template",
            ])
        out = capsys.readouterr().out
        assert "no pattern named 'nope'" in out

    def test_extra_args_without_hooks_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)
        main(["create", "workspace", "--name", "ws",
              "--template", "my_template"])
        _write_pattern(project_root, "p", """\
roles:
  play:
    prompt: prompts/play.md
    trigger:
      kind: continuous
""")

        with pytest.raises(SystemExit):
            main([
                "run", "pattern", "p",
                "--workspace",
                str(project_root / "foundry"
                    / "workspaces" / "ws"),
                "--template", "my_template",
                "--", "--game-id", "abc",
            ])
        out = capsys.readouterr().out
        assert "no project_hooks are configured" in out

    def test_runs_with_fake_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)
        main(["create", "workspace", "--name", "ws",
              "--template", "my_template"])
        _write_pattern(project_root, "tiny", """\
roles:
  play:
    prompt: prompts/play.md
    trigger:
      kind: continuous
""")

        captured: dict = {}

        class _FakeResult:
            agents_launched = 1
            cohorts_by_role = {"play": 1}

        class _FakeRunner:
            def __init__(self, pattern, **kwargs):
                captured["pattern"] = pattern
                captured["kwargs"] = kwargs

            def run(self):
                return _FakeResult()

        with patch("flywheel.cli.PatternRunner", _FakeRunner):
            main([
                "run", "pattern", "tiny",
                "--workspace",
                str(project_root / "foundry"
                    / "workspaces" / "ws"),
                "--template", "my_template",
                "--max-runtime", "5",
                "--poll-interval", "0.5",
            ])

        assert captured["pattern"].name == "tiny"
        assert captured["kwargs"]["poll_interval_s"] == 0.5
        assert captured["kwargs"]["max_total_runtime_s"] == 5

    def test_calls_project_hooks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)
        main(["create", "workspace", "--name", "ws",
              "--template", "my_template"])
        _write_pattern(project_root, "tiny", """\
roles:
  play:
    prompt: prompts/play.md
    trigger:
      kind: continuous
""")

        teardown_called = []
        init_called = []

        class _FakeHooks:
            def init(self, ws, template, pr, args):
                init_called.append((ws, args))
                return {"mcp_servers": "arc"}

            def teardown(self):
                teardown_called.append(True)

        captured: dict = {}

        class _FakeRunner:
            def __init__(self, pattern, **kwargs):
                captured["kwargs"] = kwargs

            def run(self):
                class R:
                    agents_launched = 0
                    cohorts_by_role = {"play": 0}
                return R()

        with patch(
            "flywheel.cli.PatternRunner", _FakeRunner,
        ), patch(
            "flywheel.cli.load_project_hooks_class",
            return_value=_FakeHooks,
        ):
            main([
                "run", "pattern", "tiny",
                "--workspace",
                str(project_root / "foundry"
                    / "workspaces" / "ws"),
                "--template", "my_template",
                "--project-hooks", "x:Y",
                "--", "--game-id", "abc",
            ])

        assert init_called
        # Project args were forwarded.
        assert init_called[0][1] == ["--game-id", "abc"]
        # teardown ran.
        assert teardown_called == [True]
        # The CLI built an executor_factory for the runner even
        # though the hooks did not supply one.
        assert callable(captured["kwargs"]["executor_factory"])

    def test_project_hooks_can_override_executor_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Projects that need the host-side full-stop handoff loop
        # (cyberarc) hand the runner a pre-configured
        # ``executor_factory`` from their hooks.  The CLI must
        # thread it through to :class:`PatternRunner` verbatim;
        # otherwise the default agent-only factory runs and the
        # handoff path never fires.
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)
        main(["create", "workspace", "--name", "ws",
              "--template", "my_template"])
        _write_pattern(project_root, "tiny", """\
roles:
  play:
    prompt: prompts/play.md
    trigger:
      kind: continuous
""")

        sentinel = lambda _block_def: object()

        class _Hooks:
            def init(self, ws, template, pr, args):
                return {"executor_factory": sentinel}

            def teardown(self):
                pass

        captured: dict = {}

        class _FakeRunner:
            def __init__(self, pattern, **kwargs):
                captured["kwargs"] = kwargs

            def run(self):
                class R:
                    agents_launched = 0
                    cohorts_by_role = {"play": 0}
                return R()

        with patch(
            "flywheel.cli.PatternRunner", _FakeRunner,
        ), patch(
            "flywheel.cli.load_project_hooks_class",
            return_value=_Hooks,
        ):
            main([
                "run", "pattern", "tiny",
                "--workspace",
                str(project_root / "foundry"
                    / "workspaces" / "ws"),
                "--template", "my_template",
                "--project-hooks", "x:Y",
            ])

        assert captured["kwargs"]["executor_factory"] is sentinel

    def test_project_hooks_without_executor_factory_uses_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # When hooks don't supply ``executor_factory`` the CLI
        # builds a default :class:`AgentExecutor`-backed factory
        # from CLI args + legacy battery-shaped overrides so
        # agent-only patterns keep working without project-side
        # wiring.
        project_root = make_project(tmp_path)
        monkeypatch.chdir(project_root)
        main(["create", "workspace", "--name", "ws",
              "--template", "my_template"])
        _write_pattern(project_root, "tiny", """\
roles:
  play:
    prompt: prompts/play.md
    trigger:
      kind: continuous
""")

        class _Hooks:
            def init(self, ws, template, pr, args):
                return {}

            def teardown(self):
                pass

        captured: dict = {}

        class _FakeRunner:
            def __init__(self, pattern, **kwargs):
                captured["kwargs"] = kwargs

            def run(self):
                class R:
                    agents_launched = 0
                    cohorts_by_role = {"play": 0}
                return R()

        with patch(
            "flywheel.cli.PatternRunner", _FakeRunner,
        ), patch(
            "flywheel.cli.load_project_hooks_class",
            return_value=_Hooks,
        ):
            main([
                "run", "pattern", "tiny",
                "--workspace",
                str(project_root / "foundry"
                    / "workspaces" / "ws"),
                "--template", "my_template",
                "--project-hooks", "x:Y",
            ])

        assert callable(captured["kwargs"]["executor_factory"])


class TestProjectConfigFields:
    def test_project_hooks_parsed(
        self, tmp_path: Path,
    ):
        from flywheel.config import load_project_config
        (tmp_path / "flywheel.yaml").write_text(
            "foundry_dir: foundry\n"
            "project_hooks: my.module:Hooks\n"
        )
        cfg = load_project_config(tmp_path)
        assert cfg.project_hooks == "my.module:Hooks"
        assert cfg.patterns_dir == tmp_path / "patterns"

    def test_project_hooks_wrong_type_raises(
        self, tmp_path: Path,
    ):
        from flywheel.config import load_project_config
        (tmp_path / "flywheel.yaml").write_text(
            "foundry_dir: foundry\n"
            "project_hooks: 42\n"
        )
        with pytest.raises(
                ValueError, match="project_hooks"):
            load_project_config(tmp_path)
