from __future__ import annotations

from pathlib import Path

import pytest

from flywheel.artifact_validator import ArtifactValidatorRegistry
from flywheel.config import CONFIG_FILENAME, load_project_config
from flywheel.state_validator import StateValidatorRegistry


class TestLoadProjectConfig:
    def test_loads_valid_config(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("foundry_dir: foundry\n")
        config = load_project_config(tmp_path)
        assert config.project_root == tmp_path
        assert config.foundry_dir == tmp_path / "foundry"

    def test_templates_dir(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("foundry_dir: foundry\n")
        config = load_project_config(tmp_path)
        assert config.templates_dir == tmp_path / "foundry" / "templates"
        assert config.workspace_templates_dir == (
            tmp_path / "foundry" / "templates" / "workspaces"
        )
        assert config.block_templates_dir == (
            tmp_path / "foundry" / "templates" / "blocks"
        )
        assert config.pattern_templates_dir == (
            tmp_path / "foundry" / "templates" / "patterns"
        )

    def test_missing_file_raises_with_helpful_message(self, tmp_path: Path):
        with pytest.raises(
            FileNotFoundError, match="No flywheel.yaml found"
        ) as exc_info:
            load_project_config(tmp_path)
        assert "flywheel project root" in str(exc_info.value)

    def test_missing_foundry_dir_field_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("other_field: value\n")
        with pytest.raises(ValueError, match="missing required field"):
            load_project_config(tmp_path)

    def test_invalid_yaml_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text(":\n  bad:\n[yaml")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_project_config(tmp_path)

    def test_non_mapping_yaml_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_project_config(tmp_path)

    def test_empty_file_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_project_config(tmp_path)


    def test_foundry_dir_non_string_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("foundry_dir: 42\n")
        with pytest.raises(ValueError, match="must be a string"):
            load_project_config(tmp_path)

    def test_foundry_dir_absolute_path_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("foundry_dir: /opt/foundry\n")
        with pytest.raises(ValueError, match="must be a relative path"):
            load_project_config(tmp_path)

    def test_nested_foundry_dir(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("foundry_dir: some/nested/dir\n")
        config = load_project_config(tmp_path)
        assert config.foundry_dir == tmp_path / "some" / "nested" / "dir"
        assert config.templates_dir == (
            tmp_path / "some" / "nested" / "dir" / "templates"
        )
        assert config.workspace_templates_dir == (
            tmp_path / "some" / "nested" / "dir"
            / "templates" / "workspaces"
        )


class TestHooksKeyRejected:
    """A ``hooks:`` key in ``flywheel.yaml`` must be rejected.

    The supported key is ``project_hooks``; accepting ``hooks``
    would silently route the project through the wrong
    configuration path, so the loader raises a directional
    error pointing at the correct key.
    """

    def test_hooks_key_present_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\nhooks: mymod:MyClass\n")
        with pytest.raises(
                ValueError, match="not supported"):
            load_project_config(tmp_path)

    def test_hooks_key_absent_is_fine(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\n")
        config = load_project_config(tmp_path)
        assert config.project_hooks is None


class TestArtifactValidators:
    """``artifact_validators:`` parsing + factory resolution."""

    def test_default_returns_empty_registry(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\n")
        config = load_project_config(tmp_path)
        assert config.artifact_validators is None
        registry = config.load_artifact_validator_registry()
        assert isinstance(registry, ArtifactValidatorRegistry)
        assert registry.names() == []

    def test_non_string_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\nartifact_validators: 42\n",
        )
        with pytest.raises(
                ValueError,
                match="must be a string in the form",
        ):
            load_project_config(tmp_path)

    def test_factory_resolves(
        self, tmp_path: Path, monkeypatch,
    ):
        # Build a real importable module exposing a factory.
        # Use a unique package name to avoid colliding with
        # other tests' tmp_path packages in sys.modules.
        pkg_dir = tmp_path / "pkg_factory_resolves"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "validators.py").write_text(
            "from flywheel.artifact_validator import "
            "ArtifactValidatorRegistry\n"
            "def build():\n"
            "    r = ArtifactValidatorRegistry()\n"
            "    r.register('cp', lambda n, d, p: None)\n"
            "    return r\n",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\n"
            "artifact_validators: "
            "pkg_factory_resolves.validators:build\n",
        )
        config = load_project_config(tmp_path)
        registry = config.load_artifact_validator_registry()
        assert registry.has("cp")

    def test_malformed_path_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\n"
            "artifact_validators: no_colon_here\n",
        )
        config = load_project_config(tmp_path)
        with pytest.raises(
                ValueError, match="module.path:factory_name",
        ):
            config.load_artifact_validator_registry()

    def test_factory_returns_wrong_type_raises(
        self, tmp_path: Path, monkeypatch,
    ):
        pkg_dir = tmp_path / "pkg_wrong_type"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "bad.py").write_text(
            "def build():\n    return 'not a registry'\n",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\n"
            "artifact_validators: pkg_wrong_type.bad:build\n",
        )
        config = load_project_config(tmp_path)
        with pytest.raises(
                ValueError, match="ArtifactValidatorRegistry",
        ):
            config.load_artifact_validator_registry()


class TestStateValidators:
    """``state_validators:`` parsing + factory resolution."""

    def test_default_returns_empty_registry(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\n")
        config = load_project_config(tmp_path)
        assert config.state_validators is None
        registry = config.load_state_validator_registry()
        assert isinstance(registry, StateValidatorRegistry)
        assert registry.names() == []

    def test_non_string_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\nstate_validators: 42\n",
        )
        with pytest.raises(
                ValueError,
                match="must be a string in the form",
        ):
            load_project_config(tmp_path)

    def test_factory_resolves(self, tmp_path: Path, monkeypatch):
        pkg_dir = tmp_path / "pkg_state_validators"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "validators.py").write_text(
            "from flywheel.state_validator import "
            "StateValidatorRegistry\n"
            "def build():\n"
            "    r = StateValidatorRegistry()\n"
            "    r.register('train', lambda b, d, p, k: None)\n"
            "    return r\n",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\n"
            "state_validators: pkg_state_validators.validators:build\n",
        )
        config = load_project_config(tmp_path)
        registry = config.load_state_validator_registry()
        assert registry.has("train")

    def test_malformed_path_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\n"
            "state_validators: no_colon_here\n",
        )
        config = load_project_config(tmp_path)
        with pytest.raises(
                ValueError, match="module.path:factory_name",
        ):
            config.load_state_validator_registry()

    def test_factory_returns_wrong_type_raises(
        self, tmp_path: Path, monkeypatch,
    ):
        pkg_dir = tmp_path / "pkg_wrong_state_type"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "bad.py").write_text(
            "def build():\n    return 'not a registry'\n",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\n"
            "state_validators: pkg_wrong_state_type.bad:build\n",
        )
        config = load_project_config(tmp_path)
        with pytest.raises(
                ValueError, match="StateValidatorRegistry",
        ):
            config.load_state_validator_registry()


class TestProjectConfigFrozen:
    def test_frozen(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("foundry_dir: foundry\n")
        config = load_project_config(tmp_path)
        with pytest.raises(AttributeError):
            config.project_root = tmp_path / "other"
