from __future__ import annotations

from pathlib import Path

import pytest

from flywheel.config import CONFIG_FILENAME, load_project_config


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
        assert config.templates_dir == tmp_path / "some" / "nested" / "dir" / "templates"


class TestLegacyHooksKeyRejected:
    """The legacy ``hooks:`` key was retired in P7 of the
    patterns campaign.  It now raises a directional error instead
    of silently being ignored, so workspaces created before the
    migration produce a loud failure instead of running the
    project with the wrong orchestration verb.
    """

    def test_hooks_key_present_raises(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\nhooks: mymod:MyClass\n")
        with pytest.raises(
                ValueError, match="no longer supported"):
            load_project_config(tmp_path)

    def test_hooks_key_absent_is_fine(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text(
            "foundry_dir: foundry\n")
        config = load_project_config(tmp_path)
        assert config.project_hooks is None


class TestProjectConfigFrozen:
    def test_frozen(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("foundry_dir: foundry\n")
        config = load_project_config(tmp_path)
        with pytest.raises(AttributeError):
            config.project_root = tmp_path / "other"
