from __future__ import annotations

from pathlib import Path

import pytest

from flywheel.config import CONFIG_FILENAME, load_project_config


class TestLoadProjectConfig:
    def test_loads_valid_config(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("harness_dir: workforce\n")
        config = load_project_config(tmp_path)
        assert config.project_root == tmp_path
        assert config.harness_dir == tmp_path / "workforce"

    def test_templates_dir(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("harness_dir: workforce\n")
        config = load_project_config(tmp_path)
        assert config.templates_dir == tmp_path / "workforce" / "templates"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(
            FileNotFoundError, match="No flywheel.yaml found"
        ) as exc_info:
            load_project_config(tmp_path)
        assert "flywheel project root" in str(exc_info.value)

    def test_missing_harness_dir_field_raises(self, tmp_path: Path):
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


class TestProjectConfigFrozen:
    def test_frozen(self, tmp_path: Path):
        (tmp_path / CONFIG_FILENAME).write_text("harness_dir: workforce\n")
        config = load_project_config(tmp_path)
        with pytest.raises(AttributeError):
            config.project_root = tmp_path / "other"
