"""Project configuration loading for flywheel.

Reads and validates flywheel.yaml from a project root directory.
Centralizes config loading so error handling and future validation
changes happen in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_FILENAME = "flywheel.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    """Parsed flywheel project configuration.

    Attributes:
        project_root: The project root directory containing flywheel.yaml.
        harness_dir: Path to the workforce/harness directory.
    """

    project_root: Path
    harness_dir: Path

    @property
    def templates_dir(self) -> Path:
        """Path to the templates directory."""
        return self.harness_dir / "templates"


def load_project_config(project_root: Path) -> ProjectConfig:
    """Load flywheel.yaml from a project root directory.

    Args:
        project_root: The directory containing flywheel.yaml.

    Returns:
        A ProjectConfig with resolved paths.

    Raises:
        FileNotFoundError: If flywheel.yaml does not exist.
        ValueError: If flywheel.yaml is missing required fields or
            cannot be parsed.
    """
    config_path = project_root / CONFIG_FILENAME

    if not config_path.exists():
        raise FileNotFoundError(
            f"No {CONFIG_FILENAME} found in {project_root}. "
            f"Is this a flywheel project root?"
        )

    with open(config_path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Failed to parse {config_path}: {exc}"
            ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"{config_path} must contain a YAML mapping, "
            f"got {type(data).__name__}"
        )

    if "harness_dir" not in data:
        raise ValueError(
            f"{config_path} is missing required field 'harness_dir'"
        )

    harness_dir = project_root / data["harness_dir"]

    return ProjectConfig(
        project_root=project_root,
        harness_dir=harness_dir,
    )
