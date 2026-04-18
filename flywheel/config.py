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
        foundry_dir: Path to the foundry directory.
        project_hooks: Optional Python import path for the
            project-side hooks consumed by ``flywheel run
            pattern``, in the form ``module.path:ClassName``.
            Resolved by
            :func:`flywheel.project_hooks.load_project_hooks_class`.
            Optional: a pure-pattern project that needs no
            project-side resource setup may omit it.
    """

    project_root: Path
    foundry_dir: Path
    project_hooks: str | None = None

    @property
    def templates_dir(self) -> Path:
        """Path to the templates directory."""
        return self.foundry_dir / "templates"

    @property
    def patterns_dir(self) -> Path:
        """Path to the patterns directory.

        Convention: ``<project_root>/patterns/``.  Used by
        ``flywheel run pattern`` to discover declarative patterns
        by file stem.  The directory is optional; projects that
        ship no patterns yet have an implicit empty registry.
        """
        return self.project_root / "patterns"

    @property
    def blocks_dir(self) -> Path:
        """Path to the per-block YAML directory.

        Convention: ``<project_root>/workforce/blocks/``.  Used by
        :meth:`flywheel.blocks.BlockRegistry.from_directory` to
        load block definitions referenced by name from templates.
        The directory is optional; templates that only use inline
        block definitions need not have it.
        """
        return self.project_root / "workforce" / "blocks"

    def load_block_registry(self):  # type: ignore[no-untyped-def]
        """Load the project's :class:`BlockRegistry`.

        Convenience wrapper around
        ``BlockRegistry.from_directory(self.blocks_dir)``.
        Returns an empty registry if the directory does not exist,
        which is the supported state during the Phase 2 migration
        for projects whose blocks are still inline in templates.
        """
        # Local import to avoid a config → blocks → template cycle.
        from flywheel.blocks.registry import BlockRegistry
        return BlockRegistry.from_directory(self.blocks_dir)


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

    if "foundry_dir" not in data:
        raise ValueError(
            f"{config_path} is missing required field 'foundry_dir'"
        )

    raw_foundry = data["foundry_dir"]
    if not isinstance(raw_foundry, str):
        raise ValueError(
            f"'foundry_dir' in {config_path} must be a string, "
            f"got {type(raw_foundry).__name__}"
        )

    foundry_path = Path(raw_foundry)
    if foundry_path.is_absolute() or raw_foundry.startswith("/"):
        raise ValueError(
            f"'foundry_dir' in {config_path} must be a relative path, "
            f"got {raw_foundry!r}"
        )

    foundry_dir = project_root / raw_foundry

    if "hooks" in data:
        raise ValueError(
            f"'hooks' in {config_path} is no longer supported. "
            f"The legacy AgentLoop / AgentLoopHooks path was "
            f"retired in P7 of the patterns campaign; declare "
            f"workflows under '<project>/patterns/' and wire "
            f"resource setup via 'project_hooks' instead."
        )

    project_hooks = data.get("project_hooks")
    if project_hooks is not None and not isinstance(
            project_hooks, str):
        raise ValueError(
            f"'project_hooks' in {config_path} must be a string "
            f"in the form 'module.path:ClassName', got "
            f"{type(project_hooks).__name__}"
        )

    return ProjectConfig(
        project_root=project_root,
        foundry_dir=foundry_dir,
        project_hooks=project_hooks,
    )
