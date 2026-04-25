"""Project configuration loading for flywheel.

Reads and validates flywheel.yaml from a project root directory.
Centralizes config loading so error handling and future validation
changes happen in one place.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from flywheel.artifact_validator import ArtifactValidatorRegistry
from flywheel.state_validator import StateValidatorRegistry

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
        artifact_validators: Optional Python import path for a
            zero-arg factory that returns an
            :class:`flywheel.artifact_validator.ArtifactValidatorRegistry`,
            in the form ``module.path:factory_name``.  The
            registry is consulted at every artifact-finalization
            site (``flywheel import artifact`` and post-block
            output collection).  Optional: when omitted,
            :meth:`load_artifact_validator_registry` returns an
            empty registry that accepts everything.
        state_validators: Optional Python import path for a zero-arg
            factory that returns a
            :class:`flywheel.state_validator.StateValidatorRegistry`.
            The registry is consulted during managed-state capture.
    """

    project_root: Path
    foundry_dir: Path
    project_hooks: str | None = None
    artifact_validators: str | None = None
    state_validators: str | None = None

    @property
    def templates_dir(self) -> Path:
        """Path to the root templates directory."""
        return self.foundry_dir / "templates"

    @property
    def workspace_templates_dir(self) -> Path:
        """Path to workspace template YAML files."""
        return self.templates_dir / "workspaces"

    @property
    def block_templates_dir(self) -> Path:
        """Path to per-block template YAML files.

        Convention: ``<foundry_dir>/templates/blocks/``.  Used by
        :meth:`flywheel.blocks.BlockRegistry.from_directory` to
        load block definitions referenced by name from workspace
        templates.
        The directory is optional; templates that only use inline
        block definitions need not have it.
        """
        return self.templates_dir / "blocks"

    @property
    def pattern_templates_dir(self) -> Path:
        """Path to pattern template YAML files.

        Convention: ``<foundry_dir>/templates/patterns/``.  Used by
        ``flywheel run pattern`` to discover declarative patterns
        by file stem.  The directory is optional; projects that
        ship no patterns yet have an implicit empty registry.
        """
        return self.templates_dir / "patterns"

    def load_block_registry(self):  # type: ignore[no-untyped-def]
        """Load the project's :class:`BlockRegistry`.

        Convenience wrapper around
        ``BlockRegistry.from_directory(self.block_templates_dir)``.
        Returns an empty registry if the directory does not
        exist, which is the supported state for projects whose
        blocks are still inline in templates.
        """
        # Local import to avoid a config → blocks → template cycle.
        from flywheel.blocks.registry import BlockRegistry  # noqa: PLC0415
        return BlockRegistry.from_directory(self.block_templates_dir)

    def load_artifact_validator_registry(
        self,
    ) -> ArtifactValidatorRegistry:
        """Resolve the project's artifact validator registry.

        Returns an empty registry (which accepts every name)
        when ``artifact_validators`` is unset.  When set, the
        configured ``module.path:factory`` is imported and
        called with no arguments; the result must be an
        :class:`ArtifactValidatorRegistry` instance.

        Raises:
            ValueError: If the import path is malformed, the
                factory is not callable, or the factory returns
                something other than an
                :class:`ArtifactValidatorRegistry`.
            ImportError: If the named module cannot be imported.
            AttributeError: If the named attribute does not
                exist on the imported module.
        """
        import_path = self.artifact_validators
        if import_path is None:
            return ArtifactValidatorRegistry()
        if ":" not in import_path:
            raise ValueError(
                f"'artifact_validators' import path must be "
                f"'module.path:factory_name', got "
                f"{import_path!r}"
            )
        module_path, attr_name = import_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        factory = getattr(module, attr_name)
        if not callable(factory):
            raise ValueError(
                f"'artifact_validators' target {import_path!r} "
                f"resolved to a non-callable {type(factory).__name__}"
            )
        registry = factory()
        if not isinstance(registry, ArtifactValidatorRegistry):
            raise ValueError(
                f"'artifact_validators' factory {import_path!r} "
                f"returned {type(registry).__name__}, expected "
                f"ArtifactValidatorRegistry"
            )
        return registry

    def load_state_validator_registry(self) -> StateValidatorRegistry:
        """Resolve the project's managed-state validator registry."""
        import_path = self.state_validators
        if import_path is None:
            return StateValidatorRegistry()
        if ":" not in import_path:
            raise ValueError(
                f"'state_validators' import path must be "
                f"'module.path:factory_name', got "
                f"{import_path!r}"
            )
        module_path, attr_name = import_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        factory = getattr(module, attr_name)
        if not callable(factory):
            raise ValueError(
                f"'state_validators' target {import_path!r} "
                f"resolved to a non-callable {type(factory).__name__}"
            )
        registry = factory()
        if not isinstance(registry, StateValidatorRegistry):
            raise ValueError(
                f"'state_validators' factory {import_path!r} "
                f"returned {type(registry).__name__}, expected "
                f"StateValidatorRegistry"
            )
        return registry


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
            f"'hooks' in {config_path} is not supported. "
            f"Declare workflows under "
            f"'<foundry_dir>/templates/patterns/' "
            f"and wire resource setup via 'project_hooks' "
            f"instead."
        )

    project_hooks = data.get("project_hooks")
    if project_hooks is not None and not isinstance(
            project_hooks, str):
        raise ValueError(
            f"'project_hooks' in {config_path} must be a string "
            f"in the form 'module.path:ClassName', got "
            f"{type(project_hooks).__name__}"
        )

    artifact_validators = data.get("artifact_validators")
    if artifact_validators is not None and not isinstance(
            artifact_validators, str):
        raise ValueError(
            f"'artifact_validators' in {config_path} must be a "
            f"string in the form 'module.path:factory_name', "
            f"got {type(artifact_validators).__name__}"
        )

    state_validators = data.get("state_validators")
    if state_validators is not None and not isinstance(
            state_validators, str):
        raise ValueError(
            f"'state_validators' in {config_path} must be a "
            f"string in the form 'module.path:factory_name', "
            f"got {type(state_validators).__name__}"
        )

    return ProjectConfig(
        project_root=project_root,
        foundry_dir=foundry_dir,
        project_hooks=project_hooks,
        artifact_validators=artifact_validators,
        state_validators=state_validators,
    )
