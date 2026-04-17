"""Project-wide registry of declared blocks.

A :class:`BlockRegistry` loads per-block YAML files from a
``workforce/blocks/`` directory and exposes the parsed
:class:`~flywheel.template.BlockDefinition` objects by name.
Templates can then reference blocks by name in their ``blocks:``
list and have them resolved at template-load time.

This decouples block declarations from any specific template, so
the same block can be reused by multiple templates (e.g., a
``predict`` block usable by both ``arc_play`` and a future
debugging template) without copy-pasting its full definition.

Per-block YAML schema mirrors what
:func:`flywheel.template.parse_block_definition` accepts when given
a single block-entry mapping.  See
``plans/flywheel-block-execution-refactor.md`` for the full schema.

Example:

.. code-block:: python

    from flywheel.blocks import BlockRegistry
    from flywheel.template import Template

    registry = BlockRegistry.from_directory(
        project_root / "workforce" / "blocks")
    template = Template.from_yaml(
        templates_dir / "arc_play.yaml",
        block_registry=registry,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from flywheel.template import BlockDefinition, parse_block_definition


@dataclass(frozen=True)
class BlockRegistry:
    """An immutable name → :class:`BlockDefinition` mapping.

    Construct via :meth:`from_directory` or :meth:`from_files` so
    the parser can validate each entry.  Direct construction with
    a pre-built dict is supported for tests and synthetic registries.

    Attributes:
        blocks: The underlying name → BlockDefinition map.
        sources: Per-block source-file paths, useful for error
            messages and diffing.  Empty for synthetic registries.
    """

    blocks: dict[str, BlockDefinition] = field(default_factory=dict)
    sources: dict[str, Path] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self.blocks

    def get(self, name: str) -> BlockDefinition:
        """Look up a block by name.

        Raises:
            KeyError: If no block is registered under ``name``.
        """
        if name not in self.blocks:
            raise KeyError(
                f"No block named {name!r} in registry; "
                f"known blocks: {sorted(self.blocks)}"
            )
        return self.blocks[name]

    def names(self) -> list[str]:
        """Return all registered block names, sorted."""
        return sorted(self.blocks)

    @classmethod
    def from_files(
        cls, files: list[Path],
    ) -> BlockRegistry:
        """Build a registry from explicit YAML file paths.

        Each file must contain a single block definition (a YAML
        mapping at the top level).  Block names must be unique
        across all files; the file stem must equal the block's
        ``name`` field as a sanity check.

        Args:
            files: List of YAML file paths to load.

        Returns:
            A populated registry.

        Raises:
            ValueError: For any duplicate name, missing required
                field, or stem/name mismatch.
        """
        blocks: dict[str, BlockDefinition] = {}
        sources: dict[str, Path] = {}
        for path in files:
            block = load_block_file(path)
            if path.stem != block.name:
                raise ValueError(
                    f"Block file {path} declares name "
                    f"{block.name!r} but file stem is "
                    f"{path.stem!r}; they must match"
                )
            if block.name in blocks:
                raise ValueError(
                    f"Duplicate block name {block.name!r}: "
                    f"{sources[block.name]} and {path}"
                )
            blocks[block.name] = block
            sources[block.name] = path
        return cls(blocks=blocks, sources=sources)

    @classmethod
    def from_directory(
        cls, directory: Path,
    ) -> BlockRegistry:
        """Build a registry from all ``*.yaml`` files in a directory.

        Non-recursive: only files directly under ``directory`` are
        loaded.  Files starting with an underscore are skipped so
        authors can stash partial drafts.

        If ``directory`` does not exist or contains no YAML files,
        an empty registry is returned.  Callers that require at
        least one block should check ``registry.names()`` after
        loading.

        Args:
            directory: Path to a directory of per-block YAML files.

        Returns:
            A populated registry.
        """
        if not directory.exists() or not directory.is_dir():
            return cls()
        files = sorted(
            p for p in directory.glob("*.yaml")
            if not p.name.startswith("_")
        )
        return cls.from_files(files)


def load_block_file(path: Path) -> BlockDefinition:
    """Parse a single per-block YAML file.

    Args:
        path: Path to a YAML file containing one block definition
            at the top level.

    Returns:
        The parsed :class:`BlockDefinition`.

    Raises:
        ValueError: If the YAML is empty, not a mapping, or fails
            block-definition validation.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Block file {path} is empty")
    if not isinstance(data, dict):
        raise ValueError(
            f"Block file {path} must contain a YAML mapping at the "
            f"top level, got {type(data).__name__}"
        )
    return parse_block_definition(data, source=str(path))
