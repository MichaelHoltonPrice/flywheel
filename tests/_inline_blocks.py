"""Test helpers for templates that use inline-block syntax.

Production templates only accept string block references resolved
against a :class:`BlockRegistry`; ``Template.from_yaml`` rejects
inline-dict blocks.

Many test fixtures find it convenient to declare blocks inline
rather than splitting them into per-block YAML files.  This
helper detects inline blocks in a fixture, builds an ad-hoc
registry, rewrites the template to use string references, and
parses it normally.  Production code never goes through this
path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from flywheel.blocks.registry import BlockRegistry
from flywheel.template import (
    BlockDefinition,
    Template,
    parse_block_definition,
)


def from_yaml_with_inline_blocks(path: Path) -> Template:
    """Parse a template fixture that may use inline-dict blocks.

    The helper:

    1. Reads the YAML.
    2. If ``blocks:`` contains only string references, delegates
       to :meth:`Template.from_yaml` directly.
    3. Otherwise, parses each inline-dict entry through
       :func:`parse_block_definition`, builds a synthetic
       :class:`BlockRegistry`, rewrites the template's
       ``blocks:`` list as string refs to that registry, and
       parses the rewritten file.

    Raises:
        ValueError: If the ``blocks:`` list mixes strings and
            dicts (this helper does not support the mixed form;
            use one or the other in a single fixture).
    """
    data = yaml.safe_load(path.read_text())
    raw_blocks = data.get("blocks", []) or []
    if all(isinstance(e, str) for e in raw_blocks):
        return Template.from_yaml(path)
    if not all(isinstance(e, dict) for e in raw_blocks):
        raise ValueError(
            "Mixed inline/string block lists are not supported by "
            "from_yaml_with_inline_blocks; the helper expects "
            "either all strings or all dicts."
        )
    registry_blocks: dict[str, BlockDefinition] = {
        e["name"]: parse_block_definition(e) for e in raw_blocks
    }
    registry = BlockRegistry(blocks=registry_blocks)
    rewritten = {**data, "blocks": [e["name"] for e in raw_blocks]}
    # Write the rewritten copy under a temp dir so it never touches
    # the original directory (which may be a git repo whose clean
    # state is checked by Workspace.create).
    rewritten_dir = Path(
        tempfile.mkdtemp(prefix="flywheel-test-tmpl-"))
    rewritten_path = rewritten_dir / path.name
    rewritten_path.write_text(yaml.safe_dump(rewritten))
    return Template.from_yaml(rewritten_path, block_registry=registry)
