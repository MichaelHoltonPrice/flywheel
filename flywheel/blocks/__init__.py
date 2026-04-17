"""Block-related subpackage.

Currently houses the :class:`BlockRegistry` which loads per-block
YAML files from ``<project>/workforce/blocks/``.
"""

from flywheel.blocks.registry import BlockRegistry, load_block_file

__all__ = ["BlockRegistry", "load_block_file"]
