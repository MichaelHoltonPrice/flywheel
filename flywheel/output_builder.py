"""Host-side output-builder callbacks for block executions.

An ``output_builder`` is an optional dotted-path Python callable
attached to a :class:`flywheel.template.BlockDefinition`.  When
set, it runs on the host *after* the block's container exits and
*before* flywheel's standard artifact-collection pass reads the
per-execution output directories.

The separation matters: the container body is free to produce
whatever human-readable intermediate content is natural for the
agent inside it (e.g., two markdown files), while the builder
collapses that content into the canonical artifact shape the
consuming side expects (e.g., a normalized JSON manifest or a
single summary file). Provenance fields that depend on workspace
state can be stamped here using host-side knowledge the container
never had.

The builder mutates the per-execution output tempdir in place:
it can write new files, rewrite existing ones, or leave the dir
empty (in which case the corresponding artifact is not created,
same as any empty output dir).  Files the builder writes are
picked up by the standard collection path as if the container
had written them directly.

The builder MUST NOT write outside the output tempdir. The workspace
is passed for read-only introspection. Mutating the workspace from
inside a builder is unsupported and
the workspace's durability guarantees do not cover it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flywheel.workspace import Workspace


@dataclass(frozen=True)
class OutputBuilderContext:
    """Read-only context handed to an ``output_builder`` callback.

    Attributes:
        block: Block name.
        execution_id: Channel-assigned execution ID.  Stable
            across the execution record's lifetime; suitable as
            a join key to :class:`flywheel.artifact.BlockExecution`
            metadata without duplicating that metadata into
            artifact bodies.
        outputs: Mapping from output slot name to the
            per-execution output tempdir on the host filesystem.
            The builder may read from these directories and
            write new or replacement files within them.  Paths
            outside these dirs are not part of the contract.
        workspace: The workspace the execution belongs to. Passed
            for read-only introspection. Mutating the workspace from
            inside a builder is unsupported.
    """

    block: str
    execution_id: str
    outputs: Mapping[str, Path]
    workspace: "Workspace"


OutputBuilderCallable = Callable[[OutputBuilderContext], None]


def resolve_dotted_path(path: str) -> OutputBuilderCallable:
    """Import a dotted path and return the resolved callable.

    Used at registry-load time to validate ``output_builder``
    fields.  Raises :class:`ValueError` with a useful message on
    any kind of resolution failure.

    Args:
        path: A dotted path of the form ``pkg.mod.func``.

    Returns:
        The resolved callable.

    Raises:
        ValueError: If the path can't be parsed, the module
            can't be imported, the attribute is missing, or the
            attribute isn't callable.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(
            f"output_builder dotted path must be a non-empty "
            f"string, got {path!r}")
    if "." not in path:
        raise ValueError(
            f"output_builder dotted path {path!r} must include "
            f"at least one '.' (e.g., "
            f"'mypkg.builders.my_builder')")
    module_name, _, attr = path.rpartition(".")
    try:
        import importlib
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"output_builder {path!r}: cannot import module "
            f"{module_name!r}: {exc}") from exc
    if not hasattr(module, attr):
        raise ValueError(
            f"output_builder {path!r}: module {module_name!r} "
            f"has no attribute {attr!r}")
    func = getattr(module, attr)
    if not callable(func):
        raise ValueError(
            f"output_builder {path!r}: resolved object is not "
            f"callable (got {type(func).__name__})")
    return func
