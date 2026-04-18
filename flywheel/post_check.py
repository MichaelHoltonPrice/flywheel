"""Post-execution callback protocol.

A *post-execution callback* is a project-side quality gate that
runs after :class:`flywheel.local_block.LocalBlockRecorder`
finalizes a block-execution row.  It can ask the runner that
invoked the block to stop; it cannot emit artifacts, mutate the
workspace, or change what was recorded.

Opt-in is per block, in YAML::

    # workforce/blocks/<name>.yaml
    name: <block>
    runner: lifecycle
    runner_justification: "..."
    post_check: <python.dotted.path.to.callable>

The dotted path is resolved at registry-load time
(:meth:`flywheel.blocks.registry.BlockRegistry.from_files`) so a
typo fails the run at startup, not in the middle of a long
training session.

The callable is invoked synchronously by the recorder after
``end_execution`` (or after the recorder synthesises a failed
row for a body that raised before ``end``).  Callbacks may
return a :class:`HaltDirective` to ask the host loop to stop;
they may not mutate the row, the workspace ledger, or the
artifacts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class HaltDirective:
    """A request to halt one or more runners.

    Returned from a post-execution callback.  The channel persists
    the directive on the originating ``BlockExecution`` row and on
    its per-channel halt queue, where runners pick it up via the
    ``/halt`` endpoint.

    Attributes:
        scope: Which runners to halt.  ``"caller"`` halts only the
            runner whose ``execution_id`` matches the originating
            row's ``parent_execution_id`` (typically the agent
            container that issued the tool call).  ``"run"`` halts
            every runner connected to the channel.
        reason: Human-readable explanation, surfaced verbatim in
            the runner's exit message and persisted on the row.
    """

    scope: Literal["caller", "run"]
    reason: str

    def to_dict(self) -> dict:
        """Plain dict form used for JSON wire/YAML persistence."""
        return {"scope": self.scope, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict) -> "HaltDirective":
        """Inverse of :meth:`to_dict`.  Validates the scope value."""
        scope = data.get("scope")
        if scope not in ("caller", "run"):
            raise ValueError(
                f"HaltDirective.scope must be 'caller' or 'run', "
                f"got {scope!r}")
        reason = data.get("reason", "")
        if not isinstance(reason, str):
            raise ValueError(
                f"HaltDirective.reason must be str, "
                f"got {type(reason).__name__}")
        return cls(scope=scope, reason=reason)


@dataclass(frozen=True)
class PostCheckContext:
    """Read-only view of a finalized execution, passed to a check.

    The check must not mutate any field reachable through this
    object.

    Attributes:
        block: Block name.
        execution_id: Channel-assigned execution ID.
        status: ``"succeeded"`` or ``"failed"``.
        caller: ``{"mcp_server": str, "tool": str}`` if the
            execution was tool-triggered, else ``None``.
        params: Caller-supplied params (e.g. ``{"action_id": 6}``).
        error: Body-failure error string when ``status="failed"``.
            ``None`` when ``status="succeeded"``.
        outputs: Mapping from output slot name to the per-output
            directory the block wrote into, on the host filesystem.
            Post-checks may read from these directories but must
            not modify them.  Empty mapping when the row is failed
            or when the block declared no outputs.  Note: this is
            *not* the canonical artifact directory — it's the
            ephemeral per-execution staging dir, valid for the
            duration of the post-check call.  For canonical
            access, use ``execution.output_bindings`` plus
            :meth:`flywheel.workspace.Workspace.instance_path`.
        parent_execution_id: ID of the runner that invoked this
            block, when known.
        synthetic: ``True`` if this row was synthesised by the
            channel for a request that never made it past
            ``begin`` (e.g., manifest mismatch, unknown block).
            Use this to halt the whole run on infrastructure
            failures even if the body never ran.
        workspace_path: Read-only access to the workspace
            directory.  The check must not write to it.
    """

    block: str
    execution_id: str
    status: Literal["succeeded", "failed"]
    caller: dict | None
    params: dict | None
    error: str | None
    outputs: Mapping[str, Path]
    parent_execution_id: str | None
    synthetic: bool
    workspace_path: Path


# Type alias for what a post_check callable looks like.
PostCheckCallable = Callable[
    [PostCheckContext], "HaltDirective | None"]


def resolve_dotted_path(path: str) -> PostCheckCallable:
    """Import a dotted path and return the resolved callable.

    Used at registry-load time to validate ``post_check`` fields.
    Raises :class:`ValueError` with a useful message on any kind
    of resolution failure (missing module, missing attribute, not
    callable).

    Args:
        path: A dotted path of the form ``pkg.mod.func``.

    Returns:
        The resolved callable.

    Raises:
        ValueError: If the path can't be parsed, the module can't
            be imported, the attribute is missing, or the
            attribute isn't callable.
    """
    if not isinstance(path, str) or not path:
        raise ValueError(
            f"post_check dotted path must be a non-empty string, "
            f"got {path!r}")
    if "." not in path:
        raise ValueError(
            f"post_check dotted path {path!r} must include at "
            f"least one '.' (e.g., 'mypkg.checks.my_check')")
    module_name, _, attr = path.rpartition(".")
    try:
        import importlib
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"post_check {path!r}: cannot import module "
            f"{module_name!r}: {exc}") from exc
    if not hasattr(module, attr):
        raise ValueError(
            f"post_check {path!r}: module {module_name!r} has no "
            f"attribute {attr!r}")
    func = getattr(module, attr)
    if not callable(func):
        raise ValueError(
            f"post_check {path!r}: resolved object is not "
            f"callable (got {type(func).__name__})")
    return func
