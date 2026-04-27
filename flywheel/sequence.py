"""Artifact sequence ledger types and helpers.

Sequences are append-only ordered references to immutable artifact
instances.  They are not artifact storage, mutable tags, or replacement
relations; they are a durable ordering layer over the artifact ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from flywheel.validation import validate_name

SequenceScopeKind = Literal["workspace", "run", "lane"]
SequenceScopePolicy = Literal["workspace", "enclosing_run", "enclosing_lane"]


@dataclass(frozen=True)
class SequenceScope:
    """Concrete scope for an artifact sequence."""

    kind: SequenceScopeKind
    run_id: str | None = None
    lane: str | None = None

    @staticmethod
    def workspace() -> "SequenceScope":
        """Return workspace-wide sequence scope."""
        return SequenceScope(kind="workspace")

    @staticmethod
    def run(run_id: str) -> "SequenceScope":
        """Return run-wide sequence scope."""
        if not run_id:
            raise ValueError("Run sequence scope requires run_id")
        return SequenceScope(kind="run", run_id=run_id)

    @staticmethod
    def for_lane(run_id: str, lane: str) -> "SequenceScope":
        """Return lane-scoped sequence scope."""
        if not run_id:
            raise ValueError("Lane sequence scope requires run_id")
        if not lane:
            raise ValueError("Lane sequence scope requires lane")
        return SequenceScope(kind="lane", run_id=run_id, lane=lane)

    def key(self) -> tuple[str, str | None, str | None]:
        """Return a hashable identity key."""
        return (self.kind, self.run_id, self.lane)


@dataclass(frozen=True)
class SequenceDeclaration:
    """Template-level sequence declaration on an input or output slot."""

    name: str
    scope: SequenceScopePolicy
    role: str | None = None


@dataclass(frozen=True)
class ArtifactSequenceEntry:
    """One append-only sequence membership row."""

    sequence_name: str
    scope: SequenceScope
    index: int
    artifact_id: str
    role: str | None
    recorded_at: datetime


@dataclass(frozen=True)
class SequenceEntryRef:
    """Execution-local snapshot reference to a sequence entry."""

    index: int
    artifact_id: str
    role: str | None


@dataclass(frozen=True)
class SequenceBinding:
    """Sequence snapshot consumed by one block input slot."""

    sequence_name: str
    scope: SequenceScope
    entries: list[SequenceEntryRef]


@dataclass(frozen=True)
class RunContext:
    """Optional run/lane context for resolving sequence scope policies."""

    run_id: str | None = None
    lane: str | None = None

    @staticmethod
    def empty() -> "RunContext":
        """Return an ad-hoc execution context."""
        return RunContext()


def validate_sequence_name(name: str, context: str) -> None:
    """Validate a sequence or role name with Flywheel's name rules."""
    validate_name(name, context)


def resolve_sequence_scope(
    policy: SequenceScopePolicy,
    context: RunContext,
) -> SequenceScope:
    """Resolve a template scope policy to a concrete sequence scope."""
    if policy == "workspace":
        return SequenceScope.workspace()
    if policy == "enclosing_run":
        if context.run_id is None:
            raise ValueError(
                "sequence scope 'enclosing_run' requires pattern run context"
            )
        return SequenceScope.run(context.run_id)
    if policy == "enclosing_lane":
        if context.run_id is None or context.lane is None:
            raise ValueError(
                "sequence scope 'enclosing_lane' requires pattern lane context"
            )
        return SequenceScope.for_lane(context.run_id, context.lane)
    raise ValueError(f"unknown sequence scope policy {policy!r}")
