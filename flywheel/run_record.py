"""Run-record schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from flywheel.pattern_lanes import DEFAULT_LANE


@dataclass(frozen=True)
class RunFixtureRecord:
    """Fixture artifact materialized for one run lane."""

    id: str
    lane: str
    name: str
    artifact_id: str
    source: str


@dataclass(frozen=True)
class RunMemberRecord:
    """Result of one named member within a run step."""

    name: str
    block_name: str
    status: Literal["succeeded", "failed", "skipped"]
    lane: str = DEFAULT_LANE
    execution_id: str | None = None
    output_bindings: dict[str, str] = field(default_factory=dict)
    invocation_ids: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class RunStepRecord:
    """Result of one step within a run."""

    name: str
    min_successes: int | Literal["all"]
    status: Literal["succeeded", "failed"]
    members: list[RunMemberRecord] = field(default_factory=list)


@dataclass(frozen=True)
class RunRecord:
    """A durable grouping of related block executions.

    Run-level metadata lives here rather than on block executions.
    Block executions do not carry run-specific fields.

    A workspace can hold many runs. Ad hoc work does not create a
    ``RunRecord``.

    Attributes:
        id: Unique identifier within the workspace.
        kind: What opened this run. Convention:
            ``"pattern:<name>"`` for pattern-driven runs.
        started_at: When the run opened.
        finished_at: When the run closed, or ``None`` if still
            running.
        status: ``"running"``, ``"succeeded"``, ``"failed"``,
            ``"stopped"``, or ``"interrupted"``.
        config_snapshot: Optional free-form mapping the caller
            records alongside the run for later inspection.
        params: Resolved pattern parameters supplied by the
            operator or defaults at run start.  These are durable
            orchestration inputs, not block outputs.
        lanes: Declared run-scoped artifact lanes.
        fixtures: Fixture artifacts materialized for the run.
        steps: Ordered step results recorded for the run.
        error: Optional run-level error summary.
    """

    id: str
    kind: str
    started_at: datetime
    finished_at: datetime | None = None
    status: Literal[
        "running", "succeeded", "failed", "stopped", "interrupted"
    ] = "running"
    config_snapshot: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    lanes: list[str] = field(default_factory=lambda: [DEFAULT_LANE])
    fixtures: list[RunFixtureRecord] = field(default_factory=list)
    steps: list[RunStepRecord] = field(default_factory=list)
    error: str | None = None
