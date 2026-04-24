"""Pattern next-step resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from flywheel.pattern_declaration import PatternDeclaration, PatternStep
from flywheel.run_record import RunRecord, RunStepRecord
from flywheel.workspace import Workspace


@dataclass(frozen=True)
class WorkspaceView:
    """Read-only workspace facts used by the pattern resolver."""

    execution_statuses: dict[str, str]

    @classmethod
    def from_workspace(cls, workspace: Workspace) -> WorkspaceView:
        """Build a resolver view from a workspace."""
        return cls(
            execution_statuses={
                eid: execution.status
                for eid, execution in workspace.executions.items()
            }
        )


@dataclass(frozen=True)
class RunPrefix:
    """Read-only prefix of a run."""

    steps: tuple[RunStepRecord, ...]

    @classmethod
    def from_run(cls, run: RunRecord) -> RunPrefix:
        """Build a prefix view from a run record."""
        return cls(steps=tuple(run.steps))


@dataclass(frozen=True)
class StopDecision:
    """Resolver decision to stop the run."""

    status: Literal["succeeded", "failed"]
    error: str | None = None


@dataclass(frozen=True)
class RunCohortDecision:
    """Resolver decision to execute one pattern step."""

    step: PatternStep


PatternDecision = StopDecision | RunCohortDecision


def resolve_next_step(
    pattern: PatternDeclaration,
    run_prefix: RunPrefix,
    workspace_view: WorkspaceView,
) -> PatternDecision:
    """Resolve the next cohort to run from read-only inputs."""
    # The resolver contract includes workspace state.  This
    # declaration-order resolver only needs the run prefix.
    del workspace_view
    completed_by_name = {step.name: step for step in run_prefix.steps}
    for declared_step in pattern.steps:
        completed = completed_by_name.get(declared_step.name)
        if completed is None:
            return RunCohortDecision(step=declared_step)
        if completed.status == "failed":
            return StopDecision(
                status="failed",
                error=f"step {completed.name!r} failed",
            )
    return StopDecision(status="succeeded")
