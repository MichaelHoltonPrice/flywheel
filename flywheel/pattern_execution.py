"""Pattern execution over canonical block execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from flywheel.artifact_validator import ArtifactValidatorRegistry
from flywheel.bindings import ArtifactIdBinding
from flywheel.execution import run_block
from flywheel.pattern_declaration import (
    PatternDeclaration,
    PatternMember,
    PatternStep,
    PriorOutputBinding,
)
from flywheel.pattern_resolution import (
    RunPrefix,
    StopDecision,
    WorkspaceView,
    resolve_next_step,
)
from flywheel.run_record import RunMemberRecord, RunStepRecord
from flywheel.state import pattern_state_lineage_key
from flywheel.state_validator import StateValidatorRegistry
from flywheel.template import Template
from flywheel.workspace import Workspace


class PatternRunError(RuntimeError):
    """Raised when a pattern run cannot complete successfully."""


@dataclass(frozen=True)
class PatternRunResult:
    """Result of a completed pattern run."""

    run_id: str
    status: Literal["succeeded", "failed"]


def run_pattern(
    workspace: Workspace,
    pattern: PatternDeclaration,
    template: Template,
    project_root: Path,
    *,
    validator_registry: ArtifactValidatorRegistry | None = None,
    state_validator_registry: StateValidatorRegistry | None = None,
) -> PatternRunResult:
    """Execute a pattern through canonical block execution."""
    if template.name != workspace.template_name:
        raise ValueError(
            f"Template {template.name!r} does not match workspace "
            f"template {workspace.template_name!r}"
        )

    run = workspace.begin_run(kind=f"pattern:{pattern.name}")
    try:
        while True:
            current_run = workspace.runs[run.id]
            decision = resolve_next_step(
                pattern,
                RunPrefix.from_run(current_run),
                WorkspaceView.from_workspace(workspace),
            )
            if isinstance(decision, StopDecision):
                workspace.end_run(
                    run.id,
                    decision.status,
                    error=decision.error,
                )
                if decision.status == "failed":
                    raise PatternRunError(
                        decision.error or "pattern run failed"
                    )
                return PatternRunResult(
                    run_id=run.id,
                    status=decision.status,
                )

            step_result = _execute_step(
                workspace,
                decision.step,
                template,
                project_root,
                run_id=run.id,
                validator_registry=validator_registry,
                state_validator_registry=state_validator_registry,
            )
            workspace.record_run_step(run.id, step_result)
    except KeyboardInterrupt:
        _close_running_run(
            workspace, run.id, "interrupted", error="operator interrupt")
        raise
    except Exception as exc:
        _close_running_run(
            workspace, run.id, "failed", error=str(exc))
        raise


def _execute_step(
    workspace: Workspace,
    step: PatternStep,
    template: Template,
    project_root: Path,
    *,
    run_id: str,
    validator_registry: ArtifactValidatorRegistry | None,
    state_validator_registry: StateValidatorRegistry | None,
) -> RunStepRecord:
    members: list[RunMemberRecord] = []
    min_successes = step.cohort.min_successes
    for member in step.cohort.members:
        result = _execute_member(
            workspace,
            member,
            template,
            project_root,
            run_id=run_id,
            step_name=step.name,
            validator_registry=validator_registry,
            state_validator_registry=state_validator_registry,
        )
        members.append(result)
        if min_successes == "all" and result.status != "succeeded":
            members.extend(
                RunMemberRecord(
                    name=remaining.name,
                    block_name=remaining.block,
                    status="skipped",
                    error="not launched after prior member failed",
                )
                for remaining in step.cohort.members[len(members):]
            )
            break

    status = _cohort_status(min_successes, members)
    return RunStepRecord(
        name=step.name,
        min_successes=min_successes,
        status=status,
        members=members,
    )


def _execute_member(
    workspace: Workspace,
    member: PatternMember,
    template: Template,
    project_root: Path,
    *,
    run_id: str,
    step_name: str,
    validator_registry: ArtifactValidatorRegistry | None,
    state_validator_registry: StateValidatorRegistry | None,
) -> RunMemberRecord:
    try:
        input_bindings = _resolve_member_inputs(workspace, member, run_id)
        result = run_block(
            workspace,
            member.block,
            template,
            project_root,
            input_bindings=input_bindings,
            args=member.args,
            validator_registry=validator_registry,
            state_validator_registry=state_validator_registry,
            state_lineage_key=pattern_state_lineage_key(
                run_id, step_name, member.name),
        )
    except Exception as exc:
        execution_id = getattr(exc, "flywheel_execution_id", None)
        execution = (
            workspace.executions.get(execution_id)
            if execution_id is not None else None
        )
        return RunMemberRecord(
            name=member.name,
            block_name=member.block,
            status="failed",
            execution_id=execution_id,
            output_bindings=(
                dict(execution.output_bindings)
                if execution is not None else {}
            ),
            error=str(exc),
        )

    execution = result.execution
    return RunMemberRecord(
        name=member.name,
        block_name=member.block,
        status=(
            "succeeded"
            if execution.status == "succeeded" else "failed"
        ),
        execution_id=result.execution_id,
        output_bindings=dict(execution.output_bindings),
        error=execution.error,
    )


def _resolve_member_inputs(
    workspace: Workspace,
    member: PatternMember,
    run_id: str,
) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for slot, source in member.inputs.items():
        if isinstance(source, ArtifactIdBinding):
            bindings[slot] = source.artifact_id
            continue
        if isinstance(source, PriorOutputBinding):
            bindings[slot] = _resolve_prior_output(workspace, run_id, source)
            continue
        raise TypeError(f"unknown input binding {source!r}")
    return bindings


def _resolve_prior_output(
    workspace: Workspace,
    run_id: str,
    source: PriorOutputBinding,
) -> str:
    run = workspace.runs.get(run_id)
    if run is None:
        raise PatternRunError(f"run {run_id!r} is not known")
    for step in run.steps:
        if step.name != source.from_step:
            continue
        for member in step.members:
            if member.name != source.member:
                continue
            if member.status != "succeeded":
                raise PatternRunError(
                    f"member {source.member!r} in step "
                    f"{source.from_step!r} did not succeed"
                )
            artifact_id = member.output_bindings.get(source.output)
            if artifact_id is None:
                raise PatternRunError(
                    f"member {source.member!r} in step "
                    f"{source.from_step!r} has no output "
                    f"{source.output!r}"
                )
            return artifact_id
    raise PatternRunError(
        f"no member {source.member!r} in step {source.from_step!r}"
    )


def _cohort_status(
    min_successes: int | Literal["all"],
    members: list[RunMemberRecord],
) -> Literal["succeeded", "failed"]:
    successes = sum(1 for member in members if member.status == "succeeded")
    if min_successes == "all":
        return (
            "succeeded"
            if successes == len(members) and all(
                member.status == "succeeded" for member in members
            )
            else "failed"
        )
    return "succeeded" if successes >= min_successes else "failed"


def _close_running_run(
    workspace: Workspace,
    run_id: str,
    status: Literal["failed", "interrupted"],
    *,
    error: str,
) -> None:
    run = workspace.runs.get(run_id)
    if run is not None and run.status == "running":
        workspace.end_run(run_id, status, error=error)
