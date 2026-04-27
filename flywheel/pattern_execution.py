"""Pattern execution over canonical block execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from flywheel.artifact import ArtifactInstance
from flywheel.artifact_validator import ArtifactValidatorRegistry
from flywheel.bindings import ArtifactIdBinding
from flywheel.execution import run_block
from flywheel.pattern_declaration import (
    PatternAfterEvery,
    PatternDeclaration,
    PatternForEach,
    PatternMember,
    PatternNode,
    PatternRunUntil,
    PatternStep,
    PatternUse,
    PriorOutputBinding,
)
from flywheel.pattern_lanes import DEFAULT_LANE
from flywheel.pattern_params import (
    PatternParamError,
    coerce_param_value,
    referenced_params,
    substitute_params,
)
from flywheel.pattern_resolution import (
    RunPrefix,
    StopDecision,
    WorkspaceView,
    resolve_next_step,
)
from flywheel.run_record import (
    RunFixtureRecord,
    RunMemberRecord,
    RunStepRecord,
)
from flywheel.sequence import RunContext
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
    param_overrides: dict[str, str] | None = None,
) -> PatternRunResult:
    """Execute a pattern through canonical block execution."""
    if template.name != workspace.template_name:
        raise ValueError(
            f"Template {template.name!r} does not match workspace "
            f"template {workspace.template_name!r}"
        )

    _validate_pattern_fixtures(pattern, template)
    _validate_pattern_param_references(pattern, template)
    params = resolve_pattern_params(pattern, param_overrides or {})
    run = workspace.begin_run(
        kind=f"pattern:{pattern.name}",
        params=params,
        lanes=list(pattern.lanes),
    )
    try:
        _materialize_pattern_fixtures(
            workspace,
            pattern,
            template,
            project_root,
            run_id=run.id,
            validator_registry=validator_registry,
        )
        if pattern.body:
            _execute_body(
                workspace,
                pattern.body,
                pattern,
                template,
                project_root,
                run_id=run.id,
                lane_name=(
                    pattern.lanes[0]
                    if pattern.lanes == [DEFAULT_LANE]
                    else DEFAULT_LANE
                ),
                step_prefix=[],
                validator_registry=validator_registry,
                state_validator_registry=state_validator_registry,
                params=params,
            )
            workspace.end_run(run.id, "succeeded")
            return PatternRunResult(run_id=run.id, status="succeeded")
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
                params=params,
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


def _execute_body(
    workspace: Workspace,
    nodes: list[PatternNode],
    pattern: PatternDeclaration,
    template: Template,
    project_root: Path,
    *,
    run_id: str,
    lane_name: str,
    step_prefix: list[str],
    validator_registry: ArtifactValidatorRegistry | None,
    state_validator_registry: StateValidatorRegistry | None,
    params: dict[str, object],
) -> None:
    for index, node in enumerate(nodes):
        if isinstance(node, PatternForEach):
            _execute_foreach(
                workspace,
                node,
                pattern,
                template,
                project_root,
                run_id=run_id,
                step_prefix=[*step_prefix, f"foreach_{index + 1}"],
                validator_registry=validator_registry,
                state_validator_registry=state_validator_registry,
                params=params,
            )
            continue
        if isinstance(node, PatternUse):
            _execute_use(
                workspace,
                node,
                pattern,
                template,
                project_root,
                run_id=run_id,
                lane_name=lane_name,
                step_prefix=[*step_prefix, node.pattern],
                validator_registry=validator_registry,
                state_validator_registry=state_validator_registry,
                params=params,
            )
            continue
        if isinstance(node, PatternRunUntil):
            _execute_run_until(
                workspace,
                node,
                pattern,
                template,
                project_root,
                run_id=run_id,
                lane_name=lane_name,
                step_prefix=step_prefix,
                validator_registry=validator_registry,
                state_validator_registry=state_validator_registry,
                params=params,
            )
            continue
        if isinstance(node, PatternStep):
            step_name = _generated_step_name([*step_prefix, node.name])
            step = PatternStep(name=step_name, cohort=node.cohort)
            step_result = _execute_step(
                workspace,
                step,
                template,
                project_root,
                run_id=run_id,
                lane_override=lane_name,
                validator_registry=validator_registry,
                state_validator_registry=state_validator_registry,
                params=params,
            )
            workspace.record_run_step(run_id, step_result)
            if step_result.status == "failed":
                raise PatternRunError(f"step {node.name!r} failed")
            continue
        raise TypeError(f"unknown pattern node {node!r}")


def _execute_foreach(
    workspace: Workspace,
    node: PatternForEach,
    pattern: PatternDeclaration,
    template: Template,
    project_root: Path,
    *,
    run_id: str,
    step_prefix: list[str],
    validator_registry: ArtifactValidatorRegistry | None,
    state_validator_registry: StateValidatorRegistry | None,
    params: dict[str, object],
) -> None:
    for lane_name in node.lanes:
        _execute_body(
            workspace,
            node.body,
            pattern,
            template,
            project_root,
            run_id=run_id,
            lane_name=lane_name,
            step_prefix=[*step_prefix, lane_name],
            validator_registry=validator_registry,
            state_validator_registry=state_validator_registry,
            params=params,
        )


def _execute_use(
    workspace: Workspace,
    node: PatternUse,
    pattern: PatternDeclaration,
    template: Template,
    project_root: Path,
    *,
    run_id: str,
    lane_name: str,
    step_prefix: list[str],
    validator_registry: ArtifactValidatorRegistry | None,
    state_validator_registry: StateValidatorRegistry | None,
    params: dict[str, object],
) -> None:
    subpattern = pattern.patterns.get(node.pattern)
    if subpattern is None:
        raise PatternRunError(
            f"pattern {pattern.name!r} uses unknown local pattern "
            f"{node.pattern!r}"
        )
    provided = {
        key: (
            substitute_params(value, params)
            if isinstance(value, str) else value
        )
        for key, value in node.params.items()
    }
    child_params = resolve_pattern_params(
        subpattern,
        {key: str(value) for key, value in provided.items()},
    )
    _execute_body(
        workspace,
        subpattern.body,
        subpattern,
        template,
        project_root,
        run_id=run_id,
        lane_name=lane_name,
        step_prefix=step_prefix,
        validator_registry=validator_registry,
        state_validator_registry=state_validator_registry,
        params=child_params,
    )


def _execute_run_until(
    workspace: Workspace,
    node: PatternRunUntil,
    pattern: PatternDeclaration,
    template: Template,
    project_root: Path,
    *,
    run_id: str,
    lane_name: str,
    step_prefix: list[str],
    validator_registry: ArtifactValidatorRegistry | None,
    state_validator_registry: StateValidatorRegistry | None,
    params: dict[str, object],
) -> None:
    budgets = {
        reason: _resolve_positive_int(
            budget.max,
            params,
            context=(
                f"run_until {node.name!r} continue_on[{reason!r}].max"
            ),
        )
        for reason, budget in node.continue_on.items()
    }
    reason_counts = {reason: 0 for reason in budgets}
    after_every = [
        (
            trigger,
            _resolve_positive_int(
                trigger.count,
                params,
                context=(
                    f"run_until {node.name!r} "
                    f"after_every[{trigger.reason!r}].count"
                ),
            ),
        )
        for trigger in node.after_every
    ]
    iteration = 0
    step_name = _generated_step_name([*step_prefix, node.name])
    step: RunStepRecord | None = None
    while True:
        iteration += 1
        member = PatternMember(
            name=f"iter_{iteration}",
            block=node.block,
            lane=lane_name,
            inputs=dict(node.inputs),
            args=list(node.args),
            env=dict(node.env),
        )
        result = _execute_member(
            workspace,
            member,
            template,
            project_root,
            run_id=run_id,
            step_name=step_name,
            lane_name=lane_name,
            validator_registry=validator_registry,
            state_validator_registry=state_validator_registry,
            params=params,
        )
        members = [*step.members, result] if step is not None else [result]
        step = RunStepRecord(
            name=step_name,
            min_successes=1,
            status=(
                "succeeded"
                if result.status == "succeeded" else "failed"
            ),
            members=members,
            kind="run_until",
            reason_counts=dict(reason_counts),
        )
        if iteration == 1:
            workspace.record_run_step(run_id, step)
        else:
            workspace.replace_run_step(run_id, step)
        if result.status != "succeeded":
            step = _finish_run_until_step(
                workspace,
                run_id,
                step,
                status="failed",
                stop_kind="failed",
                terminal_reason=None,
                reason_counts=reason_counts,
            )
            raise PatternRunError(
                result.error or f"run_until {node.name!r} failed"
            )
        assert result.execution_id is not None
        execution = workspace.executions[result.execution_id]
        reason = execution.termination_reason
        if reason in node.stop_on:
            _finish_run_until_step(
                workspace,
                run_id,
                step,
                status="succeeded",
                stop_kind="stop_on",
                terminal_reason=reason,
                reason_counts=reason_counts,
            )
            return
        if reason in budgets:
            reason_counts[reason] += 1
            step = _finish_run_until_step(
                workspace,
                run_id,
                step,
                status="succeeded",
                stop_kind=None,
                terminal_reason=None,
                reason_counts=reason_counts,
            )
            _execute_due_after_every(
                workspace,
                after_every,
                pattern_node=node,
                pattern=pattern,
                template=template,
                project_root=project_root,
                run_id=run_id,
                lane_name=lane_name,
                step_prefix=step_prefix,
                validator_registry=validator_registry,
                state_validator_registry=state_validator_registry,
                params=params,
                reason=reason,
                reason_count=reason_counts[reason],
            )
            if reason_counts[reason] >= budgets[reason]:
                _finish_run_until_step(
                    workspace,
                    run_id,
                    step,
                    status="succeeded",
                    stop_kind="budget_exhausted",
                    terminal_reason=reason,
                    reason_counts=reason_counts,
                )
                return
            continue
        _finish_run_until_step(
            workspace,
            run_id,
            step,
            status="failed",
            stop_kind="unexpected_reason",
            terminal_reason=reason,
            reason_counts=reason_counts,
        )
        raise PatternRunError(
            f"run_until {node.name!r} received unexpected "
            f"termination reason {reason!r}"
        )


def _finish_run_until_step(
    workspace: Workspace,
    run_id: str,
    step: RunStepRecord,
    *,
    status: Literal["succeeded", "failed"],
    stop_kind: str | None,
    terminal_reason: str | None,
    reason_counts: dict[str, int],
) -> RunStepRecord:
    updated = RunStepRecord(
        name=step.name,
        min_successes=step.min_successes,
        status=status,
        members=list(step.members),
        kind=step.kind,
        terminal_reason=terminal_reason,
        stop_kind=stop_kind,
        reason_counts=dict(reason_counts),
    )
    workspace.replace_run_step(run_id, updated)
    return updated


def _execute_due_after_every(
    workspace: Workspace,
    after_every: list[tuple[PatternAfterEvery, int]],
    *,
    pattern_node: PatternRunUntil,
    pattern: PatternDeclaration,
    template: Template,
    project_root: Path,
    run_id: str,
    lane_name: str,
    step_prefix: list[str],
    validator_registry: ArtifactValidatorRegistry | None,
    state_validator_registry: StateValidatorRegistry | None,
    params: dict[str, object],
    reason: str,
    reason_count: int,
) -> None:
    for trigger, count in after_every:
        if trigger.reason != reason:
            continue
        if reason_count % count != 0:
            continue
        occurrence = reason_count // count
        _execute_body(
            workspace,
            trigger.body,
            pattern,
            template,
            project_root,
            run_id=run_id,
            lane_name=lane_name,
            step_prefix=[
                *step_prefix,
                pattern_node.name,
                f"after_{reason}_{count}_{occurrence}",
            ],
            validator_registry=validator_registry,
            state_validator_registry=state_validator_registry,
            params=params,
        )


def _resolve_positive_int(
    value: int | str,
    params: dict[str, object],
    *,
    context: str,
) -> int:
    rendered = substitute_params(value, params) if isinstance(value, str) else value
    try:
        parsed = int(rendered)
    except (TypeError, ValueError) as exc:
        raise PatternParamError(f"{context} must resolve to an integer") from exc
    if parsed < 1:
        raise PatternParamError(f"{context} must be positive")
    return parsed


def _generated_step_name(parts: list[str]) -> str:
    cleaned = [
        "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in part)
        for part in parts
        if part
    ]
    return "__".join(cleaned)


def _execute_step(
    workspace: Workspace,
    step: PatternStep,
    template: Template,
    project_root: Path,
    *,
    run_id: str,
    lane_override: str | None = None,
    validator_registry: ArtifactValidatorRegistry | None,
    state_validator_registry: StateValidatorRegistry | None,
    params: dict[str, object],
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
            lane_name=lane_override or member.lane,
            validator_registry=validator_registry,
            state_validator_registry=state_validator_registry,
            params=params,
        )
        members.append(result)
        if min_successes == "all" and result.status != "succeeded":
            members.extend(
                RunMemberRecord(
                    name=remaining.name,
                    block_name=remaining.block,
                    status="skipped",
                    lane=lane_override or remaining.lane,
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
    lane_name: str,
    validator_registry: ArtifactValidatorRegistry | None,
    state_validator_registry: StateValidatorRegistry | None,
    params: dict[str, object],
) -> RunMemberRecord:
    try:
        input_bindings = _resolve_member_inputs(
            workspace,
            member,
            run_id,
            lane_name,
            template,
        )
        args = [substitute_params(arg, params) for arg in member.args]
        env_overlay = {
            key: substitute_params(value, params)
            for key, value in member.env.items()
        }
        resume_prompt = _build_resume_prompt(
            workspace,
            template,
            member.block,
            input_bindings,
        )
        if resume_prompt is not None:
            env_overlay.setdefault("FLYWHEEL_RESUME_PROMPT", resume_prompt)
        result = run_block(
            workspace,
            member.block,
            template,
            project_root,
            input_bindings=input_bindings,
            args=args,
            validator_registry=validator_registry,
            state_validator_registry=state_validator_registry,
            state_lineage_key=pattern_state_lineage_key(
                run_id, lane_name, member.block),
            allow_workspace_latest=False,
            env_overlay=env_overlay,
            invocation_params=params,
            run_context=RunContext(run_id=run_id, lane=lane_name),
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
            lane=lane_name,
            execution_id=execution_id,
            output_bindings=(
                dict(execution.output_bindings)
                if execution is not None else {}
            ),
            error=str(exc),
        )

    execution = result.execution
    invocation_ids = _invocation_ids_for_parent(
        workspace, result.execution_id)
    return RunMemberRecord(
        name=member.name,
        block_name=member.block,
        status=(
            "succeeded"
            if execution.status == "succeeded" else "failed"
        ),
        lane=lane_name,
        execution_id=result.execution_id,
        output_bindings=dict(execution.output_bindings),
        invocation_ids=invocation_ids,
        error=execution.error,
    )


def resolve_pattern_params(
    pattern: PatternDeclaration,
    overrides: dict[str, str],
) -> dict[str, object]:
    """Resolve operator overrides plus defaults for one pattern run."""
    unknown = set(overrides) - set(pattern.params)
    if unknown:
        raise PatternParamError(
            f"Pattern {pattern.name!r}: unknown param(s) "
            f"{sorted(unknown)!r}"
        )
    resolved: dict[str, object] = {}
    for name, param in pattern.params.items():
        if name in overrides:
            resolved[name] = _coerce_param_override(
                pattern.name, param, overrides[name])
        elif param.default is not None:
            resolved[name] = param.default
        else:
            raise PatternParamError(
                f"Pattern {pattern.name!r}: required param "
                f"{name!r} was not supplied"
            )
    return resolved


def _coerce_param_override(
    pattern_name: str,
    param,
    value: str,
) -> object:
    return coerce_param_value(
        pattern_name=pattern_name,
        name=param.name,
        value=value,
        param_type=param.type,
        source="value",
    )


def _validate_pattern_param_references(
    pattern: PatternDeclaration,
    template: Template,
) -> None:
    def validate_one(current: PatternDeclaration) -> None:
        declared = set(current.params)

        def check(value: str, context: str) -> None:
            unknown = referenced_params(value) - declared
            if unknown:
                raise PatternParamError(
                    f"Pattern {current.name!r}: {context} references "
                    f"unknown param(s) {sorted(unknown)!r}"
                )

        used_blocks: set[str] = set()
        if current.body:
            _validate_body_uses(current.body, current.patterns)
            _collect_body_param_references(
                current.body,
                check=check,
                used_blocks=used_blocks,
            )
            _validate_body_run_until_reasons(current.body, template)
        for step in current.steps:
            for member in step.cohort.members:
                used_blocks.add(member.block)
                _check_member_param_references(member, check=check)

        for block_name in sorted(used_blocks):
            block = _block_definition(template, block_name)
            for reason, routes in block.on_termination.items():
                for route in routes:
                    for arg in route.args:
                        check(
                            arg,
                            (
                                f"block {block.name!r} "
                                f"on_termination[{reason!r}] arg"
                            ),
                        )

    def visit(current: PatternDeclaration) -> None:
        validate_one(current)
        for subpattern in current.patterns.values():
            visit(subpattern)

    visit(pattern)


def _check_member_param_references(
    member: PatternMember,
    *,
    check,
) -> None:
    for arg in member.args:
        check(arg, f"member {member.name!r} arg")
    for key, value in member.env.items():
        check(value, f"member {member.name!r} env {key!r}")


def _collect_body_param_references(
    nodes: list[PatternNode],
    *,
    check,
    used_blocks: set[str],
) -> None:
    for node in nodes:
        if isinstance(node, PatternForEach):
            _collect_body_param_references(
                node.body,
                check=check,
                used_blocks=used_blocks,
            )
            continue
        if isinstance(node, PatternUse):
            for key, value in node.params.items():
                if isinstance(value, str):
                    check(value, f"use {node.pattern!r} param {key!r}")
            continue
        if isinstance(node, PatternRunUntil):
            used_blocks.add(node.block)
            member = PatternMember(
                name=node.name,
                block=node.block,
                inputs=dict(node.inputs),
                args=list(node.args),
                env=dict(node.env),
            )
            _check_member_param_references(member, check=check)
            for reason, budget in node.continue_on.items():
                if isinstance(budget.max, str):
                    check(
                        budget.max,
                        (
                            f"run_until {node.name!r} "
                            f"continue_on[{reason!r}].max"
                        ),
                    )
            for trigger in node.after_every:
                if isinstance(trigger.count, str):
                    check(
                        trigger.count,
                        (
                            f"run_until {node.name!r} "
                            f"after_every[{trigger.reason!r}].count"
                        ),
                    )
                _collect_body_param_references(
                    trigger.body,
                    check=check,
                    used_blocks=used_blocks,
                )
            continue
        if isinstance(node, PatternStep):
            for member in node.cohort.members:
                used_blocks.add(member.block)
                _check_member_param_references(member, check=check)
            continue
        raise TypeError(f"unknown pattern node {node!r}")


def _validate_body_uses(
    nodes: list[PatternNode],
    patterns: dict[str, PatternDeclaration],
) -> None:
    for node in nodes:
        if isinstance(node, PatternForEach):
            _validate_body_uses(node.body, patterns)
        elif isinstance(node, PatternRunUntil):
            for trigger in node.after_every:
                _validate_body_uses(trigger.body, patterns)
        elif isinstance(node, PatternUse):
            subpattern = patterns.get(node.pattern)
            if subpattern is None:
                raise PatternRunError(
                    f"use references unknown local pattern {node.pattern!r}"
                )
            unknown = set(node.params) - set(subpattern.params)
            if unknown:
                raise PatternParamError(
                    f"use {node.pattern!r} supplies unknown param(s) "
                    f"{sorted(unknown)!r}"
                )
            missing = {
                name
                for name, param in subpattern.params.items()
                if param.default is None and name not in node.params
            }
            if missing:
                raise PatternParamError(
                    f"use {node.pattern!r} is missing required param(s) "
                    f"{sorted(missing)!r}"
                )


def _validate_body_run_until_reasons(
    nodes: list[PatternNode],
    template: Template,
) -> None:
    for node in nodes:
        if isinstance(node, PatternForEach):
            _validate_body_run_until_reasons(node.body, template)
            continue
        if isinstance(node, PatternRunUntil):
            block = _block_definition(template, node.block)
            declared = set(block.outputs)
            required = set(node.continue_on) | set(node.stop_on)
            missing = required - declared
            if missing:
                raise PatternRunError(
                    f"run_until {node.name!r} references termination "
                    f"reason(s) {sorted(missing)!r} not declared by "
                    f"block {node.block!r}"
                )
            overlap = set(node.continue_on) & set(node.stop_on)
            if overlap:
                raise PatternRunError(
                    f"run_until {node.name!r} lists reason(s) "
                    f"{sorted(overlap)!r} in both continue_on and stop_on"
                )
            after_reasons = {trigger.reason for trigger in node.after_every}
            missing_after = after_reasons - set(node.continue_on)
            if missing_after:
                raise PatternRunError(
                    f"run_until {node.name!r} after_every references "
                    f"reason(s) {sorted(missing_after)!r} not listed "
                    "under continue_on"
                )
            for trigger in node.after_every:
                _validate_body_run_until_reasons(trigger.body, template)
            continue
        if isinstance(node, (PatternStep, PatternUse)):
            continue
        raise TypeError(f"unknown pattern node {node!r}")


def _resolve_member_inputs(
    workspace: Workspace,
    member: PatternMember,
    run_id: str,
    lane_name: str,
    template: Template,
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
    block = _block_definition(template, member.block)
    for slot in block.inputs:
        if slot.name in bindings:
            continue
        # Sequence-bound slots are resolved by execution.py at exec-prep
        # time via workspace.resolve_sequence_snapshot. Don't try to
        # resolve them as plain lane-scoped artifacts here.
        if slot.sequence is not None:
            continue
        decl = template.artifact_declaration(slot.name)
        if decl is not None and decl.kind == "git":
            continue
        artifact_id = _resolve_latest_in_lane(
            workspace, run_id, lane_name, slot.name)
        if artifact_id is not None:
            bindings[slot.name] = artifact_id
        elif not slot.optional:
            raise PatternRunError(
                f"required input {slot.name!r} for member "
                f"{member.name!r} has no artifact in lane "
                f"{lane_name!r}"
            )
    return bindings


def _block_definition(template: Template, block_name: str):
    for block in template.blocks:
        if block.name == block_name:
            return block
    raise PatternRunError(f"block {block_name!r} is not in template")


def _build_resume_prompt(
    workspace: Workspace,
    template: Template,
    block_name: str,
    input_bindings: dict[str, str],
) -> str | None:
    """Describe the concrete input artifacts mounted for a resumed agent.

    The Claude battery only consumes this when it is resuming an existing
    SDK session.  First executions ignore it, so it is safe to compute
    for every block launch.
    """
    if not input_bindings:
        return None
    block = _block_definition(template, block_name)
    slots = {slot.name: slot for slot in block.inputs}
    lines = [
        "Continue the original task.",
        "The previous external work requested by the session has completed.",
        "The following input artifacts are now mounted in this execution:",
    ]
    for slot_name in sorted(input_bindings):
        slot = slots.get(slot_name)
        if slot is None:
            continue
        artifact_id = input_bindings[slot_name]
        instance = workspace.artifacts.get(artifact_id)
        if instance is None:
            lines.append(
                f"- {slot_name}: {slot.container_path} "
                f"(artifact {artifact_id})"
            )
            continue
        files = _artifact_mounted_files(workspace, instance, slot.container_path)
        if files:
            lines.append(
                f"- {slot_name}: {slot.container_path} "
                f"(artifact {artifact_id}); files: {', '.join(files)}"
            )
        else:
            lines.append(
                f"- {slot_name}: {slot.container_path} "
                f"(artifact {artifact_id})"
            )
    lines.append("Read the relevant artifact result and continue.")
    return "\n".join(lines)


def _artifact_mounted_files(
    workspace: Workspace,
    instance: ArtifactInstance,
    container_path: str,
) -> list[str]:
    if instance.copy_path is None:
        return []
    root = workspace.path / "artifacts" / instance.copy_path
    if not root.is_dir():
        return []
    files: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        files.append(f"{container_path.rstrip('/')}/{rel}")
        if len(files) >= 10:
            files.append("...")
            break
    return files


def _resolve_latest_in_lane(
    workspace: Workspace,
    run_id: str,
    lane_name: str,
    artifact_name: str,
) -> str | None:
    run = workspace.runs.get(run_id)
    if run is None:
        raise PatternRunError(f"run {run_id!r} is not known")
    candidates: list[str] = []
    for fixture in run.fixtures:
        if fixture.lane == lane_name and fixture.name == artifact_name:
            candidates.append(fixture.artifact_id)
    for step in run.steps:
        for member in step.members:
            if member.lane != lane_name or member.status != "succeeded":
                continue
            artifact_id = member.output_bindings.get(artifact_name)
            if artifact_id is not None:
                candidates.append(artifact_id)
            for invocation_id in member.invocation_ids:
                invocation = workspace.invocations.get(invocation_id)
                if invocation is None or invocation.status != "succeeded":
                    continue
                child_id = invocation.invoked_execution_id
                if child_id is None:
                    continue
                child = workspace.executions.get(child_id)
                if child is None:
                    continue
                child_artifact_id = child.output_bindings.get(
                    artifact_name)
                if child_artifact_id is not None:
                    candidates.append(child_artifact_id)
    return candidates[-1] if candidates else None


def _invocation_ids_for_parent(
    workspace: Workspace,
    execution_id: str,
) -> list[str]:
    return [
        invocation.id
        for invocation in workspace.invocations.values()
        if invocation.invoking_execution_id == execution_id
    ]


def _validate_pattern_fixtures(
    pattern: PatternDeclaration,
    template: Template,
) -> None:
    for fixture in pattern.fixtures.values():
        declaration = template.artifact_declaration(fixture.name)
        if declaration is None:
            raise ValueError(
                f"Pattern {pattern.name!r} fixture {fixture.name!r} "
                "does not name a template artifact"
            )
        if declaration.kind != "copy":
            raise ValueError(
                f"Pattern {pattern.name!r} fixture {fixture.name!r} "
                f"targets {declaration.kind!r}; fixtures require "
                "copy artifacts"
            )


def _materialize_pattern_fixtures(
    workspace: Workspace,
    pattern: PatternDeclaration,
    template: Template,
    project_root: Path,
    *,
    run_id: str,
    validator_registry: ArtifactValidatorRegistry | None,
) -> None:
    for lane_name in pattern.lanes:
        for fixture in pattern.fixtures.values():
            declaration = template.artifact_declaration(fixture.name)
            assert declaration is not None
            source_path = (project_root / fixture.source).resolve()
            fixture_id = workspace.generate_run_fixture_id(run_id)
            source = (
                f"fixture:pattern:{pattern.name}:run:{run_id}:"
                f"lane:{lane_name}:artifact:{fixture.name}:"
                f"source:{fixture.source}"
            )
            instance: ArtifactInstance = workspace.register_artifact(
                fixture.name,
                source_path,
                source=source,
                validator_registry=validator_registry,
                declaration=declaration,
                fixture_id=fixture_id,
                persist=False,
            )
            workspace.record_run_fixture(
                run_id,
                RunFixtureRecord(
                    id=fixture_id,
                    lane=lane_name,
                    name=fixture.name,
                    artifact_id=instance.id,
                    source=fixture.source,
                ),
            )


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
