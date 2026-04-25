"""Termination-route dispatch for block invocations.

An invocation is a durable consequence of one committed block
execution: the parent announced a project-defined termination
reason, that reason has a route in the block declaration, and the
substrate runs the declared child block through the same canonical
``run_block`` path as any other execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from flywheel.artifact import BlockExecution, BlockInvocation
from flywheel.artifact_validator import ArtifactValidatorRegistry
from flywheel.state_validator import StateValidatorRegistry
from flywheel.template import (
    BlockDefinition,
    InvocationDeclaration,
    Template,
)
from flywheel.workspace import Workspace


@dataclass(frozen=True)
class InvocationDispatchResult:
    """Result of dispatching one termination-route invocation."""

    invocation: BlockInvocation
    exception: Exception | None = None


def _resolve_child_bindings(
    *,
    parent: BlockExecution,
    route: InvocationDeclaration,
    workspace: Workspace,
) -> dict[str, str]:
    """Resolve route bindings into concrete child input artifact ids."""
    bindings: dict[str, str] = {}
    for child_input, binding in route.bind.items():
        if binding.parent_output is not None:
            artifact_id = parent.output_bindings.get(binding.parent_output)
            if artifact_id is None:
                raise ValueError(
                    f"Parent execution {parent.id!r} did not commit "
                    f"output {binding.parent_output!r}"
                )
            bindings[child_input] = artifact_id
            continue
        if binding.artifact_id is not None:
            if binding.artifact_id not in workspace.artifacts:
                raise ValueError(
                    f"Invocation binding references unknown artifact "
                    f"{binding.artifact_id!r}"
                )
            bindings[child_input] = binding.artifact_id
    return bindings


def dispatch_invocations(
    *,
    workspace: Workspace,
    template: Template,
    project_root: Path,
    parent_block: BlockDefinition,
    parent_execution: BlockExecution,
    validator_registry: ArtifactValidatorRegistry | None = None,
    state_validator_registry: StateValidatorRegistry | None = None,
) -> list[InvocationDispatchResult]:
    """Run child blocks routed from a committed parent execution.

    Dispatch is intentionally host-side and post-commit.  It does
    not expose a live channel to the running parent container.
    """
    if parent_execution.status != "succeeded":
        return []
    if parent_execution.termination_reason is None:
        return []

    routes = parent_block.on_termination.get(parent_execution.termination_reason)
    if not routes:
        return []

    from flywheel.execution import run_block  # noqa: PLC0415

    results: list[InvocationDispatchResult] = []
    for route in routes:
        invocation_id = workspace.generate_invocation_id()
        input_bindings: dict[str, str] = {}
        try:
            input_bindings = _resolve_child_bindings(
                parent=parent_execution,
                route=route,
                workspace=workspace,
            )
            child_result = run_block(
                workspace,
                route.block,
                template,
                project_root,
                input_bindings=input_bindings,
                args=route.args,
                validator_registry=validator_registry,
                state_validator_registry=state_validator_registry,
                invoking_execution_id=parent_execution.id,
                dispatch_child_invocations=False,
            )
            invocation = BlockInvocation(
                id=invocation_id,
                invoking_execution_id=parent_execution.id,
                termination_reason=parent_execution.termination_reason,
                invoked_block_name=route.block,
                invoked_at=datetime.now(UTC),
                status="succeeded",
                invoked_execution_id=child_result.execution_id,
                input_bindings=dict(child_result.execution.input_bindings),
                args=list(route.args),
            )
            workspace.record_invocation(invocation)
            results.append(InvocationDispatchResult(invocation=invocation))
        except Exception as exc:
            child_id = getattr(exc, "flywheel_execution_id", None)
            if child_id is not None and child_id in workspace.executions:
                input_bindings = dict(
                    workspace.executions[child_id].input_bindings)
            invocation = BlockInvocation(
                id=invocation_id,
                invoking_execution_id=parent_execution.id,
                termination_reason=parent_execution.termination_reason,
                invoked_block_name=route.block,
                invoked_at=datetime.now(UTC),
                status="failed",
                invoked_execution_id=child_id,
                input_bindings=input_bindings,
                args=list(route.args),
                error=f"{type(exc).__name__}: {exc}",
            )
            workspace.record_invocation(invocation)
            results.append(InvocationDispatchResult(
                invocation=invocation,
                exception=exc,
            ))
    return results
