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
from typing import Any

from flywheel.artifact import BlockExecution, BlockInvocation
from flywheel.artifact_validator import ArtifactValidatorRegistry
from flywheel.pattern_params import substitute_params
from flywheel.sequence import RunContext
from flywheel.state_validator import StateValidatorRegistry
from flywheel.template import (
    BlockDefinition,
    InvocationDeclaration,
    Template,
)
from flywheel.workspace import Workspace

DEFAULT_MAX_INVOCATION_DEPTH = 8
"""Maximum committed executions allowed in one invocation chain.

The top-level block counts as parent #1 once it commits and begins
dispatching routes, so the default allows eight committed executions
and rejects the ninth launch.
"""


@dataclass(frozen=True)
class InvocationStep:
    """One committed execution in an invocation dispatch chain."""

    block_name: str
    termination_reason: str
    execution_id: str


@dataclass(frozen=True)
class InvocationChain:
    """Runtime guard state for recursive termination-route dispatch."""

    steps: tuple[InvocationStep, ...] = ()

    @classmethod
    def empty(cls) -> InvocationChain:
        """Return an empty invocation chain."""
        return cls()

    @property
    def depth(self) -> int:
        """Number of committed executions in the chain."""
        return len(self.steps)

    def extend(
        self,
        *,
        block_name: str,
        termination_reason: str,
        execution_id: str,
    ) -> InvocationChain:
        """Return a new chain with one committed execution appended."""
        return InvocationChain((
            *self.steps,
            InvocationStep(
                block_name=block_name,
                termination_reason=termination_reason,
                execution_id=execution_id,
            ),
        ))

    def contains_block(self, block_name: str) -> bool:
        """Return whether the chain already contains block_name.

        This is a runtime failsafe on top of static template cycle
        validation. It rejects same-block re-entry even through a
        different termination reason; relax this to a narrower key if
        intentional re-entry becomes a supported composition pattern.
        """
        return any(step.block_name == block_name for step in self.steps)

    def describe(self, *, next_block: str | None = None) -> str:
        """Return a human-readable chain path for diagnostics."""
        parts = [
            (
                f"{step.block_name}({step.execution_id})"
                f" --{step.termination_reason}-->"
            )
            for step in self.steps
        ]
        if next_block is not None:
            parts.append(next_block)
        return " ".join(parts) if parts else (next_block or "<empty>")


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
        if binding.parent_input is not None:
            artifact_id = parent.input_bindings.get(binding.parent_input)
            if artifact_id is None:
                raise ValueError(
                    f"Parent execution {parent.id!r} did not bind "
                    f"input {binding.parent_input!r}"
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
    params: dict[str, Any] | None = None,
    run_context: RunContext | None = None,
    invocation_chain: InvocationChain | None = None,
    max_invocation_depth: int = DEFAULT_MAX_INVOCATION_DEPTH,
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
    params = dict(params or {})
    run_context = run_context or RunContext.empty()
    invocation_chain = invocation_chain or InvocationChain.empty()
    for route in routes:
        invocation_id = workspace.generate_invocation_id()
        input_bindings: dict[str, str] = {}
        route_args = list(route.args)
        try:
            route_args = [
                substitute_params(arg, params) for arg in route.args
            ]
            if invocation_chain.depth >= max_invocation_depth:
                raise RuntimeError(
                    "invocation depth limit "
                    f"({max_invocation_depth}) exceeded for chain "
                    f"{invocation_chain.describe(next_block=route.block)}"
                )
            if invocation_chain.contains_block(route.block):
                raise RuntimeError(
                    "recursive invocation cycle detected for chain "
                    f"{invocation_chain.describe(next_block=route.block)}"
                )
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
                args=route_args,
                validator_registry=validator_registry,
                state_validator_registry=state_validator_registry,
                invoking_execution_id=parent_execution.id,
                dispatch_child_invocations=True,
                run_context=run_context,
                invocation_chain=invocation_chain,
                max_invocation_depth=max_invocation_depth,
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
                args=list(route_args),
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
                args=list(route_args),
                error=f"{type(exc).__name__}: {exc}",
            )
            workspace.record_invocation(invocation)
            results.append(InvocationDispatchResult(
                invocation=invocation,
                exception=exc,
            ))
    return results
