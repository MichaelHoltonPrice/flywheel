"""Shared block-executor protocol types.

The canonical ad hoc block-execution surface lives in
:mod:`flywheel.execution`.  This module contains protocol/result types
only; concrete container execution is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from flywheel.workspace import Workspace


@dataclass(frozen=True)
class ExecutionResult:
    """Result of a block execution through the executor protocol."""

    exit_code: int
    elapsed_s: float
    output_bindings: dict[str, str]
    execution_id: str
    status: str


@dataclass(frozen=True)
class ExecutionEvent:
    """Event payload emitted when an executor completes a block."""

    executor_type: str
    block_name: str
    execution_id: str
    status: str
    output_bindings: dict[str, str] = field(default_factory=dict)
    outputs_data: dict[str, Any] | None = None


class ExecutionHandle:
    """Handle to a running or completed block execution."""

    def is_alive(self) -> bool:
        """Check if the execution is still running."""
        raise NotImplementedError

    def stop(self, reason: str = "requested") -> None:
        """Request a graceful stop."""
        raise NotImplementedError

    def wait(self) -> ExecutionResult:
        """Block until completion and return the result."""
        raise NotImplementedError


class SyncExecutionHandle(ExecutionHandle):
    """Handle for an execution that completed synchronously."""

    def __init__(self, result: ExecutionResult):
        """Initialize the handle with its already-completed result."""
        self._result = result
        self._waited = False

    def is_alive(self) -> bool:
        """Always false because execution already completed."""
        return False

    def stop(self, reason: str = "requested") -> None:
        """No-op because execution already completed."""

    def wait(self) -> ExecutionResult:
        """Return the pre-computed result exactly once."""
        if self._waited:
            raise RuntimeError("wait() already called on this handle")
        self._waited = True
        return self._result


@runtime_checkable
class BlockExecutor(Protocol):
    """Protocol used by deferred pattern code to launch a block."""

    def launch(
        self,
        block_name: str,
        workspace: Workspace,
        input_bindings: dict[str, str],
        *,
        execution_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        allowed_blocks: list[str] | None = None,
        state_lineage_key: str | None = None,
        run_id: str | None = None,
    ) -> ExecutionHandle:
        """Launch a block execution and return a waitable handle."""
        ...
