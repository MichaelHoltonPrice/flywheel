"""Generic parallel block execution.

Launches multiple block executions simultaneously via any
``BlockExecutor`` and collects results sequentially.  This is
the generalized version of ``AgentGroup`` — it works with any
executor type, not just agent containers.

Usage::

    from flywheel.executor import ContainerExecutor, RecordExecutor
    from flywheel.block_group import BlockGroup, BlockGroupMember

    group = BlockGroup(workspace, executor)
    group.add(BlockGroupMember(block_name="eval", ...))
    group.add(BlockGroupMember(block_name="eval", ...))
    results = group.run()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from flywheel.artifact import LifecycleEvent
from flywheel.executor import BlockExecutor, ExecutionHandle, ExecutionResult
from flywheel.workspace import Workspace


@dataclass
class BlockGroupMember:
    """Configuration for one member of a block group.

    Attributes:
        block_name: The block to execute.
        input_bindings: Input artifact bindings for this member.
        kwargs: Additional keyword arguments passed to the
            executor's ``launch()`` method.
    """

    block_name: str
    input_bindings: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockGroupResult:
    """Result of a single block execution in a group.

    Attributes:
        index: Position in the group (0-based).
        result: The ``ExecutionResult`` from ``wait()``.
        block_name: The block that was executed.
    """

    index: int
    result: ExecutionResult
    block_name: str


class BlockGroup:
    """Launch and manage a group of parallel block executions.

    All executions share the same flywheel workspace.  They are
    launched simultaneously and waited on sequentially to serialize
    workspace writes.

    Args:
        workspace: The flywheel workspace.
        executor: The block executor to use for all members.
    """

    def __init__(
        self,
        workspace: Workspace,
        executor: BlockExecutor,
    ):
        """Initialize with workspace and executor."""
        self._workspace = workspace
        self._executor = executor
        self._members: list[BlockGroupMember] = []

    def add(self, member: BlockGroupMember) -> None:
        """Add a member to the group."""
        self._members.append(member)

    def run(self) -> list[BlockGroupResult]:
        """Launch all members and wait sequentially.

        Returns:
            List of ``BlockGroupResult``, one per member, in order.
        """
        if not self._members:
            return []

        # Phase 1: launch all.
        handles: list[tuple[int, BlockGroupMember, ExecutionHandle]] = []
        for i, member in enumerate(self._members):
            handle = self._executor.launch(
                block_name=member.block_name,
                workspace=self._workspace,
                input_bindings=member.input_bindings,
                **member.kwargs,
            )
            handles.append((i, member, handle))

        # Phase 2: wait sequentially.
        results: list[BlockGroupResult] = []
        for i, member, handle in handles:
            result = handle.wait()
            results.append(BlockGroupResult(
                index=i,
                result=result,
                block_name=member.block_name,
            ))

        # Record lifecycle event.
        event = LifecycleEvent(
            id=self._workspace.generate_event_id(),
            kind="block_group_completed",
            timestamp=datetime.now(UTC),
            detail={
                "members": str(len(self._members)),
                "succeeded": str(
                    sum(1 for r in results
                        if r.result.status == "succeeded")),
            },
        )
        self._workspace.add_event(event)
        self._workspace.save()

        return results
