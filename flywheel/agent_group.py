"""Backward-compatibility aliases for agent group execution.

Agent groups are now a usage pattern of :class:`BlockGroup` with
``launch_agent_block`` as the launch function.  This module
provides the old names so existing imports continue to work.

Prefer importing from ``flywheel.block_group`` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from flywheel.agent import launch_agent_block
from flywheel.block_group import BlockGroup, BlockGroupMember, BlockGroupResult
from flywheel.workspace import Workspace


@dataclass
class AgentGroupMember:
    """Legacy member config — converts to BlockGroupMember.

    Attributes:
        prompt: The system prompt for this agent.
        agent_workspace_dir: Distinct workspace subdirectory name.
        input_artifacts: Override input artifacts for this member.
        extra_env: Additional env vars merged on top of the base.
    """

    prompt: str
    agent_workspace_dir: str
    input_artifacts: dict[str, str] | None = None
    extra_env: dict[str, str] | None = None

    def to_block_member(self) -> BlockGroupMember:
        """Convert to a BlockGroupMember."""
        overrides: dict[str, Any] = {
            "prompt": self.prompt,
            "agent_workspace_dir": self.agent_workspace_dir,
        }
        if self.input_artifacts is not None:
            overrides["input_artifacts"] = self.input_artifacts
        return BlockGroupMember(
            overrides=overrides,
            merge_env=self.extra_env,
            output_dir=self.agent_workspace_dir,
        )


@dataclass
class AgentGroupResult:
    """Legacy result — wraps BlockGroupResult fields.

    Attributes:
        index: Position in the group (0-based).
        agent_result: The ``AgentResult`` from ``wait()``.
        agent_workspace_dir: The workspace subdirectory used.
        artifacts_collected: Artifact instance IDs registered.
    """

    index: int
    agent_result: Any
    agent_workspace_dir: str
    artifacts_collected: list[str] = field(default_factory=list)

    @classmethod
    def from_block_result(cls, r: BlockGroupResult) -> AgentGroupResult:
        """Convert from BlockGroupResult."""
        return cls(
            index=r.index,
            agent_result=r.result,
            agent_workspace_dir=r.output_dir or "",
            artifacts_collected=r.artifacts_collected,
        )


class AgentGroup:
    """Legacy wrapper — delegates to BlockGroup.

    Prefer using ``BlockGroup`` with ``launch_agent_block``
    directly for new code.
    """

    def __init__(
        self,
        workspace: Workspace,
        template: Any,
        project_root: Any,
        base_kwargs: dict[str, Any] | None = None,
        fallback_fn: Callable[
            [int, AgentGroupMember], dict] | None = None,
    ):
        """Initialize from workspace, template, and base kwargs."""
        merged_base = dict(base_kwargs or {})
        merged_base["workspace"] = workspace
        merged_base["template"] = template
        merged_base["project_root"] = project_root

        adapted_fallback = None
        if fallback_fn:
            self._legacy_members: list[AgentGroupMember] = []

            def _adapt(
                index: int, member: BlockGroupMember,
            ) -> dict | None:
                if index < len(self._legacy_members):
                    return fallback_fn(
                        index, self._legacy_members[index])
                return None

            adapted_fallback = _adapt
        else:
            self._legacy_members = []

        self._group = BlockGroup(
            workspace=workspace,
            launch_fn=launch_agent_block,
            base_kwargs=merged_base,
            fallback_fn=adapted_fallback,
        )

    def add(self, member: AgentGroupMember) -> None:
        """Add a member to the group."""
        self._legacy_members.append(member)
        self._group.add(member.to_block_member())

    def run(
        self,
        collect_artifacts: list[tuple[str, str]] | None = None,
    ) -> list[AgentGroupResult]:
        """Launch all members and wait sequentially."""
        block_results = self._group.run(
            collect_artifacts=collect_artifacts)
        return [
            AgentGroupResult.from_block_result(r)
            for r in block_results
        ]
