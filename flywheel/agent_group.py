"""Parallel agent group execution.

Launches multiple agent blocks simultaneously and collects their
results sequentially.  All agents share the same flywheel workspace
(for artifact tracking) but use distinct workspace subdirectories
(to avoid file conflicts).

Usage::

    from flywheel.agent import AgentBlockConfig
    from flywheel.agent_group import AgentGroup, AgentGroupMember

    base = AgentBlockConfig(
        workspace=workspace, template=template,
        project_root=project_root, prompt="ignored",
        agent_image="flywheel-claude:latest",
    )

    group = AgentGroup(base)
    group.add(AgentGroupMember(
        prompt="Analyze pattern X",
        agent_workspace_dir="explore_0",
    ))
    group.add(AgentGroupMember(
        prompt="Analyze pattern Y",
        agent_workspace_dir="explore_1",
    ))
    results = group.run(
        collect_artifacts=[("exploration_result", "exploration_result")],
    )
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from flywheel.agent import AgentResult, launch_agent_block
from flywheel.artifact import LifecycleEvent
from flywheel.workspace import Workspace


@dataclass
class AgentGroupMember:
    """Configuration for one member of an agent group.

    Overrides specific fields from the group's base config.
    Only non-None fields override the base.

    Attributes:
        prompt: The system prompt for this agent.
        agent_workspace_dir: Distinct workspace subdirectory name.
        input_artifacts: Override input artifacts for this member.
        extra_env: Additional env vars merged on top of the base.
        output_names: Override output names for this member.
    """

    prompt: str
    agent_workspace_dir: str
    input_artifacts: dict[str, str] | None = None
    extra_env: dict[str, str] | None = None
    output_names: list[str] | None = None


@dataclass
class AgentGroupResult:
    """Result of a single agent in a group.

    Attributes:
        index: Position in the group (0-based).
        agent_result: The ``AgentResult`` from ``wait()``.
        agent_workspace_dir: The workspace subdirectory used.
        artifacts_collected: Artifact instance IDs registered
            for this member.
    """

    index: int
    agent_result: AgentResult
    agent_workspace_dir: str
    artifacts_collected: list[str] = field(default_factory=list)


class AgentGroup:
    """Launch and manage a group of parallel agents.

    All agents share the same flywheel workspace but use distinct
    workspace subdirectories.  Agents are launched simultaneously
    and waited on sequentially to serialize artifact registration.

    Args:
        workspace: The flywheel workspace.
        template: The workspace template.
        project_root: Project root directory.
        base_kwargs: Keyword arguments passed to every
            ``launch_agent_block`` call.  Per-member overrides
            (prompt, agent_workspace_dir, input_artifacts,
            extra_env, output_names) are merged on top.
        fallback_fn: Optional callback to generate fallback output
            when an agent produces no output file for a collected
            artifact.  Receives ``(index, member)`` and returns a
            dict to write as JSON.  If None, missing outputs are
            silently skipped.
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
        """Initialize from workspace, template, and base launch kwargs."""
        self._workspace = workspace
        self._template = template
        self._project_root = project_root
        self._base_kwargs = base_kwargs or {}
        self._fallback_fn = fallback_fn
        self._members: list[AgentGroupMember] = []

    def add(self, member: AgentGroupMember) -> None:
        """Add a member to the group.

        Args:
            member: Configuration for this group member.
        """
        self._members.append(member)

    def run(
        self,
        collect_artifacts: list[tuple[str, str]] | None = None,
    ) -> list[AgentGroupResult]:
        """Launch all members and wait sequentially.

        Args:
            collect_artifacts: List of ``(filename_stem, artifact_name)``
                pairs.  For each member, after ``wait()``, look for a
                file matching the stem in the agent workspace and
                register it as an artifact.

        Returns:
            List of ``AgentGroupResult``, one per member, in order.
        """
        if not self._members:
            return []

        # Phase 1: launch all agents in parallel.
        handles: list[tuple[int, AgentGroupMember, Any]] = []
        for i, member in enumerate(self._members):
            kwargs = dict(self._base_kwargs)
            kwargs["workspace"] = self._workspace
            kwargs["template"] = self._template
            kwargs["project_root"] = self._project_root
            kwargs["prompt"] = member.prompt
            kwargs["agent_workspace_dir"] = member.agent_workspace_dir

            if member.input_artifacts is not None:
                kwargs["input_artifacts"] = member.input_artifacts

            if member.output_names is not None:
                kwargs["output_names"] = member.output_names

            if member.extra_env is not None:
                base_env = dict(kwargs.get("extra_env") or {})
                base_env.update(member.extra_env)
                kwargs["extra_env"] = base_env

            print(f"  [agent-group] launching member {i + 1}/"
                  f"{len(self._members)}")
            handle = launch_agent_block(**kwargs)
            handles.append((i, member, handle))

        # Phase 2: wait sequentially (serializes workspace writes).
        results: list[AgentGroupResult] = []
        for i, member, handle in handles:
            result = handle.wait()
            print(
                f"  [agent-group] member {i + 1} done: "
                f"exit={result.exit_code}"
                f" elapsed={result.elapsed_s:.0f}s"
            )

            collected: list[str] = []

            if collect_artifacts:
                agent_ws = (
                    self._workspace.path / member.agent_workspace_dir
                )
                for file_stem, artifact_name in collect_artifacts:
                    artifact_file = self._find_output_file(
                        agent_ws, file_stem)

                    if artifact_file is None and self._fallback_fn:
                        fallback_data = self._fallback_fn(i, member)
                        if fallback_data is not None:
                            artifact_file = (
                                agent_ws / f"{file_stem}.json")
                            artifact_file.write_text(
                                json.dumps(fallback_data),
                                encoding="utf-8",
                            )
                            print(
                                f"  [agent-group] member {i + 1}: "
                                f"wrote fallback for {file_stem}")

                    if artifact_file is not None:
                        inst = self._workspace.register_artifact(
                            artifact_name, artifact_file,
                            source=f"agent group member {i + 1}",
                        )
                        collected.append(inst.id)

            results.append(AgentGroupResult(
                index=i,
                agent_result=result,
                agent_workspace_dir=member.agent_workspace_dir,
                artifacts_collected=collected,
            ))

        # Record a lifecycle event for the group completion.
        event = LifecycleEvent(
            id=self._workspace.generate_event_id(),
            kind="group_completed",
            timestamp=datetime.now(UTC),
            detail={
                "members": str(len(self._members)),
                "succeeded": str(
                    sum(1 for r in results
                        if r.agent_result.exit_code == 0)),
            },
        )
        self._workspace.add_event(event)
        self._workspace.save()

        print(
            f"  [agent-group] all {len(self._members)} members "
            f"completed")

        return results

    @staticmethod
    def _find_output_file(
        agent_ws: Any, file_stem: str,
    ) -> Any:
        """Find a file in the agent workspace by stem name.

        Returns the path if found, or None.
        """
        if not hasattr(agent_ws, 'exists') or not agent_ws.exists():
            return None
        for candidate in agent_ws.iterdir():
            if candidate.is_file() and candidate.stem == file_stem:
                return candidate
        return None
