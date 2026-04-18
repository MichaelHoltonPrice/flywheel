"""Parallel execution groups.

Launches multiple executions simultaneously and collects results
sequentially.  Works with any launch function — block executors,
agent blocks, or any callable that returns a handle with ``wait()``.

Usage with a block executor::

    group = BlockGroup(
        workspace=ws,
        launch_fn=executor.launch,
        base_kwargs={"block_name": "eval"},
    )
    group.add(BlockGroupMember(
        overrides={"input_bindings": {"checkpoint": "ckpt@1"}},
    ))
    results = group.run()

Usage with agent blocks::

    from flywheel.agent import launch_agent_block

    group = BlockGroup(
        workspace=ws,
        launch_fn=launch_agent_block,
        base_kwargs={
            "workspace": ws, "template": tpl,
            "project_root": root,
            "agent_image": "flywheel-claude:latest",
        },
    )
    group.add(BlockGroupMember(
        overrides={
            "prompt": "Analyze pattern X",
            "agent_workspace_dir": "explore_0",
        },
        output_dir="explore_0",
    ))
    results = group.run(
        collect_artifacts=[("result", "exploration_result")],
    )
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flywheel.artifact import LifecycleEvent
from flywheel.workspace import Workspace


@dataclass
class BlockGroupMember:
    """Configuration for one member of a block group.

    Attributes:
        overrides: Kwargs that override ``base_kwargs`` for this
            member.  Passed directly to the launch function.
        merge_env: Extra environment variables merged into the
            base ``extra_env`` dict (rather than replacing it).
            Only relevant when ``base_kwargs`` contains
            ``extra_env``.
        output_dir: Subdirectory name (relative to
            ``workspace.path``) where this member writes output
            files.  Used by ``collect_artifacts`` to find and
            register output files.
    """

    overrides: dict[str, Any] = field(default_factory=dict)
    merge_env: dict[str, str] | None = None
    output_dir: str | None = None


@dataclass
class BlockGroupResult:
    """Result of a single execution in a group.

    Attributes:
        index: Position in the group (0-based).
        result: The result from ``handle.wait()`` (type depends
            on the launch function — ``ExecutionResult``,
            ``AgentResult``, etc.).
        output_dir: The output directory for this member, if any.
        artifacts_collected: Artifact instance IDs registered
            for this member during ``collect_artifacts``.
    """

    index: int
    result: Any
    output_dir: str | None = None
    artifacts_collected: list[str] = field(default_factory=list)


class BlockGroup:
    """Launch and manage a group of parallel executions.

    All executions share the same flywheel workspace.  They are
    launched simultaneously and waited on sequentially to serialize
    workspace writes.

    Args:
        workspace: The flywheel workspace.
        launch_fn: Callable that starts an execution and returns
            a handle with a ``wait()`` method.  Examples:
            ``executor.launch``, ``launch_agent_block``.
        base_kwargs: Default keyword arguments passed to every
            ``launch_fn`` call.  Per-member ``overrides`` are
            applied on top.
        fallback_fn: Optional callback to generate fallback output
            when a member produces no output file for a collected
            artifact.  Receives ``(index, member)`` and returns a
            dict to write as JSON, or None to skip.
    """

    def __init__(
        self,
        workspace: Workspace,
        launch_fn: Callable[..., Any],
        base_kwargs: dict[str, Any] | None = None,
        fallback_fn: Callable[
            [int, BlockGroupMember], dict | None] | None = None,
    ):
        """Initialize with workspace and launch function."""
        self._workspace = workspace
        self._launch_fn = launch_fn
        self._base_kwargs = base_kwargs or {}
        self._fallback_fn = fallback_fn
        self._members: list[BlockGroupMember] = []

    def add(self, member: BlockGroupMember) -> None:
        """Add a member to the group."""
        self._members.append(member)

    def run(
        self,
        collect_artifacts: list[tuple[str, str]] | None = None,
    ) -> list[BlockGroupResult]:
        """Launch all members and wait sequentially.

        Args:
            collect_artifacts: List of ``(filename_stem, artifact_name)``
                pairs.  For each member with an ``output_dir``,
                after ``wait()``, look for a file matching the stem
                and register it as an artifact.

        Returns:
            List of ``BlockGroupResult``, one per member, in order.
        """
        if not self._members:
            return []

        # Phase 1: launch all.
        handles: list[tuple[int, BlockGroupMember, Any]] = []
        for i, member in enumerate(self._members):
            kwargs = dict(self._base_kwargs)
            kwargs.update(member.overrides)

            if member.merge_env:
                base_env = dict(kwargs.get("extra_env") or {})
                base_env.update(member.merge_env)
                kwargs["extra_env"] = base_env

            print(f"  [block-group] launching member {i + 1}/"
                  f"{len(self._members)}")
            handle = self._launch_fn(**kwargs)
            handles.append((i, member, handle))

        # Phase 2: wait sequentially (serializes workspace writes).
        results: list[BlockGroupResult] = []
        for i, member, handle in handles:
            result = handle.wait()
            exit_code = getattr(result, "exit_code", None)
            elapsed = getattr(result, "elapsed_s", None)
            print(
                f"  [block-group] member {i + 1} done:"
                f" exit={exit_code}"
                f" elapsed={elapsed:.0f}s"
                if elapsed is not None else
                f"  [block-group] member {i + 1} done:"
                f" exit={exit_code}"
            )

            # If the caller didn't pre-stamp an output_dir
            # (the new auto-naming default for agent groups), use
            # whatever directory the launcher actually mounted.
            # AgentResult exposes ``agent_workspace_dir`` for
            # exactly this purpose; older launch_fns that don't
            # surface it leave member.output_dir as the source of
            # truth.
            output_dir = (
                member.output_dir
                or getattr(result, "agent_workspace_dir", None)
            )

            collected = self._collect(
                i, member, output_dir, collect_artifacts)

            results.append(BlockGroupResult(
                index=i,
                result=result,
                output_dir=output_dir,
                artifacts_collected=collected,
            ))

        # Record lifecycle event.
        event = LifecycleEvent(
            id=self._workspace.generate_event_id(),
            kind="group_completed",
            timestamp=datetime.now(UTC),
            detail={
                "members": str(len(self._members)),
                "succeeded": str(
                    sum(1 for r in results
                        if getattr(r.result, "exit_code", 1) == 0)),
            },
        )
        self._workspace.add_event(event)
        self._workspace.save()

        print(
            f"  [block-group] all {len(self._members)} members "
            f"completed")

        return results

    def _collect(
        self,
        index: int,
        member: BlockGroupMember,
        output_dir: str | None,
        collect_artifacts: list[tuple[str, str]] | None,
    ) -> list[str]:
        """Collect output files from a member's output directory.

        ``output_dir`` is what the caller actually wrote into —
        either the explicit ``member.output_dir`` or the
        auto-named directory the launcher reports back through
        ``AgentResult.agent_workspace_dir``.  Either way, it's
        resolved by :meth:`run` before we get here so this method
        doesn't have to re-derive it.
        """
        if not collect_artifacts or not output_dir:
            return []

        output_path = self._workspace.path / output_dir
        collected: list[str] = []

        for file_stem, artifact_name in collect_artifacts:
            artifact_file = self._find_output_file(
                output_path, file_stem)

            if artifact_file is None and self._fallback_fn:
                fallback_data = self._fallback_fn(index, member)
                if fallback_data is not None:
                    artifact_file = output_path / f"{file_stem}.json"
                    artifact_file.write_text(
                        json.dumps(fallback_data),
                        encoding="utf-8",
                    )
                    print(
                        f"  [block-group] member {index + 1}: "
                        f"wrote fallback for {file_stem}")

            if artifact_file is not None:
                inst = self._workspace.register_artifact(
                    artifact_name, artifact_file,
                    source=f"group member {index + 1}",
                )
                collected.append(inst.id)

        return collected

    @staticmethod
    def _find_output_file(
        output_dir: Path, file_stem: str,
    ) -> Path | None:
        """Find a file in the output directory by stem name."""
        if not output_dir.exists():
            return None
        for candidate in output_dir.iterdir():
            if candidate.is_file() and candidate.stem == file_stem:
                return candidate
        return None
