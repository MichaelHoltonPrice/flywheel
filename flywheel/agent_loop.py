"""Flywheel-owned agent orchestration loop.

Manages the run-decide-repeat pattern: launch an agent, wait for
it to finish, classify the exit, and ask project-provided hooks
what to do next.  Handles session resume, circuit breaking on
auth/rate-limit failures, lifecycle events, and restart from
workspace state.

Projects provide a hooks class implementing ``AgentLoopHooks``:

- ``init(workspace, template, project_root, args)``: One-time
  setup.  Parse project-specific CLI args, initialize external
  services, register initial artifacts.  Returns a dict of
  agent launch config overrides (extra_env, extra_mounts,
  output_names, mcp_servers, pre_launch_hook, etc.).
- ``decide(state) -> Action``: Given what just happened, return
  what to do next (Continue, SpawnGroup, Stop, Finished).
- ``build_prompt(action, state) -> str``: Build the prompt for
  the next agent round.

Optional hooks (detected via ``hasattr``):
- ``on_execution(event, handle)``: React to mid-execution
  artifacts (e.g., stop the agent on a policy trigger).
- ``auto_mount_artifacts() -> list[str]``: Declare artifact
  names to auto-mount on each agent launch.
- ``make_pre_launch_hook()``: Return a callback for agent
  workspace setup before container launch.

Usage::

    from flywheel.agent_loop import AgentLoop

    loop = AgentLoop(
        hooks=MyProjectHooks(config),
        base_config=AgentBlockConfig(...),
        max_rounds=30,
    )
    result = loop.run()
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from flywheel.agent import (
    AgentBlockConfig,
    AgentHandle,
    AgentResult,
    launch_agent_block,
)
from flywheel.artifact import LifecycleEvent
from flywheel.block_group import BlockGroup, BlockGroupMember
from flywheel.executor import ExecutionEvent
from flywheel.template import Template
from flywheel.workspace import Workspace

# ── Loop state and actions ───────────────────────────────────────


@dataclass(frozen=True)
class LoopState:
    """Snapshot of the loop state, passed to hooks.

    Derived from the workspace contents — no ephemeral state
    needed.  This makes the loop restartable from any checkpoint.

    Attributes:
        round_number: Current round (1-based).
        workspace: The flywheel workspace (read-only view).
        last_result: The most recent AgentResult, or None for
            the first round.
        last_exit_reason: The exit_reason from the last result.
        is_resumed: True if this loop started from an existing
            workspace with prior agent executions.
    """

    round_number: int
    workspace: Workspace
    last_result: AgentResult | None = None
    last_exit_reason: str | None = None
    is_resumed: bool = False


class Action:
    """Base class for loop actions returned by decide()."""


@dataclass
class Continue(Action):
    """Continue with a new agent round."""


@dataclass
class SpawnGroup(Action):
    """Spawn parallel sub-executions via BlockGroup.

    Attributes:
        members: List of group members to launch.
        collect_artifacts: Artifact collection spec for the group.
        base_kwargs_override: Override base kwargs for the group.
        fallback_fn: Fallback function for missing outputs.
    """

    members: list[BlockGroupMember] = field(default_factory=list)
    collect_artifacts: list[tuple[str, str]] | None = None
    base_kwargs_override: dict[str, Any] | None = None
    fallback_fn: Any = None


@dataclass
class Stop(Action):
    """Stop the loop.

    Attributes:
        reason: Why the loop is stopping.
    """

    reason: str = ""


@dataclass
class Finished(Action):
    """The task is complete.

    Attributes:
        summary: Optional summary data.
    """

    summary: dict[str, Any] | None = None


# ── Hooks protocol ──────────────────────────────────────────────


class AgentLoopHooks(Protocol):
    """Protocol for project-provided loop hooks.

    Projects implement this to control what the agent does.
    Flywheel controls how it runs.
    """

    def init(
        self,
        workspace: Workspace,
        template: Template,
        project_root: Path,
        args: list[str],
    ) -> dict[str, Any]:
        """One-time setup before the loop starts.

        Parse project-specific CLI args (passed after ``--``),
        initialize external services, register initial artifacts,
        and configure environment variables.

        Returns:
            Dict of ``AgentBlockConfig`` field overrides.  Common
            keys: ``extra_env``, ``extra_mounts``, ``output_names``,
            ``mcp_servers``, ``isolated_network``.  Unknown keys
            are ignored.
        """
        ...

    def decide(self, state: LoopState) -> Action:
        """Decide what to do after each agent round.

        Called after every agent execution completes (or on
        startup for a resumed loop).  Returns an Action.
        """
        ...

    def build_prompt(
        self, action: Action, state: LoopState,
    ) -> str:
        """Build the prompt for the next agent round.

        Called before each agent launch.  For the initial round,
        ``state.last_result`` is None.
        """
        ...


def load_hooks_class(import_path: str) -> type:
    """Import a hooks class from a ``module.path:ClassName`` string.

    Args:
        import_path: Dotted module path and class name separated by
            a colon (e.g., ``myproject.hooks:MyHooks``).

    Returns:
        The hooks class (not an instance).

    Raises:
        ValueError: If the import path format is invalid.
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    if ":" not in import_path:
        raise ValueError(
            f"Hooks import path must be 'module.path:ClassName', "
            f"got {import_path!r}"
        )
    module_path, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ── AgentLoop ────────────────────────────────────────────────────


class AgentLoop:
    """Flywheel-owned orchestration loop.

    Manages: round counting, session resume, error detection
    (auth failure, rate limit), circuit breaker, lifecycle events,
    and restart from workspace.

    Args:
        hooks: Project-provided decision and prompt hooks.
        base_config: Shared configuration for agent launches.
        max_rounds: Maximum loop iterations.
        max_consecutive_failures: Circuit breaker — stop after
            this many consecutive auth/rate-limit/zero-step exits.
    """

    def __init__(
        self,
        hooks: AgentLoopHooks,
        base_config: AgentBlockConfig,
        *,
        max_rounds: int = 10,
        max_consecutive_failures: int = 3,
    ):
        """Initialize the agent loop."""
        self._hooks = hooks
        self._config = base_config
        self._max_rounds = max_rounds
        self._max_consecutive_failures = max_consecutive_failures

    def run(self) -> dict[str, Any]:
        """Run the loop, resuming from workspace state if possible.

        Returns:
            Summary dict with round count, exit reason, and
            any summary from a Finished action.
        """
        workspace = self._config.workspace

        # Determine if this is a resume.
        agent_executions = [
            e for e in workspace.executions.values()
            if e.block_name == "__agent__"
        ]
        is_resumed = len(agent_executions) > 0
        start_round = len(agent_executions) + 1

        consecutive_failures = 0
        rounds_completed = 0
        last_result: AgentResult | None = None
        last_exit_reason: str | None = None
        groups_run = 0
        final_action: Action | None = None

        for round_num in range(
            start_round, start_round + self._max_rounds,
        ):
            state = LoopState(
                round_number=round_num,
                workspace=workspace,
                last_result=last_result,
                last_exit_reason=last_exit_reason,
                is_resumed=is_resumed and round_num == start_round,
            )

            # Decide what to do.
            if round_num == start_round and not is_resumed:
                action: Action = Continue()
            else:
                action = self._hooks.decide(state)

            if isinstance(action, Finished):
                final_action = action
                break
            if isinstance(action, Stop):
                final_action = action
                break

            if isinstance(action, SpawnGroup):
                self._run_group(action)
                groups_run += 1
                # After group, go back to decide.
                last_result = None
                last_exit_reason = None
                continue

            # Build prompt and launch agent.
            prompt = self._hooks.build_prompt(action, state)

            # Build on_execution callback if hooks support it.
            handle_ref: list[AgentHandle | None] = [None]
            on_record_cb = None
            if hasattr(self._hooks, "on_execution"):
                hooks_ref = self._hooks

                def _on_record(
                    block_name: str, outputs: dict,
                    _hooks=hooks_ref,
                    _handle_ref=handle_ref,
                ) -> None:
                    event = ExecutionEvent(
                        executor_type="record",
                        block_name=block_name,
                        execution_id="",
                        status="succeeded",
                        outputs_data=outputs,
                    )
                    _hooks.on_execution(event, _handle_ref[0])

                on_record_cb = _on_record

            # Determine predecessor for resume chain.
            predecessor_id = None
            if last_result and last_result.execution_id:
                predecessor_id = last_result.execution_id
            elif is_resumed and agent_executions:
                predecessor_id = agent_executions[-1].id

            # Auto-mount latest instances of known artifacts.
            input_artifacts = dict(
                self._config.input_artifacts or {})
            extra_env = dict(self._config.extra_env or {})

            # Session resume.
            sessions = workspace.instances_for("agent_session")
            if sessions:
                input_artifacts["agent_session"] = sessions[-1].id
                extra_env["RESUME_SESSION_FILE"] = (
                    "/input/agent_session/agent_session.jsonl")

            # Mount any declared input artifacts that exist.
            # This allows hooks to create artifacts (e.g.,
            # exploration_digest) that get auto-mounted.
            if hasattr(self._hooks, "auto_mount_artifacts"):
                for art_name in self._hooks.auto_mount_artifacts():
                    if art_name not in input_artifacts:
                        instances = workspace.instances_for(art_name)
                        if instances:
                            input_artifacts[art_name] = (
                                instances[-1].id)

            print(f"\n  [agent-loop] === Round {round_num} ===")

            handle = launch_agent_block(
                workspace=self._config.workspace,
                template=self._config.template,
                project_root=self._config.project_root,
                prompt=prompt,
                agent_image=self._config.agent_image,
                auth_volume=self._config.auth_volume,
                model=self._config.model,
                max_invocations=self._config.max_invocations,
                max_turns=self._config.max_turns,
                total_timeout=self._config.total_timeout,
                allowed_blocks=self._config.allowed_blocks,
                source_dirs=self._config.source_dirs,
                input_artifacts=input_artifacts or None,
                output_names=self._config.output_names,
                overrides=self._config.overrides,
                mcp_servers=self._config.mcp_servers,
                allowed_tools=self._config.allowed_tools,
                extra_env=extra_env or None,
                extra_mounts=self._config.extra_mounts,
                pre_launch_hook=self._config.pre_launch_hook,
                on_record=on_record_cb,
                isolated_network=self._config.isolated_network,
                agent_workspace_dir=self._config.agent_workspace_dir,
                predecessor_id=predecessor_id,
            )
            handle_ref[0] = handle
            last_result = handle.wait()
            last_exit_reason = last_result.exit_reason
            rounds_completed += 1

            print(
                f"  [agent-loop] round {round_num} done: "
                f"exit_reason={last_exit_reason}")

            # Circuit breaker.
            if last_exit_reason in (
                "auth_failure", "rate_limit", "crashed",
            ):
                consecutive_failures += 1
                if consecutive_failures >= self._max_consecutive_failures:
                    print(
                        f"  [agent-loop] circuit breaker: "
                        f"{consecutive_failures} consecutive "
                        f"failures ({last_exit_reason})")

                    event = LifecycleEvent(
                        id=workspace.generate_event_id(),
                        kind="circuit_breaker",
                        timestamp=datetime.now(UTC),
                        execution_id=last_result.execution_id,
                        detail={
                            "reason": last_exit_reason or "",
                            "consecutive": str(
                                consecutive_failures),
                        },
                    )
                    workspace.add_event(event)
                    workspace.save()

                    final_action = Stop(
                        reason=f"circuit_breaker:{last_exit_reason}")
                    break
            else:
                consecutive_failures = 0

            is_resumed = False

        summary: dict[str, Any] = {
            "rounds_completed": rounds_completed,
            "groups_run": groups_run,
            "last_exit_reason": last_exit_reason,
            "is_finished": isinstance(final_action, Finished),
        }
        if isinstance(final_action, Finished) and final_action.summary:
            summary.update(final_action.summary)
        if isinstance(final_action, Stop):
            summary["stop_reason"] = final_action.reason

        return summary

    def _run_group(self, action: SpawnGroup) -> None:
        """Execute a SpawnGroup action."""
        kwargs = dict(action.base_kwargs_override or {})
        kwargs["workspace"] = self._config.workspace
        kwargs["template"] = self._config.template
        kwargs["project_root"] = self._config.project_root

        group = BlockGroup(
            workspace=self._config.workspace,
            launch_fn=launch_agent_block,
            base_kwargs=kwargs,
            fallback_fn=action.fallback_fn,
        )
        for member in action.members:
            group.add(member)
        group.run(collect_artifacts=action.collect_artifacts)
