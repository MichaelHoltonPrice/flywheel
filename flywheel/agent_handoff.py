"""Host-side handoff loop for agent runs that may stop on tool calls.

The agent runner inside the container raises a ``tool_handoff``
state when its ``PreToolUse`` hook intercepts one or more handoff
tool calls in a single assistant turn (see
``batteries/claude/agent_runner.py``).  On that exit:

* ``.agent_state.json`` reports ``status == "tool_handoff"``;
* ``pending_tool_calls.json`` (schema_version 2) lists every
  intercepted ``tool_use`` in the order the SDK emitted them;
* ``agent_session.jsonl`` carries the SDK session with synthetic
  deny ``tool_result`` entries for each ``tool_use_id``.

This module's :func:`run_agent_with_handoffs` is the host-side
counterpart that closes the loop:

1. Launch the agent (via :func:`flywheel.agent.launch_agent_block`).
2. Wait for it to exit.
3. If the exit is ``tool_handoff``: invoke the caller-supplied
   ``block_runner`` once per pending entry (preserving SDK emission
   order), splice each real result into the workspace's
   ``agent_session.jsonl`` artifact via
   :func:`flywheel.session_splice.splice_tool_result`, remove the
   pending file, and relaunch the agent against the same workspace
   with ``RESUME_SESSION_FILE`` pointing at the spliced JSONL.
4. Repeat until the agent exits without a handoff (or
   ``max_iterations`` is hit).

The function returns the *terminal* :class:`flywheel.agent.AgentResult`
plus a list of :class:`HandoffCycle` records describing each
intermediate launch.  When no handoff occurs (the common case
today, with no tools registered in ``HANDOFF_TOOLS``) it
collapses to a single ``launch_agent_block`` + ``handle.wait()``
and is observationally identical to :func:`run_agent_block`.

Production wiring is intentionally minimal:

* The execution channel (the bridge for non-handoff tool calls)
  is still spun up per launch by :func:`launch_agent_block`.  The
  full-stop nested-block plan retires the bridge in B7; until
  then, this loop coexists with it.
* Each cycle's :class:`flywheel.artifact.BlockExecution` row is
  recorded by the underlying :class:`flywheel.agent.AgentHandle`
  with its own ``execution_id``; this loop chains them via
  ``predecessor_id`` so an operator inspecting the workspace
  sees the cycle structure explicitly.

See ``plans/full-stop-nested-blocks.md`` for the campaign and
``plans/full-stop-state-contract.md`` for the on-disk contract
this module assumes.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flywheel.agent import AgentResult, launch_agent_block
from flywheel.session_splice import SpliceError, splice_tool_result

PENDING_FILE_NAME = "pending_tool_calls.json"
SESSION_ARTIFACT_NAME = "agent_session.jsonl"
RESUME_ENV_VAR = "RESUME_SESSION_FILE"

DEFAULT_MAX_ITERATIONS = 16

DEFAULT_RESUME_PROMPT = (
    "The previous tool call returned its result.  "
    "Continue from where you left off without re-issuing the tool call."
)


@dataclass
class HandoffContext:
    """What the block runner sees for a single handoff invocation.

    The loop calls ``block_runner`` once per pending tool call,
    even when the agent emitted multiple parallel ``tool_use``
    blocks in a single assistant turn (chosen design: one block
    execution per ``tool_use``, run sequentially within one
    stop/restart cycle).

    Attributes:
        tool_name: Fully qualified MCP tool name the agent
            invoked (e.g. ``mcp__arc__take_action``).
        tool_input: Verbatim arguments the model emitted, as the
            ``PreToolUse`` hook saw them.
        session_id: Agent's current SDK session ID, as recorded
            in the pending file (or in ``.agent_state.json`` if
            the pending file omits it).
        tool_use_id: SDK-assigned correlation ID for this tool
            call; the splice key.
        agent_workspace: Host-side path to the agent's workspace
            directory (the same directory bind-mounted into the
            container at ``/workspace``).
        iteration: 0-based index of which container launch this
            handoff was surfaced by; shared across all handoffs
            from the same launch.
        index_in_iteration: 0-based position within this
            iteration's pending list.  Preserves the SDK's
            ``tool_use`` emission order within the assistant turn.
        siblings: Total number of pending tool calls in this
            iteration; ``index_in_iteration < siblings`` always.
            ``siblings == 1`` is the single-tool-use case.
    """

    tool_name: str
    tool_input: dict[str, Any]
    session_id: str
    tool_use_id: str
    agent_workspace: Path
    iteration: int
    index_in_iteration: int = 0
    siblings: int = 1


@dataclass
class HandoffResult:
    """What the block runner returns for a single handoff invocation.

    Attributes:
        content: The real ``tool_result`` payload to splice in.
            A string becomes a single text block; a list is used
            verbatim and assumed to match the SDK's expected
            content-block shape.
        is_error: Whether to mark the spliced ``tool_result`` as
            an error (e.g. the block raised).
    """

    content: str | list[dict[str, Any]]
    is_error: bool = False


BlockRunner = Callable[[HandoffContext], HandoffResult]


@dataclass
class HandoffCycle:
    """Record of one launch + handoff resolution within the loop.

    Attributes:
        iteration: 0-based launch index within this loop run.
        agent_result: The :class:`AgentResult` returned by
            ``handle.wait()`` for this cycle's launch.
        handoffs: Per-pending-entry records describing what the
            block runner produced and what was spliced in.  Empty
            on the terminating cycle (the one whose ``exit_reason``
            is *not* ``tool_handoff``).
    """

    iteration: int
    agent_result: AgentResult
    handoffs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HandoffLoopResult:
    """Outcome of a complete handoff-driven agent run.

    Attributes:
        cycles: One :class:`HandoffCycle` per container launch,
            in order.  ``cycles[-1]`` is always the terminating
            cycle; ``cycles[-1].handoffs`` is always empty.
        final_result: Convenience alias for
            ``cycles[-1].agent_result``.
        total_handoffs: Count of pending tool calls resolved
            across all cycles.
    """

    cycles: list[HandoffCycle] = field(default_factory=list)

    @property
    def final_result(self) -> AgentResult:
        """Return the terminal :class:`AgentResult`.

        Equivalent to ``cycles[-1].agent_result``.  This is the
        result whose ``exit_reason`` is *not* ``tool_handoff``
        (the loop only returns once an iteration resolves
        without queueing more pending tool calls).
        """
        return self.cycles[-1].agent_result

    @property
    def total_handoffs(self) -> int:
        """Count of pending tool calls resolved across all cycles."""
        return sum(len(c.handoffs) for c in self.cycles)


class HandoffLoopError(RuntimeError):
    """Raised when the handoff loop cannot proceed safely."""


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning ``{}`` when absent or invalid."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _normalize_pending(doc: Any) -> list[dict[str, Any]]:
    """Return the list of pending tool calls from the on-disk doc.

    Mirrors :func:`flywheel.agent_loop_driver._normalize_pending`
    so the same defensive parsing is in force whichever entry
    point the host uses.  Accepts:

    * ``{"schema_version": 2, "pending": [...]}`` — canonical;
    * a bare ``list`` of pending dicts;
    * a bare single pending dict (legacy schema_version 1).
    """
    if isinstance(doc, list):
        return [p for p in doc if isinstance(p, dict)]
    if not isinstance(doc, dict):
        return []
    pending = doc.get("pending")
    if isinstance(pending, list):
        return [p for p in pending if isinstance(p, dict)]
    if "tool_use_id" in doc and "tool_name" in doc:
        return [doc]
    return []


def _find_agent_ws(
    workspace_path: Path,
    agent_workspace_dir: str | None,
) -> Path:
    """Resolve the host-side agent workspace path for a result.

    The :class:`AgentResult` reports ``agent_workspace_dir`` as a
    workspace-relative subpath (e.g. ``agent_workspaces/abc12345``).
    The handoff loop needs the absolute host path to read the
    pending file and splice the session JSONL.
    """
    if not agent_workspace_dir:
        raise HandoffLoopError(
            "agent result has no agent_workspace_dir; cannot "
            "locate workspace for handoff resolution"
        )
    return workspace_path / agent_workspace_dir


def run_agent_with_handoffs(
    *,
    block_runner: BlockRunner | None = None,
    resume_prompt: str = DEFAULT_RESUME_PROMPT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    launch_fn: Callable[..., Any] = launch_agent_block,
    **launch_kwargs: Any,
) -> HandoffLoopResult:
    """Run an agent through any handoff cycles to completion.

    Drives the host-side loop described in this module's docstring.
    All ``launch_kwargs`` are forwarded to :func:`launch_agent_block`
    on the first cycle.  On each subsequent cycle the loop
    relaunches with ``reuse_workspace=True``, the prior cycle's
    ``execution_id`` as ``predecessor_id``, and ``extra_env`` set
    to point ``RESUME_SESSION_FILE`` at the in-workspace
    ``agent_session.jsonl`` (already spliced).

    Args:
        block_runner: Caller-supplied function invoked once per
            pending tool call.  Receives a :class:`HandoffContext`
            and returns a :class:`HandoffResult`.  When ``None``,
            a handoff exit raises :class:`HandoffLoopError`
            because the loop has no way to satisfy the pending
            tool calls.  Backward-compatible: agents launched
            with no tools listed in ``HANDOFF_TOOLS`` will never
            exit with ``tool_handoff`` and never reach this
            branch, so existing callers can pass ``block_runner=
            None`` and the loop collapses to a single
            ``launch_agent_block`` + ``handle.wait()``.
        resume_prompt: Stdin payload for relaunch cycles.  The
            agent runner sends this as a new user query *after*
            restoring the spliced session, so it must not
            re-issue the original task — the default tells the
            model to continue from where it left off without
            re-calling the tool.  Ignored on cycle 0 (which uses
            ``launch_kwargs["prompt"]``).
        max_iterations: Hard cap on the number of cycles before
            the loop gives up.  Defaults to 16; increase for
            workloads that genuinely chain that many handoffs in
            one agent run.
        launch_fn: Injection point for tests; defaults to
            :func:`launch_agent_block`.  Tests pass a fake that
            returns a stand-in handle so the loop can be exercised
            without Docker.
        **launch_kwargs: Forwarded to ``launch_fn`` on cycle 0.
            Must include the same arguments
            :func:`launch_agent_block` requires.  The loop
            mutates a copy of these kwargs for relaunches (sets
            ``reuse_workspace``, ``predecessor_id``,
            ``agent_workspace_dir``, and ``extra_env``); callers'
            originals are not modified.

    Returns:
        A :class:`HandoffLoopResult` with the per-cycle records.
        ``result.final_result`` is the terminal
        :class:`AgentResult` (the one whose ``exit_reason`` is
        not ``tool_handoff``).

    Raises:
        HandoffLoopError: when the loop cannot proceed (handoff
            requested with no ``block_runner``; pending file
            missing or malformed; session JSONL missing; splice
            failure; ``max_iterations`` exceeded).
    """
    workspace = launch_kwargs["workspace"]
    base_kwargs = dict(launch_kwargs)

    cycles: list[HandoffCycle] = []
    last_execution_id: str | None = base_kwargs.get("predecessor_id")
    last_agent_workspace_dir: str | None = base_kwargs.get(
        "agent_workspace_dir")

    for iteration in range(max_iterations):
        kwargs = dict(base_kwargs)
        if iteration == 0:
            pass
        else:
            kwargs["reuse_workspace"] = True
            kwargs["predecessor_id"] = last_execution_id
            kwargs["agent_workspace_dir"] = last_agent_workspace_dir
            kwargs["prompt"] = resume_prompt
            extra_env = dict(kwargs.get("extra_env") or {})
            extra_env[RESUME_ENV_VAR] = (
                "/workspace/" + SESSION_ARTIFACT_NAME)
            kwargs["extra_env"] = extra_env

        handle = launch_fn(**kwargs)
        try:
            agent_result = handle.wait()
        except Exception:
            raise

        last_execution_id = agent_result.execution_id
        last_agent_workspace_dir = agent_result.agent_workspace_dir

        if agent_result.exit_reason != "tool_handoff":
            cycles.append(HandoffCycle(
                iteration=iteration,
                agent_result=agent_result,
            ))
            return HandoffLoopResult(cycles=cycles)

        if block_runner is None:
            raise HandoffLoopError(
                f"agent (iteration={iteration}) exited with "
                f"exit_reason='tool_handoff' but no block_runner "
                f"was provided to resolve the pending tool calls. "
                f"Either supply block_runner or remove the "
                f"intercepted tools from HANDOFF_TOOLS."
            )

        agent_ws = _find_agent_ws(
            workspace.path, agent_result.agent_workspace_dir)
        pending_path = agent_ws / PENDING_FILE_NAME
        session_path = agent_ws / SESSION_ARTIFACT_NAME

        pending_doc = _read_json(pending_path)
        pending_list = _normalize_pending(pending_doc)
        if not pending_list:
            raise HandoffLoopError(
                f"agent exited with tool_handoff but pending "
                f"file {pending_path} has no tool calls; "
                f"cannot resolve handoff (doc={pending_doc!r})"
            )
        if not session_path.is_file():
            raise HandoffLoopError(
                f"agent exited with tool_handoff but session "
                f"artifact {session_path} is missing; cannot "
                f"splice tool results back into the conversation"
            )

        siblings = len(pending_list)
        session_id_fallback = ""
        if isinstance(pending_doc, dict):
            session_id_fallback = str(pending_doc.get(
                "session_id", "") or "")

        handoff_records: list[dict[str, Any]] = []
        for idx, pending in enumerate(pending_list):
            tool_use_id = pending.get("tool_use_id")
            tool_name = pending.get("tool_name")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                raise HandoffLoopError(
                    f"{pending_path.name} entry {idx} malformed "
                    f"(no tool_use_id): {pending}"
                )
            if not isinstance(tool_name, str) or not tool_name:
                raise HandoffLoopError(
                    f"{pending_path.name} entry {idx} malformed "
                    f"(no tool_name): {pending}"
                )

            ctx = HandoffContext(
                tool_name=tool_name,
                tool_input=pending.get("tool_input", {}) or {},
                session_id=str(pending.get(
                    "session_id", session_id_fallback) or ""),
                tool_use_id=tool_use_id,
                agent_workspace=agent_ws,
                iteration=iteration,
                index_in_iteration=idx,
                siblings=siblings,
            )

            block_result = block_runner(ctx)

            try:
                splice_line = splice_tool_result(
                    session_path,
                    tool_use_id=tool_use_id,
                    tool_result_content=block_result.content,
                    is_error=block_result.is_error,
                )
            except SpliceError as exc:
                raise HandoffLoopError(
                    f"splice failed for tool_use_id="
                    f"{tool_use_id!r} (iteration={iteration}, "
                    f"index={idx}): {exc}"
                ) from exc

            handoff_records.append({
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "tool_input": ctx.tool_input,
                "is_error": block_result.is_error,
                "index_in_iteration": idx,
                "siblings": siblings,
                "splice_line": splice_line,
            })

        cycles.append(HandoffCycle(
            iteration=iteration,
            agent_result=agent_result,
            handoffs=handoff_records,
        ))

        # Drop the pending file before relaunch so a leftover
        # cannot trick a future cycle into thinking new handoffs
        # are queued.
        with contextlib.suppress(FileNotFoundError):
            pending_path.unlink()

    raise HandoffLoopError(
        f"hit max_iterations={max_iterations} without a "
        f"non-handoff exit; last cycle exit_reason="
        f"{cycles[-1].agent_result.exit_reason!r}"
    )
