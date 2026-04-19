"""Host-side handoff loop for agent runs that may stop on tool calls.

The agent runner inside the container raises a ``tool_handoff``
state when its ``PreToolUse`` hook intercepts one or more handoff
tool calls in a single assistant turn (see
``batteries/claude/agent_runner.py``).  On that exit:

* ``.agent_state.json`` reports ``status == "tool_handoff"``;
* ``pending_tool_calls.json`` (schema_version 2) lists every
  intercepted ``tool_use`` in the order the SDK emitted them;
* the SDK session JSONL (at ``/state/session.jsonl`` inside the
  container, captured into
  ``<workspace>/state/<block>/<exec_id>/session.jsonl``) carries
  the conversation with synthetic deny ``tool_result`` entries
  for each ``tool_use_id``.

This module's :func:`run_agent_with_handoffs` is the host-side
counterpart that closes the loop:

1. Launch the agent (via :func:`flywheel.agent.launch_agent_block`).
2. Wait for it to exit.
3. If the exit is ``tool_handoff``: invoke the caller-supplied
   ``block_runner`` once per pending entry (preserving SDK
   emission order), splice each real result into the captured
   state_dir's ``session.jsonl`` via
   :func:`flywheel.session_splice.splice_tool_result`, and
   remove the pending file.  Relaunching the agent is automatic
   — the next execution's ``/state/`` populate reads the
   mutated state_dir.
4. Repeat until the agent exits without a handoff (or
   ``max_iterations`` is hit).

The function returns the *terminal* :class:`flywheel.agent.AgentResult`
plus a list of :class:`HandoffCycle` records describing each
intermediate launch.  When no handoff occurs (the common case
today, with no tools registered in ``HANDOFF_TOOLS``) it
collapses to a single ``launch_agent_block`` + ``handle.wait()``
and is observationally identical to :func:`run_agent_block`.

Production wiring is intentionally minimal:

* No host-side HTTP bridge runs.  Block executions invoked
  during a handoff are recorded directly by the host runner
  via :class:`flywheel.local_block.LocalBlockRecorder`.
* Each cycle's :class:`flywheel.artifact.BlockExecution` record is
  recorded by the underlying :class:`flywheel.agent.AgentHandle`
  with its own ``execution_id``; this loop chains them via
  ``predecessor_id`` so an operator inspecting the workspace
  sees the cycle structure explicitly.

The on-disk handoff state (the pending tool-call file the
agent runner writes when it intercepts a mapped tool) is
documented inline in :mod:`flywheel.batteries.claude.agent_runner`.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flywheel import runtime
from flywheel.agent import AgentResult, launch_agent_block
from flywheel.session_splice import SpliceError, splice_tool_result

PENDING_FILE_NAME = "pending_tool_calls.json"
SESSION_FILE_NAME = "session.jsonl"
"""Name of the session JSONL inside a captured state directory.

The agent runner writes ``/state/session.jsonl`` at container
exit; flywheel captures ``/state/`` into
``<workspace>/state/<block>/<exec_id>/``, so the session ends up
at ``<workspace>/state/<block>/<exec_id>/session.jsonl``.  The
handoff loop splices tool results into that file between
iterations; the next launch's state-populate copies the spliced
contents into the new execution's ``/state/`` mount."""

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
            cycle.  Its ``handoffs`` is empty for the natural
            termination case (agent exited with a non-handoff
            ``exit_reason``).  When the loop terminates because a
            post-check queued a halt directive between cycles,
            ``cycles[-1].handoffs`` carries the resolved handoffs
            for that final iteration and :attr:`halts` carries
            the directives that suppressed the relaunch.
        halts: Halt directives drained from ``halt_source`` that
            caused the loop to stop relaunching.  Empty when the
            loop terminated naturally (agent exit_reason was not
            ``tool_handoff``).  Each entry is whatever shape the
            ``halt_source`` callable returned -- the loop does
            not interpret the contents, only their presence.
        final_result: Convenience alias for
            ``cycles[-1].agent_result``.
        total_handoffs: Count of pending tool calls resolved
            across all cycles.
    """

    cycles: list[HandoffCycle] = field(default_factory=list)
    halts: list[Any] = field(default_factory=list)

    @property
    def final_result(self) -> AgentResult:
        """Return the terminal :class:`AgentResult`.

        Equivalent to ``cycles[-1].agent_result``.  When the
        loop terminated naturally this is the agent run whose
        ``exit_reason`` is not ``tool_handoff``; when the loop
        terminated because a halt directive fired this is the
        last agent run whose handoffs were resolved before the
        halt suppressed the next relaunch.
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
    halt_source: Callable[[], list[Any]] | None = None,
    resume_prompt: str = DEFAULT_RESUME_PROMPT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    launch_fn: Callable[..., Any] = launch_agent_block,
    **launch_kwargs: Any,
) -> HandoffLoopResult:
    """Run an agent through any handoff cycles to completion.

    Drives the host-side loop described in this module's docstring.
    All ``launch_kwargs`` are forwarded to :func:`launch_agent_block`
    on the first cycle.  On each subsequent cycle the loop
    relaunches with ``reuse_workspace=True`` (so the agent
    workspace's control files survive) and the prior cycle's
    ``execution_id`` as ``predecessor_id``.  Session resume is
    automatic: the launcher populates ``/state/`` from the prior
    execution's captured state_dir, which the handoff loop has
    just mutated in place via :func:`splice_tool_result`.

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
        halt_source: Optional zero-arg callable the loop polls
            once per cycle, *after* resolving that cycle's
            pending tool calls and *before* relaunching the agent.
            When the callable returns a non-empty list the loop
            terminates immediately, recording the directives in
            ``HandoffLoopResult.halts``.  Production wires this
            to :meth:`flywheel.local_block.LocalBlockRecorder.drain_halts`
            so a post-execution check that fires during a
            handoff can stop the run cleanly.  ``None`` (default)
            disables halt detection entirely; the loop runs
            until ``max_iterations`` or a non-handoff exit.
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
            # ``reuse_workspace`` keeps pending_tool_calls.json +
            # .agent_state.json visible to the next iteration via
            # the persistent agent-workspace bind.  The session
            # resume path no longer runs through this bind — the
            # next container's ``/state/`` is populated from the
            # previous execution's captured state_dir, which we
            # mutated in place via splice_tool_result.
            kwargs["reuse_workspace"] = True
            kwargs["predecessor_id"] = last_execution_id
            kwargs["agent_workspace_dir"] = last_agent_workspace_dir
            kwargs["prompt"] = resume_prompt

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

        # Session lives in the captured state dir.  The
        # execution record's ``state_dir`` points us there.
        # If state capture itself failed, surface that as the
        # primary error — without it the session lookup below
        # would fail with a confusing "session missing" message
        # that hides the real cause.
        execution = workspace.executions.get(
            agent_result.execution_id)
        if (execution is not None
                and execution.failure_phase
                == runtime.FAILURE_STATE_CAPTURE):
            raise HandoffLoopError(
                f"agent exited with tool_handoff but state "
                f"capture failed for execution "
                f"{agent_result.execution_id!r}: "
                f"{execution.error}; cannot resume the "
                f"conversation without captured state"
            )
        session_path: Path | None = None
        if execution is not None and execution.state_dir:
            session_path = (
                workspace.path / execution.state_dir
                / SESSION_FILE_NAME
            )

        pending_doc = _read_json(pending_path)
        pending_list = _normalize_pending(pending_doc)
        if not pending_list:
            raise HandoffLoopError(
                f"agent exited with tool_handoff but pending "
                f"file {pending_path} has no tool calls; "
                f"cannot resolve handoff (doc={pending_doc!r})"
            )
        if session_path is None or not session_path.is_file():
            raise HandoffLoopError(
                f"agent exited with tool_handoff but session "
                f"file {session_path} is missing from the "
                f"captured state_dir; cannot splice tool "
                f"results back into the conversation"
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

        # Halt check: a post-execution callable invoked inside
        # one of this cycle's block_runner calls may have queued
        # a halt directive.  Drain the queue and stop the loop
        # if anything was returned -- the agent has already had
        # its tool results spliced in but we refuse to relaunch.
        if halt_source is not None:
            drained = halt_source()
            if drained:
                return HandoffLoopResult(
                    cycles=cycles, halts=list(drained))

    raise HandoffLoopError(
        f"hit max_iterations={max_iterations} without a "
        f"non-handoff exit; last cycle exit_reason="
        f"{cycles[-1].agent_result.exit_reason!r}"
    )


# --------------------------------------------------------------------
# Threaded handle wrapper for ``PatternRunner`` compatibility.
# --------------------------------------------------------------------


class HandoffAgentHandle:
    """``PatternRunner``-compatible handle around the handoff loop.

    :class:`flywheel.pattern_runner.PatternRunner` polls handles
    via the minimal ``is_alive()`` / ``wait()`` protocol — it does
    not care whether one launch or several happen behind that
    interface.  This wrapper runs :func:`run_agent_with_handoffs`
    on a background thread so the pattern runner sees a single
    long-lived "agent" the whole time, even when the underlying
    loop is cycling through stop/restart iterations.

    The handle's ``wait()`` returns the *terminal*
    :class:`flywheel.agent.AgentResult` (the same ``AgentResult``
    a non-handoff ``AgentHandle.wait()`` would have returned for a
    one-shot run), so existing pattern-runner consumers see no
    schema change.  The full per-cycle :class:`HandoffLoopResult`
    is also exposed on the handle as ``loop_result`` for callers
    that want to inspect the chain.

    The wrapper does not implement ``stop()``: the pattern runner
    does not call it today, and a clean stop semantics across an
    in-flight handoff cycle (mid-``block_runner`` invocation,
    mid-splice) is its own design.  When that need arises it
    becomes its own piece of work.

    Attributes:
        loop_result: Populated once the loop terminates (whether
            by completion or exception).  ``None`` while the
            background thread is still running.
        loop_error: Populated if ``run_agent_with_handoffs``
            raised; ``None`` otherwise.  Re-raised by ``wait()``
            so the caller learns about the failure.
    """

    def __init__(
        self,
        *,
        block_runner: BlockRunner | None,
        launch_kwargs: dict[str, Any],
        halt_source: Callable[[], list[Any]] | None = None,
        resume_prompt: str = DEFAULT_RESUME_PROMPT,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        launch_fn: Callable[..., Any] = launch_agent_block,
    ) -> None:
        """Spawn a background thread that runs the handoff loop.

        Args:
            block_runner: Forwarded to
                :func:`run_agent_with_handoffs`.  ``None`` is the
                no-handoff backward-compat path; passing a real
                callable enables handoff resolution.
            launch_kwargs: Forwarded as the ``**launch_kwargs`` to
                :func:`run_agent_with_handoffs`.  Must include the
                same arguments :func:`launch_agent_block`
                requires.  Stored unmodified — the loop makes its
                own copies for relaunches.
            halt_source: Forwarded to
                :func:`run_agent_with_handoffs`.  Production
                passes :meth:`LocalBlockRecorder.drain_halts`.
            resume_prompt: Same as
                :func:`run_agent_with_handoffs`.
            max_iterations: Same as
                :func:`run_agent_with_handoffs`.
            launch_fn: Same as
                :func:`run_agent_with_handoffs`; default is the
                production :func:`launch_agent_block`.  Tests
                inject a fake.
        """
        self._block_runner = block_runner
        self._halt_source = halt_source
        self._launch_kwargs = launch_kwargs
        self._resume_prompt = resume_prompt
        self._max_iterations = max_iterations
        self._launch_fn = launch_fn

        self.loop_result: HandoffLoopResult | None = None
        self.loop_error: BaseException | None = None
        self._waited = False

        self._thread = threading.Thread(
            target=self._run_loop,
            name="flywheel-handoff-loop",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        """Body of the background thread.

        Captures success into ``loop_result`` and any exception
        into ``loop_error`` so ``wait()`` can re-raise on the
        caller's thread.  Catching ``BaseException`` is
        intentional: a ``KeyboardInterrupt`` raised inside the
        loop should still surface to the caller rather than
        leaking out of the daemon thread.
        """
        try:
            self.loop_result = run_agent_with_handoffs(
                block_runner=self._block_runner,
                halt_source=self._halt_source,
                resume_prompt=self._resume_prompt,
                max_iterations=self._max_iterations,
                launch_fn=self._launch_fn,
                **self._launch_kwargs,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised in wait()
            self.loop_error = exc

    def is_alive(self) -> bool:
        """Return whether the loop's background thread is running."""
        return self._thread.is_alive()

    def wait(self) -> AgentResult:
        """Block until the loop terminates and return the terminal result.

        Idempotent: calling more than once is safe and returns
        (or re-raises) the same outcome as the first call.  This
        matches the spirit of the existing one-shot
        :class:`flywheel.agent.AgentHandle.wait` while being
        gentler about repeat calls — pattern runner code that
        defensively waits twice should not blow up here.

        Raises:
            HandoffLoopError: anything the loop raised.
            Exception: any other exception raised inside the
                loop's thread (re-raised on the caller's thread).
        """
        self._thread.join()
        self._waited = True
        if self.loop_error is not None:
            raise self.loop_error
        # The loop returns a HandoffLoopResult only when the
        # terminating cycle has at least one entry; if we get
        # here without an error the result must be present.
        assert self.loop_result is not None
        return self.loop_result.final_result


def launch_agent_with_handoffs(
    *,
    block_runner: BlockRunner | None = None,
    halt_source: Callable[[], list[Any]] | None = None,
    resume_prompt: str = DEFAULT_RESUME_PROMPT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    launch_fn: Callable[..., Any] = launch_agent_block,
    **launch_kwargs: Any,
) -> HandoffAgentHandle:
    """Pattern-runner-compatible non-blocking launcher.

    Constructs a :class:`HandoffAgentHandle` so the same kwargs
    that drive :func:`flywheel.agent.launch_agent_block` (which
    :class:`flywheel.pattern_runner.PatternRunner` builds via
    ``_kwargs_for``) flow through this launcher unchanged.  The
    handoff parameters (``block_runner``, ``resume_prompt``,
    ``max_iterations``) are keyword-only and *not* in
    ``launch_kwargs``, so callers wire them up via
    :func:`functools.partial` before passing the result as
    ``PatternRunner(launch_fn=...)``.

    Typical wiring:

    .. code-block:: python

        from functools import partial
        from flywheel.agent_handoff import launch_agent_with_handoffs

        my_launch = partial(
            launch_agent_with_handoffs,
            block_runner=my_block_runner,
        )
        runner = PatternRunner(
            pattern, base_config=cfg, launch_fn=my_launch,
        )
        runner.run()

    With ``block_runner=None`` the handle wraps a one-cycle no-op
    loop that is observationally identical to a direct
    :func:`launch_agent_block` (no thread overhead worth caring
    about on the cadence pattern runner polls at).

    Args:
        block_runner: Forwarded to :class:`HandoffAgentHandle`.
        halt_source: Forwarded to :class:`HandoffAgentHandle`.
        resume_prompt: Same.
        max_iterations: Same.
        launch_fn: Same.
        **launch_kwargs: Forwarded as the loop's
            ``**launch_kwargs``.  Must include the same
            arguments :func:`launch_agent_block` requires.

    Returns:
        A :class:`HandoffAgentHandle` with its background thread
        already started.
    """
    return HandoffAgentHandle(
        block_runner=block_runner,
        launch_kwargs=launch_kwargs,
        halt_source=halt_source,
        resume_prompt=resume_prompt,
        max_iterations=max_iterations,
        launch_fn=launch_fn,
    )


def make_tool_router(
    routes: dict[str, BlockRunner],
) -> BlockRunner:
    """Compose per-tool ``BlockRunner``s into one dispatcher.

    Use this when a single agent run intercepts more than one
    handoff tool (e.g. cyberarc's ``mcp__arc__take_action`` and
    ``mcp__arc__predict_action``).  The returned callable
    satisfies the :data:`BlockRunner` protocol and is the
    ``block_runner`` you pass to
    :func:`launch_agent_with_handoffs` /
    :func:`run_agent_with_handoffs`.

    The router is intentionally a thin dict lookup: every runner
    in ``routes`` keeps its own state, construction args, and
    error contract.  Adding a new handoff tool is just one more
    entry here plus one more ``HANDOFF_TOOLS`` listing.

    Args:
        routes: Mapping from the fully qualified MCP tool name
            the agent runner's ``PreToolUse`` hook intercepts
            (e.g. ``"mcp__arc__take_action"``) to the runner
            that should handle it.  The mapping is snapshotted
            so later mutation of the original dict does not
            affect dispatch.

    Returns:
        A :data:`BlockRunner` that dispatches each
        :class:`HandoffContext` to the matching runner by
        ``ctx.tool_name``.  When the tool is not in ``routes``
        the router returns
        ``HandoffResult(content="ERROR: ...", is_error=True)``
        rather than raising, so a single misregistered tool
        cannot crash the whole handoff loop — the agent sees
        an explicit error and can adapt or terminate.
    """
    snapshot = dict(routes)

    def _router(ctx: HandoffContext) -> HandoffResult:
        runner = snapshot.get(ctx.tool_name)
        if runner is None:
            return HandoffResult(
                content=(
                    f"ERROR: no handoff runner registered for "
                    f"tool {ctx.tool_name!r}; registered tools: "
                    f"{sorted(snapshot)}"
                ),
                is_error=True,
            )
        return runner(ctx)

    return _router
