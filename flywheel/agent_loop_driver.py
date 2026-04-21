"""Drive an agent container through tool-call handoff cycles.

.. deprecated::

   This module is a legacy handoff driver that predates the
   substrate contract.  It still uses the pre-substrate
   ``RESUME_SESSION_FILE`` / ``/scratch/agent_session.jsonl``
   convention, a durable agent workspace directory, and
   ``.agent_stop`` as the cancellation sentinel — none of which
   match the contract implemented by
   :mod:`flywheel.agent_handoff` +
   :mod:`flywheel.executor.ProcessExitExecutor`.

   The only authoritative handoff path is
   :mod:`flywheel.agent_handoff`.  New code must not import
   from this module.  Existing callers and tests pinned to the
   legacy contract are tracked for migration in
   ``cyber-root/substrate-plan.md`` — until they move, this
   module exists only to keep those tests running; it is not
   part of the substrate-conformance story.

The flywheel-claude agent runner exposes a ``HANDOFF_TOOLS`` env
var.  When the agent invokes one or more matched MCP tools in a
single assistant turn, a ``PreToolUse`` hook denies each call
(recording a deny ``tool_result`` per ``tool_use_id`` in the SDK
session JSONL), captures every intercepted call into
``pending_tool_calls.json`` (``schema_version`` 2), sets
``.agent_state.json`` status to ``tool_handoff``, and exits
cleanly.  This module is the host-side counterpart: it launches
the agent container, waits for exit, and on a handoff exit runs a
caller-provided block once per pending entry (in order), splices
each real result into the saved session JSONL, and restarts the
container with ``RESUME_SESSION_FILE`` pointing at the spliced
JSONL.  The loop continues until the agent exits without any
pending handoffs.

The driver intentionally avoids the heavier ``launch_agent_block``
machinery (workspace recording, execution-channel HTTP service,
prior-artifact seeding) so it can serve as a minimal substrate for
both real production patterns and integration tests.  Callers
provide the docker command they want run; the driver only orchestrates
stop / splice / restart around it.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flywheel.session_splice import SpliceError, splice_tool_result

PENDING_FILE_NAME = "pending_tool_calls.json"
STATE_FILE_NAME = ".agent_state.json"
SESSION_ARTIFACT_NAME = "agent_session.jsonl"
PENDING_SCHEMA_VERSION = 2


@dataclass
class HandoffContext:
    """What the block runner sees for a single handoff invocation.

    The driver calls ``block_runner`` once per pending tool call,
    even when the agent emitted multiple parallel tool_use blocks
    in one assistant turn (chosen design: one block execution per
    tool_use, run sequentially within a single stop/restart cycle).

    Attributes:
        tool_name: Fully qualified MCP tool name the agent invoked
            (e.g., ``mcp__arc__take_action``).
        tool_input: Verbatim arguments the model emitted, as the
            ``PreToolUse`` hook saw them.
        session_id: Agent's current SDK session ID.
        tool_use_id: SDK-assigned correlation ID for this tool
            call; the splice key.
        workspace: Host-side path to the agent workspace directory
            (the same directory bind-mounted into the container at
            its workspace path).
        iteration: 0-based index of which container launch this is;
            shared across all handoffs surfaced by that launch.
        index_in_iteration: 0-based position within this iteration's
            pending list (preserves the SDK's tool_use emission
            order within the assistant turn).
        siblings: Total number of pending tool calls in this
            iteration; ``index_in_iteration < siblings`` always.
            ``siblings == 1`` is the single-tool-use case.
        run_id: :class:`flywheel.artifact.RunRecord` id of the
            outer run that launched the agent, or ``None`` for
            ad-hoc invocations.  Block runners forward this into
            :meth:`flywheel.local_block.LocalBlockRecorder.begin`
            so nested executions inherit the agent's run_id and
            pattern-level cadence counters can scope correctly.
    """

    tool_name: str
    tool_input: dict[str, Any]
    session_id: str
    tool_use_id: str
    workspace: Path
    iteration: int
    index_in_iteration: int = 0
    siblings: int = 1
    run_id: str | None = None


@dataclass
class HandoffResult:
    """What the block runner returns for a single handoff invocation.

    Attributes:
        content: The real tool_result payload to splice in.  A
            string becomes a single text block; a list is used
            verbatim and assumed to match the SDK's expected shape.
        is_error: Whether to mark the spliced tool_result as an
            error (e.g., the block raised).
    """

    content: str | list[dict[str, Any]]
    is_error: bool = False


BlockRunner = Callable[[HandoffContext], HandoffResult]


@dataclass
class LoopIteration:
    """Per-iteration record returned in ``LoopResult.iterations``.

    Attributes:
        container_exit_code: Docker exit code for this launch.
        state: ``.agent_state.json`` snapshot captured after exit.
        handoffs: List of handoff records (one per intercepted
            tool_use surfaced by this iteration).  Empty on the
            terminating iteration.  Each entry mirrors the
            corresponding ``HandoffContext`` plus the splice line
            number so failures can be located precisely.
    """

    container_exit_code: int
    state: dict[str, Any]
    handoffs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LoopResult:
    """Outcome of a full handoff-loop run.

    Attributes:
        iterations: One record per container launch, in order.
            The last one's ``handoffs`` is empty (the loop exits
            when an iteration completes without any pending
            handoffs).
        final_state: The final ``.agent_state.json`` contents.
        final_session_path: Path to the final session JSONL on
            disk (the workspace artifact, after all splices).
    """

    iterations: list[LoopIteration] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    final_session_path: Path | None = None


class HandoffLoopError(RuntimeError):
    """Raised when the loop cannot proceed safely."""


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning ``{}`` when absent or invalid."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_path(p: Path) -> str:
    """Format a host path for docker -v mounts (POSIX-style)."""
    return str(p).replace("\\", "/")


def _normalize_pending(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the list of pending tool calls from the on-disk doc.

    The canonical schema (``schema_version`` 2) wraps the list in
    a ``pending`` key.  We accept a handful of alternative shapes
    defensively so a future runner change or hand-edited fixture
    doesn't blow up the loop:

    * ``{"schema_version": 2, "pending": [...]}`` — canonical.
    * A bare ``list`` of pending dicts.
    * A bare single pending dict with ``tool_use_id`` (legacy
      schema_version 1; wrapped into a one-element list).
    """
    if isinstance(doc, list):
        return [p for p in doc if isinstance(p, dict)]
    pending = doc.get("pending")
    if isinstance(pending, list):
        return [p for p in pending if isinstance(p, dict)]
    if "tool_use_id" in doc and "tool_name" in doc:
        return [doc]
    return []


def _drain(stream, sink: Path) -> threading.Thread:
    """Background-stream a subprocess pipe to a file."""

    def _run() -> None:
        with sink.open("a", encoding="utf-8", errors="replace") as fh:
            for line in stream:
                fh.write(line)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


DEFAULT_RESUME_PROMPT = (
    "The previous tool call returned its result.  "
    "Continue from where you left off without re-issuing the tool call."
)


def run_with_handoffs(
    *,
    workspace: Path,
    docker_command: list[str],
    prompt: str,
    block_runner: BlockRunner,
    resume_prompt: str = DEFAULT_RESUME_PROMPT,
    extra_env_per_iteration: (
        Callable[[int, Path | None], dict[str, str]] | None
    ) = None,
    resume_env_var: str = "RESUME_SESSION_FILE",
    resume_artifact_name: str = SESSION_ARTIFACT_NAME,
    max_iterations: int = 500,
    stdout_log: Path | None = None,
    stderr_log: Path | None = None,
    run_id: str | None = None,
) -> LoopResult:
    """Drive an agent container through ``tool_handoff`` cycles.

    Each iteration:
      1. Run ``docker_command`` (with ``RESUME_SESSION_FILE`` set
         after iteration 0) and feed ``prompt`` to its stdin.
      2. Wait for the container to exit.
      3. Read ``.agent_state.json`` from the workspace.
      4. If status is not ``tool_handoff``: return.
      5. Read ``pending_tool_calls.json``; for each pending entry
         (preserving the SDK's tool_use emission order) call
         ``block_runner`` and splice the real result into the
         workspace's session JSONL artifact; remove the pending
         file; loop.

    A single agent turn can contain multiple parallel tool_use
    blocks; the design choice is one block execution per tool_use
    (Option α), run sequentially within one stop/restart cycle.
    The ``HandoffContext.siblings`` field exposes the batch size
    so callers can decide whether to specialise behaviour for
    multi-tool-use turns.

    Args:
        workspace: Host directory bind-mounted as the agent's
            workspace.  All state (``.agent_state.json``,
            ``pending_tool_calls.json``, ``agent_session.jsonl``)
            lives here.
        docker_command: Full ``docker run`` command including image
            name and any -v / -e flags the caller wants.  The
            driver appends only the env vars it owns
            (``RESUME_SESSION_FILE``, plus anything from
            ``extra_env_per_iteration``) by re-issuing the command
            list with new ``-e`` flags inserted before the image.
            For this to work the docker_command MUST place the
            image name as the LAST element.
        prompt: Stdin payload for the agent's initial launch
            (iteration 0).
        block_runner: Caller-supplied function that runs the
            block whose tool was intercepted.  Receives a
            ``HandoffContext`` and returns a ``HandoffResult``.
        resume_prompt: Stdin payload for resume iterations
            (iteration >= 1).  The agent runner sends this as a
            new user query after restoring the spliced session, so
            it must not re-issue the original task — defaults to a
            short "continue from where you left off" instruction.
        extra_env_per_iteration: Optional hook to compute
            additional env vars for each iteration.  Called with
            ``(iteration, resume_artifact_path_or_None)``.  Useful
            for callers that want to gate behavior on whether
            this is a resume or a fresh start.
        resume_env_var: Name of the env var to set on resume
            iterations.  Defaults to ``RESUME_SESSION_FILE`` to
            match what the agent runner reads.
        resume_artifact_name: Filename within the workspace whose
            contents become the resume payload.  Defaults to the
            ``agent_session.jsonl`` artifact the runner exports on
            exit.
        max_iterations: Hard cap to prevent runaway loops.
        stdout_log: Optional file to append container stdout to;
            one combined log across iterations.
        stderr_log: Same for stderr.
        run_id: Optional :class:`flywheel.artifact.RunRecord` id
            stamped onto every :class:`HandoffContext` the driver
            builds so block runners can forward it into
            :meth:`flywheel.local_block.LocalBlockRecorder.begin`.
            ``None`` means ad-hoc (no grouping).

    Returns:
        ``LoopResult`` with one ``LoopIteration`` per container
        launch and the final state dict.

    Raises:
        HandoffLoopError: when the loop cannot proceed (image
            won't run, max iterations exceeded, splice fails,
            pending file malformed).
    """
    if not docker_command:
        raise HandoffLoopError("docker_command is empty")
    workspace.mkdir(parents=True, exist_ok=True)
    state_path = workspace / STATE_FILE_NAME
    pending_path = workspace / PENDING_FILE_NAME
    session_path = workspace / resume_artifact_name

    result = LoopResult()
    last_state: dict[str, Any] = {}

    for iteration in range(max_iterations):
        iter_extra_env: dict[str, str] = {}
        resume_path: Path | None = None
        if iteration > 0:
            if not session_path.is_file():
                raise HandoffLoopError(
                    f"resume requested but session artifact "
                    f"missing: {session_path}"
                )
            resume_path = session_path
            iter_extra_env[resume_env_var] = "/scratch/" + (
                resume_artifact_name
            )
        if extra_env_per_iteration is not None:
            iter_extra_env.update(
                extra_env_per_iteration(iteration, resume_path))

        cmd = list(docker_command)
        image = cmd.pop()
        for k, v in iter_extra_env.items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.append(image)

        env = os.environ.copy()
        env["MSYS_NO_PATHCONV"] = "1"

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except FileNotFoundError as exc:
            raise HandoffLoopError(
                f"docker command not found: {cmd[0]!r}") from exc

        stdout_t = (
            _drain(process.stdout, stdout_log)
            if stdout_log is not None else None
        )
        stderr_t = (
            _drain(process.stderr, stderr_log)
            if stderr_log is not None else None
        )

        iter_prompt = prompt if iteration == 0 else resume_prompt
        try:
            process.stdin.write(iter_prompt)
            process.stdin.close()
        except BrokenPipeError:
            pass

        exit_code = process.wait()
        if stdout_t is not None:
            stdout_t.join(timeout=2)
        if stderr_t is not None:
            stderr_t.join(timeout=2)

        last_state = _read_json(state_path)
        status = last_state.get("status", "")

        if status != "tool_handoff":
            result.iterations.append(LoopIteration(
                container_exit_code=exit_code,
                state=last_state,
            ))
            break

        pending_doc = _read_json(pending_path)
        pending_list = _normalize_pending(pending_doc)
        if not pending_list:
            raise HandoffLoopError(
                f"agent exited with tool_handoff but pending file "
                f"{pending_path.name} has no tool calls: {pending_doc}"
            )

        if not session_path.is_file():
            raise HandoffLoopError(
                f"agent exited with tool_handoff but no session "
                f"artifact at {session_path}; cannot splice"
            )

        siblings = len(pending_list)
        session_id_fallback = pending_doc.get(
            "session_id", last_state.get("session_id", ""))
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
                session_id=pending.get(
                    "session_id", session_id_fallback),
                tool_use_id=tool_use_id,
                workspace=workspace,
                iteration=iteration,
                index_in_iteration=idx,
                siblings=siblings,
                run_id=run_id,
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
                    f"splice failed for tool_use_id={tool_use_id} "
                    f"(iteration={iteration}, index={idx}): {exc}"
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

        result.iterations.append(LoopIteration(
            container_exit_code=exit_code,
            state=last_state,
            handoffs=handoff_records,
        ))

        # Clear the pending file so a leftover doesn't trick the
        # next iteration into thinking a new handoff is queued.
        with contextlib.suppress(FileNotFoundError):
            pending_path.unlink()
    else:
        raise HandoffLoopError(
            f"hit max_iterations={max_iterations} without a "
            f"non-handoff exit; last state: {last_state}"
        )

    result.final_state = last_state
    result.final_session_path = (
        session_path if session_path.is_file() else None
    )
    return result


def copy_session_artifact(src: Path, dst: Path) -> None:
    """Convenience: copy a session JSONL between workspaces.

    Tests (and pattern-runner code that wants a clean per-iteration
    snapshot) sometimes need to capture the JSONL between splices.
    This is the canonical way; ``shutil.copy2`` preserves mtime
    which keeps splice diagnostics ordered correctly.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
