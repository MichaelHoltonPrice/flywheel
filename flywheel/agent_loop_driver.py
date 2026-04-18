"""Drive an agent container through tool-call handoff cycles.

The flywheel-claude agent runner exposes a ``HANDOFF_TOOLS`` env
var.  When the agent invokes a matched MCP tool, a ``PreToolUse``
hook denies the call (recording a deny ``tool_result`` in the SDK
session JSONL), writes the tool's intent to
``pending_tool_call.json``, sets ``.agent_state.json`` status to
``tool_handoff``, and exits cleanly.  This module is the host-side
counterpart: it launches the agent container, waits for exit, and
on a handoff exit runs a caller-provided block, splices the real
result into the saved session JSONL, and restarts the container
with ``RESUME_SESSION_FILE`` pointing at the spliced JSONL.  The
loop continues until the agent exits without a pending handoff.

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

PENDING_FILE_NAME = "pending_tool_call.json"
STATE_FILE_NAME = ".agent_state.json"
SESSION_ARTIFACT_NAME = "agent_session.jsonl"


@dataclass
class HandoffContext:
    """What the block runner sees for a single handoff invocation.

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
        iteration: 0-based index of which handoff in this loop run
            we're on; useful for diagnostics and per-iteration
            artifact paths.
    """

    tool_name: str
    tool_input: dict[str, Any]
    session_id: str
    tool_use_id: str
    workspace: Path
    iteration: int


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
    """Per-iteration record returned in ``LoopResult.iterations``."""

    container_exit_code: int
    state: dict[str, Any]
    handoff: dict[str, Any] | None = None
    splice_line: int | None = None


@dataclass
class LoopResult:
    """Outcome of a full handoff-loop run.

    Attributes:
        iterations: One record per container launch, in order.
            The last one's ``handoff`` is ``None`` (the loop exits
            when an iteration completes without a handoff).
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
    max_iterations: int = 16,
    stdout_log: Path | None = None,
    stderr_log: Path | None = None,
) -> LoopResult:
    """Drive an agent container through ``tool_handoff`` cycles.

    Each iteration:
      1. Run ``docker_command`` (with ``RESUME_SESSION_FILE`` set
         after iteration 0) and feed ``prompt`` to its stdin.
      2. Wait for the container to exit.
      3. Read ``.agent_state.json`` from the workspace.
      4. If status is not ``tool_handoff``: return.
      5. Read ``pending_tool_call.json``; call ``block_runner``;
         splice the real result into the workspace's session
         JSONL artifact; remove the pending file; loop.

    Args:
        workspace: Host directory bind-mounted as the agent's
            workspace.  All state (``.agent_state.json``,
            ``pending_tool_call.json``, ``agent_session.jsonl``)
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
            iter_extra_env[resume_env_var] = "/workspace/" + (
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

        pending = _read_json(pending_path)
        tool_use_id = pending.get("tool_use_id")
        tool_name = pending.get("tool_name")
        if not isinstance(tool_use_id, str) or not tool_use_id:
            raise HandoffLoopError(
                f"pending_tool_call.json malformed (no tool_use_id): "
                f"{pending}"
            )
        if not isinstance(tool_name, str) or not tool_name:
            raise HandoffLoopError(
                f"pending_tool_call.json malformed (no tool_name): "
                f"{pending}"
            )

        ctx = HandoffContext(
            tool_name=tool_name,
            tool_input=pending.get("tool_input", {}) or {},
            session_id=pending.get(
                "session_id", last_state.get("session_id", "")),
            tool_use_id=tool_use_id,
            workspace=workspace,
            iteration=iteration,
        )

        block_result = block_runner(ctx)

        if not session_path.is_file():
            raise HandoffLoopError(
                f"agent exited with tool_handoff but no session "
                f"artifact at {session_path}; cannot splice"
            )

        try:
            splice_line = splice_tool_result(
                session_path,
                tool_use_id=tool_use_id,
                tool_result_content=block_result.content,
                is_error=block_result.is_error,
            )
        except SpliceError as exc:
            raise HandoffLoopError(
                f"splice failed for tool_use_id={tool_use_id}: {exc}"
            ) from exc

        result.iterations.append(LoopIteration(
            container_exit_code=exit_code,
            state=last_state,
            handoff={
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "tool_input": ctx.tool_input,
                "is_error": block_result.is_error,
            },
            splice_line=splice_line,
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
