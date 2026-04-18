"""Live-API + Docker test: multi-``tool_use`` handoff in a single turn.

Companion to ``test_full_stop_b1_container_roundtrip.py``.  That
test covers the single-call case (one ``tool_use`` per assistant
turn -> one pending entry -> one splice).  This test exercises the
case where the model emits *multiple* ``tool_use`` blocks in a
single assistant turn and every one of them targets a
handoff-mapped tool.

Specifically it validates the post-B1 implementation choice
(Option α + serial in v1, see
``plans/full-stop-state-contract.md``):

- The PreToolUse hook captures *every* intercepted ``tool_use``
  in the turn, not just the first one.
- The runner persists ``pending_tool_calls.json`` with
  ``schema_version`` 2 and a ``pending`` array containing one
  entry per intercepted call, in emission order.
- The host-side driver iterates that list, runs each block
  serially, and splices a real ``tool_result`` in for each
  ``tool_use_id``.
- The resumed container observes a session JSONL where every
  ``tool_use`` from the original turn now has a matching real
  ``tool_result``, and the model can continue normally.

Forcing Haiku into a parallel ``tool_use`` turn is not perfectly
deterministic; we use two *different* tools (``double_it`` and
``triple_it``) and an explicit "in parallel" instruction to bias
heavily toward parallel emission.  When the model nonetheless
chooses to emit them across two turns we surface a clear
``pytest.skip`` rather than a failure -- a one-tool-per-turn
behavior is itself valid (the single-call test already covers it)
and is not a regression in the splice/driver contract.

Skipped by default; opt in with ``--run-live-api``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
import uuid
from pathlib import Path

import pytest

from flywheel.agent_loop_driver import (
    HandoffContext,
    HandoffResult,
    run_with_handoffs,
)

pytestmark = pytest.mark.live_api

AGENT_IMAGE = "flywheel-claude:latest"
AUTH_VOLUME = "claude-auth"
HAIKU_MODEL = "claude-haiku-4-5"
TOOL_DOUBLE = "mcp__multidemo__double_it"
TOOL_TRIPLE = "mcp__multidemo__triple_it"
DENY_MARKER = "handoff_to_flywheel"


def _docker_available() -> tuple[bool, str]:
    try:
        out = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"docker not runnable: {exc}"
    if out.returncode != 0:
        return False, (
            f"docker daemon unreachable: {out.stderr.strip()}")
    return True, out.stdout.strip()


def _image_present(image: str) -> bool:
    out = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True, check=False,
    )
    return out.returncode == 0


def _volume_present(volume: str) -> bool:
    out = subprocess.run(
        ["docker", "volume", "inspect", volume],
        capture_output=True, text=True, check=False,
    )
    return out.returncode == 0


@pytest.fixture(scope="module")
def docker_ready() -> str:
    ok, info = _docker_available()
    if not ok:
        pytest.skip(info)
    if not _image_present(AGENT_IMAGE):
        pytest.skip(
            f"image {AGENT_IMAGE} not present; "
            f"build it from flywheel/batteries/claude/Dockerfile.claude"
        )
    if not _volume_present(AUTH_VOLUME):
        pytest.skip(
            f"docker volume {AUTH_VOLUME} not present; "
            f"populate with Anthropic credentials before running"
        )
    return info


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """A bind-mounted agent workspace for one test."""
    ws = tmp_path / f"agent-ws-{uuid.uuid4().hex[:8]}"
    ws.mkdir()
    return ws


# Two distinct tools so the model has a natural reason to call
# them in parallel rather than sequencing the same tool twice.
_MCP_SERVER_SOURCE = textwrap.dedent("""\
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("multidemo")

    @mcp.tool()
    def double_it(value: float) -> dict:
        return {"doubled": value * 2}

    @mcp.tool()
    def triple_it(value: float) -> dict:
        return {"tripled": value * 3}

    if __name__ == "__main__":
        mcp.run()
""")

_MCP_MANIFEST = {"tools": [TOOL_DOUBLE, TOOL_TRIPLE]}


def _provision_workspace(workspace: Path) -> None:
    """Drop the multidemo MCP server + manifest into the workspace."""
    mcp_dir = workspace / ".mcp_servers"
    mcp_dir.mkdir(parents=True, exist_ok=True)
    (mcp_dir / "multidemo_mcp_server.py").write_text(
        _MCP_SERVER_SOURCE, encoding="utf-8")
    (mcp_dir / "multidemo_mcp_server.json").write_text(
        json.dumps(_MCP_MANIFEST, indent=2), encoding="utf-8")


def _docker_command(workspace: Path) -> list[str]:
    """Build the docker run command (image last, per driver contract)."""
    container_name = f"flywheel-multitest-{uuid.uuid4().hex[:8]}"
    host_ws = str(workspace).replace("\\", "/")
    handoff_tools = f"{TOOL_DOUBLE},{TOOL_TRIPLE}"
    allowed_tools = f"{TOOL_DOUBLE},{TOOL_TRIPLE}"
    return [
        "docker", "run", "--rm", "-i",
        "--name", container_name,
        "-v", f"{AUTH_VOLUME}:/home/claude/.claude",
        "-v", f"{host_ws}:/workspace",
        "-e", f"MODEL={HAIKU_MODEL}",
        "-e", "MCP_SERVERS=multidemo",
        "-e", f"HANDOFF_TOOLS={handoff_tools}",
        "-e", f"HANDOFF_DENY_MARKER={DENY_MARKER}",
        "-e", f"ALLOWED_TOOLS={allowed_tools}",
        "-e", "MAX_TURNS=6",
        "-e", "PYTHONUNBUFFERED=1",
        AGENT_IMAGE,
    ]


# Distinct tokens so we can prove every spliced result reached the
# resumed model independently.
TOKEN_DOUBLE = "FLYWHEEL-MULTI-DOUBLE-A19F-2K7M"
TOKEN_TRIPLE = "FLYWHEEL-MULTI-TRIPLE-B83C-4P6N"


def test_multi_tool_use_handoff_with_haiku(
    docker_ready: str,
    workspace: Path,
    tmp_path: Path,
) -> None:
    """End-to-end multi-``tool_use`` round-trip:

    1.  Provision workspace with the multidemo MCP server (two
        tools, both handoff-mapped).
    2.  Prompt Haiku to call ``double_it(7)`` AND ``triple_it(5)``
        *in parallel* in its first response.
    3.  Container's PreToolUse hook denies and captures every
        intercepted ``tool_use`` for the turn; runner exits with
        ``status=tool_handoff`` and ``pending_tool_calls.json``
        carrying both entries.
    4.  Driver iterates the pending list serially; block_runner
        returns a payload containing a tool-specific verification
        token for each call; splice rewrites both deny
        ``tool_results`` in the saved JSONL.
    5.  Driver relaunches the container; resumed agent sees both
        real ``tool_results``; final reply contains both tokens.

    If Haiku elects to emit the two tool_use blocks across
    *separate* assistant turns instead of in parallel, the test
    skips with a clear message -- that path is already covered by
    the single-call test and is not a regression in the
    multi-tool-use code we want to validate here.
    """
    _provision_workspace(workspace)

    handoffs_seen: list[HandoffContext] = []

    def block_runner(ctx: HandoffContext) -> HandoffResult:
        handoffs_seen.append(ctx)
        if ctx.tool_name == TOOL_DOUBLE:
            return HandoffResult(
                content=(
                    f"The double_it tool ran successfully.  "
                    f"Verification token: {TOKEN_DOUBLE}.  "
                    f"Doubled value: 14."
                ),
                is_error=False,
            )
        if ctx.tool_name == TOOL_TRIPLE:
            return HandoffResult(
                content=(
                    f"The triple_it tool ran successfully.  "
                    f"Verification token: {TOKEN_TRIPLE}.  "
                    f"Tripled value: 15."
                ),
                is_error=False,
            )
        # Defensive: an unmapped tool reached the runner.  Surface
        # as an error result so the splice still succeeds and the
        # test fails on the assertions below rather than hanging.
        return HandoffResult(
            content=f"unknown tool {ctx.tool_name!r}",
            is_error=True,
        )

    prompt = (
        "Task: I need both 2*7 and 3*5 computed by tools, in one "
        "round-trip, so I can present both numbers together.\n\n"
        "Use the two MCP tools available to you:\n"
        "  - double_it(value=7)\n"
        "  - triple_it(value=5)\n\n"
        "MUST: emit BOTH tool_use blocks in the SAME single "
        "assistant message (parallel tool calling).  Do NOT "
        "issue one tool call, wait for the result, then issue "
        "the other -- that wastes a round-trip.  Both tools "
        "are independent; there is no reason to serialize them.\n\n"
        "Once both tool results come back, write your FINAL "
        "reply.  In that final reply you MUST:\n"
        "  - quote the double_it tool's verification token "
        "    verbatim,\n"
        "  - quote the triple_it tool's verification token "
        "    verbatim,\n"
        "  - then state the doubled value and the tripled value."
    )

    stdout_log = tmp_path / "agent_stdout.log"
    stderr_log = tmp_path / "agent_stderr.log"

    result = run_with_handoffs(
        workspace=workspace,
        docker_command=_docker_command(workspace),
        prompt=prompt,
        block_runner=block_runner,
        max_iterations=6,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )

    if not handoffs_seen:
        pytest.fail(
            f"agent never invoked a handoff-mapped tool; "
            f"states: {[it.state.get('status') for it in result.iterations]}; "
            f"stderr tail: {_tail(stderr_log)}"
        )

    # Find the iteration that actually exercised multi-tool-use.
    multi_iters = [
        it for it in result.iterations
        if len(it.handoffs) >= 2
    ]
    if not multi_iters:
        max_siblings = max(
            (len(it.handoffs) for it in result.iterations), default=0)
        pytest.skip(
            f"model did not emit parallel tool_use blocks "
            f"(max siblings observed in any iteration: {max_siblings}); "
            f"single-call path is covered by "
            f"test_full_stop_b1_container_roundtrip.  "
            f"Handoff tool names seen: "
            f"{[h.tool_name for h in handoffs_seen]}"
        )

    multi_iter = multi_iters[0]
    handoffs = multi_iter.handoffs
    assert multi_iter.state.get("status") == "tool_handoff"
    assert len(handoffs) >= 2, handoffs
    siblings = handoffs[0]["siblings"]
    assert siblings == len(handoffs), (
        f"siblings={siblings} but pending list has "
        f"{len(handoffs)} entries"
    )
    indices = [h["index_in_iteration"] for h in handoffs]
    assert indices == list(range(len(handoffs))), (
        f"index_in_iteration must be 0..N-1 in order; got {indices}"
    )
    tool_use_ids = [h["tool_use_id"] for h in handoffs]
    assert len(set(tool_use_ids)) == len(tool_use_ids), (
        f"tool_use_ids must be unique within a turn; got {tool_use_ids}"
    )
    splice_lines = [h["splice_line"] for h in handoffs]
    assert all(s is not None for s in splice_lines), splice_lines
    tool_names = {h["tool_name"] for h in handoffs}
    assert TOOL_DOUBLE in tool_names and TOOL_TRIPLE in tool_names, (
        f"expected both double_it and triple_it in the parallel turn; "
        f"got {tool_names}"
    )

    # Driver must call block_runner once per pending entry in this
    # iteration, with siblings/index_in_iteration matching what the
    # iteration record exposes.
    invocations_in_iter = [
        h for h in handoffs_seen
        if h.tool_use_id in set(tool_use_ids)
    ]
    assert len(invocations_in_iter) == len(tool_use_ids), (
        f"block_runner should have been called once per pending entry "
        f"({len(tool_use_ids)}); got {len(invocations_in_iter)}.  "
        f"Seen tool_use_ids: {[h.tool_use_id for h in handoffs_seen]}"
    )
    for invocation in invocations_in_iter:
        assert invocation.siblings == siblings, (
            f"siblings mismatch in HandoffContext: "
            f"{invocation.siblings} vs {siblings}"
        )

    # Final iteration should be a clean, handoff-free exit.
    final_iter = result.iterations[-1]
    assert final_iter.handoffs == []
    assert final_iter.state.get("status") in {
        "complete", "stopped", "paused"}, (
        f"final agent state was unexpected: {final_iter.state}"
    )

    assert result.final_session_path is not None
    final_session_text = result.final_session_path.read_text(
        encoding="utf-8")
    final_assistant_text = _last_assistant_text(
        result.final_session_path)
    assert TOKEN_DOUBLE in final_assistant_text, (
        f"resumed model never echoed double_it token "
        f"{TOKEN_DOUBLE!r}.  Final assistant text: "
        f"{final_assistant_text!r}.  Spliced session contains "
        f"token? {TOKEN_DOUBLE in final_session_text}.  "
        f"Stderr tail: {_tail(stderr_log)}"
    )
    assert TOKEN_TRIPLE in final_assistant_text, (
        f"resumed model never echoed triple_it token "
        f"{TOKEN_TRIPLE!r}.  Final assistant text: "
        f"{final_assistant_text!r}.  Spliced session contains "
        f"token? {TOKEN_TRIPLE in final_session_text}.  "
        f"Stderr tail: {_tail(stderr_log)}"
    )


def _last_assistant_text(session_jsonl: Path) -> str:
    """Concatenate text from the last assistant message in the JSONL."""
    last_text = ""
    for raw in session_jsonl.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "assistant":
            continue
        msg = obj.get("message")
        content = (
            msg.get("content")
            if isinstance(msg, dict) else obj.get("content")
        )
        if not isinstance(content, list):
            continue
        chunks = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str):
                chunks.append(text)
        if chunks:
            last_text = "\n".join(chunks)
    return last_text


def _tail(path: Path, n: int = 40) -> str:
    """Return the last ``n`` lines of a log file, or ``"<empty>"``."""
    if not path.is_file():
        return "<missing>"
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return "<empty>"
    return "\n".join(lines[-n:])


_ = shutil
