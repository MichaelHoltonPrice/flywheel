"""Live-API + Docker test: full stop/restart loop with a real agent.

This is the integration test the in-process splice round-trip
demonstrator did NOT cover.  It launches the real
``flywheel-claude`` Docker container with the updated agent
runner, drives it through a tool-call handoff, runs the "block"
on the host, splices the result into the saved session JSONL,
restarts the container with the spliced JSONL on resume, and
asserts the resumed agent emits the expected text.

Compared to the in-process round-trip, this test additionally
validates:

- The PreToolUse hook works inside a containerized SDK and
  writes ``pending_tool_calls.json`` to the bind-mounted workspace.
- The agent runner exits cleanly on handoff with status
  ``tool_handoff`` and the session JSONL is exported as the
  ``agent_session.jsonl`` workspace artifact.
- The host-side ``run_with_handoffs`` driver detects the handoff,
  invokes the block runner, splices, and relaunches the
  container.
- The relaunched container reads ``RESUME_SESSION_FILE`` (set to
  the spliced JSONL) and the resumed agent treats the spliced
  tool_result as the genuine outcome of its original tool call.

Setup requirements (validated by ``test_setup_preconditions``,
which runs first):

- Docker daemon reachable.
- ``flywheel-claude:latest`` image present.
- ``claude-auth`` named volume present (provides Anthropic
  credentials inside the container).

Skipped by default; opt in with ``--run-live-api``.  The test
also short-circuits with a clear ``pytest.skip`` if any setup
precondition fails so a missing image / volume produces a
readable message rather than a Docker error wall.
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
TEST_TOOL = "mcp__splicedemo__double_it"
DENY_MARKER = "handoff_to_flywheel"


# --------------------------------------------------------------------
# Setup gates: skip cleanly when prerequisites are missing.
# --------------------------------------------------------------------

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


# --------------------------------------------------------------------
# Test MCP server: dropped into the workspace's .mcp_servers/ dir
# so the agent runner picks it up via its existing autodiscovery.
# --------------------------------------------------------------------

# Minimal stdio MCP server using the FastMCP class shipped with the
# `mcp` package (already installed in the flywheel-claude image).
# Exposes one tool, ``double_it(value: float)``.  In this test the
# PreToolUse hook denies the tool before it ever runs, but the SDK
# still requires the tool to be registered so the model knows it's
# callable.
_MCP_SERVER_SOURCE = textwrap.dedent("""\
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("splicedemo")

    @mcp.tool()
    def double_it(value: float) -> dict:
        return {"doubled": value * 2}

    if __name__ == "__main__":
        mcp.run()
""")

# Sidecar manifest read by the agent runner's MCP autodiscovery so
# the tool is on the allowed_tools list without requiring a handshake.
_MCP_MANIFEST = {"tools": [TEST_TOOL]}


def _provision_workspace(workspace: Path) -> None:
    """Drop the test MCP server + manifest into the workspace."""
    mcp_dir = workspace / ".mcp_servers"
    mcp_dir.mkdir(parents=True, exist_ok=True)
    (mcp_dir / "splicedemo_mcp_server.py").write_text(
        _MCP_SERVER_SOURCE, encoding="utf-8")
    (mcp_dir / "splicedemo_mcp_server.json").write_text(
        json.dumps(_MCP_MANIFEST, indent=2), encoding="utf-8")


def _docker_command(workspace: Path) -> list[str]:
    """Build the docker run command (image last, per driver contract)."""
    container_name = f"flywheel-test-{uuid.uuid4().hex[:8]}"
    host_ws = str(workspace).replace("\\", "/")
    return [
        "docker", "run", "--rm", "-i",
        "--name", container_name,
        "-v", f"{AUTH_VOLUME}:/home/claude/.claude",
        "-v", f"{host_ws}:/workspace",
        "-e", f"MODEL={HAIKU_MODEL}",
        "-e", "MCP_SERVERS=splicedemo",
        "-e", f"HANDOFF_TOOLS={TEST_TOOL}",
        "-e", f"HANDOFF_DENY_MARKER={DENY_MARKER}",
        "-e", f"ALLOWED_TOOLS={TEST_TOOL}",
        "-e", "MAX_TURNS=4",
        "-e", "PYTHONUNBUFFERED=1",
        AGENT_IMAGE,
    ]


# --------------------------------------------------------------------
# The test.
# --------------------------------------------------------------------

VERIFICATION_TOKEN = "FLYWHEEL-HANDOFF-7C3F-9K2D"


def test_container_handoff_roundtrip_with_haiku(
    docker_ready: str,
    workspace: Path,
    tmp_path: Path,
) -> None:
    """End-to-end with a real agent container:

    1.  Provision workspace with the splicedemo MCP server.
    2.  ``run_with_handoffs`` launches the agent container.
    3.  Haiku, prompted to call double_it on a number, invokes the tool.
    4.  Container's PreToolUse hook denies + writes
        ``pending_tool_calls.json``, status -> ``tool_handoff``,
        container exits.
    5.  Block runner returns a payload containing a unique
        verification token the model has no other way to know.
    6.  Splice rewrites the deny tool_result.
    7.  Driver relaunches the container with
        ``RESUME_SESSION_FILE=/workspace/agent_session.jsonl``.
    8.  Resumed agent emits a message that contains the token,
        which proves the spliced content reached the model
        (not just sat silently in the JSONL).

    Why a token instead of an arithmetic canary:  Haiku is smart
    enough to override an obviously-wrong tool result with its own
    math, so an assertion like ``"42" in final_text`` (with the
    splice content lying about ``2 * 7``) flakes when the model
    decides to disagree.  An opaque token has no source other than
    the splice, so the model emitting it is unambiguous evidence.
    """
    _provision_workspace(workspace)

    handoffs_seen: list[HandoffContext] = []

    def block_runner(ctx: HandoffContext) -> HandoffResult:
        handoffs_seen.append(ctx)
        return HandoffResult(
            content=(
                f"The tool ran successfully.  "
                f"Verification token: {VERIFICATION_TOKEN}.  "
                f"Doubled value: 14."
            ),
            is_error=False,
        )

    prompt = (
        "Use the double_it tool to double the number 7.  When the "
        "tool returns, repeat its verification token verbatim in "
        "your reply, then state the doubled value."
    )

    stdout_log = tmp_path / "agent_stdout.log"
    stderr_log = tmp_path / "agent_stderr.log"

    result = run_with_handoffs(
        workspace=workspace,
        docker_command=_docker_command(workspace),
        prompt=prompt,
        block_runner=block_runner,
        max_iterations=4,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )

    # --- One handoff happened, then a clean exit. ---
    assert len(handoffs_seen) == 1, (
        f"expected exactly one handoff, got {len(handoffs_seen)}; "
        f"stderr tail: {_tail(stderr_log)}"
    )
    assert handoffs_seen[0].tool_name == TEST_TOOL
    assert handoffs_seen[0].tool_input.get("value") == 7

    assert len(result.iterations) == 2, (
        f"expected 2 container launches; got "
        f"{len(result.iterations)}.  States: "
        f"{[it.state.get('status') for it in result.iterations]}"
    )

    handoff_iter = result.iterations[0]
    assert handoff_iter.state.get("status") == "tool_handoff"
    assert len(handoff_iter.handoffs) == 1
    record = handoff_iter.handoffs[0]
    assert record["splice_line"] is not None
    assert record["siblings"] == 1
    assert record["index_in_iteration"] == 0
    assert result.iterations[-1].handoffs == []

    final_iter = result.iterations[-1]
    assert final_iter.state.get("status") in {
        "complete", "stopped", "paused"}, (
        f"final agent state was unexpected: {final_iter.state}"
    )

    # --- The resumed agent echoed the verification token. ---
    assert result.final_session_path is not None
    final_session_text = result.final_session_path.read_text(
        encoding="utf-8")
    final_assistant_text = _last_assistant_text(
        result.final_session_path)
    assert VERIFICATION_TOKEN in final_assistant_text, (
        f"resumed model never echoed verification token "
        f"{VERIFICATION_TOKEN!r}.  Final assistant text: "
        f"{final_assistant_text!r}.  Spliced session contains "
        f"token? {VERIFICATION_TOKEN in final_session_text}.  "
        f"Stderr tail: {_tail(stderr_log)}"
    )


def _last_assistant_text(session_jsonl: Path) -> str:
    """Concatenate text from the last assistant message in the JSONL.

    Diagnostic helper used by the integration test; not part of the
    production splice contract.
    """
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


# Silence lint complaints about unused imports kept for downstream
# test files; ``shutil`` will be used by future block-in-container
# tests in this module.
_ = shutil
