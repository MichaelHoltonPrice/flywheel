"""Live-API + Docker test: handoff with block running in its own container.

Companion to ``test_full_stop_b1_container_roundtrip.py``.  That
test runs the agent in a container but the block (the thing the
intercepted tool maps to) inline in the test process.  This test
runs *both* containerized:

- Agent container: ``flywheel-claude:latest``, same as the
  companion test.
- Block container: ``python:3.12-slim`` running a tiny inline
  script that consumes the tool's JSON input on stdin and emits
  the real tool_result on stdout.

This validates the full-stop architecture end-to-end with no
flywheel process living on either side of the boundary except
the orchestrating driver: the agent decides what to do inside
its own container, the block executes inside its own container,
and the host only mediates by reading two JSON files and
splicing one JSONL.

Skipped by default; opt in with ``--run-live-api``.
"""

from __future__ import annotations

import json
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
BLOCK_IMAGE = "python:3.12-slim"
AUTH_VOLUME = "claude-auth"
HAIKU_MODEL = "claude-haiku-4-5"
TEST_TOOL = "mcp__splicedemo__double_it"
DENY_MARKER = "handoff_to_flywheel"


# --------------------------------------------------------------------
# Setup gates: re-checked here so this test can be run standalone.
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
    for img in (AGENT_IMAGE, BLOCK_IMAGE):
        if not _image_present(img):
            pytest.skip(
                f"image {img} not present (`docker pull {img}` "
                f"to install)"
            )
    if not _volume_present(AUTH_VOLUME):
        pytest.skip(
            f"docker volume {AUTH_VOLUME} not present; populate "
            f"with Anthropic credentials before running"
        )
    return info


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / f"agent-ws-{uuid.uuid4().hex[:8]}"
    ws.mkdir()
    return ws


# --------------------------------------------------------------------
# Inline MCP server, identical to the companion test.
# --------------------------------------------------------------------

_MCP_SERVER_SOURCE = textwrap.dedent("""\
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("splicedemo")

    @mcp.tool()
    def double_it(value: float) -> dict:
        return {"doubled": value * 2}

    if __name__ == "__main__":
        mcp.run()
""")
_MCP_MANIFEST = {"tools": [TEST_TOOL]}


def _provision_workspace(workspace: Path) -> None:
    mcp_dir = workspace / ".mcp_servers"
    mcp_dir.mkdir(parents=True, exist_ok=True)
    (mcp_dir / "splicedemo_mcp_server.py").write_text(
        _MCP_SERVER_SOURCE, encoding="utf-8")
    (mcp_dir / "splicedemo_mcp_server.json").write_text(
        json.dumps(_MCP_MANIFEST, indent=2), encoding="utf-8")


# --------------------------------------------------------------------
# Block runner: runs the block in its own python:3.12-slim container.
# --------------------------------------------------------------------

VERIFICATION_TOKEN = "FLYWHEEL-BLOCK-CONTAINER-V14-A82E"

# A minimal "block": reads the tool input from stdin as JSON,
# computes the doubled value, and prints a result line that
# includes a unique verification token.  The driver captures
# stdout verbatim and uses it as the splice payload.  Real blocks
# would be more elaborate; the splice contract is identical.
#
# The token exists so the test's success assertion can prove the
# spliced text actually flowed into the resumed agent's view of
# the conversation -- otherwise a smart model could compute 14 on
# its own and we couldn't tell whether it saw the splice.
_BLOCK_SCRIPT = textwrap.dedent(f"""\
    import json
    import sys

    payload = json.load(sys.stdin)
    value = payload.get("value", 0)
    doubled = value * 2
    print(
        f"Block computed: {{doubled}}.  "
        f"Verification token: {VERIFICATION_TOKEN}."
    )
""")


def _run_block_in_container(tool_input: dict) -> str:
    """Run the block script inside a python:3.12-slim container.

    The script reads ``tool_input`` from stdin and prints the
    result on stdout.  Container is ``--rm`` and gets no mounts;
    everything goes through stdio so the test's only host-side
    state is what comes back in the captured stdout.
    """
    cmd = [
        "docker", "run", "--rm", "-i",
        BLOCK_IMAGE, "python", "-c", _BLOCK_SCRIPT,
    ]
    proc = subprocess.run(
        cmd,
        input=json.dumps(tool_input),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"block container exited {proc.returncode}; "
            f"stderr: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _docker_command(workspace: Path) -> list[str]:
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

def test_handoff_with_block_in_separate_container(
    docker_ready: str,
    workspace: Path,
    tmp_path: Path,
) -> None:
    """Two-container end-to-end:

    - Agent container intercepts ``double_it`` via PreToolUse, exits.
    - Driver invokes the block runner.
    - Block runner spawns a *separate* python container that
      consumes the tool input on stdin and prints the result on
      stdout.  Result text crosses the container boundary as
      captured stdout.
    - Driver splices, restarts the agent container, agent echoes
      the verification token from the block container's output.

    The verification token is the success canary -- it has no
    source other than the block container's stdout, so the model
    emitting it is unambiguous evidence the splice + resume cycle
    delivered the block's real output to the model.
    """
    _provision_workspace(workspace)

    block_invocations: list[dict] = []

    def block_runner(ctx: HandoffContext) -> HandoffResult:
        out = _run_block_in_container(ctx.tool_input)
        block_invocations.append({
            "tool_input": ctx.tool_input,
            "stdout": out,
        })
        return HandoffResult(content=out, is_error=False)

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

    assert len(block_invocations) == 1, (
        f"expected exactly one block-container invocation, got "
        f"{len(block_invocations)}; "
        f"stderr tail: {_tail(stderr_log)}"
    )
    assert block_invocations[0]["tool_input"].get("value") == 7
    assert "Block computed: 14." in block_invocations[0]["stdout"]
    assert VERIFICATION_TOKEN in block_invocations[0]["stdout"]

    assert len(result.iterations) == 2
    assert result.iterations[0].state.get("status") == "tool_handoff"
    assert result.iterations[0].splice_line is not None
    assert result.iterations[-1].state.get("status") in {
        "complete", "stopped", "paused"}

    assert result.final_session_path is not None
    final_assistant_text = _last_assistant_text(
        result.final_session_path)
    assert VERIFICATION_TOKEN in final_assistant_text, (
        f"resumed model never echoed verification token "
        f"{VERIFICATION_TOKEN!r}.  Final assistant text: "
        f"{final_assistant_text!r}.  Stderr tail: "
        f"{_tail(stderr_log)}"
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
    if not path.is_file():
        return "<missing>"
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return "<empty>"
    return "\n".join(lines[-n:])
