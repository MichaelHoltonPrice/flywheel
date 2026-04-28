from __future__ import annotations

import json
import os
import subprocess
import textwrap
import uuid
from pathlib import Path

import pytest
import yaml

from flywheel.pattern_declaration import parse_pattern_declaration
from flywheel.pattern_execution import PatternRunError, run_pattern
from flywheel.workspace import Workspace
from tests._inline_blocks import from_yaml_with_inline_blocks

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.live_api
SENTINEL_SCORE = 91357
SCRATCH_SENTINEL = "SCRATCHPAD_SENTINEL_71429"


def test_claude_battery_tool_invoked_pipeline_with_real_containers(
    tmp_path: Path,
):
    """Run one real Claude-battery handoff/eval/resume pipeline."""
    if not _docker_available():
        pytest.fail("Docker is required for this live integration test")

    tag_suffix = uuid.uuid4().hex[:12]
    base_image = f"flywheel-claude-test-base:{tag_suffix}"
    agent_image = f"flywheel-claude-tool-pipeline-agent:{tag_suffix}"
    eval_image = f"flywheel-tool-pipeline-eval:{tag_suffix}"
    _build_claude_battery_image(base_image)
    if not _container_claude_auth_available(base_image):
        pytest.fail(
            "Claude auth is required. Set ANTHROPIC_API_KEY or populate "
            "the Docker volume 'claude-auth' with .credentials.json."
        )
    _build_agent_image(tmp_path / "agent-image", base_image, agent_image)
    _build_eval_image(tmp_path / "eval-image", eval_image)
    try:
        project_root = tmp_path / "project"
        project_root.mkdir()
        foundry = project_root / "foundry"
        template_path = project_root / "template.yaml"
        template_path.write_text(
            _template_yaml(agent_image=agent_image, eval_image=eval_image),
            encoding="utf-8",
        )
        template = from_yaml_with_inline_blocks(template_path)
        workspace = Workspace.create("ws", template, foundry)
        pattern = parse_pattern_declaration({
            "name": "improve",
            "do": [{
                "run_until": {
                    "name": "improve",
                    "block": "ImproveBot",
                    "continue_on": {"eval_requested": {"max": 2}},
                    "stop_on": ["normal"],
                },
            }],
        })

        try:
            result = run_pattern(workspace, pattern, template, project_root)
        except PatternRunError:
            reloaded_after_error = Workspace.load(workspace.path)
            if _has_claude_auth_failure(reloaded_after_error):
                pytest.fail(
                    "Claude authentication failed inside the battery "
                    "container. Refresh the 'claude-auth' Docker volume or "
                    "set ANTHROPIC_API_KEY before running this test."
                )
            raise

        reloaded = Workspace.load(workspace.path)
        run = reloaded.runs[result.run_id]
        step = run.steps[0]
        agent_execs = [
            reloaded.executions[member.execution_id]
            for member in step.members
            if member.execution_id
        ]
        assert [e.termination_reason for e in agent_execs] == [
            "eval_requested",
            "normal",
        ]
        assert agent_execs[0].output_bindings == {}
        assert agent_execs[0].state_snapshot_id is not None
        assert agent_execs[1].state_snapshot_id is not None
        assert reloaded.state_snapshots[
            agent_execs[1].state_snapshot_id
        ].predecessor_snapshot_id == agent_execs[0].state_snapshot_id
        first_state_dir = reloaded.state_snapshot_path(
            agent_execs[0].state_snapshot_id)
        second_state_dir = reloaded.state_snapshot_path(
            agent_execs[1].state_snapshot_id)
        _assert_clean_handoff_session(first_state_dir / "session.jsonl")
        _assert_clean_resumed_session(second_state_dir / "session.jsonl")
        _assert_session_state(first_state_dir)
        _assert_session_state(second_state_dir)
        _assert_session_telemetry(reloaded, agent_execs[0].id)
        _assert_session_telemetry(reloaded, agent_execs[1].id)
        _assert_scratchpad(first_state_dir)
        _assert_scratchpad(second_state_dir)

        invocation = reloaded.invocations[step.members[0].invocation_ids[0]]
        eval_execution = reloaded.executions[invocation.invoked_execution_id]
        assert eval_execution.termination_reason == "normal"
        assert agent_execs[1].input_bindings["score"] == (
            eval_execution.output_bindings["score"])

        final_bot_id = agent_execs[1].output_bindings["bot"]
        final_bot = reloaded.artifacts[final_bot_id]
        assert final_bot.copy_path is not None
        final_text = (
            reloaded.path / "artifacts" / final_bot.copy_path / "bot.txt"
        ).read_text(encoding="utf-8")
        assert "FINAL" in final_text
        assert f"score={SENTINEL_SCORE}" in final_text
        assert SCRATCH_SENTINEL in final_text

        usage_records = [
            record for record in reloaded.telemetry.values()
            if record.kind == "claude_usage"
        ]
        assert usage_records
        assert usage_records[-1].data.get("session_id")
    finally:
        subprocess.run(
            ["docker", "rmi", "-f", base_image, agent_image, eval_image],
            check=False,
            capture_output=True,
        )


def _template_yaml(*, agent_image: str, eval_image: str) -> str:
    env = {
        "MAX_TURNS": "8",
        "MCP_SERVER_MOUNT_DIR": "/app/agent/mcp_servers",
        "MCP_SERVERS": "test",
        "HANDOFF_TOOLS": "mcp__test__request_eval",
        "HANDOFF_TERMINATION_REASON": "eval_requested",
    }
    env["MODEL"] = os.environ.get(
        "FLYWHEEL_LIVE_TEST_MODEL",
        "claude-haiku-4-5-20251001",
    )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    data = {
        "name": "tool-pipeline",
        "artifacts": [
            {"name": "bot", "kind": "copy"},
            {"name": "score", "kind": "copy"},
        ],
        "blocks": [
            {
                "name": "ImproveBot",
                "image": agent_image,
                "docker_args": ["-v", "claude-auth:/home/claude/.claude:rw"],
                "env": env,
                "state": "managed",
                "inputs": [
                    {
                        "name": "score",
                        "container_path": "/input/score",
                        "optional": True,
                    },
                ],
                "outputs": {
                    "eval_requested": [],
                    "normal": [
                        {"name": "bot", "container_path": "/output/bot"},
                    ],
                },
                "on_termination": {
                    "eval_requested": {
                        "invoke": [{
                            "block": "EvalBot",
                        }],
                    },
                },
            },
            {
                "name": "EvalBot",
                "image": eval_image,
                "inputs": [],
                "outputs": {
                    "normal": [
                        {"name": "score", "container_path": "/output/score"},
                    ],
                },
            },
        ],
    }
    return yaml.safe_dump(data, sort_keys=False)


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except Exception:
        return False
    return True


def _has_claude_auth_failure(workspace: Workspace) -> bool:
    needle = "authentication"
    for rejection in workspace.telemetry_rejections.values():
        preserved = workspace.path / rejection.preserved_path
        if preserved.is_file():
            text = preserved.read_text(encoding="utf-8", errors="ignore")
            if needle in text.lower() or "401" in text:
                return True
    for record in workspace.telemetry.values():
        payload = json.dumps(record.data, default=str).lower()
        if record.kind == "claude_usage" and (
            "authentication" in payload or "401" in payload
        ):
            return True
    return False


def _assert_clean_handoff_session(session_path: Path) -> None:
    assert session_path.is_file()
    lines = _read_jsonl(session_path)
    text = session_path.read_text(encoding="utf-8", errors="ignore")
    assert "permission denied" not in text
    assert "permission error" not in text.lower()
    marker_index = _find_tool_result_line(lines, "Evaluation requested.")
    assert marker_index is not None
    assert _find_tool_result_line(lines, f"score: {SENTINEL_SCORE}") is None
    assert not _has_later_assistant_message(lines, marker_index)


def _assert_clean_resumed_session(session_path: Path) -> None:
    assert session_path.is_file()
    lines = _read_jsonl(session_path)
    text = session_path.read_text(encoding="utf-8", errors="ignore")
    assert "permission denied" not in text
    assert "permission error" not in text.lower()
    assert "Evaluation requested." in text
    assert "Score completed." not in text
    assert "/input/score/scores.json" in text
    assert _find_tool_result_line(lines, "Evaluation requested.") is not None
    assert _find_resume_message_line(lines, "/input/score/scores.json") is not None


def _assert_session_state(state_dir: Path) -> None:
    assert not (state_dir / "session_readback").exists()
    meta_path = state_dir / "session_meta.json"
    assert meta_path.is_file()
    active = json.loads(
        meta_path.read_text(encoding="utf-8"))
    assert active.get("session_id")
    assert active.get("cwd") == "/scratch"


def _assert_session_telemetry(
    workspace: Workspace, execution_id: str,
) -> None:
    records = [
        record for record in workspace.telemetry.values()
        if record.execution_id == execution_id
        and record.kind == "claude_session"
    ]
    assert records
    data = records[-1].data
    assert data.get("session_id")
    files = data.get("files", {})
    active_path = _telemetry_sidecar_path(
        workspace, execution_id, files["active_session"])
    assert active_path.is_file()
    messages_path = _telemetry_sidecar_path(
        workspace, execution_id, files["session_messages"]["path"])
    assert messages_path.is_file()
    assert not _telemetry_sidecar_path(
        workspace, execution_id, "/flywheel/telemetry/session_readback",
    ).exists()


def _telemetry_sidecar_path(
    workspace: Workspace, execution_id: str, container_path: str,
) -> Path:
    prefix = "/flywheel/telemetry/"
    assert container_path.startswith(prefix)
    rel = container_path[len(prefix):]
    return (
        workspace.path / "telemetry" / execution_id / rel
    )


def _assert_scratchpad(state_dir: Path) -> None:
    notes = state_dir / "scratchpad" / "notes.txt"
    assert notes.is_file()
    assert notes.read_text(encoding="utf-8").strip() == SCRATCH_SENTINEL


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _find_tool_result_line(
    rows: list[dict],
    needle: str,
) -> int | None:
    for idx, row in enumerate(rows):
        for block in _content_blocks(row):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            if _content_contains(block.get("content"), needle):
                return idx
    return None


def _find_resume_message_line(
    rows: list[dict],
    needle: str,
) -> int | None:
    for idx, row in enumerate(rows):
        if row.get("type") != "user":
            continue
        message = row.get("message")
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and needle in content:
            return idx
    return None


def _has_later_assistant_message(rows: list[dict], index: int) -> bool:
    for row in rows[index + 1:]:
        if row.get("type") == "assistant":
            return True
        message = row.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            return True
    return False


def _content_blocks(row: dict) -> list:
    message = row.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), list):
        return message["content"]
    content = row.get("content")
    if isinstance(content, list):
        return content
    return []


def _content_contains(content, needle: str) -> bool:
    if isinstance(content, str):
        return needle in content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and needle in text:
                    return True
            elif isinstance(block, str) and needle in block:
                return True
    return False


def _build_claude_battery_image(tag: str) -> None:
    battery_dir = ROOT / "batteries" / "claude"
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            tag,
            "-f",
            str(battery_dir / "Dockerfile.claude"),
            str(battery_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )


def _container_claude_auth_available(base_image: str) -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    volume = subprocess.run(
        ["docker", "volume", "inspect", "claude-auth"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if volume.returncode != 0:
        return False
    check = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            "claude-auth:/auth:ro",
            "--entrypoint",
            "python",
            base_image,
            "-c",
            (
                "from pathlib import Path; "
                "p=Path('/auth/.credentials.json'); "
                "raise SystemExit(0 if p.is_file() and p.stat().st_size else 1)"
            ),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    return check.returncode == 0


def _build_agent_image(path: Path, base_image: str, tag: str) -> None:
    mcp_dir = path / "mcp_servers"
    mcp_dir.mkdir(parents=True)
    (path / "prompt.md").write_text(_PROMPT, encoding="utf-8")
    (mcp_dir / "test_mcp_server.py").write_text(_MCP_SERVER, encoding="utf-8")
    (mcp_dir / "test_mcp_server.json").write_text(json.dumps({
        "tools": ["mcp__test__request_eval"],
    }), encoding="utf-8")
    (path / "Dockerfile").write_text(f"""\
FROM {base_image}
COPY prompt.md /app/agent/prompt.md
COPY mcp_servers/ /app/agent/mcp_servers/
RUN chown -R claude:claude /app/agent
""", encoding="utf-8")
    subprocess.run(
        ["docker", "build", "-t", tag, str(path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def _build_eval_image(path: Path, tag: str) -> None:
    path.mkdir(parents=True)
    (path / "Dockerfile").write_text("""\
FROM python:3.12-slim
COPY run.py /run.py
ENTRYPOINT ["python", "/run.py"]
""", encoding="utf-8")
    (path / "run.py").write_text(_EVAL_SCRIPT, encoding="utf-8")
    subprocess.run(
        ["docker", "build", "-t", tag, str(path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


_PROMPT = textwrap.dedent("""\
You are running a Flywheel integration test. Follow these rules exactly.

Your full task:
1. Before requesting evaluation, write
   `/scratch/.flywheel_scratchpad/notes.txt` with exactly:
   `SCRATCHPAD_SENTINEL_71429`.
2. Call the `mcp__test__request_eval` tool exactly once.
3. When the evaluation result is available at `/input/score/scores.json`,
   read `/input/score/scores.json`.
4. Read `/scratch/.flywheel_scratchpad/notes.txt`.
5. Write `/output/bot/bot.txt` with exactly this prefix:
   `FINAL score=<score from scores.json> scratchpad=<notes.txt content>`.

Do not write any output files before reading the score artifact.
Do not call `mcp__test__request_eval` more than once.
Do not call any tools after writing the final file.
""")


_MCP_SERVER = textwrap.dedent("""\
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test")

@mcp.tool()
def request_eval() -> str:
    return "Evaluation requested."

if __name__ == "__main__":
    mcp.run()
""")


_EVAL_SCRIPT = rf'''
import json
from pathlib import Path

out = Path("/output/score")
out.mkdir(parents=True, exist_ok=True)
(out / "scores.json").write_text(json.dumps({{"score": {SENTINEL_SCORE}}}), encoding="utf-8")
Path("/flywheel/termination").write_text("normal", encoding="utf-8")
'''
