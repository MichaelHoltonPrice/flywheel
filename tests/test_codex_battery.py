"""Tests for the bundled Codex agent battery."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEX_DIR = ROOT / "batteries" / "codex"


def _load_agent_runner(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("FLYWHEEL_TELEMETRY_DIR", str(tmp_path / "telemetry"))
    monkeypatch.setenv("FLYWHEEL_CODEX_EVENT_LOG", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv(
        "FLYWHEEL_CODEX_LAST_MESSAGE", str(tmp_path / "last.txt"))
    spec = importlib.util.spec_from_file_location(
        "codex_agent_runner", CODEX_DIR / "agent_runner.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_codex_dockerfile_installs_cli_and_uses_battery_contract():
    text = (CODEX_DIR / "Dockerfile.codex").read_text(encoding="utf-8")
    assert "npm install -g @openai/codex" in text
    assert "ripgrep" in text
    assert "ENV FLYWHEEL_AGENT_PROMPT=/app/agent/prompt.md" in text
    assert "ENV FLYWHEEL_SCRATCHPAD_DIR=/scratch/.flywheel_scratchpad" in text
    assert "COPY agent_runner.py /app/agent_runner.py" in text
    assert "COPY codex_handoff_hook.py /app/codex_handoff_hook.py" in text
    assert 'ENTRYPOINT ["/app/entrypoint.sh"]' in text


def test_codex_docs_use_supported_login_flow():
    readme = (ROOT / "examples" / "hello-codex" / "README.md").read_text(
        encoding="utf-8")
    spec = (ROOT / "docs" / "specs" / "codex-battery.md").read_text(
        encoding="utf-8")
    combined = readme + "\n" + spec
    assert "codex login --device-auth" in combined
    assert "codex login --with-api-key" in combined
    assert "host-codex" not in combined
    assert "USERPROFILE\\.codex" not in combined
    assert "cp -a /host-codex" not in combined


def test_codex_entrypoint_preserves_state_and_writes_termination():
    text = (CODEX_DIR / "entrypoint.sh").read_text(encoding="utf-8")
    assert "PERSISTED_SESSIONS=/flywheel/state/codex_sessions" in text
    assert "SCRATCHPAD_STATE_DIR=/flywheel/state/scratchpad" in text
    assert "RUNTIME_TELEMETRY_DIR=/tmp/flywheel-codex-telemetry" in text
    assert "chmod 700 /flywheel/state /flywheel/telemetry" in text
    assert "FLYWHEEL_TELEMETRY_DIR='$RUNTIME_TELEMETRY_DIR'" in text
    assert "python3 /app/agent_runner.py" in text
    assert 'cp -a "$RUNTIME_TELEMETRY_DIR"/. /flywheel/telemetry/' in text
    assert "echo \"$REASON\" > /flywheel/termination" in text


def test_agent_runner_default_event_log_is_not_entrypoint_capture_log():
    text = (CODEX_DIR / "agent_runner.py").read_text(encoding="utf-8")
    assert 'RUNNER_LOG=/tmp/flywheel-codex-runner.jsonl' not in text
    assert "/tmp/flywheel-codex-events.jsonl" in text


def test_agent_runner_builds_codex_exec_command(monkeypatch, tmp_path):
    module = _load_agent_runner(monkeypatch, tmp_path)
    monkeypatch.setenv("MODEL", "gpt-5.5")
    monkeypatch.setenv("REASONING_EFFORT", "high")
    monkeypatch.setenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", "200000")
    command = module._build_command("do work")
    assert command[:2] == ["codex", "exec"]
    assert "--json" in command
    assert "--output-last-message" in command
    assert command[command.index("--sandbox") + 1] == "danger-full-access"
    assert "--ask-for-approval" not in command
    assert "approval_policy=\"never\"" in command
    assert command[command.index("--model") + 1] == "gpt-5.5"
    assert "model_reasoning_effort=\"high\"" in command
    assert "model_auto_compact_token_limit=200000" in command
    assert command[command.index("--disable") + 1] == "shell_snapshot"
    assert command[-1] == "do work"


def test_agent_runner_rejects_invalid_auto_compact_limit(
        monkeypatch, tmp_path):
    module = _load_agent_runner(monkeypatch, tmp_path)
    monkeypatch.setenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", "0")
    try:
        module._build_command("do work")
    except ValueError as exc:
        assert (
            "CODEX_AUTO_COMPACT_TOKEN_LIMIT must be a positive integer"
            in str(exc)
        )
    else:
        raise AssertionError("expected invalid compact limit to fail")


def test_named_compact_limit_overrides_extra_args(monkeypatch, tmp_path):
    module = _load_agent_runner(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "CODEX_EXTRA_ARGS", "-c model_auto_compact_token_limit=900000")
    monkeypatch.setenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", "200000")
    command = module._build_command("do work")
    configured_index = command.index("model_auto_compact_token_limit=900000")
    named_index = command.index("model_auto_compact_token_limit=200000")
    assert named_index > configured_index


def test_agent_runner_can_enable_codex_shell_snapshot(monkeypatch, tmp_path):
    module = _load_agent_runner(monkeypatch, tmp_path)
    monkeypatch.setenv("CODEX_SHELL_SNAPSHOT", "true")
    command = module._build_command("do work")
    assert command[command.index("--enable") + 1] == "shell_snapshot"
    assert "--disable" not in command


def test_agent_runner_rejects_invalid_shell_snapshot_value(
        monkeypatch, tmp_path):
    module = _load_agent_runner(monkeypatch, tmp_path)
    monkeypatch.setenv("CODEX_SHELL_SNAPSHOT", "maybe")
    try:
        module._build_command("do work")
    except ValueError as exc:
        assert "CODEX_SHELL_SNAPSHOT must be a boolean" in str(exc)
    else:
        raise AssertionError("expected invalid shell snapshot flag to fail")


def test_agent_runner_classifies_known_rollout_warning(
        monkeypatch, tmp_path):
    module = _load_agent_runner(monkeypatch, tmp_path)
    assert module._is_benign_stderr(
        "2026-04-28T21:40:05.403814Z ERROR codex_core::session: "
        "failed to record rollout items: thread "
        "019dd608-bd5d-7db1-8109-1b0ecb71f865 not found\n"
    )
    assert not module._is_benign_stderr("fatal: something else failed\n")


def test_agent_runner_uses_resume_when_sessions_are_staged(
        monkeypatch, tmp_path):
    module = _load_agent_runner(monkeypatch, tmp_path)
    (tmp_path / "codex-home" / "sessions").mkdir(parents=True)
    command = module._build_command("continue")
    assert command[-3:] == ["resume", "--last", "Continue."]


def test_agent_runner_uses_flywheel_resume_prompt_when_resuming(
        monkeypatch, tmp_path):
    module = _load_agent_runner(monkeypatch, tmp_path)
    (tmp_path / "codex-home" / "sessions").mkdir(parents=True)
    monkeypatch.setenv(
        "FLYWHEEL_RESUME_PROMPT",
        "The evaluation result is mounted at /input/score/scores.json.",
    )
    command = module._build_command("original task")
    assert command[-3:] == [
        "resume",
        "--last",
        "The evaluation result is mounted at /input/score/scores.json.",
    ]


def test_agent_runner_ignores_resume_prompt_for_fresh_session(
        monkeypatch, tmp_path):
    module = _load_agent_runner(monkeypatch, tmp_path)
    monkeypatch.setenv("FLYWHEEL_RESUME_PROMPT", "resume-only context")
    command = module._build_command("original task")
    assert "resume" not in command
    assert command[-1] == "original task"


def test_agent_runner_writes_config_for_mounted_mcp_and_handoff(
        monkeypatch, tmp_path):
    module = _load_agent_runner(monkeypatch, tmp_path)
    mcp_dir = tmp_path / "mcp"
    mcp_dir.mkdir()
    (mcp_dir / "demo_mcp_server.py").write_text("print('x')\n")
    (mcp_dir / "demo_mcp_server.json").write_text(
        json.dumps({"tools": ["mcp__demo__finish"]}))
    monkeypatch.setenv("MCP_SERVER_MOUNT_DIR", str(mcp_dir))
    monkeypatch.setenv("MCP_SERVERS", "demo")
    configs = {
        "mcp__demo__finish": {
            "termination_reason": "done",
            "required_paths": ["/output/result"],
        },
    }
    module._write_codex_config(configs)
    text = (tmp_path / "codex-home" / "config.toml").read_text(
        encoding="utf-8")
    assert "[mcp_servers.demo]" in text
    assert "demo_mcp_server.py" in text
    assert "codex_hooks = true" in text
    assert "[[hooks.PostToolUse]]" in text
    assert "mcp__demo__finish" in text


def test_codex_handoff_hook_records_pending_metadata(tmp_path):
    pending = tmp_path / "pending.json"
    required = tmp_path / "result.txt"
    required.write_text("ok")
    env = os.environ.copy()
    env["FLYWHEEL_HANDOFF_PENDING"] = str(pending)
    env["HANDOFF_TOOL_CONFIG_NORMALIZED"] = json.dumps({
        "mcp__demo__finish": {
            "termination_reason": "finished",
            "required_paths": [str(required)],
            "result_path": "/input/result/result.json",
            "result_label": "Demo result",
            "placeholder_marker": "Finished.",
        }
    })
    hook_input = {
        "hook_event_name": "PostToolUse",
        "tool_name": "mcp__demo__finish",
        "tool_use_id": "call_1",
        "tool_input": {},
        "tool_response": {"content": "ok"},
    }
    proc = subprocess.run(
        [sys.executable, str(CODEX_DIR / "codex_handoff_hook.py")],
        input=json.dumps(hook_input),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    response = json.loads(proc.stdout)
    assert response["continue"] is False
    assert response["stopReason"] == "finished"
    recorded = json.loads(pending.read_text(encoding="utf-8"))
    assert recorded["tool_use_id"] == "call_1"
    assert recorded["tool_name"] == "mcp__demo__finish"
    assert recorded["result_label"] == "Demo result"


def test_agent_runner_smoke_with_fake_codex(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Write a short message.")
    workspace = tmp_path / "scratch"
    workspace.mkdir()
    telemetry = tmp_path / "telemetry"
    state = tmp_path / "state"
    state.mkdir()
    codex_home = tmp_path / "codex-home"
    fake_py = tmp_path / "fake_codex.py"
    fake_py.write_text(
        """
from __future__ import annotations
import json
import sys
from pathlib import Path

args = sys.argv[1:]
out = None
for i, arg in enumerate(args):
    if arg == "--output-last-message":
        out = Path(args[i + 1])
if out is not None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("fake final", encoding="utf-8")
print(json.dumps({"type": "thread.started", "thread_id": "thread_fake"}))
print(json.dumps({
    "type": "item.completed",
    "item": {"type": "agent_message", "text": "fake final"},
}))
print(json.dumps({
    "type": "turn.completed",
    "usage": {"input_tokens": 3, "cached_input_tokens": 1, "output_tokens": 2},
}))
""",
        encoding="utf-8",
    )
    fake_cmd = tmp_path / "fake_codex.cmd"
    fake_cmd.write_text(f'@echo off\r\n"{sys.executable}" "{fake_py}" %*\r\n')
    env = os.environ.copy()
    env.update({
        "FLYWHEEL_CODEX_FAKE": str(fake_cmd),
        "FLYWHEEL_AGENT_PROMPT": str(prompt),
        "AGENT_WORKSPACE": str(workspace),
        "CODEX_HOME": str(codex_home),
        "FLYWHEEL_TELEMETRY_DIR": str(telemetry),
        "FLYWHEEL_CODEX_EVENT_LOG": str(tmp_path / "events.jsonl"),
        "FLYWHEEL_CODEX_LAST_MESSAGE": str(tmp_path / "last.txt"),
    })
    proc = subprocess.run(
        [sys.executable, str(CODEX_DIR / "agent_runner.py")],
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    assert '"type": "agent_state"' in proc.stdout
    usage = json.loads(
        (telemetry / "codex_usage.json").read_text(encoding="utf-8"))
    assert usage["kind"] == "codex_usage"
    assert usage["source"] == "flywheel-codex"
    assert usage["data"]["thread_id"] == "thread_fake"
    assert usage["data"]["input_tokens"] == 3
    assert usage["data"]["cached_input_tokens"] == 1
    assert usage["data"]["output_tokens"] == 2
