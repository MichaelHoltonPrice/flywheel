from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from flywheel.config import load_project_config
from flywheel.container import ContainerResult
from flywheel.pattern_declaration import parse_pattern_declaration
from flywheel.pattern_execution import run_pattern
from flywheel.template import Template
from flywheel.workspace import Workspace
from tests._inline_blocks import from_yaml_with_inline_blocks

ROOT = Path(__file__).resolve().parents[1]


def test_hello_agent_template_keeps_project_env_minimal():
    project_root = ROOT / "examples" / "hello-agent"
    config = load_project_config(project_root)
    template = Template.from_yaml(
        config.workspace_templates_dir / "hello-agent.yaml",
        block_registry=config.load_block_registry(),
    )

    block = template.blocks[0]

    assert block.env == {"MAX_TURNS": "1"}


def test_claude_battery_image_owns_runtime_env_defaults():
    dockerfile = ROOT / "batteries" / "claude" / "Dockerfile.claude"
    text = dockerfile.read_text(encoding="utf-8")

    assert "ENV PYTHONUNBUFFERED=1" in text
    assert "ENV CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000" in text
    assert "ENV FLYWHEEL_AGENT_PROMPT=/app/agent/prompt.md" in text
    assert "COPY handoff_session.py /app/handoff_session.py" in text
    assert "/flywheel/control" not in text


def test_hello_agent_image_bakes_prompt_into_battery_convention():
    dockerfile = ROOT / "examples" / "hello-agent" / (
        "Dockerfile.hello-agent"
    )
    text = dockerfile.read_text(encoding="utf-8")

    assert "FROM flywheel-claude:latest" in text
    assert "COPY prompt/prompt.md /app/agent/prompt.md" in text


def test_claude_battery_entrypoint_creates_framework_subdirs():
    entrypoint = ROOT / "batteries" / "claude" / "entrypoint.sh"
    text = entrypoint.read_text(encoding="utf-8")

    assert "mkdir -p /flywheel/mcp_servers /flywheel/telemetry" in text
    assert "/flywheel/control" not in text
    assert "termination_request" not in text
    assert "chown root:root /flywheel/telemetry" in text
    assert "chmod 700 /flywheel/telemetry" in text


def test_claude_battery_entrypoint_persists_agent_scratchpad():
    dockerfile = ROOT / "batteries" / "claude" / "Dockerfile.claude"
    entrypoint = ROOT / "batteries" / "claude" / "entrypoint.sh"
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    entrypoint_text = entrypoint.read_text(encoding="utf-8")

    assert "ENV FLYWHEEL_SCRATCHPAD_DIR=/scratch/.flywheel_scratchpad" in (
        dockerfile_text)
    assert "SCRATCHPAD_STATE_DIR=/flywheel/state/scratchpad" in entrypoint_text
    assert "export FLYWHEEL_SCRATCHPAD_DIR=" in entrypoint_text
    assert "refusing unsafe FLYWHEEL_SCRATCHPAD_DIR" in entrypoint_text
    assert "FLYWHEEL_SCRATCHPAD_DIR must be under /scratch" in entrypoint_text
    assert 'cp -a "$SCRATCHPAD_STATE_DIR"/. "$SCRATCHPAD_RUNTIME_DIR"/' in (
        entrypoint_text)
    assert 'cp -a "$SCRATCHPAD_RUNTIME_DIR"/. "$SCRATCHPAD_STATE_DIR"/' in (
        entrypoint_text)
    assert 'chown -R claude:claude "$SCRATCHPAD_RUNTIME_DIR"' in (
        entrypoint_text)
    assert 'chown -R root:root "$SCRATCHPAD_STATE_DIR"' in entrypoint_text


def test_claude_battery_writes_usage_telemetry():
    runner = ROOT / "batteries" / "claude" / "agent_runner.py"
    entrypoint = ROOT / "batteries" / "claude" / "entrypoint.sh"
    runner_text = runner.read_text(encoding="utf-8")
    entrypoint_text = entrypoint.read_text(encoding="utf-8")

    assert "claude_result.json" not in runner_text
    assert "/tmp/flywheel-claude-runner.jsonl" in entrypoint_text
    assert "/flywheel/telemetry/claude_usage.json" in entrypoint_text
    assert "python3 /app/handoff_session.py" in entrypoint_text
    assert '"kind": "claude_usage"' in entrypoint_text
    assert '"source": "flywheel-claude"' in entrypoint_text


def test_claude_battery_separates_resume_state_from_session_telemetry():
    entrypoint = ROOT / "batteries" / "claude" / "entrypoint.sh"
    text = entrypoint.read_text(encoding="utf-8")

    assert 'PERSISTED_SESSION=/flywheel/state/session.jsonl' in text
    assert 'state_meta_path = Path("/flywheel/state/session_meta.json")' in text
    assert 'snapshot_dir = Path("/flywheel/telemetry/session")' in text
    assert (
        'telemetry_index_path = Path("/flywheel/telemetry/claude_session.json")'
        in text
    )
    assert 'Path("/flywheel/state/session_readback")' not in text
    assert '"kind": "claude_session"' in text


def _load_battery_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claude_battery_splices_placeholder_handoff_result_into_session(
    tmp_path: Path,
):
    module = _load_battery_module(
        "handoff_session",
        ROOT / "batteries" / "claude" / "handoff_session.py",
    )

    session = tmp_path / "session.jsonl"
    session.write_text(json.dumps({
        "message": {
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [{
                    "type": "text",
                    "text": "Evaluation requested.",
                }],
                "is_error": False,
            }],
        },
        "toolUseResult": "{\"result\":\"Evaluation requested.\"}",
        "mcpMeta": {
            "structuredContent": {"result": "Evaluation requested."},
        },
    }) + "\n", encoding="utf-8")
    score = tmp_path / "scores.json"
    score.write_text(json.dumps({
        "mean": 14.2,
        "median": 13.0,
        "episodes": 4000,
        "errors": 0,
    }), encoding="utf-8")

    count = module.splice_handoff_results(
        session,
        result_path=score,
        result_label="Evaluation",
    )

    assert count == 1
    payload = json.loads(session.read_text(encoding="utf-8"))
    block = payload["message"]["content"][0]
    assert block["is_error"] is False
    text = block["content"][0]["text"]
    assert "Evaluation completed." in text
    assert "mean: 14.2" in text
    assert "episodes: 4000" in text
    assert "handoff_to_flywheel" not in text
    assert "Evaluation requested." not in text
    assert "Evaluation requested." not in payload["toolUseResult"]
    assert "Evaluation requested." not in payload["mcpMeta"][
        "structuredContent"]["result"]


def test_claude_battery_ignores_legacy_deny_handoff_result(
    tmp_path: Path,
):
    module = _load_battery_module(
        "handoff_session",
        ROOT / "batteries" / "claude" / "handoff_session.py",
    )

    session = tmp_path / "session.jsonl"
    session.write_text(json.dumps({
        "message": {
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "permission denied: handoff_to_flywheel",
                "is_error": True,
            }],
        },
    }) + "\n", encoding="utf-8")
    score = tmp_path / "scores.json"
    score.write_text(json.dumps({"mean": 3.5}), encoding="utf-8")

    count = module.splice_handoff_results(
        session,
        result_path=score,
        result_label="Evaluation",
    )

    assert count == 0
    block = json.loads(session.read_text(encoding="utf-8"))[
        "message"]["content"][0]
    assert block["is_error"] is True
    assert block["content"] == "permission denied: handoff_to_flywheel"


def test_claude_battery_splices_missing_result_as_error(tmp_path: Path):
    module = _load_battery_module(
        "handoff_session",
        ROOT / "batteries" / "claude" / "handoff_session.py",
    )

    session = tmp_path / "session.jsonl"
    session.write_text(json.dumps({
        "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "Evaluation requested.",
            "is_error": False,
        }],
    }), encoding="utf-8")

    count = module.splice_handoff_results(
        session,
        result_path=tmp_path / "missing.json",
        result_label="Evaluation",
    )

    assert count == 1
    block = json.loads(session.read_text(encoding="utf-8"))["content"][0]
    assert block["is_error"] is True
    assert "did not produce a result artifact" in block["content"][0]["text"]


def test_claude_battery_splices_newest_handoff_by_default(
    tmp_path: Path,
):
    module = _load_battery_module(
        "handoff_session",
        ROOT / "batteries" / "claude" / "handoff_session.py",
    )
    session = tmp_path / "session.jsonl"
    session.write_text(
        "\n".join([
            json.dumps({"content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_old",
                "content": "Evaluation requested.",
                "is_error": False,
            }]}),
            json.dumps({"content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_new",
                "content": "Evaluation requested.",
                "is_error": False,
            }]}),
        ]),
        encoding="utf-8",
    )
    score = tmp_path / "scores.json"
    score.write_text(json.dumps({"mean": 88}), encoding="utf-8")

    count = module.splice_handoff_results(
        session,
        result_path=score,
        result_label="Evaluation",
    )

    assert count == 1
    lines = [
        json.loads(line)
        for line in session.read_text(encoding="utf-8").splitlines()
    ]
    assert lines[0]["content"][0]["is_error"] is False
    assert lines[0]["content"][0]["content"] == "Evaluation requested."
    assert lines[1]["content"][0]["is_error"] is False
    assert "mean: 88" in lines[1]["content"][0]["content"][0]["text"]


def test_claude_battery_splices_exact_tool_use_id(tmp_path: Path):
    module = _load_battery_module(
        "handoff_session",
        ROOT / "batteries" / "claude" / "handoff_session.py",
    )
    session = tmp_path / "session.jsonl"
    session.write_text(
        "\n".join([
            json.dumps({"content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_old",
                "content": "Evaluation requested.",
                "is_error": False,
            }]}),
            json.dumps({"content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_new",
                "content": "Evaluation requested.",
                "is_error": False,
            }]}),
        ]),
        encoding="utf-8",
    )
    score = tmp_path / "scores.json"
    score.write_text(json.dumps({"mean": 77}), encoding="utf-8")

    count = module.splice_handoff_results(
        session,
        result_path=score,
        result_label="Evaluation",
        tool_use_id="toolu_old",
    )

    assert count == 1
    lines = [
        json.loads(line)
        for line in session.read_text(encoding="utf-8").splitlines()
    ]
    assert lines[0]["content"][0]["is_error"] is False
    assert "mean: 77" in lines[0]["content"][0]["content"][0]["text"]
    assert lines[1]["content"][0]["is_error"] is False
    assert lines[1]["content"][0]["content"] == "Evaluation requested."


def test_claude_battery_splicer_ignores_non_placeholder_hits(
    tmp_path: Path,
):
    module = _load_battery_module(
        "handoff_session",
        ROOT / "batteries" / "claude" / "handoff_session.py",
    )
    session = tmp_path / "session.jsonl"
    original = {
        "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": [{
                "type": "text",
                "text": "stdout happened to mention handoff_to_flywheel",
            }],
            "is_error": False,
        }],
    }
    session.write_text(json.dumps(original), encoding="utf-8")
    score = tmp_path / "scores.json"
    score.write_text(json.dumps({"mean": 99}), encoding="utf-8")

    count = module.splice_handoff_results(
        session,
        result_path=score,
        result_label="Evaluation",
    )

    assert count == 0
    assert json.loads(session.read_text(encoding="utf-8")) == original


def test_claude_battery_splicer_preserves_session_mode(tmp_path: Path):
    if os.name == "nt":
        pytest.skip("Windows chmod does not preserve POSIX mode bits")
    module = _load_battery_module(
        "handoff_session",
        ROOT / "batteries" / "claude" / "handoff_session.py",
    )
    session = tmp_path / "session.jsonl"
    session.write_text(json.dumps({
        "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "Evaluation requested.",
            "is_error": False,
        }],
    }), encoding="utf-8")
    os.chmod(session, 0o640)
    score = tmp_path / "scores.json"
    score.write_text(json.dumps({"mean": 1}), encoding="utf-8")

    module.splice_handoff_results(
        session,
        result_path=score,
        result_label="Evaluation",
    )

    assert session.stat().st_mode & 0o777 == 0o640


def test_claude_battery_handoff_post_hook_requires_paths(tmp_path: Path):
    module = _load_battery_module(
        "agent_runner",
        ROOT / "batteries" / "claude" / "agent_runner.py",
    )
    module._HANDOFF_STATE["pending"] = []
    hook = module._build_handoff_post_hook(
        {"mcp__cyberloop__request_eval"},
        [tmp_path / "missing.py"],
    )

    result = asyncio.run(hook({
        "tool_name": "mcp__cyberloop__request_eval",
        "tool_input": {},
    }, "toolu_1", None))

    assert len(module._HANDOFF_STATE["pending"]) == 1
    assert module._HANDOFF_STATE["pending"][0]["tool_use_id"] == "toolu_1"
    assert module._HANDOFF_STATE["pending"][0]["missing_required_paths"]
    assert result["continue"] is False
    reason = result["stopReason"]
    assert "missing required path" in reason
    assert "handoff_to_flywheel" not in reason


def test_claude_battery_handoff_post_hook_captures_first_only(
    tmp_path: Path,
):
    module = _load_battery_module(
        "agent_runner",
        ROOT / "batteries" / "claude" / "agent_runner.py",
    )
    module._HANDOFF_STATE["pending"] = []
    bot = tmp_path / "bot.py"
    bot.write_text("def player_fn(env, obs_json, action_labels): return 0")
    hook = module._build_handoff_post_hook(
        {"mcp__cyberloop__request_eval"},
        [bot],
    )

    first = asyncio.run(hook({
        "tool_name": "mcp__cyberloop__request_eval",
        "tool_input": {"artifact_path": "bot.py"},
    }, "toolu_1", None))
    second = asyncio.run(hook({
        "tool_name": "mcp__cyberloop__request_eval",
        "tool_input": {},
    }, "toolu_2", None))

    assert len(module._HANDOFF_STATE["pending"]) == 1
    assert module._HANDOFF_STATE["pending"][0]["tool_use_id"] == "toolu_1"
    assert first["continue"] is False
    assert first["stopReason"] == "handoff_to_flywheel"
    second_reason = second["stopReason"]
    assert "already pending" in second_reason


def test_claude_battery_handoff_post_hook_records_per_tool_metadata(
    tmp_path: Path,
):
    module = _load_battery_module(
        "agent_runner",
        ROOT / "batteries" / "claude" / "agent_runner.py",
    )
    module._HANDOFF_STATE["pending"] = []
    prediction_request = tmp_path / "prediction_request.json"
    prediction_request.write_text("{}", encoding="utf-8")
    hook = module._build_handoff_post_hook({
        "mcp__arc__predict_action": {
            "termination_reason": "prediction_requested",
            "required_paths": [prediction_request],
            "result_path": "/input/prediction/prediction.json",
            "result_label": "Prediction",
            "placeholder_marker": "Prediction requested.",
        },
        "mcp__arc__take_action": {
            "termination_reason": "action_requested",
            "required_paths": [],
            "result_path": "/input/game_step/game_step.json",
            "result_label": "Action",
            "placeholder_marker": "Action requested.",
        },
    })

    result = asyncio.run(hook({
        "tool_name": "mcp__arc__predict_action",
        "tool_input": {"action": 1},
    }, "toolu_predict", None))

    assert result["continue"] is False
    pending = module._HANDOFF_STATE["pending"][0]
    assert pending["tool_use_id"] == "toolu_predict"
    assert pending["termination_reason"] == "prediction_requested"
    assert pending["result_path"] == "/input/prediction/prediction.json"
    assert pending["result_label"] == "Prediction"
    assert pending["placeholder_marker"] == "Prediction requested."


def test_eval_handoff_resumes_with_result_artifact_prompt(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    foundry = project_root / "foundry"
    template_path = project_root / "template.yaml"
    template_path.write_text("""\
artifacts:
  - name: bot
    kind: copy
  - name: score
    kind: copy

blocks:
  - name: ImproveBot
    image: agent:latest
    state: managed
    inputs:
      - name: bot
        container_path: /input/bot
      - name: score
        container_path: /input/score
        optional: true
    outputs:
      eval_requested:
        - name: bot
          container_path: /output/bot
      normal:
        - name: bot
          container_path: /output/bot
    on_termination:
      eval_requested:
        invoke:
          - block: EvalBot
            bind:
              bot: bot
  - name: EvalBot
    image: eval:latest
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      normal:
        - name: score
          container_path: /output/score
""")
    template = from_yaml_with_inline_blocks(template_path)
    workspace = Workspace.create("ws", template, foundry)
    fixture = project_root / "assets" / "bot"
    fixture.mkdir(parents=True)
    (fixture / "bot.py").write_text("BASE", encoding="utf-8")
    pattern = parse_pattern_declaration({
        "name": "improve",
        "lanes": ["A"],
        "fixtures": {"bot": "assets/bot"},
        "do": [
            {"name": "improve_1", "cohort": {
                "foreach": "lanes",
                "block": "ImproveBot",
            }},
            {"name": "improve_2", "cohort": {
                "foreach": "lanes",
                "block": "ImproveBot",
            }},
        ],
    })
    agent_runs = 0

    def fake_container(config, args=None):
        del args
        nonlocal agent_runs
        mounts = {
            container: Path(host)
            for host, container, _mode in config.mounts
        }
        flywheel_dir = mounts["/flywheel"]
        if config.image == "agent:latest":
            agent_runs += 1
            bot_out = mounts["/output/bot"] / "bot.py"
            bot_out.parent.mkdir(parents=True, exist_ok=True)
            if agent_runs == 1:
                assert (mounts["/input/bot"] / "bot.py").read_text(
                    encoding="utf-8") == "BASE"
                bot_out.write_text("BOT1", encoding="utf-8")
                session = flywheel_dir / "state" / "session.jsonl"
                session.parent.mkdir(parents=True, exist_ok=True)
                session.write_text(json.dumps({
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_eval",
                    "content": [{
                        "type": "text",
                        "text": "Evaluation requested.",
                    }],
                    "is_error": False,
                }],
            }) + "\n", encoding="utf-8")
                (flywheel_dir / "termination").write_text(
                    "eval_requested", encoding="utf-8")
                return ContainerResult(exit_code=0, elapsed_s=0.1)

            score_path = mounts["/input/score"] / "scores.json"
            assert score_path.is_file()
            session = flywheel_dir / "state" / "session.jsonl"
            assert session.is_file()
            resume_prompt = config.env.get("FLYWHEEL_RESUME_PROMPT", "")
            assert "/input/score/scores.json" in resume_prompt
            assert "Continue the original task." in resume_prompt
            text = session.read_text(encoding="utf-8")
            assert "permission denied: handoff_to_flywheel" not in text
            assert "Evaluation requested." in text
            assert "mean: 42.0" not in text
            bot_out.write_text("BOT2", encoding="utf-8")
            (flywheel_dir / "termination").write_text(
                "normal", encoding="utf-8")
            return ContainerResult(exit_code=0, elapsed_s=0.1)

        if config.image == "eval:latest":
            assert (mounts["/input/bot"] / "bot.py").read_text(
                encoding="utf-8") == "BOT1"
            score = mounts["/output/score"] / "scores.json"
            score.parent.mkdir(parents=True, exist_ok=True)
            score.write_text(json.dumps({
                "mean": 42.0,
                "episodes": 4000,
                "errors": 0,
            }), encoding="utf-8")
            (flywheel_dir / "termination").write_text(
                "normal", encoding="utf-8")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        raise AssertionError(config.image)

    with patch("flywheel.execution.run_container",
               side_effect=fake_container):
        result = run_pattern(workspace, pattern, template, project_root)

    reloaded = Workspace.load(workspace.path)
    run = reloaded.runs[result.run_id]
    first_agent = reloaded.executions[run.steps[0].members[0].execution_id]
    second_agent = reloaded.executions[run.steps[1].members[0].execution_id]
    invocation = reloaded.invocations[run.steps[0].members[0].invocation_ids[0]]
    eval_execution = reloaded.executions[invocation.invoked_execution_id]

    assert agent_runs == 2
    assert first_agent.termination_reason == "eval_requested"
    assert second_agent.termination_reason == "normal"
    assert second_agent.input_bindings["score"] == (
        eval_execution.output_bindings["score"])
    assert first_agent.state_snapshot_id is not None
    assert second_agent.state_snapshot_id is not None
    assert reloaded.state_snapshots[
        second_agent.state_snapshot_id].predecessor_snapshot_id == (
            first_agent.state_snapshot_id)
