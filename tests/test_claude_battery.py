from __future__ import annotations

from pathlib import Path

from flywheel.config import load_project_config
from flywheel.template import Template

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
    assert "chown claude:claude /flywheel/termination_request" in text
    assert "chown root:root /flywheel/telemetry" in text
    assert "chmod 700 /flywheel/telemetry" in text


def test_claude_battery_writes_usage_telemetry():
    runner = ROOT / "batteries" / "claude" / "agent_runner.py"
    entrypoint = ROOT / "batteries" / "claude" / "entrypoint.sh"
    runner_text = runner.read_text(encoding="utf-8")
    entrypoint_text = entrypoint.read_text(encoding="utf-8")

    assert "claude_result.json" not in runner_text
    assert "/tmp/flywheel-claude-runner.jsonl" in entrypoint_text
    assert "/flywheel/telemetry/claude_usage.json" in entrypoint_text
    assert '"kind": "claude_usage"' in entrypoint_text
    assert '"source": "flywheel-claude"' in entrypoint_text
