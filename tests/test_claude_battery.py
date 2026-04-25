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


def test_claude_battery_entrypoint_creates_framework_subdirs():
    entrypoint = ROOT / "batteries" / "claude" / "entrypoint.sh"
    text = entrypoint.read_text(encoding="utf-8")

    assert "mkdir -p /flywheel/control /flywheel/mcp_servers" in text
    assert "chown -R claude:claude /flywheel/control" in text
