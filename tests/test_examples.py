from __future__ import annotations

from pathlib import Path

from flywheel.config import load_project_config
from flywheel.template import Template


def test_hello_agent_example_templates_parse():
    project_root = Path(__file__).resolve().parents[1] / "examples" / (
        "hello-agent"
    )

    config = load_project_config(project_root)
    template = Template.from_yaml(
        config.workspace_templates_dir / "hello-agent.yaml",
        block_registry=config.load_block_registry(),
    )

    assert template.name == "hello-agent"
    assert template.artifacts == []
    assert [block.name for block in template.blocks] == ["HelloAgent"]
    block = template.blocks[0]
    assert block.image == "flywheel-hello-agent:latest"
    assert block.state == "managed"
    assert block.inputs == []
    assert "claude-auth:/home/claude/.claude:rw" in block.docker_args
    assert block.outputs_for("normal") == []


def test_hello_codex_example_templates_parse():
    project_root = Path(__file__).resolve().parents[1] / "examples" / (
        "hello-codex"
    )

    config = load_project_config(project_root)
    template = Template.from_yaml(
        config.workspace_templates_dir / "hello-codex.yaml",
        block_registry=config.load_block_registry(),
    )

    assert template.name == "hello-codex"
    assert template.artifacts == []
    assert [block.name for block in template.blocks] == ["HelloCodex"]
    block = template.blocks[0]
    assert block.image == "flywheel-hello-codex:latest"
    assert block.state == "managed"
    assert block.inputs == []
    assert "codex-auth:/home/codex/.codex:rw" in block.docker_args
    assert block.outputs_for("normal") == []


def test_claude_battery_uses_agent_status_for_termination():
    entrypoint = (
        Path(__file__).resolve().parents[1]
        / "batteries" / "claude" / "entrypoint.sh"
    )

    text = entrypoint.read_text()
    assert "/tmp/flywheel-claude-runner.jsonl" in text
    assert 'candidate.get("type") == "agent_state"' in text
    assert "complete" in text
    assert "/flywheel/termination" in text
