from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from flywheel.agent import run_agent_block
from flywheel.container import ContainerResult
from flywheel.workspace import Workspace
from tests._inline_blocks import from_yaml_with_inline_blocks


AGENT_TEMPLATE_YAML = """\
artifacts: []

blocks:
  - name: agent
    image: agent:latest
    env:
      BASE_ENV: base
    state: managed
    outputs:
      normal: []
"""


def test_run_agent_block_uses_canonical_managed_state_path(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    foundry_dir = project_root / "foundry"
    foundry_dir.mkdir()
    template_path = project_root / "template.yaml"
    template_path.write_text(AGENT_TEMPLATE_YAML)
    template = from_yaml_with_inline_blocks(template_path)
    workspace = Workspace.create("ws", template, foundry_dir)

    seen: dict[str, object] = {}

    def fake_run_container(config, args=None):
        mounts = {container: host for host, container, mode in config.mounts}
        seen["image"] = config.image
        seen["args"] = args
        seen["env"] = dict(config.env)
        seen["mounts"] = dict(mounts)

        flywheel_dir = Path(mounts["/flywheel"])
        flywheel_dir.mkdir(parents=True, exist_ok=True)
        (flywheel_dir / "termination").write_text("normal")

        state_dir = flywheel_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "session.txt").write_text("hello state")

        control_dir = flywheel_dir / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        (control_dir / "agent_exit_state.json").write_text(json.dumps({
            "status": "complete",
            "reason": "done",
        }))
        (control_dir / "pending_tool_calls.json").write_text(json.dumps({
            "pending": [],
        }))
        assert (Path(mounts["/prompt"]) / "prompt.md").read_text() == (
            "hello agent"
        )
        return ContainerResult(exit_code=0, elapsed_s=1.25)

    with patch("flywheel.agent.run_container", side_effect=fake_run_container):
        result = run_agent_block(
            workspace=workspace,
            template=template,
            project_root=project_root,
            prompt="hello agent",
            block_name="agent",
            model="test-model",
            max_turns=3,
            state_lineage_key="agent-session",
        )

    assert result.exit_code == 0
    assert result.elapsed_s == 1.25
    assert result.exit_reason == "completed"
    assert result.exit_state == {"status": "complete", "reason": "done"}
    assert result.pending_tool_calls == []

    execution = workspace.executions[result.execution_id]
    assert execution.block_name == "agent"
    assert execution.status == "succeeded"
    assert execution.state_mode == "managed"
    assert execution.state_snapshot_id is not None

    snapshot = workspace.latest_state_snapshot("agent-session")
    assert snapshot is not None
    assert snapshot.produced_by == result.execution_id
    assert (
        workspace.state_snapshot_path(snapshot.id) / "session.txt"
    ).read_text() == "hello state"

    assert seen["image"] == "agent:latest"
    assert seen["args"] is None
    env = seen["env"]
    assert env["BASE_ENV"] == "base"
    assert env["MODEL"] == "test-model"
    assert env["MAX_TURNS"] == "3"
    assert "/flywheel" in seen["mounts"]
    assert "/flywheel/control" not in seen["mounts"]
    assert "/prompt" in seen["mounts"]
