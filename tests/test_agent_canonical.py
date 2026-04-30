from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from flywheel.container import ContainerResult
from flywheel.execution import run_block
from flywheel.workspace import Workspace
from tests._inline_blocks import from_yaml_with_inline_blocks

AGENT_TEMPLATE_YAML = """\
blocks:
  - name: agent
    image: agent:latest
    network: bridge
    env:
      BASE_ENV: base
      MAX_TURNS: "3"
      MODEL: test-model
    docker_args:
      - -v
      - claude-auth:/home/claude/.claude:rw
    state: managed
    outputs:
      normal: []
"""


def test_claude_battery_uses_canonical_managed_state_path(
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
        seen["network"] = config.network
        seen["mounts"] = dict(mounts)
        seen["docker_args"] = list(config.docker_args)

        flywheel_dir = Path(mounts["/flywheel"])
        flywheel_dir.mkdir(parents=True, exist_ok=True)
        (flywheel_dir / "termination").write_text("normal")

        state_dir = flywheel_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "session.txt").write_text("hello state")

        control_dir = flywheel_dir / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        (control_dir / "agent_exit_state.json").write_text("{}")
        (control_dir / "pending_tool_calls.json").write_text(
            '{"pending": []}')
        return ContainerResult(exit_code=0, elapsed_s=1.25)

    with patch("flywheel.execution.run_container",
               side_effect=fake_run_container):
        result = run_block(
            workspace=workspace,
            block_name="agent",
            template=template,
            project_root=project_root,
            state_lineage_key="agent-session",
        )

    assert result.container_result.exit_code == 0
    assert result.container_result.elapsed_s == 1.25

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
    assert seen["network"] == "bridge"
    assert seen["args"] is None
    env = seen["env"]
    assert env["BASE_ENV"] == "base"
    assert env["MODEL"] == "test-model"
    assert env["MAX_TURNS"] == "3"
    assert "-v" in seen["docker_args"]
    assert "claude-auth:/home/claude/.claude:rw" in seen["docker_args"]
    assert "/flywheel" in seen["mounts"]
    assert "/flywheel/control" not in seen["mounts"]
    assert "/prompt" not in seen["mounts"]
