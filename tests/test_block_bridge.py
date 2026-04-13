"""Tests for the block bridge record mode.

Record mode creates artifacts and execution records without
launching a Docker container -- used for provenance tracking of
actions that already happened (e.g., game steps via REST API).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from flywheel.artifact import ArtifactInstance
from flywheel.block_bridge import (
    RECORD_SENTINEL,
    _process_record_invocation,
)
from flywheel.template import Template
from flywheel.workspace import Workspace

TEMPLATE_YAML = """\
artifacts:
  - name: game_session
    kind: copy
  - name: game_step
    kind: copy
  - name: game_history
    kind: copy

blocks:
  - name: game_step
    image: "__record__"
    inputs:
      - name: game_session
        container_path: /input/game_session
    outputs:
      - name: game_session
        container_path: /output/game_session
      - name: game_step
        container_path: /output/game_step
"""

CONTAINER_BLOCK_YAML = """\
artifacts:
  - name: checkpoint
    kind: copy
  - name: score
    kind: copy

blocks:
  - name: eval_bot
    image: eval:latest
    inputs: [checkpoint]
    outputs: [score]
"""


def _make_workspace(tmp_path: Path, yaml: str = TEMPLATE_YAML) -> tuple[
    Template, Workspace,
]:
    """Create a template and workspace in tmp_path (no git needed)."""
    tmpl_path = tmp_path / "test.yaml"
    tmpl_path.write_text(yaml)
    template = Template.from_yaml(tmpl_path)

    ws_path = tmp_path / "workspace"
    ws_path.mkdir()
    (ws_path / "artifacts").mkdir()

    ws = Workspace(
        name="test",
        path=ws_path,
        template_name="test",
        created_at=datetime.now(UTC),
        artifact_declarations={
            a.name: a.kind for a in template.artifacts
        },
        artifacts={},
    )
    ws.save()
    return template, ws


def _seed_session(ws: Workspace) -> ArtifactInstance:
    """Create an initial game_session artifact in the workspace."""
    aid = ws.generate_artifact_id("game_session")
    artifact_dir = ws.path / "artifacts" / aid
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "game_session.json").write_text(
        json.dumps({"card_id": "c1", "guid": "g1",
                     "level": 0, "action_count": 0}),
    )
    inst = ArtifactInstance(
        id=aid, name="game_session", kind="copy",
        created_at=datetime.now(UTC), copy_path=aid,
    )
    ws.add_artifact(inst)
    ws.save()
    return inst


class TestRecordMode:
    def test_creates_artifacts_and_execution(self, tmp_path: Path):
        template, ws = _make_workspace(tmp_path)
        session = _seed_session(ws)

        resp = _process_record_invocation(
            request_id="test_0001",
            block_name="game_step",
            inputs={"game_session": session.id},
            outputs={
                "game_session": {
                    "card_id": "c1", "guid": "g1",
                    "level": 0, "action_count": 1,
                },
                "game_step": {
                    "step_index": 1,
                    "action": {"action": 6, "x": 10, "y": 20},
                    "pre_state": [[0, 1], [2, 3]],
                    "post_state": [[0, 1], [2, 4]],
                    "score_before": 0,
                    "score_after": 0,
                },
            },
            elapsed_s=0.25,
            template=template,
            workspace=ws,
        )

        assert resp["ok"] is True
        assert "execution_id" in resp
        assert "game_session_artifact_id" in resp
        assert "game_step_artifact_id" in resp

        # Verify artifacts exist in workspace.
        session_aid = resp["game_session_artifact_id"]
        step_aid = resp["game_step_artifact_id"]
        assert session_aid in ws.artifacts
        assert step_aid in ws.artifacts

        # Verify artifact files on disk.
        session_file = (
            ws.path / "artifacts" / session_aid / "game_session.json"
        )
        assert session_file.exists()
        session_data = json.loads(session_file.read_text())
        assert session_data["action_count"] == 1

        step_file = (
            ws.path / "artifacts" / step_aid / "game_step.json"
        )
        assert step_file.exists()
        step_data = json.loads(step_file.read_text())
        assert step_data["step_index"] == 1
        assert step_data["action"]["x"] == 10

        # Verify execution record.
        exec_id = resp["execution_id"]
        assert exec_id in ws.executions
        execution = ws.executions[exec_id]
        assert execution.block_name == "game_step"
        assert execution.status == "succeeded"
        assert execution.image == RECORD_SENTINEL
        assert execution.elapsed_s == 0.25
        assert execution.input_bindings["game_session"] == session.id
        assert execution.output_bindings["game_session"] == session_aid
        assert execution.output_bindings["game_step"] == step_aid

    def test_session_chaining(self, tmp_path: Path):
        """Output session from step 1 is used as input for step 2."""
        template, ws = _make_workspace(tmp_path)
        session = _seed_session(ws)

        # Step 1.
        resp1 = _process_record_invocation(
            request_id="test_0001",
            block_name="game_step",
            inputs={"game_session": session.id},
            outputs={
                "game_session": {"card_id": "c1", "guid": "g1",
                                 "level": 0, "action_count": 1},
                "game_step": {"step_index": 1, "action": {"action": 1},
                              "pre_state": [], "post_state": [],
                              "score_before": 0, "score_after": 0},
            },
            elapsed_s=0.1,
            template=template,
            workspace=ws,
        )
        assert resp1["ok"]
        new_session_id = resp1["game_session_artifact_id"]

        # Step 2 — uses step 1's output session.
        resp2 = _process_record_invocation(
            request_id="test_0002",
            block_name="game_step",
            inputs={"game_session": new_session_id},
            outputs={
                "game_session": {"card_id": "c1", "guid": "g1",
                                 "level": 0, "action_count": 2},
                "game_step": {"step_index": 2, "action": {"action": 2},
                              "pre_state": [], "post_state": [],
                              "score_before": 0, "score_after": 0},
            },
            elapsed_s=0.1,
            template=template,
            workspace=ws,
        )
        assert resp2["ok"]

        # Verify the chain.
        exec2 = ws.executions[resp2["execution_id"]]
        assert exec2.input_bindings["game_session"] == new_session_id

    def test_unknown_block_returns_error(self, tmp_path: Path):
        template, ws = _make_workspace(tmp_path)

        resp = _process_record_invocation(
            request_id="test_0001",
            block_name="nonexistent",
            inputs={},
            outputs={},
            elapsed_s=None,
            template=template,
            workspace=ws,
        )
        assert resp["ok"] is False
        assert resp["error_type"] == "unknown_block"

    def test_non_record_block_rejected(self, tmp_path: Path):
        template, ws = _make_workspace(tmp_path, CONTAINER_BLOCK_YAML)

        resp = _process_record_invocation(
            request_id="test_0001",
            block_name="eval_bot",
            inputs={},
            outputs={},
            elapsed_s=None,
            template=template,
            workspace=ws,
        )
        assert resp["ok"] is False
        assert resp["error_type"] == "not_record_block"

    def test_unknown_input_artifact_returns_error(self, tmp_path: Path):
        template, ws = _make_workspace(tmp_path)

        resp = _process_record_invocation(
            request_id="test_0001",
            block_name="game_step",
            inputs={"game_session": "game_session@nonexistent"},
            outputs={
                "game_session": {"card_id": "c1"},
                "game_step": {"step_index": 1},
            },
            elapsed_s=None,
            template=template,
            workspace=ws,
        )
        assert resp["ok"] is False
        assert resp["error_type"] == "unknown_artifact"

    def test_missing_required_input_no_instances(self, tmp_path: Path):
        template, ws = _make_workspace(tmp_path)

        resp = _process_record_invocation(
            request_id="test_0001",
            block_name="game_step",
            inputs={},  # No session provided, none in workspace.
            outputs={
                "game_session": {"card_id": "c1"},
                "game_step": {"step_index": 1},
            },
            elapsed_s=None,
            template=template,
            workspace=ws,
        )
        assert resp["ok"] is False
        assert resp["error_type"] == "missing_input"

    def test_auto_resolves_latest_session(self, tmp_path: Path):
        """When no input ID is provided, uses the latest instance."""
        template, ws = _make_workspace(tmp_path)
        session = _seed_session(ws)

        resp = _process_record_invocation(
            request_id="test_0001",
            block_name="game_step",
            inputs={},  # Empty — should auto-resolve.
            outputs={
                "game_session": {"card_id": "c1", "action_count": 1},
                "game_step": {"step_index": 1},
            },
            elapsed_s=0.1,
            template=template,
            workspace=ws,
        )
        assert resp["ok"]
        exec_rec = ws.executions[resp["execution_id"]]
        assert exec_rec.input_bindings["game_session"] == session.id

    def test_allowed_blocks_filter(self, tmp_path: Path):
        template, ws = _make_workspace(tmp_path)

        resp = _process_record_invocation(
            request_id="test_0001",
            block_name="game_step",
            inputs={},
            outputs={},
            elapsed_s=None,
            template=template,
            workspace=ws,
            allowed_blocks=["other_block"],
        )
        assert resp["ok"] is False
        assert resp["error_type"] == "block_not_allowed"

    def test_missing_output_data_skipped(self, tmp_path: Path):
        """Output slot with no data in the payload is silently skipped."""
        template, ws = _make_workspace(tmp_path)
        session = _seed_session(ws)

        resp = _process_record_invocation(
            request_id="test_0001",
            block_name="game_step",
            inputs={"game_session": session.id},
            outputs={
                # game_session provided, game_step omitted.
                "game_session": {"card_id": "c1", "action_count": 1},
            },
            elapsed_s=0.1,
            template=template,
            workspace=ws,
        )
        assert resp["ok"]
        assert "game_session_artifact_id" in resp
        # game_step was not provided, so no artifact created.
        assert "game_step_artifact_id" not in resp
        exec_rec = ws.executions[resp["execution_id"]]
        assert "game_step" not in exec_rec.output_bindings

    def test_persists_to_disk(self, tmp_path: Path):
        """Verify workspace.yaml is updated after recording."""
        template, ws = _make_workspace(tmp_path)
        session = _seed_session(ws)

        _process_record_invocation(
            request_id="test_0001",
            block_name="game_step",
            inputs={"game_session": session.id},
            outputs={
                "game_session": {"card_id": "c1", "action_count": 1},
                "game_step": {"step_index": 1},
            },
            elapsed_s=0.1,
            template=template,
            workspace=ws,
        )

        # Reload from disk.
        reloaded = Workspace.load(ws.path)
        # Original session + new session + new step = 3.
        assert len(reloaded.artifacts) == 3
        assert len(reloaded.executions) == 1
