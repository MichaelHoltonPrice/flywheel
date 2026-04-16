"""Tests for RecordExecutor.

Ported from the record-mode tests in the former test_block_bridge.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywheel.executor import RECORD_SENTINEL, RecordExecutor
from flywheel.template import Template
from flywheel.workspace import Workspace

TEMPLATE_REQUIRED_INPUT_YAML = """\
artifacts:
  - name: game_session
    kind: copy
  - name: game_step
    kind: copy

blocks:
  - name: game_step
    image: "__record__"
    inputs:
      - name: game_session
        container_path: /input/game_session
    outputs:
      - name: game_step
        container_path: /output/game_step
"""

TEMPLATE_YAML = """\
artifacts:
  - name: game_session
    kind: copy
  - name: game_step
    kind: copy

blocks:
  - name: game_step
    image: "__record__"
    inputs:
      - name: game_session
        container_path: /input/game_session
        optional: true
    outputs:
      - name: game_session
        container_path: /output/game_session
      - name: game_step
        container_path: /output/game_step

  - name: container_block
    image: "test:latest"
    inputs: [game_step]
    outputs: [game_step]
"""


def _setup(tmp_path: Path) -> tuple[Workspace, Template]:
    """Create a workspace with the test template."""
    template_file = tmp_path / "test.yaml"
    template_file.write_text(TEMPLATE_YAML)
    template = Template.from_yaml(template_file)

    foundry = tmp_path / "foundry"
    foundry.mkdir()
    ws = Workspace.create("test_ws", template, foundry)
    return ws, template


class TestRecordExecutorBasics:
    def test_creates_output_artifacts(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        handle = executor.launch(
            "game_step", ws, {},
            outputs_data={
                "game_session": {"card_id": "abc"},
                "game_step": {"step_index": 0, "action": "click"},
            },
        )
        result = handle.wait()

        assert result.exit_code == 0
        assert result.status == "succeeded"
        assert "game_session" in result.output_bindings
        assert "game_step" in result.output_bindings

    def test_records_execution(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        handle = executor.launch(
            "game_step", ws, {},
            outputs_data={"game_step": {"step": 1}},
        )
        result = handle.wait()

        assert result.execution_id in ws.executions
        ex = ws.executions[result.execution_id]
        assert ex.block_name == "game_step"
        assert ex.status == "succeeded"
        assert ex.image == RECORD_SENTINEL

    def test_writes_json_file(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        data = {"step_index": 5, "action": "reset"}
        handle = executor.launch(
            "game_step", ws, {},
            outputs_data={"game_step": data},
        )
        result = handle.wait()

        aid = result.output_bindings["game_step"]
        art_dir = ws.path / "artifacts" / aid
        json_file = art_dir / "game_step.json"
        assert json_file.exists()
        assert json.loads(json_file.read_text()) == data

    def test_elapsed_s_recorded(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        handle = executor.launch(
            "game_step", ws, {},
            outputs_data={"game_step": {}},
            elapsed_s=1.5,
        )
        result = handle.wait()

        ex = ws.executions[result.execution_id]
        assert ex.elapsed_s == 1.5
        assert result.elapsed_s == 1.5

    def test_saves_workspace(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        executor.launch(
            "game_step", ws, {},
            outputs_data={"game_step": {}},
        ).wait()

        # Reload and verify persistence.
        loaded = Workspace.load(ws.path)
        assert len(loaded.executions) == 1


class TestRecordExecutorInputs:
    def test_explicit_input_binding(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        # Create an initial session artifact.
        handle1 = executor.launch(
            "game_step", ws, {},
            outputs_data={
                "game_session": {"card_id": "first"},
                "game_step": {"step": 0},
            },
        )
        r1 = handle1.wait()
        session_id = r1.output_bindings["game_session"]

        # Use explicit input binding.
        handle2 = executor.launch(
            "game_step", ws,
            {"game_session": session_id},
            outputs_data={"game_step": {"step": 1}},
        )
        r2 = handle2.wait()

        ex = ws.executions[r2.execution_id]
        assert ex.input_bindings["game_session"] == session_id

    def test_auto_resolves_latest_required_input(
        self, tmp_path: Path,
    ):
        """Required inputs auto-resolve to latest instance."""
        # Use the template with required input.
        template_file = tmp_path / "required.yaml"
        template_file.write_text(TEMPLATE_REQUIRED_INPUT_YAML)
        template = Template.from_yaml(template_file)
        foundry = tmp_path / "foundry_auto"
        foundry.mkdir()
        ws = Workspace.create("auto_ws", template, foundry)

        executor = RecordExecutor(template)

        # Manually create a session artifact.
        session_file = tmp_path / "session.json"
        session_file.write_text('{"card_id":"auto"}')
        ws.register_artifact(
            "game_session", session_file, source="test")

        # Launch without explicit binding — should auto-resolve.
        handle = executor.launch(
            "game_step", ws, {},
            outputs_data={"game_step": {"step": 1}},
        )
        result = handle.wait()

        ex = ws.executions[result.execution_id]
        assert "game_session" in ex.input_bindings


class TestRecordExecutorErrors:
    def test_unknown_block_raises(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        with pytest.raises(ValueError, match="not found"):
            executor.launch("nonexistent", ws, {})

    def test_non_record_block_raises(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        with pytest.raises(ValueError, match="not a record block"):
            executor.launch("container_block", ws, {})

    def test_block_not_allowed_raises(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        with pytest.raises(ValueError, match="not in allowed"):
            executor.launch(
                "game_step", ws, {},
                allowed_blocks=["other_block"],
            )

    def test_unknown_input_artifact_raises(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        with pytest.raises(ValueError, match="not found"):
            executor.launch(
                "game_step", ws,
                {"game_session": "nonexistent@abc"},
            )

    def test_slot_mismatch_raises(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        # Create a game_step artifact (wrong type for session slot).
        executor.launch(
            "game_step", ws, {},
            outputs_data={"game_step": {"step": 0}},
        ).wait()
        step_id = ws.instances_for("game_step")[-1].id

        with pytest.raises(ValueError, match="slot expects"):
            executor.launch(
                "game_step", ws,
                {"game_session": step_id},
            )

    def test_missing_required_input_raises(self, tmp_path: Path):
        """No session exists and none provided — should raise."""
        # Use the template with required (non-optional) input.
        template_file = tmp_path / "required.yaml"
        template_file.write_text(TEMPLATE_REQUIRED_INPUT_YAML)
        template = Template.from_yaml(template_file)
        foundry = tmp_path / "foundry_req"
        foundry.mkdir()
        ws = Workspace.create("req_ws", template, foundry)

        executor = RecordExecutor(template)

        # game_session is required but no instances exist.
        with pytest.raises(ValueError, match="not provided"):
            executor.launch(
                "game_step", ws, {},
                outputs_data={"game_step": {}},
            )

    def test_skips_missing_output_data(self, tmp_path: Path):
        ws, template = _setup(tmp_path)
        executor = RecordExecutor(template)

        # Only provide game_step, not game_session.
        handle = executor.launch(
            "game_step", ws, {},
            outputs_data={"game_step": {"step": 0}},
        )
        result = handle.wait()

        # game_session not in output_bindings.
        assert "game_step" in result.output_bindings
        assert "game_session" not in result.output_bindings
