"""Tests for derived (rollup) input slots.

Phase 4 of the block-execution refactor introduces ``derive_from``
on ``InputSlot``: at every ``/execution/begin`` the channel rebuilds
the rollup from its source artifact's instances, so blocks like
``predict`` see a freshly assembled ``game_history`` per call.

These tests pin down:

- YAML parsing rules (``derive_from`` + ``derive_kind`` required
  together, unsupported kinds rejected).
- :func:`_resolve_inputs_for_begin` rebuilds derived slots and
  binds the new instance.
- Each begin produces a NEW rollup instance even when the source
  hasn't changed (always-rebuild semantics).
- Source-empty case treats the slot as absent rather than raising.
- End-to-end: a lifecycle begin pins
  ``input_bindings[<rollup>]`` to the freshly minted instance.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flywheel.artifact import ArtifactInstance
from flywheel.execution_channel import (
    ExecutionChannel,
    _resolve_inputs_for_begin,
)
from flywheel.blocks.registry import BlockRegistry
from flywheel.template import (
    InputSlot,
    Template,
    _parse_input_slots,
    parse_block_definition,
)
from flywheel.workspace import Workspace


TEMPLATE_YAML = """\
artifacts:
  - name: game_step
    kind: copy
  - name: game_history
    kind: copy
  - name: prediction
    kind: copy

blocks:
  - predict
"""

PREDICT_BLOCK_YAML = """\
name: predict
runner: lifecycle
runner_justification: "Tool-triggered logical block."
inputs:
  - name: game_history
    container_path: /input/game_history
    derive_from: game_step
    derive_kind: jsonl_concat
outputs:
  - name: prediction
    container_path: /output/prediction
"""


def _make_workspace(tmp_path: Path) -> tuple[Template, Workspace]:
    import yaml as _yaml

    registry = BlockRegistry(blocks={
        "predict": parse_block_definition(
            _yaml.safe_load(PREDICT_BLOCK_YAML)),
    })
    tmpl_path = tmp_path / "test.yaml"
    tmpl_path.write_text(TEMPLATE_YAML)
    template = Template.from_yaml(tmpl_path, block_registry=registry)
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


def _seed_step(ws: Workspace, payload: dict) -> str:
    aid = ws.generate_artifact_id("game_step")
    art_dir = ws.path / "artifacts" / aid
    art_dir.mkdir(parents=True)
    (art_dir / "game_step.json").write_text(json.dumps(payload))
    ws.add_artifact(ArtifactInstance(
        id=aid, name="game_step", kind="copy",
        created_at=datetime.now(UTC), copy_path=aid,
    ))
    return aid


class TestInputSlotParsing:
    def test_derive_from_alone_rejected(self):
        with pytest.raises(ValueError, match="derive_kind"):
            _parse_input_slots([
                {"name": "game_history", "derive_from": "game_step"},
            ])

    def test_derive_kind_alone_rejected(self):
        with pytest.raises(ValueError, match="derive_from"):
            _parse_input_slots([
                {"name": "game_history",
                 "derive_kind": "jsonl_concat"},
            ])

    def test_unsupported_kind_rejected(self):
        with pytest.raises(ValueError, match="unsupported derive_kind"):
            _parse_input_slots([
                {"name": "game_history",
                 "derive_from": "game_step",
                 "derive_kind": "no_such_thing"},
            ])

    def test_jsonl_concat_accepted(self):
        slots = _parse_input_slots([
            {"name": "game_history",
             "derive_from": "game_step",
             "derive_kind": "jsonl_concat",
             "container_path": "/input/game_history"},
        ])
        assert len(slots) == 1
        assert slots[0].derive_from == "game_step"
        assert slots[0].derive_kind == "jsonl_concat"

    def test_string_form_has_no_derive_fields(self):
        slots = _parse_input_slots(["predictor"])
        assert slots[0].derive_from is None
        assert slots[0].derive_kind is None


class TestResolverRebuildsDerivedInputs:
    def test_resolver_rebuilds_and_binds_fresh(
            self, tmp_path: Path):
        template, ws = _make_workspace(tmp_path)
        block_def = template.blocks[0]
        _seed_step(ws, {"step_index": 0, "action": 1})
        _seed_step(ws, {"step_index": 1, "action": 2})

        bindings, paths, _hashes = _resolve_inputs_for_begin(
            block_def, ws)

        # The rollup exists in the bindings.
        assert "game_history" in bindings
        history_id = bindings["game_history"]

        # And it's a new artifact instance, registered in the workspace.
        assert history_id in ws.artifacts
        assert ws.artifacts[history_id].name == "game_history"

        # The rollup file contains both source steps as JSONL.
        history_path = Path(paths["game_history"]) / "history.jsonl"
        lines = history_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["step_index"] == 0
        assert json.loads(lines[1])["step_index"] == 1

    def test_each_resolve_creates_a_new_rollup_instance(
            self, tmp_path: Path):
        template, ws = _make_workspace(tmp_path)
        block_def = template.blocks[0]
        _seed_step(ws, {"step_index": 0})

        b1, _, _ = _resolve_inputs_for_begin(block_def, ws)
        b2, _, _ = _resolve_inputs_for_begin(block_def, ws)

        # Always-rebuild: same source, different rollup ids.  The
        # ledger row pins exactly which rollup version this begin
        # saw.
        assert b1["game_history"] != b2["game_history"]
        all_rollups = ws.instances_for("game_history")
        assert len(all_rollups) == 2

    def test_picks_up_new_source_between_begins(
            self, tmp_path: Path):
        template, ws = _make_workspace(tmp_path)
        block_def = template.blocks[0]
        _seed_step(ws, {"step_index": 0})

        b1, p1, _ = _resolve_inputs_for_begin(block_def, ws)
        h1 = (Path(p1["game_history"]) / "history.jsonl").read_text()
        assert h1.count("\n") == 1

        _seed_step(ws, {"step_index": 1})
        b2, p2, _ = _resolve_inputs_for_begin(block_def, ws)
        h2 = (Path(p2["game_history"]) / "history.jsonl").read_text()
        assert h2.count("\n") == 2
        assert b2["game_history"] != b1["game_history"]

    def test_empty_source_skips_slot(self, tmp_path: Path):
        template, ws = _make_workspace(tmp_path)
        block_def = template.blocks[0]
        # No game_step seeded; rollup of nothing → slot absent.
        bindings, paths, _ = _resolve_inputs_for_begin(
            block_def, ws)
        assert "game_history" not in bindings
        assert "game_history" not in paths
        # No rollup instance was registered either.
        assert ws.instances_for("game_history") == []


class TestLifecycleBeginRebuildsDerivedInputs:
    """End-to-end: begin response shows the freshly minted rollup."""

    def test_begin_pins_freshly_rebuilt_rollup(
            self, tmp_path: Path):
        template, ws = _make_workspace(tmp_path)
        _seed_step(ws, {"step_index": 0})
        _seed_step(ws, {"step_index": 1})
        ws.save()

        chan = ExecutionChannel(template=template, workspace=ws)
        port = chan.start()
        try:
            time.sleep(0.05)
            url = f"http://127.0.0.1:{port}/execution/begin"
            req = urllib.request.Request(
                url,
                data=json.dumps({"block": "predict"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as r:
                resp = json.loads(r.read())
        finally:
            chan.stop(timeout=5.0)

        assert resp["ok"] is True
        assert "game_history" in resp["input_bindings"]
        rollup_id = resp["input_bindings"]["game_history"]

        ws2 = Workspace.load(ws.path)
        assert rollup_id in ws2.artifacts
        assert ws2.artifacts[rollup_id].name == "game_history"

        history_path = (
            Path(resp["input_paths"]["game_history"])
            / "history.jsonl")
        assert history_path.exists()
        lines = history_path.read_text().strip().splitlines()
        assert len(lines) == 2


class TestInputSlotConstructionDefensive:
    """Constructing InputSlot in code (bypassing the parser) still
    surfaces unsupported derive_kind at resolve time."""

    def test_unknown_kind_raises_at_resolve(self, tmp_path: Path):
        from flywheel.execution_channel import (
            _materialize_derived_input)
        _, ws = _make_workspace(tmp_path)
        _seed_step(ws, {"x": 1})
        bad_slot = InputSlot(
            name="game_history",
            container_path="/input/game_history",
            derive_from="game_step",
            derive_kind="not_a_real_kind",
        )
        with pytest.raises(ValueError, match="not_a_real_kind"):
            _materialize_derived_input(bad_slot, ws)
