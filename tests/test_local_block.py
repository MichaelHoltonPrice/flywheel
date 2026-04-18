"""Deterministic unit tests for :mod:`flywheel.local_block`.

The recorder replaces the loopback ``BlockChannelClient`` →
``ExecutionChannel`` round-trip used by host-side block runners
in B5/B6.  These tests pin the contract every cyberarc runner
relies on:

- ``begin`` resolves declared inputs against the workspace's
  latest instances; missing required inputs raise a typed error
  *before* any row is written; missing optional inputs are
  silently skipped.
- ``set_output`` buffers payloads; on clean exit the recorder
  registers each as a ``copy`` artifact and writes a single
  ``"succeeded"`` row.  No intermediate ``"running"`` row exists.
- Body exceptions propagate after the recorder writes a single
  ``"failed"`` row with the expected ``error`` string and zero
  output bindings.
- Output write failures roll back partial artifacts and surface
  as a failed row carrying the underlying error message.
- Post-execution checks run after the row is durable; halt
  directives queue on the recorder and drain through
  ``drain_halts``.
- Scratch directories survive the body for read/write use and
  are cleaned up regardless of whether the body raised.

Tests use a real :class:`flywheel.workspace.Workspace` against
``tmp_path`` — no HTTP, no threads, no docker.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from flywheel.artifact import ArtifactInstance, BlockExecution
from flywheel.blocks.registry import BlockRegistry
from flywheel.local_block import (
    LocalBlockError,
    LocalBlockRecorder,
    LocalExecutionContext,
)
from flywheel.post_check import HaltDirective, PostCheckContext
from flywheel.template import Template, parse_block_definition
from flywheel.workspace import Workspace

TEMPLATE_YAML = """\
artifacts:
  - name: predictor
    kind: copy
  - name: game_history
    kind: copy
  - name: prediction
    kind: copy
  - name: game_step
    kind: copy
  - name: scratch
    kind: copy

blocks:
  - predict
  - noop
  - emits_two
"""

PREDICT_BLOCK_YAML = """\
name: predict
runner: lifecycle
runner_justification: "Tool-triggered logical block; no container body."
inputs:
  - name: predictor
    container_path: /input/predictor
  - name: game_history
    container_path: /input/game_history
    optional: true
outputs:
  - name: prediction
    container_path: /output/prediction
"""

NOOP_BLOCK_YAML = """\
name: noop
runner: lifecycle
runner_justification: "Test-only no-op lifecycle block."
inputs: []
outputs: []
"""

EMITS_TWO_BLOCK_YAML = """\
name: emits_two
runner: lifecycle
runner_justification: "Test-only block that emits two output slots."
inputs: []
outputs:
  - name: prediction
    container_path: /output/prediction
  - name: game_step
    container_path: /output/game_step
"""


def _make_workspace(tmp_path: Path) -> tuple[Template, Workspace]:
    """Build a template and empty workspace under ``tmp_path``."""
    registry = BlockRegistry(blocks={
        "predict": parse_block_definition(
            yaml.safe_load(PREDICT_BLOCK_YAML)),
        "noop": parse_block_definition(
            yaml.safe_load(NOOP_BLOCK_YAML)),
        "emits_two": parse_block_definition(
            yaml.safe_load(EMITS_TWO_BLOCK_YAML)),
    })
    tmpl_path = tmp_path / "test.yaml"
    tmpl_path.write_text(TEMPLATE_YAML)
    template = Template.from_yaml(
        tmpl_path, block_registry=registry)

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


def _seed_artifact(
    ws: Workspace, name: str, data: dict,
) -> str:
    """Seed a copy artifact instance and return its ID."""
    aid = ws.generate_artifact_id(name)
    art_dir = ws.path / "artifacts" / aid
    art_dir.mkdir(parents=True)
    (art_dir / f"{name}.json").write_text(json.dumps(data))
    inst = ArtifactInstance(
        id=aid,
        name=name,
        kind="copy",
        created_at=datetime.now(UTC),
        copy_path=aid,
    )
    ws.add_artifact(inst)
    return aid


@pytest.fixture()
def recorder_factory(tmp_path: Path):
    """Yield ``(make_recorder, workspace)`` for tests."""
    template, ws = _make_workspace(tmp_path)

    def _make(post_checks=None) -> LocalBlockRecorder:
        return LocalBlockRecorder(
            workspace=ws,
            template=template,
            post_checks=post_checks,
        )

    return _make, ws


class TestBeginInputResolution:
    """``begin`` must resolve inputs the same way the bridge did."""

    def test_unknown_block_raises_typed_error(
            self, recorder_factory):
        make, _ = recorder_factory
        rec = make()
        with pytest.raises(LocalBlockError) as exc_info, rec.begin(block="does_not_exist"):
            pass
        assert exc_info.value.error_type == "unknown_block"

    def test_missing_required_input_raises_before_row(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        executions_before = len(ws.executions)
        with pytest.raises(LocalBlockError) as exc_info, rec.begin(block="predict"):
            pass
        assert exc_info.value.error_type == "missing_input"
        assert "predictor" in str(exc_info.value)
        assert len(ws.executions) == executions_before, (
            "missing-input rejection must not leave a row")

    def test_missing_optional_input_is_silently_skipped(
            self, recorder_factory):
        make, ws = recorder_factory
        _seed_artifact(ws, "predictor", {"v": 1})
        rec = make()
        with rec.begin(block="predict") as ctx:
            assert "predictor" in ctx.input_bindings
            assert "game_history" not in ctx.input_bindings

    def test_inputs_resolve_to_latest_instance(
            self, recorder_factory):
        make, ws = recorder_factory
        first = _seed_artifact(ws, "predictor", {"v": 1})
        latest = _seed_artifact(ws, "predictor", {"v": 2})
        rec = make()
        with rec.begin(block="predict") as ctx:
            assert ctx.input_bindings["predictor"] == latest
            assert ctx.input_bindings["predictor"] != first

    def test_input_paths_are_absolute_host_paths(
            self, recorder_factory):
        make, ws = recorder_factory
        aid = _seed_artifact(ws, "predictor", {"v": 1})
        rec = make()
        with rec.begin(block="predict") as ctx:
            path = Path(ctx.input_paths["predictor"])
            assert path.is_absolute()
            assert path.exists()
            assert path == ws.path / "artifacts" / aid


class TestSucceededRow:
    """Clean exit writes one ``"succeeded"`` row + the artifacts."""

    def test_succeeded_row_replaces_intermediate(
            self, recorder_factory):
        make, ws = recorder_factory
        _seed_artifact(ws, "predictor", {"v": 1})
        rec = make()
        with rec.begin(
            block="predict",
            params={"action_id": 6, "x": 10, "y": 20},
            caller={"mcp_server": "arc",
                    "tool": "predict_action"},
        ) as ctx:
            ctx.set_output("prediction", {
                "predicted_state": [[0, 1]],
                "rationale": "deterministic",
            })

        rows = list(ws.executions.values())
        assert len(rows) == 1, (
            "no intermediate `running` row should survive")
        row = rows[0]
        assert row.status == "succeeded"
        assert row.block_name == "predict"
        assert row.params == {
            "action_id": 6, "x": 10, "y": 20}
        assert row.caller == {
            "mcp_server": "arc", "tool": "predict_action"}
        assert "prediction" in row.output_bindings

    def test_artifact_file_written_with_payload(
            self, recorder_factory):
        make, ws = recorder_factory
        _seed_artifact(ws, "predictor", {"v": 1})
        rec = make()
        with rec.begin(block="predict") as ctx:
            ctx.set_output("prediction", {
                "predicted_state": [[0, 1], [2, 3]]})

        row = next(iter(ws.executions.values()))
        aid = row.output_bindings["prediction"]
        path = (ws.path / "artifacts" / aid / "prediction.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["predicted_state"] == [[0, 1], [2, 3]]

    def test_output_bindings_surfaced_on_context(
            self, recorder_factory):
        make, ws = recorder_factory
        _seed_artifact(ws, "predictor", {"v": 1})
        rec = make()
        captured: dict[str, str] = {}
        with rec.begin(block="predict") as ctx:
            ctx.set_output("prediction", {"v": 1})
        captured.update(ctx.output_bindings)
        row = next(iter(ws.executions.values()))
        assert captured == row.output_bindings

    def test_unset_output_slot_is_skipped(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="emits_two") as ctx:
            ctx.set_output("prediction", {"only": "this one"})

        row = next(iter(ws.executions.values()))
        assert row.status == "succeeded"
        assert "prediction" in row.output_bindings
        assert "game_step" not in row.output_bindings

    def test_set_output_twice_raises(self, recorder_factory):
        make, _ = recorder_factory
        rec = make()
        with (
            pytest.raises(ValueError, match="already set"),
            rec.begin(block="emits_two") as ctx,
        ):
            ctx.set_output("prediction", {"v": 1})
            ctx.set_output("prediction", {"v": 2})

    def test_parent_execution_id_recorded(self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(
            block="noop", parent_execution_id="exec_parent01"):
            pass
        row = next(iter(ws.executions.values()))
        assert row.parent_execution_id == "exec_parent01"


class TestFailedRow:
    """Body exceptions produce a single ``"failed"`` row."""

    def test_body_exception_propagates(self, recorder_factory):
        make, _ = recorder_factory
        rec = make()
        with pytest.raises(RuntimeError, match="boom"), rec.begin(block="noop"):
            raise RuntimeError("boom")

    def test_failed_row_records_error_string(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with pytest.raises(RuntimeError), rec.begin(block="noop"):
            raise RuntimeError("boom")
        row = next(iter(ws.executions.values()))
        assert row.status == "failed"
        assert row.error == "RuntimeError: boom"
        assert row.output_bindings == {}

    def test_failed_row_does_not_register_outputs(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        artifacts_before = dict(ws.artifacts)
        with pytest.raises(RuntimeError), rec.begin(block="emits_two") as ctx:
            ctx.set_output("prediction", {"v": 1})
            raise RuntimeError("boom")
        assert ws.artifacts == artifacts_before


class TestPostCheck:
    """Post-checks run after the row lands and queue halts."""

    def test_post_check_invoked_with_finalized_context(
            self, recorder_factory):
        make, ws = recorder_factory
        _seed_artifact(ws, "predictor", {"v": 1})
        seen: list[PostCheckContext] = []

        def check(ctx: PostCheckContext) -> None:
            seen.append(ctx)
            return None

        rec = make(post_checks={"predict": check})
        with rec.begin(
            block="predict",
            caller={"mcp_server": "arc",
                    "tool": "predict_action"},
        ) as ctx:
            ctx.set_output("prediction", {"v": 1})

        assert len(seen) == 1
        seen_ctx = seen[0]
        assert seen_ctx.block == "predict"
        assert seen_ctx.status == "succeeded"
        assert seen_ctx.caller == {
            "mcp_server": "arc", "tool": "predict_action"}
        assert seen_ctx.outputs == {"prediction": {"v": 1}}

    def test_halt_directive_queues_and_persists(
            self, recorder_factory):
        make, ws = recorder_factory

        def check(_: PostCheckContext) -> HaltDirective:
            return HaltDirective(
                scope="run", reason="enough already")

        rec = make(post_checks={"noop": check})
        with rec.begin(block="noop"):
            pass

        halts = rec.drain_halts()
        assert len(halts) == 1
        assert halts[0].scope == "run"
        assert halts[0].reason == "enough already"
        assert rec.drain_halts() == [], (
            "drain must clear the queue")

        row = next(iter(ws.executions.values()))
        assert row.halt_directive == {
            "scope": "run", "reason": "enough already"}

    def test_post_check_failure_recorded_on_row(
            self, recorder_factory):
        make, ws = recorder_factory

        def check(_: PostCheckContext) -> HaltDirective:
            raise RuntimeError("check broken")

        rec = make(post_checks={"noop": check})
        with rec.begin(block="noop"):
            pass

        row = next(iter(ws.executions.values()))
        assert row.status == "succeeded"
        assert "RuntimeError: check broken" in (
            row.post_check_error or "")

    def test_post_check_invalid_return_recorded(
            self, recorder_factory):
        make, ws = recorder_factory

        def check(_: PostCheckContext) -> str:  # type: ignore[return-value]
            return "not a HaltDirective"

        rec = make(post_checks={"noop": check})
        with rec.begin(block="noop"):
            pass

        row = next(iter(ws.executions.values()))
        assert row.halt_directive is None
        assert "expected HaltDirective" in (
            row.post_check_error or "")

    def test_no_post_check_leaves_row_unchanged(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="noop"):
            pass
        row = next(iter(ws.executions.values()))
        assert row.halt_directive is None
        assert row.post_check_error is None


class TestScratchAndCleanup:
    """Scratch dirs are usable during the body and removed after."""

    def test_scratch_dir_exists_during_body(
            self, recorder_factory):
        make, _ = recorder_factory
        rec = make()
        seen: list[Path] = []
        with rec.begin(block="noop") as ctx:
            seen.append(Path(ctx.scratch_dir))
            assert Path(ctx.scratch_dir).is_dir()
        assert not seen[0].exists()

    def test_scratch_dir_removed_on_failed_body(
            self, recorder_factory):
        make, _ = recorder_factory
        rec = make()
        scratch_path: list[Path] = []
        with pytest.raises(RuntimeError), rec.begin(block="noop") as ctx:
            scratch_path.append(Path(ctx.scratch_dir))
            raise RuntimeError("boom")
        assert not scratch_path[0].exists()


class TestNoIntermediateRunningRow:
    """Critical contract: no row exists with status=='running'."""

    def test_in_body_workspace_has_no_intermediate_row(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        observed: dict[str, str] = {}
        with rec.begin(block="noop"):
            observed["count_during_body"] = (
                str(len(ws.executions)))
            observed["running_during_body"] = str(sum(
                1 for r in ws.executions.values()
                if r.status == "running"))
        assert observed["count_during_body"] == "0"
        assert observed["running_during_body"] == "0"
        assert len(ws.executions) == 1
        row = next(iter(ws.executions.values()))
        assert row.status == "succeeded"


class TestLocalExecutionContext:
    """Independent unit coverage of the dataclass surface."""

    def test_set_output_buffers_data(self):
        ctx = LocalExecutionContext(
            execution_id="exec_test01",
            block_name="noop",
            input_bindings={},
            input_paths={},
            scratch_dir="/tmp/x",
            params={},
            parent_execution_id=None,
        )
        ctx.set_output("a", {"v": 1})
        ctx.set_output("b", [1, 2, 3])
        assert ctx.outputs == {"a": {"v": 1}, "b": [1, 2, 3]}


class TestRowIsolation:
    """Recorder writes nothing under failure scenarios mid-sequence."""

    def test_two_sequential_runs_record_two_rows(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="noop"):
            pass
        with rec.begin(block="noop"):
            pass
        rows = list(ws.executions.values())
        assert len(rows) == 2
        assert all(r.status == "succeeded" for r in rows)
        assert rows[0].id != rows[1].id

    def test_failed_then_succeeded_runs_recorded_separately(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with pytest.raises(RuntimeError), rec.begin(block="noop"):
            raise RuntimeError("first attempt failed")
        with rec.begin(block="noop"):
            pass
        rows = sorted(
            ws.executions.values(), key=lambda r: r.started_at)
        assert [r.status for r in rows] == ["failed", "succeeded"]


class TestExecutionRowFields:
    """Exhaustive contract checks for the fields downstream consumes."""

    def test_runner_field_propagates(self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="noop", runner="lifecycle"):
            pass
        row: BlockExecution = next(iter(ws.executions.values()))
        assert row.runner == "lifecycle"

    def test_image_pulled_from_block_definition(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="noop"):
            pass
        row = next(iter(ws.executions.values()))
        # noop has no container image; BlockDefinition surfaces
        # an empty string for that, not None — the recorder
        # passes it through verbatim.
        assert row.image == ""

    def test_elapsed_s_is_recorded_as_float(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="noop"):
            pass
        row = next(iter(ws.executions.values()))
        assert isinstance(row.elapsed_s, float)
        assert row.elapsed_s >= 0.0
