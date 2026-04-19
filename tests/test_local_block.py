"""Deterministic unit tests for :mod:`flywheel.local_block`.

The recorder is the host-side write surface for nested-block
executions invoked during a full-stop handoff.  These tests pin
the contract every cyberarc runner relies on:

- ``begin`` resolves declared inputs against the workspace's
  latest instances; missing required inputs raise a typed error
  *before* any execution is opened; missing optional inputs are
  silently skipped.
- The body writes files into ``ctx.output_dir(name)``; on clean
  exit the recorder registers a ``copy`` artifact from each
  per-output directory that has contents (and appends to the
  workspace's canonical ``incremental`` instance for incremental
  outputs), then writes a single ``"succeeded"`` execution record.  No
  intermediate ``"running"`` execution record exists.
- Body exceptions propagate after the recorder writes a single
  ``"failed"`` execution record with the expected ``error`` string and zero
  output bindings.
- Output write failures roll back partial copy artifacts and
  surface as a failed execution carrying the underlying error message.
- Post-execution checks run after the execution record is durable; halt
  directives queue on the recorder and drain through
  ``drain_halts``.
- Scratch directories survive the body for read/write use and
  are cleaned up regardless of whether the body raised.

Tests use a real :class:`flywheel.workspace.Workspace` against
``tmp_path`` — no HTTP, no threads, no docker.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    kind: incremental
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
  - emits_incremental
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

EMITS_INCREMENTAL_BLOCK_YAML = """\
name: emits_incremental
runner: lifecycle
runner_justification: "Test-only block that appends to an incremental output."
inputs: []
outputs:
  - name: game_history
    container_path: /output/game_history
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
        "emits_incremental": parse_block_definition(
            yaml.safe_load(EMITS_INCREMENTAL_BLOCK_YAML)),
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


def _emit_copy_output(
    ctx: LocalExecutionContext, name: str, data: dict[str, Any],
) -> None:
    """Write a JSON file into the per-output dir for *name*.

    Helper that mirrors what cyberarc runners do: the body's
    contribution to a copy output is one file per execution.
    """
    out_dir = ctx.output_dir(name)
    (out_dir / f"{name}.json").write_text(
        json.dumps(data), encoding="utf-8")


def _emit_incremental_entries(
    ctx: LocalExecutionContext,
    name: str,
    entries: list[Any],
) -> None:
    """Append entries to the per-output ``entries.jsonl`` file."""
    out_dir = ctx.output_dir(name)
    entries_file = out_dir / "entries.jsonl"
    with entries_file.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


@pytest.fixture()
def recorder_factory(tmp_path: Path, monkeypatch):
    """Yield ``(make_recorder, workspace)`` for tests."""
    template, ws = _make_workspace(tmp_path)
    monkeypatch.setenv("FLYWHEEL_RETAIN_FAILED_OUTPUT_DIRS", "0")

    def _make(post_checks=None) -> LocalBlockRecorder:
        return LocalBlockRecorder(
            workspace=ws,
            template=template,
            post_checks=post_checks,
        )

    return _make, ws


class TestBeginInputResolution:
    """``begin`` resolves declared inputs to latest instances."""

    def test_unknown_block_raises_typed_error(
            self, recorder_factory):
        make, _ = recorder_factory
        rec = make()
        with pytest.raises(LocalBlockError) as exc_info, rec.begin(block="does_not_exist"):
            pass
        assert exc_info.value.error_type == "unknown_block"

    def test_missing_required_input_raises_before_execution(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        executions_before = len(ws.executions)
        with pytest.raises(LocalBlockError) as exc_info, rec.begin(block="predict"):
            pass
        assert exc_info.value.error_type == "missing_input"
        assert "predictor" in str(exc_info.value)
        assert len(ws.executions) == executions_before, (
            "missing-input rejection must not leave an execution")

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
            # Paths point at per-mount staging tempdirs, not
            # the canonical artifact directory.
            assert path != ws.path / "artifacts" / aid

    def test_input_paths_are_staging_copies_not_canonical(
            self, recorder_factory):
        make, ws = recorder_factory
        aid = _seed_artifact(ws, "predictor", {"v": 1})
        canonical = ws.path / "artifacts" / aid
        rec = make()
        observed: dict[str, Path] = {}
        with rec.begin(block="predict") as ctx:
            staged = Path(ctx.input_paths["predictor"])
            observed["staged"] = staged
            # A modification of the staging copy must not
            # mutate canonical state.
            (staged / "tampered.txt").write_text("scratch")
            assert not (canonical / "tampered.txt").exists()
            # Each input file is present in the staging copy.
            assert (staged / "predictor.json").is_file()
        # Tempdir is cleaned up after the block exits.
        assert not observed["staged"].exists()
        assert canonical.exists()
        assert not (canonical / "tampered.txt").exists()

    def test_input_staging_runs_per_begin(
            self, recorder_factory):
        make, ws = recorder_factory
        _seed_artifact(ws, "predictor", {"v": 1})
        rec = make()
        with rec.begin(block="predict") as ctx:
            first = Path(ctx.input_paths["predictor"])
        with rec.begin(block="predict") as ctx:
            second = Path(ctx.input_paths["predictor"])
        assert first != second


class TestSucceededExecution:
    """Clean exit writes one ``"succeeded"`` execution + the artifacts."""

    def test_succeeded_execution_replaces_intermediate(
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
            _emit_copy_output(ctx, "prediction", {
                "predicted_state": [[0, 1]],
                "rationale": "deterministic",
            })

        executions = list(ws.executions.values())
        assert len(executions) == 1, (
            "no intermediate `running` execution should survive")
        execution = executions[0]
        assert execution.status == "succeeded"
        assert execution.block_name == "predict"
        assert execution.params == {
            "action_id": 6, "x": 10, "y": 20}
        assert execution.caller == {
            "mcp_server": "arc", "tool": "predict_action"}
        assert "prediction" in execution.output_bindings

    def test_artifact_file_written_with_payload(
            self, recorder_factory):
        make, ws = recorder_factory
        _seed_artifact(ws, "predictor", {"v": 1})
        rec = make()
        with rec.begin(block="predict") as ctx:
            _emit_copy_output(ctx, "prediction", {
                "predicted_state": [[0, 1], [2, 3]]})

        execution = next(iter(ws.executions.values()))
        aid = execution.output_bindings["prediction"]
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
            _emit_copy_output(ctx, "prediction", {"v": 1})
        captured.update(ctx.output_bindings)
        execution = next(iter(ws.executions.values()))
        assert captured == execution.output_bindings

    def test_unset_output_slot_is_skipped(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="emits_two") as ctx:
            _emit_copy_output(ctx, "prediction", {"only": "this one"})

        execution = next(iter(ws.executions.values()))
        assert execution.status == "succeeded"
        assert "prediction" in execution.output_bindings
        assert "game_step" not in execution.output_bindings

    def test_output_dir_for_unknown_slot_raises(
            self, recorder_factory):
        make, _ = recorder_factory
        rec = make()
        with rec.begin(block="emits_two") as ctx, pytest.raises(KeyError):
            ctx.output_dir("not_a_slot")

    def test_parent_execution_id_recorded(self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(
            block="noop", parent_execution_id="exec_parent01"):
            pass
        execution = next(iter(ws.executions.values()))
        assert execution.parent_execution_id == "exec_parent01"

    def test_copy_output_directory_with_multiple_files(
            self, recorder_factory):
        make, ws = recorder_factory
        _seed_artifact(ws, "predictor", {"v": 1})
        rec = make()
        with rec.begin(block="predict") as ctx:
            out = ctx.output_dir("prediction")
            (out / "first.json").write_text('{"a": 1}')
            (out / "second.txt").write_text("hello")

        execution = next(iter(ws.executions.values()))
        aid = execution.output_bindings["prediction"]
        out_dir = ws.path / "artifacts" / aid
        assert (out_dir / "first.json").exists()
        assert (out_dir / "second.txt").read_text() == "hello"


class TestIncrementalOutputs:
    """Incremental outputs append to a workspace-canonical instance."""

    def test_first_append_creates_canonical_instance(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="emits_incremental") as ctx:
            _emit_incremental_entries(
                ctx, "game_history",
                [{"step": 0, "action": "noop"}],
            )

        latest = ws.latest_incremental_instance("game_history")
        assert latest is not None
        execution = next(iter(ws.executions.values()))
        assert execution.output_bindings["game_history"] == latest.id
        assert ws.read_incremental_entries(latest.id) == [
            {"step": 0, "action": "noop"},
        ]

    def test_subsequent_appends_reuse_canonical_instance(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()

        with rec.begin(block="emits_incremental") as ctx:
            _emit_incremental_entries(
                ctx, "game_history", [{"step": 0}])
        with rec.begin(block="emits_incremental") as ctx:
            _emit_incremental_entries(
                ctx, "game_history", [{"step": 1}, {"step": 2}])

        latest = ws.latest_incremental_instance("game_history")
        executions = sorted(
            ws.executions.values(), key=lambda r: r.started_at)
        assert all(
            r.output_bindings["game_history"] == latest.id
            for r in executions
        )
        assert ws.read_incremental_entries(latest.id) == [
            {"step": 0}, {"step": 1}, {"step": 2},
        ]
        # Only one canonical instance should ever exist for v1.
        assert len([
            a for a in ws.artifacts.values()
            if a.name == "game_history" and a.kind == "incremental"
        ]) == 1

    def test_empty_entries_file_skips_binding(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="emits_incremental") as ctx:
            (ctx.output_dir("game_history")
             / "entries.jsonl").write_text("")

        execution = next(iter(ws.executions.values()))
        assert execution.status == "succeeded"
        assert "game_history" not in execution.output_bindings
        assert ws.latest_incremental_instance(
            "game_history") is None

    def test_no_entries_file_skips_binding(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="emits_incremental"):
            pass

        execution = next(iter(ws.executions.values()))
        assert "game_history" not in execution.output_bindings


class TestFailedExecution:
    """Body exceptions produce a single ``"failed"`` execution record."""

    def test_body_exception_propagates(self, recorder_factory):
        make, _ = recorder_factory
        rec = make()
        with pytest.raises(RuntimeError, match="boom"), rec.begin(block="noop"):
            raise RuntimeError("boom")

    def test_failed_execution_records_error_string(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with pytest.raises(RuntimeError), rec.begin(block="noop"):
            raise RuntimeError("boom")
        execution = next(iter(ws.executions.values()))
        assert execution.status == "failed"
        assert execution.error == "RuntimeError: boom"
        assert execution.output_bindings == {}

    def test_failed_execution_does_not_register_outputs(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        artifacts_before = dict(ws.artifacts)
        with pytest.raises(RuntimeError), rec.begin(block="emits_two") as ctx:
            _emit_copy_output(ctx, "prediction", {"v": 1})
            raise RuntimeError("boom")
        assert ws.artifacts == artifacts_before


class TestPostCheck:
    """Post-checks run after the execution lands and queue halts."""

    def test_post_check_invoked_with_finalized_context(
            self, recorder_factory):
        make, ws = recorder_factory
        _seed_artifact(ws, "predictor", {"v": 1})
        seen: list[PostCheckContext] = []
        # Captured during the check, before output-dir cleanup
        # runs in the recorder's finally block.
        observed_payload: dict = {}

        def check(ctx: PostCheckContext) -> None:
            seen.append(ctx)
            out_path = ctx.outputs["prediction"]
            payload_file = out_path / "prediction.json"
            observed_payload["exists"] = payload_file.exists()
            observed_payload["text"] = payload_file.read_text()
            return None

        rec = make(post_checks={"predict": check})
        with rec.begin(
            block="predict",
            caller={"mcp_server": "arc",
                    "tool": "predict_action"},
        ) as ctx:
            _emit_copy_output(ctx, "prediction", {"v": 1})

        assert len(seen) == 1
        seen_ctx = seen[0]
        assert seen_ctx.block == "predict"
        assert seen_ctx.status == "succeeded"
        assert seen_ctx.caller == {
            "mcp_server": "arc", "tool": "predict_action"}
        assert "prediction" in seen_ctx.outputs
        assert isinstance(seen_ctx.outputs["prediction"], Path)
        assert observed_payload["exists"] is True
        assert json.loads(observed_payload["text"]) == {"v": 1}

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

        execution = next(iter(ws.executions.values()))
        assert execution.halt_directive == {
            "scope": "run", "reason": "enough already"}

    def test_post_check_failure_recorded_on_execution(
            self, recorder_factory):
        make, ws = recorder_factory

        def check(_: PostCheckContext) -> HaltDirective:
            raise RuntimeError("check broken")

        rec = make(post_checks={"noop": check})
        with rec.begin(block="noop"):
            pass

        execution = next(iter(ws.executions.values()))
        assert execution.status == "succeeded"
        assert "RuntimeError: check broken" in (
            execution.post_check_error or "")

    def test_post_check_invalid_return_recorded(
            self, recorder_factory):
        make, ws = recorder_factory

        def check(_: PostCheckContext) -> str:  # type: ignore[return-value]
            return "not a HaltDirective"

        rec = make(post_checks={"noop": check})
        with rec.begin(block="noop"):
            pass

        execution = next(iter(ws.executions.values()))
        assert execution.halt_directive is None
        assert "expected HaltDirective" in (
            execution.post_check_error or "")

    def test_no_post_check_leaves_execution_unchanged(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="noop"):
            pass
        execution = next(iter(ws.executions.values()))
        assert execution.halt_directive is None
        assert execution.post_check_error is None


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

    def test_output_root_created_during_body(
            self, recorder_factory):
        make, _ = recorder_factory
        rec = make()
        seen: list[Path] = []
        with rec.begin(block="emits_two") as ctx:
            assert ctx.output_root is not None
            seen.append(ctx.output_root)
            assert ctx.output_root.is_dir()
            for slot in ("prediction", "game_step"):
                assert ctx.output_dir(slot).is_dir()
        assert not seen[0].exists()

    def test_output_root_removed_on_success(
            self, recorder_factory):
        make, _ = recorder_factory
        rec = make()
        seen: list[Path] = []
        with rec.begin(block="emits_two") as ctx:
            seen.append(ctx.output_root)
            _emit_copy_output(ctx, "prediction", {"v": 1})
        assert not seen[0].exists()

    def test_output_root_removed_on_failure_when_disabled(
            self, recorder_factory):
        make, _ = recorder_factory
        rec = make()
        seen: list[Path] = []
        with pytest.raises(RuntimeError), rec.begin(block="emits_two") as ctx:
            seen.append(ctx.output_root)
            raise RuntimeError("boom")
        assert not seen[0].exists()

    def test_output_root_retained_on_failure_by_default(
            self, tmp_path: Path, monkeypatch):
        template, ws = _make_workspace(tmp_path)
        monkeypatch.delenv(
            "FLYWHEEL_RETAIN_FAILED_OUTPUT_DIRS", raising=False)
        rec = LocalBlockRecorder(
            workspace=ws, template=template)
        seen: list[Path] = []
        with pytest.raises(RuntimeError), rec.begin(block="emits_two") as ctx:
            seen.append(ctx.output_root)
            raise RuntimeError("boom")
        assert seen[0].exists(), (
            "default behaviour retains output dir on failure")
        # Cleanup so tmp_path teardown doesn't trip on stale state.
        shutil.rmtree(seen[0], ignore_errors=True)


class TestNoIntermediateRunningExecution:
    """Critical contract: no execution exists with status=='running'."""

    def test_in_body_workspace_has_no_intermediate_execution(
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
        execution = next(iter(ws.executions.values()))
        assert execution.status == "succeeded"


class TestLocalExecutionContext:
    """Independent unit coverage of the dataclass surface."""

    def test_output_dir_returns_expected_path(self, tmp_path: Path):
        out_dir = tmp_path / "outputs" / "a"
        out_dir.mkdir(parents=True)
        ctx = LocalExecutionContext(
            execution_id="exec_test01",
            block_name="noop",
            input_bindings={},
            input_paths={},
            scratch_dir=str(tmp_path / "scratch"),
            params={},
            parent_execution_id=None,
            output_root=tmp_path / "outputs",
            _output_dirs={"a": out_dir},
        )
        assert ctx.output_dir("a") == out_dir

    def test_output_dir_unknown_raises(self, tmp_path: Path):
        ctx = LocalExecutionContext(
            execution_id="exec_test01",
            block_name="noop",
            input_bindings={},
            input_paths={},
            scratch_dir=str(tmp_path),
            params={},
            parent_execution_id=None,
            output_root=tmp_path,
            _output_dirs={},
        )
        with pytest.raises(KeyError):
            ctx.output_dir("missing")


class TestExecutionIsolation:
    """Recorder writes nothing under failure scenarios mid-sequence."""

    def test_two_sequential_runs_record_two_executions(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="noop"):
            pass
        with rec.begin(block="noop"):
            pass
        executions = list(ws.executions.values())
        assert len(executions) == 2
        assert all(r.status == "succeeded" for r in executions)
        assert executions[0].id != executions[1].id

    def test_failed_then_succeeded_runs_recorded_separately(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with pytest.raises(RuntimeError), rec.begin(block="noop"):
            raise RuntimeError("first attempt failed")
        with rec.begin(block="noop"):
            pass
        executions = sorted(
            ws.executions.values(), key=lambda r: r.started_at)
        assert [r.status for r in executions] == ["failed", "succeeded"]


class TestExecutionRecordFields:
    """Exhaustive contract checks for the fields downstream consumes."""

    def test_runner_field_propagates(self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="noop", runner="lifecycle"):
            pass
        execution: BlockExecution = next(iter(ws.executions.values()))
        assert execution.runner == "lifecycle"

    def test_image_pulled_from_block_definition(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="noop"):
            pass
        execution = next(iter(ws.executions.values()))
        # noop has no container image; BlockDefinition surfaces
        # an empty string for that, not None — the recorder
        # passes it through verbatim.
        assert execution.image == ""

    def test_elapsed_s_is_recorded_as_float(
            self, recorder_factory):
        make, ws = recorder_factory
        rec = make()
        with rec.begin(block="noop"):
            pass
        execution = next(iter(ws.executions.values()))
        assert isinstance(execution.elapsed_s, float)
        assert execution.elapsed_s >= 0.0
