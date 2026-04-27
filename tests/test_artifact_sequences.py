from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from flywheel.artifact import BlockExecution
from flywheel.container import ContainerResult
from flywheel.execution import RuntimeResult, run_block
from flywheel.persistent_runtime import PersistentRuntimeResult
from flywheel.run_record import RunMemberRecord, RunStepRecord
from flywheel.sequence import RunContext, SequenceScope
from flywheel.state import state_compatibility_identity
from flywheel.template import Template
from flywheel.workspace import Workspace
from tests._inline_blocks import from_yaml_with_inline_blocks


SEQUENCE_TEMPLATE_YAML = """\
artifacts:
  - name: game_state
    kind: copy
  - name: action
    kind: copy
  - name: checkpoint
    kind: copy

blocks:
  - name: state
    image: state:latest
    outputs:
      normal:
        - name: game_state
          container_path: /output/game_state
          sequence:
            name: arc_game
            scope: enclosing_lane
            role: state
  - name: action
    image: action:latest
    outputs:
      normal:
        - name: action
          container_path: /output/action
          sequence:
            name: arc_game
            scope: enclosing_lane
            role: action
  - name: history
    image: history:latest
    inputs:
      - name: arc_history
        container_path: /input/arc_history
        sequence:
          name: arc_game
          scope: enclosing_lane
    outputs:
      normal:
        - name: checkpoint
          container_path: /output/checkpoint
"""


def _workspace(tmp_path: Path, template: Template) -> Workspace:
    ws_path = tmp_path / "ws"
    (ws_path / "artifacts").mkdir(parents=True)
    (ws_path / "proposals").mkdir()
    (ws_path / "states").mkdir()
    return Workspace(
        name="ws",
        path=ws_path,
        template_name=template.name,
        created_at=datetime.now(UTC),
        artifact_declarations={a.name: a.kind for a in template.artifacts},
        artifacts={},
    )


def _template(tmp_path: Path, text: str = SEQUENCE_TEMPLATE_YAML) -> Template:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "template.yaml"
    path.write_text(text, encoding="utf-8")
    return from_yaml_with_inline_blocks(path)


class SequenceRunner:
    def __init__(self) -> None:
        self.manifests: list[dict] = []

    def run(self, plan, args=None):
        if plan.block_name in {"state", "action"}:
            slot = plan.block_name if plan.block_name == "action" else "game_state"
            out = plan.proposal_dirs[slot]
            (out / f"{plan.block_name}.txt").write_text(
                plan.block_name,
                encoding="utf-8",
            )
        elif plan.block_name == "history":
            host_path = next(
                Path(host)
                for host, container, _mode in plan.mounts
                if container == "/input/arc_history"
            )
            self.manifests.append(json.loads(
                (host_path / "manifest.json").read_text(encoding="utf-8")
            ))
            checkpoint = plan.proposal_dirs["checkpoint"]
            (checkpoint / "ok.txt").write_text("ok", encoding="utf-8")
        return RuntimeResult(
            termination_reason="normal",
            container_result=ContainerResult(exit_code=0, elapsed_s=0.1),
            announcement="normal",
        )


class EmptyRunner:
    def run(self, plan, args=None):
        return RuntimeResult(
            termination_reason="normal",
            container_result=ContainerResult(exit_code=0, elapsed_s=0.1),
            announcement="normal",
        )


def test_template_parses_sequence_inputs_and_outputs(tmp_path: Path) -> None:
    template = _template(tmp_path)
    state = next(block for block in template.blocks if block.name == "state")
    history = next(block for block in template.blocks if block.name == "history")

    output_sequence = state.outputs_for("normal")[0].sequence
    assert output_sequence is not None
    assert output_sequence.name == "arc_game"
    assert output_sequence.scope == "enclosing_lane"
    assert output_sequence.role == "state"

    input_sequence = history.inputs[0].sequence
    assert input_sequence is not None
    assert input_sequence.name == "arc_game"
    assert input_sequence.scope == "enclosing_lane"
    assert input_sequence.role is None


def test_input_sequence_slot_need_not_be_artifact(tmp_path: Path) -> None:
    template = _template(tmp_path)
    assert template.artifact_declaration("arc_history") is None


def test_input_sequence_rejects_role(tmp_path: Path) -> None:
    text = SEQUENCE_TEMPLATE_YAML.replace(
        "scope: enclosing_lane\n    outputs:",
        "scope: enclosing_lane\n          role: state\n    outputs:",
    )
    with pytest.raises(ValueError, match="sequence.role is only valid"):
        _template(tmp_path, text)


def test_unknown_sequence_scope_rejected(tmp_path: Path) -> None:
    text = SEQUENCE_TEMPLATE_YAML.replace(
        "scope: enclosing_lane",
        "scope: execution",
        1,
    )
    with pytest.raises(ValueError, match="sequence.scope"):
        _template(tmp_path, text)


def test_lane_scoped_sequence_appends_and_stages_manifest(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path)
    ws = _workspace(tmp_path, template)
    run = ws.begin_run("pattern:test", lanes=["lane_a", "lane_b"])
    runner = SequenceRunner()

    state_result = run_block(
        ws,
        "state",
        template,
        tmp_path,
        container_runner=runner,
        run_context=RunContext(run.id, "lane_a"),
    )
    ws.record_run_step(
        run.id,
        RunStepRecord(
            name="state_a",
            min_successes="all",
            status="succeeded",
            members=[
                RunMemberRecord(
                    name="state",
                    block_name="state",
                    status="succeeded",
                    lane="lane_a",
                    execution_id=state_result.execution_id,
                    output_bindings=dict(state_result.execution.output_bindings),
                )
            ],
        ),
    )
    action_result = run_block(
        ws,
        "action",
        template,
        tmp_path,
        container_runner=runner,
        run_context=RunContext(run.id, "lane_a"),
    )
    ws.record_run_step(
        run.id,
        RunStepRecord(
            name="action_a",
            min_successes="all",
            status="succeeded",
            members=[
                RunMemberRecord(
                    name="action",
                    block_name="action",
                    status="succeeded",
                    lane="lane_a",
                    execution_id=action_result.execution_id,
                    output_bindings=dict(action_result.execution.output_bindings),
                )
            ],
        ),
    )

    history_result = run_block(
        ws,
        "history",
        template,
        tmp_path,
        container_runner=runner,
        run_context=RunContext(run.id, "lane_a"),
    )

    manifest = runner.manifests[-1]
    assert manifest["manifest_version"] == 1
    assert manifest["sequence_name"] == "arc_game"
    assert manifest["scope"] == {
        "kind": "lane",
        "run_id": run.id,
        "lane": "lane_a",
    }
    assert manifest["length"] == 2
    assert [entry["directory"] for entry in manifest["entries"]] == [
        "00000_state",
        "00001_action",
    ]
    assert [entry["role"] for entry in manifest["entries"]] == [
        "state",
        "action",
    ]
    binding = history_result.execution.input_sequence_bindings["arc_history"]
    assert [entry.artifact_id for entry in binding.entries] == [
        state_result.execution.output_bindings["game_state"],
        action_result.execution.output_bindings["action"],
    ]


def test_empty_sequence_input_is_bound_with_empty_manifest(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path)
    ws = _workspace(tmp_path, template)
    run = ws.begin_run("pattern:test", lanes=["lane_a"])
    runner = SequenceRunner()

    result = run_block(
        ws,
        "history",
        template,
        tmp_path,
        container_runner=runner,
        run_context=RunContext(run.id, "lane_a"),
    )

    assert runner.manifests[-1]["length"] == 0
    assert runner.manifests[-1]["entries"] == []
    assert result.execution.input_bindings == {}
    assert result.execution.input_sequence_bindings["arc_history"].entries == []


def test_rejected_output_does_not_append_sequence(tmp_path: Path) -> None:
    template = _template(tmp_path)
    ws = _workspace(tmp_path, template)
    run = ws.begin_run("pattern:test", lanes=["lane_a"])

    with pytest.raises(RuntimeError, match="no output bytes written"):
        run_block(
            ws,
            "state",
            template,
            tmp_path,
            container_runner=EmptyRunner(),
            run_context=RunContext(run.id, "lane_a"),
        )

    assert ws.sequence_entries == []


def test_partial_output_failure_does_not_append_any_sequence(
    tmp_path: Path,
) -> None:
    text = """\
artifacts:
  - name: game_state
    kind: copy
  - name: action
    kind: copy

blocks:
  - name: mixed
    image: mixed:latest
    outputs:
      normal:
        - name: game_state
          container_path: /output/game_state
          sequence:
            name: arc_game
            scope: enclosing_lane
            role: state
        - name: action
          container_path: /output/action
          sequence:
            name: arc_game
            scope: enclosing_lane
            role: action
"""
    template = _template(tmp_path, text)
    ws = _workspace(tmp_path, template)
    run = ws.begin_run("pattern:test", lanes=["lane_a"])

    class PartialRunner:
        def run(self, plan, args=None):
            (plan.proposal_dirs["game_state"] / "state.txt").write_text(
                "state",
                encoding="utf-8",
            )
            return RuntimeResult(
                termination_reason="normal",
                container_result=ContainerResult(exit_code=0, elapsed_s=0.1),
                announcement="normal",
            )

    with pytest.raises(RuntimeError, match="no output bytes written"):
        run_block(
            ws,
            "mixed",
            template,
            tmp_path,
            container_runner=PartialRunner(),
            run_context=RunContext(run.id, "lane_a"),
        )

    execution = next(iter(ws.executions.values()))
    assert execution.status == "failed"
    assert "game_state" in execution.output_bindings
    assert execution.rejected_outputs["action"].reason == "no output bytes written"
    assert ws.sequence_entries == []


def test_sequence_append_happens_after_artifact_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _template(tmp_path)
    ws = _workspace(tmp_path, template)
    run = ws.begin_run("pattern:test", lanes=["lane_a"])
    calls: list[str] = []

    original_register = ws.register_artifact
    original_append = ws.record_sequence_entry

    def register_spy(*args, **kwargs):
        calls.append("register")
        return original_register(*args, **kwargs)

    def append_spy(*args, **kwargs):
        calls.append("append")
        return original_append(*args, **kwargs)

    monkeypatch.setattr(ws, "register_artifact", register_spy)
    monkeypatch.setattr(ws, "record_sequence_entry", append_spy)

    run_block(
        ws,
        "state",
        template,
        tmp_path,
        container_runner=SequenceRunner(),
        run_context=RunContext(run.id, "lane_a"),
    )

    assert calls == ["register", "append"]


def test_enclosing_run_output_requires_run_context(tmp_path: Path) -> None:
    text = SEQUENCE_TEMPLATE_YAML.replace(
        "scope: enclosing_lane",
        "scope: enclosing_run",
        1,
    )
    template = _template(tmp_path, text)
    ws = _workspace(tmp_path, template)

    with pytest.raises(ValueError, match="requires pattern run context"):
        run_block(ws, "state", template, tmp_path, container_runner=EmptyRunner())
    assert ws.executions == {}


def test_enclosing_lane_input_requires_run_context_before_container(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path)
    ws = _workspace(tmp_path, template)
    runner = SequenceRunner()

    with pytest.raises(ValueError, match="requires pattern lane context"):
        run_block(ws, "history", template, tmp_path, container_runner=runner)

    assert runner.manifests == []
    assert ws.executions == {}


def test_sequence_indices_are_partitioned_by_scope(tmp_path: Path) -> None:
    template = _template(tmp_path)
    ws = _workspace(tmp_path, template)
    src = tmp_path / "src"
    src.mkdir()
    (src / "x.txt").write_text("x", encoding="utf-8")
    artifact = ws.register_artifact("game_state", src, source="test")
    run = ws.begin_run("pattern:test", lanes=["lane_a", "lane_b"])

    ws.record_sequence_entry(
        sequence_name="arc_game",
        scope=SequenceScope.for_lane(run.id, "lane_a"),
        artifact_id=artifact.id,
        role="state",
    )
    ws.record_sequence_entry(
        sequence_name="arc_game",
        scope=SequenceScope.for_lane(run.id, "lane_a"),
        artifact_id=artifact.id,
        role="state",
    )
    ws.record_sequence_entry(
        sequence_name="arc_game",
        scope=SequenceScope.for_lane(run.id, "lane_b"),
        artifact_id=artifact.id,
        role="state",
    )
    ws.record_sequence_entry(
        sequence_name="arc_game",
        scope=SequenceScope.workspace(),
        artifact_id=artifact.id,
        role="state",
    )
    ws.record_sequence_entry(
        sequence_name="arc_game",
        scope=SequenceScope.workspace(),
        artifact_id=artifact.id,
        role="state",
    )

    assert [
        entry.index
        for entry in ws.resolve_sequence_snapshot(
            "arc_game", SequenceScope.for_lane(run.id, "lane_a")
        )
    ] == [0, 1]
    assert [
        entry.index
        for entry in ws.resolve_sequence_snapshot(
            "arc_game", SequenceScope.for_lane(run.id, "lane_b")
        )
    ] == [0]
    assert [
        entry.index
        for entry in ws.resolve_sequence_snapshot(
            "arc_game", SequenceScope.workspace()
        )
    ] == [0, 1]


def test_sequence_entries_round_trip_and_gaps_are_rejected(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path)
    ws = _workspace(tmp_path, template)
    src = tmp_path / "src"
    src.mkdir()
    (src / "x.txt").write_text("x", encoding="utf-8")
    artifact = ws.register_artifact("game_state", src, source="test")
    ws.record_sequence_entry(
        sequence_name="arc_game",
        scope=SequenceScope.workspace(),
        artifact_id=artifact.id,
        role="state",
    )

    loaded = Workspace.load(ws.path)
    assert len(loaded.sequence_entries) == 1
    assert loaded.sequence_entries[0].artifact_id == artifact.id

    yaml_path = ws.path / "workspace.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["sequence_entries"][0]["index"] = 1
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid indices"):
        Workspace.load(ws.path)


def test_sequence_input_binding_load_rejects_unknown_artifact(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path)
    ws = _workspace(tmp_path, template)
    run = ws.begin_run("pattern:test", lanes=["lane_a"])
    runner = SequenceRunner()
    run_block(
        ws,
        "state",
        template,
        tmp_path,
        container_runner=runner,
        run_context=RunContext(run.id, "lane_a"),
    )
    consumer = run_block(
        ws,
        "history",
        template,
        tmp_path,
        container_runner=runner,
        run_context=RunContext(run.id, "lane_a"),
    )

    yaml_path = ws.path / "workspace.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["executions"][consumer.execution_id]["input_sequence_bindings"][
        "arc_history"
    ]["entries"][0]["artifact_id"] = "game_state@missing"
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="references unknown artifact"):
        Workspace.load(ws.path)


def test_sequence_input_binding_is_frozen_snapshot(tmp_path: Path) -> None:
    template = _template(tmp_path)
    ws = _workspace(tmp_path, template)
    run = ws.begin_run("pattern:test", lanes=["lane_a"])
    runner = SequenceRunner()

    first = run_block(
        ws,
        "state",
        template,
        tmp_path,
        container_runner=runner,
        run_context=RunContext(run.id, "lane_a"),
    )
    consumer = run_block(
        ws,
        "history",
        template,
        tmp_path,
        container_runner=runner,
        run_context=RunContext(run.id, "lane_a"),
    )
    second = run_block(
        ws,
        "state",
        template,
        tmp_path,
        container_runner=runner,
        run_context=RunContext(run.id, "lane_a"),
    )

    assert len(ws.sequence_entries) == 2
    binding = ws.executions[consumer.execution_id].input_sequence_bindings[
        "arc_history"
    ]
    assert [entry.artifact_id for entry in binding.entries] == [
        first.execution.output_bindings["game_state"]
    ]
    assert second.execution.output_bindings["game_state"] not in {
        entry.artifact_id for entry in binding.entries
    }


def test_invoked_child_inherits_run_context_for_sequence_append(
    tmp_path: Path,
) -> None:
    text = """\
artifacts:
  - name: game_state
    kind: copy
  - name: trigger
    kind: copy

blocks:
  - name: parent
    image: parent:latest
    outputs:
      eval_requested:
        - name: trigger
          container_path: /output/trigger
    on_termination:
      eval_requested:
        invoke:
          - block: child
  - name: child
    image: child:latest
    outputs:
      normal:
        - name: game_state
          container_path: /output/game_state
          sequence:
            name: arc_game
            scope: enclosing_lane
            role: state
"""
    template = _template(tmp_path, text)
    ws = _workspace(tmp_path, template)
    run = ws.begin_run("pattern:test", lanes=["lane_a"])

    def fake_run_container(config, args=None):
        mounts = {container: Path(host) for host, container, _mode in config.mounts}
        if config.image == "parent:latest":
            (mounts["/output/trigger"] / "trigger.txt").write_text(
                "go",
                encoding="utf-8",
            )
            (mounts["/flywheel"] / "termination").write_text(
                "eval_requested",
                encoding="utf-8",
            )
        else:
            (mounts["/output/game_state"] / "state.txt").write_text(
                "state",
                encoding="utf-8",
            )
            (mounts["/flywheel"] / "termination").write_text(
                "normal",
                encoding="utf-8",
            )
        return ContainerResult(exit_code=0, elapsed_s=0.1)

    with patch("flywheel.execution.run_container", side_effect=fake_run_container):
        result = run_block(
            ws,
            "parent",
            template,
            tmp_path,
            run_context=RunContext(run.id, "lane_a"),
        )

    invocation = next(iter(ws.invocations.values()))
    child = ws.executions[invocation.invoked_execution_id]
    assert child.invoking_execution_id == result.execution_id
    assert len(ws.sequence_entries) == 1
    entry = ws.sequence_entries[0]
    assert entry.scope == SequenceScope.for_lane(run.id, "lane_a")
    assert entry.artifact_id == child.output_bindings["game_state"]


def test_persistent_sequence_input_uses_request_input_root(
    tmp_path: Path,
) -> None:
    text = SEQUENCE_TEMPLATE_YAML.replace(
        "  - name: history\n    image: history:latest",
        "  - name: history\n    image: history:latest\n    lifecycle: workspace_persistent",
    )
    template = _template(tmp_path, text)
    ws = _workspace(tmp_path, template)
    run = ws.begin_run("pattern:test", lanes=["lane_a"])
    one_shot = SequenceRunner()
    run_block(
        ws,
        "state",
        template,
        tmp_path,
        container_runner=one_shot,
        run_context=RunContext(run.id, "lane_a"),
    )

    class PersistentRunner:
        def __init__(self) -> None:
            self.seen_plan = None

        def run(self, plan, args=None):
            self.seen_plan = plan
            manifest_path = (
                plan.proposals_root
                / "input"
                / "arc_history"
                / "manifest.json"
            )
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest["length"] == 1
            assert not any(
                container == "/input/arc_history"
                for _host, container, _mode in plan.mounts
            )
            (plan.proposal_dirs["checkpoint"] / "ok.txt").write_text(
                "ok",
                encoding="utf-8",
            )
            return PersistentRuntimeResult(
                termination_reason="normal",
                container_result=ContainerResult(exit_code=0, elapsed_s=0.1),
                announcement="normal",
            )

    persistent = PersistentRunner()
    run_block(
        ws,
        "history",
        template,
        tmp_path,
        persistent_runner=persistent,
        run_context=RunContext(run.id, "lane_a"),
    )

    assert persistent.seen_plan is not None
    assert persistent.seen_plan.runner == "container_persistent"


def test_sequence_declarations_affect_state_compatibility_hash(
    tmp_path: Path,
) -> None:
    base = _template(tmp_path / "base", """\
artifacts:
  - name: game_state
    kind: copy

blocks:
  - name: stateful
    image: stateful:latest
    state: managed
    inputs:
      - name: game_state
        container_path: /input/game_state
    outputs:
      normal:
        - name: game_state
          container_path: /output/game_state
""")
    with_sequence = _template(tmp_path / "seq", """\
artifacts:
  - name: game_state
    kind: copy

blocks:
  - name: stateful
    image: stateful:latest
    state: managed
    inputs:
      - name: game_state
        container_path: /input/game_state
        sequence:
          name: arc_game
          scope: workspace
    outputs:
      normal:
        - name: game_state
          container_path: /output/game_state
          sequence:
            name: arc_game
            scope: workspace
            role: state
""")

    assert (
        state_compatibility_identity(base.blocks[0])["block_template_hash"]
        != state_compatibility_identity(
            with_sequence.blocks[0])["block_template_hash"]
    )


def test_lane_scoped_sequence_rejects_artifact_from_other_lane(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path)
    ws = _workspace(tmp_path, template)
    run = ws.begin_run("pattern:test", lanes=["lane_a", "lane_b"])
    execution = BlockExecution(
        id="exec_a",
        block_name="state",
        started_at=datetime.now(UTC),
        status="succeeded",
        input_bindings={},
        output_bindings={},
        runner="container_one_shot",
        termination_reason="normal",
    )
    ws.record_execution(execution)
    src = tmp_path / "src"
    src.mkdir()
    (src / "x.txt").write_text("x", encoding="utf-8")
    artifact = ws.register_artifact(
        "game_state",
        src,
        produced_by=execution.id,
        persist=True,
    )
    ws.record_run_step(
        run.id,
        RunStepRecord(
            name="state_a",
            min_successes="all",
            status="succeeded",
            members=[
                RunMemberRecord(
                    name="state",
                    block_name="state",
                    status="succeeded",
                    lane="lane_a",
                    execution_id=execution.id,
                )
            ],
        ),
    )

    with pytest.raises(ValueError, match="was not produced in lane"):
        ws.record_sequence_entry(
            sequence_name="arc_game",
            scope=SequenceScope.for_lane(run.id, "lane_b"),
            artifact_id=artifact.id,
            role="state",
        )
