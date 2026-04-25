from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from flywheel.artifact_validator import ArtifactValidatorRegistry
from flywheel.container import ContainerResult
from flywheel.pattern_declaration import parse_pattern_declaration
from flywheel.pattern_execution import (
    run_pattern,
)
from flywheel.pattern_resolution import (
    RunPrefix,
    StopDecision,
    WorkspaceView,
    resolve_next_step,
)
from flywheel.run_record import RunMemberRecord, RunRecord, RunStepRecord
from flywheel.state import pattern_state_lineage_key
from flywheel.template import Template
from flywheel.workspace import Workspace
from tests._inline_blocks import from_yaml_with_inline_blocks

TEMPLATE_YAML = """\
artifacts:
  - name: checkpoint
    kind: copy
  - name: score
    kind: copy

blocks:
  - name: train
    image: train:latest
    outputs:
      - name: checkpoint
        container_path: /output/checkpoint
  - name: eval
    image: eval:latest
    inputs:
      - name: checkpoint
        container_path: /input/checkpoint
    outputs:
      - name: score
        container_path: /output/score
"""


def _setup_workspace(
    tmp_path: Path,
    template_yaml: str = TEMPLATE_YAML,
) -> tuple[Path, Template, Workspace]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    foundry_dir = project_root / "foundry"
    templates_dir = foundry_dir / "templates" / "workspaces"
    templates_dir.mkdir(parents=True)
    template_path = templates_dir / "test.yaml"
    template_path.write_text(template_yaml)
    template = from_yaml_with_inline_blocks(template_path)
    workspace = Workspace.create("ws", template, foundry_dir)
    return project_root, template, workspace


def _fake_container(config, args=None):
    args = args or []
    if "--fail" in args:
        return ContainerResult(exit_code=1, elapsed_s=0.1)
    for host, container_path, mode in config.mounts:
        if mode != "rw":
            continue
        path = Path(host)
        if container_path == "/flywheel":
            (path / "termination").write_text("normal")
            state_dir = path / "state"
            if state_dir.exists():
                (state_dir / "marker.txt").write_text("state")
        elif container_path.endswith("checkpoint"):
            (path / "checkpoint.pt").write_text("weights")
        elif container_path.endswith("score"):
            (path / "score.json").write_text('{"mean": 1.0}')
    return ContainerResult(exit_code=0, elapsed_s=0.1)


def _declaration():
    return parse_pattern_declaration({
        "name": "train_eval",
        "steps": [
            {
                "name": "train",
                "cohort": {
                    "min_successes": "all",
                    "members": [
                        {"name": "train_dueling", "block": "train"}
                    ],
                },
            },
            {
                "name": "eval",
                "cohort": {
                    "min_successes": "all",
                    "members": [
                        {
                            "name": "eval_dueling",
                            "block": "eval",
                            "inputs": {
                                "checkpoint": {
                                    "from_step": "train",
                                    "member": "train_dueling",
                                    "output": "checkpoint",
                                }
                            },
                        }
                    ],
                },
            },
        ],
    })


def test_resolver_emits_first_cohort_and_is_deterministic():
    pattern = _declaration()
    run = RunRecord(
        id="run_1",
        kind="pattern:train_eval",
        started_at=datetime.now(UTC),
    )
    prefix = RunPrefix.from_run(run)
    view = WorkspaceView(execution_statuses={})

    first = resolve_next_step(pattern, prefix, view)
    second = resolve_next_step(pattern, prefix, view)

    assert first == second
    assert first.step.name == "train"


def test_resolver_emits_second_cohort_after_first_succeeds():
    pattern = _declaration()
    run = RunRecord(
        id="run_1",
        kind="pattern:train_eval",
        started_at=datetime.now(UTC),
        steps=[
            RunStepRecord(
                name="train",
                min_successes="all",
                status="succeeded",
                members=[
                    RunMemberRecord(
                        name="train_dueling",
                        block_name="train",
                        status="succeeded",
                        execution_id="exec_1",
                        output_bindings={"checkpoint": "checkpoint@1"},
                    )
                ],
            )
        ],
    )

    decision = resolve_next_step(
        pattern, RunPrefix.from_run(run), WorkspaceView(execution_statuses={}))

    assert decision.step.name == "eval"


def test_resolver_stops_succeeded_after_final_cohort():
    pattern = _declaration()
    run = RunRecord(
        id="run_1",
        kind="pattern:train_eval",
        started_at=datetime.now(UTC),
        steps=[
            RunStepRecord("train", "all", "succeeded", []),
            RunStepRecord("eval", "all", "succeeded", []),
        ],
    )

    decision = resolve_next_step(
        pattern, RunPrefix.from_run(run), WorkspaceView(execution_statuses={}))

    assert decision == StopDecision(status="succeeded")


def test_resolver_stops_failed_after_failed_step():
    pattern = _declaration()
    run = RunRecord(
        id="run_1",
        kind="pattern:train_eval",
        started_at=datetime.now(UTC),
        steps=[RunStepRecord("train", "all", "failed", [])],
    )

    decision = resolve_next_step(
        pattern, RunPrefix.from_run(run), WorkspaceView(execution_statuses={}))

    assert decision == StopDecision(
        status="failed", error="step 'train' failed")


def test_run_pattern_records_train_eval_membership(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(tmp_path)

    with patch("flywheel.execution.run_container", side_effect=_fake_container):
        result = run_pattern(
            workspace, _declaration(), template, project_root)

    reloaded = Workspace.load(workspace.path)
    run = reloaded.runs[result.run_id]
    assert run.status == "succeeded"
    assert [step.name for step in run.steps] == ["train", "eval"]
    train_member = run.steps[0].members[0]
    eval_member = run.steps[1].members[0]
    assert train_member.status == "succeeded"
    assert eval_member.status == "succeeded"
    train_execution = reloaded.executions[train_member.execution_id]
    eval_execution = reloaded.executions[eval_member.execution_id]
    assert eval_execution.input_bindings == {
        "checkpoint": train_execution.output_bindings["checkpoint"]
    }


def test_run_pattern_uses_artifact_validators(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(tmp_path)
    seen: list[str] = []
    registry = ArtifactValidatorRegistry()

    def record_validation(name, declaration, staged_path):
        del declaration
        assert staged_path.exists()
        seen.append(name)

    registry.register("checkpoint", record_validation)
    registry.register("score", record_validation)

    with patch("flywheel.execution.run_container", side_effect=_fake_container):
        run_pattern(
            workspace,
            _declaration(),
            template,
            project_root,
            validator_registry=registry,
        )

    assert seen == ["checkpoint", "score"]


def test_run_pattern_derives_managed_state_lineage(
    tmp_path: Path,
):
    template_yaml = TEMPLATE_YAML.replace(
        "    image: train:latest",
        "    image: train:latest\n    state: managed",
    )
    project_root, template, workspace = _setup_workspace(
        tmp_path, template_yaml)

    with patch("flywheel.execution.run_container", side_effect=_fake_container):
        result = run_pattern(
            workspace, _declaration(), template, project_root)

    reloaded = Workspace.load(workspace.path)
    run = reloaded.runs[result.run_id]
    train_member = run.steps[0].members[0]
    train_execution = reloaded.executions[train_member.execution_id]
    assert train_execution.state_mode == "managed"
    assert train_execution.state_snapshot_id is not None

    snapshot = reloaded.state_snapshots[train_execution.state_snapshot_id]
    assert snapshot.lineage_key == pattern_state_lineage_key(
        result.run_id, "train", "train_dueling")
    assert (
        reloaded.state_snapshot_path(snapshot.id) / "marker.txt"
    ).read_text() == "state"


def test_pattern_state_lineage_key_is_injective_for_underscores():
    run_id = "run_collision"

    assert pattern_state_lineage_key(
        run_id, "train_dueling", "a") != pattern_state_lineage_key(
            run_id, "train", "dueling_a")


def test_min_successes_one_runs_all_members_and_succeeds(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(tmp_path)
    pattern = parse_pattern_declaration({
        "name": "try_two",
        "steps": [
            {
                "name": "train",
                "cohort": {
                    "min_successes": 1,
                    "members": [
                        {
                            "name": "bad",
                            "block": "train",
                            "args": ["--fail"],
                        },
                        {"name": "good", "block": "train"},
                    ],
                },
            }
        ],
    })

    with patch("flywheel.execution.run_container", side_effect=_fake_container):
        result = run_pattern(workspace, pattern, template, project_root)

    run = Workspace.load(workspace.path).runs[result.run_id]
    assert run.status == "succeeded"
    assert [member.status for member in run.steps[0].members] == [
        "failed",
        "succeeded",
    ]


def test_min_successes_all_stops_after_first_failure(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(tmp_path)
    pattern = parse_pattern_declaration({
        "name": "all_required",
        "steps": [
            {
                "name": "train",
                "cohort": {
                    "min_successes": "all",
                    "members": [
                        {
                            "name": "bad",
                            "block": "train",
                            "args": ["--fail"],
                        },
                        {"name": "not_launched", "block": "train"},
                    ],
                },
            }
        ],
    })

    with patch("flywheel.execution.run_container", side_effect=_fake_container):
        try:
            run_pattern(workspace, pattern, template, project_root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected pattern failure")

    run = next(iter(Workspace.load(workspace.path).runs.values()))
    assert run.status == "failed"
    assert [member.status for member in run.steps[0].members] == [
        "failed",
        "skipped",
    ]
    assert len(workspace.executions) == 1
