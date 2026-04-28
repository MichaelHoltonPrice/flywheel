from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from flywheel.artifact_validator import ArtifactValidatorRegistry
from flywheel.container import ContainerResult
from flywheel.pattern_declaration import parse_pattern_declaration
from flywheel.pattern_execution import (
    resolve_pattern_params,
    run_pattern,
)
from flywheel.pattern_lanes import DEFAULT_LANE
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

INVOCATION_FAILURE_TEMPLATE_YAML = """\
artifacts:
  - name: bot
    kind: copy
  - name: score
    kind: copy

blocks:
  - name: agent
    image: agent:latest
    outputs:
      eval_requested:
        - name: bot
          container_path: /output/bot
    on_termination:
      eval_requested:
        invoke:
          - block: eval_bad
            bind:
              bot: bot
  - name: eval_bad
    image: eval-bad:latest
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      - name: score
        container_path: /output/score
"""

LANE_TEMPLATE_YAML = """\
artifacts:
  - name: bot
    kind: copy

blocks:
  - name: improve
    image: improve:latest
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      - name: bot
        container_path: /output/bot
"""

OPTIONAL_LANE_TEMPLATE_YAML = """\
artifacts:
  - name: bot
    kind: copy
  - name: note
    kind: copy

blocks:
  - name: optional
    image: optional:latest
    inputs:
      - name: bot
        container_path: /input/bot
        optional: true
      - name: note
        container_path: /input/note
        optional: true
    outputs:
      - name: note
        container_path: /output/note
"""

INVOCATION_OUTPUT_TEMPLATE_YAML = """\
artifacts:
  - name: bot
    kind: copy
  - name: score
    kind: copy

blocks:
  - name: agent
    image: agent:latest
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      eval_requested:
        - name: bot
          container_path: /output/bot
    on_termination:
      eval_requested:
        invoke:
          - block: eval
            bind:
              bot: bot
  - name: eval
    image: eval:latest
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      - name: score
        container_path: /output/score
  - name: consume_score
    image: consume-score:latest
    inputs:
      - name: score
        container_path: /input/score
    outputs:
      - name: bot
        container_path: /output/bot
"""

RUN_UNTIL_TEMPLATE_YAML = """\
artifacts:
  - name: bot
    kind: copy
  - name: score
    kind: copy
  - name: ideas
    kind: copy

blocks:
  - name: ImproveBot
    image: improve:latest
    state: managed
    inputs:
      - name: bot
        container_path: /input/bot
      - name: score
        container_path: /input/score
        optional: true
    outputs:
      eval_requested:
        - name: bot
          container_path: /output/bot
      normal:
        - name: bot
          container_path: /output/bot
      aborted:
        - name: bot
          container_path: /output/bot
    on_termination:
      eval_requested:
        invoke:
          - block: EvalBot
            bind:
              bot: bot
  - name: EvalBot
    image: eval:latest
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      - name: score
        container_path: /output/score
  - name: Brainstorm
    image: brainstorm:latest
    inputs:
      - name: score
        container_path: /input/score
        optional: true
    outputs:
      - name: ideas
        container_path: /output/ideas
"""

PARAM_INVOCATION_TEMPLATE_YAML = """\
artifacts:
  - name: bot
    kind: copy
  - name: score
    kind: copy

blocks:
  - name: improve
    image: improve:latest
    outputs:
      eval_requested:
        - name: bot
          container_path: /output/bot
    on_termination:
      eval_requested:
        invoke:
          - block: eval
            bind:
              bot: bot
            args:
              - --episodes
              - ${params.eval_episodes}
  - name: eval
    image: eval:latest
    inputs:
      - name: bot
        container_path: /input/bot
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


def test_pattern_params_persist_and_substitute_into_env_args_and_invocations(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, PARAM_INVOCATION_TEMPLATE_YAML)
    pattern = parse_pattern_declaration({
        "name": "improve",
        "params": {
            "model": {
                "type": "string",
                "default": "claude-sonnet",
            },
            "eval_episodes": {
                "type": "int",
                "default": 4000,
            },
        },
        "steps": [
            {
                "name": "improve",
                "cohort": {
                    "min_successes": "all",
                    "members": [
                        {
                            "name": "agent",
                            "block": "improve",
                            "args": [
                                "--budget",
                                "${params.eval_episodes}",
                            ],
                            "env": {
                                "MODEL": "${params.model}",
                            },
                        },
                    ],
                },
            }
        ],
    })
    seen: list[tuple[str, list[str] | None, dict[str, str]]] = []

    def fake_container(config, args=None):
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        seen.append((config.image, args, dict(config.env)))
        if config.image == "improve:latest":
            assert config.env["MODEL"] == "claude-opus"
            assert args == ["--budget", "123"]
            bot = mounts["/output/bot"] / "bot.py"
            bot.parent.mkdir(parents=True, exist_ok=True)
            bot.write_text("BOT")
            (mounts["/flywheel"] / "termination").write_text(
                "eval_requested")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        if config.image == "eval:latest":
            assert args == ["--episodes", "123"]
            score = mounts["/output/score"] / "score.json"
            score.parent.mkdir(parents=True, exist_ok=True)
            score.write_text("{}")
            (mounts["/flywheel"] / "termination").write_text("normal")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        raise AssertionError(config.image)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        result = run_pattern(
            workspace,
            pattern,
            template,
            project_root,
            param_overrides={
                "model": "claude-opus",
                "eval_episodes": "123",
            },
        )

    reloaded = Workspace.load(workspace.path)
    run = reloaded.runs[result.run_id]
    assert run.params == {
        "model": "claude-opus",
        "eval_episodes": 123,
    }
    assert [item[0] for item in seen] == ["improve:latest", "eval:latest"]
    invocation = next(iter(reloaded.invocations.values()))
    assert invocation.args == ["--episodes", "123"]


def test_pattern_param_resolution_coerces_defaults_and_overrides():
    pattern = parse_pattern_declaration({
        "name": "params",
        "params": {
            "model": {"type": "string", "default": "sonnet"},
            "episodes": {"type": "int", "default": 100},
            "threshold": {"type": "float", "default": 1},
            "dry_run": {"type": "bool", "default": "false"},
        },
        "steps": [
            {
                "name": "noop",
                "cohort": {
                    "members": [
                        {"name": "member", "block": "train"},
                    ],
                },
            }
        ],
    })

    assert resolve_pattern_params(pattern, {
        "episodes": "4000",
        "dry_run": "yes",
    }) == {
        "model": "sonnet",
        "episodes": 4000,
        "threshold": 1.0,
        "dry_run": True,
    }


def test_required_pattern_param_without_default_fails_before_run(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, PARAM_INVOCATION_TEMPLATE_YAML)
    pattern = parse_pattern_declaration({
        "name": "improve",
        "params": {
            "model": {"type": "string"},
            "eval_episodes": {"type": "int", "default": 10},
        },
        "steps": [
            {
                "name": "improve",
                "cohort": {
                    "members": [
                        {"name": "agent", "block": "improve"},
                    ],
                },
            }
        ],
    })

    try:
        run_pattern(workspace, pattern, template, project_root)
    except ValueError as exc:
        assert "required param 'model'" in str(exc)
    else:
        raise AssertionError("expected required param failure")

    assert Workspace.load(workspace.path).runs == {}


def test_pattern_param_override_type_failure_happens_before_run(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, PARAM_INVOCATION_TEMPLATE_YAML)
    pattern = parse_pattern_declaration({
        "name": "improve",
        "params": {
            "eval_episodes": {"type": "int"},
        },
        "steps": [
            {
                "name": "improve",
                "cohort": {
                    "members": [
                        {"name": "agent", "block": "improve"},
                    ],
                },
            }
        ],
    })

    try:
        run_pattern(
            workspace,
            pattern,
            template,
            project_root,
            param_overrides={"eval_episodes": "foo"},
        )
    except ValueError as exc:
        assert "not a valid int" in str(exc)
    else:
        raise AssertionError("expected type failure")

    assert Workspace.load(workspace.path).runs == {}


def test_bool_pattern_param_round_trips_through_workspace_yaml(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(tmp_path)
    pattern = parse_pattern_declaration({
        "name": "train_eval",
        "params": {
            "dry_run": {"type": "bool", "default": False},
        },
        "steps": [
            {
                "name": "train",
                "cohort": {
                    "members": [
                        {"name": "train_dueling", "block": "train"},
                    ],
                },
            }
        ],
    })

    with patch("flywheel.execution.run_container", side_effect=_fake_container):
        result = run_pattern(
            workspace,
            pattern,
            template,
            project_root,
            param_overrides={"dry_run": "true"},
        )

    assert Workspace.load(workspace.path).runs[result.run_id].params == {
        "dry_run": True,
    }


def test_use_with_bool_param_coerces_literal_bool(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(
        tmp_path, RUN_UNTIL_TEMPLATE_YAML)
    fixture_bot = project_root / "foundry" / "templates" / "assets" / "bot"
    fixture_bot.mkdir(parents=True)
    (fixture_bot / "bot.py").write_text("BASE")
    pattern = parse_pattern_declaration({
        "name": "bool_use",
        "fixtures": {"bot": "foundry/templates/assets/bot"},
        "do": [{"use": "inner", "with": {"dry_run": True}}],
        "patterns": {
            "inner": {
                "params": {"dry_run": {"type": "bool"}},
                "do": [
                    {
                        "run_until": {
                            "block": "ImproveBot",
                            "env": {"DRY_RUN": "${params.dry_run}"},
                            "continue_on": {"eval_requested": {"max": 1}},
                            "stop_on": ["normal"],
                        },
                    }
                ],
            }
        },
    })
    seen_env: list[str] = []

    def fake_container(config, args=None):
        del args
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        seen_env.append(config.env["DRY_RUN"])
        output = mounts["/output/bot"] / "bot.py"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("DONE")
        state_file = mounts["/flywheel"] / "state" / "session.txt"
        state_file.write_text("DONE")
        (mounts["/flywheel"] / "termination").write_text("normal")
        return ContainerResult(exit_code=0, elapsed_s=0.1)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        result = run_pattern(workspace, pattern, template, project_root)

    assert seen_env == ["true"]
    assert Workspace.load(workspace.path).runs[result.run_id].status == (
        "succeeded")


def test_unknown_member_param_reference_fails_before_run(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(
        tmp_path, PARAM_INVOCATION_TEMPLATE_YAML)
    pattern = parse_pattern_declaration({
        "name": "improve",
        "params": {
            "model": {"type": "string", "default": "m"},
        },
        "steps": [
            {
                "name": "improve",
                "cohort": {
                    "members": [
                        {
                            "name": "agent",
                            "block": "improve",
                            "args": ["${params.mdoel}"],
                        },
                    ],
                },
            }
        ],
    })

    try:
        run_pattern(workspace, pattern, template, project_root)
    except ValueError as exc:
        assert "unknown param" in str(exc)
        assert "mdoel" in str(exc)
    else:
        raise AssertionError("expected param reference failure")

    assert Workspace.load(workspace.path).runs == {}


def test_unknown_invocation_route_param_reference_fails_before_run(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, PARAM_INVOCATION_TEMPLATE_YAML)
    pattern = parse_pattern_declaration({
        "name": "improve",
        "params": {
            "episodes": {"type": "int", "default": 10},
        },
        "steps": [
            {
                "name": "improve",
                "cohort": {
                    "members": [
                        {"name": "agent", "block": "improve"},
                    ],
                },
            }
        ],
    })

    try:
        run_pattern(workspace, pattern, template, project_root)
    except ValueError as exc:
        assert "unknown param" in str(exc)
        assert "eval_episodes" in str(exc)
    else:
        raise AssertionError("expected route param reference failure")

    assert Workspace.load(workspace.path).runs == {}


def test_unknown_pattern_param_override_is_rejected(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(
        tmp_path, PARAM_INVOCATION_TEMPLATE_YAML)
    pattern = parse_pattern_declaration({
        "name": "improve",
        "params": {
            "model": {"type": "string"},
        },
        "steps": [
            {
                "name": "improve",
                "cohort": {
                    "members": [
                        {"name": "agent", "block": "improve"},
                    ],
                },
            }
        ],
    })

    try:
        run_pattern(
            workspace,
            pattern,
            template,
            project_root,
            param_overrides={"unknown": "x", "model": "m"},
        )
    except ValueError as exc:
        assert "unknown param" in str(exc)
    else:
        raise AssertionError("expected unknown param failure")


def test_run_until_rejects_reason_not_declared_by_block(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(
        tmp_path, RUN_UNTIL_TEMPLATE_YAML)
    pattern = parse_pattern_declaration({
        "name": "bad_reason",
        "do": [
            {
                "run_until": {
                    "block": "ImproveBot",
                    "continue_on": {"missing": {"max": 1}},
                    "stop_on": ["normal"],
                },
            }
        ],
    })

    try:
        run_pattern(workspace, pattern, template, project_root)
    except RuntimeError as exc:
        assert "not declared" in str(exc)
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected run_until validation failure")


def test_run_until_rejects_reason_in_continue_and_stop(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(
        tmp_path, RUN_UNTIL_TEMPLATE_YAML)
    pattern = parse_pattern_declaration({
        "name": "overlap",
        "do": [
            {
                "run_until": {
                    "block": "ImproveBot",
                    "continue_on": {"normal": {"max": 1}},
                    "stop_on": ["normal"],
                },
            }
        ],
    })

    try:
        run_pattern(workspace, pattern, template, project_root)
    except RuntimeError as exc:
        assert "both continue_on and stop_on" in str(exc)
    else:
        raise AssertionError("expected run_until validation failure")


def test_use_param_validation_catches_unknown_and_missing_params(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, RUN_UNTIL_TEMPLATE_YAML)
    for bad_with, expected in (
        ({"unknown": "x"}, "unknown param"),
        ({}, "missing required param"),
    ):
        pattern = parse_pattern_declaration({
            "name": "bad_use",
            "do": [{"use": "inner", "with": bad_with}],
            "patterns": {
                "inner": {
                    "params": {"needed": {"type": "string"}},
                    "do": [
                        {
                            "run_until": {
                                "block": "ImproveBot",
                                "continue_on": {
                                    "eval_requested": {"max": 1},
                                },
                                "stop_on": ["normal"],
                            },
                        }
                    ],
                }
            },
        })
        try:
            run_pattern(workspace, pattern, template, project_root)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("expected use param validation failure")


def test_pattern_lanes_keep_artifact_pedigrees_separate(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(
        tmp_path, LANE_TEMPLATE_YAML)
    baseline = project_root / "foundry" / "templates" / "assets" / "bot"
    baseline.mkdir(parents=True)
    (baseline / "bot.py").write_text("BASE")
    pattern = parse_pattern_declaration({
        "name": "improve",
        "lanes": ["A", "B"],
        "fixtures": {"bot": "foundry/templates/assets/bot"},
        "steps": [
            {
                "name": "round_1",
                "cohort": {
                    "foreach": "lanes",
                    "block": "improve",
                },
            },
            {
                "name": "round_2",
                "cohort": {
                    "foreach": "lanes",
                    "block": "improve",
                },
            },
        ],
    })
    seen_inputs: list[str] = []

    def fake_container(config, args=None):
        del args
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        source = (mounts["/input/bot"] / "bot.py").read_text()
        seen_inputs.append(source)
        output = mounts["/output/bot"] / "bot.py"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{source}->{len(seen_inputs)}")
        (mounts["/flywheel"] / "termination").write_text("normal")
        return ContainerResult(exit_code=0, elapsed_s=0.1)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        result = run_pattern(workspace, pattern, template, project_root)

    reloaded = Workspace.load(workspace.path)
    run = reloaded.runs[result.run_id]
    assert run.lanes == ["A", "B"]
    assert [(f.lane, f.name) for f in run.fixtures] == [
        ("A", "bot"),
        ("B", "bot"),
    ]
    assert len({f.artifact_id for f in run.fixtures}) == 2
    for fixture in run.fixtures:
        assert reloaded.artifacts[fixture.artifact_id].fixture_id == fixture.id
        assert reloaded.artifacts[fixture.artifact_id].produced_by is None
    assert seen_inputs == [
        "BASE",
        "BASE",
        "BASE->1",
        "BASE->2",
    ]
    round_2 = run.steps[1]
    lane_a = round_2.members[0]
    lane_b = round_2.members[1]
    assert lane_a.lane == "A"
    assert lane_b.lane == "B"
    exec_a = reloaded.executions[lane_a.execution_id]
    exec_b = reloaded.executions[lane_b.execution_id]
    assert exec_a.input_bindings["bot"] == run.steps[0].members[0].output_bindings["bot"]
    assert exec_b.input_bindings["bot"] == run.steps[0].members[1].output_bindings["bot"]


def test_pattern_optional_inputs_do_not_use_workspace_global_latest(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, OPTIONAL_LANE_TEMPLATE_YAML)
    external_bot = project_root / "external_bot"
    external_bot.mkdir()
    (external_bot / "bot.py").write_text("GLOBAL")
    workspace.register_artifact("bot", external_bot)
    pattern = parse_pattern_declaration({
        "name": "optional_inputs",
        "lanes": ["A"],
        "steps": [
            {
                "name": "optional",
                "cohort": {
                    "foreach": "lanes",
                    "block": "optional",
                },
            }
        ],
    })

    def fake_container(config, args=None):
        del args
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        assert "/input/bot" not in mounts
        note = mounts["/output/note"] / "note.txt"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text("ok")
        (mounts["/flywheel"] / "termination").write_text("normal")
        return ContainerResult(exit_code=0, elapsed_s=0.1)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        result = run_pattern(workspace, pattern, template, project_root)

    reloaded = Workspace.load(workspace.path)
    member = reloaded.runs[result.run_id].steps[0].members[0]
    execution = reloaded.executions[member.execution_id]
    assert "bot" not in execution.input_bindings


def test_pattern_fixtures_ignore_workspace_global_latest(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, LANE_TEMPLATE_YAML)
    external_bot = project_root / "external_bot"
    external_bot.mkdir()
    (external_bot / "bot.py").write_text("GLOBAL")
    global_instance = workspace.register_artifact("bot", external_bot)
    fixture_bot = project_root / "foundry" / "templates" / "assets" / "bot"
    fixture_bot.mkdir(parents=True)
    (fixture_bot / "bot.py").write_text("FIXTURE")
    pattern = parse_pattern_declaration({
        "name": "fixture_inputs",
        "lanes": ["A"],
        "fixtures": {"bot": "foundry/templates/assets/bot"},
        "steps": [
            {
                "name": "improve",
                "cohort": {
                    "foreach": "lanes",
                    "block": "improve",
                },
            }
        ],
    })

    def fake_container(config, args=None):
        del args
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        assert (mounts["/input/bot"] / "bot.py").read_text() == "FIXTURE"
        output = mounts["/output/bot"] / "bot.py"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("NEXT")
        (mounts["/flywheel"] / "termination").write_text("normal")
        return ContainerResult(exit_code=0, elapsed_s=0.1)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        result = run_pattern(workspace, pattern, template, project_root)

    reloaded = Workspace.load(workspace.path)
    run = reloaded.runs[result.run_id]
    execution = reloaded.executions[run.steps[0].members[0].execution_id]
    assert execution.input_bindings["bot"] == run.fixtures[0].artifact_id
    assert execution.input_bindings["bot"] != global_instance.id
    assert reloaded.artifacts[run.fixtures[0].artifact_id].fixture_id == (
        run.fixtures[0].id)


def test_invocation_child_output_feeds_next_step_in_lane(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(
        tmp_path, INVOCATION_OUTPUT_TEMPLATE_YAML)
    fixture_bot = project_root / "foundry" / "templates" / "assets" / "bot"
    fixture_bot.mkdir(parents=True)
    (fixture_bot / "bot.py").write_text("BASE")
    pattern = parse_pattern_declaration({
        "name": "invocation_lane",
        "lanes": ["A"],
        "fixtures": {"bot": "foundry/templates/assets/bot"},
        "steps": [
            {
                "name": "agent",
                "cohort": {
                    "foreach": "lanes",
                    "block": "agent",
                },
            },
            {
                "name": "consume",
                "cohort": {
                    "foreach": "lanes",
                    "block": "consume_score",
                },
            },
        ],
    })
    consumed_scores: list[str] = []

    def fake_container(config, args=None):
        del args
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        if config.image == "agent:latest":
            assert (mounts["/input/bot"] / "bot.py").read_text() == "BASE"
            bot = mounts["/output/bot"] / "bot.py"
            bot.parent.mkdir(parents=True, exist_ok=True)
            bot.write_text("AGENT")
            (mounts["/flywheel"] / "termination").write_text(
                "eval_requested")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        if config.image == "eval:latest":
            assert (mounts["/input/bot"] / "bot.py").read_text() == "AGENT"
            score = mounts["/output/score"] / "score.json"
            score.parent.mkdir(parents=True, exist_ok=True)
            score.write_text("SCORE")
            (mounts["/flywheel"] / "termination").write_text("normal")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        if config.image == "consume-score:latest":
            consumed_scores.append(
                (mounts["/input/score"] / "score.json").read_text())
            bot = mounts["/output/bot"] / "bot.py"
            bot.parent.mkdir(parents=True, exist_ok=True)
            bot.write_text("NEXT")
            (mounts["/flywheel"] / "termination").write_text("normal")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        raise AssertionError(config.image)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        result = run_pattern(workspace, pattern, template, project_root)

    reloaded = Workspace.load(workspace.path)
    run = reloaded.runs[result.run_id]
    agent_member = run.steps[0].members[0]
    consume_member = run.steps[1].members[0]
    invocation = reloaded.invocations[agent_member.invocation_ids[0]]
    child_execution = reloaded.executions[invocation.invoked_execution_id]
    consume_execution = reloaded.executions[consume_member.execution_id]

    assert consumed_scores == ["SCORE"]
    assert consume_execution.input_bindings["score"] == (
        child_execution.output_bindings["score"])


def test_run_until_executes_lane_major_and_stops_on_normal(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, RUN_UNTIL_TEMPLATE_YAML)
    fixture_bot = project_root / "foundry" / "templates" / "assets" / "bot"
    fixture_bot.mkdir(parents=True)
    (fixture_bot / "bot.py").write_text("BASE")
    pattern = parse_pattern_declaration({
        "name": "improve",
        "params": {
            "model": {"type": "string", "default": "sonnet"},
            "max_evals": {"type": "int", "default": 3},
        },
        "fixtures": {"bot": "foundry/templates/assets/bot"},
        "do": [
            {
                "foreach": {"count": 2},
                "do": [
                    {
                        "use": "lane_loop",
                        "with": {
                            "model": "${params.model}",
                            "max_evals": "${params.max_evals}",
                        },
                    }
                ],
            }
        ],
        "patterns": {
            "lane_loop": {
                "params": {
                    "model": {"type": "string"},
                    "max_evals": {"type": "int"},
                },
                "do": [
                    {
                        "run_until": {
                            "name": "improve",
                            "block": "ImproveBot",
                            "env": {"MODEL": "${params.model}"},
                            "continue_on": {
                                "eval_requested": {
                                    "max": "${params.max_evals}",
                                },
                            },
                            "stop_on": ["normal"],
                        }
                    }
                ],
            }
        },
    })
    seen_improve_inputs: list[str] = []
    seen_score_inputs: list[bool] = []

    def fake_container(config, args=None):
        del args
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        if config.image == "improve:latest":
            source = (mounts["/input/bot"] / "bot.py").read_text()
            seen_improve_inputs.append(source)
            seen_score_inputs.append("/input/score" in mounts)
            state_file = mounts["/flywheel"] / "state" / "session.txt"
            if "->eval2" in source:
                reason = "normal"
                next_bot = f"{source}->done"
            elif "->eval1" in source:
                reason = "eval_requested"
                next_bot = f"{source}->eval2"
                assert state_file.read_text() in {"BASE", "BASE->eval1"}
            else:
                reason = "eval_requested"
                next_bot = f"{source}->eval1"
            state_file.write_text(next_bot)
            output = mounts["/output/bot"] / "bot.py"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(next_bot)
            (mounts["/flywheel"] / "termination").write_text(reason)
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        if config.image == "eval:latest":
            bot_text = (mounts["/input/bot"] / "bot.py").read_text()
            score = mounts["/output/score"] / "score.json"
            score.parent.mkdir(parents=True, exist_ok=True)
            score.write_text(f"score:{bot_text}")
            (mounts["/flywheel"] / "termination").write_text("normal")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        raise AssertionError(config.image)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        result = run_pattern(workspace, pattern, template, project_root)

    reloaded = Workspace.load(workspace.path)
    run = reloaded.runs[result.run_id]
    assert run.lanes == ["lane_0", "lane_1"]
    assert run.status == "succeeded"
    assert seen_improve_inputs == [
        "BASE",
        "BASE->eval1",
        "BASE->eval1->eval2",
        "BASE",
        "BASE->eval1",
        "BASE->eval1->eval2",
    ]
    assert seen_score_inputs == [False, True, True, False, True, True]
    assert [step.kind for step in run.steps] == ["run_until", "run_until"]
    assert [len(step.members) for step in run.steps] == [3, 3]
    assert [member.lane for step in run.steps for member in step.members] == [
        "lane_0", "lane_0", "lane_0", "lane_1", "lane_1", "lane_1",
    ]
    assert [member.name for member in run.steps[0].members] == [
        "iter_1", "iter_2", "iter_3"]
    assert run.steps[0].terminal_reason == "normal"
    assert run.steps[0].stop_kind == "stop_on"
    assert run.steps[0].reason_counts == {"eval_requested": 2}
    lane_0_snapshots = [
        reloaded.state_snapshots[
            reloaded.executions[member.execution_id]
            .state_snapshot_id
        ]
        for member in run.steps[0].members
    ]
    assert {
        snapshot.lineage_key for snapshot in lane_0_snapshots
    } == {
        pattern_state_lineage_key(result.run_id, "lane_0", "ImproveBot")
    }
    assert lane_0_snapshots[1].predecessor_snapshot_id == (
        lane_0_snapshots[0].id)
    assert lane_0_snapshots[2].predecessor_snapshot_id == (
        lane_0_snapshots[1].id)


def test_run_until_budget_stops_after_max_continue_reason(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, RUN_UNTIL_TEMPLATE_YAML)
    fixture_bot = project_root / "foundry" / "templates" / "assets" / "bot"
    fixture_bot.mkdir(parents=True)
    (fixture_bot / "bot.py").write_text("BASE")
    pattern = parse_pattern_declaration({
        "name": "budget",
        "fixtures": {"bot": "foundry/templates/assets/bot"},
        "do": [
            {
                "run_until": {
                    "name": "improve",
                    "block": "ImproveBot",
                    "continue_on": {"eval_requested": {"max": 2}},
                    "stop_on": ["normal"],
                },
            }
        ],
    })
    seen: list[str] = []

    def fake_container(config, args=None):
        del args
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        if config.image == "improve:latest":
            source = (mounts["/input/bot"] / "bot.py").read_text()
            seen.append(source)
            output = mounts["/output/bot"] / "bot.py"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(f"{source}->{len(seen)}")
            state_file = mounts["/flywheel"] / "state" / "session.txt"
            state_file.write_text(f"{source}->{len(seen)}")
            (mounts["/flywheel"] / "termination").write_text(
                "eval_requested")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        if config.image == "eval:latest":
            score = mounts["/output/score"] / "score.json"
            score.parent.mkdir(parents=True, exist_ok=True)
            score.write_text("{}")
            (mounts["/flywheel"] / "termination").write_text("normal")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        raise AssertionError(config.image)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        result = run_pattern(workspace, pattern, template, project_root)

    run = Workspace.load(workspace.path).runs[result.run_id]
    assert seen == ["BASE", "BASE->1"]
    assert len(run.steps) == 1
    assert run.steps[0].kind == "run_until"
    assert [member.name for member in run.steps[0].members] == [
        "iter_1", "iter_2"]
    assert run.steps[0].status == "succeeded"
    assert run.steps[0].terminal_reason == "eval_requested"
    assert run.steps[0].stop_kind == "budget_exhausted"
    assert run.steps[0].reason_counts == {"eval_requested": 2}


def test_run_until_after_every_runs_after_invocation_before_next_iteration(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, RUN_UNTIL_TEMPLATE_YAML)
    fixture_bot = project_root / "foundry" / "templates" / "assets" / "bot"
    fixture_bot.mkdir(parents=True)
    (fixture_bot / "bot.py").write_text("BASE")
    pattern = parse_pattern_declaration({
        "name": "periodic",
        "params": {
            "interval": {"type": "int", "default": 2},
        },
        "fixtures": {"bot": "foundry/templates/assets/bot"},
        "do": [
            {
                "run_until": {
                    "name": "improve",
                    "block": "ImproveBot",
                    "continue_on": {"eval_requested": {"max": 3}},
                    "stop_on": ["normal"],
                    "after_every": [
                        {
                            "reason": "eval_requested",
                            "count": "${params.interval}",
                            "do": [
                                {
                                    "cohort": {
                                        "members": [
                                            {
                                                "name": "brainstorm",
                                                "block": "Brainstorm",
                                            }
                                        ],
                                    }
                                }
                            ],
                        }
                    ],
                },
            }
        ],
    })
    order: list[str] = []
    brainstorm_scores: list[str] = []

    def fake_container(config, args=None):
        del args
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        if config.image == "improve:latest":
            order.append("improve")
            source = (mounts["/input/bot"] / "bot.py").read_text()
            output = mounts["/output/bot"] / "bot.py"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(f"{source}->{len(order)}")
            state_file = mounts["/flywheel"] / "state" / "session.txt"
            state_file.write_text(source)
            (mounts["/flywheel"] / "termination").write_text(
                "eval_requested")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        if config.image == "eval:latest":
            order.append("eval")
            bot_text = (mounts["/input/bot"] / "bot.py").read_text()
            score = mounts["/output/score"] / "score.json"
            score.parent.mkdir(parents=True, exist_ok=True)
            score.write_text(f"score:{bot_text}")
            (mounts["/flywheel"] / "termination").write_text("normal")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        if config.image == "brainstorm:latest":
            order.append("brainstorm")
            brainstorm_scores.append(
                (mounts["/input/score"] / "score.json").read_text())
            ideas = mounts["/output/ideas"] / "ideas.md"
            ideas.parent.mkdir(parents=True, exist_ok=True)
            ideas.write_text("try a different button")
            (mounts["/flywheel"] / "termination").write_text("normal")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        raise AssertionError(config.image)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        result = run_pattern(workspace, pattern, template, project_root)

    reloaded = Workspace.load(workspace.path)
    run = reloaded.runs[result.run_id]
    assert order == [
        "improve", "eval",
        "improve", "eval",
        "brainstorm",
        "improve", "eval",
    ]
    assert brainstorm_scores == ["score:BASE->1->3"]
    assert [step.kind for step in run.steps] == ["run_until", "cohort"]
    assert run.steps[1].name == (
        "improve__after_eval_requested_2_1__brainstorm")
    assert run.steps[1].members[0].lane == DEFAULT_LANE
    assert run.steps[0].reason_counts == {"eval_requested": 3}


def test_run_until_after_every_reason_must_be_continue_reason(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, RUN_UNTIL_TEMPLATE_YAML)
    pattern = parse_pattern_declaration({
        "name": "bad_periodic",
        "do": [
            {
                "run_until": {
                    "name": "improve",
                    "block": "ImproveBot",
                    "continue_on": {"eval_requested": {"max": 3}},
                    "stop_on": ["normal"],
                    "after_every": [
                        {
                            "reason": "normal",
                            "count": 1,
                            "do": [
                                {
                                    "cohort": {
                                        "members": [
                                            {
                                                "name": "brainstorm",
                                                "block": "Brainstorm",
                                            }
                                        ],
                                    }
                                }
                            ],
                        }
                    ],
                },
            }
        ],
    })

    with patch("flywheel.execution.run_container") as run_container:
        try:
            run_pattern(workspace, pattern, template, project_root)
        except RuntimeError as exc:
            assert "not listed under continue_on" in str(exc)
        else:
            raise AssertionError("expected after_every validation failure")

    assert run_container.call_count == 0


def test_run_until_unexpected_declared_reason_fails_run(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, RUN_UNTIL_TEMPLATE_YAML)
    fixture_bot = project_root / "foundry" / "templates" / "assets" / "bot"
    fixture_bot.mkdir(parents=True)
    (fixture_bot / "bot.py").write_text("BASE")
    pattern = parse_pattern_declaration({
        "name": "unexpected",
        "fixtures": {"bot": "foundry/templates/assets/bot"},
        "do": [
            {
                "run_until": {
                    "name": "improve",
                    "block": "ImproveBot",
                    "continue_on": {"eval_requested": {"max": 2}},
                    "stop_on": ["normal"],
                },
            }
        ],
    })

    def fake_container(config, args=None):
        del args
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        if config.image == "improve:latest":
            output = mounts["/output/bot"] / "bot.py"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("ABORT")
            state_file = mounts["/flywheel"] / "state" / "session.txt"
            state_file.write_text("ABORT")
            (mounts["/flywheel"] / "termination").write_text("aborted")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        raise AssertionError(config.image)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        try:
            run_pattern(workspace, pattern, template, project_root)
        except RuntimeError as exc:
            assert "unexpected termination reason 'aborted'" in str(exc)
        else:
            raise AssertionError("expected run failure")

    run = next(iter(Workspace.load(workspace.path).runs.values()))
    assert run.status == "failed"
    assert run.error == (
        "run_until 'improve' received unexpected termination reason 'aborted'"
    )
    assert len(run.steps) == 1
    step = run.steps[0]
    assert step.kind == "run_until"
    assert step.status == "failed"
    assert step.terminal_reason == "aborted"
    assert step.stop_kind == "unexpected_reason"
    assert step.members[0].status == "succeeded"


def test_run_until_fail_on_reason_fails_run_legibly(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, RUN_UNTIL_TEMPLATE_YAML)
    fixture_bot = project_root / "foundry" / "templates" / "assets" / "bot"
    fixture_bot.mkdir(parents=True)
    (fixture_bot / "bot.py").write_text("BASE")
    pattern = parse_pattern_declaration({
        "name": "fail_on",
        "fixtures": {"bot": "foundry/templates/assets/bot"},
        "do": [
            {
                "run_until": {
                    "name": "improve",
                    "block": "ImproveBot",
                    "continue_on": {"eval_requested": {"max": 2}},
                    "stop_on": ["normal"],
                    "fail_on": ["aborted"],
                },
            }
        ],
    })

    def fake_container(config, args=None):
        del args
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        if config.image == "improve:latest":
            output = mounts["/output/bot"] / "bot.py"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("ABORT")
            state_file = mounts["/flywheel"] / "state" / "session.txt"
            state_file.write_text("ABORT")
            (mounts["/flywheel"] / "termination").write_text("aborted")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        raise AssertionError(config.image)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        try:
            run_pattern(workspace, pattern, template, project_root)
        except RuntimeError as exc:
            assert "failed on termination reason 'aborted'" in str(exc)
        else:
            raise AssertionError("expected run failure")

    run = next(iter(Workspace.load(workspace.path).runs.values()))
    assert run.status == "failed"
    assert run.error == (
        "run_until 'improve' failed on termination reason 'aborted'"
    )
    step = run.steps[0]
    assert step.kind == "run_until"
    assert step.status == "failed"
    assert step.terminal_reason == "aborted"
    assert step.stop_kind == "fail_on"
    assert step.members[0].status == "succeeded"


def test_run_until_member_failure_fails_loop_without_more_iterations(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, RUN_UNTIL_TEMPLATE_YAML)
    fixture_bot = project_root / "foundry" / "templates" / "assets" / "bot"
    fixture_bot.mkdir(parents=True)
    (fixture_bot / "bot.py").write_text("BASE")
    pattern = parse_pattern_declaration({
        "name": "failure",
        "fixtures": {"bot": "foundry/templates/assets/bot"},
        "do": [
            {
                "run_until": {
                    "name": "improve",
                    "block": "ImproveBot",
                    "continue_on": {"eval_requested": {"max": 3}},
                    "stop_on": ["normal"],
                },
            }
        ],
    })
    calls = 0

    def fake_container(config, args=None):
        nonlocal calls
        del args
        calls += 1
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        if config.image == "improve:latest" and calls == 1:
            output = mounts["/output/bot"] / "bot.py"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("NEXT")
            state_file = mounts["/flywheel"] / "state" / "session.txt"
            state_file.write_text("NEXT")
            (mounts["/flywheel"] / "termination").write_text(
                "eval_requested")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        if config.image == "eval:latest":
            score = mounts["/output/score"] / "score.json"
            score.parent.mkdir(parents=True, exist_ok=True)
            score.write_text("{}")
            (mounts["/flywheel"] / "termination").write_text("normal")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        if config.image == "improve:latest":
            return ContainerResult(exit_code=1, elapsed_s=0.1)
        raise AssertionError(config.image)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        try:
            run_pattern(workspace, pattern, template, project_root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected run failure")

    reloaded = Workspace.load(workspace.path)
    run = next(iter(reloaded.runs.values()))
    assert run.status == "failed"
    assert len(run.steps) == 1
    step = run.steps[0]
    assert step.kind == "run_until"
    assert step.status == "failed"
    assert step.stop_kind == "failed"
    assert [member.status for member in step.members] == [
        "succeeded", "failed"]
    assert len(reloaded.invocations) == 1


def test_pattern_member_stays_parent_when_invoked_child_fails(
    tmp_path: Path,
):
    project_root, template, workspace = _setup_workspace(
        tmp_path, INVOCATION_FAILURE_TEMPLATE_YAML)
    pattern = parse_pattern_declaration({
        "name": "improve",
        "steps": [
            {
                "name": "improve",
                "cohort": {
                    "min_successes": "all",
                    "members": [
                        {"name": "agent", "block": "agent"},
                    ],
                },
            }
        ],
    })

    def fake_container(config, args=None):
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        if config.image == "agent:latest":
            bot_dir = mounts["/output/bot"]
            bot_dir.mkdir(parents=True, exist_ok=True)
            (bot_dir / "bot.py").write_text("def player_fn(*_): return 0")
            (mounts["/flywheel"] / "termination").write_text(
                "eval_requested")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        if config.image == "eval-bad:latest":
            assert (mounts["/input/bot"] / "bot.py").exists()
            (mounts["/flywheel"] / "termination").write_text("normal")
            return ContainerResult(exit_code=0, elapsed_s=0.1)
        raise AssertionError(config.image)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        result = run_pattern(workspace, pattern, template, project_root)

    reloaded = Workspace.load(workspace.path)
    run = reloaded.runs[result.run_id]
    assert run.status == "succeeded"
    member = run.steps[0].members[0]
    assert member.status == "succeeded"
    parent_execution = reloaded.executions[member.execution_id]
    assert parent_execution.block_name == "agent"
    invocation = next(iter(reloaded.invocations.values()))
    assert invocation.status == "failed"
    assert invocation.invoked_execution_id in reloaded.executions
    assert reloaded.executions[invocation.invoked_execution_id].block_name == (
        "eval_bad")


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


def test_pattern_fixture_uses_artifact_validator(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(
        tmp_path, LANE_TEMPLATE_YAML)
    fixture_bot = project_root / "foundry" / "templates" / "assets" / "bot"
    fixture_bot.mkdir(parents=True)
    (fixture_bot / "bot.py").write_text("BASE")
    pattern = parse_pattern_declaration({
        "name": "fixture_validation",
        "lanes": ["A"],
        "fixtures": {"bot": "foundry/templates/assets/bot"},
        "steps": [
            {
                "name": "improve",
                "cohort": {
                    "foreach": "lanes",
                    "block": "improve",
                },
            }
        ],
    })
    seen: list[str] = []
    seen_payloads: list[str] = []
    registry = ArtifactValidatorRegistry()

    def validate_bot(name, declaration, staged_path):
        del declaration
        seen.append(name)
        seen_payloads.append((staged_path / "bot.py").read_text())

    registry.register("bot", validate_bot)

    def fake_container(config, args=None):
        del args
        mounts = {
            container_path: Path(host)
            for host, container_path, _mode in config.mounts
        }
        output = mounts["/output/bot"] / "bot.py"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("NEXT")
        (mounts["/flywheel"] / "termination").write_text("normal")
        return ContainerResult(exit_code=0, elapsed_s=0.1)

    with patch("flywheel.execution.run_container", side_effect=fake_container):
        run_pattern(
            workspace,
            pattern,
            template,
            project_root,
            validator_registry=registry,
        )

    assert seen == ["bot", "bot"]
    assert seen_payloads == ["BASE", "NEXT"]


def test_missing_pattern_fixture_fails_run_before_first_step(tmp_path: Path):
    project_root, template, workspace = _setup_workspace(
        tmp_path, LANE_TEMPLATE_YAML)
    pattern = parse_pattern_declaration({
        "name": "missing_fixture",
        "lanes": ["A"],
        "fixtures": {"bot": "foundry/templates/assets/missing"},
        "steps": [
            {
                "name": "improve",
                "cohort": {
                    "foreach": "lanes",
                    "block": "improve",
                },
            }
        ],
    })

    with patch("flywheel.execution.run_container") as run_container:
        try:
            run_pattern(workspace, pattern, template, project_root)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected fixture materialization failure")

    assert run_container.call_count == 0
    run = next(iter(Workspace.load(workspace.path).runs.values()))
    assert run.status == "failed"
    assert run.steps == []
    assert run.fixtures == []


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
        result.run_id, DEFAULT_LANE, "train")
    assert (
        reloaded.state_snapshot_path(snapshot.id) / "marker.txt"
    ).read_text() == "state"


def test_pattern_state_lineage_key_is_injective_for_underscores():
    run_id = "run_collision"

    assert pattern_state_lineage_key(
        run_id, "x_train", "dueling") != pattern_state_lineage_key(
            run_id, "x", "train_dueling")
    assert pattern_state_lineage_key(
        run_id, "x", "train") != pattern_state_lineage_key(
            run_id, "x", "train_dueling")


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
