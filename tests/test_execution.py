from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from flywheel import runtime
from flywheel.artifact_validator import (
    ArtifactValidationError,
    ArtifactValidatorRegistry,
)
from flywheel.container import ContainerResult
from flywheel.execution import RuntimeResult, run_block
from flywheel.state_validator import (
    StateValidationError,
    StateValidatorRegistry,
)
from flywheel.template import InvocationBinding, Template
from flywheel.workspace import Workspace
from tests._inline_blocks import from_yaml_with_inline_blocks
from tests.conftest import _init_git_repo

TEMPLATE_YAML = """\
artifacts:
  - name: engine
    kind: git
    repo: "."
    path: src
  - name: checkpoint
    kind: copy
  - name: score
    kind: copy

blocks:
  - name: train
    image: train:latest
    docker_args: ["--gpus", "all", "--shm-size", "8g"]
    inputs:
      - name: engine
        container_path: /input/engine
      - name: checkpoint
        container_path: /input/checkpoint
        optional: true
    outputs:
      - name: checkpoint
        container_path: /output
  - name: eval
    image: eval:latest
    inputs:
      - name: checkpoint
        container_path: /input/checkpoint
    outputs:
      - name: score
        container_path: /output
"""

STATE_TEMPLATE_YAML = """\
artifacts:
  - name: checkpoint
    kind: copy

blocks:
  - name: stateful
    image: stateful:latest
    state: managed
    inputs: []
    outputs:
      normal:
        - name: checkpoint
          container_path: /output
"""

INVOCATION_TEMPLATE_YAML = """\
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
          - block: eval
            bind:
              bot: bot
            args: ["--episodes", "2"]
  - name: eval
    image: eval:latest
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      - name: score
        container_path: /output/score
"""

MULTI_INVOCATION_TEMPLATE_YAML = """\
artifacts:
  - name: bot
    kind: copy
  - name: score
    kind: copy
  - name: audit
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
          - block: audit
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
  - name: audit
    image: audit:latest
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      - name: audit
        container_path: /output/audit
"""

IMPLICIT_INVOCATION_TEMPLATE_YAML = """\
artifacts:
  - name: bot
    kind: copy
  - name: score
    kind: copy
  - name: config
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
          - block: eval
            bind:
              bot: bot
  - name: eval
    image: eval:latest
    inputs:
      - name: bot
        container_path: /input/bot
      - name: config
        container_path: /input/config
    outputs:
      - name: score
        container_path: /output/score
"""


def _setup_git_project(tmp_path: Path) -> tuple[Path, Path, Template]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _init_git_repo(project_root)
    (project_root / "src").mkdir()
    (project_root / "src" / "main.rs").write_text("// engine")
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", "add src"],
        check=True, capture_output=True,
    )
    foundry_dir = project_root / "foundry"
    foundry_dir.mkdir()
    templates_dir = foundry_dir / "templates" / "workspaces"
    templates_dir.mkdir(parents=True)
    template_path = templates_dir / "test_template.yaml"
    template_path.write_text(TEMPLATE_YAML)
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", "add foundry"],
        check=True, capture_output=True,
    )
    template = from_yaml_with_inline_blocks(template_path)
    return project_root, foundry_dir, template


def _setup_state_project(
    tmp_path: Path,
    template_yaml: str = STATE_TEMPLATE_YAML,
) -> tuple[Path, Path, Path, Template]:
    project_root = tmp_path / "state_project"
    project_root.mkdir()
    foundry_dir = project_root / "foundry"
    foundry_dir.mkdir()
    templates_dir = foundry_dir / "templates" / "workspaces"
    templates_dir.mkdir(parents=True)
    template_path = templates_dir / "state_template.yaml"
    template_path.write_text(template_yaml)
    template = from_yaml_with_inline_blocks(template_path)
    return project_root, foundry_dir, template_path, template


def _commit_all(project_root: Path, message: str = "auto") -> None:
    subprocess.run(
        ["git", "-C", str(project_root), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "commit", "-m", message],
        check=True, capture_output=True,
    )


def _fake_run_with_output(
    output_files: dict[str, str],
    termination_reason: str | None = "normal",
):
    """Build a ``run_container`` stub that mimics a happy-path block.

    Writes ``output_files`` into every per-slot proposal mount
    (matching the proposal-then-forge contract) and announces
    ``termination_reason`` via the ``/flywheel/termination``
    sidecar.  Pass ``termination_reason=None`` to simulate a
    block that exits cleanly without announcing — exercises the
    substrate's ``protocol_violation`` path.
    """
    def fake_run(config, args=None):
        for host, container, mode in config.mounts:
            if mode != "rw":
                continue
            if container == "/flywheel":
                if termination_reason is not None:
                    (Path(host) / "termination").write_text(
                        termination_reason,
                    )
                continue
            for name, content in output_files.items():
                file_path = Path(host) / name
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
        return ContainerResult(exit_code=0, elapsed_s=1.0)
    return fake_run


class TestBlockLookup:
    def test_nonexistent_block_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        with pytest.raises(KeyError, match="nonexistent"):
            run_block(ws, "nonexistent", template, project_root)

    def test_valid_block_found(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            result = run_block(ws, "train", template, project_root)
        assert result.container_result.exit_code == 0
        assert result.execution_id in ws.executions
        assert result.execution == ws.executions[result.execution_id]


class TestBlockInvocation:
    def test_termination_route_invokes_child_block(self, tmp_path: Path):
        project_root, foundry_dir, _template_path, template = (
            _setup_state_project(tmp_path, INVOCATION_TEMPLATE_YAML)
        )
        ws = Workspace.create("test_ws", template, foundry_dir)
        seen: dict[str, object] = {}

        def fake_run(config, args=None):
            mounts = {
                container: Path(host)
                for host, container, _mode in config.mounts
            }
            if config.image == "agent:latest":
                seen["parent_mounts"] = dict(mounts)
                bot_dir = mounts["/output/bot"]
                bot_dir.mkdir(parents=True, exist_ok=True)
                (bot_dir / "bot.py").write_text(
                    "def player_fn(env, obs_json, action_labels):\n"
                    "    return 0\n",
                    encoding="utf-8",
                )
                (mounts["/flywheel"] / "termination").write_text(
                    "eval_requested")
                return ContainerResult(exit_code=0, elapsed_s=3.0)

            if config.image == "eval:latest":
                seen["child_args"] = args
                bot_path = mounts["/input/bot"] / "bot.py"
                assert bot_path.read_text(encoding="utf-8").startswith(
                    "def player_fn")
                score_dir = mounts["/output/score"]
                score_dir.mkdir(parents=True, exist_ok=True)
                (score_dir / "scores.json").write_text(json.dumps({
                    "mean": 2,
                    "episodes": 2,
                }))
                (mounts["/flywheel"] / "termination").write_text("normal")
                return ContainerResult(exit_code=0, elapsed_s=1.0)

            raise AssertionError(config.image)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            result = run_block(ws, "agent", template, project_root)

        assert result.execution.status == "succeeded"
        assert result.execution.termination_reason == "eval_requested"
        assert "bot" in result.execution.output_bindings
        assert len(ws.invocations) == 1
        invocation = next(iter(ws.invocations.values()))
        assert invocation.invoking_execution_id == result.execution_id
        assert invocation.termination_reason == "eval_requested"
        assert invocation.status == "succeeded"
        assert invocation.invoked_execution_id in ws.executions
        child = ws.executions[invocation.invoked_execution_id]
        assert child.block_name == "eval"
        assert child.invoking_execution_id == result.execution_id
        assert child.input_bindings["bot"] == (
            result.execution.output_bindings["bot"])
        assert seen["child_args"] == ["--episodes", "2"]
        assert ws.artifacts[invocation.input_bindings["bot"]].name == "bot"
        assert "/scratch" not in seen["parent_mounts"]

        reloaded = Workspace.load(ws.path)
        loaded_invocation = reloaded.invocations[invocation.id]
        assert loaded_invocation.invoked_execution_id == (
            invocation.invoked_execution_id)
        assert loaded_invocation.input_bindings == invocation.input_bindings
        loaded_child = reloaded.executions[invocation.invoked_execution_id]
        assert loaded_child.invoking_execution_id == result.execution_id

    def test_ad_hoc_invocation_route_substitutes_params(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _template_path, template = (
            _setup_state_project(tmp_path, INVOCATION_TEMPLATE_YAML)
        )
        ws = Workspace.create("test_ws", template, foundry_dir)
        agent = next(block for block in template.blocks if block.name == "agent")
        eval_block = next(block for block in template.blocks if block.name == "eval")
        route = agent.on_termination["eval_requested"][0]
        agent = replace(
            agent,
            on_termination={
                "eval_requested": [
                    replace(
                        route,
                        args=[
                            "--episodes",
                            "${params.eval_episodes}",
                            "--dry-run",
                            "${params.dry_run}",
                        ],
                    )
                ]
            },
        )
        template = Template(
            name=template.name,
            artifacts=template.artifacts,
            blocks=[agent, eval_block],
        )
        seen_args: list[str] | None = None

        def fake_run(config, args=None):
            nonlocal seen_args
            mounts = {
                container: Path(host)
                for host, container, _mode in config.mounts
            }
            if config.image == "agent:latest":
                bot_dir = mounts["/output/bot"]
                bot_dir.mkdir(parents=True, exist_ok=True)
                (bot_dir / "bot.py").write_text("def player_fn(*_): return 0")
                (mounts["/flywheel"] / "termination").write_text(
                    "eval_requested")
                return ContainerResult(exit_code=0, elapsed_s=1.0)
            if config.image == "eval:latest":
                seen_args = list(args or [])
                score_dir = mounts["/output/score"]
                score_dir.mkdir(parents=True, exist_ok=True)
                (score_dir / "scores.json").write_text("{}")
                (mounts["/flywheel"] / "termination").write_text("normal")
                return ContainerResult(exit_code=0, elapsed_s=1.0)
            raise AssertionError(config.image)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(
                ws,
                "agent",
                template,
                project_root,
                invocation_params={
                    "eval_episodes": 4000,
                    "dry_run": True,
                },
            )

        assert seen_args == ["--episodes", "4000", "--dry-run", "true"]

    def test_missing_ad_hoc_invocation_param_records_failed_invocation(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _template_path, template = (
            _setup_state_project(tmp_path, INVOCATION_TEMPLATE_YAML)
        )
        ws = Workspace.create("test_ws", template, foundry_dir)
        agent = next(block for block in template.blocks if block.name == "agent")
        eval_block = next(block for block in template.blocks if block.name == "eval")
        route = agent.on_termination["eval_requested"][0]
        agent = replace(
            agent,
            on_termination={
                "eval_requested": [
                    replace(route, args=["${params.eval_episodes}"])
                ]
            },
        )
        template = Template(
            name=template.name,
            artifacts=template.artifacts,
            blocks=[agent, eval_block],
        )

        def fake_run(config, args=None):
            del args
            mounts = {
                container: Path(host)
                for host, container, _mode in config.mounts
            }
            assert config.image == "agent:latest"
            bot_dir = mounts["/output/bot"]
            bot_dir.mkdir(parents=True, exist_ok=True)
            (bot_dir / "bot.py").write_text("def player_fn(*_): return 0")
            (mounts["/flywheel"] / "termination").write_text(
                "eval_requested")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            result = run_block(ws, "agent", template, project_root)

        assert result.execution.status == "succeeded"
        invocation = next(iter(ws.invocations.values()))
        assert invocation.status == "failed"
        assert invocation.invoked_execution_id is None
        assert "unknown pattern param 'eval_episodes'" in invocation.error

    def test_no_route_for_termination_reason_runs_only_parent(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _template_path, template = (
            _setup_state_project(tmp_path, INVOCATION_TEMPLATE_YAML)
        )
        ws = Workspace.create("test_ws", template, foundry_dir)

        def fake_run(config, args=None):
            mounts = {
                container: Path(host)
                for host, container, _mode in config.mounts
            }
            assert config.image == "agent:latest"
            bot_dir = mounts["/output/bot"]
            bot_dir.mkdir(parents=True, exist_ok=True)
            (bot_dir / "bot.py").write_text("def player_fn(*_): return 0")
            (mounts["/flywheel"] / "termination").write_text(
                "eval_requested")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        block = next(block for block in template.blocks if block.name == "agent")
        template = Template(
            name=template.name,
            artifacts=template.artifacts,
            blocks=[
                block.__class__(
                    **{
                        **block.__dict__,
                        "on_termination": {},
                    },
                ),
                next(block for block in template.blocks if block.name == "eval"),
            ],
        )

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "agent", template, project_root)

        assert len(ws.executions) == 1
        assert ws.invocations == {}

    def test_child_failure_records_invocation_and_continues_routes(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _template_path, template = (
            _setup_state_project(tmp_path, MULTI_INVOCATION_TEMPLATE_YAML)
        )
        ws = Workspace.create("test_ws", template, foundry_dir)

        def fake_run(config, args=None):
            mounts = {
                container: Path(host)
                for host, container, _mode in config.mounts
            }
            if config.image == "agent:latest":
                bot_dir = mounts["/output/bot"]
                bot_dir.mkdir(parents=True, exist_ok=True)
                (bot_dir / "bot.py").write_text("def player_fn(*_): return 0")
                (mounts["/flywheel"] / "termination").write_text(
                    "eval_requested")
                return ContainerResult(exit_code=0, elapsed_s=1.0)
            if config.image == "eval-bad:latest":
                assert (mounts["/input/bot"] / "bot.py").exists()
                (mounts["/flywheel"] / "termination").write_text(
                    "normal")
                return ContainerResult(exit_code=0, elapsed_s=1.0)
            if config.image == "audit:latest":
                assert (mounts["/input/bot"] / "bot.py").exists()
                audit_dir = mounts["/output/audit"]
                audit_dir.mkdir(parents=True, exist_ok=True)
                (audit_dir / "audit.txt").write_text("checked")
                (mounts["/flywheel"] / "termination").write_text(
                    "normal")
                return ContainerResult(exit_code=0, elapsed_s=1.0)
            raise AssertionError(config.image)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            result = run_block(ws, "agent", template, project_root)

        assert result.execution.status == "succeeded"
        assert result.execution.block_name == "agent"
        invocations = sorted(
            ws.invocations.values(), key=lambda inv: inv.invoked_block_name)
        assert [inv.invoked_block_name for inv in invocations] == [
            "audit", "eval_bad"]
        audit_inv = invocations[0]
        eval_inv = invocations[1]
        assert audit_inv.status == "succeeded"
        assert ws.executions[audit_inv.invoked_execution_id].status == (
            "succeeded")
        assert eval_inv.status == "failed"
        assert eval_inv.error is not None
        assert "no output bytes written" in eval_inv.error
        assert eval_inv.invoked_execution_id in ws.executions
        assert ws.executions[eval_inv.invoked_execution_id].status == "failed"

        reloaded = Workspace.load(ws.path)
        loaded_eval_inv = reloaded.invocations[eval_inv.id]
        assert loaded_eval_inv.status == "failed"
        assert loaded_eval_inv.error == eval_inv.error

    def test_invocation_supports_implicit_child_input_resolution(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _template_path, template = (
            _setup_state_project(tmp_path, IMPLICIT_INVOCATION_TEMPLATE_YAML)
        )
        ws = Workspace.create("test_ws", template, foundry_dir)
        config_source = tmp_path / "config"
        config_source.mkdir()
        (config_source / "settings.json").write_text("{}")
        config_artifact = ws.register_artifact("config", config_source)

        def fake_run(config, args=None):
            mounts = {
                container: Path(host)
                for host, container, _mode in config.mounts
            }
            if config.image == "agent:latest":
                bot_dir = mounts["/output/bot"]
                bot_dir.mkdir(parents=True, exist_ok=True)
                (bot_dir / "bot.py").write_text("def player_fn(*_): return 0")
                (mounts["/flywheel"] / "termination").write_text(
                    "eval_requested")
                return ContainerResult(exit_code=0, elapsed_s=1.0)
            if config.image == "eval:latest":
                assert (mounts["/input/bot"] / "bot.py").exists()
                assert (mounts["/input/config"] / "settings.json").exists()
                score_dir = mounts["/output/score"]
                score_dir.mkdir(parents=True, exist_ok=True)
                (score_dir / "scores.json").write_text('{"mean": 1}')
                (mounts["/flywheel"] / "termination").write_text("normal")
                return ContainerResult(exit_code=0, elapsed_s=1.0)
            raise AssertionError(config.image)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "agent", template, project_root)

        invocation = next(iter(ws.invocations.values()))
        assert invocation.input_bindings["config"] == config_artifact.id

    def test_invocation_supports_concrete_artifact_id_binding(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _template_path, template = (
            _setup_state_project(tmp_path, INVOCATION_TEMPLATE_YAML)
        )
        ws = Workspace.create("test_ws", template, foundry_dir)
        bot_source = tmp_path / "bot"
        bot_source.mkdir()
        (bot_source / "bot.py").write_text("def player_fn(*_): return 1")
        seed_bot = ws.register_artifact("bot", bot_source)

        agent = next(block for block in template.blocks if block.name == "agent")
        eval_block = next(block for block in template.blocks if block.name == "eval")
        route = agent.on_termination["eval_requested"][0]
        agent = replace(
            agent,
            outputs={"eval_requested": []},
            on_termination={
                "eval_requested": [
                    replace(
                        route,
                        bind={
                            "bot": InvocationBinding(
                                artifact_id=seed_bot.id)
                        },
                    )
                ]
            },
        )
        template = Template(
            name=template.name,
            artifacts=template.artifacts,
            blocks=[agent, eval_block],
        )

        def fake_run(config, args=None):
            mounts = {
                container: Path(host)
                for host, container, _mode in config.mounts
            }
            if config.image == "agent:latest":
                (mounts["/flywheel"] / "termination").write_text(
                    "eval_requested")
                return ContainerResult(exit_code=0, elapsed_s=1.0)
            if config.image == "eval:latest":
                assert (mounts["/input/bot"] / "bot.py").read_text() == (
                    "def player_fn(*_): return 1")
                score_dir = mounts["/output/score"]
                score_dir.mkdir(parents=True, exist_ok=True)
                (score_dir / "scores.json").write_text('{"mean": 1}')
                (mounts["/flywheel"] / "termination").write_text("normal")
                return ContainerResult(exit_code=0, elapsed_s=1.0)
            raise AssertionError(config.image)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "agent", template, project_root)

        invocation = next(iter(ws.invocations.values()))
        assert invocation.input_bindings["bot"] == seed_bot.id


class TestInputResolution:
    def test_git_input_creates_new_instance(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        engine_instances = ws.instances_for("engine")
        assert len(engine_instances) == 2
        assert engine_instances[0].id == "engine@baseline"
        assert engine_instances[1].id.startswith("engine@")
        assert engine_instances[1].id != "engine@baseline"

    def test_git_baseline_preserved(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        baseline_commit = ws.artifacts["engine@baseline"].commit
        _commit_all(project_root, "add workspace")

        (project_root / "src" / "new.rs").write_text("// new")
        _commit_all(project_root, "new code")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        assert ws.artifacts["engine@baseline"].commit == baseline_commit
        new_engine = [
            a for a in ws.instances_for("engine")
            if a.id != "engine@baseline"
        ][0]
        assert new_engine.commit != baseline_commit

    def test_optional_input_skipped_when_missing(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        captured_config = {}

        def fake_run(config, args=None):
            captured_config["config"] = config
            for host, container, mode in config.mounts:
                if mode != "rw":
                    continue
                if container == "/flywheel":
                    (Path(host) / "termination").write_text(
                        "normal")
                    continue
                (Path(host) / "model.pt").write_text("weights")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        config = captured_config["config"]
        checkpoint_mounts = [
            (h, c, m) for h, c, m in config.mounts
            if c == "/input/checkpoint"
        ]
        assert len(checkpoint_mounts) == 0

    def test_required_input_raises_when_missing(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        with pytest.raises(ValueError, match="not available"):
            run_block(ws, "eval", template, project_root)


class TestRepeatedExecution:
    def test_second_execution_produces_new_artifact(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)
            _commit_all(project_root, "after first train")
            run_block(ws, "train", template, project_root)

        cp_instances = [
            a for a in ws.artifacts.values()
            if a.name == "checkpoint" and a.kind == "copy"
        ]
        assert len(cp_instances) == 2
        assert cp_instances[0].id != cp_instances[1].id

    def test_second_execution_uses_first_output_as_input(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)
            _commit_all(project_root, "after first train")
            run_block(ws, "train", template, project_root)

        # Find the two executions
        execs = sorted(ws.executions.values(), key=lambda e: e.started_at)
        assert len(execs) == 2
        first_output = execs[0].output_bindings["checkpoint"]
        second_input = execs[1].input_bindings["checkpoint"]
        second_output = execs[1].output_bindings["checkpoint"]
        assert second_input == first_output
        assert second_output != first_output


class TestEmptyOutput:
    def test_empty_output_recorded_as_collect_rejection(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        # Container exits cleanly and announces "normal" but
        # writes no output bytes.  Per the block-execution spec
        # this is an ``output_collect`` rejection: the block
        # promised an output that the proposal directory did not
        # receive.
        def empty_run(config, args=None):
            for host, container, mode in config.mounts:
                if mode == "rw" and container == "/flywheel":
                    (Path(host) / "termination").write_text(
                        "normal")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with (
            patch("flywheel.execution.run_container", side_effect=empty_run),
            pytest.raises(
                RuntimeError,
                match="no output bytes written",
            ),
        ):
            run_block(ws, "train", template, project_root)

        ex = next(iter(ws.executions.values()))
        assert ex.status == "failed"
        assert ex.failure_phase == "output_collect"
        assert ex.output_bindings == {}
        assert "checkpoint" in ex.rejected_outputs
        assert (
            ex.rejected_outputs["checkpoint"].phase
            == "output_collect"
        )
        assert len(ws.instances_for("checkpoint")) == 0


class TestExecutionRecords:
    def test_successful_execution_recorded(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        assert len(ws.executions) == 1
        ex = next(iter(ws.executions.values()))
        assert ex.block_name == "train"
        assert ex.status == "succeeded"
        assert "engine" in ex.input_bindings
        assert "checkpoint" in ex.output_bindings

    def test_failed_execution_recorded(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        with patch("flywheel.execution.run_container") as mock_run:
            mock_run.return_value = ContainerResult(exit_code=1, elapsed_s=5.0)
            with pytest.raises(RuntimeError, match="exited with code 1"):
                run_block(ws, "train", template, project_root)

        assert len(ws.executions) == 1
        ex = next(iter(ws.executions.values()))
        assert ex.status == "failed"

    def test_execution_persisted(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        reloaded = Workspace.load(ws.path)
        assert len(reloaded.executions) == 1
        cp_instances = reloaded.instances_for("checkpoint")
        assert len(cp_instances) == 1


class TestResourceConfig:
    def test_docker_args_passed_to_container(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        captured_config = {}

        def fake_run(config, args=None):
            captured_config["config"] = config
            for host, container, mode in config.mounts:
                if mode != "rw":
                    continue
                if container == "/flywheel":
                    (Path(host) / "termination").write_text(
                        "normal")
                    continue
                (Path(host) / "model.pt").write_text("weights")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)

        config = captured_config["config"]
        assert "--gpus" in config.docker_args
        assert "--shm-size" in config.docker_args


class TestErrorCases:
    def test_dirty_git_repo_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")
        (project_root / "src" / "dirty.rs").write_text("// dirty")

        with pytest.raises(RuntimeError, match="uncommitted changes"):
            run_block(ws, "train", template, project_root)

    def test_template_mismatch_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        other_path = foundry_dir / "templates" / "other.yaml"
        other_path.write_text(TEMPLATE_YAML)
        other_template = from_yaml_with_inline_blocks(other_path)

        with pytest.raises(ValueError, match="does not match"):
            run_block(ws, "train", other_template, project_root)

    def test_deleted_git_path_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        shutil.rmtree(project_root / "src")
        _commit_all(project_root, "remove src")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            run_block(ws, "train", template, project_root)

    def test_extra_args_passed_through(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        captured_args = {}

        def fake_run(config, args=None):
            captured_args["args"] = args
            for host, container, mode in config.mounts:
                if mode != "rw":
                    continue
                if container == "/flywheel":
                    (Path(host) / "termination").write_text(
                        "normal")
                    continue
                (Path(host) / "model.pt").write_text("weights")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(
                ws, "train", template, project_root,
                args=["--subclass", "dueling"],
            )

        assert captured_args["args"] == ["--subclass", "dueling"]


class TestInputBindings:
    def test_explicit_binding_used(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        # Run train twice to get two checkpoints
        fake_run = _fake_run_with_output({"model.pt": "weights_v1"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)
            _commit_all(project_root, "after first train")

        first_cp = ws.instances_for("checkpoint")[0].id

        fake_run2 = _fake_run_with_output({"model.pt": "weights_v2"})
        with patch("flywheel.execution.run_container", side_effect=fake_run2):
            run_block(ws, "train", template, project_root)
            _commit_all(project_root, "after second train")

        # Run eval with explicit binding to the first checkpoint
        def capturing_run(config, args=None):
            for host, container, mode in config.mounts:
                if mode != "rw":
                    continue
                if container == "/flywheel":
                    (Path(host) / "termination").write_text(
                        "normal")
                    continue
                (Path(host) / "scores.json").write_text("{}")
            return ContainerResult(exit_code=0, elapsed_s=1.0)

        with patch("flywheel.execution.run_container", side_effect=capturing_run):
            run_block(
                ws, "eval", template, project_root,
                input_bindings={"checkpoint": first_cp},
            )

        eval_exec = [
            e for e in ws.executions.values() if e.block_name == "eval"
        ][0]
        assert eval_exec.input_bindings["checkpoint"] == first_cp

    def test_invalid_binding_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        with pytest.raises(ValueError, match="not found in workspace"):
            run_block(
                ws, "eval", template, project_root,
                input_bindings={"checkpoint": "checkpoint@nonexistent"},
            )


class TestFailureCleanup:
    def test_failed_execution_cleans_output_dirs(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        with patch("flywheel.execution.run_container") as mock_run:
            mock_run.return_value = ContainerResult(exit_code=1, elapsed_s=1.0)
            with pytest.raises(RuntimeError):
                run_block(ws, "train", template, project_root)

        # No checkpoint directories should remain
        checkpoint_dirs = [
            d for d in (ws.path / "artifacts").iterdir()
            if d.name.startswith("checkpoint@")
        ]
        assert len(checkpoint_dirs) == 0

    def test_failed_execution_still_records_git_inputs(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        with patch("flywheel.execution.run_container") as mock_run:
            mock_run.return_value = ContainerResult(exit_code=1, elapsed_s=1.0)
            with pytest.raises(RuntimeError):
                run_block(ws, "train", template, project_root)

        # Git instances are inputs — committed even on failure
        engine_instances = ws.instances_for("engine")
        assert len(engine_instances) == 2
        assert engine_instances[0].id == "engine@baseline"
        assert engine_instances[1].id.startswith("engine@")

    def test_wrong_slot_binding_raises(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(ws, "train", template, project_root)
            _commit_all(project_root, "after train")

        # Try to bind engine@baseline to the checkpoint slot
        with pytest.raises(ValueError, match="belongs to"):
            run_block(
                ws, "eval", template, project_root,
                input_bindings={"checkpoint": "engine@baseline"},
            )

    def test_interrupted_execution_recorded(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        with patch("flywheel.execution.run_container") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()
            with pytest.raises(KeyboardInterrupt):
                run_block(ws, "train", template, project_root)

        assert len(ws.executions) == 1
        ex = next(iter(ws.executions.values()))
        assert ex.status == "interrupted"

        # No checkpoint directories should remain
        checkpoint_dirs = [
            d for d in (ws.path / "artifacts").iterdir()
            if d.name.startswith("checkpoint@")
        ]
        assert len(checkpoint_dirs) == 0

        # Git instances are inputs — committed before container runs
        engine_instances = ws.instances_for("engine")
        assert len(engine_instances) == 2
        engine_id = ex.input_bindings["engine"]
        assert engine_id in ws.artifacts


class TestOneShotContainerRunner:
    @staticmethod
    def _workspace_for_runner_test(tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")
        return project_root, template, ws

    @staticmethod
    def _only_execution(ws: Workspace):
        return next(iter(ws.executions.values()))

    def test_run_block_delegates_body_run_to_container_runner(
        self, tmp_path: Path,
    ):
        project_root, template, ws = self._workspace_for_runner_test(
            tmp_path)

        class FakeContainerRunner:
            def __init__(self):
                self.seen_args = None
                self.seen_plan = None

            def run(self, plan, args=None):
                self.seen_plan = plan
                self.seen_args = args
                out = plan.proposal_dirs["checkpoint"]
                (out / "model.pt").write_text("weights")
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.25),
                )

        runner = FakeContainerRunner()
        with patch(
            "flywheel.execution.run_container",
            side_effect=AssertionError("default container runner used"),
        ):
            result = run_block(
                ws, "train", template, project_root,
                args=["--example-flag", "example-value"],
                container_runner=runner,
            )

        assert result.container_result.exit_code == 0
        assert runner.seen_args == ["--example-flag", "example-value"]
        assert runner.seen_plan is not None
        assert runner.seen_plan.block_name == "train"
        ex = ws.executions[runner.seen_plan.execution_id]
        assert ex.status == "succeeded"
        assert ex.termination_reason == "normal"
        assert "checkpoint" in ex.output_bindings

    def test_runner_reported_crash_records_runtime_failure(
        self, tmp_path: Path,
    ):
        project_root, template, ws = self._workspace_for_runner_test(
            tmp_path)

        class CrashRunner:
            def run(self, plan, args=None):
                return RuntimeResult(
                    termination_reason=runtime.TERMINATION_REASON_CRASH,
                    container_result=ContainerResult(
                        exit_code=137, elapsed_s=0.5),
                    error="OOM-killed",
                )

        with pytest.raises(RuntimeError, match="OOM-killed") as exc_info:
            run_block(
                ws, "train", template, project_root,
                container_runner=CrashRunner(),
            )

        ex = self._only_execution(ws)
        assert exc_info.value.flywheel_execution_id == ex.id
        assert ex.status == "failed"
        assert ex.termination_reason == runtime.TERMINATION_REASON_CRASH
        assert ex.failure_phase == runtime.FAILURE_INVOKE
        assert ex.error == "OOM-killed"
        assert ex.exit_code == 137
        assert ex.output_bindings == {}

    def test_runner_reported_timeout_surfaces_runner_error(
        self, tmp_path: Path,
    ):
        project_root, template, ws = self._workspace_for_runner_test(
            tmp_path)

        class TimeoutRunner:
            def run(self, plan, args=None):
                return RuntimeResult(
                    termination_reason=runtime.TERMINATION_REASON_TIMEOUT,
                    container_result=ContainerResult(
                        exit_code=-1, elapsed_s=30.0),
                    error="scheduler deadline exceeded",
                )

        with pytest.raises(
            RuntimeError, match="scheduler deadline exceeded",
        ):
            run_block(
                ws, "train", template, project_root,
                container_runner=TimeoutRunner(),
            )

        ex = self._only_execution(ws)
        assert ex.status == "failed"
        assert ex.termination_reason == runtime.TERMINATION_REASON_TIMEOUT
        assert ex.failure_phase == runtime.FAILURE_INVOKE
        assert ex.error == "scheduler deadline exceeded"

    def test_runner_reported_protocol_violation_records_protocol_failure(
        self, tmp_path: Path,
    ):
        project_root, template, ws = self._workspace_for_runner_test(
            tmp_path)

        class ProtocolViolationRunner:
            def run(self, plan, args=None):
                return RuntimeResult(
                    termination_reason=(
                        runtime.TERMINATION_REASON_PROTOCOL_VIOLATION
                    ),
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.25),
                    announcement="reserved:crash",
                    error="reserved termination reason announced",
                )

        with pytest.raises(
            RuntimeError, match="reserved termination reason announced",
        ):
            run_block(
                ws, "train", template, project_root,
                container_runner=ProtocolViolationRunner(),
            )

        ex = self._only_execution(ws)
        assert ex.status == "failed"
        assert (
            ex.termination_reason
            == runtime.TERMINATION_REASON_PROTOCOL_VIOLATION
        )
        assert ex.failure_phase == runtime.FAILURE_OUTPUT_PROTOCOL
        assert ex.error == "reserved termination reason announced"
        assert ex.output_bindings == {}

    def test_runner_error_whitespace_uses_crash_fallback(
        self, tmp_path: Path,
    ):
        project_root, template, ws = self._workspace_for_runner_test(
            tmp_path)

        class WhitespaceErrorRunner:
            def run(self, plan, args=None):
                return RuntimeResult(
                    termination_reason=runtime.TERMINATION_REASON_CRASH,
                    container_result=ContainerResult(
                        exit_code=2, elapsed_s=0.1),
                    error="   ",
                )

        with pytest.raises(RuntimeError) as exc_info:
            run_block(
                ws, "train", template, project_root,
                container_runner=WhitespaceErrorRunner(),
            )

        ex = self._only_execution(ws)
        assert ex.error == "container exited with code 2"
        assert str(exc_info.value) == (
            "Block 'train' runtime failed: container exited with code 2"
        )

    def test_runner_crash_without_exit_metadata_uses_fallback(
        self, tmp_path: Path,
    ):
        project_root, template, ws = self._workspace_for_runner_test(
            tmp_path)

        class MissingExitMetadataRunner:
            def run(self, plan, args=None):
                return RuntimeResult(
                    termination_reason=runtime.TERMINATION_REASON_CRASH,
                    container_result=None,
                )

        with pytest.raises(RuntimeError) as exc_info:
            run_block(
                ws, "train", template, project_root,
                container_runner=MissingExitMetadataRunner(),
            )

        ex = self._only_execution(ws)
        assert ex.error == "container crashed without exit metadata"
        assert str(exc_info.value) == (
            "Block 'train' runtime failed: "
            "container crashed without exit metadata"
        )

    def test_runner_protocol_violation_without_error_uses_fallback(
        self, tmp_path: Path,
    ):
        project_root, template, ws = self._workspace_for_runner_test(
            tmp_path)

        class ProtocolFallbackRunner:
            def run(self, plan, args=None):
                return RuntimeResult(
                    termination_reason=(
                        runtime.TERMINATION_REASON_PROTOCOL_VIOLATION
                    ),
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                    announcement="unknown_reason",
                )

        with pytest.raises(RuntimeError) as exc_info:
            run_block(
                ws, "train", template, project_root,
                container_runner=ProtocolFallbackRunner(),
            )

        ex = self._only_execution(ws)
        assert ex.failure_phase == runtime.FAILURE_OUTPUT_PROTOCOL
        assert ex.error == (
            "protocol violation: termination announcement "
            "'unknown_reason' did not match any declared reason "
            "['normal']"
        )
        assert str(exc_info.value) == f"Block 'train' runtime failed: {ex.error}"

    def test_runner_returned_interrupted_records_interruption(
        self, tmp_path: Path,
    ):
        project_root, template, ws = self._workspace_for_runner_test(
            tmp_path)

        class InterruptedRunner:
            def run(self, plan, args=None):
                return RuntimeResult(
                    termination_reason=(
                        runtime.TERMINATION_REASON_INTERRUPTED
                    ),
                    container_result=None,
                    error="scheduler cancelled mid-step",
                )

        with pytest.raises(KeyboardInterrupt) as exc_info:
            run_block(
                ws, "train", template, project_root,
                container_runner=InterruptedRunner(),
            )

        ex = self._only_execution(ws)
        assert ex.status == "interrupted"
        assert ex.termination_reason == (
            runtime.TERMINATION_REASON_INTERRUPTED
        )
        assert ex.failure_phase == runtime.FAILURE_INVOKE
        assert ex.error == "scheduler cancelled mid-step"
        assert str(exc_info.value) == (
            "Block 'train' runtime failed: scheduler cancelled mid-step"
        )

    def test_project_defined_success_ignores_runner_error(
        self, tmp_path: Path,
    ):
        project_root, template, ws = self._workspace_for_runner_test(
            tmp_path)

        class SuccessfulRunnerWithError:
            def run(self, plan, args=None):
                out = plan.proposal_dirs["checkpoint"]
                (out / "model.pt").write_text("weights")
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                    error="diagnostic that should not enter ledger",
                )

        run_block(
            ws, "train", template, project_root,
            container_runner=SuccessfulRunnerWithError(),
        )

        ex = self._only_execution(ws)
        assert ex.status == "succeeded"
        assert ex.error is None

    def test_project_defined_commit_failure_ignores_runner_error(
        self, tmp_path: Path,
    ):
        project_root, template, ws = self._workspace_for_runner_test(
            tmp_path)

        class MissingOutputRunnerWithError:
            def run(self, plan, args=None):
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                    error="runner diagnostic should not win",
                )

        with pytest.raises(RuntimeError, match="output_collect"):
            run_block(
                ws, "train", template, project_root,
                container_runner=MissingOutputRunnerWithError(),
            )

        ex = self._only_execution(ws)
        assert ex.status == "failed"
        assert ex.failure_phase == runtime.FAILURE_OUTPUT_COLLECT
        assert ex.error == "output_collect: checkpoint: no output bytes written"


class TestManagedStateExecution:
    """Managed state restore and capture through canonical run_block."""

    def test_state_snapshot_restores_into_next_execution(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _, template = _setup_state_project(
            tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        seen_markers: list[str | None] = []

        class StatefulRunner:
            def __init__(self, next_marker: str):
                self.next_marker = next_marker

            def run(self, plan, args=None):
                assert plan.state_mount_dir is not None
                assert not any(
                    mount[1] == runtime.STATE_MOUNT_PATH
                    for mount in plan.mounts
                )
                assert any(
                    mount[1] == "/flywheel" for mount in plan.mounts
                )
                marker = plan.state_mount_dir / "marker.txt"
                seen_markers.append(
                    marker.read_text() if marker.exists() else None
                )
                marker.write_text(self.next_marker)
                (plan.proposal_dirs["checkpoint"] / "model.pt").write_text(
                    self.next_marker)
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                )

        run_block(
            ws, "stateful", template, project_root,
            state_lineage_key="lineage_a",
            container_runner=StatefulRunner("first"),
        )
        run_block(
            ws, "stateful", template, project_root,
            state_lineage_key="lineage_a",
            container_runner=StatefulRunner("second"),
        )

        assert seen_markers == [None, "first"]
        assert len(ws.state_snapshots) == 2
        latest = ws.latest_state_snapshot("lineage_a")
        assert latest is not None
        assert (ws.state_snapshot_path(latest.id) / "marker.txt").read_text() == (
            "second"
        )
        executions = list(ws.executions.values())
        assert executions[0].state_mode == "managed"
        assert executions[0].state_snapshot_id is not None
        assert executions[1].state_snapshot_id == latest.id

    def test_produced_artifact_and_state_persist_with_execution(
        self, tmp_path: Path, monkeypatch,
    ):
        project_root, foundry_dir, _, template = _setup_state_project(
            tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        original_save = ws.save

        def assert_no_dangling_produced_by():
            for artifact in ws.artifacts.values():
                if artifact.produced_by is not None:
                    assert artifact.produced_by in ws.executions
            for snapshot in ws.state_snapshots.values():
                assert snapshot.produced_by in ws.executions
            original_save()

        monkeypatch.setattr(ws, "save", assert_no_dangling_produced_by)

        class StatefulRunner:
            def run(self, plan, args=None):
                assert plan.state_mount_dir is not None
                (plan.state_mount_dir / "marker.txt").write_text("state")
                (plan.proposal_dirs["checkpoint"] / "model.pt").write_text(
                    "weights")
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                )

        run_block(
            ws, "stateful", template, project_root,
            state_lineage_key="lineage_a",
            container_runner=StatefulRunner(),
        )

        assert len(ws.state_snapshots) == 1
        assert len(ws.instances_for("checkpoint")) == 1

    def test_managed_state_requires_lineage_key(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _, template = _setup_state_project(
            tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        with pytest.raises(ValueError, match="state lineage key"):
            run_block(ws, "stateful", template, project_root)

        ex = next(iter(ws.executions.values()))
        assert ex.status == "failed"
        assert ex.failure_phase == runtime.FAILURE_STAGE_IN
        assert ex.state_mode == "managed"

    def test_reserved_failure_does_not_capture_state(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _, template = _setup_state_project(
            tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        seen_markers: list[str | None] = []

        class SuccessRunner:
            def run(self, plan, args=None):
                assert plan.state_mount_dir is not None
                (plan.state_mount_dir / "marker.txt").write_text("good")
                (plan.proposal_dirs["checkpoint"] / "model.pt").write_text(
                    "weights")
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                )

        class CrashRunner:
            def run(self, plan, args=None):
                assert plan.state_mount_dir is not None
                (plan.state_mount_dir / "marker.txt").write_text("bad")
                return RuntimeResult(
                    termination_reason=runtime.TERMINATION_REASON_CRASH,
                    container_result=ContainerResult(
                        exit_code=1, elapsed_s=0.1),
                )

        class ReadRunner:
            def run(self, plan, args=None):
                assert plan.state_mount_dir is not None
                marker = plan.state_mount_dir / "marker.txt"
                seen_markers.append(
                    marker.read_text() if marker.exists() else None
                )
                (plan.state_mount_dir / "marker.txt").write_text("after")
                (plan.proposal_dirs["checkpoint"] / "model.pt").write_text(
                    "weights")
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                )

        run_block(
            ws, "stateful", template, project_root,
            state_lineage_key="lineage_a",
            container_runner=SuccessRunner(),
        )
        with pytest.raises(RuntimeError):
            run_block(
                ws, "stateful", template, project_root,
                state_lineage_key="lineage_a",
                container_runner=CrashRunner(),
            )
        run_block(
            ws, "stateful", template, project_root,
            state_lineage_key="lineage_a",
            container_runner=ReadRunner(),
        )

        assert seen_markers == ["good"]
        assert len(ws.state_snapshots) == 2

    def test_state_capture_failure_records_without_outputs(
        self, tmp_path: Path, monkeypatch,
    ):
        project_root, foundry_dir, _, template = _setup_state_project(
            tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        class SuccessRunner:
            def run(self, plan, args=None):
                assert plan.state_mount_dir is not None
                (plan.state_mount_dir / "marker.txt").write_text("state")
                (plan.proposal_dirs["checkpoint"] / "model.pt").write_text(
                    "weights")
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                )

        def fail_capture(**kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(ws, "register_state_snapshot", fail_capture)

        with pytest.raises(RuntimeError, match="state_capture"):
            run_block(
                ws, "stateful", template, project_root,
                state_lineage_key="lineage_a",
                container_runner=SuccessRunner(),
            )

        ex = next(iter(ws.executions.values()))
        assert ex.status == "failed"
        assert ex.failure_phase == runtime.FAILURE_STATE_CAPTURE
        assert ex.output_bindings == {}
        assert ws.instances_for("checkpoint") == []
        assert len(ws.state_snapshots) == 0
        assert (ws.path / "state_recovery" / ex.id / "state").exists()

    def test_state_validator_rejection_records_recovery(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _, template = _setup_state_project(
            tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        class SuccessRunner:
            def run(self, plan, args=None):
                assert plan.state_mount_dir is not None
                (plan.state_mount_dir / "marker.txt").write_text("bad")
                (plan.proposal_dirs["checkpoint"] / "model.pt").write_text(
                    "weights")
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                )

        def reject_state(name, block_def, path, lineage_key):
            assert name == "stateful"
            assert block_def.name == "stateful"
            assert lineage_key == "lineage_a"
            assert (path / "marker.txt").read_text() == "bad"
            (path / "marker.txt").write_text("mutated by validator")
            raise StateValidationError("marker rejected")

        registry = StateValidatorRegistry({"stateful": reject_state})

        with pytest.raises(
            StateValidationError, match="marker rejected",
        ) as exc_info:
            run_block(
                ws, "stateful", template, project_root,
                state_lineage_key="lineage_a",
                state_validator_registry=registry,
                container_runner=SuccessRunner(),
            )

        ex = next(iter(ws.executions.values()))
        assert ex.status == "failed"
        assert ex.failure_phase == runtime.FAILURE_STATE_VALIDATE
        assert ex.output_bindings == {}
        assert "state_validate: marker rejected" in (ex.error or "")
        assert ws.instances_for("checkpoint") == []
        assert len(ws.state_snapshots) == 0
        recovery = ws.path / "state_recovery" / ex.id / "state"
        assert exc_info.value.path == recovery
        assert (recovery / "marker.txt").read_text() == "bad"

    def test_state_validator_mutation_does_not_change_snapshot(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _, template = _setup_state_project(
            tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        class SuccessRunner:
            def run(self, plan, args=None):
                assert plan.state_mount_dir is not None
                (plan.state_mount_dir / "marker.txt").write_text("original")
                (plan.proposal_dirs["checkpoint"] / "model.pt").write_text(
                    "weights")
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                )

        def mutating_validator(name, block_def, path, lineage_key):
            assert (path / "marker.txt").read_text() == "original"
            (path / "marker.txt").write_text("mutated by validator")
            (path / "extra.txt").write_text("validator side effect")

        registry = StateValidatorRegistry({"stateful": mutating_validator})

        run_block(
            ws, "stateful", template, project_root,
            state_lineage_key="lineage_a",
            state_validator_registry=registry,
            container_runner=SuccessRunner(),
        )

        latest = ws.latest_state_snapshot("lineage_a")
        assert latest is not None
        snapshot_path = ws.state_snapshot_path(latest.id)
        assert (snapshot_path / "marker.txt").read_text() == "original"
        assert not (snapshot_path / "extra.txt").exists()

    def test_artifact_validators_do_not_validate_state(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, _, template = _setup_state_project(
            tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)

        class SuccessRunner:
            def run(self, plan, args=None):
                assert plan.state_mount_dir is not None
                (plan.state_mount_dir / "marker.txt").write_text("state")
                (plan.proposal_dirs["checkpoint"] / "model.pt").write_text(
                    "weights")
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                )

        def stateful_artifact_validator(name, declaration, path):
            raise AssertionError("artifact validator saw state")

        registry = ArtifactValidatorRegistry({
            "stateful": stateful_artifact_validator,
        })

        run_block(
            ws, "stateful", template, project_root,
            state_lineage_key="lineage_a",
            validator_registry=registry,
            container_runner=SuccessRunner(),
        )

        assert len(ws.state_snapshots) == 1
        assert ws.instances_for("checkpoint")

    def test_incompatible_snapshot_rejects_before_running(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, template_path, template = (
            _setup_state_project(tmp_path)
        )
        ws = Workspace.create("test_ws", template, foundry_dir)

        class SuccessRunner:
            def run(self, plan, args=None):
                assert plan.state_mount_dir is not None
                (plan.state_mount_dir / "marker.txt").write_text("state")
                (plan.proposal_dirs["checkpoint"] / "model.pt").write_text(
                    "weights")
                return RuntimeResult(
                    termination_reason="normal",
                    container_result=ContainerResult(
                        exit_code=0, elapsed_s=0.1),
                )

        run_block(
            ws, "stateful", template, project_root,
            state_lineage_key="lineage_a",
            container_runner=SuccessRunner(),
        )

        changed_template_yaml = STATE_TEMPLATE_YAML.replace(
            "stateful:latest", "stateful:v2")
        template_path.write_text(changed_template_yaml)
        changed_template = from_yaml_with_inline_blocks(template_path)

        class ShouldNotRun:
            def run(self, plan, args=None):
                raise AssertionError("runtime should not start")

        with pytest.raises(ValueError, match="not compatible"):
            run_block(
                ws, "stateful", changed_template, project_root,
                state_lineage_key="lineage_a",
                container_runner=ShouldNotRun(),
            )

        assert len(ws.state_snapshots) == 1
        executions = list(ws.executions.values())
        assert executions[-1].failure_phase == runtime.FAILURE_STAGE_IN
        assert executions[-1].state_mode == "managed"


class TestArtifactValidation:
    """``run_block`` artifact-validator integration.

    Covers commit-passing-slots semantics plus quarantine of
    rejected outputs.
    """

    def test_validator_pass_commits_output(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        seen: list[tuple[str, str, str]] = []

        def validator(name, decl, path):
            # Snapshot the staged contents synchronously; the
            # staging dir is renamed into place once this returns.
            assert path.is_dir()
            seen.append((
                name, path.name,
                (path / "model.pt").read_text(),
            ))

        registry = ArtifactValidatorRegistry({"checkpoint": validator})

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(
                ws, "train", template, project_root,
                validator_registry=registry,
            )

        assert len(seen) == 1
        name, dirname, content = seen[0]
        assert name == "checkpoint"
        # Validator sees the staged candidate produced by
        # ``Workspace.register_artifact``; the staging directory
        # mirrors the canonical artifact's shape but lives under
        # a transient ``_staging-checkpoint-...`` name.
        assert dirname.startswith("_staging-checkpoint-")
        assert content == "weights"
        ex = next(iter(ws.executions.values()))
        assert ex.status == "succeeded"
        assert "checkpoint" in ex.output_bindings

    def test_validator_rejection_records_failure(
        self, tmp_path: Path,
    ):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        def reject(name, decl, path):
            raise ArtifactValidationError("checkpoint missing")

        registry = ArtifactValidatorRegistry({"checkpoint": reject})

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with (
            patch(
                "flywheel.execution.run_container",
                side_effect=fake_run,
            ),
            pytest.raises(
                ArtifactValidationError,
                match="checkpoint missing",
            ),
        ):
            run_block(
                ws, "train", template, project_root,
                validator_registry=registry,
            )

        # No checkpoint instance committed; execution recorded
        # as failed with the validation phase.
        assert ws.instances_for("checkpoint") == []
        ex = next(iter(ws.executions.values()))
        assert ex.status == "failed"
        assert ex.failure_phase == runtime.FAILURE_OUTPUT_VALIDATE
        assert "checkpoint missing" in (ex.error or "")
        assert ex.output_bindings == {}

        # Rejected bytes are preserved under quarantine; the
        # ledger references the workspace-relative path.
        rec = ex.rejected_outputs["checkpoint"]
        assert rec.reason == "checkpoint missing"
        assert rec.quarantine_path == (
            f"quarantine/{ex.id}/checkpoint"
        )
        quarantined = ws.path / rec.quarantine_path / "model.pt"
        assert quarantined.read_text() == "weights"

        # Reject path is cleaned up out of artifact storage.
        artifacts_dir = ws.path / "artifacts"
        leftover = [
            d for d in artifacts_dir.iterdir()
            if d.name.startswith("checkpoint@")
        ]
        assert leftover == []

    def test_quarantine_io_failure_preserves_validation_signal(
        self, tmp_path: Path,
    ):
        # If quarantine itself can't preserve the bytes, the
        # validation failure remains the primary signal:
        # failure_phase stays output_validate, the validator's
        # reason is reported, and quarantine_path is None.
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        def reject(name, decl, path):
            raise ArtifactValidationError("checkpoint missing")

        registry = ArtifactValidatorRegistry({"checkpoint": reject})

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with (
            patch(
                "flywheel.execution.run_container",
                side_effect=fake_run,
            ),
            patch(
                "flywheel.execution.quarantine_slot",
                return_value=None,
            ),
            pytest.raises(
                ArtifactValidationError,
                match="checkpoint missing",
            ),
        ):
            run_block(
                ws, "train", template, project_root,
                validator_registry=registry,
            )

        ex = next(iter(ws.executions.values()))
        assert ex.failure_phase == runtime.FAILURE_OUTPUT_VALIDATE
        assert "checkpoint missing" in (ex.error or "")
        rec = ex.rejected_outputs["checkpoint"]
        assert rec.reason == "checkpoint missing"
        assert rec.quarantine_path is None

    def test_no_validator_registered_skips(self, tmp_path: Path):
        project_root, foundry_dir, template = _setup_git_project(tmp_path)
        ws = Workspace.create("test_ws", template, foundry_dir)
        _commit_all(project_root, "add workspace")

        # Registry exists, but checkpoint is not registered: this
        # is the documented "no validator => accepted" path.
        registry = ArtifactValidatorRegistry()

        fake_run = _fake_run_with_output({"model.pt": "weights"})
        with patch("flywheel.execution.run_container", side_effect=fake_run):
            run_block(
                ws, "train", template, project_root,
                validator_registry=registry,
            )

        ex = next(iter(ws.executions.values()))
        assert ex.status == "succeeded"
        assert "checkpoint" in ex.output_bindings
