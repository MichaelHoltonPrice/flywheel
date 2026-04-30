from __future__ import annotations

from pathlib import Path

import pytest

from flywheel.template import (
    ArtifactDeclaration,
    BlockDefinition,
)
from tests._inline_blocks import (
    from_yaml_with_inline_blocks as _from_yaml_with_inline_blocks,
)

VALID_TEMPLATE_YAML = """\
artifacts:
  - name: game_engine
    kind: git
    repo: "."
    path: crates/engine
  - name: checkpoint
    kind: copy
  - name: score
    kind: copy

blocks:
  - name: train
    image: cyberloop-train:latest
    inputs: [checkpoint]
    outputs:
      normal:
        - checkpoint
  - name: eval
    image: cyberloop-eval:latest
    inputs: [checkpoint]
    outputs:
      normal:
        - score
"""


@pytest.fixture()
def valid_template_path(tmp_path: Path) -> Path:
    path = tmp_path / "my_template.yaml"
    path.write_text(VALID_TEMPLATE_YAML)
    return path


class TestFromYaml:
    def test_name_from_file_stem(self, valid_template_path: Path):
        template = _from_yaml_with_inline_blocks(valid_template_path)
        assert template.name == "my_template"

    def test_artifacts_parsed(self, valid_template_path: Path):
        template = _from_yaml_with_inline_blocks(valid_template_path)
        assert len(template.artifacts) == 3

    def test_git_artifact_fields(self, valid_template_path: Path):
        template = _from_yaml_with_inline_blocks(valid_template_path)
        git_art = template.artifacts[0]
        assert git_art.name == "game_engine"
        assert git_art.kind == "git"
        assert git_art.repo == "."
        assert git_art.path == "crates/engine"

    def test_copy_artifact_fields(self, valid_template_path: Path):
        template = _from_yaml_with_inline_blocks(valid_template_path)
        copy_art = template.artifacts[1]
        assert copy_art.name == "checkpoint"
        assert copy_art.kind == "copy"
        assert copy_art.repo is None
        assert copy_art.path is None

    def test_blocks_parsed(self, valid_template_path: Path):
        template = _from_yaml_with_inline_blocks(valid_template_path)
        assert len(template.blocks) == 2

    def test_block_fields(self, valid_template_path: Path):
        template = _from_yaml_with_inline_blocks(valid_template_path)
        train_block = template.blocks[0]
        assert train_block.name == "train"
        assert train_block.image == "cyberloop-train:latest"
        assert len(train_block.inputs) == 1
        assert train_block.inputs[0].name == "checkpoint"
        assert train_block.inputs[0].container_path == "/input/checkpoint"
        train_outputs = train_block.all_output_slots()
        assert len(train_outputs) == 1
        assert train_outputs[0].name == "checkpoint"
        assert train_outputs[0].container_path == "/output/checkpoint"

    def test_eval_block_fields(self, valid_template_path: Path):
        template = _from_yaml_with_inline_blocks(valid_template_path)
        eval_block = template.blocks[1]
        assert eval_block.name == "eval"
        assert eval_block.image == "cyberloop-eval:latest"
        assert len(eval_block.inputs) == 1
        assert eval_block.inputs[0].name == "checkpoint"
        eval_outputs = eval_block.all_output_slots()
        assert len(eval_outputs) == 1
        assert eval_outputs[0].name == "score"


class TestGitArtifactValidation:
    def test_missing_repo_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: code
    kind: git
    path: src
blocks: []
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="requires 'repo'"):
            _from_yaml_with_inline_blocks(path)

    def test_missing_path_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: code
    kind: git
    repo: "."
blocks: []
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="requires 'path'"):
            _from_yaml_with_inline_blocks(path)

    def test_missing_both_repo_and_path_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: code
    kind: git
blocks: []
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="requires 'repo'"):
            _from_yaml_with_inline_blocks(path)


class TestBlockArtifactRefValidation:
    def test_undeclared_input_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: process
    image: img:latest
    inputs: [nonexistent]
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="not declared in artifacts"):
            _from_yaml_with_inline_blocks(path)

    def test_undeclared_output_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: process
    image: img:latest
    inputs: [data]
    outputs:
      normal:
        - nonexistent
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="not declared in artifacts"):
            _from_yaml_with_inline_blocks(path)

    def test_undeclared_input_names_block(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: my_block
    image: img:latest
    inputs: [missing]
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="my_block"):
            _from_yaml_with_inline_blocks(path)


class TestInvocationRouteValidation:
    BASE = """\
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
    inputs:
      - name: config
        container_path: /input/config
    outputs:
      eval_requested:
        - name: bot
          container_path: /output/bot
    {route}
  - name: eval
    image: eval:latest
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      normal:
        - name: score
          container_path: /output/score
"""

    def _write(self, tmp_path: Path, route: str) -> Path:
        path = tmp_path / "invocation.yaml"
        path.write_text(self.BASE.format(route=route))
        return path

    def test_parses_parent_output_binding(self, tmp_path: Path):
        path = self._write(tmp_path, """\
on_termination:
      eval_requested:
        invoke:
          - block: eval
            bind:
              bot: bot""")
        template = _from_yaml_with_inline_blocks(path)
        agent = template.blocks[0]
        route = agent.on_termination["eval_requested"][0]
        assert route.block == "eval"
        assert route.bind["bot"].parent_output == "bot"

    def test_parses_required_expected_termination_reasons(
        self, tmp_path: Path,
    ):
        path = self._write(tmp_path, """\
on_termination:
      eval_requested:
        invoke:
          - block: eval
            required: true
            expected_termination_reasons:
              - normal
            bind:
              bot: bot""")
        template = _from_yaml_with_inline_blocks(path)
        route = template.blocks[0].on_termination["eval_requested"][0]
        assert route.required is True
        assert route.expected_termination_reasons == ("normal",)

    def test_rejects_unknown_expected_termination_reason(
        self, tmp_path: Path,
    ):
        path = self._write(tmp_path, """\
on_termination:
      eval_requested:
        invoke:
          - block: eval
            expected_termination_reasons:
              - missing
            bind:
              bot: bot""")
        with pytest.raises(ValueError, match="expects child"):
            _from_yaml_with_inline_blocks(path)

    def test_rejects_route_for_undeclared_reason(self, tmp_path: Path):
        path = self._write(tmp_path, """\
on_termination:
      done:
        invoke:
          - block: eval""")
        with pytest.raises(ValueError, match="not declared under outputs"):
            _from_yaml_with_inline_blocks(path)

    def test_rejects_unknown_child_block(self, tmp_path: Path):
        path = self._write(tmp_path, """\
on_termination:
      eval_requested:
        invoke:
          - block: missing""")
        with pytest.raises(ValueError, match="unknown block"):
            _from_yaml_with_inline_blocks(path)

    def test_rejects_unknown_child_input(self, tmp_path: Path):
        path = self._write(tmp_path, """\
on_termination:
      eval_requested:
        invoke:
          - block: eval
            bind:
              missing: bot""")
        with pytest.raises(ValueError, match="has no such input"):
            _from_yaml_with_inline_blocks(path)

    def test_rejects_parent_output_not_declared_for_reason(
        self, tmp_path: Path,
    ):
        path = self._write(tmp_path, """\
on_termination:
      eval_requested:
        invoke:
          - block: eval
            bind:
              bot: score""")
        with pytest.raises(ValueError, match="not declared"):
            _from_yaml_with_inline_blocks(path)

    def test_parses_long_form_parent_output_binding(self, tmp_path: Path):
        path = self._write(tmp_path, """\
on_termination:
      eval_requested:
        invoke:
          - block: eval
            bind:
              bot:
                parent_output: bot""")
        template = _from_yaml_with_inline_blocks(path)
        route = template.blocks[0].on_termination["eval_requested"][0]
        assert route.bind["bot"].parent_output == "bot"

    def test_parses_parent_input_binding(self, tmp_path: Path):
        path = self._write(tmp_path, """\
on_termination:
      eval_requested:
        invoke:
          - block: eval
            bind:
              bot:
                parent_input: config""")
        template = _from_yaml_with_inline_blocks(path)
        route = template.blocks[0].on_termination["eval_requested"][0]
        assert route.bind["bot"].parent_input == "config"

    def test_rejects_unknown_long_form_binding_key(self, tmp_path: Path):
        path = self._write(tmp_path, """\
on_termination:
      eval_requested:
        invoke:
          - block: eval
            bind:
              bot:
                source: bot""")
        with pytest.raises(ValueError, match="unknown key"):
            _from_yaml_with_inline_blocks(path)

    def test_rejects_managed_state_child(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: bot
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
          - block: stateful
            bind:
              bot: bot
  - name: stateful
    image: stateful:latest
    state: managed
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      normal: []
"""
        path = tmp_path / "managed_child.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="managed-state child"):
            _from_yaml_with_inline_blocks(path)

    def test_rejects_invocation_route_cycle(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: bot
    kind: copy
blocks:
  - name: first
    image: first:latest
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      next:
        - name: bot
          container_path: /output/bot
    on_termination:
      next:
        invoke:
          - block: second
            bind:
              bot: bot
  - name: second
    image: second:latest
    inputs:
      - name: bot
        container_path: /input/bot
    outputs:
      next:
        - name: bot
          container_path: /output/bot
    on_termination:
      next:
        invoke:
          - block: first
            bind:
              bot: bot
"""
        path = tmp_path / "cycle.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="route cycle"):
            _from_yaml_with_inline_blocks(path)


class TestDataclassProperties:
    def test_artifact_declaration_frozen(self):
        decl = ArtifactDeclaration(name="x", kind="copy")
        with pytest.raises(AttributeError):
            decl.name = "y"

    def test_block_definition_frozen(self):
        block = BlockDefinition(name="b", image="img", inputs=[], outputs={})
        with pytest.raises(AttributeError):
            block.name = "other"

    def test_template_frozen(self, valid_template_path: Path):
        template = _from_yaml_with_inline_blocks(valid_template_path)
        with pytest.raises(AttributeError):
            template.name = "other"


class TestMinimalTemplate:
    def test_no_artifacts_no_blocks(self, tmp_path: Path):
        yaml_content = """\
artifacts: []
blocks: []
"""
        path = tmp_path / "empty.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.name == "empty"
        assert template.artifacts == []
        assert template.blocks == []

    def test_copy_only_no_blocks(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks: []
"""
        path = tmp_path / "simple.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert len(template.artifacts) == 1
        assert template.artifacts[0].kind == "copy"


class TestDuplicateArtifactNames:
    def test_duplicate_name_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
  - name: data
    kind: copy
blocks: []
"""
        path = tmp_path / "dup.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="Duplicate artifact name"):
            _from_yaml_with_inline_blocks(path)


class TestUnknownKind:
    def test_unknown_kind_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: gti
blocks: []
"""
        path = tmp_path / "bad_kind.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="unknown kind"):
            _from_yaml_with_inline_blocks(path)


class TestGitArtifactCannotBeBlockOutput:
    def test_git_output_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: engine
    kind: git
    repo: "."
    path: src
blocks:
  - name: build
    image: build:latest
    inputs: []
    outputs:
      normal:
        - engine
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="git artifact and cannot be a block output"):
            _from_yaml_with_inline_blocks(path)

    def test_git_input_is_allowed(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: engine
    kind: git
    repo: "."
    path: src
  - name: result
    kind: copy
blocks:
  - name: eval
    image: eval:latest
    inputs: [engine]
    outputs:
      normal:
        - result
"""
        path = tmp_path / "ok.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].inputs[0].name == "engine"


class TestDuplicateBlockNames:
    def test_duplicate_block_name_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: process
    image: img:latest
    inputs: []
    outputs:
      normal:
        - data
  - name: process
    image: img2:latest
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "dup_block.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="Duplicate block name"):
            _from_yaml_with_inline_blocks(path)


class TestDuplicateOutputSlots:
    def test_duplicate_output_in_block_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: process
    image: img:latest
    inputs: []
    outputs:
      normal:
        - data
        - data
"""
        path = tmp_path / "dup_output.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="duplicate output"):
            _from_yaml_with_inline_blocks(path)


class TestNameValidation:
    def test_empty_artifact_name_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: ""
    kind: copy
blocks: []
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="must not be empty"):
            _from_yaml_with_inline_blocks(path)

    def test_artifact_name_with_slash_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: "my/artifact"
    kind: copy
blocks: []
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="invalid"):
            _from_yaml_with_inline_blocks(path)

    def test_block_name_with_space_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: "my block"
    image: img:latest
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="invalid"):
            _from_yaml_with_inline_blocks(path)

class TestExpandedBlockFormat:
    def test_expanded_inputs_with_container_path(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: checkpoint
    kind: copy
blocks:
  - name: train
    image: train:latest
    inputs:
      - name: checkpoint
        container_path: /data/input
        optional: true
    outputs:
      normal:
        - checkpoint
"""
        path = tmp_path / "expanded.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        inp = template.blocks[0].inputs[0]
        assert inp.name == "checkpoint"
        assert inp.container_path == "/data/input"
        assert inp.optional is True

    def test_expanded_outputs_with_container_path(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: checkpoint
    kind: copy
blocks:
  - name: train
    image: train:latest
    inputs: []
    outputs:
      normal:
        - name: checkpoint
          container_path: /output
"""
        path = tmp_path / "expanded.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        out = template.blocks[0].all_output_slots()[0]
        assert out.name == "checkpoint"
        assert out.container_path == "/output"

    def test_flat_outputs_are_rejected(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: checkpoint
    kind: copy
blocks:
  - name: train
    image: train:latest
    inputs: []
    outputs:
      - name: checkpoint
        container_path: /output
"""
        path = tmp_path / "flat_outputs.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="mapping keyed by termination"):
            _from_yaml_with_inline_blocks(path)

    def test_resource_config(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: checkpoint
    kind: copy
blocks:
  - name: train
    image: train:latest
    docker_args: ["--gpus", "all", "--shm-size", "8g"]
    env:
      OMP_NUM_THREADS: "1"
    inputs: []
    outputs:
      normal:
        - checkpoint
"""
        path = tmp_path / "resources.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        block = template.blocks[0]
        assert block.docker_args == ["--gpus", "all", "--shm-size", "8g"]
        assert block.env == {"OMP_NUM_THREADS": "1"}

    def test_defaults_for_optional_fields(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: process
    image: proc:latest
    inputs: [data]
    outputs:
      normal:
        - data
"""
        path = tmp_path / "defaults.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        block = template.blocks[0]
        assert block.docker_args == []
        assert block.env == {}
        assert block.inputs[0].optional is False
        assert block.inputs[0].container_path == "/input/data"
        assert (
            block.all_output_slots()[0].container_path == "/output/data"
        )


class TestNetworkField:
    def test_network_parses(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    network: cyberloop-cua
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "network.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].network == "cyberloop-cua"

    def test_network_defaults_to_none(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "network.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].network is None

    @pytest.mark.parametrize("value", ["", "   "])
    def test_network_must_be_non_empty(self, tmp_path: Path, value: str):
        yaml_content = f"""\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    network: {value!r}
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "network.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="network.*non-empty"):
            _from_yaml_with_inline_blocks(path)

    @pytest.mark.parametrize("flag", [
        "--network",
        "--network=bridge",
        "--net",
        "--net=bridge",
    ])
    def test_network_flags_rejected_in_docker_args(
        self, tmp_path: Path, flag: str,
    ):
        yaml_content = f"""\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    docker_args:
      - {flag}
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad_network_arg.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="use.*network"):
            _from_yaml_with_inline_blocks(path)

    def test_network_alias_is_not_network_opt_in(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    docker_args:
      - --network-alias=proc
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "network_alias.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        block = template.blocks[0]
        assert block.network is None
        assert block.docker_args == ["--network-alias=proc"]

    def test_network_rejected_on_lifecycle_runner(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    runner: lifecycle
    runner_justification: "test fixture"
    network: bridge
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad_network.yaml"
        path.write_text(yaml_content)
        with pytest.raises(
            ValueError, match="only valid for runner 'container'",
        ):
            _from_yaml_with_inline_blocks(path)


class TestNameValidationAccepted:
    def test_valid_names_with_hyphens_and_underscores(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: my-data_01
    kind: copy
blocks:
  - name: my-block_v2
    image: img:latest
    inputs: []
    outputs:
      normal:
        - my-data_01
"""
        path = tmp_path / "ok.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.artifacts[0].name == "my-data_01"
        assert template.blocks[0].name == "my-block_v2"


class TestSameArtifactInInputAndOutput:
    def test_same_artifact_in_input_and_output(self, tmp_path: Path):
        """Session appears in both inputs and outputs — must be valid."""
        yaml_content = """\
artifacts:
  - name: session
    kind: copy
blocks:
  - name: step
    runner: lifecycle
    runner_justification: "Test fixture"
    inputs: [session]
    outputs:
      normal:
        - session
"""
        path = tmp_path / "lifecycle.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].inputs[0].name == "session"
        assert (
            template.blocks[0].all_output_slots()[0].name == "session"
        )


class TestLifecycle:
    """``lifecycle:`` on a block YAML declares the container-lifetime model."""

    def test_default_is_one_shot(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "default.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].lifecycle == "one_shot"

    def test_workspace_persistent_parses(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    lifecycle: workspace_persistent
    network: bridge
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "persistent.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].lifecycle == "workspace_persistent"
        assert template.blocks[0].network == "bridge"

    def test_workspace_persistent_requires_network(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    lifecycle: workspace_persistent
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "persistent.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="requires.*network"):
            _from_yaml_with_inline_blocks(path)

    def test_unknown_lifecycle_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    lifecycle: forever
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="unknown lifecycle"):
            _from_yaml_with_inline_blocks(path)

    def test_persistent_rejected_on_non_container(self, tmp_path: Path):
        """Lifecycle only applies to container runners."""
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    runner: lifecycle
    runner_justification: "test fixture"
    lifecycle: workspace_persistent
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(
            ValueError, match="only valid for runner 'container'",
        ):
            _from_yaml_with_inline_blocks(path)

    def test_explicit_one_shot_rejected_on_non_container(
        self, tmp_path: Path,
    ):
        """Even the default value is rejected — the key itself is the error.

        ``lifecycle`` has no meaning for non-container blocks, so
        declaring it at all on one is a config mistake worth
        surfacing.  Matches the docstring contract.
        """
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    runner: lifecycle
    runner_justification: "test fixture"
    lifecycle: one_shot
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(
            ValueError, match="only valid for runner 'container'",
        ):
            _from_yaml_with_inline_blocks(path)


class TestStateField:
    """``state:`` on a block YAML declares private state support."""

    def test_default_is_none(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "default.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].state == "none"

    def test_state_true_parses_as_managed(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    state: true
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "stateful.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].state == "managed"

    def test_named_state_modes_parse(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    state: unmanaged
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "stateful.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].state == "unmanaged"

    def test_unknown_state_mode_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    state: "yes"
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="unknown state mode"):
            _from_yaml_with_inline_blocks(path)

    def test_state_false_is_rejected(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    state: false
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="'state' must be"):
            _from_yaml_with_inline_blocks(path)

    def test_state_rejected_on_non_container(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    runner: lifecycle
    runner_justification: "test fixture"
    state: true
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(
            ValueError, match="only valid for runner 'container'",
        ):
            _from_yaml_with_inline_blocks(path)

    def test_managed_state_with_workspace_persistent_rejected(
        self, tmp_path: Path,
    ):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    lifecycle: workspace_persistent
    network: bridge
    state: managed
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(
            ValueError, match="mutually exclusive",
        ):
            _from_yaml_with_inline_blocks(path)

    def test_unmanaged_state_with_workspace_persistent_parses(
        self, tmp_path: Path,
    ):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    lifecycle: workspace_persistent
    network: bridge
    state: unmanaged
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "persistent.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].state == "unmanaged"


class TestStopTimeout:
    """``stop_timeout_s:`` on a block YAML sets the cooperative
    stop grace window before forced termination."""

    def test_default_is_thirty(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "default.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].stop_timeout_s == 30

    def test_explicit_override_parses(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    stop_timeout_s: 120
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "custom.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].stop_timeout_s == 120

    def test_zero_is_allowed(self, tmp_path: Path):
        """Zero means 'skip cooperative, go straight to TERM';
        explicitly permitted for containers that don't poll."""
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    stop_timeout_s: 0
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "zero.yaml"
        path.write_text(yaml_content)
        template = _from_yaml_with_inline_blocks(path)
        assert template.blocks[0].stop_timeout_s == 0

    def test_negative_rejected(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    stop_timeout_s: -1
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(
            ValueError, match="must be non-negative",
        ):
            _from_yaml_with_inline_blocks(path)

    def test_non_int_rejected(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    stop_timeout_s: "30"
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(
            ValueError, match="non-negative integer",
        ):
            _from_yaml_with_inline_blocks(path)

    def test_rejected_on_non_container(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    runner: lifecycle
    runner_justification: "test fixture"
    stop_timeout_s: 15
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(
            ValueError, match="only valid for runner 'container'",
        ):
            _from_yaml_with_inline_blocks(path)


class TestUnknownBlockKeys:
    """Strict top-level key validation on block YAMLs."""

    def test_unknown_key_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    notes: random extra field
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="unknown top-level"):
            _from_yaml_with_inline_blocks(path)

    def test_typo_in_known_key_raises(self, tmp_path: Path):
        """A typo like ``lifecylce`` surfaces instead of being dropped."""
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    lifecylce: workspace_persistent
    inputs: []
    outputs:
      normal:
        - data
"""
        path = tmp_path / "typo.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="lifecylce"):
            _from_yaml_with_inline_blocks(path)


class TestUnknownSlotKeys:
    """Strict key validation extends into nested input/output mappings."""

    def test_unknown_key_in_input_slot_raises(self, tmp_path: Path):
        """``optionl`` (typo of ``optional``) should surface."""
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    inputs:
      - name: data
        container_path: /input/data
        optionl: true
    outputs:
      normal:
        - data
"""
        path = tmp_path / "bad_in.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="optionl"):
            _from_yaml_with_inline_blocks(path)

    def test_unknown_key_in_output_slot_raises(self, tmp_path: Path):
        """``container_pat`` (typo of ``container_path``) should surface."""
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: proc
    image: proc:latest
    inputs: []
    outputs:
      normal:
        - name: data
          container_pat: /output/data
"""
        path = tmp_path / "bad_out.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="container_pat"):
            _from_yaml_with_inline_blocks(path)


