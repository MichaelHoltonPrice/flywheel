from __future__ import annotations

from pathlib import Path

import pytest

from flywheel.template import ArtifactDeclaration, BlockDefinition, Template

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
    outputs: [checkpoint]
  - name: eval
    image: cyberloop-eval:latest
    inputs: [checkpoint]
    outputs: [score]
"""


@pytest.fixture()
def valid_template_path(tmp_path: Path) -> Path:
    path = tmp_path / "my_template.yaml"
    path.write_text(VALID_TEMPLATE_YAML)
    return path


class TestFromYaml:
    def test_name_from_file_stem(self, valid_template_path: Path):
        template = Template.from_yaml(valid_template_path)
        assert template.name == "my_template"

    def test_artifacts_parsed(self, valid_template_path: Path):
        template = Template.from_yaml(valid_template_path)
        assert len(template.artifacts) == 3

    def test_git_artifact_fields(self, valid_template_path: Path):
        template = Template.from_yaml(valid_template_path)
        git_art = template.artifacts[0]
        assert git_art.name == "game_engine"
        assert git_art.kind == "git"
        assert git_art.repo == "."
        assert git_art.path == "crates/engine"

    def test_copy_artifact_fields(self, valid_template_path: Path):
        template = Template.from_yaml(valid_template_path)
        copy_art = template.artifacts[1]
        assert copy_art.name == "checkpoint"
        assert copy_art.kind == "copy"
        assert copy_art.repo is None
        assert copy_art.path is None

    def test_blocks_parsed(self, valid_template_path: Path):
        template = Template.from_yaml(valid_template_path)
        assert len(template.blocks) == 2

    def test_block_fields(self, valid_template_path: Path):
        template = Template.from_yaml(valid_template_path)
        train_block = template.blocks[0]
        assert train_block.name == "train"
        assert train_block.image == "cyberloop-train:latest"
        assert len(train_block.inputs) == 1
        assert train_block.inputs[0].name == "checkpoint"
        assert train_block.inputs[0].container_path == "/input/checkpoint"
        assert len(train_block.outputs) == 1
        assert train_block.outputs[0].name == "checkpoint"
        assert train_block.outputs[0].container_path == "/output/checkpoint"

    def test_eval_block_fields(self, valid_template_path: Path):
        template = Template.from_yaml(valid_template_path)
        eval_block = template.blocks[1]
        assert eval_block.name == "eval"
        assert eval_block.image == "cyberloop-eval:latest"
        assert len(eval_block.inputs) == 1
        assert eval_block.inputs[0].name == "checkpoint"
        assert len(eval_block.outputs) == 1
        assert eval_block.outputs[0].name == "score"


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
            Template.from_yaml(path)

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
            Template.from_yaml(path)

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
            Template.from_yaml(path)


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
    outputs: [data]
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="not declared in artifacts"):
            Template.from_yaml(path)

    def test_undeclared_output_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: process
    image: img:latest
    inputs: [data]
    outputs: [nonexistent]
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="not declared in artifacts"):
            Template.from_yaml(path)

    def test_undeclared_input_names_block(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: my_block
    image: img:latest
    inputs: [missing]
    outputs: [data]
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="my_block"):
            Template.from_yaml(path)


class TestDataclassProperties:
    def test_artifact_declaration_frozen(self):
        decl = ArtifactDeclaration(name="x", kind="copy")
        with pytest.raises(AttributeError):
            decl.name = "y"

    def test_block_definition_frozen(self):
        block = BlockDefinition(name="b", image="img", inputs=[], outputs=[])
        with pytest.raises(AttributeError):
            block.name = "other"

    def test_template_frozen(self, valid_template_path: Path):
        template = Template.from_yaml(valid_template_path)
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
        template = Template.from_yaml(path)
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
        template = Template.from_yaml(path)
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
            Template.from_yaml(path)


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
            Template.from_yaml(path)


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
    outputs: [engine]
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="git artifact and cannot be a block output"):
            Template.from_yaml(path)

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
    outputs: [result]
"""
        path = tmp_path / "ok.yaml"
        path.write_text(yaml_content)
        template = Template.from_yaml(path)
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
    outputs: [data]
  - name: process
    image: img2:latest
    inputs: []
    outputs: [data]
"""
        path = tmp_path / "dup_block.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="Duplicate block name"):
            Template.from_yaml(path)


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
    outputs: [data, data]
"""
        path = tmp_path / "dup_output.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="duplicate output"):
            Template.from_yaml(path)


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
            Template.from_yaml(path)

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
            Template.from_yaml(path)

    def test_block_name_with_space_raises(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: data
    kind: copy
blocks:
  - name: "my block"
    image: img:latest
    inputs: []
    outputs: [data]
"""
        path = tmp_path / "bad.yaml"
        path.write_text(yaml_content)
        with pytest.raises(ValueError, match="invalid"):
            Template.from_yaml(path)

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
    outputs: [checkpoint]
"""
        path = tmp_path / "expanded.yaml"
        path.write_text(yaml_content)
        template = Template.from_yaml(path)
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
      - name: checkpoint
        container_path: /output
"""
        path = tmp_path / "expanded.yaml"
        path.write_text(yaml_content)
        template = Template.from_yaml(path)
        out = template.blocks[0].outputs[0]
        assert out.name == "checkpoint"
        assert out.container_path == "/output"

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
    outputs: [checkpoint]
"""
        path = tmp_path / "resources.yaml"
        path.write_text(yaml_content)
        template = Template.from_yaml(path)
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
    outputs: [data]
"""
        path = tmp_path / "defaults.yaml"
        path.write_text(yaml_content)
        template = Template.from_yaml(path)
        block = template.blocks[0]
        assert block.docker_args == []
        assert block.env == {}
        assert block.inputs[0].optional is False
        assert block.inputs[0].container_path == "/input/data"
        assert block.outputs[0].container_path == "/output/data"


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
    outputs: [my-data_01]
"""
        path = tmp_path / "ok.yaml"
        path.write_text(yaml_content)
        template = Template.from_yaml(path)
        assert template.artifacts[0].name == "my-data_01"
        assert template.blocks[0].name == "my-block_v2"


class TestRecordSentinelImage:
    def test_record_image_parses(self, tmp_path: Path):
        yaml_content = """\
artifacts:
  - name: session
    kind: copy
  - name: step
    kind: copy
blocks:
  - name: game_step
    image: "__record__"
    inputs:
      - name: session
        container_path: /input/session
    outputs:
      - name: session
        container_path: /output/session
      - name: step
        container_path: /output/step
"""
        path = tmp_path / "record.yaml"
        path.write_text(yaml_content)
        template = Template.from_yaml(path)
        assert len(template.blocks) == 1
        assert template.blocks[0].image == "__record__"
        assert len(template.blocks[0].inputs) == 1
        assert len(template.blocks[0].outputs) == 2

    def test_same_artifact_in_input_and_output(self, tmp_path: Path):
        """Session appears in both inputs and outputs — must be valid."""
        yaml_content = """\
artifacts:
  - name: session
    kind: copy
blocks:
  - name: step
    image: "__record__"
    inputs: [session]
    outputs: [session]
"""
        path = tmp_path / "record.yaml"
        path.write_text(yaml_content)
        template = Template.from_yaml(path)
        assert template.blocks[0].inputs[0].name == "session"
        assert template.blocks[0].outputs[0].name == "session"
