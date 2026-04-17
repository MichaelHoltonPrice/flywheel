"""Tests for the per-block YAML registry and template integration.

Covers Phase 2 of the block-execution refactor:

- ``BlockRegistry.from_directory`` loads ``workforce/blocks/*.yaml``.
- Block YAML schema validation (runner, image, implementation,
  runner_justification rules).
- Template integration: string entries in ``blocks:`` resolve to
  registry blocks; inline-dict entries still work; mixed lists
  load.
- ``ProjectConfig.load_block_registry`` auto-discovers from
  ``<project_root>/workforce/blocks``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from flywheel.blocks import BlockRegistry, load_block_file
from flywheel.config import load_project_config
from flywheel.template import (
    BlockImplementation,
    Template,
    parse_block_definition,
)


# A minimal valid container-block YAML body.
CONTAINER_BLOCK = {
    "name": "train",
    "image": "train:latest",
    "inputs": [
        {"name": "engine", "container_path": "/input/engine"},
    ],
    "outputs": [
        {"name": "checkpoint", "container_path": "/output/checkpoint"},
    ],
}

INPROCESS_BLOCK = {
    "name": "predict",
    "runner": "inprocess",
    "runner_justification": (
        "Lightweight Python evaluation; container overhead would "
        "dominate the actual compute."
    ),
    "inputs": [
        {"name": "predictor", "container_path": "/input/predictor"},
    ],
    "outputs": [
        {"name": "prediction",
         "container_path": "/output/prediction"},
    ],
    "implementation": {
        "python_module": "mypkg.blocks.predict_block",
        "entry": "run",
    },
}


# Template with no inline blocks; everything comes from the registry.
TEMPLATE_REGISTRY_ONLY = """\
artifacts:
  - name: engine
    kind: copy
  - name: checkpoint
    kind: copy
  - name: predictor
    kind: copy
  - name: prediction
    kind: copy

blocks:
  - train
  - predict
"""


def _write_yaml(path: Path, body: dict) -> None:
    path.write_text(yaml.safe_dump(body, sort_keys=False))


@pytest.fixture()
def blocks_dir(tmp_path: Path) -> Path:
    """A directory with two valid block YAML files."""
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    _write_yaml(blocks / "train.yaml", CONTAINER_BLOCK)
    _write_yaml(blocks / "predict.yaml", INPROCESS_BLOCK)
    return blocks


class TestParseBlockDefinition:
    def test_container_block_minimum(self):
        block = parse_block_definition(CONTAINER_BLOCK)
        assert block.name == "train"
        assert block.runner == "container"
        assert block.image == "train:latest"
        assert block.implementation is None
        assert block.runner_justification is None
        assert len(block.inputs) == 1
        assert block.inputs[0].name == "engine"
        assert block.outputs[0].name == "checkpoint"

    def test_inprocess_block_minimum(self):
        block = parse_block_definition(INPROCESS_BLOCK)
        assert block.name == "predict"
        assert block.runner == "inprocess"
        assert block.image == ""
        assert block.runner_justification.startswith("Lightweight")
        assert isinstance(block.implementation, BlockImplementation)
        assert (block.implementation.python_module
                == "mypkg.blocks.predict_block")
        assert block.implementation.entry == "run"

    def test_inprocess_implementation_default_entry(self):
        body = {
            **INPROCESS_BLOCK,
            "implementation": {
                "python_module": "x.y",
                # entry omitted; should default to "run".
            },
        }
        block = parse_block_definition(body)
        assert block.implementation.entry == "run"

    def test_unknown_runner_rejected(self):
        body = {**INPROCESS_BLOCK, "runner": "wasm"}
        with pytest.raises(ValueError, match="unknown runner"):
            parse_block_definition(body)

    def test_container_requires_image(self):
        body = {**CONTAINER_BLOCK}
        body.pop("image")
        with pytest.raises(
                ValueError, match="requires.*non-empty 'image'"):
            parse_block_definition(body)

    def test_container_forbids_implementation(self):
        body = {
            **CONTAINER_BLOCK,
            "implementation": {"python_module": "x"},
        }
        with pytest.raises(
                ValueError, match="must not declare 'implementation'"):
            parse_block_definition(body)

    def test_inprocess_forbids_image(self):
        body = {**INPROCESS_BLOCK, "image": "leaked:latest"}
        with pytest.raises(
                ValueError, match="must not declare 'image'"):
            parse_block_definition(body)

    def test_inprocess_requires_justification(self):
        body = {**INPROCESS_BLOCK}
        body.pop("runner_justification")
        with pytest.raises(
                ValueError,
                match="requires 'runner_justification'"):
            parse_block_definition(body)

    def test_inprocess_requires_implementation(self):
        body = {**INPROCESS_BLOCK}
        body.pop("implementation")
        with pytest.raises(
                ValueError, match="requires 'implementation'"):
            parse_block_definition(body)

    def test_implementation_requires_python_module(self):
        body = {
            **INPROCESS_BLOCK,
            "implementation": {"entry": "run"},
        }
        with pytest.raises(
                ValueError, match="missing.*'python_module'"):
            parse_block_definition(body)

    def test_duplicate_output_rejected(self):
        body = {
            **CONTAINER_BLOCK,
            "outputs": [
                {"name": "checkpoint",
                 "container_path": "/output/checkpoint"},
                {"name": "checkpoint",
                 "container_path": "/output/checkpoint2"},
            ],
        }
        with pytest.raises(
                ValueError, match="duplicate output"):
            parse_block_definition(body)


class TestLoadBlockFile:
    def test_loads_valid_file(self, blocks_dir: Path):
        block = load_block_file(blocks_dir / "train.yaml")
        assert block.name == "train"

    def test_empty_file_rejected(self, tmp_path: Path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_block_file(f)

    def test_top_level_list_rejected(self, tmp_path: Path):
        f = tmp_path / "list.yaml"
        f.write_text("- a\n- b\n")
        with pytest.raises(ValueError, match="mapping"):
            load_block_file(f)


class TestBlockRegistry:
    def test_from_directory_loads_all(self, blocks_dir: Path):
        registry = BlockRegistry.from_directory(blocks_dir)
        assert sorted(registry.names()) == ["predict", "train"]
        assert "train" in registry
        assert "predict" in registry
        assert registry.get("train").name == "train"

    def test_unknown_name_raises(self, blocks_dir: Path):
        registry = BlockRegistry.from_directory(blocks_dir)
        with pytest.raises(KeyError, match="bogus"):
            registry.get("bogus")

    def test_missing_directory_returns_empty(self, tmp_path: Path):
        registry = BlockRegistry.from_directory(
            tmp_path / "does-not-exist")
        assert registry.names() == []

    def test_underscore_files_skipped(self, tmp_path: Path):
        d = tmp_path / "blocks"
        d.mkdir()
        _write_yaml(d / "train.yaml", CONTAINER_BLOCK)
        _write_yaml(d / "_draft_predict.yaml", INPROCESS_BLOCK)
        registry = BlockRegistry.from_directory(d)
        assert registry.names() == ["train"]

    def test_stem_must_match_name(self, tmp_path: Path):
        d = tmp_path / "blocks"
        d.mkdir()
        _write_yaml(d / "wrong_name.yaml", CONTAINER_BLOCK)
        with pytest.raises(
                ValueError, match="stem is 'wrong_name'"):
            BlockRegistry.from_directory(d)

    def test_duplicate_name_across_files_rejected(
            self, tmp_path: Path):
        d = tmp_path / "blocks"
        d.mkdir()
        _write_yaml(d / "train.yaml", CONTAINER_BLOCK)
        # Same name, different stem (will hit stem check first).
        # Force a duplicate by giving two files the same logical
        # name but having matching stems impossible — instead use
        # an aliased copy.
        body2 = {**CONTAINER_BLOCK, "image": "train:other"}
        _write_yaml(d / "train2.yaml", body2)
        # train2.yaml's name is still "train" — first hits stem
        # mismatch.
        with pytest.raises(ValueError, match="stem is 'train2'"):
            BlockRegistry.from_directory(d)

    def test_sources_recorded(self, blocks_dir: Path):
        registry = BlockRegistry.from_directory(blocks_dir)
        assert registry.sources["train"].name == "train.yaml"


class TestTemplateRegistryIntegration:
    def test_string_block_reference_resolves(
            self, tmp_path: Path, blocks_dir: Path):
        tmpl_path = tmp_path / "test_tmpl.yaml"
        tmpl_path.write_text(TEMPLATE_REGISTRY_ONLY)
        registry = BlockRegistry.from_directory(blocks_dir)

        template = Template.from_yaml(
            tmpl_path, block_registry=registry)

        names = [b.name for b in template.blocks]
        assert names == ["train", "predict"]
        assert template.blocks[1].runner == "inprocess"

    def test_string_reference_without_registry_raises(
            self, tmp_path: Path):
        tmpl_path = tmp_path / "t.yaml"
        tmpl_path.write_text(TEMPLATE_REGISTRY_ONLY)
        with pytest.raises(
                ValueError,
                match="no block_registry was provided"):
            Template.from_yaml(tmpl_path)

    def test_unknown_block_name_raises(
            self, tmp_path: Path, blocks_dir: Path):
        tmpl_path = tmp_path / "t.yaml"
        tmpl_path.write_text(
            "artifacts:\n"
            "  - {name: engine, kind: copy}\n"
            "  - {name: checkpoint, kind: copy}\n"
            "blocks:\n"
            "  - bogus\n"
        )
        registry = BlockRegistry.from_directory(blocks_dir)
        with pytest.raises(
                ValueError, match="unknown block 'bogus'"):
            Template.from_yaml(tmpl_path, block_registry=registry)

    def test_mixed_inline_and_registry_no_longer_supported(
            self, tmp_path: Path, blocks_dir: Path):
        # Phase 5 of the block-execution refactor removed inline
        # block definitions; templates may only reference blocks
        # by name from the registry.
        tmpl_path = tmp_path / "t.yaml"
        tmpl_path.write_text(
            "artifacts:\n"
            "  - {name: engine, kind: copy}\n"
            "  - {name: checkpoint, kind: copy}\n"
            "  - {name: misc, kind: copy}\n"
            "blocks:\n"
            "  - train\n"
            "  - name: inline_block\n"
            "    image: inline:latest\n"
            "    inputs: [misc]\n"
            "    outputs: [misc]\n"
        )
        registry = BlockRegistry.from_directory(blocks_dir)
        with pytest.raises(
                ValueError, match="unsupported type 'dict'"):
            Template.from_yaml(
                tmpl_path, block_registry=registry)

    def test_block_input_must_be_declared_artifact(
            self, tmp_path: Path):
        # Build a registry where the block's input refers to an
        # artifact the template doesn't declare.
        d = tmp_path / "blocks"
        d.mkdir()
        body = {
            **CONTAINER_BLOCK,
            "inputs": [{"name": "missing_artifact",
                        "container_path": "/x"}],
        }
        _write_yaml(d / "train.yaml", body)
        registry = BlockRegistry.from_directory(d)

        tmpl = tmp_path / "t.yaml"
        tmpl.write_text(
            "artifacts:\n"
            "  - {name: checkpoint, kind: copy}\n"
            "blocks: [train]\n"
        )
        with pytest.raises(
                ValueError,
                match="input 'missing_artifact' not declared"):
            Template.from_yaml(tmpl, block_registry=registry)

    def test_unsupported_block_entry_type_raises(
            self, tmp_path: Path):
        tmpl = tmp_path / "t.yaml"
        tmpl.write_text(
            "artifacts: []\n"
            "blocks:\n"
            "  - 42\n"
        )
        with pytest.raises(
                ValueError,
                match="unsupported type 'int'"):
            Template.from_yaml(tmpl)


class TestInlineBlockRemoved:
    def test_inline_block_definition_is_now_an_error(
            self, tmp_path: Path):
        # Phase 5 of the block-execution refactor removed support
        # for inline-dict block definitions in templates.
        tmpl = tmp_path / "t.yaml"
        tmpl.write_text(
            "artifacts:\n"
            "  - {name: misc, kind: copy}\n"
            "blocks:\n"
            "  - name: foo\n"
            "    image: foo:latest\n"
            "    inputs: [misc]\n"
            "    outputs: [misc]\n"
        )
        with pytest.raises(
                ValueError, match="unsupported type 'dict'"):
            Template.from_yaml(tmpl)

    def test_registry_only_template_loads_cleanly(
            self, tmp_path: Path, blocks_dir: Path):
        tmpl_path = tmp_path / "t.yaml"
        tmpl_path.write_text(TEMPLATE_REGISTRY_ONLY)
        registry = BlockRegistry.from_directory(blocks_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails
            Template.from_yaml(
                tmpl_path, block_registry=registry)


class TestProjectConfigBlocksDir:
    def test_blocks_dir_default_path(self, tmp_path: Path):
        (tmp_path / "flywheel.yaml").write_text(
            "foundry_dir: foundry\n")
        (tmp_path / "foundry").mkdir()
        config = load_project_config(tmp_path)
        assert config.blocks_dir == (
            tmp_path / "workforce" / "blocks")

    def test_load_block_registry_empty_when_dir_missing(
            self, tmp_path: Path):
        (tmp_path / "flywheel.yaml").write_text(
            "foundry_dir: foundry\n")
        (tmp_path / "foundry").mkdir()
        config = load_project_config(tmp_path)
        registry = config.load_block_registry()
        assert registry.names() == []

    def test_load_block_registry_loads_workforce_blocks(
            self, tmp_path: Path):
        (tmp_path / "flywheel.yaml").write_text(
            "foundry_dir: foundry\n")
        (tmp_path / "foundry").mkdir()
        wf = tmp_path / "workforce" / "blocks"
        wf.mkdir(parents=True)
        _write_yaml(wf / "train.yaml", CONTAINER_BLOCK)

        config = load_project_config(tmp_path)
        registry = config.load_block_registry()
        assert registry.names() == ["train"]
