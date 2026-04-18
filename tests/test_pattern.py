"""Tests for :mod:`flywheel.pattern` — parsing and validation only.

The runner that consumes these patterns is exercised separately
in ``test_pattern_runner.py``; here we lock down the load-time
contract: every malformed shape raises a clear ``ValueError``
with the offending pattern + role mentioned, and every valid
shape round-trips into the expected dataclasses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flywheel.blocks.registry import BlockRegistry
from flywheel.pattern import (
    ContinuousTrigger,
    EveryNExecutionsTrigger,
    OnEventTrigger,
    OnRequestTrigger,
    Pattern,
    Role,
    discover_patterns,
)
from flywheel.template import BlockDefinition


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(body)
    return path


def _registry_with(*block_names: str) -> BlockRegistry:
    """Build a synthetic registry holding the requested block names."""
    blocks = {
        n: BlockDefinition(name=n, image="example:latest")
        for n in block_names
    }
    return BlockRegistry(blocks=blocks)


VALID_MULTI_ROLE_YAML = """\
description: One play agent and a periodic brainstorm cohort.
roles:
  play:
    prompt: workforce/prompts/play.md
    model: claude-sonnet-4-6
    cardinality: 1
    trigger:
      kind: continuous
    inputs: [predictor, mechanics_summary]
    outputs: [game_log]
    mcp_servers: arc
  brainstorm:
    prompt: workforce/prompts/brainstorm.md
    cardinality: 6
    trigger:
      kind: every_n_executions
      of_block: take_action
      n: 20
    outputs: [brainstorm_result]
"""


class TestValidPattern:
    def test_round_trip(self, tmp_path: Path):
        path = _write(
            tmp_path, "play_brainstorm", VALID_MULTI_ROLE_YAML)
        registry = _registry_with("take_action")
        pattern = Pattern.from_yaml(
            path, block_registry=registry)

        assert pattern.name == "play_brainstorm"
        assert pattern.description.startswith("One play agent")
        assert [r.name for r in pattern.roles] == [
            "play", "brainstorm"]

    def test_play_role_fields(self, tmp_path: Path):
        path = _write(
            tmp_path, "play_brainstorm", VALID_MULTI_ROLE_YAML)
        pattern = Pattern.from_yaml(
            path, block_registry=_registry_with("take_action"))
        play = pattern.roles[0]

        assert isinstance(play, Role)
        assert play.prompt == "workforce/prompts/play.md"
        assert play.model == "claude-sonnet-4-6"
        assert play.cardinality == 1
        assert isinstance(play.trigger, ContinuousTrigger)
        assert play.inputs == ["predictor", "mechanics_summary"]
        assert play.outputs == ["game_log"]
        assert play.mcp_servers == "arc"

    def test_brainstorm_role_trigger(self, tmp_path: Path):
        path = _write(
            tmp_path, "play_brainstorm", VALID_MULTI_ROLE_YAML)
        pattern = Pattern.from_yaml(
            path, block_registry=_registry_with("take_action"))
        brainstorm = pattern.roles[1]

        assert brainstorm.cardinality == 6
        assert isinstance(brainstorm.trigger, EveryNExecutionsTrigger)
        assert brainstorm.trigger.of_block == "take_action"
        assert brainstorm.trigger.n == 20
        assert brainstorm.model is None
        assert brainstorm.mcp_servers is None

    def test_no_block_registry_skips_trigger_validation(
            self, tmp_path: Path):
        # Useful for unit-testing patterns without a project layout.
        path = _write(
            tmp_path, "play_brainstorm", VALID_MULTI_ROLE_YAML)
        pattern = Pattern.from_yaml(path)
        assert pattern.roles[1].trigger.of_block == "take_action"


class TestTriggerVariants:
    def test_on_request(self, tmp_path: Path):
        path = _write(tmp_path, "fanout", """\
roles:
  brainstorm:
    prompt: p.md
    trigger:
      kind: on_request
      tool: request_brainstorm
""")
        pattern = Pattern.from_yaml(path)
        trigger = pattern.roles[0].trigger
        assert isinstance(trigger, OnRequestTrigger)
        assert trigger.tool == "request_brainstorm"

    def test_on_event(self, tmp_path: Path):
        path = _write(tmp_path, "surprise", """\
roles:
  reactor:
    prompt: p.md
    trigger:
      kind: on_event
      event: surprise
""")
        pattern = Pattern.from_yaml(path)
        trigger = pattern.roles[0].trigger
        assert isinstance(trigger, OnEventTrigger)
        assert trigger.event == "surprise"

    def test_unknown_kind_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  r:
    prompt: p.md
    trigger:
      kind: at_full_moon
""")
        with pytest.raises(ValueError, match="unknown trigger kind"):
            Pattern.from_yaml(path)


class TestEveryNExecutionsValidation:
    def test_unknown_block_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  r:
    prompt: p.md
    trigger:
      kind: every_n_executions
      of_block: not_a_real_block
      n: 5
""")
        with pytest.raises(
                ValueError,
                match="references unknown block"):
            Pattern.from_yaml(
                path, block_registry=_registry_with("take_action"))

    def test_missing_n_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  r:
    prompt: p.md
    trigger:
      kind: every_n_executions
      of_block: take_action
""")
        with pytest.raises(ValueError, match="requires 'n'"):
            Pattern.from_yaml(path)

    def test_zero_n_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  r:
    prompt: p.md
    trigger:
      kind: every_n_executions
      of_block: take_action
      n: 0
""")
        with pytest.raises(ValueError, match="requires 'n'"):
            Pattern.from_yaml(path)

    def test_missing_of_block_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  r:
    prompt: p.md
    trigger:
      kind: every_n_executions
      n: 5
""")
        with pytest.raises(ValueError, match="requires 'of_block'"):
            Pattern.from_yaml(path)


class TestRoleValidation:
    def test_missing_prompt_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  r:
    trigger:
      kind: continuous
""")
        with pytest.raises(ValueError, match="'prompt'"):
            Pattern.from_yaml(path)

    def test_zero_cardinality_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  r:
    prompt: p.md
    cardinality: 0
    trigger:
      kind: continuous
""")
        with pytest.raises(
                ValueError, match="'cardinality' must be a positive"):
            Pattern.from_yaml(path)

    def test_missing_trigger_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  r:
    prompt: p.md
""")
        with pytest.raises(ValueError, match="'trigger' must be"):
            Pattern.from_yaml(path)

    def test_invalid_role_name_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  "with spaces":
    prompt: p.md
    trigger:
      kind: continuous
""")
        with pytest.raises(ValueError, match="Role name"):
            Pattern.from_yaml(path)


class TestPatternValidation:
    def test_empty_file_raises(self, tmp_path: Path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            Pattern.from_yaml(path)

    def test_top_level_list_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", "- not a mapping\n")
        with pytest.raises(ValueError, match="mapping at the top"):
            Pattern.from_yaml(path)

    def test_missing_roles_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", "description: nope\n")
        with pytest.raises(
                ValueError, match="'roles' must be a non-empty"):
            Pattern.from_yaml(path)

    def test_empty_roles_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad", "roles: {}\n")
        with pytest.raises(
                ValueError, match="'roles' must be a non-empty"):
            Pattern.from_yaml(path)

    def test_invalid_pattern_name_raises(self, tmp_path: Path):
        path = _write(tmp_path, "bad name", """\
roles:
  r:
    prompt: p.md
    trigger:
      kind: continuous
""")
        # ``tmp_path / "bad name.yaml"`` exists fine on Windows;
        # validation kicks in on the stem.
        with pytest.raises(ValueError, match="Pattern name"):
            Pattern.from_yaml(path)


class TestMaterializeKeyRejected:
    """A ``materialize`` key in a role spec must be rejected.

    Per-execution rollups are expressed by listing a first-class
    ``incremental`` artifact in :attr:`Role.inputs`; the
    in-process recorder appends to it live and per-mount input
    staging re-reads the latest snapshot on every relaunch.
    Accepting the unsupported key would let the role launch
    with stale or absent inputs, so the loader rejects it.
    """

    def test_materialize_key_is_rejected(self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  r:
    prompt: p.md
    trigger:
      kind: continuous
    materialize:
      game_history: take_action
""")
        with pytest.raises(
                ValueError,
                match="'materialize' is not supported"):
            Pattern.from_yaml(path)


class TestDiscoverPatterns:
    def test_indexes_yaml_files(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text("roles: {}")
        (tmp_path / "b.yaml").write_text("roles: {}")
        (tmp_path / "_draft.yaml").write_text("roles: {}")
        (tmp_path / "readme.txt").write_text("ignore me")

        found = discover_patterns(tmp_path)
        assert set(found) == {"a", "b"}
        assert found["a"].name == "a.yaml"

    def test_missing_directory_returns_empty(self, tmp_path: Path):
        assert discover_patterns(tmp_path / "nope") == {}
