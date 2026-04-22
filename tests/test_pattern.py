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
    BlockInstance,
    ContinuousTrigger,
    EveryNExecutionsTrigger,
    OnEventTrigger,
    OnRequestTrigger,
    OnToolTrigger,
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
    cardinality: 1
    trigger:
      kind: continuous
    inputs: [predictor, mechanics_summary]
    outputs: [game_log]
    overrides:
      model: claude-sonnet-4-6
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
        assert play.cardinality == 1
        assert isinstance(play.trigger, ContinuousTrigger)
        assert play.inputs == ["predictor", "mechanics_summary"]
        assert play.outputs == ["game_log"]
        assert play.overrides == {
            "model": "claude-sonnet-4-6",
            "mcp_servers": "arc",
        }

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
        assert brainstorm.overrides == {}

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


class TestPauseField:
    """``pause`` on ``every_n_executions`` — parser + validation."""

    _INSTANCES_YAML = """\
instances:
  play:
    block: play
    trigger: {{kind: continuous}}
    prompt: p.md
  brainstorm:
    block: brainstorm
    trigger:
      kind: every_n_executions
      of_block: play
      n: 20
      pause: {pause_list}
    prompt: b.md
"""

    def test_default_empty_tuple(self):
        trigger = EveryNExecutionsTrigger(of_block="play", n=5)
        assert trigger.pause == ()

    def test_parses_list_of_names(self, tmp_path: Path):
        path = _write(
            tmp_path, "p",
            self._INSTANCES_YAML.format(pause_list="[play]"))
        pattern = Pattern.from_yaml(path)
        brainstorm = next(
            i for i in pattern.instances if i.name == "brainstorm")
        assert brainstorm.trigger.pause == ("play",)

    def test_empty_list_parses_as_empty_tuple(
            self, tmp_path: Path):
        path = _write(
            tmp_path, "p",
            self._INSTANCES_YAML.format(pause_list="[]"))
        pattern = Pattern.from_yaml(path)
        brainstorm = next(
            i for i in pattern.instances if i.name == "brainstorm")
        assert brainstorm.trigger.pause == ()

    def test_rejects_non_list(self, tmp_path: Path):
        path = _write(
            tmp_path, "bad",
            self._INSTANCES_YAML.format(pause_list="'play'"))
        with pytest.raises(
                ValueError, match="'pause' must be a list"):
            Pattern.from_yaml(path)

    def test_rejects_empty_entry(self, tmp_path: Path):
        path = _write(
            tmp_path, "bad",
            self._INSTANCES_YAML.format(pause_list="['']"))
        with pytest.raises(
                ValueError,
                match="'pause' entries must be non-empty"):
            Pattern.from_yaml(path)

    def test_rejects_unknown_instance(self, tmp_path: Path):
        path = _write(
            tmp_path, "bad",
            self._INSTANCES_YAML.format(
                pause_list="[no_such_instance]"))
        with pytest.raises(
                ValueError,
                match="'pause' references unknown instance"):
            Pattern.from_yaml(path)

    def test_rejects_self_reference(self, tmp_path: Path):
        path = _write(
            tmp_path, "bad",
            self._INSTANCES_YAML.format(pause_list="[brainstorm]"))
        with pytest.raises(
                ValueError,
                match="'pause' cannot reference the cohort itself"):
            Pattern.from_yaml(path)

    def test_validation_applies_to_roles_grammar(
            self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  play:
    prompt: p.md
    trigger: {kind: continuous}
  cohort:
    prompt: c.md
    trigger:
      kind: every_n_executions
      of_block: play
      n: 5
      pause: [no_such_role]
""")
        with pytest.raises(
                ValueError,
                match="'pause' references unknown instance"):
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


class TestOverridesField:
    """The free-form ``overrides:`` map carries battery-specific knobs.

    Top-level keys that used to be typed (``model``,
    ``mcp_servers``, ``allowed_tools``, ``max_turns``,
    ``total_timeout``) are no longer accepted directly on a
    role / instance; the parser rejects them with a pointed
    error so a stale YAML fails loudly instead of silently
    losing the value.  Their replacement is the free-form
    ``overrides:`` mapping the runner forwards verbatim into
    each launch's ``overrides`` dict.
    """

    @pytest.mark.parametrize("legacy_key,legacy_value", [
        ("model", "claude-sonnet-4-6"),
        ("mcp_servers", "arc"),
        ("allowed_tools", "Read,Write"),
        ("max_turns", 10),
        ("total_timeout", 60),
    ])
    def test_role_rejects_legacy_battery_key(
            self, tmp_path: Path,
            legacy_key: str, legacy_value: object):
        path = _write(tmp_path, "bad", f"""\
roles:
  r:
    prompt: p.md
    trigger:
      kind: continuous
    {legacy_key}: {legacy_value!r}
""")
        with pytest.raises(
                ValueError,
                match=r"no longer supported.*overrides:"):
            Pattern.from_yaml(path)

    def test_instance_rejects_legacy_battery_key(
            self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
instances:
  play:
    block: play
    trigger: {kind: continuous}
    prompt: p.md
    model: claude-sonnet-4-6
""")
        with pytest.raises(
                ValueError,
                match=r"no longer supported.*overrides:"):
            Pattern.from_yaml(path)

    def test_role_overrides_round_trip(self, tmp_path: Path):
        path = _write(tmp_path, "p", """\
roles:
  play:
    prompt: p.md
    trigger:
      kind: continuous
    overrides:
      model: claude-sonnet-4-6
      max_turns: 12
      custom_executor_knob: hello
""")
        pattern = Pattern.from_yaml(path)
        assert pattern.roles[0].overrides == {
            "model": "claude-sonnet-4-6",
            "max_turns": 12,
            "custom_executor_knob": "hello",
        }

    def test_instance_overrides_round_trip(self, tmp_path: Path):
        path = _write(tmp_path, "p", """\
instances:
  play:
    block: play
    trigger: {kind: continuous}
    prompt: p.md
    overrides:
      model: claude-sonnet-4-6
      whatever: 123
""")
        pattern = Pattern.from_yaml(path)
        inst = pattern.instances[0]
        assert inst.overrides == {
            "model": "claude-sonnet-4-6",
            "whatever": 123,
        }

    @pytest.mark.parametrize("reserved_key,reserved_value", [
        ("prompt", "'other.md'"),
        ("prompt_substitutions", "{foo: bar}"),
        ("predecessor_id", "'exec_123'"),
        ("extra_env", "{FOO: bar}"),
        ("extra_mounts", "[]"),
    ])
    def test_overrides_rejects_runner_owned_keys(
            self, tmp_path: Path,
            reserved_key: str, reserved_value: str):
        path = _write(tmp_path, "bad", f"""\
roles:
  r:
    prompt: p.md
    trigger:
      kind: continuous
    overrides:
      {reserved_key}: {reserved_value}
""")
        with pytest.raises(
                ValueError,
                match=rf"'overrides\.{reserved_key}' is reserved"):
            Pattern.from_yaml(path)

    def test_overrides_must_be_mapping(self, tmp_path: Path):
        path = _write(tmp_path, "bad", """\
roles:
  r:
    prompt: p.md
    trigger:
      kind: continuous
    overrides: not-a-mapping
""")
        with pytest.raises(
                ValueError, match="'overrides' must be a mapping"):
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
                ValueError,
                match="must declare either 'instances'",
        ):
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


class TestInstancesGrammar:
    """Patterns may declare topology as ``instances:`` referencing
    blocks, with ``on_tool`` triggers for reactive dispatch."""

    def test_instances_parses_with_continuous_trigger(
        self, tmp_path: Path,
    ):
        body = (
            "instances:\n"
            "  play:\n"
            "    block: play\n"
            "    trigger: {kind: continuous}\n"
            "    cardinality: 1\n"
            "    prompt: workforce/prompts/arc.md\n"
        )
        path = _write(tmp_path, "inst", body)
        pattern = Pattern.from_yaml(path)
        assert len(pattern.instances) == 1
        assert len(pattern.roles) == 0
        inst = pattern.instances[0]
        assert inst.name == "play"
        assert inst.block == "play"
        assert inst.cardinality == 1
        assert isinstance(
            inst.trigger, ContinuousTrigger)
        assert inst.prompt == "workforce/prompts/arc.md"

    def test_on_tool_trigger_parses(self, tmp_path: Path):
        body = (
            "instances:\n"
            "  play:\n"
            "    block: play\n"
            "    trigger: {kind: continuous}\n"
            "    prompt: p.md\n"
            "  execute_action:\n"
            "    block: ExecuteAction\n"
            "    trigger:\n"
            "      kind: on_tool\n"
            "      instance: play\n"
            "      tool: mcp__arc__take_action\n"
        )
        path = _write(tmp_path, "ontool", body)
        pattern = Pattern.from_yaml(path)
        ea = next(
            i for i in pattern.instances
            if i.name == "execute_action")
        assert isinstance(ea.trigger, OnToolTrigger)
        assert ea.trigger.instance == "play"
        assert (
            ea.trigger.tool == "mcp__arc__take_action")

    def test_on_tool_unknown_instance_raises(
        self, tmp_path: Path,
    ):
        body = (
            "instances:\n"
            "  play:\n"
            "    block: play\n"
            "    trigger: {kind: continuous}\n"
            "    prompt: p.md\n"
            "  ea:\n"
            "    block: ExecuteAction\n"
            "    trigger:\n"
            "      kind: on_tool\n"
            "      instance: nope\n"
            "      tool: mcp__arc__take_action\n"
        )
        path = _write(tmp_path, "bad", body)
        with pytest.raises(
            ValueError,
            match=(
                "on_tool trigger references unknown "
                "instance 'nope'"),
        ):
            Pattern.from_yaml(path)

    def test_both_roles_and_instances_raises(
        self, tmp_path: Path,
    ):
        body = (
            "roles:\n"
            "  a: {prompt: p.md, trigger: {kind: continuous}}\n"
            "instances:\n"
            "  a:\n"
            "    block: a\n"
            "    trigger: {kind: continuous}\n"
        )
        path = _write(tmp_path, "bad", body)
        with pytest.raises(
            ValueError, match="not both",
        ):
            Pattern.from_yaml(path)

    def test_instance_missing_block_raises(
        self, tmp_path: Path,
    ):
        body = (
            "instances:\n"
            "  play:\n"
            "    trigger: {kind: continuous}\n"
        )
        path = _write(tmp_path, "bad", body)
        with pytest.raises(
            ValueError,
            match="'block' must be a non-empty string",
        ):
            Pattern.from_yaml(path)
