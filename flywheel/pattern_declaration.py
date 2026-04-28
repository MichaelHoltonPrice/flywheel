"""Declarative pattern schema.

Patterns describe ordered cohorts of block executions.  They do
not describe how artifacts are staged or committed; each member is
executed through the canonical block execution path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from flywheel.bindings import ArtifactIdBinding
from flywheel.pattern_lanes import DEFAULT_LANE
from flywheel.pattern_params import ParamType, coerce_param_value
from flywheel.validation import validate_name


@dataclass(frozen=True)
class PriorOutputBinding:
    """Bind an input slot to a prior member's output."""

    from_step: str
    member: str
    output: str


InputBinding = ArtifactIdBinding | PriorOutputBinding
MinSuccesses = Literal["all"] | int


@dataclass(frozen=True)
class PatternParam:
    """Operator-supplied pattern input declared by a pattern."""

    name: str
    type: ParamType
    default: str | int | float | bool | None = None


@dataclass(frozen=True)
class PatternFixture:
    """Project source materialized into one artifact per lane."""

    name: str
    source: str


@dataclass(frozen=True)
class PatternMember:
    """One block execution request within a cohort."""

    name: str
    block: str
    lane: str = DEFAULT_LANE
    inputs: dict[str, InputBinding] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PatternCohort:
    """A semantic group of peer member executions."""

    min_successes: MinSuccesses
    members: list[PatternMember]


@dataclass(frozen=True)
class PatternStep:
    """One ordered pattern step."""

    name: str
    cohort: PatternCohort


@dataclass(frozen=True)
class PatternForEach:
    """Run a body once for each generated lane."""

    lanes: list[str]
    body: list[PatternNode]


@dataclass(frozen=True)
class PatternUse:
    """Run a locally-declared sub-pattern in the current lane."""

    pattern: str
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RunUntilContinuation:
    """Budget for a termination reason that continues a loop."""

    max: int | str


@dataclass(frozen=True)
class PatternAfterEvery:
    """Run a body after a counted continuation reason."""

    reason: str
    count: int | str
    body: list[PatternNode]


@dataclass(frozen=True)
class PatternRunUntil:
    """Repeatedly run one block until a termination reason stops it."""

    name: str
    block: str
    inputs: dict[str, InputBinding] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    continue_on: dict[str, RunUntilContinuation] = field(default_factory=dict)
    stop_on: list[str] = field(default_factory=list)
    fail_on: list[str] = field(default_factory=list)
    after_every: list[PatternAfterEvery] = field(default_factory=list)


PatternNode = PatternStep | PatternForEach | PatternUse | PatternRunUntil


@dataclass(frozen=True)
class PatternDeclaration:
    """A parsed pattern declaration."""

    name: str
    params: dict[str, PatternParam]
    lanes: list[str]
    fixtures: dict[str, PatternFixture]
    steps: list[PatternStep]
    body: list[PatternNode] = field(default_factory=list)
    patterns: dict[str, PatternDeclaration] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> PatternDeclaration:
        """Load a pattern declaration from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Pattern {path} must contain a YAML mapping"
            )
        return parse_pattern_declaration(data, source=str(path))


def parse_pattern_declaration(
    data: dict,
    *,
    source: str = "<pattern>",
) -> PatternDeclaration:
    """Parse and validate a pattern declaration mapping."""
    unknown = set(data) - {
        "name", "params", "lanes", "fixtures", "steps", "do", "patterns",
    }
    if unknown:
        raise ValueError(
            f"Pattern {source}: unknown top-level key(s) "
            f"{sorted(unknown)!r}"
        )
    name = data.get("name")
    if not isinstance(name, str):
        raise ValueError(f"Pattern {source}: missing string 'name'")
    validate_name(name, "Pattern")
    params = _parse_params(data.get("params", {}), pattern_name=name)
    fixtures = _parse_fixtures(data.get("fixtures", {}), pattern_name=name)
    steps: list[PatternStep] = []
    body: list[PatternNode] = []
    if "steps" in data and "do" in data:
        raise ValueError(
            f"Pattern {name!r}: declare either 'steps' or 'do', not both"
        )
    patterns = _parse_local_patterns(
        data.get("patterns", {}), parent_name=name, source=source)
    if "steps" in data:
        lanes = _parse_lanes(data.get("lanes"), pattern_name=name)
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(
                f"Pattern {name!r}: 'steps' must be a non-empty list"
            )
        step_names: set[str] = set()
        for raw_step in raw_steps:
            step = _parse_step(raw_step, pattern_name=name, lanes=lanes)
            if step.name in step_names:
                raise ValueError(
                    f"Pattern {name!r}: duplicate step name "
                    f"{step.name!r}"
                )
            step_names.add(step.name)
            steps.append(step)
    elif "do" in data:
        body = _parse_body(data.get("do"), pattern_name=name)
        declared_lanes = _collect_body_lanes(body)
        lanes = (
            _parse_lanes(data.get("lanes"), pattern_name=name)
            if "lanes" in data else declared_lanes or [DEFAULT_LANE]
        )
        if declared_lanes and _body_has_root_non_foreach(body):
            raise ValueError(
                f"Pattern {name!r}: a root 'do' body that declares "
                "foreach lanes must not mix root non-foreach nodes; "
                "put non-foreach work inside the foreach body"
            )
    else:
        raise ValueError(
            f"Pattern {name!r}: declare non-empty 'steps' or 'do'"
        )
    return PatternDeclaration(
        name=name,
        params=params,
        lanes=lanes,
        fixtures=fixtures,
        steps=steps,
        body=body,
        patterns=patterns,
    )


def _parse_params(
    raw: object,
    *,
    pattern_name: str,
) -> dict[str, PatternParam]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern {pattern_name!r}: 'params' must be a mapping"
        )
    params: dict[str, PatternParam] = {}
    for name, spec in raw.items():
        if not isinstance(name, str):
            raise ValueError(
                f"Pattern {pattern_name!r}: param names must be strings"
            )
        validate_name(name, "Pattern param")
        if not isinstance(spec, dict):
            raise ValueError(
                f"Pattern {pattern_name!r}: param {name!r} must be "
                "a mapping"
            )
        unknown = set(spec) - {"type", "default"}
        if unknown:
            raise ValueError(
                f"Pattern {pattern_name!r}: param {name!r} has "
                f"unknown key(s) {sorted(unknown)!r}"
            )
        param_type = spec.get("type")
        if param_type not in ("string", "int", "float", "bool"):
            raise ValueError(
                f"Pattern {pattern_name!r}: param {name!r} type "
                "must be one of string, int, float, bool"
            )
        default = spec.get("default")
        if default is not None:
            default = coerce_param_value(
                pattern_name=pattern_name,
                name=name,
                value=default,
                param_type=param_type,
                source="default",
            )
        params[name] = PatternParam(
            name=name,
            type=param_type,
            default=default,
        )
    return params


def _parse_lanes(raw: object, *, pattern_name: str) -> list[str]:
    if raw is None:
        return [DEFAULT_LANE]
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"Pattern {pattern_name!r}: 'lanes' must be a "
            "non-empty list"
        )
    lanes: list[str] = []
    seen: set[str] = set()
    for lane in raw:
        if not isinstance(lane, str):
            raise ValueError(
                f"Pattern {pattern_name!r}: lane names must be strings"
            )
        validate_name(lane, "Pattern lane")
        if lane in seen:
            raise ValueError(
                f"Pattern {pattern_name!r}: duplicate lane {lane!r}"
            )
        seen.add(lane)
        lanes.append(lane)
    return lanes


def _parse_fixtures(
    raw: object,
    *,
    pattern_name: str,
) -> dict[str, PatternFixture]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern {pattern_name!r}: 'fixtures' must be a mapping"
        )
    fixtures: dict[str, PatternFixture] = {}
    for name, source in raw.items():
        if not isinstance(name, str):
            raise ValueError(
                f"Pattern {pattern_name!r}: fixture names must be strings"
            )
        validate_name(name, "Fixture artifact")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"Pattern {pattern_name!r}: fixture {name!r} source "
                "must be a non-empty string"
            )
        fixtures[name] = PatternFixture(name=name, source=source)
    return fixtures


def _parse_local_patterns(
    raw: object,
    *,
    parent_name: str,
    source: str,
) -> dict[str, PatternDeclaration]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern {parent_name!r}: 'patterns' must be a mapping"
        )
    patterns: dict[str, PatternDeclaration] = {}
    for name, spec in raw.items():
        if not isinstance(name, str):
            raise ValueError(
                f"Pattern {parent_name!r}: local pattern names must "
                "be strings"
            )
        validate_name(name, "Local pattern")
        if not isinstance(spec, dict):
            raise ValueError(
                f"Pattern {parent_name!r}: local pattern {name!r} "
                "must be a mapping"
            )
        merged = {"name": name, **spec}
        if "fixtures" in merged:
            raise ValueError(
                f"Pattern {parent_name!r}: local pattern {name!r} "
                "must not declare fixtures; fixtures belong to the "
                "owning run"
            )
        patterns[name] = parse_pattern_declaration(
            merged,
            source=f"{source} patterns[{name!r}]",
        )
    return patterns


def _parse_body(raw: object, *, pattern_name: str) -> list[PatternNode]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"Pattern {pattern_name!r}: 'do' must be a non-empty list"
        )
    return [
        _parse_body_node(item, pattern_name=pattern_name, index=index)
        for index, item in enumerate(raw)
    ]


def _parse_body_node(
    raw: object,
    *,
    pattern_name: str,
    index: int,
) -> PatternNode:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern {pattern_name!r}: do[{index}] must be a mapping"
        )
    present = [key for key in ("foreach", "use", "run_until", "cohort")
               if key in raw]
    if len(present) != 1:
        raise ValueError(
            f"Pattern {pattern_name!r}: do[{index}] must declare exactly "
            "one of foreach, use, run_until, or cohort"
        )
    if "cohort" in raw:
        return _parse_body_cohort_node(
            raw,
            pattern_name=pattern_name,
            lanes=[DEFAULT_LANE],
            index=index,
        )
    if "foreach" in raw:
        unknown = set(raw) - {"foreach", "do"}
        if unknown:
            raise ValueError(
                f"Pattern {pattern_name!r}: foreach node has unknown "
                f"key(s) {sorted(unknown)!r}"
            )
        return _parse_foreach_node(raw, pattern_name=pattern_name)
    if "use" in raw:
        unknown = set(raw) - {"use", "with"}
        if unknown:
            raise ValueError(
                f"Pattern {pattern_name!r}: use node has unknown "
                f"key(s) {sorted(unknown)!r}"
            )
        target = raw["use"]
        if not isinstance(target, str):
            raise ValueError(
                f"Pattern {pattern_name!r}: use target must be a string"
            )
        validate_name(target, "Local pattern")
        raw_params = raw.get("with", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            raise ValueError(
                f"Pattern {pattern_name!r}: use {target!r} 'with' "
                "must be a mapping"
            )
        params: dict[str, object] = {}
        for key, value in raw_params.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Pattern {pattern_name!r}: use {target!r} "
                    "param names must be strings"
                )
            validate_name(key, "Pattern param")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"Pattern {pattern_name!r}: use {target!r} "
                    f"param {key!r} must be a scalar"
                )
            params[key] = value
        return PatternUse(pattern=target, params=params)
    return _parse_run_until_node(raw["run_until"], pattern_name=pattern_name)


def _parse_body_cohort_node(
    raw: dict,
    *,
    pattern_name: str,
    lanes: list[str],
    index: int,
) -> PatternStep:
    if "name" in raw:
        return _parse_step(raw, pattern_name=pattern_name, lanes=lanes)
    cohort = raw.get("cohort")
    name = f"cohort_{index + 1}"
    if isinstance(cohort, dict):
        members = cohort.get("members")
        if isinstance(members, list) and len(members) == 1:
            member = members[0]
            if isinstance(member, dict) and isinstance(member.get("name"), str):
                name = member["name"]
    return _parse_step(
        {"name": name, "cohort": cohort},
        pattern_name=pattern_name,
        lanes=lanes,
    )


def _parse_foreach_node(
    raw: dict,
    *,
    pattern_name: str,
) -> PatternForEach:
    spec = raw["foreach"]
    if not isinstance(spec, dict):
        raise ValueError(
            f"Pattern {pattern_name!r}: foreach must be a mapping"
        )
    unknown = set(spec) - {"count"}
    if unknown:
        raise ValueError(
            f"Pattern {pattern_name!r}: foreach has unknown key(s) "
            f"{sorted(unknown)!r}"
        )
    count = spec.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(
            f"Pattern {pattern_name!r}: foreach.count must be a "
            "positive integer"
        )
    lanes = [f"lane_{index}" for index in range(count)]
    return PatternForEach(
        lanes=lanes,
        body=_parse_body(raw.get("do"), pattern_name=pattern_name),
    )


def _parse_run_until_node(
    raw: object,
    *,
    pattern_name: str,
) -> PatternRunUntil:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern {pattern_name!r}: run_until must be a mapping"
        )
    unknown = set(raw) - {
        "name", "block", "inputs", "args", "env", "continue_on", "stop_on",
        "fail_on", "after_every",
    }
    if unknown:
        raise ValueError(
            f"Pattern {pattern_name!r}: run_until has unknown key(s) "
            f"{sorted(unknown)!r}"
        )
    block = raw.get("block")
    if not isinstance(block, str):
        raise ValueError(
            f"Pattern {pattern_name!r}: run_until missing string 'block'"
        )
    validate_name(block, "Block")
    name = raw.get("name", block)
    if not isinstance(name, str):
        raise ValueError(
            f"Pattern {pattern_name!r}: run_until name must be a string"
        )
    validate_name(name, "run_until")
    args = raw.get("args", [])
    if not isinstance(args, list) or not all(
        isinstance(item, str) for item in args
    ):
        raise ValueError(
            f"Pattern {pattern_name!r}: run_until 'args' must be a "
            "list of strings"
        )
    continue_on = _parse_continue_on(
        raw.get("continue_on"), pattern_name=pattern_name, node_name=name)
    stop_on = _parse_stop_on(
        raw.get("stop_on"), pattern_name=pattern_name, node_name=name)
    fail_on = _parse_fail_on(
        raw.get("fail_on", []), pattern_name=pattern_name, node_name=name)
    if not continue_on:
        raise ValueError(
            f"Pattern {pattern_name!r}: run_until {name!r} must "
            "declare at least one continue_on reason"
        )
    if not stop_on:
        raise ValueError(
            f"Pattern {pattern_name!r}: run_until {name!r} must "
            "declare at least one stop_on reason"
        )
    return PatternRunUntil(
        name=name,
        block=block,
        inputs=_parse_inputs(raw.get("inputs", {}), member_name=name),
        args=list(args),
        env=_parse_env(raw.get("env", {}), owner=(
            f"Pattern {pattern_name!r} run_until {name!r}")),
        continue_on=continue_on,
        stop_on=stop_on,
        fail_on=fail_on,
        after_every=_parse_after_every(
            raw.get("after_every", []),
            pattern_name=pattern_name,
            node_name=name,
        ),
    )


def _parse_after_every(
    raw: object,
    *,
    pattern_name: str,
    node_name: str,
) -> list[PatternAfterEvery]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            f"Pattern {pattern_name!r}: run_until {node_name!r} "
            "'after_every' must be a list"
        )
    parsed: list[PatternAfterEvery] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"Pattern {pattern_name!r}: run_until {node_name!r} "
                f"after_every[{index}] must be a mapping"
            )
        unknown = set(item) - {"reason", "count", "do"}
        if unknown:
            raise ValueError(
                f"Pattern {pattern_name!r}: run_until {node_name!r} "
                f"after_every[{index}] has unknown key(s) "
                f"{sorted(unknown)!r}"
            )
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(
                f"Pattern {pattern_name!r}: run_until {node_name!r} "
                f"after_every[{index}].reason must be a non-empty string"
            )
        count = item.get("count")
        if isinstance(count, int) and not isinstance(count, bool):
            if count < 1:
                raise ValueError(
                    f"Pattern {pattern_name!r}: run_until {node_name!r} "
                    f"after_every[{index}].count must be positive"
                )
            count_value: int | str = count
        elif isinstance(count, str) and count:
            count_value = count
        else:
            raise ValueError(
                f"Pattern {pattern_name!r}: run_until {node_name!r} "
                f"after_every[{index}].count must be an int or string"
            )
        parsed.append(PatternAfterEvery(
            reason=reason,
            count=count_value,
            body=_parse_body(item.get("do"), pattern_name=pattern_name),
        ))
    return parsed


def _parse_continue_on(
    raw: object,
    *,
    pattern_name: str,
    node_name: str,
) -> dict[str, RunUntilContinuation]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"Pattern {pattern_name!r}: run_until {node_name!r} "
            "'continue_on' must be a non-empty mapping"
        )
    parsed: dict[str, RunUntilContinuation] = {}
    for reason, spec in raw.items():
        if not isinstance(reason, str) or not reason:
            raise ValueError(
                f"Pattern {pattern_name!r}: run_until {node_name!r} "
                "continue_on reasons must be non-empty strings"
            )
        if isinstance(spec, int) and not isinstance(spec, bool):
            max_value: int | str = spec
        elif isinstance(spec, str):
            max_value = spec
        elif isinstance(spec, dict) and set(spec) == {"max"}:
            value = spec["max"]
            if not isinstance(value, (int, str)) or isinstance(value, bool):
                raise ValueError(
                    f"Pattern {pattern_name!r}: run_until {node_name!r} "
                    f"continue_on[{reason!r}].max must be an int or string"
                )
            max_value = value
        else:
            raise ValueError(
                f"Pattern {pattern_name!r}: run_until {node_name!r} "
                f"continue_on[{reason!r}] must be an int, string, "
                "or {max: ...}"
            )
        parsed[reason] = RunUntilContinuation(max=max_value)
    return parsed


def _parse_stop_on(
    raw: object,
    *,
    pattern_name: str,
    node_name: str,
) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"Pattern {pattern_name!r}: run_until {node_name!r} "
            "'stop_on' must be a non-empty list"
        )
    reasons: list[str] = []
    seen: set[str] = set()
    for reason in raw:
        if not isinstance(reason, str) or not reason:
            raise ValueError(
                f"Pattern {pattern_name!r}: run_until {node_name!r} "
                "stop_on reasons must be non-empty strings"
            )
        if reason in seen:
            raise ValueError(
                f"Pattern {pattern_name!r}: run_until {node_name!r} "
                f"duplicate stop_on reason {reason!r}"
            )
        seen.add(reason)
        reasons.append(reason)
    return reasons


def _parse_fail_on(
    raw: object,
    *,
    pattern_name: str,
    node_name: str,
) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            f"Pattern {pattern_name!r}: run_until {node_name!r} "
            "'fail_on' must be a list"
        )
    reasons: list[str] = []
    seen: set[str] = set()
    for reason in raw:
        if not isinstance(reason, str) or not reason:
            raise ValueError(
                f"Pattern {pattern_name!r}: run_until {node_name!r} "
                "fail_on reasons must be non-empty strings"
            )
        if reason in seen:
            raise ValueError(
                f"Pattern {pattern_name!r}: run_until {node_name!r} "
                f"duplicate fail_on reason {reason!r}"
            )
        seen.add(reason)
        reasons.append(reason)
    return reasons


def _collect_body_lanes(nodes: list[PatternNode]) -> list[str]:
    lanes: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        if isinstance(node, PatternForEach):
            for lane in node.lanes:
                if lane not in seen:
                    seen.add(lane)
                    lanes.append(lane)
    return lanes


def _body_has_root_non_foreach(nodes: list[PatternNode]) -> bool:
    return any(not isinstance(node, PatternForEach) for node in nodes)


def _parse_step(
    raw: object,
    *,
    pattern_name: str,
    lanes: list[str],
) -> PatternStep:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern {pattern_name!r}: step entries must be mappings"
        )
    unknown = set(raw) - {"name", "cohort"}
    if unknown:
        raise ValueError(
            f"Pattern {pattern_name!r}: step has unknown key(s) "
            f"{sorted(unknown)!r}"
        )
    name = raw.get("name")
    if not isinstance(name, str):
        raise ValueError(
            f"Pattern {pattern_name!r}: step missing string 'name'"
        )
    validate_name(name, "Pattern step")
    cohort = _parse_cohort(raw.get("cohort"), step_name=name, lanes=lanes)
    return PatternStep(name=name, cohort=cohort)


def _parse_cohort(
    raw: object,
    *,
    step_name: str,
    lanes: list[str],
) -> PatternCohort:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern step {step_name!r}: 'cohort' must be a mapping"
        )
    unknown = set(raw) - {
        "min_successes", "members", "foreach", "block", "inputs", "args",
        "env",
    }
    if unknown:
        raise ValueError(
            f"Pattern step {step_name!r}: cohort has unknown key(s) "
            f"{sorted(unknown)!r}"
        )
    min_successes = raw.get("min_successes", "all")
    min_successes = _parse_min_successes(
        min_successes, step_name=step_name)
    raw_members = raw.get("members")
    if raw_members is not None and "foreach" in raw:
        raise ValueError(
            f"Pattern step {step_name!r}: cohort must not declare "
            "both members and foreach"
        )
    if "foreach" in raw:
        return _parse_foreach_cohort(raw, step_name=step_name, lanes=lanes)
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError(
            f"Pattern step {step_name!r}: members must be a "
            "non-empty list"
        )
    members: list[PatternMember] = []
    member_names: set[str] = set()
    for raw_member in raw_members:
        member = _parse_member(
            raw_member, step_name=step_name, lanes=lanes)
        if member.name in member_names:
            raise ValueError(
                f"Pattern step {step_name!r}: duplicate member "
                f"name {member.name!r}"
            )
        member_names.add(member.name)
        members.append(member)
    return PatternCohort(
        min_successes=min_successes,
        members=members,
    )


def _parse_foreach_cohort(
    raw: dict,
    *,
    step_name: str,
    lanes: list[str],
) -> PatternCohort:
    target = raw.get("foreach")
    if target != "lanes":
        raise ValueError(
            f"Pattern step {step_name!r}: foreach must be 'lanes'"
        )
    block = raw.get("block")
    if not isinstance(block, str):
        raise ValueError(
            f"Pattern step {step_name!r}: foreach cohort missing "
            "string 'block'"
        )
    validate_name(block, "Block")
    inputs = _parse_inputs(raw.get("inputs", {}), member_name=block)
    args = raw.get("args", [])
    if not isinstance(args, list) or not all(
        isinstance(item, str) for item in args
    ):
        raise ValueError(
            f"Pattern step {step_name!r}: foreach 'args' must be a "
            "list of strings"
        )
    env = _parse_env(raw.get("env", {}), owner=f"Pattern step {step_name!r}")
    return PatternCohort(
        min_successes=_parse_min_successes(
            raw.get("min_successes", "all"),
            step_name=step_name,
        ),
        members=[
            PatternMember(
                name=lane,
                block=block,
                lane=lane,
                inputs=dict(inputs),
                args=list(args),
                env=dict(env),
            )
            for lane in lanes
        ],
    )


def _parse_min_successes(
    value: object,
    *,
    step_name: str,
) -> MinSuccesses:
    if value == "all" or value == 1:
        return value
    raise ValueError(
        f"Pattern step {step_name!r}: min_successes must be "
        "'all' or 1"
    )


def _parse_member(
    raw: object,
    *,
    step_name: str,
    lanes: list[str],
) -> PatternMember:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern step {step_name!r}: member entries must be "
            "mappings"
        )
    unknown = set(raw) - {"name", "block", "lane", "inputs", "args", "env"}
    if unknown:
        raise ValueError(
            f"Pattern step {step_name!r}: member has unknown "
            f"key(s) {sorted(unknown)!r}"
        )
    name = raw.get("name")
    block = raw.get("block")
    if not isinstance(name, str):
        raise ValueError(
            f"Pattern step {step_name!r}: member missing string "
            "'name'"
        )
    if not isinstance(block, str):
        raise ValueError(
            f"Pattern member {name!r}: missing string 'block'"
        )
    validate_name(name, "Pattern member")
    validate_name(block, "Block")
    lane = raw.get("lane", DEFAULT_LANE)
    if not isinstance(lane, str):
        raise ValueError(
            f"Pattern member {name!r}: lane must be a string"
        )
    validate_name(lane, "Pattern lane")
    if lane not in lanes:
        raise ValueError(
            f"Pattern member {name!r}: lane {lane!r} is not declared"
        )
    args = raw.get("args", [])
    if not isinstance(args, list) or not all(
        isinstance(item, str) for item in args
    ):
        raise ValueError(
            f"Pattern member {name!r}: 'args' must be a list of "
            "strings"
        )
    inputs = _parse_inputs(raw.get("inputs", {}), member_name=name)
    env = _parse_env(raw.get("env", {}), owner=f"Pattern member {name!r}")
    return PatternMember(
        name=name,
        block=block,
        lane=lane,
        inputs=inputs,
        args=list(args),
        env=env,
    )


def _parse_env(raw: object, *, owner: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{owner}: 'env' must be a mapping")
    parsed: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{owner}: env keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError(f"{owner}: env value for {key!r} must be a string")
        parsed[key] = value
    return parsed


def _parse_inputs(
    raw: object,
    *,
    member_name: str,
) -> dict[str, InputBinding]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern member {member_name!r}: 'inputs' must be a "
            "mapping"
        )
    parsed: dict[str, InputBinding] = {}
    for slot, binding in raw.items():
        if not isinstance(slot, str):
            raise ValueError(
                f"Pattern member {member_name!r}: input slot names "
                "must be strings"
            )
        validate_name(slot, "Input slot")
        if not isinstance(binding, dict):
            raise ValueError(
                f"Pattern member {member_name!r}: input {slot!r} "
                "must be a mapping"
            )
        keys = set(binding)
        if keys == {"artifact_id"}:
            artifact_id = binding["artifact_id"]
            if not isinstance(artifact_id, str):
                raise ValueError(
                    f"Pattern member {member_name!r}: input "
                    f"{slot!r} artifact_id must be a string"
                )
            parsed[slot] = ArtifactIdBinding(artifact_id=artifact_id)
            continue
        if keys == {"from_step", "member", "output"}:
            from_step = binding["from_step"]
            member = binding["member"]
            output = binding["output"]
            if not all(
                isinstance(value, str)
                for value in (from_step, member, output)
            ):
                raise ValueError(
                    f"Pattern member {member_name!r}: input "
                    f"{slot!r} prior-output fields must be strings"
                )
            parsed[slot] = PriorOutputBinding(
                from_step=from_step,
                member=member,
                output=output,
            )
            continue
        raise ValueError(
            f"Pattern member {member_name!r}: input {slot!r} must "
            "declare either artifact_id or from_step/member/output"
        )
    return parsed
