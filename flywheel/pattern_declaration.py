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
class PatternMember:
    """One block execution request within a cohort."""

    name: str
    block: str
    inputs: dict[str, InputBinding] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)


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
class PatternDeclaration:
    """A parsed pattern declaration."""

    name: str
    steps: list[PatternStep]

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
    unknown = set(data) - {"name", "steps"}
    if unknown:
        raise ValueError(
            f"Pattern {source}: unknown top-level key(s) "
            f"{sorted(unknown)!r}"
        )
    name = data.get("name")
    if not isinstance(name, str):
        raise ValueError(f"Pattern {source}: missing string 'name'")
    validate_name(name, "Pattern")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(
            f"Pattern {name!r}: 'steps' must be a non-empty list"
        )

    steps: list[PatternStep] = []
    step_names: set[str] = set()
    for raw_step in raw_steps:
        step = _parse_step(raw_step, pattern_name=name)
        if step.name in step_names:
            raise ValueError(
                f"Pattern {name!r}: duplicate step name "
                f"{step.name!r}"
            )
        step_names.add(step.name)
        steps.append(step)
    return PatternDeclaration(name=name, steps=steps)


def _parse_step(raw: object, *, pattern_name: str) -> PatternStep:
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
    cohort = _parse_cohort(raw.get("cohort"), step_name=name)
    return PatternStep(name=name, cohort=cohort)


def _parse_cohort(raw: object, *, step_name: str) -> PatternCohort:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern step {step_name!r}: 'cohort' must be a mapping"
        )
    unknown = set(raw) - {"min_successes", "members"}
    if unknown:
        raise ValueError(
            f"Pattern step {step_name!r}: cohort has unknown key(s) "
            f"{sorted(unknown)!r}"
        )
    min_successes = raw.get("min_successes", "all")
    if min_successes != "all" and min_successes != 1:
        raise ValueError(
            f"Pattern step {step_name!r}: min_successes must be "
            "'all' or 1"
        )
    raw_members = raw.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError(
            f"Pattern step {step_name!r}: members must be a "
            "non-empty list"
        )
    members: list[PatternMember] = []
    member_names: set[str] = set()
    for raw_member in raw_members:
        member = _parse_member(raw_member, step_name=step_name)
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


def _parse_member(raw: object, *, step_name: str) -> PatternMember:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern step {step_name!r}: member entries must be "
            "mappings"
        )
    unknown = set(raw) - {"name", "block", "inputs", "args"}
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
    args = raw.get("args", [])
    if not isinstance(args, list) or not all(
        isinstance(item, str) for item in args
    ):
        raise ValueError(
            f"Pattern member {name!r}: 'args' must be a list of "
            "strings"
        )
    inputs = _parse_inputs(raw.get("inputs", {}), member_name=name)
    return PatternMember(
        name=name,
        block=block,
        inputs=inputs,
        args=list(args),
    )


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
