from __future__ import annotations

from flywheel.pattern_declaration import (
    PriorOutputBinding,
    parse_pattern_declaration,
)


def _pattern(overrides: dict | None = None) -> dict:
    data = {
        "name": "train_eval",
        "steps": [
            {
                "name": "train",
                "cohort": {
                    "min_successes": "all",
                    "members": [
                        {
                            "name": "train_dueling",
                            "block": "train",
                            "args": ["--subclass", "dueling"],
                        }
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
    }
    if overrides:
        data.update(overrides)
    return data


def test_pattern_yaml_parses_into_declaration_model():
    pattern = parse_pattern_declaration(_pattern())

    assert pattern.name == "train_eval"
    assert [step.name for step in pattern.steps] == ["train", "eval"]
    eval_member = pattern.steps[1].cohort.members[0]
    binding = eval_member.inputs["checkpoint"]
    assert isinstance(binding, PriorOutputBinding)
    assert binding.from_step == "train"
    assert binding.member == "train_dueling"
    assert binding.output == "checkpoint"


def test_unknown_success_rule_fails_at_parse_time():
    data = _pattern()
    data["steps"][0]["cohort"]["min_successes"] = 2

    try:
        parse_pattern_declaration(data)
    except ValueError as exc:
        assert "min_successes" in str(exc)
    else:
        raise AssertionError("expected parse failure")


def test_step_names_are_unique():
    data = _pattern()
    data["steps"][1]["name"] = "train"

    try:
        parse_pattern_declaration(data)
    except ValueError as exc:
        assert "duplicate step" in str(exc)
    else:
        raise AssertionError("expected parse failure")


def test_member_names_are_unique_within_step():
    data = _pattern()
    members = data["steps"][0]["cohort"]["members"]
    members.append({**members[0]})

    try:
        parse_pattern_declaration(data)
    except ValueError as exc:
        assert "duplicate member" in str(exc)
    else:
        raise AssertionError("expected parse failure")


def test_empty_patterns_fail_at_parse_time():
    try:
        parse_pattern_declaration({"name": "empty", "steps": []})
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected parse failure")
