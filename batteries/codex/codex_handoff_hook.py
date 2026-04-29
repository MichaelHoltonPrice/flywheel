#!/usr/bin/env python3
"""Codex PostToolUse hook that converts selected MCP calls into handoffs."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def _load_configs() -> dict:
    raw = os.environ.get("HANDOFF_TOOL_CONFIG_NORMALIZED", "")
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _missing_paths(paths: list[str]) -> list[str]:
    missing = []
    for value in paths:
        path = Path(value)
        if not path.exists():
            missing.append(value)
    return missing


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({
            "systemMessage": f"Flywheel handoff hook ignored invalid input: {exc}",
        }))
        return 0

    tool_name = event.get("tool_name")
    configs = _load_configs()
    config = configs.get(tool_name)
    if not isinstance(config, dict):
        return 0

    required_paths = config.get("required_paths", [])
    if not isinstance(required_paths, list):
        required_paths = []
    missing = _missing_paths([str(path) for path in required_paths])
    if missing:
        print(json.dumps({
            "systemMessage": (
                "Flywheel handoff request ignored; missing required path(s): "
                + ", ".join(missing)
            ),
        }))
        return 0

    pending = {
        "tool_use_id": str(event.get("tool_use_id") or ""),
        "tool_name": str(tool_name or ""),
        "termination_reason": str(
            config.get("termination_reason") or "tool_handoff"),
        "result_path": str(
            config.get("result_path") or "/input/score/scores.json"),
        "result_label": str(config.get("result_label") or "Tool result"),
        "placeholder_marker": str(
            config.get("placeholder_marker") or "Tool handoff requested."),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    pending_path = Path(os.environ.get(
        "FLYWHEEL_HANDOFF_PENDING", "/tmp/flywheel-codex-handoff.json"))
    pending_path.write_text(
        json.dumps(pending, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "continue": False,
        "stopReason": pending["termination_reason"],
        "systemMessage": pending["placeholder_marker"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
