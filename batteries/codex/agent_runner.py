#!/usr/bin/env python3
"""Run Codex CLI as a Flywheel one-shot agent battery.

The runner wraps ``codex exec`` and translates its JSONL event stream into
Flywheel-owned runtime metadata. It deliberately does not know about Flywheel
workspaces, artifact forging, pattern lanes, or invocation routing. Those
remain substrate responsibilities outside the battery.

Environment variables:
    MODEL
        Optional Codex model slug.
    REASONING_EFFORT
        Optional reasoning effort passed through to Codex config.
    CODEX_AUTO_COMPACT_TOKEN_LIMIT
        Optional positive integer token threshold for Codex automatic
        conversation compaction.
    FLYWHEEL_AGENT_PROMPT
        Image-local prompt file. Defaults to ``/app/agent/prompt.md``.
    MCP_SERVERS
        Comma-separated project MCP server names. Server files are discovered
        from ``MCP_SERVER_MOUNT_DIR`` by looking for ``*_mcp_server.py``.
    HANDOFF_TOOL_CONFIG
        Optional JSON object keyed by MCP tool name. Each value may set
        ``termination_reason``, ``required_paths``, ``result_path``,
        ``result_label``, and ``placeholder_marker``. A Codex PostToolUse hook
        stops the turn when one of these tools completes.
    HANDOFF_TOOLS
        Simple comma-separated handoff tool list. Used only when
        ``HANDOFF_TOOL_CONFIG`` is unset.
    HANDOFF_TERMINATION_REASON
        Termination reason for the simple ``HANDOFF_TOOLS`` shape.
    HANDOFF_REQUIRED_PATHS
        Comma-separated required paths for the simple shape.
    HANDOFF_RESULT_PATH, HANDOFF_RESULT_LABEL, HANDOFF_PLACEHOLDER_MARKER
        Result metadata for the simple shape and continuation prompt.
    CODEX_EXTRA_ARGS
        Optional shell-split extra arguments appended before the prompt.
    CODEX_SANDBOX
        Codex sandbox mode. Defaults to ``danger-full-access`` inside the
        already-isolated Flywheel container.
    CODEX_APPROVAL_POLICY
        Codex approval policy. Defaults to ``never`` for unattended blocks.
    FLYWHEEL_CODEX_FAKE
        Test hook: path to an executable that mimics ``codex``.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", "/scratch"))
DEFAULT_PROMPT_FILE = Path("/app/agent/prompt.md")
MCP_SERVER_MOUNT_DIR = "/flywheel/mcp_servers"
CODEX_HOME = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
FLYWHEEL_TELEMETRY_DIR = Path(os.environ.get(
    "FLYWHEEL_TELEMETRY_DIR", "/tmp/flywheel-codex-telemetry"))
EVENT_LOG = Path(os.environ.get(
    "FLYWHEEL_CODEX_EVENT_LOG", "/tmp/flywheel-codex-events.jsonl"))
LAST_MESSAGE_PATH = Path(os.environ.get(
    "FLYWHEEL_CODEX_LAST_MESSAGE", "/tmp/flywheel-codex-last-message.txt"))

DEFAULT_HANDOFF_PLACEHOLDER_MARKER = "Tool handoff requested."
HANDOFF_PENDING_PATH = Path(os.environ.get(
    "FLYWHEEL_HANDOFF_PENDING", "/tmp/flywheel-codex-handoff.json"))
BENIGN_STDERR_PATTERNS = [
    re.compile(
        r"^\S+\s+ERROR\s+codex_core::session: "
        r"failed to record rollout items: thread .* not found$"
    ),
]


def _emit(data: dict[str, Any]) -> None:
    print(json.dumps(data, default=str), flush=True)


def _load_prompt() -> str:
    prompt_file = Path(os.environ.get(
        "FLYWHEEL_AGENT_PROMPT", str(DEFAULT_PROMPT_FILE)))
    if not prompt_file.is_file():
        raise RuntimeError(
            f"Prompt file not found at {prompt_file}; derive an agent "
            "image from the Codex battery and copy the project prompt into "
            "that path, or set FLYWHEEL_AGENT_PROMPT."
        )
    prompt = prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        raise RuntimeError(f"Prompt file {prompt_file} is empty")
    return prompt


def _scan_mounted_servers() -> dict[str, dict[str, Any]]:
    mount_dir = Path(os.environ.get(
        "MCP_SERVER_MOUNT_DIR", MCP_SERVER_MOUNT_DIR))
    if not mount_dir.is_dir():
        return {}
    servers: dict[str, dict[str, Any]] = {}
    for py_file in sorted(mount_dir.glob("*_mcp_server.py")):
        name = py_file.name.removesuffix("_mcp_server.py")
        manifest = py_file.with_suffix(".json")
        tools: list[str] = []
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                raw_tools = data.get("tools", [])
                if isinstance(raw_tools, list):
                    tools = [str(tool) for tool in raw_tools]
            except Exception:
                tools = []
        servers[name] = {
            "command": "python3",
            "args": [str(py_file)],
            "env": dict(os.environ),
            "tools": tools,
        }
    return servers


def _legacy_handoff_config(
    tools: set[str],
    required_paths: list[Path],
    *,
    termination_reason: str,
    result_path: str,
    result_label: str,
    placeholder_marker: str,
) -> dict[str, dict[str, Any]]:
    return {
        tool: {
            "termination_reason": termination_reason,
            "required_paths": [str(path) for path in required_paths],
            "result_path": result_path,
            "result_label": result_label,
            "placeholder_marker": placeholder_marker,
        }
        for tool in tools
    }


def _parse_handoff_tool_config(raw: str) -> dict[str, dict[str, Any]]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"HANDOFF_TOOL_CONFIG was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("HANDOFF_TOOL_CONFIG must be a JSON object")

    configs: dict[str, dict[str, Any]] = {}
    for tool_name, value in parsed.items():
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise RuntimeError("HANDOFF_TOOL_CONFIG has an invalid tool name")
        if not isinstance(value, dict):
            raise RuntimeError(
                f"HANDOFF_TOOL_CONFIG[{tool_name!r}] must be an object")
        termination_reason = str(
            value.get("termination_reason") or "tool_handoff").strip()
        if not termination_reason:
            raise RuntimeError(
                f"HANDOFF_TOOL_CONFIG[{tool_name!r}].termination_reason "
                "must not be empty")
        paths_raw = value.get("required_paths", [])
        if isinstance(paths_raw, str):
            required_paths = [
                p.strip() for p in paths_raw.split(",") if p.strip()]
        elif isinstance(paths_raw, list):
            required_paths = []
            for item in paths_raw:
                if not isinstance(item, str) or not item.strip():
                    raise RuntimeError(
                        f"HANDOFF_TOOL_CONFIG[{tool_name!r}].required_paths "
                        "entries must be non-empty strings")
                required_paths.append(item.strip())
        else:
            raise RuntimeError(
                f"HANDOFF_TOOL_CONFIG[{tool_name!r}].required_paths must be "
                "a list or comma-separated string")
        configs[tool_name.strip()] = {
            "termination_reason": termination_reason,
            "required_paths": required_paths,
            "result_path": str(
                value.get("result_path") or "/input/score/scores.json"),
            "result_label": str(value.get("result_label") or "Tool result"),
            "placeholder_marker": str(
                value.get("placeholder_marker")
                or DEFAULT_HANDOFF_PLACEHOLDER_MARKER
            ),
        }
    return configs


def _handoff_configs() -> dict[str, dict[str, Any]]:
    configs = _parse_handoff_tool_config(
        os.environ.get("HANDOFF_TOOL_CONFIG", ""))
    if configs:
        return configs
    tools = {
        tool.strip()
        for tool in os.environ.get("HANDOFF_TOOLS", "").split(",")
        if tool.strip()
    }
    required_paths = [
        Path(path.strip())
        for path in os.environ.get("HANDOFF_REQUIRED_PATHS", "").split(",")
        if path.strip()
    ]
    if not tools:
        return {}
    reason = os.environ.get(
        "HANDOFF_TERMINATION_REASON", "tool_handoff").strip()
    return _legacy_handoff_config(
        tools,
        required_paths,
        termination_reason=reason or "tool_handoff",
        result_path=os.environ.get(
            "HANDOFF_RESULT_PATH", "/input/score/scores.json"),
        result_label=os.environ.get("HANDOFF_RESULT_LABEL", "Tool result"),
        placeholder_marker=os.environ.get(
            "HANDOFF_PLACEHOLDER_MARKER",
            DEFAULT_HANDOFF_PLACEHOLDER_MARKER,
        ),
    )


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _write_codex_config(handoff_configs: dict[str, dict[str, Any]]) -> None:
    CODEX_HOME.mkdir(parents=True, exist_ok=True)
    lines = [
        'cli_auth_credentials_store = "file"',
        "",
        "[features]",
        "codex_hooks = true" if handoff_configs else "codex_hooks = false",
        "",
    ]

    mounted = _scan_mounted_servers()
    requested = [
        name.strip()
        for name in os.environ.get("MCP_SERVERS", "").split(",")
        if name.strip()
    ]
    for name in requested:
        config = mounted.get(name)
        if config is None:
            _emit({
                "type": "warning",
                "message": f"Unknown MCP server: {name!r}",
            })
            continue
        lines.extend([
            f"[mcp_servers.{name}]",
            f"command = {_toml_string(config['command'])}",
            "args = ["
            + ", ".join(_toml_string(arg) for arg in config["args"])
            + "]",
            "required = true",
            "tool_timeout_sec = 600",
            "",
            f"[mcp_servers.{name}.env]",
        ])
        env = config.get("env", {})
        for key in sorted(env):
            if key.startswith("_"):
                continue
            lines.append(f"{key} = {_toml_string(str(env[key]))}")
        lines.append("")

    if handoff_configs:
        tools_pattern = "|".join(
            sorted(tool.replace("\\", "\\\\") for tool in handoff_configs))
        lines.extend([
            "[[hooks.PostToolUse]]",
            f"matcher = {_toml_string(tools_pattern)}",
            "",
            "[[hooks.PostToolUse.hooks]]",
            'type = "command"',
            'command = "python3 /app/codex_handoff_hook.py"',
            "timeout = 30",
            'statusMessage = "Checking Flywheel handoff"',
            "",
        ])

    (CODEX_HOME / "config.toml").write_text(
        "\n".join(lines), encoding="utf-8")
    _emit({
        "type": "codex_config_written",
        "mcp_servers": requested,
        "handoff_tools": sorted(handoff_configs),
    })


def _build_command(prompt: str) -> list[str]:
    executable = os.environ.get("FLYWHEEL_CODEX_FAKE", "codex")
    command = [executable, "exec"]
    command.extend(["--json"])
    command.extend(["--output-last-message", str(LAST_MESSAGE_PATH)])
    command.extend([
        "--sandbox", os.environ.get("CODEX_SANDBOX", "danger-full-access"),
    ])
    approval_policy = os.environ.get("CODEX_APPROVAL_POLICY", "never").strip()
    if approval_policy:
        command.extend([
            "-c", f"approval_policy={json.dumps(approval_policy)}",
        ])
    if os.environ.get("CODEX_SKIP_GIT_REPO_CHECK", "1") != "0":
        command.append("--skip-git-repo-check")
    model = os.environ.get("MODEL", "").strip()
    if model:
        command.extend(["--model", model])
    effort = os.environ.get("REASONING_EFFORT", "").strip()
    if effort:
        command.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])
    extra = os.environ.get("CODEX_EXTRA_ARGS", "").strip()
    if extra:
        command.extend(shlex.split(extra))
    compact_limit = _positive_int_env("CODEX_AUTO_COMPACT_TOKEN_LIMIT")
    if compact_limit is not None:
        command.extend([
            "-c",
            f"model_auto_compact_token_limit={compact_limit}",
        ])
    if (CODEX_HOME / "sessions").exists():
        command.extend(["resume", "--last"])
    command.append(prompt)
    return command


def _positive_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _record_usage(events: list[dict[str, Any]]) -> None:
    telemetry_path = FLYWHEEL_TELEMETRY_DIR / "codex_usage.json"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    usage_events = [
        event for event in events
        if event.get("type") in ("turn.completed", "turn.failed")
    ]
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    for event in usage_events:
        usage = event.get("usage", {})
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or 0)
            cached_input_tokens += int(
                usage.get("cached_input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)

    thread_ids = [
        event.get("thread_id") for event in events
        if isinstance(event.get("thread_id"), str)
    ]
    latest_message = ""
    if LAST_MESSAGE_PATH.exists():
        latest_message = LAST_MESSAGE_PATH.read_text(
            encoding="utf-8", errors="replace")

    telemetry_path.write_text(json.dumps({
        "kind": "codex_usage",
        "source": "flywheel-codex",
        "data": {
            "thread_id": thread_ids[-1] if thread_ids else "",
            "model": os.environ.get("MODEL", ""),
            "event_count": len(events),
            "usage_event_count": len(usage_events),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "last_message": latest_message,
        },
    }, indent=2), encoding="utf-8")


def _is_benign_stderr(stderr: str) -> bool:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return True
    return all(
        any(pattern.match(line) for pattern in BENIGN_STDERR_PATTERNS)
        for line in lines
    )


def _run_codex(prompt: str) -> int:
    command = _build_command(prompt)
    _emit({"type": "codex_command", "argv": command[:-1] + ["<prompt>"]})
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    proc = subprocess.Popen(
        command,
        cwd=str(WORKSPACE),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    with EVENT_LOG.open("w", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                _emit({"type": "codex_stdout", "text": stripped})
                continue
            events.append(event)
            _emit({"type": "codex_event", "event": event})
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    rc = proc.wait()
    if stderr:
        FLYWHEEL_TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        (FLYWHEEL_TELEMETRY_DIR / "codex_stderr.txt").write_text(
            stderr, encoding="utf-8")
        if _is_benign_stderr(stderr):
            _emit({
                "type": "codex_stderr_suppressed",
                "reason": "known_nonfatal_rollout_recorder_warning",
            })
        else:
            _emit({"type": "codex_stderr", "preview": stderr[:4000]})
    _record_usage(events)
    session_ids = [
        event.get("thread_id") for event in events
        if isinstance(event.get("thread_id"), str)
    ]
    if session_ids:
        _emit({
            "type": "agent_state",
            "status": "complete" if rc == 0 else "failed",
            "session_id": session_ids[-1],
            "timestamp": time.time(),
        })
    if HANDOFF_PENDING_PATH.exists():
        try:
            pending = json.loads(HANDOFF_PENDING_PATH.read_text(
                encoding="utf-8"))
        except Exception:
            pending = {}
        if isinstance(pending, dict):
            reason = pending.get("termination_reason") or "tool_handoff"
            _emit({"type": "handoff_pending", "pending": [pending]})
            _emit({
                "type": "agent_state",
                "status": reason,
                "session_id": session_ids[-1] if session_ids else "",
                "timestamp": time.time(),
            })
    return rc


def main() -> int:
    try:
        prompt = _load_prompt()
        handoff_configs = _handoff_configs()
        os.environ["HANDOFF_TOOL_CONFIG_NORMALIZED"] = json.dumps(
            handoff_configs, sort_keys=True)
        os.environ["FLYWHEEL_HANDOFF_PENDING"] = str(HANDOFF_PENDING_PATH)
        _write_codex_config(handoff_configs)
        return _run_codex(prompt)
    except Exception as exc:  # noqa: BLE001
        _emit({
            "type": "error",
            "message": str(exc),
            "error_type": type(exc).__name__,
            "timestamp": datetime.now(UTC).isoformat(),
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
