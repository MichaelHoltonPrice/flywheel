#!/usr/bin/env python3
"""Agent runner using the Claude Agent SDK.

Runs a Claude Code agent via the SDK's query() function and streams
JSON events to stdout. Supports pause/resume for long-running agents:

- **Max turns**: Set MAX_TURNS to limit turns per query session.
  When reached, the runner pauses and waits for a resume signal.
- **Rate limit**: Detects rate limit rejections and pauses
  automatically, waiting for a resume signal.
- **Resume**: On pause, the runner writes session state to
  .agent_state.json and polls for .agent_resume in the workspace.
  Write a prompt to that file (or leave it empty for a default) to
  continue. Alternatively, set RESUME_SESSION at startup to resume
  a specific session immediately.

Environment variables:
    MODEL           — Model to use (e.g., claude-sonnet-4-6)
    EVAL_ENDPOINT   — URL of the host-side eval HTTP service
    ALLOWED_TOOLS   — Comma-separated tool whitelist
    MAX_TURNS       — Maximum turns per query session (optional)
    MCP_SERVERS     — Comma-separated list of MCP servers to enable.
                      Known servers: eval, arc.
    RESUME_SESSION  — Session ID to resume on startup (optional)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anyio
from claude_agent_sdk import ClaudeAgentOptions, query

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", "/workspace"))
STATE_FILE = WORKSPACE / ".agent_state.json"
RESUME_FILE = WORKSPACE / ".agent_resume"
POLL_INTERVAL = 5  # seconds between resume-file checks


# ------------------------------------------------------------------
# MCP server registry
# ------------------------------------------------------------------

def _eval_server():
    """Block invocation proxy — requires EVAL_ENDPOINT to be set."""
    endpoint = os.environ.get("EVAL_ENDPOINT", "")
    if not endpoint:
        return None
    eval_block = os.environ.get("EVAL_BLOCK", "eval_bot")
    config = {
        "command": "python3",
        "args": ["/app/eval_mcp_server.py"],
        "env": {"EVAL_ENDPOINT": endpoint, "EVAL_BLOCK": eval_block},
    }
    tools = ["mcp__run_eval__evaluate"]
    return config, tools


def _arc_server():
    """ARC-AGI-3 game interaction — requires GAME_ID and ARC_SERVER_URL."""
    game_id = os.environ.get("GAME_ID", "")
    server_url = os.environ.get("ARC_SERVER_URL", "")
    if not game_id or not server_url:
        return None
    config = {
        "command": "python3",
        "args": ["/app/arc_mcp_server.py"],
        "env": {"GAME_ID": game_id, "ARC_SERVER_URL": server_url},
    }
    tools = [
        "mcp__arc__start_game",
        "mcp__arc__take_action",
        "mcp__arc__reset_level",
        "mcp__arc__get_status",
    ]
    return config, tools


_MCP_REGISTRY = {
    "eval": ("run_eval", _eval_server),
    "arc": ("arc", _arc_server),
}


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------

def _serialize(obj):
    """Recursively serialize an SDK object to a JSON-safe dict."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        d = {k: _serialize(v) for k, v in obj.__dict__.items()}
        cls = type(obj).__name__
        if "type" not in d:
            _BLOCK_TYPES = {
                "TextBlock": "text",
                "ToolUseBlock": "tool_use",
                "ToolResultBlock": "tool_result",
                "ThinkingBlock": "thinking",
            }
            d["type"] = _BLOCK_TYPES.get(cls, cls)
        return d
    return str(obj)


# ------------------------------------------------------------------
# Event emission and state management
# ------------------------------------------------------------------

def _emit(data: dict) -> None:
    """Emit a JSON event to stdout."""
    print(json.dumps(data, default=str), flush=True)


def _save_state(session_id: str, status: str, reason: str = "") -> None:
    """Persist agent state for resume and external visibility."""
    state = {
        "session_id": session_id,
        "status": status,
        "reason": reason,
        "timestamp": time.time(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))
    _emit({"type": "agent_state", **state})


async def _wait_for_resume() -> str:
    """Block until .agent_resume file appears in the workspace.

    Returns the file content as the resume prompt (or a default).
    The host writes this file when it's time to continue.
    """
    _emit({
        "type": "agent_state",
        "status": "waiting_for_resume",
        "resume_file": str(RESUME_FILE),
    })
    while True:
        if RESUME_FILE.exists():
            text = RESUME_FILE.read_text().strip()
            RESUME_FILE.unlink()
            return text or "Continue from where you left off."
        await anyio.sleep(POLL_INTERVAL)


def _get_session_id(message) -> str | None:
    """Extract session_id from an SDK message if present."""
    sid = getattr(message, "session_id", None)
    return sid if sid else None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main() -> None:
    """Run the Claude Code agent with pause/resume support."""
    # --- Determine initial state ---
    session_id = os.environ.get("RESUME_SESSION", "")

    # Check for saved paused state from a previous session.
    if not session_id and STATE_FILE.exists():
        try:
            saved = json.loads(STATE_FILE.read_text())
            if saved.get("status") == "paused":
                session_id = saved.get("session_id", "")
        except (json.JSONDecodeError, OSError):
            pass

    if session_id:
        prompt = "Continue from where you left off."
    else:
        prompt = sys.stdin.read()
        if not prompt.strip():
            _emit({"type": "error", "message": "Empty prompt"})
            sys.exit(1)

    # --- Setup ---
    claude_json = os.path.expanduser("~/.claude.json")
    if not os.path.exists(claude_json):
        with open(claude_json, "w") as f:
            f.write("{}")

    model = os.environ.get("MODEL", "")
    allowed_tools_str = os.environ.get(
        "ALLOWED_TOOLS", "Read,Write,Edit,Glob,Grep",
    )
    allowed_tools = [
        t.strip() for t in allowed_tools_str.split(",") if t.strip()
    ]
    max_turns_str = os.environ.get("MAX_TURNS", "")
    max_turns = int(max_turns_str) if max_turns_str else None

    # Register MCP servers declared in MCP_SERVERS.
    mcp_servers_str = os.environ.get("MCP_SERVERS", "")
    requested = [
        s.strip() for s in mcp_servers_str.split(",") if s.strip()
    ]
    mcp_servers = {}
    for name in requested:
        entry = _MCP_REGISTRY.get(name)
        if entry is None:
            _emit({
                "type": "warning",
                "message": f"Unknown MCP server: {name!r}",
            })
            continue
        server_id, factory = entry
        result = factory()
        if result is None:
            continue
        config, tools = result
        mcp_servers[server_id] = config
        for t in tools:
            if t not in allowed_tools:
                allowed_tools.append(t)

    # --- Pause/resume loop ---
    _TYPE_MAP = {
        "AssistantMessage": "assistant",
        "UserMessage": "user",
        "SystemMessage": "system",
        "ResultMessage": "result",
        "RateLimitEvent": "rate_limit",
    }

    while True:
        # Build options for this query session.
        options = ClaudeAgentOptions(
            cwd="/workspace",
            allowed_tools=allowed_tools,
            permission_mode="bypassPermissions",
        )
        if model:
            options.model = model
        if mcp_servers:
            options.mcp_servers = mcp_servers
        if max_turns:
            options.max_turns = max_turns
        if session_id:
            options.resume = session_id

        # Run one query session.
        pause_reason = None

        try:
            async for message in query(prompt=prompt, options=options):
                try:
                    cls_name = type(message).__name__

                    # Capture session_id from any message that has it.
                    sid = _get_session_id(message)
                    if sid:
                        session_id = sid

                    # --- Rate limit detection ---
                    if cls_name == "RateLimitEvent":
                        status = "unknown"
                        info = getattr(message, "rate_limit_info", None)
                        if info:
                            status = getattr(info, "status", "unknown")
                        if status == "rejected":
                            pause_reason = "rate_limit"
                            _emit({
                                "type": "rate_limit",
                                "status": "rejected",
                            })
                            break
                        # Warning or allowed — log and continue.
                        _emit({"type": "rate_limit", "status": status})
                        continue

                    # --- Max turns detection ---
                    if cls_name == "ResultMessage":
                        subtype = getattr(message, "subtype", "")
                        if subtype == "error_max_turns":
                            pause_reason = "max_turns"

                    # --- Serialize and emit ---
                    if hasattr(message, "model_dump"):
                        data = message.model_dump()
                    elif hasattr(message, "to_dict"):
                        data = message.to_dict()
                    elif hasattr(message, "__dict__"):
                        data = message.__dict__
                        if "content" in data and isinstance(
                            data["content"], list,
                        ):
                            data["content"] = [
                                _serialize(b) for b in data["content"]
                            ]
                    else:
                        data = {"type": "unknown", "repr": repr(message)}

                    data["type"] = _TYPE_MAP.get(cls_name, cls_name)
                    _emit(data)

                except Exception as e:
                    _emit({"type": "error", "message": str(e)})

        except Exception as e:
            err = str(e).lower()
            if ("rate" in err and "limit" in err) or "overloaded" in err:
                pause_reason = "rate_limit"
                _emit({"type": "error", "message": f"Rate limit: {e}"})
            elif pause_reason:
                # Expected: CLI exits non-zero after a known pause
                # trigger (e.g., exit code 1 on max_turns).
                pass
            elif "auth" in err or "401" in err:
                pause_reason = "auth_error"
                _emit({"type": "error", "message": f"Auth error: {e}"})
            else:
                # Unknown error — still pause rather than crash,
                # so we preserve the session for potential resume.
                pause_reason = "error"
                _emit({"type": "error", "message": str(e)})

        # --- Decide next action ---
        if pause_reason:
            _save_state(session_id, "paused", pause_reason)
            prompt = await _wait_for_resume()
            continue

        # Natural completion.
        _save_state(session_id, "complete")
        break


if __name__ == "__main__":
    anyio.run(main)
