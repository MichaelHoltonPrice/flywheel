#!/usr/bin/env python3
"""Agent runner using the Claude Agent SDK (ClaudeSDKClient).

Runs a Claude Code agent via ClaudeSDKClient in streaming mode,
which keeps a persistent connection to the CLI subprocess.  This
enables mid-session /compact without killing the session.

Context compaction
------------------

After each assistant response, the runner checks token usage.  When
usage crosses COMPACT_THRESHOLD of the estimated context window, the
runner sends /compact as a new message in the live session.  The CLI
compacts the conversation history and the agent continues working —
all within the same subprocess, with no session interruption.

Pause/resume
------------

- **Rate limit**: Detected from RateLimitEvent; pauses and waits
  for .agent_resume file.
- **Auth error**: Detected from error messages; pauses and waits.
- **External resume**: Write a prompt (or empty) to .agent_resume
  in the workspace to continue after a pause.

Environment variables:
    MODEL           — Model to use (e.g., claude-sonnet-4-6)
    EVAL_ENDPOINT   — URL of the host-side eval HTTP service
    ALLOWED_TOOLS   — Comma-separated tool whitelist
    MAX_TURNS       — Total turn budget for the agent (optional)
    MCP_SERVERS     — Comma-separated list of MCP servers to enable.
                      Built-in: eval. Projects can mount additional
                      servers at /workspace/.mcp_servers/.
    RESUME_SESSION  — Session ID to resume on startup (optional)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anyio
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", "/workspace"))
STATE_FILE = WORKSPACE / ".agent_state.json"
RESUME_FILE = WORKSPACE / ".agent_resume"
POLL_INTERVAL = 5  # seconds between resume-file checks

# Proactive compaction: compact when input tokens exceed this fraction
# of the estimated context window.
COMPACT_THRESHOLD = 0.20
DEFAULT_CONTEXT_WINDOW = 200_000


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


_MCP_REGISTRY = {
    "eval": ("run_eval", _eval_server),
}

# Default directory where project-provided MCP servers are mounted.
MCP_SERVER_MOUNT_DIR = "/workspace/.mcp_servers"


def _scan_mounted_servers() -> dict:
    """Discover project-provided MCP servers from the mount directory.

    Scans for ``*_mcp_server.py`` files. For each, derives the server
    name (strip ``_mcp_server.py`` suffix), builds a config, and
    returns a registry dict matching ``_MCP_REGISTRY`` format.

    An optional sidecar ``*_mcp_server.json`` manifest provides tool
    names to add to ``allowed_tools``. If absent, tools are discovered
    via MCP handshake (no pre-registration).

    Returns:
        Dict mapping server names to ``(server_id, factory)`` tuples.
    """
    mount_dir = Path(os.environ.get(
        "MCP_SERVER_MOUNT_DIR", MCP_SERVER_MOUNT_DIR))
    if not mount_dir.is_dir():
        return {}
    servers = {}
    for py_file in sorted(mount_dir.glob("*_mcp_server.py")):
        name = py_file.name.removesuffix("_mcp_server.py")
        config = {
            "command": "python3",
            "args": [str(py_file)],
            "env": dict(os.environ),
        }
        # Read tool names from sidecar manifest if present.
        manifest = py_file.with_suffix(".json")
        tools: list[str] = []
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                tools = data.get("tools", [])
            except Exception:
                pass
        # Capture config and tools in the closure.
        def _factory(c=config, t=tools):
            return c, t
        servers[name] = (name, _factory)
    return servers


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

_TYPE_MAP = {
    "AssistantMessage": "assistant",
    "UserMessage": "user",
    "SystemMessage": "system",
    "ResultMessage": "result",
    "RateLimitEvent": "rate_limit",
}


def _emit(data: dict) -> None:
    """Emit a JSON event to stdout."""
    print(json.dumps(data, default=str), flush=True)


def _save_state(session_id: str, status: str, reason: str = "") -> None:
    """Persist agent state for resume and external visibility.

    NOTE: The session_id saved here is only useful for resume if
    the container is still running.  The SDK stores session history
    as local files at ~/.claude/projects/<cwd>/<session-id>.jsonl
    inside the container — these are lost when the container dies.
    To support cross-container resume, mount a persistent volume
    at /home/claude/.claude/projects/.  See docs/architecture.md.
    """
    state = {
        "session_id": session_id,
        "status": status,
        "reason": reason,
        "timestamp": time.time(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))
    _emit({"type": "agent_state", **state})


async def _wait_for_resume() -> str:
    """Block until .agent_resume file appears in the workspace."""
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


def _emit_message(message) -> None:
    """Serialize and emit an SDK message."""
    cls_name = type(message).__name__
    if hasattr(message, "model_dump"):
        data = message.model_dump()
    elif hasattr(message, "to_dict"):
        data = message.to_dict()
    elif hasattr(message, "__dict__"):
        data = message.__dict__
        if "content" in data and isinstance(data["content"], list):
            data["content"] = [_serialize(b) for b in data["content"]]
    else:
        data = {"type": "unknown", "repr": repr(message)}
    data["type"] = _TYPE_MAP.get(cls_name, cls_name)
    _emit(data)


def _get_input_tokens(message) -> int:
    """Extract total input tokens from an AssistantMessage."""
    usage = getattr(message, "usage", None)
    if not isinstance(usage, dict):
        return 0
    inp = usage.get("input_tokens", 0)
    cache_c = usage.get("cache_creation_input_tokens", 0)
    cache_r = usage.get("cache_read_input_tokens", 0)
    total = inp + cache_c + cache_r
    return total


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main() -> None:
    """Run the Claude Code agent with mid-session compaction."""
    # --- Read initial prompt ---
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

    # Register MCP servers (built-in + project-mounted).
    mounted_servers = _scan_mounted_servers()
    mcp_servers_str = os.environ.get("MCP_SERVERS", "")
    requested = [
        s.strip() for s in mcp_servers_str.split(",") if s.strip()
    ]
    mcp_servers = {}
    for name in requested:
        entry = _MCP_REGISTRY.get(name)
        if entry is None:
            entry = mounted_servers.get(name)
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

    # Context window estimate.
    context_window = DEFAULT_CONTEXT_WINDOW
    if model and "1m" in model.lower():
        context_window = 1_000_000
    compact_token_limit = int(context_window * COMPACT_THRESHOLD)

    # --- Build options ---
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

    # --- Connect and run ---
    last_input_tokens = 0
    session_id = ""

    try:
        async with ClaudeSDKClient(options=options) as client:
            # Send initial prompt.
            await client.query(prompt)

            # Process responses.  receive_response() yields messages
            # until a ResultMessage, then stops.  After compaction or
            # follow-up queries, we call receive_response() again to
            # continue processing.
            while True:
                completed = False
                pause_reason = None

                async for message in client.receive_response():
                    try:
                        cls_name = type(message).__name__

                        # Capture session_id.
                        sid = getattr(message, "session_id", None)
                        if sid:
                            session_id = sid

                        # Track token usage.
                        if isinstance(message, AssistantMessage):
                            tokens = _get_input_tokens(message)
                            if tokens > 0:
                                last_input_tokens = tokens

                            # Mid-session compaction trigger.
                            # Breaking out of receive_response() does
                            # NOT kill the ClaudeSDKClient connection.
                            # We can send /compact and then continue.
                            if last_input_tokens >= compact_token_limit:
                                pause_reason = "compact_needed"
                                _emit_message(message)
                                break

                        # Rate limit detection.
                        if cls_name == "RateLimitEvent":
                            info = getattr(
                                message, "rate_limit_info", None,
                            )
                            status = "unknown"
                            if info:
                                status = getattr(
                                    info, "status", "unknown",
                                )
                            if status == "rejected":
                                pause_reason = "rate_limit"
                                _emit({
                                    "type": "rate_limit",
                                    "status": "rejected",
                                })
                                break
                            _emit({
                                "type": "rate_limit",
                                "status": status,
                            })
                            continue

                        # Completion detection.
                        if isinstance(message, ResultMessage):
                            subtype = getattr(message, "subtype", "")
                            if subtype == "success":
                                completed = True

                        # Emit the message.
                        _emit_message(message)

                    except Exception as e:
                        _emit({
                            "type": "error",
                            "message": str(e),
                        })

                # --- After receive_response() returns ---

                # Natural completion.
                if completed:
                    _save_state(session_id, "complete")
                    return

                # External pause (rate limit, auth).
                if pause_reason and pause_reason != "compact_needed":
                    _save_state(session_id, "paused", pause_reason)
                    resume_prompt = await _wait_for_resume()
                    await client.query(resume_prompt)
                    continue

                # --- Proactive compaction ---
                # Triggered either mid-session (compact_needed) or
                # after a ResultMessage when tokens are high.
                if (
                    pause_reason == "compact_needed"
                    or last_input_tokens >= compact_token_limit
                ):
                    tokens_before = last_input_tokens
                    _emit({
                        "type": "compact",
                        "message": (
                            f"Context at {tokens_before:,} tokens "
                            f"(limit {compact_token_limit:,}), "
                            f"sending /compact"
                        ),
                        "tokens_before": tokens_before,
                        "threshold": compact_token_limit,
                    })

                    # Interrupt the current generation so the CLI
                    # can accept the /compact command.  Then drain
                    # any remaining messages from the interrupted
                    # response before sending /compact.
                    await client.interrupt()
                    async for msg in client.receive_response():
                        _emit_message(msg)

                    # Send /compact in the live session.
                    await client.query("/compact")
                    async for msg in client.receive_response():
                        sid = getattr(msg, "session_id", None)
                        if sid:
                            session_id = sid
                        # Track post-compact tokens.
                        if isinstance(msg, AssistantMessage):
                            tokens = _get_input_tokens(msg)
                            if tokens > 0:
                                last_input_tokens = tokens

                    _emit({
                        "type": "compact",
                        "message": "Compaction complete",
                        "tokens_before": tokens_before,
                        "tokens_after": last_input_tokens,
                        "success": True,
                    })

                    # Raise the threshold so we don't compact again
                    # immediately if tokens didn't drop much.  Next
                    # compaction triggers at current level + 50% of
                    # the original threshold.
                    compact_token_limit = (
                        last_input_tokens
                        + int(context_window * COMPACT_THRESHOLD * 0.5)
                    )

                    # Tell the agent to keep working.
                    await client.query(
                        "Your context was compacted automatically. "
                        "This is routine. Resume what you were doing."
                    )
                    continue

                # ResultMessage without completion and without needing
                # compact — this is error_max_turns or similar.  The
                # agent used all its turns.
                _save_state(session_id, "paused", "max_turns")
                resume_prompt = await _wait_for_resume()
                await client.query(resume_prompt)

    except Exception as e:
        err = str(e).lower()
        if "auth" in err or "401" in err:
            _emit({"type": "error", "message": f"Auth error: {e}"})
            _save_state(session_id, "paused", "auth_error")
            # Can't resume inside ClaudeSDKClient after auth failure.
            return
        _emit({"type": "error", "message": str(e)})
        _save_state(session_id, "paused", "error")


if __name__ == "__main__":
    anyio.run(main)
