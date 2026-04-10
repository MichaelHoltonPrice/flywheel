#!/usr/bin/env python3
"""Agent runner using the Claude Agent SDK.

Reads a prompt from stdin, runs a Claude Code agent via the SDK's
query() function, and streams JSON events to stdout. The SDK
manages context compaction automatically.

Environment variables:
    MODEL           — Model to use (e.g., claude-sonnet-4-6)
    EVAL_ENDPOINT   — URL of the host-side eval HTTP service
    ALLOWED_TOOLS   — Comma-separated tool whitelist
    MAX_TURNS       — Maximum agent turns (optional)
    MCP_SERVERS     — Comma-separated list of MCP servers to enable.
                      Known servers: eval.
"""

from __future__ import annotations

import json
import os
import sys

import anyio
from claude_agent_sdk import ClaudeAgentOptions, query

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


async def main() -> None:
    """Run the Claude Code agent, streaming events to stdout."""
    prompt = sys.stdin.read()
    if not prompt.strip():
        print(
            json.dumps({"type": "error", "message": "Empty prompt"}),
            file=sys.stdout,
        )
        sys.exit(1)

    # Create ~/.claude.json if it doesn't exist (Claude Code expects it).
    claude_json = os.path.expanduser("~/.claude.json")
    if not os.path.exists(claude_json):
        with open(claude_json, "w") as f:
            f.write("{}")

    # Configuration from environment.
    model = os.environ.get("MODEL", "")
    allowed_tools_str = os.environ.get(
        "ALLOWED_TOOLS", "Read,Write,Edit,Glob,Grep",
    )
    allowed_tools = [
        t.strip() for t in allowed_tools_str.split(",") if t.strip()
    ]
    max_turns_str = os.environ.get("MAX_TURNS", "")

    # Register MCP servers declared in MCP_SERVERS.
    mcp_servers_str = os.environ.get("MCP_SERVERS", "")
    requested = [
        s.strip() for s in mcp_servers_str.split(",") if s.strip()
    ]

    mcp_servers = {}
    for name in requested:
        entry = _MCP_REGISTRY.get(name)
        if entry is None:
            print(
                json.dumps({
                    "type": "warning",
                    "message": f"Unknown MCP server: {name!r}",
                }),
                file=sys.stderr,
            )
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

    # Build options.
    options = ClaudeAgentOptions(
        cwd="/workspace",
        allowed_tools=allowed_tools,
        permission_mode="bypassPermissions",
    )
    if model:
        options.model = model
    if mcp_servers:
        options.mcp_servers = mcp_servers
    if max_turns_str:
        options.max_turns = int(max_turns_str)

    # Run the agent and stream events to stdout.
    _TYPE_MAP = {
        "AssistantMessage": "assistant",
        "UserMessage": "user",
        "SystemMessage": "system",
        "ResultMessage": "result",
        "RateLimitEvent": "rate_limit",
    }

    async for message in query(prompt=prompt, options=options):
        try:
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

            cls_name = type(message).__name__
            data["type"] = _TYPE_MAP.get(cls_name, cls_name)

            print(json.dumps(data, default=str), flush=True)
        except Exception as e:
            print(
                json.dumps({"type": "error", "message": str(e)}),
                flush=True,
            )


if __name__ == "__main__":
    anyio.run(main)
