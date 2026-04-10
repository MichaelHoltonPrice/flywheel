#!/usr/bin/env python3
"""MCP stdio server for nested block execution requests.

Runs inside an agent container as a subprocess. Exposes tools
for invoking flywheel block executions via the host-side block
bridge HTTP service.

The ``evaluate`` tool is a convenience wrapper that invokes a
named block (defaulting to ``eval_bot``) — the name reflects
the most common use case but the mechanism is generic.

CRITICAL: This script must NEVER print to stdout — doing so would
corrupt the JSON-RPC stdio channel. All diagnostics go to stderr.
"""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

_BRIDGE_ENDPOINT = os.environ.get("EVAL_ENDPOINT", "")
_DEFAULT_BLOCK = os.environ.get("EVAL_BLOCK", "eval_bot")

mcp = FastMCP("eval")


@mcp.tool()
def evaluate(artifact_path: str) -> str:
    """Evaluate an artifact by invoking a block execution.

    Triggers a block execution on the host via the block bridge
    service. The block to invoke is configured by the project
    (defaults to ``eval_bot``).

    Args:
        artifact_path: Path to the artifact file, relative to
            /workspace.

    Returns:
        JSON string with execution results. Contains "ok" (bool),
        "score" (float) on success, or "error_type" and "message"
        on failure.
    """
    if not _BRIDGE_ENDPOINT:
        return (
            '{"ok": false, "error_type": "config_error",'
            ' "message": "EVAL_ENDPOINT not set"}'
        )

    try:
        response = httpx.post(
            _BRIDGE_ENDPOINT,
            json={
                "block_name": _DEFAULT_BLOCK,
                "artifact_path": artifact_path,
            },
            timeout=300.0,
        )
        response.raise_for_status()
        return response.text
    except httpx.TimeoutException:
        return (
            '{"ok": false, "error_type": "timeout",'
            ' "message": "Request timed out after 300s"}'
        )
    except Exception as e:
        msg = str(e).replace('"', '\\"')
        return (
            '{"ok": false, "error_type": "request_error",'
            f' "message": "{msg}"'
            "}"
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
