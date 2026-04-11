#!/usr/bin/env python3
"""ARC-AGI-3 game interaction MCP server (REST API client).

Provides tools for an agent to play ARC-AGI-3 games by calling a
remote game server via HTTP. The agent never sees the game source
code — it only receives frames and state through the API, matching
the competition setup.

Environment variables:
    ARC_SERVER_URL  — Base URL of the game server
                      (e.g., "http://host.docker.internal:8001").
    GAME_ID         — Game to play (e.g., "vc33-9851e02b").
"""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arc")

# Global game state.
_client: httpx.Client | None = None
_base_url: str = ""
_game_id: str = ""
_card_id: str = ""
_guid: str = ""
_action_count: int = 0


def _frame_to_text(frame: list) -> str:
    """Convert a frame (list of lists) to compact text.

    Each row is a line of space-separated integers.
    """
    lines = []
    for row in frame:
        lines.append(" ".join(str(int(v)) for v in row))
    return "\n".join(lines)


def _format_response(data: dict, action_desc: str = "") -> str:
    """Format an API response as structured text for the agent."""
    parts = []
    if action_desc:
        parts.append(f"ACTION: {action_desc}")
    parts.append(f"ACTIONS_TAKEN: {_action_count}")
    parts.append(f"STATE: {data.get('state', 'UNKNOWN')}")
    parts.append(f"LEVELS_COMPLETED: {data.get('levels_completed', '?')}")
    parts.append(f"WIN_LEVELS: {data.get('win_levels', '?')}")
    parts.append(f"AVAILABLE_ACTIONS: {data.get('available_actions', [])}")

    frame = data.get("frame")
    if frame and isinstance(frame, list) and len(frame) > 0:
        grid = frame[0]  # First frame in the list.
        if isinstance(grid, list) and len(grid) > 0:
            h = len(grid)
            w = len(grid[0]) if grid else 0
            # Compute unique colors.
            colors = set()
            for row in grid:
                colors.update(row)
            parts.append(f"FRAME_SHAPE: {h}x{w}")
            parts.append(f"COLORS_PRESENT: {sorted(colors)}")
            parts.append("FRAME:")
            parts.append(_frame_to_text(grid))

    return "\n".join(parts)


def _post(endpoint: str, payload: dict) -> dict:
    """POST to the game server and return JSON response."""
    if _client is None:
        return {"error": "Not connected. Call start_game() first."}
    payload["card_id"] = _card_id
    payload["game_id"] = _game_id
    if _guid:
        payload["guid"] = _guid
    r = _client.post(
        f"{_base_url}/api/cmd/{endpoint}",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "detail": r.text[:300]}
    return r.json()


@mcp.tool()
def start_game() -> str:
    """Start (or restart) the current game. Returns the initial frame.

    The game server and game ID are configured via environment
    variables. Call this once at the beginning.
    """
    global _client, _base_url, _game_id, _card_id, _guid, _action_count

    _base_url = os.environ.get("ARC_SERVER_URL", "")
    _game_id = os.environ.get("GAME_ID", "")

    if not _base_url:
        return "ERROR: ARC_SERVER_URL environment variable not set."
    if not _game_id:
        return "ERROR: GAME_ID environment variable not set."

    _client = httpx.Client(timeout=30.0)
    _action_count = 0
    _guid = ""

    # Open a scorecard.
    r = _client.post(
        f"{_base_url}/api/scorecard/open",
        json={},
        headers={"Content-Type": "application/json"},
    )
    if r.status_code != 200:
        return f"ERROR: Failed to open scorecard: {r.status_code} {r.text[:200]}"
    _card_id = r.json().get("card_id", "")

    # Reset (start the game).
    data = _post("RESET", {})
    if "error" in data:
        return f"ERROR: {data['error']} {data.get('detail', '')}"
    _guid = data.get("guid", "")

    return _format_response(data, "start_game")


@mcp.tool()
def take_action(action: int, x: int = -1, y: int = -1) -> str:
    """Take a game action and return the resulting frame.

    Args:
        action: Action number (1-5 for simple actions, 6 for click
            with coordinates, 7 for undo). Do NOT send 0 (RESET)
            here — use reset_level() instead.
        x: X coordinate (0-63) for ACTION6 clicks. Ignored for
            other actions.
        y: Y coordinate (0-63) for ACTION6 clicks. Ignored for
            other actions.

    Returns:
        The new game state including the frame, state, levels
        completed, and available actions.
    """
    global _action_count, _guid

    if _client is None:
        return "ERROR: No game started. Call start_game() first."

    if action < 1 or action > 7:
        return f"ERROR: Invalid action {action}. Must be 1-7."

    payload = {}
    if action == 6:
        payload["x"] = x
        payload["y"] = y

    _action_count += 1
    data = _post(f"ACTION{action}", payload)

    if "error" in data:
        _action_count -= 1  # Don't count failed actions.
        return f"ERROR: {data['error']} {data.get('detail', '')}"

    if "guid" in data:
        _guid = data["guid"]

    desc = f"ACTION{action}"
    if action == 6:
        desc = f"ACTION6(x={x}, y={y})"

    return _format_response(data, desc)


@mcp.tool()
def reset_level() -> str:
    """Reset the current level (sends RESET action).

    Use this to restart the current level if you're stuck or
    want to try a different approach.
    """
    global _guid

    if _client is None:
        return "ERROR: No game started. Call start_game() first."

    data = _post("RESET", {})
    if "error" in data:
        return f"ERROR: {data['error']} {data.get('detail', '')}"

    if "guid" in data:
        _guid = data["guid"]

    return _format_response(data, "RESET (level restart)")


@mcp.tool()
def get_status() -> str:
    """Get current game status without taking an action.

    Returns: actions taken and connection status.
    """
    if _client is None:
        return "ERROR: No game started. Call start_game() first."

    parts = [
        f"ACTIONS_TAKEN: {_action_count}",
        f"GAME_ID: {_game_id}",
        f"SERVER: {_base_url}",
        "GAME_STARTED: true",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
