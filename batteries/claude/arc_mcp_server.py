#!/usr/bin/env python3
"""ARC-AGI-3 game interaction MCP server (REST API client).

Provides tools for an agent to play ARC-AGI-3 games by calling a
remote game server via HTTP. The agent never sees the game source
code -- it only receives frames and state through the API, matching
the competition setup.

The game is initialized by the host before the agent starts.  The
agent receives ARC_CARD_ID and ARC_GUID as environment variables
and connects automatically on the first tool call.

After each action, the server POSTs step data to the host-side
block bridge in record mode, creating ``game_step`` and
``game_session`` artifacts with full provenance tracking. Tracking
failures are silently dropped -- they never break gameplay.

Environment variables:
    ARC_SERVER_URL  -- Base URL of the game server
                      (e.g., "http://host.docker.internal:8001").
    GAME_ID         -- Game being played (e.g., "vc33-9851e02b").
    ARC_CARD_ID     -- Scorecard ID (created by host).
    ARC_GUID        -- Game session GUID (created by host).
    EVAL_ENDPOINT   -- Block bridge URL for artifact tracking
                      (optional; no tracking if absent).
    ARC_SESSION_ARTIFACT_ID -- Initial game_session artifact ID
                      (optional; set by host for provenance chain).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arc")

# Global game state -- set from environment on first tool call.
_client: httpx.Client | None = None
_base_url: str = ""
_game_id: str = ""
_card_id: str = ""
_guid: str = ""
_action_count: int = 0

# Artifact tracking -- POST step data to the block bridge.
_bridge_endpoint: str = ""
_current_session_id: str = ""
_last_frame: list | None = None  # last known frame for pre-state
_last_score: int = 0


def _hydrate_initial_state() -> None:
    """Read the initial frame written by the host into tracking globals.

    The host writes ``.arc_initial_state.json`` to /workspace before
    the agent starts. This gives the first game_step a valid pre_state.
    """
    global _last_frame, _last_score
    state_path = Path("/workspace/.arc_initial_state.json")
    if not state_path.exists():
        return
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        _last_frame = data.get("frame")
        _last_score = data.get("score", 0)
    except Exception:
        pass  # Best-effort; don't break startup.


def _ensure_connected() -> str | None:
    """Initialize connection from environment variables on first call.

    Returns an error string if required env vars are missing, or
    None on success.
    """
    global _client, _base_url, _game_id, _card_id, _guid
    global _bridge_endpoint, _current_session_id

    if _client is not None:
        return None

    _base_url = os.environ.get("ARC_SERVER_URL", "")
    _game_id = os.environ.get("GAME_ID", "")
    _card_id = os.environ.get("ARC_CARD_ID", "")
    _guid = os.environ.get("ARC_GUID", "")

    if not _base_url:
        return "ERROR: ARC_SERVER_URL environment variable not set."
    if not _game_id:
        return "ERROR: GAME_ID environment variable not set."
    if not _card_id:
        return "ERROR: ARC_CARD_ID environment variable not set."
    if not _guid:
        return "ERROR: ARC_GUID environment variable not set."

    _client = httpx.Client(timeout=30.0)

    # Hydrate initial frame from host-written file so the first
    # game_step artifact has a valid pre_state.
    _hydrate_initial_state()

    # Artifact tracking (optional -- graceful if absent).
    _bridge_endpoint = os.environ.get("EVAL_ENDPOINT", "")
    _current_session_id = os.environ.get(
        "ARC_SESSION_ARTIFACT_ID", "")

    return None


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


def _record_step(
    action_desc: dict,
    pre_frame: list | None,
    post_data: dict,
    score_before: int,
    elapsed: float,
) -> None:
    """POST step data to the block bridge for artifact tracking.

    Silently drops errors — tracking failures must never break
    gameplay.
    """
    global _current_session_id

    if not _bridge_endpoint:
        return

    post_frame = post_data.get("frame", [[]])[0]
    score_after = post_data.get("score", score_before)

    try:
        resp = httpx.post(
            _bridge_endpoint,
            json={
                "mode": "record",
                "block_name": "game_step",
                "inputs": {
                    "game_session": _current_session_id,
                } if _current_session_id else {},
                "outputs": {
                    "game_session": {
                        "card_id": _card_id,
                        "guid": _guid,
                        "level": post_data.get(
                            "levels_completed", 0),
                        "action_count": _action_count,
                        "state": post_data.get(
                            "state", "UNKNOWN"),
                    },
                    "game_step": {
                        "step_index": _action_count,
                        "action": action_desc,
                        "pre_state": pre_frame,
                        "post_state": post_frame,
                        "score_before": score_before,
                        "score_after": score_after,
                        "level": post_data.get(
                            "levels_completed", 0),
                        "state": post_data.get(
                            "state", "UNKNOWN"),
                    },
                },
                "elapsed_s": elapsed,
            },
            timeout=10.0,
        )
        data = resp.json()
        if data.get("ok"):
            new_session = data.get("game_session_artifact_id", "")
            if new_session:
                _current_session_id = new_session
    except Exception:
        pass  # Never break gameplay on tracking failure.


def _post(endpoint: str, payload: dict) -> dict:
    """POST to the game server and return JSON response."""
    if _client is None:
        return {"error": "Not connected."}
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
def take_action(action: int, x: int = -1, y: int = -1) -> str:
    """Take a game action and return the resulting frame.

    Args:
        action: Action number (1-5 for simple actions, 6 for click
            with coordinates, 7 for undo). Do NOT send 0 (RESET)
            here -- use reset_level() instead.
        x: X coordinate (0-63) for ACTION6 clicks. Ignored for
            other actions.
        y: Y coordinate (0-63) for ACTION6 clicks. Ignored for
            other actions.

    Returns:
        The new game state including the frame, state, levels
        completed, and available actions.
    """
    global _action_count, _guid, _last_frame, _last_score

    err = _ensure_connected()
    if err:
        return err

    if action < 1 or action > 7:
        return f"ERROR: Invalid action {action}. Must be 1-7."

    payload = {}
    if action == 6:
        payload["x"] = x
        payload["y"] = y

    pre_frame = _last_frame
    score_before = _last_score
    t0 = time.monotonic()

    _action_count += 1
    data = _post(f"ACTION{action}", payload)

    if "error" in data:
        _action_count -= 1  # Don't count failed actions.
        return f"ERROR: {data['error']} {data.get('detail', '')}"

    elapsed = time.monotonic() - t0

    if "guid" in data:
        _guid = data["guid"]

    # Update tracking state.
    post_frame = data.get("frame", [[]])[0]
    _last_frame = post_frame
    _last_score = data.get("score", _last_score)

    desc = f"ACTION{action}"
    action_desc: dict = {"action": action}
    if action == 6:
        desc = f"ACTION6(x={x}, y={y})"
        action_desc["x"] = x
        action_desc["y"] = y

    _record_step(action_desc, pre_frame, data, score_before, elapsed)

    return _format_response(data, desc)


@mcp.tool()
def reset_level() -> str:
    """Reset the current level (sends RESET action).

    Use this to restart the current level if you're stuck or
    want to try a different approach.
    """
    global _action_count, _guid, _last_frame, _last_score

    err = _ensure_connected()
    if err:
        return err

    pre_frame = _last_frame
    score_before = _last_score
    t0 = time.monotonic()

    _action_count += 1
    data = _post("RESET", {})
    if "error" in data:
        _action_count -= 1  # Don't count failed resets.
        return f"ERROR: {data['error']} {data.get('detail', '')}"

    elapsed = time.monotonic() - t0

    if "guid" in data:
        _guid = data["guid"]

    # Update tracking state.
    post_frame = data.get("frame", [[]])[0]
    _last_frame = post_frame
    _last_score = data.get("score", _last_score)

    _record_step(
        {"action": 0, "type": "RESET"},
        pre_frame, data, score_before, elapsed,
    )

    return _format_response(data, "RESET (level restart)")


@mcp.tool()
def get_status() -> str:
    """Get current game status without taking an action.

    Returns: actions taken and connection status.
    """
    err = _ensure_connected()
    if err:
        return err

    parts = [
        f"ACTIONS_TAKEN: {_action_count}",
        f"GAME_ID: {_game_id}",
        f"SERVER: {_base_url}",
        f"CARD_ID: {_card_id}",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
