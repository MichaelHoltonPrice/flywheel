#!/usr/bin/env python3
"""ARC-AGI-3 game interaction MCP server.

Provides tools for an agent to play ARC-AGI-3 games: start a game,
take actions, and query game state. Runs as a stdio subprocess
inside the agent container.

Environment variables:
    GAME_ID         — Game to play (e.g., "vc33-9851e02b").
    ENVIRONMENTS_DIR — Path to cached game files (default: /game_files).
"""

from __future__ import annotations

import os

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine.enums import GameAction
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arc")

# Global game state.
_arcade: Arcade | None = None
_env = None
_action_count: int = 0
_last_frame: np.ndarray | None = None


def _frame_to_text(frame: np.ndarray) -> str:
    """Convert a 64x64 numpy frame to a compact text representation.

    Each row is a line of space-separated integers. This is more
    token-efficient than JSON nested lists and easy to parse with
    Python.
    """
    lines = []
    for row in frame:
        lines.append(" ".join(str(int(v)) for v in row))
    return "\n".join(lines)


def _format_obs(obs, action_desc: str = "") -> str:
    """Format an observation as a structured text block for the agent."""
    global _last_frame, _action_count

    frame = obs.frame[0] if obs.frame else None
    if frame is not None:
        _last_frame = frame

    parts = []
    if action_desc:
        parts.append(f"ACTION: {action_desc}")
    parts.append(f"ACTIONS_TAKEN: {_action_count}")
    parts.append(f"STATE: {obs.state.name}")
    parts.append(f"LEVELS_COMPLETED: {obs.levels_completed}")
    parts.append(f"WIN_LEVELS: {obs.win_levels}")
    parts.append(f"AVAILABLE_ACTIONS: {obs.available_actions}")

    if frame is not None:
        # Include frame dimensions and unique colors.
        unique = np.unique(frame)
        parts.append(f"FRAME_SHAPE: {frame.shape[0]}x{frame.shape[1]}")
        parts.append(f"COLORS_PRESENT: {unique.tolist()}")
        parts.append("FRAME:")
        parts.append(_frame_to_text(frame))

    return "\n".join(parts)


@mcp.tool()
def start_game() -> str:
    """Start (or restart) the current game. Returns the initial frame.

    The game is specified by the GAME_ID environment variable.
    Call this once at the beginning. Use reset_level() to restart
    the current level.
    """
    global _arcade, _env, _action_count, _last_frame

    game_id = os.environ.get("GAME_ID", "")
    if not game_id:
        return "ERROR: GAME_ID environment variable not set."

    envs_dir = os.environ.get("ENVIRONMENTS_DIR", "/game_files")

    _arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=envs_dir,
    )
    _env = _arcade.make(game_id)
    _action_count = 0
    _last_frame = None

    obs = _env.reset()
    return _format_obs(obs, "start_game")


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
    global _action_count

    if _env is None:
        return "ERROR: No game started. Call start_game() first."

    if action < 1 or action > 7:
        return f"ERROR: Invalid action {action}. Must be 1-7."

    game_action = GameAction(action)
    data = {"x": x, "y": y} if action == 6 else None

    _action_count += 1
    obs = _env.step(game_action, data=data)

    if obs is None:
        return f"ERROR: step() returned None for action {action}."

    desc = f"ACTION{action}"
    if action == 6:
        desc = f"ACTION6(x={x}, y={y})"

    return _format_obs(obs, desc)


@mcp.tool()
def reset_level() -> str:
    """Reset the current level (sends RESET action).

    Use this to restart the current level if you're stuck or
    want to try a different approach. Does not count toward
    your action total.
    """
    if _env is None:
        return "ERROR: No game started. Call start_game() first."

    obs = _env.step(GameAction.RESET)
    if obs is None:
        return "ERROR: reset returned None."

    return _format_obs(obs, "RESET (level restart)")


@mcp.tool()
def get_status() -> str:
    """Get current game status without taking an action.

    Returns: actions taken, current state, levels completed,
    and frame dimensions.
    """
    if _env is None:
        return "ERROR: No game started. Call start_game() first."

    parts = [
        f"ACTIONS_TAKEN: {_action_count}",
        "GAME_STARTED: true",
    ]

    if _last_frame is not None:
        unique = np.unique(_last_frame)
        parts.append(f"FRAME_SHAPE: {_last_frame.shape[0]}x{_last_frame.shape[1]}")
        parts.append(f"COLORS_PRESENT: {unique.tolist()}")

    return "\n".join(parts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
