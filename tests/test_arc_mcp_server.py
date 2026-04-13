"""Tests for the ARC-AGI-3 MCP server (batteries/claude/arc_mcp_server.py).

Uses mocked HTTP calls and file I/O to test game interaction logic,
action counting, initial state hydration, and artifact tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import batteries.claude.arc_mcp_server as srv


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset all module-level globals before each test."""
    srv._client = None
    srv._base_url = ""
    srv._game_id = ""
    srv._card_id = ""
    srv._guid = ""
    srv._action_count = 0
    srv._bridge_endpoint = ""
    srv._current_session_id = ""
    srv._last_frame = None
    srv._last_score = 0
    yield
    srv._client = None
    srv._base_url = ""
    srv._game_id = ""
    srv._card_id = ""
    srv._guid = ""
    srv._action_count = 0
    srv._bridge_endpoint = ""
    srv._current_session_id = ""
    srv._last_frame = None
    srv._last_score = 0


def _fake_env(extra: dict | None = None) -> dict:
    """Return a minimal set of env vars for connection."""
    env = {
        "ARC_SERVER_URL": "http://localhost:8001",
        "GAME_ID": "vc33-test",
        "ARC_CARD_ID": "card123",
        "ARC_GUID": "guid456",
    }
    if extra:
        env.update(extra)
    return env


def _game_response(
    frame: list | None = None,
    score: int = 0,
    state: str = "NOT_FINISHED",
    levels_completed: int = 0,
    guid: str = "guid456",
) -> dict:
    """Build a fake game server response."""
    resp: dict = {
        "state": state,
        "score": score,
        "levels_completed": levels_completed,
        "win_levels": 7,
        "available_actions": [6],
        "guid": guid,
    }
    if frame is not None:
        resp["frame"] = [frame]
    else:
        resp["frame"] = [[[0, 1], [2, 3]]]
    return resp


def _connect(env: dict | None = None, initial_frame: list | None = None):
    """Set globals as if _ensure_connected succeeded."""
    e = _fake_env(env)
    srv._client = MagicMock()
    srv._base_url = e["ARC_SERVER_URL"]
    srv._game_id = e["GAME_ID"]
    srv._card_id = e["ARC_CARD_ID"]
    srv._guid = e["ARC_GUID"]
    srv._bridge_endpoint = e.get("EVAL_ENDPOINT", "")
    srv._current_session_id = e.get("ARC_SESSION_ARTIFACT_ID", "")
    if initial_frame is not None:
        srv._last_frame = initial_frame
        srv._last_score = 0


class TestHydrateInitialState:
    def test_reads_initial_state_file(self, tmp_path: Path):
        state_file = tmp_path / ".arc_initial_state.json"
        state_file.write_text(json.dumps({
            "frame": [[1, 2], [3, 4]],
            "score": 5,
        }))

        with patch(
            "batteries.claude.arc_mcp_server.Path",
        ) as mock_path:
            mock_path.return_value = state_file
            srv._hydrate_initial_state()

        assert srv._last_frame == [[1, 2], [3, 4]]
        assert srv._last_score == 5

    def test_missing_file_is_noop(self):
        with patch(
            "batteries.claude.arc_mcp_server.Path",
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            srv._hydrate_initial_state()

        assert srv._last_frame is None
        assert srv._last_score == 0

    def test_malformed_json_is_noop(self, tmp_path: Path):
        state_file = tmp_path / ".arc_initial_state.json"
        state_file.write_text("not valid json {{{")

        with patch(
            "batteries.claude.arc_mcp_server.Path",
        ) as mock_path:
            mock_path.return_value = state_file
            srv._hydrate_initial_state()

        assert srv._last_frame is None
        assert srv._last_score == 0


class TestTakeAction:
    def test_increments_action_count(self):
        _connect(initial_frame=[[0, 0], [0, 0]])
        resp = _game_response()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = resp
        srv._client.post.return_value = mock_response

        assert srv._action_count == 0
        srv.take_action(6, x=10, y=20)
        assert srv._action_count == 1
        srv.take_action(6, x=11, y=21)
        assert srv._action_count == 2

    def test_rollback_on_error(self):
        _connect()
        error_response = MagicMock()
        error_response.status_code = 200
        error_response.json.return_value = {"error": "bad action"}
        srv._client.post.return_value = error_response

        srv._action_count = 5
        result = srv.take_action(6, x=0, y=0)
        assert "ERROR" in result
        assert srv._action_count == 5  # Rolled back.

    def test_updates_last_frame_and_score(self):
        _connect(initial_frame=[[0, 0], [0, 0]])
        new_frame = [[1, 1], [1, 1]]
        resp = _game_response(frame=new_frame, score=3)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = resp
        srv._client.post.return_value = mock_response

        srv.take_action(6, x=5, y=5)
        assert srv._last_frame == new_frame
        assert srv._last_score == 3

    def test_invalid_action_rejected(self):
        _connect()
        result = srv.take_action(0)
        assert "ERROR" in result
        assert srv._action_count == 0

    def test_pre_state_from_initial_hydration(self):
        """First action uses the hydrated initial frame as pre_state."""
        initial = [[7, 7], [3, 3]]
        _connect(initial_frame=initial, env={
            "EVAL_ENDPOINT": "http://bridge:9000",
        })

        post_frame = [[7, 7], [3, 0]]
        resp = _game_response(frame=post_frame, score=0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = resp
        srv._client.post.return_value = mock_response

        recorded_steps = []

        def capture_record(*args, **kwargs):
            recorded_steps.append(args)

        with patch.object(srv, "_record_step", side_effect=capture_record):
            srv.take_action(6, x=1, y=1)

        assert len(recorded_steps) == 1
        # pre_frame should be the initial hydrated frame, not None.
        pre_frame = recorded_steps[0][1]
        assert pre_frame == initial


class TestResetLevel:
    def test_increments_action_count(self):
        _connect()
        resp = _game_response()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = resp
        srv._client.post.return_value = mock_response

        assert srv._action_count == 0
        srv.reset_level()
        assert srv._action_count == 1

    def test_rollback_on_error(self):
        _connect()
        error_response = MagicMock()
        error_response.status_code = 200
        error_response.json.return_value = {"error": "server error"}
        srv._client.post.return_value = error_response

        srv._action_count = 3
        result = srv.reset_level()
        assert "ERROR" in result
        assert srv._action_count == 3  # Rolled back.

    def test_monotonic_with_actions(self):
        """Actions and resets produce monotonically increasing counts."""
        _connect(initial_frame=[[0]])
        resp = _game_response()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = resp
        srv._client.post.return_value = mock_response

        srv.take_action(6, x=0, y=0)  # count = 1
        srv.take_action(6, x=1, y=1)  # count = 2
        srv.reset_level()              # count = 3
        srv.take_action(6, x=2, y=2)  # count = 4

        assert srv._action_count == 4


class TestRecordStep:
    def test_posts_to_bridge(self):
        srv._bridge_endpoint = "http://bridge:9000"
        srv._current_session_id = "game_session@init"
        srv._card_id = "card1"
        srv._guid = "guid1"
        srv._action_count = 1

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ok": True,
            "game_session_artifact_id": "game_session@step1",
        }

        with patch("batteries.claude.arc_mcp_server.httpx.post",
                    return_value=mock_resp) as mock_post:
            srv._record_step(
                {"action": 6, "x": 10, "y": 20},
                [[0, 0]],  # pre_frame
                {"frame": [[[1, 1]]], "score": 0,
                 "levels_completed": 0, "state": "NOT_FINISHED"},
                0,  # score_before
                0.5,  # elapsed
            )

        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["mode"] == "record"
        assert payload["block_name"] == "game_step"
        assert payload["inputs"]["game_session"] == "game_session@init"
        assert payload["outputs"]["game_step"]["step_index"] == 1
        assert payload["outputs"]["game_step"]["pre_state"] == [[0, 0]]

    def test_updates_session_id_from_response(self):
        srv._bridge_endpoint = "http://bridge:9000"
        srv._current_session_id = "game_session@init"
        srv._card_id = "c1"
        srv._guid = "g1"
        srv._action_count = 1

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ok": True,
            "game_session_artifact_id": "game_session@new",
        }

        with patch("batteries.claude.arc_mcp_server.httpx.post",
                    return_value=mock_resp):
            srv._record_step(
                {"action": 1}, None,
                {"frame": [[[0]]], "score": 0}, 0, 0.1,
            )

        assert srv._current_session_id == "game_session@new"

    def test_no_bridge_is_noop(self):
        srv._bridge_endpoint = ""  # No bridge configured.
        srv._action_count = 1

        with patch("batteries.claude.arc_mcp_server.httpx.post") as mock_post:
            srv._record_step(
                {"action": 1}, None,
                {"frame": [[[0]]], "score": 0}, 0, 0.1,
            )

        mock_post.assert_not_called()

    def test_bridge_error_is_silent(self):
        srv._bridge_endpoint = "http://bridge:9000"
        srv._current_session_id = "game_session@init"
        srv._card_id = "c1"
        srv._guid = "g1"
        srv._action_count = 1

        with patch("batteries.claude.arc_mcp_server.httpx.post",
                    side_effect=ConnectionError("bridge down")):
            # Should not raise.
            srv._record_step(
                {"action": 1}, None,
                {"frame": [[[0]]], "score": 0}, 0, 0.1,
            )

        # Session ID unchanged on failure.
        assert srv._current_session_id == "game_session@init"


class TestEnsureConnected:
    def test_missing_env_var_returns_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ARC_SERVER_URL", raising=False)
        monkeypatch.delenv("GAME_ID", raising=False)
        monkeypatch.delenv("ARC_CARD_ID", raising=False)
        monkeypatch.delenv("ARC_GUID", raising=False)

        err = srv._ensure_connected()
        assert err is not None
        assert "ARC_SERVER_URL" in err

    def test_connects_with_all_vars(self, monkeypatch: pytest.MonkeyPatch):
        for key, val in _fake_env().items():
            monkeypatch.setenv(key, val)

        with patch(
            "batteries.claude.arc_mcp_server._hydrate_initial_state",
        ):
            err = srv._ensure_connected()

        assert err is None
        assert srv._client is not None
        assert srv._base_url == "http://localhost:8001"

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch):
        for key, val in _fake_env().items():
            monkeypatch.setenv(key, val)

        with patch(
            "batteries.claude.arc_mcp_server._hydrate_initial_state",
        ):
            srv._ensure_connected()
            first_client = srv._client
            srv._ensure_connected()
            assert srv._client is first_client
