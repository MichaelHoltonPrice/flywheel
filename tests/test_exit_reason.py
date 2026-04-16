"""Tests for _classify_exit() exit reason classification."""

from __future__ import annotations

import json
from pathlib import Path

from flywheel.agent import _classify_exit


class TestClassifyExit:
    def test_stopped_returns_stopped(self, tmp_path: Path):
        assert _classify_exit(0, "exploration_request", tmp_path) == "stopped"

    def test_stopped_regardless_of_exit_code(self, tmp_path: Path):
        assert _classify_exit(1, "timeout", tmp_path) == "stopped"

    def test_auth_failure_from_state_file(self, tmp_path: Path):
        state = {"status": "paused", "reason": "auth_error"}
        (tmp_path / ".agent_state.json").write_text(json.dumps(state))
        assert _classify_exit(0, None, tmp_path) == "auth_failure"

    def test_auth_failure_substring_match(self, tmp_path: Path):
        state = {"status": "paused", "reason": "authentication failed"}
        (tmp_path / ".agent_state.json").write_text(json.dumps(state))
        assert _classify_exit(0, None, tmp_path) == "auth_failure"

    def test_rate_limit_from_state_file(self, tmp_path: Path):
        state = {"status": "paused", "reason": "rate_limit (retry 5/5)"}
        (tmp_path / ".agent_state.json").write_text(json.dumps(state))
        assert _classify_exit(0, None, tmp_path) == "rate_limit"

    def test_max_turns_from_state_file(self, tmp_path: Path):
        state = {"status": "complete", "reason": "max_turns"}
        (tmp_path / ".agent_state.json").write_text(json.dumps(state))
        assert _classify_exit(0, None, tmp_path) == "max_turns"

    def test_completed_from_state_file(self, tmp_path: Path):
        state = {"status": "complete", "reason": "success"}
        (tmp_path / ".agent_state.json").write_text(json.dumps(state))
        assert _classify_exit(0, None, tmp_path) == "completed"

    def test_completed_without_state_file(self, tmp_path: Path):
        assert _classify_exit(0, None, tmp_path) == "completed"

    def test_crashed_nonzero_no_state(self, tmp_path: Path):
        assert _classify_exit(1, None, tmp_path) == "crashed"

    def test_crashed_nonzero_with_empty_state(self, tmp_path: Path):
        (tmp_path / ".agent_state.json").write_text("{}")
        assert _classify_exit(1, None, tmp_path) == "crashed"

    def test_malformed_state_file_defaults(self, tmp_path: Path):
        (tmp_path / ".agent_state.json").write_text("not json")
        assert _classify_exit(0, None, tmp_path) == "completed"

    def test_malformed_state_file_nonzero(self, tmp_path: Path):
        (tmp_path / ".agent_state.json").write_text("not json")
        assert _classify_exit(1, None, tmp_path) == "crashed"

    def test_stop_reason_takes_precedence(self, tmp_path: Path):
        """Stop reason wins even if state file says auth_failure."""
        state = {"status": "paused", "reason": "auth_error"}
        (tmp_path / ".agent_state.json").write_text(json.dumps(state))
        assert _classify_exit(0, "exploration_request", tmp_path) == "stopped"
