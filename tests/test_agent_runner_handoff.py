"""Unit tests for the agent runner's handoff hook plumbing.

The hook is what turns a tool_use block into a pending entry on
disk.  Multi-tool-use turns rely on the hook firing once per
intercepted tool_use within a single assistant message and
appending each capture into the same pending list.  These tests
exercise that contract without an SDK, without a model, and
without a container.

Also covers the on-disk envelope that ``_persist_handoff`` writes
(``schema_version`` 2, plural ``pending`` array) so the host-side
driver's ``_normalize_pending`` decoder is being fed what the
runner actually emits.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from flywheel.agent_loop_driver import _normalize_pending


def _import_runner():
    """Import ``agent_runner`` with all SDK / heavy deps mocked."""
    sdk_mock = type(sys)("claude_agent_sdk")
    sdk_mock.ClaudeAgentOptions = object
    sdk_mock.ClaudeSDKClient = object
    sdk_mock.HookMatcher = object
    types_mock = type(sys)("claude_agent_sdk.types")
    types_mock.AssistantMessage = object
    types_mock.ResultMessage = object

    with patch.dict(sys.modules, {
        "claude_agent_sdk": sdk_mock,
        "claude_agent_sdk.types": types_mock,
        "anyio": type(sys)("anyio"),
        "mcp": type(sys)("mcp"),
        "mcp.server": type(sys)("mcp.server"),
        "mcp.server.fastmcp": type(sys)("mcp.server.fastmcp"),
    }):
        mod_name = "batteries.claude.agent_runner"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        batteries_path = str(
            Path(__file__).parent.parent / "batteries" / "claude")
        for p in [batteries_path]:
            if p not in sys.path:
                sys.path.insert(0, p)

        spec = importlib.util.spec_from_file_location(
            "agent_runner",
            Path(__file__).parent.parent
            / "batteries" / "claude" / "agent_runner.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_runner = _import_runner()


@pytest.fixture(autouse=True)
def _reset_handoff_state():
    """Each test gets a fresh empty pending list."""
    _runner._HANDOFF_STATE["pending"] = []
    yield
    _runner._HANDOFF_STATE["pending"] = []


def _call_hook(hook, *, tool_name: str, tool_input: dict, tool_use_id):
    """Invoke the async PreToolUse hook synchronously for testing."""
    return asyncio.run(hook(
        {"tool_name": tool_name, "tool_input": tool_input},
        tool_use_id,
        context=None,
    ))


# --------------------------------------------------------------------
# _build_handoff_hook: what the PreToolUse hook does.
# --------------------------------------------------------------------


class TestBuildHandoffHook:
    """Tests for the PreToolUse hook factory.

    The hook factory closes over a set of handoff-mapped tool names
    and a deny marker.  The returned hook:
      * Captures every intercepted tool_use into the shared pending
        list (one append per call, preserving order).
      * Returns a deny decision with the marker for every match.
      * Returns ``{}`` (proceed normally) for non-matched tools.
      * Returns ``{}`` when ``tool_use_id`` is missing.
    """

    def test_unmapped_tool_passes_through(self):
        hook = _runner._build_handoff_hook(
            handoff_tools={"mcp__demo__handoff_me"},
            deny_marker="handoff_to_flywheel",
        )
        result = _call_hook(
            hook,
            tool_name="mcp__demo__some_other",
            tool_input={"x": 1},
            tool_use_id="toolu_unrelated",
        )
        assert result == {}
        assert _runner._HANDOFF_STATE["pending"] == []

    def test_missing_tool_use_id_passes_through(self):
        """Without a tool_use_id we cannot key a splice, so we
        must let the SDK proceed rather than capture half-data."""
        hook = _runner._build_handoff_hook(
            handoff_tools={"mcp__demo__handoff_me"},
            deny_marker="handoff_to_flywheel",
        )
        result = _call_hook(
            hook,
            tool_name="mcp__demo__handoff_me",
            tool_input={"x": 1},
            tool_use_id=None,
        )
        assert result == {}
        assert _runner._HANDOFF_STATE["pending"] == []

    def test_matched_tool_captures_and_denies(self):
        hook = _runner._build_handoff_hook(
            handoff_tools={"mcp__demo__handoff_me"},
            deny_marker="handoff_to_flywheel",
        )
        result = _call_hook(
            hook,
            tool_name="mcp__demo__handoff_me",
            tool_input={"value": 7},
            tool_use_id="toolu_one",
        )
        decision = result["hookSpecificOutput"]
        assert decision["hookEventName"] == "PreToolUse"
        assert decision["permissionDecision"] == "deny"
        assert "handoff_to_flywheel" in decision["permissionDecisionReason"]

        pending = _runner._HANDOFF_STATE["pending"]
        assert len(pending) == 1
        entry = pending[0]
        assert entry["tool_use_id"] == "toolu_one"
        assert entry["tool_name"] == "mcp__demo__handoff_me"
        assert entry["tool_input"] == {"value": 7}
        assert "captured_at" in entry

    def test_multiple_matched_tool_uses_in_one_turn_all_captured(self):
        """The core multi-tool-use contract: when the SDK fires
        the hook once per tool_use in a single assistant message,
        every match appends to the pending list in emission order."""
        hook = _runner._build_handoff_hook(
            handoff_tools={
                "mcp__demo__double_it",
                "mcp__demo__triple_it",
            },
            deny_marker="handoff_to_flywheel",
        )
        first = _call_hook(
            hook,
            tool_name="mcp__demo__double_it",
            tool_input={"value": 7},
            tool_use_id="toolu_first",
        )
        second = _call_hook(
            hook,
            tool_name="mcp__demo__triple_it",
            tool_input={"value": 5},
            tool_use_id="toolu_second",
        )
        third = _call_hook(
            hook,
            tool_name="mcp__demo__double_it",
            tool_input={"value": 11},
            tool_use_id="toolu_third",
        )

        for decision in (first, second, third):
            assert decision["hookSpecificOutput"][
                "permissionDecision"] == "deny"
            assert "handoff_to_flywheel" in decision[
                "hookSpecificOutput"]["permissionDecisionReason"]

        pending = _runner._HANDOFF_STATE["pending"]
        assert [e["tool_use_id"] for e in pending] == [
            "toolu_first", "toolu_second", "toolu_third",
        ]
        assert [e["tool_name"] for e in pending] == [
            "mcp__demo__double_it",
            "mcp__demo__triple_it",
            "mcp__demo__double_it",
        ]
        assert [e["tool_input"]["value"] for e in pending] == [7, 5, 11]

    def test_mixed_matched_and_unmatched_in_one_turn(self):
        """A turn with one mapped + one unmapped tool: only the
        mapped one is captured; the unmapped one passes through
        with an empty hook decision."""
        hook = _runner._build_handoff_hook(
            handoff_tools={"mcp__demo__handoff_me"},
            deny_marker="handoff_to_flywheel",
        )
        passthrough = _call_hook(
            hook,
            tool_name="mcp__demo__plain_tool",
            tool_input={"a": 1},
            tool_use_id="toolu_plain",
        )
        captured = _call_hook(
            hook,
            tool_name="mcp__demo__handoff_me",
            tool_input={"b": 2},
            tool_use_id="toolu_handoff",
        )
        assert passthrough == {}
        assert captured["hookSpecificOutput"][
            "permissionDecision"] == "deny"
        pending = _runner._HANDOFF_STATE["pending"]
        assert len(pending) == 1
        assert pending[0]["tool_use_id"] == "toolu_handoff"

    def test_custom_deny_marker_threaded_through(self):
        hook = _runner._build_handoff_hook(
            handoff_tools={"mcp__demo__handoff_me"},
            deny_marker="custom_marker_xyz",
        )
        result = _call_hook(
            hook,
            tool_name="mcp__demo__handoff_me",
            tool_input={},
            tool_use_id="toolu_custom",
        )
        reason = result["hookSpecificOutput"]["permissionDecisionReason"]
        assert "custom_marker_xyz" in reason
        assert "handoff_to_flywheel" not in reason


# --------------------------------------------------------------------
# _persist_handoff: what lands on disk.
# --------------------------------------------------------------------


class TestPersistHandoff:
    """The schema the runner writes to ``pending_tool_calls.json``.

    The host-side driver's ``_normalize_pending`` decoder must be
    able to read what the runner emits, so the schema is part of a
    cross-module contract: ``schema_version`` 2 with a plural
    ``pending`` array.  These tests pin the wire format.
    """

    def test_no_pending_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _runner, "PENDING_TOOL_CALLS_FILE",
            tmp_path / "pending_tool_calls.json",
        )
        _runner._persist_handoff(session_id="sess-empty")
        assert not (tmp_path / "pending_tool_calls.json").exists()

    def test_single_pending_envelope(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _runner, "PENDING_TOOL_CALLS_FILE",
            tmp_path / "pending_tool_calls.json",
        )
        _runner._HANDOFF_STATE["pending"] = [
            {
                "tool_use_id": "toolu_solo",
                "tool_name": "mcp__demo__handoff_me",
                "tool_input": {"value": 7},
                "captured_at": "2026-04-17T15:23:01+00:00",
            },
        ]
        _runner._persist_handoff(session_id="sess-solo")

        doc = json.loads(
            (tmp_path / "pending_tool_calls.json").read_text(
                encoding="utf-8"))
        assert doc["schema_version"] == 2
        assert doc["session_id"] == "sess-solo"
        assert len(doc["pending"]) == 1
        entry = doc["pending"][0]
        assert entry["tool_use_id"] == "toolu_solo"
        assert entry["tool_name"] == "mcp__demo__handoff_me"
        assert entry["tool_input"] == {"value": 7}

    def test_multi_pending_preserves_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _runner, "PENDING_TOOL_CALLS_FILE",
            tmp_path / "pending_tool_calls.json",
        )
        _runner._HANDOFF_STATE["pending"] = [
            {
                "tool_use_id": f"toolu_{i}",
                "tool_name": "mcp__demo__handoff_me",
                "tool_input": {"i": i},
                "captured_at": f"2026-04-17T15:23:0{i}+00:00",
            }
            for i in range(4)
        ]
        _runner._persist_handoff(session_id="sess-batch")

        doc = json.loads(
            (tmp_path / "pending_tool_calls.json").read_text(
                encoding="utf-8"))
        assert doc["schema_version"] == 2
        assert doc["session_id"] == "sess-batch"
        assert [e["tool_use_id"] for e in doc["pending"]] == [
            "toolu_0", "toolu_1", "toolu_2", "toolu_3",
        ]
        assert [e["tool_input"]["i"] for e in doc["pending"]] == [
            0, 1, 2, 3,
        ]


# --------------------------------------------------------------------
# Cross-module contract: what the runner writes is what the driver
# reads.  Catches schema drift between the two modules.
# --------------------------------------------------------------------


def test_runner_envelope_decodes_via_driver_normalize_pending(
    tmp_path, monkeypatch,
):
    """The runner writes ``pending_tool_calls.json``; the driver
    decodes it via ``_normalize_pending``.  This test pins that the
    two agree on the on-disk shape so a schema change in one cannot
    silently break the other.
    """
    monkeypatch.setattr(
        _runner, "PENDING_TOOL_CALLS_FILE",
        tmp_path / "pending_tool_calls.json",
    )
    _runner._HANDOFF_STATE["pending"] = [
        {
            "tool_use_id": "toolu_a",
            "tool_name": "mcp__demo__one",
            "tool_input": {"x": 1},
            "captured_at": "2026-04-17T15:23:01+00:00",
        },
        {
            "tool_use_id": "toolu_b",
            "tool_name": "mcp__demo__two",
            "tool_input": {"y": 2},
            "captured_at": "2026-04-17T15:23:02+00:00",
        },
    ]
    _runner._persist_handoff(session_id="sess-cross")
    doc = json.loads(
        (tmp_path / "pending_tool_calls.json").read_text(
            encoding="utf-8"))

    decoded = _normalize_pending(doc)
    assert [e["tool_use_id"] for e in decoded] == ["toolu_a", "toolu_b"]
    assert [e["tool_name"] for e in decoded] == [
        "mcp__demo__one", "mcp__demo__two",
    ]
    assert [e["tool_input"] for e in decoded] == [{"x": 1}, {"y": 2}]
