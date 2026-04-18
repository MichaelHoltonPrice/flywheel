"""Unit tests for ``flywheel.session_splice``.

Covers the algorithm against synthetic JSONL fixtures shaped like
the Claude Agent SDK's saved sessions.  The shapes are documented
in ``plans/full-stop-state-contract.md``; the live-API integration
test under ``tests/integration/`` validates that the assumed shapes
match what the SDK actually writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywheel.session_splice import (
    SpliceError,
    find_pending_deny_tool_use_ids,
    splice_tool_result,
)


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    """Write entries as newline-delimited JSON, with trailing newline."""
    path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assistant_with_tool_use(
    tool_use_id: str, tool_name: str = "mcp__test__do_thing",
) -> dict:
    return {
        "type": "assistant",
        "sessionId": "session-uuid",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll call the tool now."},
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": tool_name,
                    "input": {"x": 1, "y": 2},
                },
            ],
        },
    }


def _user_with_deny(
    tool_use_id: str,
    deny_text: str = (
        "permission denied: handoff_to_flywheel"
    ),
) -> dict:
    return {
        "type": "user",
        "sessionId": "session-uuid",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [
                        {"type": "text", "text": deny_text},
                    ],
                    "is_error": True,
                },
            ],
        },
    }


class TestSpliceHappyPath:

    def test_replaces_text_payload(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [
            {"type": "user", "sessionId": "s",
             "message": {"role": "user",
                         "content": [{"type": "text",
                                      "text": "hi"}]}},
            _assistant_with_tool_use("toolu_X"),
            _user_with_deny("toolu_X"),
        ])

        line = splice_tool_result(
            path,
            tool_use_id="toolu_X",
            tool_result_content="real result text",
        )
        assert line == 3

        entries = _read_jsonl(path)
        result_block = entries[2]["message"]["content"][0]
        assert result_block["type"] == "tool_result"
        assert result_block["tool_use_id"] == "toolu_X"
        assert result_block["is_error"] is False
        assert result_block["content"] == [
            {"type": "text", "text": "real result text"},
        ]

    def test_replaces_list_payload(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            _user_with_deny("toolu_X"),
        ])

        payload = [
            {"type": "text", "text": "outputs:"},
            {"type": "text", "text": "score=42"},
        ]
        splice_tool_result(
            path,
            tool_use_id="toolu_X",
            tool_result_content=payload,
        )

        result_block = _read_jsonl(path)[1]["message"]["content"][0]
        assert result_block["content"] == payload
        assert result_block["is_error"] is False

    def test_marks_error_when_block_failed(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            _user_with_deny("toolu_X"),
        ])

        splice_tool_result(
            path,
            tool_use_id="toolu_X",
            tool_result_content="block raised RuntimeError",
            is_error=True,
        )

        result_block = _read_jsonl(path)[1]["message"]["content"][0]
        assert result_block["is_error"] is True

    def test_preserves_other_fields_on_block(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        deny = _user_with_deny("toolu_X")
        deny["message"]["content"][0]["custom_meta"] = "preserve_me"
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            deny,
        ])

        splice_tool_result(
            path,
            tool_use_id="toolu_X",
            tool_result_content="real",
        )

        result_block = _read_jsonl(path)[1]["message"]["content"][0]
        assert result_block["custom_meta"] == "preserve_me"

    def test_preserves_other_blocks_in_same_envelope(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        deny = _user_with_deny("toolu_X")
        deny["message"]["content"].append({
            "type": "tool_result",
            "tool_use_id": "toolu_OTHER",
            "content": [{"type": "text", "text": "untouched"}],
            "is_error": False,
        })
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            deny,
        ])

        splice_tool_result(
            path,
            tool_use_id="toolu_X",
            tool_result_content="real",
        )

        blocks = _read_jsonl(path)[1]["message"]["content"]
        spliced = next(b for b in blocks
                       if b["tool_use_id"] == "toolu_X")
        untouched = next(b for b in blocks
                         if b["tool_use_id"] == "toolu_OTHER")
        assert spliced["content"] == [
            {"type": "text", "text": "real"}]
        assert untouched["content"] == [
            {"type": "text", "text": "untouched"}]

    def test_supports_bare_content_envelope_shape(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        bare_user = {
            "type": "user",
            "sessionId": "s",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_X",
                    "content": "deny: handoff_to_flywheel",
                    "is_error": True,
                },
            ],
        }
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            bare_user,
        ])

        splice_tool_result(
            path,
            tool_use_id="toolu_X",
            tool_result_content="real",
        )

        result_block = _read_jsonl(path)[1]["content"][0]
        assert result_block["content"] == [
            {"type": "text", "text": "real"}]


class TestSpliceErrors:

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SpliceError, match="not found"):
            splice_tool_result(
                tmp_path / "nope.jsonl",
                tool_use_id="x",
                tool_result_content="r",
            )

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(SpliceError, match="empty"):
            splice_tool_result(
                path, tool_use_id="x", tool_result_content="r")

    def test_invalid_json_line(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(SpliceError, match="invalid JSON"):
            splice_tool_result(
                path, tool_use_id="x", tool_result_content="r")

    def test_no_match_raises_and_leaves_file_unchanged(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        original = [
            _assistant_with_tool_use("toolu_X"),
            _user_with_deny("toolu_X"),
        ]
        _write_jsonl(path, original)
        original_text = path.read_text(encoding="utf-8")

        with pytest.raises(SpliceError, match="not found"):
            splice_tool_result(
                path,
                tool_use_id="toolu_NOT_THERE",
                tool_result_content="r",
            )

        assert path.read_text(encoding="utf-8") == original_text

    def test_two_envelopes_match_same_id_raises(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            _user_with_deny("toolu_X"),
            _user_with_deny("toolu_X"),
        ])
        original_text = path.read_text(encoding="utf-8")

        with pytest.raises(SpliceError, match="invariant violation"):
            splice_tool_result(
                path, tool_use_id="toolu_X",
                tool_result_content="r")

        assert path.read_text(encoding="utf-8") == original_text

    def test_two_blocks_in_one_envelope_raises(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        deny = _user_with_deny("toolu_X")
        deny["message"]["content"].append(
            dict(deny["message"]["content"][0]))
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            deny,
        ])
        original_text = path.read_text(encoding="utf-8")

        with pytest.raises(SpliceError, match="invariant violation"):
            splice_tool_result(
                path, tool_use_id="toolu_X",
                tool_result_content="r")

        assert path.read_text(encoding="utf-8") == original_text

    def test_invalid_payload_type_raises(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            _user_with_deny("toolu_X"),
        ])

        with pytest.raises(SpliceError, match="must be str or list"):
            splice_tool_result(
                path,
                tool_use_id="toolu_X",
                tool_result_content=42,  # type: ignore[arg-type]
            )

    def test_invalid_block_in_payload_raises(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            _user_with_deny("toolu_X"),
        ])

        with pytest.raises(SpliceError, match="missing 'type'"):
            splice_tool_result(
                path,
                tool_use_id="toolu_X",
                tool_result_content=[{"text": "no type"}],
            )


class TestAtomicity:

    def test_lines_with_no_content_array_are_preserved_verbatim(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        weird_event = {
            "type": "system",
            "subtype": "init",
            "sessionId": "s",
            "extra": [1, 2, 3],
        }
        _write_jsonl(path, [
            weird_event,
            _assistant_with_tool_use("toolu_X"),
            _user_with_deny("toolu_X"),
        ])

        splice_tool_result(
            path, tool_use_id="toolu_X",
            tool_result_content="r")

        assert _read_jsonl(path)[0] == weird_event

    def test_trailing_newline_preserved(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        path.write_text(
            json.dumps(_assistant_with_tool_use("toolu_X")) + "\n"
            + json.dumps(_user_with_deny("toolu_X")) + "\n",
            encoding="utf-8",
        )
        splice_tool_result(
            path, tool_use_id="toolu_X",
            tool_result_content="r")
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_no_trailing_newline_preserved(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        path.write_text(
            json.dumps(_assistant_with_tool_use("toolu_X")) + "\n"
            + json.dumps(_user_with_deny("toolu_X")),
            encoding="utf-8",
        )
        splice_tool_result(
            path, tool_use_id="toolu_X",
            tool_result_content="r")
        assert not path.read_text(
            encoding="utf-8").endswith("\n")


class TestFindPendingDenyToolUseIds:

    def test_finds_single_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            _user_with_deny("toolu_X"),
        ])
        assert find_pending_deny_tool_use_ids(path) == ["toolu_X"]

    def test_finds_multiple_markers_in_order(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_A"),
            _user_with_deny("toolu_A"),
            _assistant_with_tool_use("toolu_B"),
            _user_with_deny("toolu_B"),
        ])
        assert find_pending_deny_tool_use_ids(path) == [
            "toolu_A", "toolu_B"]

    def test_ignores_already_spliced_results(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            _user_with_deny("toolu_X"),
        ])
        splice_tool_result(
            path, tool_use_id="toolu_X",
            tool_result_content="real")
        assert find_pending_deny_tool_use_ids(path) == []

    def test_supports_string_content_form(
            self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_X",
                        "content": (
                            "permission denied: handoff_to_flywheel"
                        ),
                        "is_error": True,
                    }],
                },
            },
        ])
        assert find_pending_deny_tool_use_ids(path) == ["toolu_X"]

    def test_custom_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        _write_jsonl(path, [
            _assistant_with_tool_use("toolu_X"),
            _user_with_deny(
                "toolu_X", deny_text="my-other-marker"),
        ])
        assert find_pending_deny_tool_use_ids(
            path, deny_marker="my-other-marker") == ["toolu_X"]
        assert find_pending_deny_tool_use_ids(
            path, deny_marker="handoff_to_flywheel") == []
