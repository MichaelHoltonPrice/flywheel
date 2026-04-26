#!/usr/bin/env python3
"""Splice delayed handoff results into a Claude SDK session JSONL."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class SpliceError(RuntimeError):
    """Raised when a session JSONL cannot be spliced safely."""


def splice_handoff_results(
    session_jsonl_path: Path,
    *,
    result_path: Path,
    deny_marker: str,
    result_label: str,
) -> int:
    """Replace pending handoff deny results in ``session_jsonl_path``."""
    if not session_jsonl_path.is_file():
        return 0

    raw = session_jsonl_path.read_text(encoding="utf-8")
    if not raw.strip():
        return 0

    content, is_error = _build_result_payload(
        result_path=result_path,
        result_label=result_label,
    )

    new_lines: list[str] = []
    replacements = 0
    for lineno, line in enumerate(raw.splitlines(keepends=False), 1):
        if not line.strip():
            new_lines.append(line)
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpliceError(
                f"line {lineno}: invalid JSON: {exc}") from exc
        blocks = _locate_content_array(obj)
        if blocks is None:
            new_lines.append(line)
            continue
        line_replacements = _splice_blocks(
            blocks,
            deny_marker=deny_marker,
            payload=content,
            is_error=is_error,
        )
        replacements += line_replacements
        new_lines.append(
            json.dumps(obj, separators=(",", ":"))
            if line_replacements else line
        )

    if replacements:
        new_text = "\n".join(new_lines)
        if raw.endswith("\n"):
            new_text += "\n"
        _atomic_write(session_jsonl_path, new_text)
    return replacements


def _build_result_payload(
    *,
    result_path: Path,
    result_label: str,
) -> tuple[list[dict[str, Any]], bool]:
    if not result_path.is_file():
        text = (
            f"{result_label} did not produce a result artifact at "
            f"{result_path}."
        )
        return [{"type": "text", "text": text}], True

    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        text = (
            f"{result_label} result at {result_path} was not valid JSON: "
            f"{type(exc).__name__}: {exc}"
        )
        return [{"type": "text", "text": text}], True

    lines = [f"{result_label} completed."]
    if data.get("aborted"):
        lines[0] = f"{result_label} aborted."
        reason = data.get("abort_reason")
        message = data.get("abort_message")
        if reason:
            lines.append(f"abort_reason: {reason}")
        if message:
            lines.append(f"abort_message: {message}")
    for key in (
        "mean",
        "median",
        "std",
        "min",
        "max",
        "p25",
        "p75",
        "episodes",
        "elapsed_s",
        "errors",
        "error_count",
        "seed",
    ):
        if key in data:
            lines.append(f"{key}: {data[key]}")
    lines.append(f"Full result JSON is available at {result_path}.")
    return [{"type": "text", "text": "\n".join(lines)}], False


def _locate_content_array(obj: dict[str, Any]) -> list[Any] | None:
    msg = obj.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
            return content
    content = obj.get("content")
    if isinstance(content, list):
        return content
    return None


def _splice_blocks(
    blocks: list[Any],
    *,
    deny_marker: str,
    payload: list[dict[str, Any]],
    is_error: bool,
) -> int:
    replacements = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if not _is_handoff_deny(block, deny_marker):
            continue
        block["content"] = payload
        block["is_error"] = is_error
        replacements += 1
    return replacements


def _is_handoff_deny(block: dict[str, Any], deny_marker: str) -> bool:
    if block.get("type") != "tool_result":
        return False
    if block.get("is_error") is not True:
        return False
    expected = f"permission denied: {deny_marker}"
    return _content_has_marker(block.get("content"), expected)


def _content_has_marker(content: Any, marker: str) -> bool:
    if isinstance(content, str):
        return marker in content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and marker in text:
                    return True
            elif isinstance(block, str) and marker in block:
                return True
    return False


def _atomic_write(path: Path, text: str) -> None:
    parent = path.parent
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = 0o600
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--deny-marker", default="handoff_to_flywheel")
    parser.add_argument("--label", default="Tool result")
    args = parser.parse_args()

    try:
        count = splice_handoff_results(
            Path(args.session),
            result_path=Path(args.result),
            deny_marker=args.deny_marker,
            result_label=args.label,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[handoff-session] splice failed: "
            f"{type(exc).__name__}: {exc}",
            file=os.sys.stderr,
        )
        return 0
    if count:
        print(f"[handoff-session] spliced {count} handoff result(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
