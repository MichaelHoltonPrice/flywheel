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
    placeholder_marker: str = "Evaluation requested.",
    result_label: str,
    tool_use_id: str | None = None,
    replace_all: bool = False,
) -> int:
    """Replace pending handoff placeholder results in a session JSONL.

    The current handoff path lets the MCP tool complete and then stops
    Claude Code from a PostToolUse hook with ``continue: false``.  That
    leaves a successful placeholder tool_result in the SDK session.  For
    migration, this also recognizes the old PreToolUse deny marker.
    """
    if not session_jsonl_path.is_file():
        return 0

    raw = session_jsonl_path.read_text(encoding="utf-8")
    if not raw.strip():
        return 0

    content, is_error = _build_result_payload(
        result_path=result_path,
        result_label=result_label,
    )

    parsed_lines: list[tuple[str, dict[str, Any] | None]] = []
    candidates: list[tuple[int, dict[str, Any]]] = []
    for lineno, line in enumerate(raw.splitlines(keepends=False), 1):
        if not line.strip():
            parsed_lines.append((line, None))
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpliceError(
                f"line {lineno}: invalid JSON: {exc}") from exc
        parsed_lines.append((line, obj))
        blocks = _locate_content_array(obj)
        if blocks is None:
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if not _is_pending_handoff_result(
                block,
                deny_marker=deny_marker,
                placeholder_marker=placeholder_marker,
            ):
                continue
            if tool_use_id and block.get("tool_use_id") != tool_use_id:
                continue
            candidates.append((len(parsed_lines) - 1, block))

    if tool_use_id and not candidates:
        raise SpliceError(f"no pending handoff tool_result for {tool_use_id}")

    selected = candidates if replace_all else candidates[-1:]
    for line_index, block in selected:
        block["content"] = content
        block["is_error"] = is_error
        obj = parsed_lines[line_index][1]
        if obj is not None:
            _update_tool_result_metadata(
                obj,
                payload=content,
                is_error=is_error,
            )

    replacements = len(selected)
    if replacements:
        touched = {idx for idx, _block in selected}
        new_lines: list[str] = []
        for idx, (line, obj) in enumerate(parsed_lines):
            if obj is not None and (replace_all or idx in touched):
                new_lines.append(json.dumps(obj, separators=(",", ":")))
            else:
                new_lines.append(line)
        new_text = "\n".join(new_lines)
        if raw.endswith("\n"):
            new_text += "\n"
        _atomic_write(session_jsonl_path, new_text)
    return replacements


def splice_handoff_results_in_blocks(
    blocks: list[Any],
    *,
    result_path: Path,
    deny_marker: str,
    placeholder_marker: str = "Evaluation requested.",
    result_label: str,
    tool_use_id: str | None = None,
    replace_all: bool = False,
) -> int:
    """Replace pending handoff results inside one content block list."""
    content, is_error = _build_result_payload(
        result_path=result_path,
        result_label=result_label,
    )
    candidates = [
        block for block in blocks
        if (
            isinstance(block, dict)
            and _is_pending_handoff_result(
                block,
                deny_marker=deny_marker,
                placeholder_marker=placeholder_marker,
            )
            and (tool_use_id is None or block.get("tool_use_id") == tool_use_id)
        )
    ]
    selected = candidates if replace_all else candidates[-1:]
    for block in selected:
        block["content"] = content
        block["is_error"] = is_error
    return len(selected)


def handoff_result_text(*, result_path: Path, result_label: str) -> tuple[str, bool]:
    """Build the text payload used for a completed handoff result."""
    content, is_error = _build_result_payload(
        result_path=result_path,
        result_label=result_label,
    )
    texts: list[str] = []
    for block in content:
        text = block.get("text") if isinstance(block, dict) else None
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts), is_error


def _splice_blocks(
    blocks: list[Any],
    *,
    deny_marker: str,
    payload: list[dict[str, Any]],
    is_error: bool,
    tool_use_id: str | None = None,
    replace_all: bool = False,
) -> int:
    candidates = [
        block for block in blocks
        if (
            isinstance(block, dict)
            and _is_pending_handoff_result(
                block,
                deny_marker=deny_marker,
                placeholder_marker="Evaluation requested.",
            )
            and (tool_use_id is None or block.get("tool_use_id") == tool_use_id)
        )
    ]
    selected = candidates if replace_all else candidates[-1:]
    for block in selected:
        block["content"] = payload
        block["is_error"] = is_error
    return len(selected)


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
        "score",
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


def _update_tool_result_metadata(
    obj: dict[str, Any],
    *,
    payload: list[dict[str, Any]],
    is_error: bool,
) -> None:
    """Update Claude Code's duplicated tool-result metadata fields."""
    text = "\n".join(
        block.get("text", "")
        for block in payload
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )
    if "toolUseResult" in obj:
        obj["toolUseResult"] = text
    if isinstance(obj.get("mcpMeta"), dict):
        obj["mcpMeta"] = {"structuredContent": {"result": text}}
    if isinstance(obj.get("tool_use_result"), dict):
        obj["tool_use_result"] = {
            "content": text,
            "is_error": is_error,
        }


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


def _is_pending_handoff_result(
    block: dict[str, Any],
    *,
    deny_marker: str,
    placeholder_marker: str,
) -> bool:
    if block.get("type") != "tool_result":
        return False
    if block.get("is_error") is not True:
        return _content_has_marker(block.get("content"), placeholder_marker)
    # Backward compatibility for sessions captured by the old PreToolUse
    # deny handoff path.
    return _is_handoff_deny(block, deny_marker)


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
    parser.add_argument(
        "--meta",
        default=None,
        help=(
            "Optional JSON file with result_path, result_label, "
            "placeholder_marker, and tool_use_id for the pending handoff."
        ),
    )
    parser.add_argument("--deny-marker", default="handoff_to_flywheel")
    parser.add_argument(
        "--placeholder-marker",
        default="Evaluation requested.",
        help="Text identifying the successful placeholder tool_result.",
    )
    parser.add_argument("--label", default="Tool result")
    parser.add_argument("--tool-use-id", default=None)
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help="Replace every matching pending handoff result.",
    )
    args = parser.parse_args()

    result_path = Path(args.result)
    result_label = args.label
    placeholder_marker = args.placeholder_marker
    tool_use_id = args.tool_use_id
    if args.meta:
        try:
            meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                if isinstance(meta.get("result_path"), str):
                    result_path = Path(meta["result_path"])
                if isinstance(meta.get("result_label"), str):
                    result_label = meta["result_label"]
                if isinstance(meta.get("placeholder_marker"), str):
                    placeholder_marker = meta["placeholder_marker"]
                if isinstance(meta.get("tool_use_id"), str):
                    tool_use_id = meta["tool_use_id"]
        except Exception as exc:  # noqa: BLE001
            print(
                f"[handoff-session] ignored invalid metadata file "
                f"{args.meta}: {type(exc).__name__}: {exc}",
                file=os.sys.stderr,
            )

    try:
        count = splice_handoff_results(
            Path(args.session),
            result_path=result_path,
            deny_marker=args.deny_marker,
            placeholder_marker=placeholder_marker,
            result_label=result_label,
            tool_use_id=tool_use_id,
            replace_all=args.replace_all,
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
