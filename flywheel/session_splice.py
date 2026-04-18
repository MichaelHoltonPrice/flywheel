"""Splice a real tool_result into a saved Claude Agent SDK session.

The full-stop nested-block boundary works by intercepting an
agent's tool call via the SDK's ``PreToolUse`` hook, denying the
local execution, exiting the container, running the requested
block as a normal flywheel execution, then restarting the agent
with the real result delivered as the resolution of the original
tool call.  The "delivering the real result" step is this module.

The SDK's deny path leaves a synthetic tool_result in the session
JSONL keyed to the original ``tool_use_id``.  Before the agent
resumes, we rewrite that entry so its content is the executed
block's output and its ``is_error`` flag matches reality.  The
SDK then resumes from a JSONL that records a normal tool-use →
tool-result cycle, and the model's view of the conversation is
indistinguishable from a tool that simply took a long time to
return.

The full design contract lives at
``plans/full-stop-state-contract.md``.  The invariants this
module depends on are documented there and validated by the B1
unit tests + the live-API integration test.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class SpliceError(RuntimeError):
    """Raised when splice cannot complete safely.

    The splice prefers to fail loudly rather than write a
    half-modified JSONL: every error path leaves the input file
    untouched on disk.
    """


def splice_tool_result(
    session_jsonl_path: str | os.PathLike[str],
    *,
    tool_use_id: str,
    tool_result_content: str | list[dict[str, Any]],
    is_error: bool = False,
) -> int:
    """Replace a deny tool_result with a real one in a session JSONL.

    Reads ``session_jsonl_path``, finds the unique tool_result
    block whose ``tool_use_id`` matches ``tool_use_id``, replaces
    its ``content`` with ``tool_result_content`` and its
    ``is_error`` flag with ``is_error``, and writes the modified
    JSONL back atomically (temp file in the same directory then
    ``os.replace``).

    Args:
        session_jsonl_path: Path to the saved session JSONL.
            Must exist and be non-empty.
        tool_use_id: The ``tool_use_id`` to splice; must appear
            in exactly one tool_result block in the JSONL.
        tool_result_content: The real result to inject.  A
            ``str`` becomes a single text block (``[{"type":
            "text", "text": str}]``) per the Anthropic content
            convention; a ``list`` is used verbatim and assumed
            to already match the SDK's expected shape.
        is_error: Whether the result represents a failure.

    Returns:
        The line number (1-indexed) within the JSONL whose
        message envelope held the spliced block.  Useful for
        diagnostics; callers usually ignore it.

    Raises:
        SpliceError: if the file is missing, empty, malformed,
            contains zero matching tool_result blocks, or
            contains more than one matching block.  All raises
            leave the file unchanged on disk.
    """
    path = Path(session_jsonl_path)
    if not path.is_file():
        raise SpliceError(f"session JSONL not found: {path}")

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise SpliceError(f"session JSONL is empty: {path}")

    payload = _normalize_tool_result_content(tool_result_content)

    new_lines: list[str] = []
    matched_line: int | None = None

    for lineno, line in enumerate(raw.splitlines(keepends=False), 1):
        if not line.strip():
            new_lines.append(line)
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpliceError(
                f"line {lineno}: invalid JSON: {exc}") from exc

        content = _locate_content_array(obj)
        if content is None:
            new_lines.append(line)
            continue

        replaced_in_line = _splice_blocks(
            content,
            tool_use_id=tool_use_id,
            payload=payload,
            is_error=is_error,
        )
        if replaced_in_line == 0:
            new_lines.append(line)
            continue

        if replaced_in_line > 1:
            raise SpliceError(
                f"line {lineno}: tool_use_id {tool_use_id!r} "
                f"matched {replaced_in_line} blocks in one "
                f"envelope; invariant violation")

        if matched_line is not None:
            raise SpliceError(
                f"tool_use_id {tool_use_id!r} matched on lines "
                f"{matched_line} and {lineno}; invariant violation")

        matched_line = lineno
        new_lines.append(json.dumps(obj, separators=(",", ":")))

    if matched_line is None:
        raise SpliceError(
            f"tool_use_id {tool_use_id!r} not found in {path}; "
            f"the splice key may be wrong or the file may have "
            f"already been spliced")

    new_text = "\n".join(new_lines)
    if raw.endswith("\n"):
        new_text += "\n"

    _atomic_write(path, new_text)
    return matched_line


def _normalize_tool_result_content(
    content: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Coerce ``str`` payloads to the SDK's text-block list shape."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        for i, block in enumerate(content):
            if not isinstance(block, dict):
                raise SpliceError(
                    f"tool_result_content[{i}] is not a dict")
            if "type" not in block:
                raise SpliceError(
                    f"tool_result_content[{i}] missing 'type'")
        return content
    raise SpliceError(
        f"tool_result_content must be str or list, got "
        f"{type(content).__name__}")


def _locate_content_array(
    obj: dict[str, Any],
) -> list[Any] | None:
    """Return the message-envelope's content array if present.

    The SDK has historically used both ``message.content`` and
    bare ``content``; we resolve at runtime.  Returns the live
    list (mutating it mutates the parsed object) or ``None``
    when neither shape is present.
    """
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
    tool_use_id: str,
    payload: list[dict[str, Any]],
    is_error: bool,
) -> int:
    """Splice in a real tool_result; return number of replacements.

    Mutates ``blocks`` in place.  Returns 0 when no block in the
    envelope matched, 1 on a successful splice, or N>1 if the
    same ``tool_use_id`` appears more than once in the same
    envelope (caller treats that as an invariant violation).
    """
    replacements = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_result":
            continue
        if block.get("tool_use_id") != tool_use_id:
            continue
        block["content"] = payload
        block["is_error"] = is_error
        replacements += 1
    return replacements


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via temp file + ``os.replace``.

    Crash-safe: a partial write never replaces the existing file.
    Uses the parent directory for the temp file so the rename is
    atomic on the same filesystem.
    """
    parent = path.parent
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def find_pending_deny_tool_use_ids(
    session_jsonl_path: str | os.PathLike[str],
    *,
    deny_marker: str = "handoff_to_flywheel",
) -> list[str]:
    """Scan a JSONL for deny tool_results from our PreToolUse hook.

    Diagnostic helper: returns every ``tool_use_id`` whose paired
    tool_result content carries our deny marker substring.  Used
    by the runner during recovery (a saved session that contains
    multiple deny markers means multiple pending tool calls were
    captured but only some were spliced).

    Returns the IDs in the order they appear in the file.  Does
    not modify the file.
    """
    path = Path(session_jsonl_path)
    if not path.is_file():
        raise SpliceError(f"session JSONL not found: {path}")

    found: list[str] = []
    for line in _iter_jsonl_lines(path):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = _locate_content_array(obj)
        if content is None:
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            if not _content_has_marker(
                    block.get("content"), deny_marker):
                continue
            tool_use_id = block.get("tool_use_id")
            if isinstance(tool_use_id, str):
                found.append(tool_use_id)
    return found


def _iter_jsonl_lines(path: Path) -> Iterable[str]:
    """Yield non-empty lines from a JSONL file."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped.strip():
                yield stripped


def _content_has_marker(content: Any, marker: str) -> bool:
    """Whether a tool_result's content carries a substring marker.

    Tolerates both string content and list-of-blocks content.
    """
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
