"""Termination-channel parsing and substrate normalization.

The block-execution spec defines a per-runtime channel that
blocks use to announce *why* they ended.  The substrate then
normalizes the announcement against the block's declared
termination reasons and the closed set of substrate-reserved
values, and stores the result on
:attr:`flywheel.artifact.BlockExecution.termination_reason`.

This module owns the pure parsing and normalization rules.  It
holds no I/O — runtimes read from their channel (sidecar file,
HTTP response, etc.) and call :func:`normalize_termination_reason`
with the raw announcement plus a few substrate-observed signals.
The status-mapping rule lives here too so every commit path
derives :attr:`BlockExecution.status` from
``termination_reason`` identically.

See ``flywheel/docs/specs/block-execution.md`` § "Termination
reasons" and § "Status mapping".
"""

from __future__ import annotations

from pathlib import Path

from flywheel.runtime import (
    RESERVED_TERMINATION_REASONS,
    TERMINATION_REASON_CRASH,
    TERMINATION_REASON_INTERRUPTED,
    TERMINATION_REASON_PROTOCOL_VIOLATION,
    TERMINATION_REASON_TIMEOUT,
)


def read_termination_sidecar(termination_file: Path) -> str | None:
    """Parse the ephemeral runtime's ``/flywheel/termination`` file.

    The file must contain a single non-empty line of UTF-8 text,
    optionally trailed by a newline.  Surrounding whitespace on
    that line is stripped and the result returned verbatim.

    Anything that does not parse as "exactly one non-empty line"
    is treated as no announcement and yields ``None``:

    * the file does not exist;
    * the file is empty or contains only whitespace / blank lines;
    * the file has multiple non-blank lines;
    * the file is not valid UTF-8.

    Returns the announced reason as a stripped string, or
    ``None`` for any of the above conditions.
    """
    if not termination_file.exists():
        return None
    try:
        raw = termination_file.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    # Strip exactly one trailing newline if present; any other
    # newline (embedded, leading blank line, multiple lines)
    # disqualifies the announcement.
    if raw.endswith("\r\n"):
        raw = raw[:-2]
    elif raw.endswith("\n"):
        raw = raw[:-1]
    if "\n" in raw or "\r" in raw:
        return None
    text = raw.strip()
    if not text:
        return None
    return text


def normalize_termination_reason(
    *,
    announcement: str | None,
    declared_reasons: set[str],
    crashed: bool = False,
    interrupted: bool = False,
    timed_out: bool = False,
) -> str:
    """Apply the substrate normalization rules to a raw announcement.

    Returns the canonical termination reason that goes onto
    :attr:`flywheel.artifact.BlockExecution.termination_reason`.

    Substrate-observed events override any block announcement:

    * ``interrupted=True`` → ``"interrupted"`` (operator/stop signal).
    * ``timed_out=True`` → ``"timeout"`` (deadline exceeded).
    * ``crashed=True`` → ``"crash"`` (non-zero exit, container
      died, dropped HTTP connection, etc.).

    Otherwise the block's clean-exit announcement is normalized
    against ``declared_reasons``:

    * ``announcement is None`` → ``"protocol_violation"``
      (the block did not announce).
    * Announcement matches a substrate-reserved name (collision)
      → ``"protocol_violation"``.
    * Announcement matches a declared project reason → that
      value verbatim.
    * Announcement matches no declared reason → ``"protocol_violation"``.
    """
    if interrupted:
        return TERMINATION_REASON_INTERRUPTED
    if timed_out:
        return TERMINATION_REASON_TIMEOUT
    if crashed:
        return TERMINATION_REASON_CRASH
    if announcement is None:
        return TERMINATION_REASON_PROTOCOL_VIOLATION
    if announcement in RESERVED_TERMINATION_REASONS:
        return TERMINATION_REASON_PROTOCOL_VIOLATION
    if announcement not in declared_reasons:
        return TERMINATION_REASON_PROTOCOL_VIOLATION
    return announcement


def derive_status(
    termination_reason: str,
    *,
    all_expected_committed: bool,
) -> str:
    """Derive ``BlockExecution.status`` from termination_reason.

    Implements the spec's "status mapping" table:

    * ``"interrupted"`` → ``"interrupted"``.
    * ``"crash"`` / ``"timeout"`` / ``"protocol_violation"`` →
      ``"failed"``.
    * Any other (project-defined) reason → ``"succeeded"`` when
      every expected output slot reached the forge,
      ``"failed"`` otherwise.

    ``all_expected_committed`` is the per-execution signal the
    commit step computes: ``True`` iff every slot the
    ``termination_reason`` mapped to was successfully forged
    (no rejected slots for project-defined reasons; trivially
    ``True`` for reserved reasons whose expected set is empty).
    """
    if termination_reason == TERMINATION_REASON_INTERRUPTED:
        return "interrupted"
    if termination_reason in (
        TERMINATION_REASON_CRASH,
        TERMINATION_REASON_TIMEOUT,
        TERMINATION_REASON_PROTOCOL_VIOLATION,
    ):
        return "failed"
    return "succeeded" if all_expected_committed else "failed"
