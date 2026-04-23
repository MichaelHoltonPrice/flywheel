"""Workspace-internal quarantine for validator-rejected output bytes.

When a block produces an output that fails validation, the bytes
themselves are still useful: an operator can inspect them to see
*why* the validator rejected them, or correct them and register a
superseding instance via ``flywheel fix execution``.  Without
quarantine, those bytes vanish with the producer's tempdir as
soon as the failure is recorded.

The convention is::

    <workspace>/quarantine/<execution_id>/<slot_name>/

One subdir per ``(execution_id, slot_name)`` pair, which is the
durable handle a :class:`flywheel.artifact.RejectionRef` uses.
The path stored on the ledger is *workspace-relative* (e.g.
``"quarantine/exec_a3f7b2/checkpoint"``) so the workspace
remains relocatable.

Quarantine is best-effort.  The validation failure is the
primary signal; missing bytes just mean the operator cannot
recover them from this particular failure.  Callers must
therefore tolerate ``None`` returns from
:func:`quarantine_slot`.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def quarantine_slot(
    workspace_path: Path,
    execution_id: str,
    slot_name: str,
    src: Path,
) -> str | None:
    """Preserve a rejected slot's bytes under the workspace.

    Copies the directory ``src`` to
    ``<workspace_path>/quarantine/<execution_id>/<slot_name>/``
    and returns the workspace-relative path of the destination
    on success.

    On any I/O failure — missing source, permission denied,
    pre-existing destination from a duplicate execution id —
    logs a warning and returns ``None``.  The caller records the
    validation failure either way; ``None`` simply means the
    bytes were not preserved this time.

    Args:
        workspace_path: Absolute path of the owning workspace.
        execution_id: The :class:`BlockExecution` whose output
            slot was rejected.
        slot_name: Output slot name within that execution.
        src: Directory containing the rejected bytes.  Treated
            as read-only.

    Returns:
        The workspace-relative path the bytes were preserved at
        (e.g. ``"quarantine/exec_a3f/checkpoint"``), or ``None``
        when preservation failed.
    """
    rel = Path("quarantine") / execution_id / slot_name
    dst = workspace_path / rel
    try:
        if not src.is_dir():
            logger.warning(
                "quarantine: source %s is not a directory; "
                "cannot preserve rejected slot %r of execution %r",
                src, slot_name, execution_id,
            )
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    except Exception:  # noqa: BLE001
        logger.warning(
            "quarantine: failed to preserve rejected slot %r of "
            "execution %r at %s",
            slot_name, execution_id, dst,
            exc_info=True,
        )
        return None
    return rel.as_posix()
