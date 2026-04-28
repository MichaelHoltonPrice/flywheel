"""Unit tests for :mod:`flywheel.quarantine`.

Covers the workspace-relative path convention, copy-tree
behavior, and the best-effort I/O contract: any failure yields
``None`` rather than propagating, so the caller's primary
signal (the validation failure) is never overwritten by a
secondary preservation failure.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

from flywheel.quarantine import quarantine_slot


def _make_slot(tmp_path: Path, name: str = "src") -> Path:
    """Return a directory containing one small file."""
    src = tmp_path / name
    src.mkdir()
    (src / "out.txt").write_text("payload")
    return src


class TestQuarantineSlot:
    def test_copies_into_workspace_relative_path(
        self, tmp_path: Path,
    ):
        ws = tmp_path / "ws"
        ws.mkdir()
        src = _make_slot(tmp_path)

        rel = quarantine_slot(ws, "exec_a3f", "checkpoint", src)

        assert rel == "quarantine/exec_a3f/checkpoint"
        # Bytes preserved at the canonical location.
        copied = ws / rel / "out.txt"
        assert copied.read_text() == "payload"

    def test_returns_none_when_source_missing(
        self, tmp_path: Path,
    ):
        # Source directory does not exist on disk; preservation
        # cannot succeed.  Caller still gets a usable signal
        # (None) rather than an exception.
        ws = tmp_path / "ws"
        ws.mkdir()
        missing = tmp_path / "does_not_exist"

        rel = quarantine_slot(ws, "e1", "slot", missing)

        assert rel is None
        assert not (ws / "quarantine").exists()

    def test_returns_none_when_source_is_file(
        self, tmp_path: Path,
    ):
        # Quarantine targets the per-slot directory shape that
        # block outputs and imports use; a bare file is not a
        # valid source.
        ws = tmp_path / "ws"
        ws.mkdir()
        bare = tmp_path / "bare.txt"
        bare.write_text("hi")

        rel = quarantine_slot(ws, "e1", "slot", bare)

        assert rel is None

    def test_returns_none_on_copy_failure(
        self, tmp_path: Path,
    ):
        # Simulate disk-full / permission-denied during the
        # copy.  The validation outcome is what matters; the
        # caller treats None as "bytes not preserved this time"
        # and proceeds.
        ws = tmp_path / "ws"
        ws.mkdir()
        src = _make_slot(tmp_path)

        with patch(
            "flywheel.quarantine.shutil.copytree",
            side_effect=OSError("no space left on device"),
        ):
            rel = quarantine_slot(ws, "e1", "slot", src)

        assert rel is None

    def test_distinct_executions_do_not_collide(
        self, tmp_path: Path,
    ):
        # Two failures of two different executions touching the
        # same slot name keep their bytes separate, since the
        # execution id is part of the path.
        ws = tmp_path / "ws"
        ws.mkdir()
        src1 = _make_slot(tmp_path, "src1")
        src2 = _make_slot(tmp_path, "src2")

        rel1 = quarantine_slot(ws, "e1", "checkpoint", src1)
        rel2 = quarantine_slot(ws, "e2", "checkpoint", src2)

        assert rel1 != rel2
        assert (ws / rel1 / "out.txt").exists()
        assert (ws / rel2 / "out.txt").exists()

    def test_duplicate_call_for_same_slot_returns_none(
        self, tmp_path: Path,
    ):
        # Quarantine paths are immutable per (exec_id, slot);
        # a second call for the same pair should not silently
        # overwrite the first preserved copy.
        ws = tmp_path / "ws"
        ws.mkdir()
        src1 = _make_slot(tmp_path, "src1")
        (src1 / "marker.txt").write_text("first")
        rel1 = quarantine_slot(ws, "e1", "slot", src1)
        assert rel1 is not None

        src2 = _make_slot(tmp_path, "src2")
        (src2 / "marker.txt").write_text("second")
        rel2 = quarantine_slot(ws, "e1", "slot", src2)

        # ``shutil.copytree`` refuses to overwrite an existing
        # destination; we surface that as ``None`` and leave the
        # original quarantined copy untouched.
        assert rel2 is None
        marker = ws / rel1 / "marker.txt"
        assert marker.read_text() == "first"

    def test_creates_intermediate_dirs(self, tmp_path: Path):
        # Workspace may not have a ``quarantine/`` dir yet on
        # the first failure; the helper should create it.
        ws = tmp_path / "ws"
        ws.mkdir()
        assert not (ws / "quarantine").exists()
        src = _make_slot(tmp_path)

        rel = quarantine_slot(ws, "e1", "slot", src)

        assert rel is not None
        assert (ws / "quarantine").is_dir()
        # Cleanup so pytest's tmp_path cleanup doesn't choke on
        # any odd permissions on Windows.
        shutil.rmtree(ws / "quarantine", ignore_errors=True)
