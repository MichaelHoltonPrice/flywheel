"""Smoke tests for :class:`flywheel.run_defaults.RunDefaults`.

The struct is dataclass-only with no behavior of its own; these
tests pin the construction surface so a future field addition or
type-tightening shows up as a deliberate change rather than a
silent regression.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flywheel.run_defaults import RunDefaults


def test_run_defaults_minimum_construction(tmp_path: Path) -> None:
    """Three required fields, ``defaults`` defaults to empty dict."""
    ws = MagicMock()
    template = MagicMock()
    rd = RunDefaults(
        workspace=ws, template=template, project_root=tmp_path,
    )
    assert rd.workspace is ws
    assert rd.template is template
    assert rd.project_root == tmp_path
    assert rd.defaults == {}


def test_run_defaults_carries_arbitrary_defaults(
    tmp_path: Path,
) -> None:
    """``defaults`` is free-form: any keys an executor wants."""
    rd = RunDefaults(
        workspace=MagicMock(),
        template=MagicMock(),
        project_root=tmp_path,
        defaults={
            "model": "claude-sonnet-4-6",
            "run_id_prefix": "exp42",
            "max_turns": 200,
        },
    )
    assert rd.defaults["model"] == "claude-sonnet-4-6"
    assert rd.defaults["run_id_prefix"] == "exp42"
    assert rd.defaults["max_turns"] == 200


def test_run_defaults_is_frozen(tmp_path: Path) -> None:
    """Frozen so mutation surfaces a clear error.

    The runner must trust the struct is stable across the run;
    a frozen dataclass prevents the subtlest source of "config
    changed mid-run" bugs.
    """
    rd = RunDefaults(
        workspace=MagicMock(),
        template=MagicMock(),
        project_root=tmp_path,
    )
    with pytest.raises(Exception):
        rd.workspace = MagicMock()  # type: ignore[misc]
