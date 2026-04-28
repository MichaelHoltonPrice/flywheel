"""Boundary guards for substrate code and pattern execution."""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

from flywheel import cli
from flywheel.artifact import BlockExecution

FLYWHEEL_ROOT = Path(__file__).resolve().parent.parent / "flywheel"

SUBSTRATE_MODULES = [
    "artifact.py",
    "artifact_validator.py",
    "config.py",
    "container.py",
    "execution.py",
    "input_staging.py",
    "output_builder.py",
    "quarantine.py",
    "runtime.py",
    "template.py",
    "termination.py",
    "validation.py",
    "workspace.py",
]

HIGHER_LAYER_MODULES = {
    "flywheel.executor",
    "flywheel.post_check",
}


def test_run_pattern_command_does_not_reference_higher_layer_runners():
    """The pattern command must not import battery runners."""
    source = inspect.getsource(cli.run_pattern_command)
    forbidden = [
        "discover_patterns",
    ]
    for token in forbidden:
        assert token not in source


def test_block_execution_has_no_pattern_grouping_field():
    """Pattern grouping belongs on RunRecord, not BlockExecution."""
    names = {f.name for f in fields(BlockExecution)}
    assert "run_id" not in names


def test_substrate_modules_do_not_import_higher_layers():
    """Canonical substrate code must not depend on batteries/patterns."""
    offenders: list[str] = []
    for filename in SUBSTRATE_MODULES:
        path = FLYWHEEL_ROOT / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
            for name in imported:
                if any(
                    name == forbidden or name.startswith(f"{forbidden}.")
                    for forbidden in HIGHER_LAYER_MODULES
                ):
                    offenders.append(f"{filename}:{node.lineno}:{name}")

    assert offenders == []


def test_private_workspace_mutators_are_not_called_outside_workspace():
    """Artifact/execution ledger writes must use sanctioned APIs."""
    private_mutators = {"_add_artifact", "_add_execution"}
    offenders: list[str] = []
    for path in sorted(FLYWHEEL_ROOT.glob("*.py")):
        if path.name == "workspace.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in private_mutators:
                offenders.append(f"{path.name}:{node.lineno}:{node.attr}")

    assert offenders == []
