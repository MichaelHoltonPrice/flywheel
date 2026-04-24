"""Boundary guards for substrate code and unsupported pattern execution."""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import pytest

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
    "flywheel.agent",
    "flywheel.agent_executor",
    "flywheel.agent_group",
    "flywheel.block_group",
    "flywheel.executor",
    "flywheel.pattern",
    "flywheel.pattern_handoff",
    "flywheel.pattern_runner",
    "flywheel.post_check",
    "flywheel.project_hooks",
    "flywheel.run_defaults",
}


def test_run_pattern_is_unavailable_before_loading_project_config():
    """The CLI must not load project configuration for unsupported commands."""
    with patch(
        "flywheel.cli.load_project_config",
        side_effect=AssertionError("project config should not load"),
    ), pytest.raises(NotImplementedError, match="not currently supported"):
        cli.main([
            "run", "pattern", "demo",
            "--workspace", "foundry/workspaces/ws",
            "--template", "template",
        ])


def test_run_pattern_command_does_not_reference_higher_layer_runners():
    """The unsupported command must not import pattern/battery runners."""
    source = inspect.getsource(cli.run_pattern_command)
    forbidden = [
        "AgentExecutor",
        "PatternRunner",
        "discover_patterns",
        "load_project_hooks_class",
        "RunDefaults",
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
