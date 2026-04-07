"""CLI entry point for flywheel.

Supports: flywheel create workspace --name NAME --template TEMPLATE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from flywheel.template import Template
from flywheel.workspace import Workspace


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the appropriate command.

    Args:
        argv: Command-line arguments. Defaults to sys.argv when None.
    """
    parser = argparse.ArgumentParser(prog="flywheel")
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser("create")
    create_sub = create_parser.add_subparsers(dest="resource")

    ws_parser = create_sub.add_parser("workspace")
    ws_parser.add_argument("--name", required=True)
    ws_parser.add_argument("--template", required=True)

    args = parser.parse_args(argv)

    if args.command == "create" and getattr(args, "resource", None) == "workspace":
        create_workspace(args.name, args.template)
    else:
        parser.print_help()
        sys.exit(1)


def create_workspace(name: str, template_name: str) -> None:
    """Create a workspace from project root (cwd).

    Reads flywheel.yaml to find the workforce dir (harness_dir).
    Looks for template at workforce_dir/templates/{template_name}.yaml.

    Args:
        name: Workspace name.
        template_name: Name of the template file (without .yaml extension).

    Raises:
        FileNotFoundError: If flywheel.yaml or the template file is missing.
    """
    project_root = Path.cwd()
    config_path = project_root / "flywheel.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    workforce_dir = project_root / config["harness_dir"]
    template_path = workforce_dir / "templates" / f"{template_name}.yaml"
    template = Template.from_yaml(template_path)

    ws = Workspace.create(name, template, workforce_dir)
    print(f"Created workspace {ws.name!r} at {ws.path}")
