"""CLI entry point for flywheel.

Supports:
    flywheel create workspace --name NAME --template TEMPLATE
    flywheel run block --workspace PATH --block BLOCK --template TEMPLATE
        [--bind SLOT=ARTIFACT_ID ...] [-- extra container args...]
    flywheel import artifact --workspace PATH --name NAME
        --from SOURCE [--source TEXT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flywheel.config import load_project_config
from flywheel.execution import run_block
from flywheel.template import Template
from flywheel.workspace import Workspace


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the appropriate command.

    Args:
        argv: Command-line arguments. Defaults to sys.argv when None.
    """
    parser = argparse.ArgumentParser(prog="flywheel")
    subparsers = parser.add_subparsers(dest="command")

    # flywheel create workspace
    create_parser = subparsers.add_parser("create")
    create_sub = create_parser.add_subparsers(dest="resource")

    ws_parser = create_sub.add_parser("workspace")
    ws_parser.add_argument("--name", required=True)
    ws_parser.add_argument("--template", required=True)

    # flywheel import artifact
    import_parser = subparsers.add_parser("import")
    import_sub = import_parser.add_subparsers(dest="resource")

    import_art_parser = import_sub.add_parser("artifact")
    import_art_parser.add_argument("--workspace", required=True)
    import_art_parser.add_argument("--name", required=True)
    import_art_parser.add_argument(
        "--from", dest="source_path", required=True,
        help="Path to the file or directory to import.",
    )
    import_art_parser.add_argument(
        "--source", default=None,
        help="Free-text provenance description (defaults to source path).",
    )

    # flywheel run block
    run_parser = subparsers.add_parser("run")
    run_sub = run_parser.add_subparsers(dest="target")

    block_parser = run_sub.add_parser("block")
    block_parser.add_argument("--workspace", required=True)
    block_parser.add_argument("--block", required=True)
    block_parser.add_argument("--template", required=True)
    block_parser.add_argument(
        "--bind", action="append", default=[],
        help="Bind an input slot to a specific artifact ID, "
        "as SLOT=ARTIFACT_ID (repeatable).",
    )

    # Split on '--' to separate flywheel args from container args
    if argv is None:
        argv = sys.argv[1:]
    if "--" in argv:
        split_idx = argv.index("--")
        flywheel_argv = argv[:split_idx]
        extra_container_args = argv[split_idx + 1:]
    else:
        flywheel_argv = argv
        extra_container_args = []

    args = parser.parse_args(flywheel_argv)

    if args.command == "create" and getattr(args, "resource", None) == "workspace":
        create_workspace(args.name, args.template)
    elif args.command == "import" and getattr(args, "resource", None) == "artifact":
        import_artifact(
            args.workspace, args.name, args.source_path, args.source,
        )
    elif args.command == "run" and getattr(args, "target", None) == "block":
        bindings = _parse_bindings(args.bind)
        run_block_command(
            args.workspace, args.block, args.template,
            bindings, extra_container_args,
        )
    else:
        parser.print_help()
        sys.exit(1)


def _parse_bindings(raw: list[str]) -> dict[str, str]:
    """Parse --bind SLOT=ARTIFACT_ID arguments into a dict.

    Args:
        raw: List of "SLOT=ARTIFACT_ID" strings from argparse.

    Returns:
        A dict mapping slot names to artifact instance IDs.

    Raises:
        ValueError: If any entry is not in SLOT=ARTIFACT_ID format.
    """
    bindings: dict[str, str] = {}
    for entry in raw:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --bind format {entry!r}, "
                f"expected SLOT=ARTIFACT_ID"
            )
        slot, artifact_id = entry.split("=", 1)
        bindings[slot] = artifact_id
    return bindings


def create_workspace(name: str, template_name: str) -> None:
    """Create a workspace from project root (cwd).

    Reads flywheel.yaml to find the foundry dir (foundry_dir).
    Looks for template at foundry_dir/templates/{template_name}.yaml.

    Args:
        name: Workspace name.
        template_name: Name of the template file (without .yaml extension).

    Raises:
        FileNotFoundError: If flywheel.yaml or the template file is missing.
        ValueError: If flywheel.yaml is malformed.
    """
    project_root = Path.cwd()
    config = load_project_config(project_root)

    template_path = config.templates_dir / f"{template_name}.yaml"
    template = Template.from_yaml(template_path)

    ws = Workspace.create(name, template, config.foundry_dir)
    print(f"Created workspace {ws.name!r} at {ws.path}")


def import_artifact(
    workspace_path: str,
    name: str,
    source_path: str,
    source: str | None,
) -> None:
    """Import an external file or directory as a workspace artifact.

    Args:
        workspace_path: Path to the workspace directory.
        name: Artifact declaration name.
        source_path: Path to the file or directory to import.
        source: Free-text provenance description.

    Raises:
        ValueError: If the workspace YAML is malformed, the artifact
            name is not declared, or it is not a copy artifact.
        FileNotFoundError: If the workspace or source path does not
            exist.
    """
    ws = Workspace.load(Path(workspace_path))
    instance = ws.register_artifact(
        name, Path(source_path), source=source,
    )
    print(f"Imported {name!r} as {instance.id!r}")


def run_block_command(
    workspace_path: str,
    block_name: str,
    template_name: str,
    bindings: dict[str, str],
    extra_args: list[str],
) -> None:
    """Run a block within an existing workspace.

    Reads flywheel.yaml to find the foundry dir and project root.
    Loads the workspace and template, then executes the block.

    Args:
        workspace_path: Path to the workspace directory.
        block_name: Name of the block to execute.
        template_name: Name of the template file (without .yaml extension).
        bindings: Explicit input bindings mapping slot names to artifact IDs.
        extra_args: Extra arguments passed to the container entrypoint.

    Raises:
        FileNotFoundError: If flywheel.yaml or the template file is missing.
        ValueError: If flywheel.yaml is malformed or inputs are missing.
        KeyError: If the block is not found in the template.
        RuntimeError: If the container exits with non-zero code.
    """
    config = load_project_config(Path.cwd())

    template_path = config.templates_dir / f"{template_name}.yaml"
    template = Template.from_yaml(template_path)

    ws = Workspace.load(Path(workspace_path))
    result = run_block(
        ws, block_name, template, config.project_root,
        input_bindings=bindings or None,
        args=extra_args or None,
    )
    print(
        f"Block {block_name!r} completed: "
        f"exit_code={result.exit_code}, elapsed={result.elapsed_s:.1f}s"
    )
