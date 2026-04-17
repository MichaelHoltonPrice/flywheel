"""CLI entry point for flywheel.

Supports:
    flywheel create workspace --name NAME --template TEMPLATE
    flywheel run block --workspace PATH --block BLOCK --template TEMPLATE
        [--bind SLOT=ARTIFACT_ID ...] [-- extra container args...]
    flywheel run agent --workspace PATH --template TEMPLATE
        --prompt-file FILE [--model MODEL] [--max-invocations N]
        [--allowed-block BLOCK ...]
        [--input-artifact NAME=ARTIFACT_ID ...]
        [-- container override args...]
    flywheel run loop --workspace PATH --template TEMPLATE
        [--hooks MODULE:CLASS] [--model MODEL] [--max-rounds N]
        [-- project-specific args...]
    flywheel import artifact --workspace PATH --name NAME
        --from SOURCE [--source TEXT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from flywheel.agent import AgentBlockConfig, run_agent_block
from flywheel.agent_loop import AgentLoop, load_hooks_class
from flywheel.config import load_project_config
from flywheel.execution import run_block
from flywheel.template import Template, check_service_dependencies
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

    # flywheel run agent
    agent_parser = run_sub.add_parser("agent")
    agent_parser.add_argument("--workspace", required=True)
    agent_parser.add_argument("--template", required=True)
    agent_parser.add_argument("--prompt-file", required=True,
                              help="Path to the agent system prompt file.")
    agent_parser.add_argument("--model", default=None)
    agent_parser.add_argument("--max-invocations", type=int, default=None,
                              help="Max nested block invocations.")
    agent_parser.add_argument("--max-turns", type=int, default=None)
    agent_parser.add_argument("--total-timeout", type=int, default=14400,
                              help="Max wall-clock seconds (default: 14400 = 4h).")
    agent_parser.add_argument("--auth-volume", default="claude-auth")
    agent_parser.add_argument("--agent-image", default="flywheel-claude:latest")
    agent_parser.add_argument(
        "--allowed-block", action="append", default=[],
        help="Block the agent can invoke (repeatable; default: all).")
    agent_parser.add_argument(
        "--source-dir", action="append", default=[],
        help="Source directory to mount read-only (repeatable).")
    agent_parser.add_argument(
        "--output", action="append", default=[],
        help="Artifact name to collect from agent workspace (repeatable).")
    agent_parser.add_argument(
        "--mcp-servers", default=None,
        help="Comma-separated MCP server names (default: eval).")
    agent_parser.add_argument(
        "--allowed-tools", default=None,
        help="Comma-separated tool whitelist (default: Read,Write,Edit,Glob,Grep).")
    agent_parser.add_argument(
        "--env", action="append", default=[], dest="extra_env",
        help="Extra env var as KEY=VALUE (repeatable).")
    agent_parser.add_argument(
        "--mount", action="append", default=[], dest="extra_mounts",
        help="Extra mount as HOST:CONTAINER:MODE (repeatable).")
    agent_parser.add_argument(
        "--input-artifact", action="append", default=[],
        dest="input_artifacts",
        help="Mount a workspace artifact at /input/NAME inside "
        "the agent container, as NAME=ARTIFACT_ID (repeatable).")

    # flywheel run loop
    loop_parser = run_sub.add_parser("loop")
    loop_parser.add_argument("--workspace", required=True)
    loop_parser.add_argument("--template", required=True)
    loop_parser.add_argument(
        "--hooks", default=None,
        help="Hooks class as module.path:ClassName. "
        "Overrides the hooks key in flywheel.yaml.")
    loop_parser.add_argument("--model", default=None)
    loop_parser.add_argument("--max-rounds", type=int, default=10)
    loop_parser.add_argument("--max-turns", type=int, default=200)
    loop_parser.add_argument("--total-timeout", type=int, default=14400,
                             help="Max wall-clock seconds per agent "
                             "round (default: 14400 = 4h).")
    loop_parser.add_argument("--auth-volume", default="claude-auth")
    loop_parser.add_argument("--agent-image",
                             default="flywheel-claude:latest")
    loop_parser.add_argument(
        "--max-consecutive-failures", type=int, default=3,
        help="Circuit breaker threshold (default: 3).")

    # flywheel materialize
    mat_parser = subparsers.add_parser("materialize")
    mat_parser.add_argument("--workspace", required=True)
    mat_parser.add_argument(
        "--from", dest="source_name", required=True,
        help="Source artifact declaration name.",
    )
    mat_parser.add_argument(
        "--to", dest="target_name", required=True,
        help="Target artifact declaration name.",
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
    elif args.command == "run" and getattr(args, "target", None) == "agent":
        run_agent_command(args, extra_container_args)
    elif args.command == "run" and getattr(args, "target", None) == "loop":
        run_loop_command(args, extra_container_args)
    elif args.command == "materialize":
        materialize_command(
            args.workspace, args.source_name, args.target_name,
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
    template = Template.from_yaml(
        template_path,
        block_registry=config.load_block_registry(),
    )

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


def materialize_command(
    workspace_path: str,
    source_name: str,
    target_name: str,
) -> None:
    """Assemble incremental artifacts into a single JSONL artifact.

    Args:
        workspace_path: Path to the workspace directory.
        source_name: Artifact declaration name to read from.
        target_name: Artifact declaration name to write to.

    Raises:
        ValueError: If the workspace YAML is malformed, artifact names
            are not declared, or no source instances exist.
    """
    ws = Workspace.load(Path(workspace_path))
    count = len(ws.instances_for(source_name))
    instance = ws.materialize_sequence(source_name, target_name)
    print(f"Materialized {count} {source_name!r} instances "
          f"into {instance.id!r}")


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
    template = Template.from_yaml(
        template_path,
        block_registry=config.load_block_registry(),
    )

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


def run_agent_command(args, extra_args: list[str]) -> None:
    """Run an agent block within an existing workspace.

    Reads the prompt file, starts the block bridge, and launches
    the agent container.

    Args:
        args: Parsed argparse namespace with agent-specific fields.
        extra_args: Extra arguments passed to invoked containers.
    """
    config = load_project_config(Path.cwd())

    template_path = config.templates_dir / f"{args.template}.yaml"
    template = Template.from_yaml(
        template_path,
        block_registry=config.load_block_registry(),
    )

    ws = Workspace.load(Path(args.workspace))

    prompt_path = Path(args.prompt_file)
    prompt = prompt_path.read_text(encoding="utf-8")

    # Parse extra_args as --key value overrides for invoked containers.
    overrides = _parse_overrides(extra_args)

    # Substitute {{KEY}} placeholders in the prompt with override values.
    for key, value in overrides.items():
        prompt = prompt.replace("{{" + key.upper() + "}}", value)

    # Parse --env KEY=VALUE arguments into a dict.
    extra_env = {}
    for entry in args.extra_env:
        if "=" in entry:
            k, v = entry.split("=", 1)
            extra_env[k] = v

    # Parse --mount HOST:CONTAINER:MODE arguments into tuples.
    extra_mounts = []
    for entry in args.extra_mounts:
        parts = entry.split(":")
        if len(parts) >= 3:
            # Rejoin first parts in case of Windows drive letter (C:...)
            # Format: HOST:CONTAINER:MODE
            mode = parts[-1]
            container_path = parts[-2]
            host_path = ":".join(parts[:-2])
            extra_mounts.append((host_path, container_path, mode))

    # Parse --input-artifact NAME=ARTIFACT_ID arguments into a dict.
    # Mirrors --bind on `flywheel run block`, but expressed as
    # --input-artifact since these become /input/NAME mounts inside
    # the agent container rather than slot-bound block inputs.
    input_artifacts: dict[str, str] = {}
    for entry in args.input_artifacts:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --input-artifact format {entry!r}, "
                f"expected NAME=ARTIFACT_ID"
            )
        name, artifact_id = entry.split("=", 1)
        input_artifacts[name] = artifact_id

    result = run_agent_block(
        workspace=ws,
        template=template,
        project_root=config.project_root,
        prompt=prompt,
        agent_image=args.agent_image,
        auth_volume=args.auth_volume,
        model=args.model,
        max_invocations=args.max_invocations,
        max_turns=args.max_turns,
        total_timeout=args.total_timeout,
        allowed_blocks=args.allowed_block or None,
        source_dirs=args.source_dir or None,
        input_artifacts=input_artifacts or None,
        output_names=args.output or None,
        overrides=overrides or None,
        mcp_servers=args.mcp_servers,
        allowed_tools=args.allowed_tools,
        extra_env=extra_env or None,
        extra_mounts=extra_mounts or None,
    )
    print(
        f"Agent completed: exit_code={result.exit_code}, "
        f"elapsed={result.elapsed_s:.1f}s, "
        f"invocations={result.evals_run}"
    )


def run_loop_command(args, extra_args: list[str]) -> None:
    """Run an agent loop with project-provided hooks.

    Loads the hooks class (from ``--hooks`` or ``flywheel.yaml``),
    calls ``hooks.init()`` for project-specific setup, builds the
    agent config from CLI flags merged with hooks overrides, and
    runs the loop.

    Args:
        args: Parsed argparse namespace with loop-specific fields.
        extra_args: Project-specific arguments passed after ``--``.
    """
    config = load_project_config(Path.cwd())

    template_path = config.templates_dir / f"{args.template}.yaml"
    template = Template.from_yaml(
        template_path,
        block_registry=config.load_block_registry(),
    )

    ws = Workspace.load(Path(args.workspace))

    # Check service dependencies.
    warnings = check_service_dependencies(template)
    for w in warnings:
        print(f"  [flywheel] WARNING: {w}")

    # Resolve hooks class.
    hooks_path = args.hooks or config.hooks
    if not hooks_path:
        print("ERROR: No hooks specified. Use --hooks or set "
              "'hooks' in flywheel.yaml.")
        sys.exit(1)

    hooks_cls = load_hooks_class(hooks_path)
    hooks = hooks_cls()

    # Project-specific initialization.
    overrides: dict[str, Any] = {}
    if hasattr(hooks, "init"):
        overrides = hooks.init(
            ws, template, config.project_root, extra_args,
        ) or {}

    # Build agent config from CLI flags + hooks overrides.
    agent_config = AgentBlockConfig(
        workspace=ws,
        template=template,
        project_root=config.project_root,
        prompt="",  # Set by hooks.build_prompt()
        agent_image=overrides.get(
            "agent_image", args.agent_image),
        auth_volume=overrides.get(
            "auth_volume", args.auth_volume),
        model=overrides.get("model", args.model),
        max_turns=overrides.get("max_turns", args.max_turns),
        total_timeout=overrides.get(
            "total_timeout", args.total_timeout),
        output_names=overrides.get("output_names"),
        mcp_servers=overrides.get("mcp_servers"),
        allowed_tools=overrides.get("allowed_tools"),
        extra_env=overrides.get("extra_env"),
        extra_mounts=overrides.get("extra_mounts"),
        pre_launch_hook=overrides.get("pre_launch_hook"),
        isolated_network=overrides.get(
            "isolated_network", True),
    )

    # Run the loop.
    loop = AgentLoop(
        hooks=hooks,
        base_config=agent_config,
        max_rounds=args.max_rounds,
        max_consecutive_failures=args.max_consecutive_failures,
    )
    result = loop.run()

    # Print summary.
    print(f"\nLoop complete: "
          f"{result.get('rounds_completed', 0)} rounds, "
          f"exit={result.get('last_exit_reason', 'unknown')}")
    if result.get("is_finished"):
        print("  Game finished!")
    if result.get("stop_reason"):
        print(f"  Stop reason: {result['stop_reason']}")


def _parse_overrides(args: list[str]) -> dict[str, str]:
    """Parse --key value pairs from extra args into a dict.

    Args:
        args: List of CLI arguments (e.g., ["--subclass", "dueling"]).

    Returns:
        Dict mapping keys (without --) to values.
    """
    overrides = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args):
            key = args[i][2:].replace("-", "_")
            overrides[key] = args[i + 1]
            i += 2
        else:
            i += 1
    return overrides
