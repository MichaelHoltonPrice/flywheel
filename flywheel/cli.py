"""CLI entry point for flywheel.

Supports:
    flywheel create workspace --name NAME --template TEMPLATE
    flywheel run block --workspace PATH --block BLOCK --template TEMPLATE
        [--bind SLOT=ARTIFACT_ID ...] [-- extra container args...]
    flywheel run agent --workspace PATH --template TEMPLATE
        --prompt-file FILE [--model MODEL]
        [--allowed-block BLOCK ...]
        [--input-artifact NAME=ARTIFACT_ID ...]
        [-- container override args...]
    flywheel run pattern PATTERN_NAME --workspace PATH
        --template TEMPLATE
    flywheel import artifact --workspace PATH --name NAME
        --from SOURCE [--source TEXT]
    flywheel fix execution --workspace PATH --execution EXEC_ID
        --slot SLOT --from SOURCE_DIR [--reason TEXT]
    flywheel amend artifact --workspace PATH --artifact ARTIFACT_ID
        --from SOURCE_DIR [--reason TEXT]

Pattern execution runs declared cohorts through canonical block
execution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flywheel.agent import run_agent_block
from flywheel.artifact import RejectionRef, SupersedesRef
from flywheel.artifact_validator import ArtifactValidatorRegistry
from flywheel.config import ProjectConfig, load_project_config
from flywheel.execution import run_block
from flywheel.pattern_declaration import PatternDeclaration
from flywheel.pattern_execution import PatternRunError, run_pattern
from flywheel.template import ArtifactDeclaration, Template
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
        help=(
            "Path to the directory whose contents become the new "
            "artifact instance.  Must be a directory; wrap a "
            "single file in a directory and pass that."
        ),
    )
    import_art_parser.add_argument(
        "--source", default=None,
        help="Free-text provenance description (defaults to source path).",
    )

    # flywheel fix execution
    fix_parser = subparsers.add_parser(
        "fix",
        help=(
            "Register a corrected artifact for a quarantined "
            "output slot of a failed execution."),
    )
    fix_sub = fix_parser.add_subparsers(dest="resource")

    fix_exec_parser = fix_sub.add_parser(
        "execution",
        help=(
            "Register corrected bytes as a new artifact instance "
            "that supersedes the rejected output slot of a "
            "failed execution."),
    )
    fix_exec_parser.add_argument("--workspace", required=True)
    fix_exec_parser.add_argument(
        "--execution", required=True, dest="execution_id",
        help=(
            "Execution id whose rejected slot is being "
            "superseded."),
    )
    fix_exec_parser.add_argument(
        "--slot", required=True,
        help=(
            "Output slot name within the execution.  Must appear "
            "in that execution's rejected_outputs."),
    )
    fix_exec_parser.add_argument(
        "--from", dest="source_path", required=True,
        help=(
            "Path to the directory whose contents become the "
            "corrected artifact instance.  Must be a directory."),
    )
    fix_exec_parser.add_argument(
        "--reason", default=None,
        help=(
            "Optional human-readable reason recorded on the "
            "successor instance."),
    )

    # flywheel amend artifact
    amend_parser = subparsers.add_parser(
        "amend",
        help=(
            "Register a corrective successor to an accepted "
            "artifact instance."),
    )
    amend_sub = amend_parser.add_subparsers(dest="resource")

    amend_art_parser = amend_sub.add_parser(
        "artifact",
        help=(
            "Register corrected bytes as a new artifact instance "
            "that supersedes an accepted predecessor."),
    )
    amend_art_parser.add_argument("--workspace", required=True)
    amend_art_parser.add_argument(
        "--artifact", required=True, dest="artifact_id",
        help="Predecessor artifact id being superseded.",
    )
    amend_art_parser.add_argument(
        "--from", dest="source_path", required=True,
        help=(
            "Path to the directory whose contents become the "
            "successor artifact instance.  Must be a directory."),
    )
    amend_art_parser.add_argument(
        "--reason", default=None,
        help=(
            "Optional human-readable reason recorded on the "
            "successor instance."),
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
    agent_parser.add_argument(
        "--block-name", required=True,
        help=(
            "Block name to record this execution under.  "
            "Also keys the /state/ restore chain."
        ),
    )
    agent_parser.add_argument("--prompt-file", required=True,
                              help="Path to the agent system prompt file.")
    agent_parser.add_argument("--model", default=None)
    agent_parser.add_argument("--max-turns", type=int, default=None)
    agent_parser.add_argument("--total-timeout", type=int, default=14400,
                              help="Max wall-clock seconds (default: 14400 = 4h).")
    agent_parser.add_argument("--auth-volume", default="claude-auth")
    agent_parser.add_argument("--agent-image", default="flywheel-claude:latest")
    agent_parser.add_argument(
        "--source-dir", action="append", default=[],
        help="Source directory to mount read-only (repeatable).")
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

    # flywheel run pattern
    pattern_parser = run_sub.add_parser("pattern")
    pattern_parser.add_argument(
        "pattern_name",
        help=(
            "Pattern name (file stem under "
            "<foundry_dir>/templates/patterns/)."
        ))
    pattern_parser.add_argument("--workspace", required=True)
    pattern_parser.add_argument("--template", required=True)

    # flywheel container — manage persistent request-response runtimes
    container_parser = subparsers.add_parser(
        "container",
        help=(
            "Manage long-lived request-response runtimes "
            "(workspace-persistent containers)."),
    )
    container_sub = container_parser.add_subparsers(
        dest="container_action")

    cstop_parser = container_sub.add_parser(
        "stop",
        help=(
            "Tear down a request-response runtime via the "
            "/scratch/.stop sentinel, falling back to "
            "SIGTERM/SIGKILL."),
    )
    cstop_parser.add_argument("--workspace", required=True)
    cstop_parser.add_argument("--template", required=True)
    cstop_parser.add_argument(
        "--block", required=True,
        help="Block name whose runtime should be stopped.")
    cstop_parser.add_argument(
        "--reason", default="cli_stop",
        help=(
            "Free-form reason string recorded with the teardown "
            "(default: cli_stop)."),
    )

    clist_parser = container_sub.add_parser(
        "list",
        help=(
            "List the request-response runtimes Docker reports "
            "for this workspace."),
    )
    clist_parser.add_argument("--workspace", required=True)

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
    elif args.command == "fix" and getattr(args, "resource", None) == "execution":
        fix_execution(
            workspace_path=args.workspace,
            execution_id=args.execution_id,
            slot=args.slot,
            source_path=args.source_path,
            reason=args.reason,
        )
    elif args.command == "amend" and getattr(args, "resource", None) == "artifact":
        amend_artifact(
            workspace_path=args.workspace,
            artifact_id=args.artifact_id,
            source_path=args.source_path,
            reason=args.reason,
        )
    elif args.command == "run" and getattr(args, "target", None) == "block":
        bindings = _parse_bindings(args.bind)
        run_block_command(
            args.workspace, args.block, args.template,
            bindings, extra_container_args,
        )
    elif args.command == "run" and getattr(args, "target", None) == "agent":
        run_agent_command(args, extra_container_args)
    elif args.command == "run" and getattr(args, "target", None) == "pattern":
        run_pattern_command(args, extra_container_args)
    elif (
        args.command == "container"
        and getattr(args, "container_action", None) == "stop"
    ):
        container_stop_command(
            workspace=args.workspace,
            template_name=args.template,
            block_name=args.block,
            reason=args.reason,
        )
    elif (
        args.command == "container"
        and getattr(args, "container_action", None) == "list"
    ):
        container_list_command(workspace=args.workspace)
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
    Looks for the workspace template at
    foundry_dir/templates/workspaces/{template_name}.yaml.

    Args:
        name: Workspace name.
        template_name: Name of the template file (without .yaml extension).

    Raises:
        FileNotFoundError: If flywheel.yaml or the template file is missing.
        ValueError: If flywheel.yaml is malformed.
    """
    project_root = Path.cwd()
    config = load_project_config(project_root)

    template_path = (
        config.workspace_templates_dir / f"{template_name}.yaml"
    )
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
    """Import an external directory as a workspace artifact.

    Artifact instances are always directory-shaped (mirroring
    block output slots), so ``source_path`` must be a directory;
    callers with a single file should wrap it in a directory
    first.

    Args:
        workspace_path: Path to the workspace directory.
        name: Artifact declaration name.
        source_path: Path to the directory whose contents
            become the new artifact instance.
        source: Free-text provenance description.

    Raises:
        ValueError: If the workspace YAML is malformed, the artifact
            name is not declared, it is not a copy artifact, or
            ``source_path`` is not a directory.
        FileNotFoundError: If the workspace or source path does not
            exist.
        flywheel.artifact_validator.ArtifactValidationError: If the
            project registered a validator for ``name`` and the
            source fails it.
    """
    config = load_project_config(Path.cwd())
    ws = Workspace.load(Path(workspace_path))
    validator_registry = config.load_artifact_validator_registry()
    declaration = None
    template_name = ws.template_name
    if template_name:
        template_path = (
            config.workspace_templates_dir / f"{template_name}.yaml"
        )
        if template_path.is_file():
            template = Template.from_yaml(
                template_path,
                block_registry=config.load_block_registry(),
            )
            for decl in template.artifacts:
                if decl.name == name:
                    declaration = decl
                    break
    instance = ws.register_artifact(
        name, Path(source_path), source=source,
        validator_registry=validator_registry,
        declaration=declaration,
    )
    print(f"Imported {name!r} as {instance.id!r}")


def _resolve_validator_and_declaration(
    config: ProjectConfig,
    workspace: Workspace,
    name: str,
) -> tuple[ArtifactValidatorRegistry, ArtifactDeclaration | None]:
    """Resolve validator registry and declaration for an artifact name.

    Mirrors the loading shape of :func:`import_artifact` so
    every entry point that lands new bytes runs the same
    validator the original import would have.  Factored here
    because three CLI verbs (``import artifact``,
    ``fix execution``, ``amend artifact``) all need the same
    pair.

    Args:
        config: The loaded project config (already resolved
            against the project root).
        workspace: The workspace receiving the new instance;
            its ``template_name`` selects which template the
            declaration is read from.
        name: The artifact declaration name.

    Returns:
        ``(validator_registry, declaration)``.  ``declaration``
        is ``None`` when the workspace has no template_name, the
        template file is missing, or the template doesn't
        declare ``name`` — matching ``import_artifact``'s
        existing behaviour.
    """
    validator_registry = config.load_artifact_validator_registry()
    declaration: ArtifactDeclaration | None = None
    template_name = workspace.template_name
    if template_name:
        template_path = (
            config.workspace_templates_dir / f"{template_name}.yaml"
        )
        if template_path.is_file():
            template = Template.from_yaml(
                template_path,
                block_registry=config.load_block_registry(),
            )
            for decl in template.artifacts:
                if decl.name == name:
                    declaration = decl
                    break
    return validator_registry, declaration


def fix_execution(
    *,
    workspace_path: str,
    execution_id: str,
    slot: str,
    source_path: str,
    reason: str | None,
) -> None:
    """Register a corrected artifact for a quarantined slot.

    Looks up ``execution_id`` in the workspace, locates the
    rejected output ``slot`` (must appear in
    :attr:`flywheel.artifact.BlockExecution.rejected_outputs`),
    and registers the corrected bytes as a fresh artifact
    instance whose ``supersedes`` points at that
    ``RejectionRef``.  The corrected bytes are run through the
    project's validator registry the same way a fresh
    ``import artifact`` call would be — successors get no
    validation discount.

    Args:
        workspace_path: Path to the workspace directory.
        execution_id: The failed execution being fixed.
        slot: The output slot name within that execution; also
            the artifact declaration name used for the new
            instance.
        source_path: Directory whose contents become the new
            instance.
        reason: Optional human-readable reason.

    Raises:
        ValueError: If the execution / slot / source path are
            invalid, or the predecessor existence check on the
            workspace fails.
        FileNotFoundError: If the workspace or source path does
            not exist.
        flywheel.artifact_validator.ArtifactValidationError: If
            the corrected bytes fail validation.
    """
    config = load_project_config(Path.cwd())
    ws = Workspace.load(Path(workspace_path))
    validator_registry, declaration = (
        _resolve_validator_and_declaration(config, ws, slot)
    )
    instance = ws.register_artifact(
        slot, Path(source_path),
        validator_registry=validator_registry,
        declaration=declaration,
        supersedes=SupersedesRef(rejection=RejectionRef(
            execution_id=execution_id, slot=slot,
        )),
        supersedes_reason=reason,
    )
    print(
        f"Registered {slot!r} as {instance.id!r} "
        f"superseding rejected slot {slot!r} of "
        f"execution {execution_id!r}"
    )


def amend_artifact(
    *,
    workspace_path: str,
    artifact_id: str,
    source_path: str,
    reason: str | None,
) -> None:
    """Register a corrective successor to an accepted artifact.

    Looks up ``artifact_id`` in the workspace to read its
    declaration name (lineage is same-name; see
    :class:`flywheel.artifact.SupersedesRef`), then registers
    the corrected bytes as a fresh artifact instance whose
    ``supersedes`` points at that predecessor id.  Successors
    are validated by the same registry a fresh import would
    use.

    Args:
        workspace_path: Path to the workspace directory.
        artifact_id: The accepted predecessor being superseded.
        source_path: Directory whose contents become the new
            instance.
        reason: Optional human-readable reason.

    Raises:
        ValueError: If the predecessor does not exist, or the
            source path / workspace are invalid.
        FileNotFoundError: If the workspace or source path does
            not exist.
        flywheel.artifact_validator.ArtifactValidationError: If
            the corrected bytes fail validation.
    """
    config = load_project_config(Path.cwd())
    ws = Workspace.load(Path(workspace_path))
    predecessor = ws.artifacts.get(artifact_id)
    if predecessor is None:
        raise ValueError(
            f"Artifact {artifact_id!r} does not exist in this "
            f"workspace"
        )
    validator_registry, declaration = (
        _resolve_validator_and_declaration(
            config, ws, predecessor.name,
        )
    )
    instance = ws.register_artifact(
        predecessor.name, Path(source_path),
        validator_registry=validator_registry,
        declaration=declaration,
        supersedes=SupersedesRef(artifact_id=artifact_id),
        supersedes_reason=reason,
    )
    print(
        f"Registered {predecessor.name!r} as {instance.id!r} "
        f"superseding {artifact_id!r}"
    )


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

    template_path = (
        config.workspace_templates_dir / f"{template_name}.yaml"
    )
    template = Template.from_yaml(
        template_path,
        block_registry=config.load_block_registry(),
    )

    ws = Workspace.load(Path(workspace_path))
    validator_registry = config.load_artifact_validator_registry()
    result = run_block(
        ws, block_name, template, config.project_root,
        input_bindings=bindings or None,
        args=extra_args or None,
        validator_registry=validator_registry,
    )
    print(
        f"Block {block_name!r} completed: "
        f"exit_code={result.exit_code}, elapsed={result.elapsed_s:.1f}s"
    )


def run_agent_command(args, extra_args: list[str]) -> None:
    """Run an agent block within an existing workspace.

    Reads the prompt file and launches the agent container.

    Args:
        args: Parsed argparse namespace with agent-specific fields.
        extra_args: Extra arguments passed to invoked containers.
    """
    config = load_project_config(Path.cwd())

    template_path = config.workspace_templates_dir / f"{args.template}.yaml"
    block_registry = config.load_block_registry()
    template = Template.from_yaml(
        template_path,
        block_registry=block_registry,
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
        block_name=args.block_name,
        agent_image=args.agent_image,
        auth_volume=args.auth_volume,
        model=args.model,
        max_turns=args.max_turns,
        total_timeout=args.total_timeout,
        source_dirs=args.source_dir or None,
        input_artifacts=input_artifacts or None,
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


def run_pattern_command(args, extra_args: list[str]) -> None:
    """Run a pattern through canonical block execution."""
    if extra_args:
        print(
            "ERROR: flywheel run pattern does not accept trailing "
            f"arguments after --: {extra_args!r}"
        )
        sys.exit(1)

    config = load_project_config(Path.cwd())
    pattern_path = (
        config.pattern_templates_dir / f"{args.pattern_name}.yaml"
    )
    if not pattern_path.exists():
        print(
            f"ERROR: no pattern named {args.pattern_name!r} in "
            f"{config.pattern_templates_dir}"
        )
        sys.exit(1)

    block_registry = config.load_block_registry()
    template = Template.from_yaml(
        config.workspace_templates_dir / f"{args.template}.yaml",
        block_registry=block_registry,
    )
    workspace = Workspace.load(Path(args.workspace))
    pattern = PatternDeclaration.from_yaml(pattern_path)
    validator_registry = config.load_artifact_validator_registry()

    try:
        result = run_pattern(
            workspace,
            pattern,
            template,
            config.project_root,
            validator_registry=validator_registry,
        )
    except PatternRunError as exc:
        print(f"Pattern {pattern.name!r} failed: {exc}")
        sys.exit(1)

    print(
        f"Pattern {pattern.name!r} completed: "
        f"run_id={result.run_id}, status={result.status}"
    )


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


def container_stop_command(
    *,
    workspace: str,
    template_name: str,
    block_name: str,
    reason: str,
) -> None:
    """Report that persistent runtime management is deferred."""
    del workspace, template_name, block_name, reason
    raise NotImplementedError(
        "persistent container runtime management is deferred until "
        "the persistent execution path is rebuilt on the canonical "
        "block execution pipeline"
    )


def container_list_command(*, workspace: str) -> None:
    """Report that persistent runtime management is deferred."""
    del workspace
    raise NotImplementedError(
        "persistent container runtime management is deferred until "
        "the persistent execution path is rebuilt on the canonical "
        "block execution pipeline"
    )

def _load_template_for(
    workspace_path: Path, template_name: str,
) -> Template:
    """Load the template declared for a given workspace.

    Walks up from the workspace path to find ``flywheel.yaml``,
    loads the project config, resolves the named template, and
    returns it with a registered block registry so lifecycle /
    block lookups work.
    """
    project_root = workspace_path.parent
    for candidate in [project_root, *project_root.parents]:
        if (candidate / "flywheel.yaml").is_file():
            project_root = candidate
            break
    config = load_project_config(project_root)
    registry = config.load_block_registry()
    template_path = (
        config.foundry_dir / "templates"
        / f"{template_name}.yaml"
    )
    return Template.from_yaml(
        template_path, block_registry=registry)
