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
        --template TEMPLATE [--project-hooks MODULE:CLASS]
        [--model MODEL] [--max-runtime SECONDS]
        [-- project-specific args...]
    flywheel import artifact --workspace PATH --name NAME
        --from SOURCE [--source TEXT]

Multi-round / multi-agent workflows are expressed as patterns
under ``<project>/patterns/`` and run via
``flywheel run pattern``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from flywheel.agent import AgentBlockConfig, run_agent_block
from flywheel.agent_executor import AgentExecutor
from flywheel.config import load_project_config
from flywheel.execution import run_block
from flywheel.pattern import Pattern, discover_patterns
from flywheel.pattern_runner import PatternRunner
from flywheel.project_hooks import load_project_hooks_class
from flywheel.run_defaults import RunDefaults
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
        help="Pattern name (file stem under <project>/patterns/).")
    pattern_parser.add_argument("--workspace", required=True)
    pattern_parser.add_argument("--template", required=True)
    pattern_parser.add_argument(
        "--project-hooks", default=None,
        dest="project_hooks",
        help="Project hooks class as module.path:ClassName. "
        "Overrides 'project_hooks' in flywheel.yaml.")
    pattern_parser.add_argument("--model", default=None)
    pattern_parser.add_argument(
        "--max-runtime", type=int, default=None,
        dest="max_runtime",
        help="Hard wall-clock cap on the whole pattern run, in "
        "seconds.  Default: wait until all continuous-role "
        "agents finish naturally.")
    pattern_parser.add_argument(
        "--poll-interval", type=float, default=1.0,
        dest="poll_interval",
        help="Seconds between ledger scans for trigger "
        "evaluation (default: 1.0).")
    pattern_parser.add_argument(
        "--total-timeout", type=int, default=14400,
        help="Per-agent wall-clock cap (default: 14400 = 4h). "
        "Roles can override this in their YAML.")
    pattern_parser.add_argument(
        "--max-turns", type=int, default=200)
    pattern_parser.add_argument(
        "--auth-volume", default="claude-auth")
    pattern_parser.add_argument(
        "--agent-image", default="flywheel-claude:latest")

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

    Reads the prompt file and launches the agent container.

    Args:
        args: Parsed argparse namespace with agent-specific fields.
        extra_args: Extra arguments passed to invoked containers.
    """
    config = load_project_config(Path.cwd())

    template_path = config.templates_dir / f"{args.template}.yaml"
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
    """Run a declarative pattern with optional project hooks.

    Discovers ``<project_root>/patterns/<name>.yaml`` (mirrors
    template discovery), loads project hooks if configured,
    builds a :class:`RunDefaults` and an
    :class:`flywheel.executor.BlockExecutor` factory from CLI
    flags + hook overrides, and hands off to
    :class:`PatternRunner`.

    Project hooks are expected to return:

    * ``executor_factory`` (optional): the
      :class:`flywheel.pattern_runner.ExecutorFactory` the
      runner will use for every block dispatch.  Hooks that
      need agent-battery wiring (e.g. cyberarc's ``on_tool``
      handoff loop) construct an
      :class:`flywheel.agent_executor.AgentExecutor` here and
      return a factory that hands it out.  When absent, the
      CLI builds a default agent-only factory from CLI flags
      + the legacy battery-shaped overrides below.
    * ``per_instance_runtime_config`` (optional): forwarded
      verbatim to the runner's same-named parameter.
    * ``defaults`` (optional): a free-form dict merged into
      :attr:`RunDefaults.defaults`.  Each executor reads what
      it cares about and ignores the rest.
    * Legacy battery-shaped keys (``agent_image``, ``model``,
      ``mcp_servers``, ``extra_env``, ``extra_mounts``,
      ``prompt_substitutions``, ...): consumed only when the
      hooks did *not* supply ``executor_factory``.  In that
      mode the CLI builds a single
      :class:`flywheel.agent_executor.AgentExecutor` carrying
      these defaults and uses it for every block.

    Args:
        args: Parsed argparse namespace with pattern-specific
            fields.
        extra_args: Project-specific arguments passed after
            ``--``; forwarded verbatim to the project hooks'
            ``init``.
    """
    config = load_project_config(Path.cwd())

    template_path = config.templates_dir / f"{args.template}.yaml"
    block_registry = config.load_block_registry()
    template = Template.from_yaml(
        template_path, block_registry=block_registry)

    ws = Workspace.load(Path(args.workspace))

    patterns = discover_patterns(config.patterns_dir)
    if args.pattern_name not in patterns:
        known = sorted(patterns) or ["<none>"]
        print(
            f"ERROR: no pattern named {args.pattern_name!r} in "
            f"{config.patterns_dir}.  Known patterns: "
            f"{', '.join(known)}"
        )
        sys.exit(1)

    pattern = Pattern.from_yaml(
        patterns[args.pattern_name],
        block_registry=block_registry,
    )
    print(
        f"  [flywheel] running pattern {pattern.name!r} "
        f"({len(pattern.roles)} role(s))"
    )

    hooks_path = args.project_hooks or config.project_hooks
    hooks = None
    overrides: dict[str, Any] = {}
    if hooks_path:
        hooks_cls = load_project_hooks_class(hooks_path)
        hooks = hooks_cls()
        if hasattr(hooks, "init"):
            overrides = hooks.init(
                ws, template, config.project_root, extra_args,
            ) or {}
    elif extra_args:
        # Loud failure: project args were passed but nothing
        # is wired up to receive them.  Better to fail than to
        # let a typo silently drop a flag.
        print(
            f"ERROR: extra args {extra_args!r} were passed "
            f"after `--` but no project_hooks are configured "
            f"to consume them.  Set 'project_hooks' in "
            f"flywheel.yaml or pass --project-hooks."
        )
        sys.exit(1)

    executor_factory = overrides.get("executor_factory")
    if executor_factory is None:
        # No project-supplied factory: build a default
        # AgentExecutor configured from CLI args + the legacy
        # battery-shaped overrides hooks may still set.  This
        # is the path agent-only patterns take when their
        # project doesn't need any host-side dispatch beyond
        # what the agent battery already provides.
        agent_executor = AgentExecutor(
            template=template,
            project_root=config.project_root,
            agent_image=overrides.get(
                "agent_image", args.agent_image),
            auth_volume=overrides.get(
                "auth_volume", args.auth_volume),
            model=overrides.get("model", args.model),
            max_turns=overrides.get(
                "max_turns", args.max_turns),
            total_timeout=overrides.get(
                "total_timeout", args.total_timeout),
            mcp_servers=overrides.get("mcp_servers"),
            allowed_tools=overrides.get("allowed_tools"),
            extra_env=overrides.get("extra_env"),
            extra_mounts=overrides.get("extra_mounts"),
            isolated_network=overrides.get(
                "isolated_network", True),
        )
        executor_factory = lambda _block_def: agent_executor

    defaults_bag: dict[str, Any] = dict(
        overrides.get("defaults") or {})
    if "prompt_substitutions" in overrides:
        defaults_bag.setdefault(
            "prompt_substitutions",
            overrides["prompt_substitutions"])

    run_defaults = RunDefaults(
        workspace=ws,
        template=template,
        project_root=config.project_root,
        defaults=defaults_bag,
    )

    runner_kwargs: dict[str, Any] = {}
    if "per_instance_runtime_config" in overrides:
        runner_kwargs["per_instance_runtime_config"] = (
            overrides["per_instance_runtime_config"])

    try:
        runner = PatternRunner(
            pattern,
            defaults=run_defaults,
            executor_factory=executor_factory,
            poll_interval_s=args.poll_interval,
            max_total_runtime_s=args.max_runtime,
            **runner_kwargs,
        )
        result = runner.run()
    finally:
        if hooks is not None and hasattr(hooks, "teardown"):
            hooks.teardown()

    print(
        f"\nPattern {pattern.name!r} complete: "
        f"{result.agents_launched} agent(s) launched"
    )
    for role_name, count in result.cohorts_by_role.items():
        print(f"  {role_name}: {count} cohort(s)")


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
    """Tear down the request-response runtime for a workspace / block.

    Loads the project config to resolve the block registry and
    template, constructs a :class:`RequestResponseExecutor`
    against the live workspace, and calls its
    :meth:`~flywheel.executor.RequestResponseExecutor.shutdown`.
    Shutdown is best-effort; an absent runtime is a no-op.
    """
    from flywheel.executor import RequestResponseExecutor

    ws_path = Path(workspace)
    ws = Workspace.load(ws_path)
    template = _load_template_for(ws_path, template_name)
    executor = RequestResponseExecutor(template)
    key = RequestResponseExecutor._attachment_key(block_name, ws)
    handled = executor.shutdown(
        key, reason=reason,
        block_name=block_name, workspace=ws,
    )
    if handled:
        print(
            f"[container stop] block={block_name!r} "
            f"workspace={ws.name!r} — teardown signalled"
        )
    else:
        print(
            f"[container stop] block={block_name!r} "
            f"workspace={ws.name!r} — no matching runtime "
            f"found (already stopped or never started)"
        )


def container_list_command(*, workspace: str) -> None:
    """Enumerate request-response runtimes for a workspace.

    Reports whichever runtimes Docker currently has labelled for
    this workspace, regardless of whether the current flywheel
    process started them.
    """
    import subprocess

    from flywheel.executor import _RUNTIME_LABEL_WORKSPACE

    ws_path = Path(workspace)
    ws = Workspace.load(ws_path)
    filter_arg = f"label={_RUNTIME_LABEL_WORKSPACE}={ws.name}"
    try:
        result = subprocess.run(
            [
                "docker", "ps",
                "--filter", filter_arg,
                "--format",
                "{{.Names}}\t{{.Status}}\t{{.Image}}",
            ],
            capture_output=True, check=False, text=True,
        )
    except FileNotFoundError:
        print("[container list] docker not found")
        sys.exit(1)
    if result.returncode != 0:
        print(
            f"[container list] docker ps failed: "
            f"{result.stderr.strip()}"
        )
        sys.exit(1)
    lines = [
        line for line in result.stdout.splitlines() if line]
    if not lines:
        print(
            f"[container list] no runtimes labelled for "
            f"workspace {ws.name!r}")
        return
    print(
        f"[container list] runtimes for workspace "
        f"{ws.name!r}:"
    )
    for line in lines:
        print(f"  {line}")


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
