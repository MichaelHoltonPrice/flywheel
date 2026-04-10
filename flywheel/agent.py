"""Agent block execution orchestration.

Runs an AI agent (Claude Code) inside a Docker container with access
to a block bridge service that lets it trigger nested block executions.
The agent reads source code, writes artifacts, and iteratively
invokes other blocks (e.g., evaluations) through an MCP tool.

The agent container receives:
- A workspace directory (read-write) for writing artifacts
- Source code directories (read-only)
- Previous artifacts as optional inputs (read-only)
- A block bridge endpoint for triggering nested block executions

Each block the agent invokes becomes a tracked block execution
with full artifact provenance in the workspace.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flywheel.block_bridge import BlockBridgeService
from flywheel.template import Template
from flywheel.workspace import Workspace


@dataclass(frozen=True)
class AgentResult:
    """Result of an agent block execution.

    Attributes:
        exit_code: The agent container's exit code.
        elapsed_s: Wall-clock time in seconds.
        evals_run: Number of evaluations the agent triggered.
    """

    exit_code: int
    elapsed_s: float
    evals_run: int


def _resolve_path(path: Path) -> str:
    """Resolve a path for Docker volume mounts (Windows-safe)."""
    resolved = str(path.resolve())
    if sys.platform == "win32":
        resolved = resolved.replace("\\", "/")
    return resolved


def run_agent_block(
    workspace: Workspace,
    template: Template,
    project_root: Path,
    prompt: str,
    agent_image: str = "flywheel-claude:latest",
    auth_volume: str = "claude-auth",
    model: str | None = None,
    max_invocations: int | None = None,
    max_turns: int | None = None,
    allowed_blocks: list[str] | None = None,
    source_dirs: list[str] | None = None,
    input_artifacts: dict[str, str] | None = None,
    output_names: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> AgentResult:
    """Run an agent block execution.

    Launches a Claude Code agent in a Docker container with a block
    bridge service that lets it invoke other block executions defined
    in the template. Each invocation creates real artifact instances
    and block execution records in the workspace.

    Args:
        workspace: The flywheel workspace for artifact tracking.
        template: The template containing block definitions that the
            agent can invoke.
        project_root: Path to the project root (for source mounts).
        prompt: The system prompt for the agent.
        agent_image: Docker image for the agent container.
        auth_volume: Docker named volume containing API credentials.
        model: Model name (e.g., "claude-sonnet-4-6").
        max_invocations: Maximum block invocations the agent can
            trigger (None = unlimited).
        max_turns: Maximum agent conversation turns.
        allowed_blocks: Block names the agent is allowed to invoke.
            None means all blocks in the template.
        source_dirs: Paths to source directories (relative to
            project_root) to mount read-only.
        input_artifacts: Maps mount names to artifact instance IDs
            for optional inputs (e.g., previous bot/summary).
        output_names: Artifact declaration names to collect from
            the agent workspace after completion.
        overrides: CLI flag overrides passed to invoked containers
            (e.g., {"subclass": "dueling", "episodes": "4000"}).

    Returns:
        AgentResult with exit code, elapsed time, and invocation count.
    """
    # Create agent workspace directory.
    agent_ws = workspace.path / "agent_workspace"
    agent_ws.mkdir(parents=True, exist_ok=True)

    # Snapshot execution count to compute invocations later.
    executions_before = len(workspace.executions)

    # Start block bridge service.
    bridge = BlockBridgeService(
        template=template,
        workspace=workspace,
        overrides=overrides,
        allowed_blocks=allowed_blocks,
        max_invocations=max_invocations,
    )
    port = bridge.start()
    bridge_endpoint = f"http://host.docker.internal:{port}"

    # Build Docker command.
    cmd = ["docker", "run", "--rm", "-i"]

    # Auth volume.
    cmd.extend(["-v", f"{auth_volume}:/home/claude/.claude"])

    # Agent workspace (read-write).
    cmd.extend(["-v", f"{_resolve_path(agent_ws)}:/workspace"])

    # Source directories (read-only).
    if source_dirs:
        for i, src in enumerate(source_dirs):
            src_path = project_root / src
            mount_point = "/source" if i == 0 else f"/source{i + 1}"
            cmd.extend([
                "-v", f"{_resolve_path(src_path)}:{mount_point}:ro",
            ])

    # Input artifacts (read-only).
    if input_artifacts:
        for mount_name, artifact_id in input_artifacts.items():
            if artifact_id not in workspace.artifacts:
                continue
            inst = workspace.artifacts[artifact_id]
            if inst.kind == "copy" and inst.copy_path:
                host_path = workspace.path / "artifacts" / inst.copy_path
                if host_path.exists():
                    cmd.extend([
                        "-v",
                        f"{_resolve_path(host_path)}:/input/{mount_name}:ro",
                    ])

    # Environment variables.
    # EVAL_BLOCK tells the MCP server which block to invoke by
    # default when the agent calls evaluate(). Projects can override
    # this to point at a different block name.
    default_block = allowed_blocks[0] if allowed_blocks else "eval_bot"
    env_vars = {
        "EVAL_ENDPOINT": bridge_endpoint,
        "EVAL_BLOCK": default_block,
        "MCP_SERVERS": "eval",
        "ALLOWED_TOOLS": "Read,Write,Edit,Glob,Grep",
    }
    if model:
        env_vars["MODEL"] = model
    if max_turns is not None:
        env_vars["MAX_TURNS"] = str(max_turns)

    for key, value in env_vars.items():
        cmd.extend(["-e", f"{key}={value}"])

    cmd.append(agent_image)

    # Launch agent container.
    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"

    print(f"  [agent] launching {agent_image}")
    start = time.monotonic()
    event_log = workspace.path / "agent_events.jsonl"

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )

        # Send prompt to stdin.
        process.stdin.write(prompt)
        process.stdin.close()

        # Stream stdout (JSON events from agent_runner.py).
        with open(event_log, "w") as log_f:
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                log_f.write(line + "\n")
                log_f.flush()

                # Print a summary of agent activity.
                try:
                    event = json.loads(line)
                    _log_event(event)
                except json.JSONDecodeError:
                    pass

        process.wait()
    except KeyboardInterrupt:
        print("  [agent] interrupted — terminating container")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        bridge.stop()

    elapsed = time.monotonic() - start

    # Collect output artifacts from agent workspace.
    if output_names:
        for name in output_names:
            # Convention: agent writes {name}.py or {name}.txt to
            # /workspace. Match by exact stem.
            for candidate in agent_ws.iterdir():
                if candidate.is_file() and candidate.stem == name:
                    workspace.register_artifact(
                        name, candidate,
                        source=f"agent output ({agent_image})",
                    )
                    break

    evals_run = len(workspace.executions) - executions_before

    print(
        f"  [agent] completed: exit_code={process.returncode}, "
        f"elapsed={elapsed:.1f}s, evals={evals_run}"
    )

    return AgentResult(
        exit_code=process.returncode,
        elapsed_s=elapsed,
        evals_run=evals_run,
    )


def _log_event(event: dict) -> None:
    """Print a concise summary of an agent event."""
    event_type = event.get("type", "")

    if event_type == "assistant":
        content = event.get("content", [])
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "")
                    preview = text[:120].replace("\n", " ")
                    if preview:
                        # Replace non-ASCII to avoid cp1252 errors on Windows.
                        safe = preview.encode("ascii", "replace").decode("ascii")
                        print(f"  [agent] {safe}")
                elif btype == "tool_use":
                    tool = block.get("name", "?")
                    if tool.startswith("mcp__"):
                        short = tool.split("__")[-1]
                        print(f"  [agent] [mcp] {short}()")
                    else:
                        print(f"  [agent] [tool] {tool}()")

    elif event_type == "result":
        subtype = event.get("subtype", "")
        print(f"  [agent] result: {subtype}")

    elif event_type == "error":
        msg = event.get("message", "unknown error")
        print(f"  [agent] ERROR: {msg}")
