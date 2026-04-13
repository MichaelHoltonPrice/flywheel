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
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
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


# Default total timeout for an agent step (4 hours).
DEFAULT_TOTAL_TIMEOUT = 14400


def _arc_post(url: str, payload: dict) -> tuple[int, dict]:
    """POST JSON to the ARC game server.

    Args:
        url: Full endpoint URL.
        payload: JSON-serializable request body.

    Returns:
        (status_code, body) tuple.
    """
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:200]}


def _init_arc_game(
    server_url: str, game_id: str,
) -> tuple[str, str, str, dict]:
    """Create a scorecard and start an ARC-AGI-3 game on the host game server.

    Args:
        server_url: Base URL of the game server (e.g., ``http://localhost:8001``).
        game_id: Game identifier (e.g., ``vc33-9851e02b``).

    Returns:
        (card_id, guid, initial_frame, raw_data) -- the scorecard ID,
        game session GUID, formatted initial frame text for the agent
        prompt, and the raw RESET response for artifact creation.

    Raises:
        RuntimeError: If the game server is unreachable or returns errors.
    """
    # Open a scorecard.
    status, resp = _arc_post(f"{server_url}/api/scorecard/open", {})
    if status != 200:
        raise RuntimeError(
            f"Failed to open scorecard: {status} {resp}"
        )
    card_id = resp.get("card_id", "")

    # RESET starts the game and returns the initial frame.
    status, data = _arc_post(
        f"{server_url}/api/cmd/RESET",
        {"card_id": card_id, "game_id": game_id},
    )
    if status != 200:
        raise RuntimeError(f"Failed to start game: {status} {data}")
    guid = data.get("guid", "")

    # Format the initial frame for the agent prompt.
    frame_lines: list[str] = []
    frame = data.get("frame")
    if frame and isinstance(frame, list) and len(frame) > 0:
        grid = frame[0]
        if isinstance(grid, list) and len(grid) > 0:
            for row in grid:
                frame_lines.append(" ".join(str(int(v)) for v in row))

    initial_frame = (
        f"STATE: {data.get('state', 'UNKNOWN')}\n"
        f"LEVELS_COMPLETED: {data.get('levels_completed', '?')}\n"
        f"WIN_LEVELS: {data.get('win_levels', '?')}\n"
        f"AVAILABLE_ACTIONS: {data.get('available_actions', [])}\n"
    )
    if frame_lines:
        h = len(frame_lines)
        w = len(frame_lines[0].split()) if frame_lines else 0
        colors = set()
        for row in data["frame"][0]:
            colors.update(row)
        initial_frame += (
            f"FRAME_SHAPE: {h}x{w}\n"
            f"COLORS_PRESENT: {sorted(colors)}\n"
            f"FRAME:\n" + "\n".join(frame_lines)
        )

    return card_id, guid, initial_frame, data


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
    total_timeout: int = DEFAULT_TOTAL_TIMEOUT,
    allowed_blocks: list[str] | None = None,
    source_dirs: list[str] | None = None,
    input_artifacts: dict[str, str] | None = None,
    output_names: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
    mcp_servers: str | None = None,
    allowed_tools: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_mounts: list[tuple[str, str, str]] | None = None,
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
        total_timeout: Maximum wall-clock seconds for the entire
            agent step. The agent container is killed if exceeded.
            Default: 14400 (4 hours).
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
        mcp_servers: Comma-separated MCP server names to enable
            (default: "eval"). Use "arc" for ARC-AGI-3 games.
        allowed_tools: Comma-separated tool whitelist for the agent
            (default: "Read,Write,Edit,Glob,Grep").
        extra_env: Additional environment variables to pass to the
            agent container.
        extra_mounts: Additional volume mounts as (host, container,
            mode) tuples.

    Returns:
        AgentResult with exit code, elapsed time, and invocation count.
    """
    # Create agent workspace directory (fresh each step).
    agent_ws = workspace.path / "agent_workspace"
    if agent_ws.exists():
        shutil.rmtree(agent_ws)
    agent_ws.mkdir(parents=True)

    # Seed the agent workspace with the latest artifacts from
    # prior steps so the agent can continue where the previous
    # step left off. For each output name, find the latest
    # artifact instance and copy its contents into the workspace.
    if output_names:
        for name in output_names:
            instances = workspace.instances_for(name)
            if instances:
                latest = instances[-1]  # sorted by created_at
                if latest.kind == "copy" and latest.copy_path:
                    src_dir = workspace.path / "artifacts" / latest.copy_path
                    if src_dir.exists():
                        for f in src_dir.iterdir():
                            if f.is_file():
                                shutil.copy2(f, agent_ws / f.name)

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

    # Extra mounts (e.g., game files for ARC-AGI-3).
    if extra_mounts:
        for host_path, container_path, mode in extra_mounts:
            cmd.extend([
                "-v", f"{_resolve_path(Path(host_path))}:{container_path}:{mode}",
            ])

    # Environment variables.
    default_block = allowed_blocks[0] if allowed_blocks else "eval_bot"
    env_vars = {
        "EVAL_ENDPOINT": bridge_endpoint,
        "EVAL_BLOCK": default_block,
        "MCP_SERVERS": mcp_servers or "eval",
        "ALLOWED_TOOLS": allowed_tools or "Read,Write,Edit,Glob,Grep",
    }
    if model:
        env_vars["MODEL"] = model
    if max_turns is not None:
        env_vars["MAX_TURNS"] = str(max_turns)
    # Disable Python output buffering so the host sees events
    # immediately (without this, stdout is block-buffered inside
    # Docker and the host event log lags behind).
    env_vars["PYTHONUNBUFFERED"] = "1"

    if extra_env:
        env_vars.update(extra_env)

    # ARC-AGI-3: create scorecard and start game on the host side
    # before launching the container.  The agent receives the card
    # ID, GUID, and initial frame — it never opens scorecards.
    arc_server = env_vars.get("ARC_SERVER_URL", "")
    arc_game = env_vars.get("GAME_ID", "")
    if arc_server and arc_game:
        # Rewrite ARC_SERVER_URL for the container (Docker alias).
        host_url = arc_server.replace(
            "host.docker.internal", "localhost",
        ).replace("localhost", "host.docker.internal")
        card_id, guid, initial_frame, raw_data = _init_arc_game(
            # Use localhost URL for host-side HTTP call.
            arc_server.replace("host.docker.internal", "localhost"),
            arc_game,
        )
        env_vars["ARC_CARD_ID"] = card_id
        env_vars["ARC_GUID"] = guid
        env_vars["ARC_SERVER_URL"] = host_url
        # Inject the initial frame into the prompt so the agent
        # can start playing immediately.
        prompt += f"\n\n# Initial game state\n\n{initial_frame}"
        print(f"  [agent] ARC game initialized: card={card_id[:8]}... "
              f"guid={guid[:8]}...")

        # Write initial frame to workspace so the MCP server can
        # hydrate pre-state tracking for the first game step.
        initial_state_file = agent_ws / ".arc_initial_state.json"
        initial_grid = raw_data.get("frame", [[]])[0]
        initial_state_file.write_text(json.dumps({
            "frame": initial_grid,
            "score": raw_data.get("score", 0),
        }), encoding="utf-8")

        # Create initial game_spec and game_session artifacts if the
        # template declares them.
        if "game_spec" in workspace.artifact_declarations:
            grid = raw_data.get("frame", [[]])[0]
            grid_h = len(grid) if grid else 0
            grid_w = len(grid[0]) if grid and grid[0] else 0
            spec_data = {
                "game_id": arc_game,
                "server_url": arc_server,
                "available_actions": raw_data.get(
                    "available_actions", []),
                "win_levels": raw_data.get("win_levels"),
                "grid_size": f"{grid_h}x{grid_w}",
            }
            spec_file = agent_ws / "_game_spec.json"
            spec_file.write_text(
                json.dumps(spec_data, indent=2), encoding="utf-8")
            workspace.register_artifact(
                "game_spec", spec_file,
                source="game initialization",
            )
            spec_file.unlink()

        if "game_session" in workspace.artifact_declarations:
            session_data = {
                "card_id": card_id,
                "guid": guid,
                "level": raw_data.get("levels_completed", 0),
                "action_count": 0,
                "state": raw_data.get("state", "initialized"),
            }
            session_file = agent_ws / "_game_session.json"
            session_file.write_text(
                json.dumps(session_data, indent=2), encoding="utf-8")
            session_instance = workspace.register_artifact(
                "game_session", session_file,
                source="game initialization",
            )
            session_file.unlink()
            env_vars["ARC_SESSION_ARTIFACT_ID"] = session_instance.id

    for key, value in env_vars.items():
        cmd.extend(["-e", f"{key}={value}"])

    cmd.append(agent_image)

    # Launch agent container.
    env = os.environ.copy()
    env["MSYS_NO_PATHCONV"] = "1"

    print(f"  [agent] launching {agent_image}")
    start = time.monotonic()
    event_log = workspace.path / "agent_events.jsonl"

    process = None
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

        # Drain stderr in a background thread to prevent deadlock
        # if the child writes more than the OS pipe buffer allows.
        stderr_log = workspace.path / "agent_stderr.log"
        def _drain_stderr():
            with open(stderr_log, "w", encoding="utf-8") as f:
                for line in process.stderr:
                    f.write(line)
        stderr_thread = threading.Thread(
            target=_drain_stderr, daemon=True)
        stderr_thread.start()

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

                # Kill the agent if the total timeout is exceeded.
                if time.monotonic() - start > total_timeout:
                    print(f"  [agent] total timeout "
                          f"({total_timeout}s) exceeded -- killing")
                    process.kill()
                    break

        process.wait()
        stderr_thread.join(timeout=5)
    except KeyboardInterrupt:
        if process is not None:
            print("  [agent] interrupted -- terminating container")
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
    if output_names and agent_ws.exists():
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

    exit_code = process.returncode if process is not None else -1

    print(
        f"  [agent] completed: exit_code={exit_code}, "
        f"elapsed={elapsed:.1f}s, evals={evals_run}"
    )

    return AgentResult(
        exit_code=exit_code,
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
