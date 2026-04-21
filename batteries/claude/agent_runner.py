#!/usr/bin/env python3
"""Agent runner using the Claude Agent SDK (ClaudeSDKClient).

Runs a Claude Code agent via ClaudeSDKClient in streaming mode,
which keeps a persistent connection to the CLI subprocess.  This
enables mid-session /compact without killing the session.

Context compaction
------------------

After each assistant response, the runner checks token usage.  When
usage crosses COMPACT_THRESHOLD of the estimated context window, the
runner sends /compact as a new message in the live session.  The CLI
compacts the conversation history and the agent continues working —
all within the same subprocess, with no session interruption.

Pause/resume
------------

- **Rate limit**: Detected from RateLimitEvent; auto-retries with
  exponential backoff (60s, 120s, 300s, 300s, 300s).  Falls back
  to .agent_resume after exhausting retries.
- **Auth error**: Detected from error messages; pauses and waits
  for .agent_resume file.
- **External resume**: Write a prompt (or empty) to .agent_resume
  in the workspace to continue after a pause.

Environment variables:
    MODEL           — Model to use (e.g., claude-sonnet-4-6)
    ALLOWED_TOOLS   — Comma-separated tool whitelist
    MAX_TURNS       — Total turn budget for the agent (optional)
    MCP_SERVERS     — Comma-separated list of MCP servers to enable.
                      Projects mount servers at
                      /flywheel/mcp_servers/.
    HANDOFF_TOOLS   — Comma-separated MCP tool names to intercept
                      via a PreToolUse hook.  When the agent calls
                      one or more in a single assistant turn, the
                      hook denies each (recording deny tool_results
                      in the session), captures every intercepted
                      call into ``/flywheel/control/
                      pending_tool_calls.json``, sets the exit
                      state in ``/flywheel/control/
                      agent_exit_state.json`` to status
                      ``tool_handoff``, and signals the runner to
                      exit cleanly so a host-side driver can run
                      the blocks out-of-band, splice every real
                      result into the captured state_dir's
                      session.jsonl, and let the next launch pick
                      it up via the ``/flywheel/state/`` populate.
                      Neither file is an artifact — they are
                      framework-owned runtime data the agent
                      launcher reads from the control tempdir
                      after the container exits.
    HANDOFF_DENY_MARKER — Substring that the splice helper uses to
                      locate the deny tool_result on disk.  Defaults
                      to ``handoff_to_flywheel``.

    Session resume:
        The runner reads ``/flywheel/state/session.jsonl`` on
        startup if flywheel populated one from a prior execution's
        captured state; otherwise it starts a fresh conversation.
        The session is written back to
        ``/flywheel/state/session.jsonl`` in a ``finally:`` at
        exit so flywheel captures it regardless of how the run
        ended.  No env var governs this — the presence or absence
        of the file is the whole signal.
    COMPACT_TOKEN_LIMIT — Explicit token count at which to trigger
                      compaction. Overrides the default percentage-
                      based calculation. Use for large-context models
                      or image-heavy workloads where the default is
                      too aggressive.
    TOOLS           — Comma-separated list of built-in tools to
                      enable. Overrides the default tool set. Use to
                      restrict agents to a safe subset (e.g., no web
                      access). If unset, all built-in tools are
                      available.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
)
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", "/scratch"))
STOP_FILE = WORKSPACE / ".stop"
# Framework-owned control files live under ``/flywheel/control/``
# so they don't clutter the agent's scratchpad namespace or the
# ``/output/<slot>/`` mounts that carry real artifact outputs.
# Flywheel mounts a host tempdir here; the runner reads/writes
# the files listed below and flywheel collects them after exit
# via :class:`flywheel.agent.AgentHandle.wait`.  Nothing under
# ``/flywheel/control/`` is an artifact.
CONTROL_DIR = Path("/flywheel/control")
RESUME_FILE = CONTROL_DIR / ".agent_resume"
POLL_INTERVAL = 5  # seconds between resume-file checks

# Rate limit auto-retry: sleep with exponential backoff before
# retrying.  Falls back to .agent_resume after max retries.
RATE_LIMIT_BACKOFFS = [60, 120, 300, 300, 300]  # seconds per attempt

# Proactive compaction: compact when input tokens exceed this fraction
# of the estimated context window.
COMPACT_THRESHOLD = 0.20
DEFAULT_CONTEXT_WINDOW = 200_000

# Session persistence: the SDK session JSONL is the container's
# private memory across restarts.  It lives at
# ``/flywheel/state/session.jsonl`` — flywheel populates
# ``/flywheel/state/`` from the prior execution's captured state
# at launch and captures its final contents after the container
# exits.  Not an artifact; not in the artifact graph.  See
# ``cyber-root/substrate-contract.md`` for the runtime contract.
STATE_DIR = Path("/flywheel/state")
SESSION_FILE = STATE_DIR / "session.jsonl"
SDK_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Prompt delivery: the pattern runner renders the prompt into a
# tempdir and mounts it read-only at ``/prompt``.  Reading the
# prompt from a file (rather than stdin) frees stdin for log
# capture and matches the runtime contract's "inputs arrive via
# file mounts" convention.
PROMPT_FILE = Path("/prompt/prompt.md")

# Orchestration control outputs.  The runner writes them to
# ``/flywheel/control/`` and flywheel's agent launcher reads them
# back from the host-side mount after the container exits.
# Neither file is an artifact — they are framework-owned runtime
# data consumed by the host handoff loop.
PENDING_TOOL_CALLS_FILE = (
    CONTROL_DIR / "pending_tool_calls.json")
EXIT_STATE_FILE = CONTROL_DIR / "agent_exit_state.json"
PENDING_TOOL_CALLS_SCHEMA_VERSION = 2
DEFAULT_HANDOFF_DENY_MARKER = "handoff_to_flywheel"

# Module-level handoff state.  The PreToolUse hook appends each
# intercepted tool_use to ``pending`` (preserving emission order);
# the main message loop reads the list after each message to know
# whether to exit cleanly with status ``tool_handoff``.  A dict so
# the hook (a free function passed to the SDK) can mutate the same
# object the loop reads.
_HANDOFF_STATE: dict[str, Any] = {"pending": []}



def _encode_cwd(cwd: str) -> str:
    """Encode a cwd path the way the Claude SDK does internally.

    Every non-alphanumeric character is replaced with a hyphen.
    For cwd="/scratch", the result is "-scratch".
    """
    return "".join(c if c.isalnum() else "-" for c in cwd)


def _sdk_session_path(
    session_id: str, cwd: str = "/scratch",
) -> Path:
    """Return the path where the Claude SDK stores session history."""
    encoded = _encode_cwd(cwd)
    return SDK_PROJECTS_DIR / encoded / f"{session_id}.jsonl"


def _export_session(session_id: str) -> None:
    """Copy the SDK session JSONL into ``/flywheel/state/session.jsonl``.

    Called in a finally block on exit.  Flywheel captures the
    contents of ``/flywheel/state/`` after the container exits and
    populates it from that capture on the next execution's launch,
    so this single file is enough to resume the conversation in a
    fresh container.

    Must not raise — it runs during teardown and a failure here
    shouldn't mask the real exit reason.
    """
    if not session_id:
        return
    try:
        src = _sdk_session_path(session_id)
        if src.exists():
            SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, SESSION_FILE)
            _emit({
                "type": "session_export",
                "session_id": session_id,
                "path": str(SESSION_FILE),
            })
        else:
            _emit({
                "type": "session_export_skip",
                "session_id": session_id,
                "reason": f"SDK session file not found: {src}",
            })
    except Exception as exc:
        _emit({
            "type": "session_export_error",
            "session_id": session_id,
            "message": str(exc),
        })


# ------------------------------------------------------------------
# Tool-call handoff hook
# ------------------------------------------------------------------

def _build_handoff_hook(
    handoff_tools: set[str],
    deny_marker: str,
):
    """Return a PreToolUse hook that intercepts handoff-mapped tools.

    The returned hook appends every intercepted tool_use into
    ``_HANDOFF_STATE["pending"]`` (preserving the SDK's emission
    order within an assistant turn) and returns a deny decision so
    the SDK records a deny tool_result for each ``tool_use_id``.
    The actual exit signaling is left to the main message loop
    (which polls ``_HANDOFF_STATE`` after each message).

    Multi-tool-use turns: the SDK fires PreToolUse for each
    tool_use in a single assistant message in order; we capture
    every match and the host-side driver runs / splices all of
    them in one stop/restart cycle.

    Args:
        handoff_tools: Set of fully qualified MCP tool names whose
            invocations should be intercepted.  Other tool names
            pass through (the hook returns an empty dict, which the
            SDK treats as "no opinion, proceed normally").
        deny_marker: Substring embedded in the deny reason; the
            host-side splice helper uses this to locate the deny
            tool_results on disk.

    Returns:
        An async hook callable matching the SDK's hook signature.
    """
    async def _hook(
        input_data: dict,
        tool_use_id: str | None,
        context,
    ) -> dict:
        tool_name = input_data.get("tool_name", "")
        if tool_name not in handoff_tools:
            return {}
        if tool_use_id is None:
            return {}
        _HANDOFF_STATE["pending"].append({
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "tool_input": input_data.get("tool_input", {}),
            "captured_at": datetime.now(UTC).isoformat(),
        })
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"permission denied: {deny_marker}"
                ),
            },
        }
    return _hook


def _persist_handoff(session_id: str) -> None:
    """Write the pending tool calls + state file for host pickup.

    Called by the main loop after it observes the handoff signal
    and the SDK has flushed the deny tool_results to the session
    JSONL.  The pending file is the canonical handoff payload; the
    state file mirrors the count + first tool name for quick visibility.
    """
    pending = list(_HANDOFF_STATE["pending"])
    if not pending:
        return
    payload = {
        "schema_version": PENDING_TOOL_CALLS_SCHEMA_VERSION,
        "session_id": session_id,
        "pending": pending,
    }
    PENDING_TOOL_CALLS_FILE.parent.mkdir(
        parents=True, exist_ok=True)
    PENDING_TOOL_CALLS_FILE.write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    _emit({
        "type": "handoff_pending",
        "count": len(pending),
        "tool_use_ids": [p["tool_use_id"] for p in pending],
        "tool_names": [p["tool_name"] for p in pending],
        "path": str(PENDING_TOOL_CALLS_FILE),
    })


# ------------------------------------------------------------------
# MCP server registry
# ------------------------------------------------------------------

# Built-in MCP servers used to ship the in-container ``eval`` proxy
# that round-tripped to the host bridge.  The bridge is gone; the
# only servers the runner now exposes are the project-provided ones
# scanned from ``MCP_SERVER_MOUNT_DIR``.
_MCP_REGISTRY: dict[str, tuple[str, Any]] = {}

# Default directory where project-provided MCP servers are mounted.
MCP_SERVER_MOUNT_DIR = "/flywheel/mcp_servers"


def _scan_mounted_servers() -> dict:
    """Discover project-provided MCP servers from the mount directory.

    Scans for ``*_mcp_server.py`` files. For each, derives the server
    name (strip ``_mcp_server.py`` suffix), builds a config, and
    returns a registry dict matching ``_MCP_REGISTRY`` format.

    An optional sidecar ``*_mcp_server.json`` manifest provides tool
    names to add to ``allowed_tools``. If absent, tools are discovered
    via MCP handshake (no pre-registration).

    Returns:
        Dict mapping server names to ``(server_id, factory)`` tuples.
    """
    mount_dir = Path(os.environ.get(
        "MCP_SERVER_MOUNT_DIR", MCP_SERVER_MOUNT_DIR))
    if not mount_dir.is_dir():
        return {}
    servers = {}
    for py_file in sorted(mount_dir.glob("*_mcp_server.py")):
        name = py_file.name.removesuffix("_mcp_server.py")
        config = {
            "command": "python3",
            "args": [str(py_file)],
            "env": dict(os.environ),
        }
        # Read tool names from sidecar manifest if present.
        manifest = py_file.with_suffix(".json")
        tools: list[str] = []
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                tools = data.get("tools", [])
            except Exception:
                pass
        # Capture config and tools in the closure.
        def _factory(c=config, t=tools):
            return c, t
        servers[name] = (name, _factory)
    return servers


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------

def _serialize(obj):
    """Recursively serialize an SDK object to a JSON-safe dict."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        d = {k: _serialize(v) for k, v in obj.__dict__.items()}
        cls = type(obj).__name__
        if "type" not in d:
            _BLOCK_TYPES = {
                "TextBlock": "text",
                "ToolUseBlock": "tool_use",
                "ToolResultBlock": "tool_result",
                "ThinkingBlock": "thinking",
            }
            d["type"] = _BLOCK_TYPES.get(cls, cls)
        return d
    return str(obj)


# ------------------------------------------------------------------
# Event emission and state management
# ------------------------------------------------------------------

_TYPE_MAP = {
    "AssistantMessage": "assistant",
    "UserMessage": "user",
    "SystemMessage": "system",
    "ResultMessage": "result",
    "RateLimitEvent": "rate_limit",
}


def _emit(data: dict) -> None:
    """Emit a JSON event to stdout."""
    print(json.dumps(data, default=str), flush=True)


def _save_state(session_id: str, status: str, reason: str = "") -> None:
    """Persist agent exit state into the declared output slot.

    The host-side handoff loop reads the resulting artifact to
    classify ``AgentResult.exit_reason`` and to decide whether to
    relaunch after a handoff.  Written into
    ``/output/agent_exit_state/`` so it lands as a first-class
    artifact instance tied to this execution.
    """
    state = {
        "session_id": session_id,
        "status": status,
        "reason": reason,
        "timestamp": time.time(),
    }
    EXIT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    EXIT_STATE_FILE.write_text(json.dumps(state, indent=2))
    _emit({"type": "agent_state", **state})


async def _wait_for_resume() -> str:
    """Block until .agent_resume file appears in the workspace."""
    _emit({
        "type": "agent_state",
        "status": "waiting_for_resume",
        "resume_file": str(RESUME_FILE),
    })
    while True:
        if RESUME_FILE.exists():
            text = RESUME_FILE.read_text().strip()
            RESUME_FILE.unlink()
            return text or "Continue from where you left off."
        await anyio.sleep(POLL_INTERVAL)


def _emit_message(message) -> None:
    """Serialize and emit an SDK message."""
    cls_name = type(message).__name__
    if hasattr(message, "model_dump"):
        data = message.model_dump()
    elif hasattr(message, "to_dict"):
        data = message.to_dict()
    elif hasattr(message, "__dict__"):
        data = message.__dict__
        if "content" in data and isinstance(data["content"], list):
            data["content"] = [_serialize(b) for b in data["content"]]
    else:
        data = {"type": "unknown", "repr": repr(message)}
    data["type"] = _TYPE_MAP.get(cls_name, cls_name)
    _emit(data)


def _get_input_tokens(message) -> int:
    """Extract total input tokens from an AssistantMessage."""
    usage = getattr(message, "usage", None)
    if not isinstance(usage, dict):
        return 0
    inp = usage.get("input_tokens", 0)
    cache_c = usage.get("cache_creation_input_tokens", 0)
    cache_r = usage.get("cache_read_input_tokens", 0)
    total = inp + cache_c + cache_r
    return total


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main() -> None:
    """Run the Claude Code agent with mid-session compaction."""
    # --- Read initial prompt from the /prompt/ mount ---
    if not PROMPT_FILE.is_file():
        _emit({
            "type": "error",
            "message": (
                f"Prompt file not found at {PROMPT_FILE}; the "
                f"launcher must mount a read-only prompt "
                f"directory at /prompt/ with a prompt.md file."
            ),
        })
        sys.exit(1)
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    if not prompt.strip():
        _emit({"type": "error", "message": "Empty prompt"})
        sys.exit(1)

    # --- Setup ---
    claude_json = os.path.expanduser("~/.claude.json")
    if not os.path.exists(claude_json):
        with open(claude_json, "w") as f:
            f.write("{}")

    model = os.environ.get("MODEL", "")
    allowed_tools_str = os.environ.get(
        "ALLOWED_TOOLS", "Read,Write,Edit,Glob,Grep",
    )
    allowed_tools = [
        t.strip() for t in allowed_tools_str.split(",") if t.strip()
    ]
    max_turns_str = os.environ.get("MAX_TURNS", "")
    max_turns = int(max_turns_str) if max_turns_str else None

    # Register MCP servers (built-in + project-mounted).
    mounted_servers = _scan_mounted_servers()
    mcp_servers_str = os.environ.get("MCP_SERVERS", "")
    requested = [
        s.strip() for s in mcp_servers_str.split(",") if s.strip()
    ]
    mcp_servers = {}
    for name in requested:
        entry = _MCP_REGISTRY.get(name)
        if entry is None:
            entry = mounted_servers.get(name)
        if entry is None:
            _emit({
                "type": "warning",
                "message": f"Unknown MCP server: {name!r}",
            })
            continue
        server_id, factory = entry
        result = factory()
        if result is None:
            continue
        config, tools = result
        mcp_servers[server_id] = config
        for t in tools:
            if t not in allowed_tools:
                allowed_tools.append(t)

    # Context window estimate.
    context_window = DEFAULT_CONTEXT_WINDOW
    if model and "1m" in model.lower():
        context_window = 1_000_000
    compact_env = os.environ.get("COMPACT_TOKEN_LIMIT", "")
    if compact_env:
        compact_token_limit = int(compact_env)
    else:
        compact_token_limit = int(context_window * COMPACT_THRESHOLD)

    # --- Built-in tool whitelist ---
    tools_str = os.environ.get("TOOLS", "")
    tools_whitelist = [
        t.strip() for t in tools_str.split(",") if t.strip()
    ] if tools_str else None

    # --- Tool-call handoff hook ---
    handoff_tools_str = os.environ.get("HANDOFF_TOOLS", "")
    handoff_tools = {
        t.strip() for t in handoff_tools_str.split(",") if t.strip()
    }
    handoff_deny_marker = os.environ.get(
        "HANDOFF_DENY_MARKER", DEFAULT_HANDOFF_DENY_MARKER)
    handoff_hooks_config = None
    if handoff_tools:
        hook_callable = _build_handoff_hook(
            handoff_tools, handoff_deny_marker)
        handoff_hooks_config = {
            "PreToolUse": [
                HookMatcher(matcher=t, hooks=[hook_callable])
                for t in sorted(handoff_tools)
            ],
        }
        _emit({
            "type": "handoff_hook_registered",
            "tools": sorted(handoff_tools),
            "deny_marker": handoff_deny_marker,
        })

    # --- Build options ---
    options = ClaudeAgentOptions(
        cwd="/scratch",
        allowed_tools=allowed_tools,
        permission_mode="bypassPermissions",
    )
    if tools_whitelist is not None:
        options.tools = tools_whitelist
    if model:
        options.model = model
    if handoff_hooks_config is not None:
        options.hooks = handoff_hooks_config
    if mcp_servers:
        options.mcp_servers = mcp_servers
    if max_turns:
        options.max_turns = max_turns

    # --- Session resume from /flywheel/state/ ---
    # Flywheel populates ``/flywheel/state/`` with the prior execution's
    # captured state before launching us.  If a session file is
    # present, extract the SDK session id from its first line and
    # hand it to the SDK so the conversation continues.  Absence
    # of the file means "first execution in this lineage";
    # nothing to resume.
    session_id = ""
    if SESSION_FILE.exists():
        resume_sid = SESSION_FILE.stem
        try:
            first_line = SESSION_FILE.read_text(
                encoding="utf-8").split("\n", 1)[0]
            entry = json.loads(first_line)
            sid_from_file = entry.get("sessionId", "")
            if sid_from_file:
                resume_sid = sid_from_file
        except Exception:
            pass  # Fall back to filename stem.
        sdk_dest = _sdk_session_path(resume_sid)
        sdk_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SESSION_FILE, sdk_dest)
        options.resume = resume_sid
        session_id = resume_sid
        _emit({
            "type": "session_resume",
            "session_id": resume_sid,
            "source": str(SESSION_FILE),
        })

    # --- Connect and run ---
    last_input_tokens = 0
    rate_limit_retries = 0

    try:
        async with ClaudeSDKClient(options=options) as client:
            # Send initial prompt.
            await client.query(prompt)

            # Process responses.  receive_response() yields messages
            # until a ResultMessage, then stops.  After compaction or
            # follow-up queries, we call receive_response() again to
            # continue processing.
            while True:
                completed = False
                pause_reason = None

                async for message in client.receive_response():
                    try:
                        cls_name = type(message).__name__

                        # Capture session_id.
                        sid = getattr(message, "session_id", None)
                        if sid:
                            session_id = sid

                        # Track token usage.
                        if isinstance(message, AssistantMessage):
                            rate_limit_retries = 0  # Reset on success.
                            tokens = _get_input_tokens(message)
                            if tokens > 0:
                                last_input_tokens = tokens

                            # Mid-session compaction trigger.
                            # Breaking out of receive_response() does
                            # NOT kill the ClaudeSDKClient connection.
                            # We can send /compact and then continue.
                            if last_input_tokens >= compact_token_limit:
                                pause_reason = "compact_needed"
                                _emit_message(message)
                                break

                        # Rate limit detection.
                        if cls_name == "RateLimitEvent":
                            info = getattr(
                                message, "rate_limit_info", None,
                            )
                            status = "unknown"
                            if info:
                                status = getattr(
                                    info, "status", "unknown",
                                )
                            if status == "rejected":
                                pause_reason = "rate_limit"
                                _emit({
                                    "type": "rate_limit",
                                    "status": "rejected",
                                })
                                break
                            _emit({
                                "type": "rate_limit",
                                "status": status,
                            })
                            continue

                        # Completion detection.
                        if isinstance(message, ResultMessage):
                            subtype = getattr(message, "subtype", "")
                            if subtype == "success":
                                completed = True

                        # Emit the message.
                        _emit_message(message)

                        # Graceful stop: the host wrote .agent_stop
                        # to the workspace.  The current tool response
                        # is already in the session — exit cleanly so
                        # the finally block exports it.
                        if STOP_FILE.exists():
                            pause_reason = "stop_requested"
                            break

                        # Tool-call handoff: PreToolUse hook captured
                        # one or more mapped tools in this turn.
                        # All deny tool_results are now in the live
                        # session; exit cleanly so the finally block
                        # exports the JSONL and the host driver can
                        # splice + restart.
                        if _HANDOFF_STATE["pending"]:
                            pause_reason = "tool_handoff"
                            break

                    except Exception as e:
                        _emit({
                            "type": "error",
                            "message": str(e),
                        })

                # --- After receive_response() returns ---

                # Graceful stop.
                if pause_reason == "stop_requested":
                    _emit({
                        "type": "agent_state",
                        "status": "stopping",
                        "reason": "stop_requested",
                    })
                    STOP_FILE.unlink(missing_ok=True)
                    _save_state(session_id, "stopped", "stop_requested")
                    return

                # Tool-call handoff.
                if pause_reason == "tool_handoff":
                    pending = list(_HANDOFF_STATE["pending"])
                    _persist_handoff(session_id)
                    if pending:
                        first = pending[0]["tool_name"]
                        reason = (
                            first if len(pending) == 1
                            else f"{first} (+{len(pending) - 1} more)"
                        )
                    else:
                        reason = ""
                    _save_state(
                        session_id, "tool_handoff", reason)
                    return

                # Natural completion.
                if completed:
                    _save_state(session_id, "complete")
                    return

                # Rate limit: auto-retry with backoff.
                if pause_reason == "rate_limit":
                    if rate_limit_retries < len(RATE_LIMIT_BACKOFFS):
                        delay = RATE_LIMIT_BACKOFFS[rate_limit_retries]
                        rate_limit_retries += 1
                        _emit({
                            "type": "rate_limit_retry",
                            "attempt": rate_limit_retries,
                            "max_attempts": len(RATE_LIMIT_BACKOFFS),
                            "delay_s": delay,
                        })
                        _save_state(
                            session_id, "paused",
                            f"rate_limit (retry {rate_limit_retries}"
                            f"/{len(RATE_LIMIT_BACKOFFS)} in {delay}s)",
                        )
                        await anyio.sleep(delay)
                        await client.query(
                            "Continue from where you left off."
                        )
                        continue
                    # Exhausted retries — fall through to manual resume.
                    _emit({
                        "type": "rate_limit_exhausted",
                        "attempts": rate_limit_retries,
                    })

                # External pause (auth error, exhausted rate limit).
                if pause_reason and pause_reason != "compact_needed":
                    _save_state(session_id, "paused", pause_reason)
                    resume_prompt = await _wait_for_resume()
                    await client.query(resume_prompt)
                    continue

                # --- Proactive compaction ---
                # Triggered either mid-session (compact_needed) or
                # after a ResultMessage when tokens are high.
                if (
                    pause_reason == "compact_needed"
                    or last_input_tokens >= compact_token_limit
                ):
                    tokens_before = last_input_tokens
                    _emit({
                        "type": "compact",
                        "message": (
                            f"Context at {tokens_before:,} tokens "
                            f"(limit {compact_token_limit:,}), "
                            f"sending /compact"
                        ),
                        "tokens_before": tokens_before,
                        "threshold": compact_token_limit,
                    })

                    # Interrupt the current generation so the CLI
                    # can accept the /compact command.  Then drain
                    # any remaining messages from the interrupted
                    # response before sending /compact.
                    await client.interrupt()
                    async for msg in client.receive_response():
                        _emit_message(msg)

                    # Send /compact in the live session.
                    await client.query("/compact")
                    async for msg in client.receive_response():
                        sid = getattr(msg, "session_id", None)
                        if sid:
                            session_id = sid
                        # Track post-compact tokens.
                        if isinstance(msg, AssistantMessage):
                            tokens = _get_input_tokens(msg)
                            if tokens > 0:
                                last_input_tokens = tokens

                    _emit({
                        "type": "compact",
                        "message": "Compaction complete",
                        "tokens_before": tokens_before,
                        "tokens_after": last_input_tokens,
                        "success": True,
                    })

                    # Raise the threshold so we don't compact again
                    # immediately if tokens didn't drop much.  Next
                    # compaction triggers at current level + 50% of
                    # the original threshold.
                    compact_token_limit = (
                        last_input_tokens
                        + int(context_window * COMPACT_THRESHOLD * 0.5)
                    )

                    # Tell the agent to keep working.
                    await client.query(
                        "Your context was compacted automatically. "
                        "This is routine. Resume what you were doing."
                    )
                    continue

                # ResultMessage without completion and without needing
                # compact — this is error_max_turns or similar.  The
                # agent used all its turns.  Exit cleanly so the
                # orchestrator can relaunch with session resume.
                _save_state(session_id, "complete", "max_turns")
                return

    except Exception as e:
        err = str(e).lower()
        if "auth" in err or "401" in err:
            _emit({"type": "error", "message": f"Auth error: {e}"})
            _save_state(session_id, "paused", "auth_error")
            # Can't resume inside ClaudeSDKClient after auth failure.
            return
        _emit({"type": "error", "message": str(e)})
        _save_state(session_id, "paused", "error")
    finally:
        # Export the session JSONL for artifact collection, regardless
        # of how the agent exited (success, error, SIGTERM).
        _export_session(session_id)


if __name__ == "__main__":
    anyio.run(main)
