#!/bin/bash
# Entrypoint for the flywheel-claude container.
#
# When NETWORK_ISOLATION=1, configures iptables to allow only:
#   - Anthropic API (api.anthropic.com, port 443)
#   - DNS (port 53, for API IP rotation)
#   - Loopback (MCP servers, internal comms)
#   - host.docker.internal TCP ports listed in
#     ``HOST_WHITELIST_PORTS`` (comma-separated).  Default is
#     empty, i.e. no host access at all.  Projects that need
#     a specific host-side port declare it per-instance via
#     ``extra_env: HOST_WHITELIST_PORTS: <ports>`` in the
#     pattern YAML.
#
# Then drops to the non-root 'claude' user and runs the agent.
# The claude user cannot modify the iptables rules.
set -e

if [ "${NETWORK_ISOLATION}" = "1" ]; then
    # Resolve allowed destinations at startup.
    API_IPS=$(getent ahosts api.anthropic.com | awk '{print $1}' | sort -u)
    HOST_IP=$(getent ahosts host.docker.internal 2>/dev/null | awk '{print $1}' | head -1)

    # Default policy: drop all outbound.
    iptables -P OUTPUT DROP

    # Allow loopback.
    iptables -A OUTPUT -o lo -j ACCEPT

    # Allow established/related connections.
    iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

    # Allow DNS (Claude CLI may re-resolve during session).
    iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
    iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

    # Allow Anthropic API.
    for ip in $API_IPS; do
        iptables -A OUTPUT -d "$ip" -p tcp --dport 443 -j ACCEPT
    done

    # Allow host machine on explicitly whitelisted TCP ports only.
    # Unset/empty ``HOST_WHITELIST_PORTS`` means no host access;
    # that is the intended default and closes the accidental
    # host-reachability channel the blanket host ACCEPT used to
    # open.  Non-numeric entries abort container start so
    # operator typos surface loudly instead of silently dropping
    # a rule.
    if [ -n "$HOST_IP" ] && [ -n "${HOST_WHITELIST_PORTS}" ]; then
        IFS=',' read -ra _ports <<< "$HOST_WHITELIST_PORTS"
        for _port in "${_ports[@]}"; do
            _port=$(echo "$_port" | tr -d '[:space:]')
            if [ -z "$_port" ]; then
                continue
            fi
            if ! [[ "$_port" =~ ^[0-9]+$ ]]; then
                echo "[entrypoint] HOST_WHITELIST_PORTS contains non-numeric entry: $_port" >&2
                exit 1
            fi
            iptables -A OUTPUT -d "$HOST_IP" -p tcp --dport "$_port" -j ACCEPT
            echo "[entrypoint] host.docker.internal:$_port allowed"
        done
    fi

    # Explicit final drop.
    iptables -A OUTPUT -j DROP

    echo "[entrypoint] network isolation active"
fi

# --- Auth volume containment ---
#
# /home/claude/.claude is a shared Docker volume populated from
# the operator's host ~/.claude/.  The host directory carries a
# lot more than the in-container CLI needs: prior session JSONLs
# (~/.claude/projects/), shell history, plugins, telemetry,
# operator settings.  Anything we leave in place is reachable by
# the agent — past sessions become spoiler material, host
# settings (e.g. effortLevel) silently shape agent behavior.
#
# We defend in depth: even if the refresh command copied the
# whole host ~/.claude/ into the volume, we scrub the
# container's view at every start and synthesize a known-good
# settings.json with only the flags the headless CLI requires.
# The refresh-script-of-the-day is irrelevant; the agent only
# ever sees the post-entrypoint state.
#
# We do NOT touch the host filesystem -- the volume is a
# Docker-managed named volume separate from the host's
# ~/.claude/.  Removing files inside the container only removes
# them from the volume.

# 1. Allowlist scrub: keep only .credentials.json at top level.
find /home/claude/.claude -mindepth 1 -maxdepth 1 \
    ! -name '.credentials.json' \
    -exec rm -rf {} +

# 2. Synthesize the settings the headless CLI needs.
#    skipDangerousModePermissionPrompt is required so the SDK's
#    permission_mode=bypassPermissions actually skips the prompt
#    (otherwise the CLI hangs waiting for input).
cat > /home/claude/.claude/settings.json <<'JSON'
{
  "skipDangerousModePermissionPrompt": true
}
JSON
chown claude:claude /home/claude/.claude/settings.json
echo "[entrypoint] auth volume scrubbed; settings synthesized"

# --- Privilege split: stage state, lock /flywheel/, run as claude ---
#
# The agent process and the SDK's spawned ``claude`` CLI share
# a UID (the SDK can't spawn its child as a different user), so
# anything readable by the runner is also readable by the
# agent's Bash tool.  We can't stop the SDK's live session
# (~/.claude/projects/<encoded_cwd>/<sid>.jsonl) from being
# claude-readable while the SDK is using it, but we CAN keep
# the persisted ``/flywheel/state/session.jsonl`` (which
# survives across container launches and would otherwise spoil
# every relaunch) out of claude's reach entirely.
#
# Stage the persisted session into the SDK's expected location
# (claude-readable), then lock ``/flywheel/state/`` to root:700
# so the agent can't read or modify the persisted copy.  After
# the agent exits, copy the SDK's updated session back as root.
# Net effect: the persisted session lineage is owned by root
# end-to-end; only the live SDK working file is reachable by
# the agent.

SCRATCH_ENC=-scratch
PROJECTS_DIR=/home/claude/.claude/projects
SDK_SESSION_DIR="$PROJECTS_DIR/$SCRATCH_ENC"
PERSISTED_SESSION=/flywheel/state/session.jsonl
PERSISTED_HANDOFF=/flywheel/state/handoff_pending.json
SCRATCHPAD_STATE_DIR=/flywheel/state/scratchpad
export FLYWHEEL_SCRATCHPAD_DIR="${FLYWHEEL_SCRATCHPAD_DIR:-/scratch/.flywheel_scratchpad}"
SCRATCHPAD_RUNTIME_DIR="$FLYWHEEL_SCRATCHPAD_DIR"
if [ -z "$SCRATCHPAD_RUNTIME_DIR" ] \
    || [ "$SCRATCHPAD_RUNTIME_DIR" = "/" ] \
    || [ "$SCRATCHPAD_RUNTIME_DIR" = "/scratch" ]; then
    echo "[entrypoint] refusing unsafe FLYWHEEL_SCRATCHPAD_DIR: $SCRATCHPAD_RUNTIME_DIR" >&2
    exit 1
fi
case "$SCRATCHPAD_RUNTIME_DIR" in
    /scratch/*) ;;
    *)
        echo "[entrypoint] FLYWHEEL_SCRATCHPAD_DIR must be under /scratch: $SCRATCHPAD_RUNTIME_DIR" >&2
        exit 1
        ;;
esac

if [ -f "$PERSISTED_SESSION" ] && [ "${FLYWHEEL_ENABLE_SESSION_SPLICE:-0}" = "1" ]; then
    if [ -f "$PERSISTED_HANDOFF" ]; then
        python3 /app/handoff_session.py \
            --session "$PERSISTED_SESSION" \
            --result "${HANDOFF_RESULT_PATH:-/input/score/scores.json}" \
            --meta "$PERSISTED_HANDOFF" \
            --placeholder-marker "${HANDOFF_PLACEHOLDER_MARKER:-Evaluation requested.}" \
            --label "${HANDOFF_RESULT_LABEL:-Tool result}" || true
    else
        python3 /app/handoff_session.py \
            --session "$PERSISTED_SESSION" \
            --result "${HANDOFF_RESULT_PATH:-/input/score/scores.json}" \
            --placeholder-marker "${HANDOFF_PLACEHOLDER_MARKER:-Evaluation requested.}" \
            --label "${HANDOFF_RESULT_LABEL:-Tool result}" || true
    fi
fi

mkdir -p "$SDK_SESSION_DIR"
if [ -f "$PERSISTED_SESSION" ]; then
    SID=$(python3 -c '
import json, sys
try:
    line = open(sys.argv[1]).readline().strip()
    print(json.loads(line).get("sessionId", ""))
except Exception:
    pass
' "$PERSISTED_SESSION" 2>/dev/null || echo "")
    if [ -n "$SID" ]; then
        cp "$PERSISTED_SESSION" "$SDK_SESSION_DIR/$SID.jsonl"
        echo "[entrypoint] staged session $SID for SDK resume"
    fi
fi

rm -rf "$SCRATCHPAD_RUNTIME_DIR"
mkdir -p "$SCRATCHPAD_RUNTIME_DIR"
if [ -d "$SCRATCHPAD_STATE_DIR" ]; then
    cp -a "$SCRATCHPAD_STATE_DIR"/. "$SCRATCHPAD_RUNTIME_DIR"/
    echo "[entrypoint] staged scratchpad from managed state"
else
    echo "[entrypoint] initialized empty scratchpad"
fi
chown -R claude:claude "$SCRATCHPAD_RUNTIME_DIR"
chmod 700 "$SCRATCHPAD_RUNTIME_DIR" 2>/dev/null || true

chown -R claude:claude "$PROJECTS_DIR"

# Lock the persisted-state directory so claude cannot read it.
# Per docs/specs/state.md, ``state: none`` blocks do not receive a
# /flywheel/state mount.  This battery still wants a single code path
# for the lock and the post-agent scratchpad/session writes, so
# create the directory if it is missing.  Without a host mount the
# directory lives only in the container's overlay filesystem and is
# discarded on exit, which is exactly what ``state: none`` means.
mkdir -p /flywheel/state
chmod 700 /flywheel/state
chown -R root:root /flywheel/state

mkdir -p /flywheel/mcp_servers /flywheel/telemetry
chown root:root /flywheel/telemetry
chmod 700 /flywheel/telemetry

# /flywheel/mcp_servers is read-only project code already
# chowned to claude in the Dockerfile.  Nothing to do.

echo "[entrypoint] /flywheel/state locked to root; running agent as claude"

# Run the agent as claude WITHOUT exec so we resume here when
# it exits and can sync the SDK's updated session back to
# root-owned state.  ``set +e`` so we capture the rc instead of
# crashing on a non-zero agent exit.
set +e
RUNNER_LOG=/tmp/flywheel-claude-runner.jsonl
: > "$RUNNER_LOG"
chown root:root "$RUNNER_LOG"
chmod 600 "$RUNNER_LOG"
set -o pipefail
su -s /bin/bash claude -c "python3 /app/agent_runner.py" | tee "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
set +o pipefail

rm -rf "$SCRATCHPAD_STATE_DIR"
mkdir -p "$SCRATCHPAD_STATE_DIR"
if [ -d "$SCRATCHPAD_RUNTIME_DIR" ]; then
    cp -a "$SCRATCHPAD_RUNTIME_DIR"/. "$SCRATCHPAD_STATE_DIR"/
fi
chown -R root:root "$SCRATCHPAD_STATE_DIR"
chmod -R go-rwx "$SCRATCHPAD_STATE_DIR" 2>/dev/null || true
echo "[entrypoint] persisted scratchpad to managed state"

# Sync the latest SDK session back to /flywheel/state so the
# next launch can populate from it.  The SDK may have written a
# new session id (a fresh conversation, or a /compact branch);
# pick the most recently modified ``*.jsonl``.
LATEST=$(ls -t "$SDK_SESSION_DIR"/*.jsonl 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    if cp "$LATEST" "$PERSISTED_SESSION" 2>/dev/null; then
        chown root:root "$PERSISTED_SESSION" 2>/dev/null || true
        chmod 600 "$PERSISTED_SESSION" 2>/dev/null || true
        echo "[entrypoint] persisted SDK session $(basename "$LATEST" .jsonl)"
    else
        echo "[entrypoint] failed to persist SDK session" >&2
    fi
fi

python3 - <<'PY'
import json
from pathlib import Path

log_path = Path("/tmp/flywheel-claude-runner.jsonl")
pending_path = Path("/flywheel/state/handoff_pending.json")
latest_pending = None
try:
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            candidate = json.loads(line)
        except Exception:
            continue
        if candidate.get("type") == "handoff_pending":
            pending = candidate.get("pending")
            if isinstance(pending, list) and pending:
                first = pending[0]
                if isinstance(first, dict):
                    latest_pending = first
except Exception:
    latest_pending = None

if latest_pending is None:
    try:
        pending_path.unlink()
    except FileNotFoundError:
        pass
    raise SystemExit(0)

meta = {}
for key in (
    "tool_use_id",
    "tool_name",
    "termination_reason",
    "result_path",
    "result_label",
    "placeholder_marker",
):
    value = latest_pending.get(key)
    if isinstance(value, str) and value:
        meta[key] = value

pending_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
pending_path.chmod(0o600)
PY
chown root:root "$PERSISTED_HANDOFF" 2>/dev/null || true

HOME=/home/claude python3 - <<'PY'
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

os.environ["HOME"] = "/home/claude"

sdk_session_dir = Path("/home/claude/.claude/projects/-scratch")
persisted_session = Path("/flywheel/state/session.jsonl")
state_meta_path = Path("/flywheel/state/session_meta.json")
# Resume-critical state stays in /flywheel/state.  Full readbacks,
# raw JSONL copies, and SDK inspection output are telemetry sidecars.
snapshot_dir = Path("/flywheel/telemetry/session")
jsonl_dir = snapshot_dir / "jsonl"
telemetry_index_path = Path("/flywheel/telemetry/claude_session.json")
now = datetime.now(timezone.utc).isoformat()

try:
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for src in sdk_session_dir.glob("*.jsonl"):
        dst = jsonl_dir / src.name
        shutil.copy2(src, dst)
        os.chown(dst, 0, 0)
        os.chmod(dst, 0o600)
except Exception as exc:  # noqa: BLE001
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "copy_error.json").write_text(
        json.dumps({
            "error": type(exc).__name__,
            "message": str(exc),
        }, indent=2),
        encoding="utf-8",
    )

session_id = ""
try:
    first = persisted_session.read_text(encoding="utf-8").splitlines()[0]
    session_id = json.loads(first).get("sessionId", "")
except Exception:
    pass

(snapshot_dir / "active_session.json").write_text(
    json.dumps({"session_id": session_id}, indent=2),
    encoding="utf-8",
)
state_meta_path.write_text(
    json.dumps({
        "session_id": session_id,
        "cwd": "/scratch",
        "saved_at": now,
        "source": "flywheel-claude",
    }, indent=2),
    encoding="utf-8",
)
os.chown(state_meta_path, 0, 0)
os.chmod(state_meta_path, 0o600)

if session_id:
    try:
        from claude_agent_sdk import get_session_info, get_session_messages

        def encode(value):
            if hasattr(value, "model_dump"):
                return value.model_dump()
            if hasattr(value, "to_dict"):
                return value.to_dict()
            if hasattr(value, "__dict__"):
                return {
                    key: encode(val)
                    for key, val in value.__dict__.items()
                    if not key.startswith("_")
                }
            if isinstance(value, (list, tuple)):
                return [encode(item) for item in value]
            if isinstance(value, dict):
                return {key: encode(val) for key, val in value.items()}
            return value

        messages = get_session_messages(session_id, directory="/scratch")
        info = get_session_info(session_id, directory="/scratch")
        (snapshot_dir / "session_messages.json").write_text(
            json.dumps([encode(msg) for msg in messages], indent=2, default=str),
            encoding="utf-8",
        )
        (snapshot_dir / "session_info.json").write_text(
            json.dumps(encode(info), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        (snapshot_dir / "sdk_readback_error.json").write_text(
            json.dumps({
                "error": type(exc).__name__,
                "message": str(exc),
            }, indent=2),
            encoding="utf-8",
        )

def telemetry_path(path):
    try:
        rel = path.relative_to(Path("/flywheel/telemetry"))
    except ValueError:
        return str(path)
    return f"/flywheel/telemetry/{rel.as_posix()}"

jsonl_files = []
for path in sorted(jsonl_dir.glob("*.jsonl")):
    jsonl_files.append({
        "path": telemetry_path(path),
        "bytes": path.stat().st_size,
    })

session_files = {
    "active_session": telemetry_path(snapshot_dir / "active_session.json"),
    "raw_jsonl_dir": telemetry_path(jsonl_dir),
    "jsonl_files": jsonl_files,
}
for name in ("session_messages.json", "session_info.json", "sdk_readback_error.json"):
    path = snapshot_dir / name
    if path.exists():
        session_files[name.removesuffix(".json")] = {
            "path": telemetry_path(path),
            "bytes": path.stat().st_size,
        }

telemetry_index_path.write_text(
    json.dumps({
        "kind": "claude_session",
        "source": "flywheel-claude",
        "data": {
            "session_id": session_id,
            "cwd": "/scratch",
            "saved_at": now,
            "files": session_files,
        },
    }, indent=2),
    encoding="utf-8",
)

for path in snapshot_dir.rglob("*"):
    try:
        os.chown(path, 0, 0)
        if path.is_file():
            os.chmod(path, 0o600)
    except Exception:
        pass
try:
    os.chown(telemetry_index_path, 0, 0)
    os.chmod(telemetry_index_path, 0o600)
except Exception:
    pass
PY

python3 - <<'PY'
import json
from pathlib import Path

result_path = Path("/tmp/flywheel-claude-runner.jsonl")
telemetry_path = Path("/flywheel/telemetry/claude_usage.json")
results = []
try:
    for line in result_path.read_text(encoding="utf-8").splitlines():
        try:
            candidate = json.loads(line)
        except Exception:
            continue
        if candidate.get("type") == "result":
            results.append(candidate)
except Exception:
    raise SystemExit(0)
if not results:
    raise SystemExit(0)

def slim(result):
    data = {}
    for key in (
        "model",
        "total_cost_usd",
        "duration_ms",
        "duration_api_ms",
        "is_error",
        "num_turns",
        "stop_reason",
        "session_id",
        "uuid",
        "usage",
        "model_usage",
        "result",
        "structured_output",
        "permission_denials",
        "errors",
    ):
        if key in result:
            data[key] = result[key]
    return data

items = [slim(result) for result in results]
total_cost = 0.0
has_cost = False
for item in items:
    cost = item.get("total_cost_usd")
    if isinstance(cost, (int, float)):
        total_cost += float(cost)
        has_cost = True

latest = dict(items[-1])
latest["results"] = items
latest["summary"] = {
    "result_count": len(items),
    "total_cost_usd": total_cost if has_cost else None,
}

telemetry_path.write_text(json.dumps({
    "kind": "claude_usage",
    "source": "flywheel-claude",
    "data": latest,
}, indent=2), encoding="utf-8")
PY

python3 - <<'PY'
import json
from pathlib import Path

result_path = Path("/tmp/flywheel-claude-runner.jsonl")
compact_events_path = Path("/flywheel/telemetry/compact_events.json")
compact_events = []
try:
    for line in result_path.read_text(encoding="utf-8").splitlines():
        try:
            candidate = json.loads(line)
        except Exception:
            continue
        if candidate.get("type") == "compact_hook":
            compact_events.append(candidate)
except Exception:
    raise SystemExit(0)

if compact_events:
    compact_events_path.write_text(
        json.dumps({
            "kind": "claude_compact_events",
            "source": "flywheel-claude",
            "data": {"events": compact_events},
        }, indent=2),
        encoding="utf-8",
    )
PY

if [ "$RC" -eq 0 ]; then
    REASON=$(python3 - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/flywheel-claude-runner.jsonl")
status = None
try:
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            candidate = json.loads(line)
        except Exception:
            continue
        if candidate.get("type") == "agent_state":
            status = candidate.get("status")
except Exception:
    pass
print("normal" if status in (None, "", "complete") else status)
PY
)
    echo "$REASON" > /flywheel/termination
fi

exit "$RC"
