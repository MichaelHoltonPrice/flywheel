#!/bin/bash
# Entrypoint for the flywheel-codex container.
set -e

PHASE_LOG=/tmp/flywheel-codex-phases.tsv
: > "$PHASE_LOG"
phase() {
    python3 - "$PHASE_LOG" "$1" <<'PY'
import sys
import time

path, name = sys.argv[1], sys.argv[2]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(f"{name}\t{time.monotonic():.9f}\n")
PY
}

phase entrypoint_start

if [ "${NETWORK_ISOLATION}" = "1" ]; then
    CODEX_ALLOWED_HOSTS="${CODEX_ALLOWED_HOSTS:-api.openai.com,auth.openai.com,chatgpt.com,ab.chatgpt.com,chat.openai.com}"
    HOST_IP=$(getent ahosts host.docker.internal 2>/dev/null | awk '{print $1}' | head -1)
    iptables -P OUTPUT DROP
    iptables -A OUTPUT -o lo -j ACCEPT
    iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
    iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
    IFS=',' read -ra _codex_hosts <<< "$CODEX_ALLOWED_HOSTS"
    for _host in "${_codex_hosts[@]}"; do
        _host=$(echo "$_host" | tr -d '[:space:]')
        if [ -z "$_host" ]; then
            continue
        fi
        if ! [[ "$_host" =~ ^[A-Za-z0-9._-]+$ ]]; then
            echo "[entrypoint] CODEX_ALLOWED_HOSTS contains invalid host: $_host" >&2
            exit 1
        fi
        _ips=$(getent ahosts "$_host" | awk '{print $1}' | sort -u)
        if [ -z "$_ips" ]; then
            echo "[entrypoint] CODEX_ALLOWED_HOSTS host did not resolve: $_host" >&2
            exit 1
        fi
        for ip in $_ips; do
            iptables -A OUTPUT -d "$ip" -p tcp --dport 443 -j ACCEPT
        done
    done
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
        done
    fi
    iptables -A OUTPUT -j DROP
    echo "[entrypoint] network isolation active"
fi
phase network_setup_done

export CODEX_HOME="${CODEX_HOME:-/home/codex/.codex}"
export FLYWHEEL_SCRATCHPAD_DIR="${FLYWHEEL_SCRATCHPAD_DIR:-/scratch}"
SCRATCHPAD_RUNTIME_DIR="$FLYWHEEL_SCRATCHPAD_DIR"
SCRATCHPAD_STATE_DIR=/flywheel/state/scratchpad
PERSISTED_SESSIONS=/flywheel/state/codex_sessions
PERSISTED_SESSION_ID=/flywheel/state/session_id
PERSISTED_HANDOFF=/flywheel/state/handoff_pending.json
RUNNER_LOG=/tmp/flywheel-codex-runner.jsonl
RUNTIME_TELEMETRY_DIR=/tmp/flywheel-codex-telemetry

if [ -z "$SCRATCHPAD_RUNTIME_DIR" ] \
    || [ "$SCRATCHPAD_RUNTIME_DIR" = "/" ]; then
    echo "[entrypoint] refusing unsafe FLYWHEEL_SCRATCHPAD_DIR: $SCRATCHPAD_RUNTIME_DIR" >&2
    exit 1
fi
case "$SCRATCHPAD_RUNTIME_DIR" in
    /scratch|/scratch/*) ;;
    *)
        echo "[entrypoint] FLYWHEEL_SCRATCHPAD_DIR must be under /scratch: $SCRATCHPAD_RUNTIME_DIR" >&2
        exit 1
        ;;
esac

mkdir -p "$CODEX_HOME"
# Keep credentials but remove stale config and unrelated local session history.
find "$CODEX_HOME" -mindepth 1 -maxdepth 1 \
    ! -name 'auth.json' \
    ! -name 'config.toml' \
    -exec rm -rf {} +
rm -f "$CODEX_HOME/config.toml"

if [ -d "$PERSISTED_SESSIONS" ]; then
    mkdir -p "$CODEX_HOME/sessions"
    cp -a "$PERSISTED_SESSIONS"/. "$CODEX_HOME/sessions"/
    echo "[entrypoint] staged Codex sessions"
fi
chown -R codex:codex "$CODEX_HOME"
phase codex_home_stage_done

mkdir -p "$SCRATCHPAD_RUNTIME_DIR"
find "$SCRATCHPAD_RUNTIME_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
if [ -d "$SCRATCHPAD_STATE_DIR" ]; then
    cp -a "$SCRATCHPAD_STATE_DIR"/. "$SCRATCHPAD_RUNTIME_DIR"/
    echo "[entrypoint] staged scratchpad from managed state"
else
    echo "[entrypoint] initialized empty scratchpad"
fi
chown -R codex:codex "$SCRATCHPAD_RUNTIME_DIR"
chmod 700 "$SCRATCHPAD_RUNTIME_DIR" 2>/dev/null || true
phase scratchpad_stage_done

mkdir -p /flywheel/state /flywheel/mcp_servers /flywheel/telemetry
chown -R root:root /flywheel/state /flywheel/telemetry
chmod 700 /flywheel/state /flywheel/telemetry

rm -rf "$RUNTIME_TELEMETRY_DIR"
mkdir -p "$RUNTIME_TELEMETRY_DIR"
chown -R codex:codex "$RUNTIME_TELEMETRY_DIR"

set +e
: > "$RUNNER_LOG"
chown root:root "$RUNNER_LOG"
chmod 600 "$RUNNER_LOG"
set -o pipefail
phase agent_process_start
su -s /bin/bash codex -c "FLYWHEEL_TELEMETRY_DIR='$RUNTIME_TELEMETRY_DIR' python3 /app/agent_runner.py" | tee "$RUNNER_LOG"
RC=${PIPESTATUS[0]}
phase agent_process_done
set +o pipefail
set -e

if [ -d "$RUNTIME_TELEMETRY_DIR" ]; then
    cp -a "$RUNTIME_TELEMETRY_DIR"/. /flywheel/telemetry/
    chown -R root:root /flywheel/telemetry
    chmod -R go-rwx /flywheel/telemetry
fi
phase agent_telemetry_copy_done

rm -rf "$SCRATCHPAD_STATE_DIR"
mkdir -p "$SCRATCHPAD_STATE_DIR"
if [ -d "$SCRATCHPAD_RUNTIME_DIR" ]; then
    cp -a "$SCRATCHPAD_RUNTIME_DIR"/. "$SCRATCHPAD_STATE_DIR"/
fi
phase scratchpad_checkpoint_done

rm -rf "$PERSISTED_SESSIONS"
if [ -d "$CODEX_HOME/sessions" ]; then
    mkdir -p "$PERSISTED_SESSIONS"
    cp -a "$CODEX_HOME/sessions"/. "$PERSISTED_SESSIONS"/
fi
phase sessions_checkpoint_done

python3 - <<'PY'
import json
from pathlib import Path

log_path = Path("/tmp/flywheel-codex-runner.jsonl")
session_path = Path("/flywheel/state/session_id")
pending_path = Path("/flywheel/state/handoff_pending.json")
latest_session = ""
latest_pending = None
try:
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue
        if event.get("type") == "agent_state" and event.get("session_id"):
            latest_session = str(event["session_id"])
        if event.get("type") == "handoff_pending":
            pending = event.get("pending")
            if isinstance(pending, list) and pending:
                first = pending[0]
                if isinstance(first, dict):
                    latest_pending = first
except Exception:
    pass

if latest_session:
    session_path.write_text(latest_session + "\n", encoding="utf-8")
if latest_pending is None:
    try:
        pending_path.unlink()
    except FileNotFoundError:
        pass
else:
    pending_path.write_text(
        json.dumps(latest_pending, indent=2, sort_keys=True),
        encoding="utf-8",
    )
PY
phase session_metadata_done

chown -R root:root /flywheel/state /flywheel/telemetry
chmod -R go-rwx /flywheel/state /flywheel/telemetry 2>/dev/null || true
phase permissions_done

if [ "$RC" -eq 0 ]; then
    REASON=$(python3 - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/flywheel-codex-runner.jsonl")
status = None
try:
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue
        if event.get("type") == "agent_state":
            status = event.get("status")
except Exception:
    pass
print("normal" if status in (None, "", "complete") else status)
PY
)
    echo "$REASON" > /flywheel/termination
fi
phase termination_done

python3 - "$PHASE_LOG" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
marks = []
try:
    for line in path.read_text(encoding="utf-8").splitlines():
        name, raw = line.split("\t", 1)
        marks.append((name, float(raw)))
except Exception:
    marks = []

durations = {}
for (name, at), (next_name, next_at) in zip(marks, marks[1:]):
    durations[f"{name}_to_{next_name}_s"] = round(next_at - at, 6)
if marks:
    durations["entrypoint_total_s"] = round(marks[-1][1] - marks[0][1], 6)

telemetry = {
    "kind": "codex_entrypoint_phases",
    "source": "flywheel-codex",
    "data": {
        "schema_version": 1,
        "marks": [name for name, _ in marks],
        "phase_timings_s": durations,
    },
}
Path("/flywheel/telemetry/codex_entrypoint_phases.json").write_text(
    json.dumps(telemetry, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY

exit "$RC"
