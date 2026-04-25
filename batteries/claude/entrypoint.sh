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
chown -R claude:claude "$PROJECTS_DIR"

# Lock the persisted-state directory so claude cannot read it.
chmod 700 /flywheel/state
chown -R root:root /flywheel/state

# /flywheel/control is a write-only handoff drop for the agent
# runner; claude needs to write JSON files here, the launcher
# reads them after container exit.
mkdir -p /flywheel/control /flywheel/mcp_servers
chown -R claude:claude /flywheel/control
chmod 700 /flywheel/control

# /flywheel/mcp_servers is read-only project code already
# chowned to claude in the Dockerfile.  Nothing to do.

echo "[entrypoint] /flywheel/state locked to root; running agent as claude"

# Run the agent as claude WITHOUT exec so we resume here when
# it exits and can sync the SDK's updated session back to
# root-owned state.  ``set +e`` so we capture the rc instead of
# crashing on a non-zero agent exit.
set +e
su -s /bin/bash claude -c "python3 /app/agent_runner.py"
RC=$?

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

if [ "$RC" -eq 0 ]; then
    REASON=$(python3 - <<'PY'
import json
from pathlib import Path

path = Path("/flywheel/control/agent_exit_state.json")
try:
    status = json.loads(path.read_text(encoding="utf-8")).get("status")
except Exception:
    status = None
print("normal" if status in (None, "", "complete") else status)
PY
)
    echo "$REASON" > /flywheel/termination
fi

exit "$RC"
