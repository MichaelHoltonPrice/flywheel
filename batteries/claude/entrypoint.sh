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

# Drop to claude user and run agent.
exec su -s /bin/bash claude -c "python3 /app/agent_runner.py"
