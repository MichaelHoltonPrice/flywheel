#!/bin/bash
# Entrypoint for the flywheel-claude container.
#
# When NETWORK_ISOLATION=1, configures iptables to allow only:
#   - Anthropic API (api.anthropic.com, port 443)
#   - Host machine (host.docker.internal, any port)
#   - DNS (port 53, for API IP rotation)
#   - Loopback (MCP servers, internal comms)
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

    # Allow host machine (game server + project MCP servers).
    if [ -n "$HOST_IP" ]; then
        iptables -A OUTPUT -d "$HOST_IP" -j ACCEPT
    fi

    # Explicit final drop.
    iptables -A OUTPUT -j DROP

    echo "[entrypoint] network isolation active"
fi

# Drop to claude user and run agent.
exec su -s /bin/bash claude -c "python3 /app/agent_runner.py"
