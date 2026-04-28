#!/bin/bash
# Entrypoint for flywheel-desktop.
#
# Starts a virtual X display, a minimal window manager, the optional
# project GUI command, and a Python HTTP server. The server exposes:
#   * Flywheel persistent-runtime control endpoints on FLYWHEEL_CONTROL_PORT
#   * desktop screenshot/input/file endpoints on DESKTOP_API_PORT
set -euo pipefail

export DISPLAY="${DESKTOP_DISPLAY:-:99}"
WIDTH="${DESKTOP_WIDTH:-1280}"
HEIGHT="${DESKTOP_HEIGHT:-720}"
DPI="${DESKTOP_DPI:-96}"

mkdir -p "${DESKTOP_SHARED_DIR:-/desktop/shared}" /tmp/flywheel-desktop

Xvfb "$DISPLAY" -screen 0 "${WIDTH}x${HEIGHT}x24" -dpi "$DPI" \
    >/tmp/flywheel-desktop/xvfb.log 2>&1 &
XVFB_PID=$!

cleanup() {
    if [ -n "${APP_PID:-}" ]; then
        kill "$APP_PID" >/dev/null 2>&1 || true
    fi
    kill "$XVFB_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 50); do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    echo "[desktop] Xvfb did not become ready on $DISPLAY" >&2
    exit 1
fi

openbox >/tmp/flywheel-desktop/openbox.log 2>&1 || true &

if [ -n "${DESKTOP_APP_COMMAND:-}" ]; then
    sh -c "exec $DESKTOP_APP_COMMAND" >/tmp/flywheel-desktop/app.log 2>&1 &
    APP_PID=$!
    export DESKTOP_APP_PID="$APP_PID"
fi

exec python3 /app/desktop_server.py
