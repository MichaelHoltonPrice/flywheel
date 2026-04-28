#!/usr/bin/env python3
# ruff: noqa: D101,D102,D103,D107
"""Generic desktop-service HTTP server.

The server intentionally contains no agent or project-specific logic. It
serves a Flywheel persistent-runtime control surface plus a generic desktop
surface that controllers can drive over a project Docker network.
"""

from __future__ import annotations

import json
import mimetypes
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

try:
    import mss
except Exception:  # pragma: no cover - exercised only in a broken image
    mss = None


PROTOCOL_VERSION = "1"
VERSION = "1"


class DesktopConfig:
    """Runtime configuration shared by request handlers."""

    def __init__(self) -> None:
        self.width = _env_int("DESKTOP_WIDTH", 1280)
        self.height = _env_int("DESKTOP_HEIGHT", 720)
        self.dpi = _env_int("DESKTOP_DPI", 96)
        self.display = os.environ.get("DISPLAY") or os.environ.get(
            "DESKTOP_DISPLAY", ":99")
        self.shared_dir = Path(os.environ.get(
            "DESKTOP_SHARED_DIR", "/desktop/shared")).resolve()
        self.app_command = os.environ.get("DESKTOP_APP_COMMAND", "")
        self.app_pid = _env_int_optional("DESKTOP_APP_PID")
        self.control_port = _env_int("FLYWHEEL_CONTROL_PORT", 8099)
        self.desktop_port = _env_int("DESKTOP_API_PORT", 8080)
        self.lock = threading.Lock()
        self.shared_dir.mkdir(parents=True, exist_ok=True)

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_int_optional(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


CONFIG = DesktopConfig()


def _read_json(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    return data


def _write_json(handler: BaseHTTPRequestHandler, status: int, data: Any) -> None:
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _write_empty(handler: BaseHTTPRequestHandler, status: int = 204) -> None:
    handler.send_response(status)
    handler.send_header("Content-Length", "0")
    handler.end_headers()


def _write_error(
    handler: BaseHTTPRequestHandler,
    status: int,
    error: str,
    detail: str = "",
) -> None:
    _write_json(handler, status, {"error": error, "detail": detail})


def _coerce_coordinate(value: Any, name: str, *, upper_bound: int) -> int:
    """Return a window-relative integer coordinate or raise ValueError."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0 or parsed >= upper_bound:
        raise ValueError(f"{name} must be in [0, {upper_bound - 1}]")
    return parsed


def _coerce_duration_ms(value: Any, default: int = 100) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("duration_ms must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_ms must be an integer") from exc
    if parsed < 0 or parsed > 60_000:
        raise ValueError("duration_ms must be in [0, 60000]")
    return parsed


def _app_status() -> dict[str, Any]:
    pid = CONFIG.app_pid
    if pid is None:
        return {"pid": None, "status": "not_configured"}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {"pid": pid, "status": "stopped"}
    except PermissionError:
        return {"pid": pid, "status": "unknown"}
    return {"pid": pid, "status": "running"}


def health_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "display": {
            "width": CONFIG.width,
            "height": CONFIG.height,
            "dpi": CONFIG.dpi,
            "display": CONFIG.display,
        },
        "app": _app_status(),
    }


def _xdotool(args: list[str]) -> None:
    env = os.environ.copy()
    env["DISPLAY"] = CONFIG.display
    subprocess.run(["xdotool", *args], check=True, env=env)


def click(payload: dict[str, Any]) -> None:
    x = _coerce_coordinate(payload.get("x"), "x", upper_bound=CONFIG.width)
    y = _coerce_coordinate(payload.get("y"), "y", upper_bound=CONFIG.height)
    button_name = str(payload.get("button", "left")).lower()
    buttons = {"left": "1", "middle": "2", "right": "3"}
    if button_name not in buttons:
        raise ValueError("button must be left, middle, or right")
    _xdotool(["mousemove", str(x), str(y)])
    repeat = "2" if bool(payload.get("double", False)) else "1"
    _xdotool(["click", "--repeat", repeat, buttons[button_name]])


def move(payload: dict[str, Any]) -> None:
    x = _coerce_coordinate(payload.get("x"), "x", upper_bound=CONFIG.width)
    y = _coerce_coordinate(payload.get("y"), "y", upper_bound=CONFIG.height)
    _xdotool(["mousemove", str(x), str(y)])


def type_text(payload: dict[str, Any]) -> None:
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    _xdotool(["type", "--clearmodifiers", "--", text])


def key(payload: dict[str, Any]) -> None:
    key_name = payload.get("key")
    if not isinstance(key_name, str) or not key_name.strip():
        raise ValueError("key must be a non-empty string")
    modifiers = payload.get("modifiers", [])
    if modifiers is None:
        modifiers = []
    if not isinstance(modifiers, list) or not all(
            isinstance(item, str) for item in modifiers):
        raise ValueError("modifiers must be a list of strings")
    combo = "+".join([*modifiers, key_name])
    _xdotool(["key", "--clearmodifiers", combo])


def drag(payload: dict[str, Any]) -> None:
    start = payload.get("from")
    end = payload.get("to")
    if not isinstance(start, list) or len(start) != 2:
        raise ValueError("from must be [x, y]")
    if not isinstance(end, list) or len(end) != 2:
        raise ValueError("to must be [x, y]")
    x1 = _coerce_coordinate(start[0], "from[0]", upper_bound=CONFIG.width)
    y1 = _coerce_coordinate(start[1], "from[1]", upper_bound=CONFIG.height)
    x2 = _coerce_coordinate(end[0], "to[0]", upper_bound=CONFIG.width)
    y2 = _coerce_coordinate(end[1], "to[1]", upper_bound=CONFIG.height)
    duration_ms = _coerce_duration_ms(payload.get("duration_ms"), default=250)
    _xdotool(["mousemove", str(x1), str(y1)])
    _xdotool(["mousedown", "1"])
    time.sleep(duration_ms / 1000.0)
    _xdotool(["mousemove", str(x2), str(y2)])
    _xdotool(["mouseup", "1"])


def wait(payload: dict[str, Any]) -> None:
    duration_ms = _coerce_duration_ms(payload.get("duration_ms"), default=100)
    time.sleep(duration_ms / 1000.0)


def screenshot_bytes(fmt: str, quality: int) -> tuple[bytes, str]:
    if mss is None:
        raise RuntimeError("mss is not available")
    normalized = fmt.lower()
    if normalized not in {"png", "jpeg", "jpg"}:
        raise ValueError("format must be png or jpeg")
    with mss.mss() as grabber:
        monitor = {
            "top": 0,
            "left": 0,
            "width": CONFIG.width,
            "height": CONFIG.height,
        }
        raw = grabber.grab(monitor)
        image = Image.frombytes("RGB", raw.size, raw.rgb)
    output = BytesIO()
    if normalized == "png":
        image.save(output, format="PNG")
        return output.getvalue(), "image/png"
    q = min(max(int(quality), 1), 100)
    image.save(output, format="JPEG", quality=q)
    return output.getvalue(), "image/jpeg"


def _safe_shared_path(relative_url_path: str) -> Path:
    rel = unquote(relative_url_path).lstrip("/")
    if not rel:
        raise ValueError("file path is required")
    path = (CONFIG.shared_dir / rel).resolve()
    if path != CONFIG.shared_dir and CONFIG.shared_dir not in path.parents:
        raise ValueError("file path escapes shared directory")
    return path


def _wait_for_app_running(*, timeout_s: float, render_delay_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _app_status().get("status") == "running":
            if render_delay_s > 0:
                time.sleep(render_delay_s)
            return
        time.sleep(0.1)
    raise RuntimeError("desktop app did not become running")


def _terminate_app() -> None:
    if CONFIG.app_pid is None:
        return
    try:
        os.kill(CONFIG.app_pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(CONFIG.app_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)


def _start_app() -> None:
    env = os.environ.copy()
    env["DISPLAY"] = CONFIG.display
    log_path = Path("/tmp/flywheel-desktop/app.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    try:
        proc = subprocess.Popen(
            CONFIG.app_command,
            shell=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
    finally:
        log_handle.close()
    CONFIG.app_pid = proc.pid


def _restart_app(
    *,
    wait_for_running: bool = False,
    timeout_s: float = 30.0,
    render_delay_s: float = 1.0,
) -> None:
    if not CONFIG.app_command:
        raise RuntimeError("DESKTOP_APP_COMMAND is not configured")
    with CONFIG.lock:
        _terminate_app()
        _start_app()
        if wait_for_running:
            _wait_for_app_running(
                timeout_s=timeout_s,
                render_delay_s=render_delay_s,
            )


class Handler(BaseHTTPRequestHandler):
    server_version = "FlywheelDesktop/1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                _write_json(self, 200, health_payload())
                return
            if parsed.path == "/screenshot":
                query = parse_qs(parsed.query)
                fmt = query.get("format", ["jpeg"])[0]
                quality_text = query.get("quality", ["80"])[0]
                try:
                    quality = int(quality_text)
                except ValueError as exc:
                    raise ValueError("quality must be an integer") from exc
                data, content_type = screenshot_bytes(fmt, quality)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path.startswith("/files/"):
                file_path = _safe_shared_path(parsed.path.removeprefix("/files/"))
                if not file_path.is_file():
                    _write_error(self, 404, "not_found", str(file_path))
                    return
                data = file_path.read_bytes()
                content_type = (
                    mimetypes.guess_type(file_path.name)[0]
                    or "application/octet-stream"
                )
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        except ValueError as exc:
            _write_error(self, 400, "bad_request", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            _write_error(self, 500, type(exc).__name__, str(exc))
            return
        _write_error(self, 404, "not_found", parsed.path)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/files/"):
                file_path = _safe_shared_path(parsed.path.removeprefix("/files/"))
                if file_path.exists():
                    file_path.unlink()
                _write_empty(self)
                return
        except Exception as exc:  # noqa: BLE001
            _write_error(self, 500, type(exc).__name__, str(exc))
            return
        _write_error(self, 404, "not_found", parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            if parsed.path == "/execute":
                if self.server.server_port != CONFIG.control_port:
                    _write_error(self, 404, "not_found", parsed.path)
                    return
                payload = _read_json(raw)
                termination_path = payload.get("termination_path")
                if isinstance(termination_path, str) and termination_path:
                    Path(termination_path).write_text("normal\n", encoding="utf-8")
                _write_json(self, 200, {
                    "status": "succeeded",
                    "termination_reason": "normal",
                })
                return
            if parsed.path == "/click":
                click(_read_json(raw))
                _write_empty(self)
                return
            if parsed.path == "/move":
                move(_read_json(raw))
                _write_empty(self)
                return
            if parsed.path == "/type":
                type_text(_read_json(raw))
                _write_empty(self)
                return
            if parsed.path == "/key":
                key(_read_json(raw))
                _write_empty(self)
                return
            if parsed.path == "/drag":
                drag(_read_json(raw))
                _write_empty(self)
                return
            if parsed.path == "/wait":
                wait(_read_json(raw))
                _write_empty(self)
                return
            if parsed.path == "/reset":
                payload = _read_json(raw)
                timeout_s = float(payload.get("timeout_s", 30.0))
                render_delay_s = (
                    _coerce_duration_ms(
                        payload.get("render_delay_ms"),
                        default=1000,
                    ) / 1000.0
                )
                _restart_app(
                    wait_for_running=bool(payload.get("wait_for_running")),
                    timeout_s=timeout_s,
                    render_delay_s=render_delay_s,
                )
                _write_json(self, 200, health_payload())
                return
            if parsed.path.startswith("/files/"):
                file_path = _safe_shared_path(parsed.path.removeprefix("/files/"))
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(raw)
                _write_empty(self, 201)
                return
        except ValueError as exc:
            _write_error(self, 400, "bad_request", str(exc))
            return
        except subprocess.CalledProcessError as exc:
            _write_error(self, 502, "input_command_failed", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            _write_error(self, 500, type(exc).__name__, str(exc))
            return
        _write_error(self, 404, "not_found", parsed.path)


def _serve(port: int) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    SERVERS.append(server)
    server.serve_forever()


SERVERS: list[ThreadingHTTPServer] = []


def _shutdown_servers(*_args: Any) -> None:
    for server in list(SERVERS):
        threading.Thread(target=server.shutdown, daemon=True).start()


def main() -> None:
    signal.signal(signal.SIGTERM, _shutdown_servers)
    signal.signal(signal.SIGINT, _shutdown_servers)
    control_port = CONFIG.control_port
    desktop_port = CONFIG.desktop_port
    ports = [control_port]
    if desktop_port != control_port:
        ports.append(desktop_port)
    threads = []
    for port in ports:
        thread = threading.Thread(target=_serve, args=(port,), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
