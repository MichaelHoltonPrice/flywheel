from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DESKTOP_DIR = ROOT / "batteries" / "desktop"


def _load_desktop_server():
    spec = importlib.util.spec_from_file_location(
        "desktop_server", DESKTOP_DIR / "desktop_server.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DesktopServer:
    def __init__(self, module):
        self.module = module
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _request(method: str, url: str, payload: dict | bytes | None = None):
    data = None
    headers = {}
    if isinstance(payload, dict):
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif isinstance(payload, bytes):
        data = payload
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        raw = response.read()
        content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("application/json") and raw:
        return json.loads(raw.decode("utf-8"))
    return raw


def test_desktop_battery_is_agent_agnostic():
    dockerfile = (DESKTOP_DIR / "Dockerfile.desktop").read_text(
        encoding="utf-8")
    server = (DESKTOP_DIR / "desktop_server.py").read_text(encoding="utf-8")
    entrypoint = (DESKTOP_DIR / "entrypoint.sh").read_text(encoding="utf-8")

    combined = "\n".join([dockerfile, server, entrypoint]).lower()
    assert "claude" not in combined
    assert "anthropic" not in combined
    assert "cyberloop" not in combined
    assert "decker" not in combined


def test_desktop_server_has_no_flywheel_imports():
    server = (DESKTOP_DIR / "desktop_server.py").read_text(encoding="utf-8")

    assert "from flywheel" not in server
    assert "import flywheel" not in server


def test_desktop_battery_uses_separate_control_and_desktop_ports():
    dockerfile = (DESKTOP_DIR / "Dockerfile.desktop").read_text(
        encoding="utf-8")
    server = (DESKTOP_DIR / "desktop_server.py").read_text(encoding="utf-8")

    assert "DESKTOP_API_PORT=8080" in dockerfile
    assert "FLYWHEEL_CONTROL_PORT" in server
    assert "DESKTOP_API_PORT" in server
    assert "def _serve(port: int)" in server


def test_flywheel_core_has_no_cua_or_desktop_concepts():
    terms = ["cua", "screenshot", "click", "decker", "desktop service"]
    offenders: list[str] = []
    for path in (ROOT / "flywheel").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for term in terms:
            if term in text:
                offenders.append(f"{path.relative_to(ROOT)}:{term}")

    assert offenders == []


def test_coordinate_validation_rejects_out_of_bounds():
    desktop_server = _load_desktop_server()

    assert desktop_server._coerce_coordinate(0, "x", upper_bound=10) == 0
    assert desktop_server._coerce_coordinate("9", "x", upper_bound=10) == 9
    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        desktop_server._coerce_coordinate(10, "x", upper_bound=10)
    with pytest.raises(ValueError, match="integer"):
        desktop_server._coerce_coordinate(True, "x", upper_bound=10)


def test_shared_file_paths_cannot_escape_shared_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DESKTOP_SHARED_DIR", str(tmp_path))
    desktop_server = _load_desktop_server()

    inside = desktop_server._safe_shared_path("state/save.json")
    assert inside == tmp_path / "state" / "save.json"
    with pytest.raises(ValueError, match="escapes"):
        desktop_server._safe_shared_path("../outside.json")


def test_files_roundtrip_and_execute_is_control_port_only(
        tmp_path, monkeypatch):
    monkeypatch.setenv("DESKTOP_SHARED_DIR", str(tmp_path))
    desktop_server = _load_desktop_server()

    with _DesktopServer(desktop_server) as server:
        desktop_server.CONFIG.control_port = server.server.server_port
        result = _request("POST", f"{server.url}/files/state/save.bin", b"abc")
        assert result == b""
        assert (tmp_path / "state" / "save.bin").read_bytes() == b"abc"
        assert _request("GET", f"{server.url}/files/state/save.bin") == b"abc"

        termination = tmp_path / "termination"
        response = _request("POST", f"{server.url}/execute", {
            "termination_path": str(termination),
        })
        assert response["status"] == "succeeded"
        assert termination.read_text(encoding="utf-8") == "normal\n"

    with _DesktopServer(desktop_server) as server:
        desktop_server.CONFIG.control_port = server.server.server_port + 1
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _request("POST", f"{server.url}/execute", {
                "termination_path": str(tmp_path / "bad"),
            })
        assert excinfo.value.code == 404


def test_click_route_validates_and_calls_xdotool(tmp_path, monkeypatch):
    monkeypatch.setenv("DESKTOP_SHARED_DIR", str(tmp_path))
    desktop_server = _load_desktop_server()
    calls = []
    monkeypatch.setattr(desktop_server, "_xdotool", calls.append)

    with _DesktopServer(desktop_server) as server:
        _request("POST", f"{server.url}/click", {
            "x": 7,
            "y": 9,
            "button": "right",
        })
        assert calls == [["mousemove", "7", "9"], [
            "click", "--repeat", "1", "3"]]
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _request("POST", f"{server.url}/click", {
                "x": -1,
                "y": 9,
            })
        assert excinfo.value.code == 400
