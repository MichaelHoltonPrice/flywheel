"""Tests for dynamic MCP server discovery in the agent runner.

Tests _scan_mounted_servers() which scans a directory for
*_mcp_server.py files and optional .json sidecar manifests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from batteries.claude.agent_runner import _scan_mounted_servers


class TestScanMountedServers:
    def test_discovers_python_file(self, tmp_path: Path):
        (tmp_path / "arc_mcp_server.py").write_text("# server")

        with patch.dict(
            "os.environ",
            {"MCP_SERVER_MOUNT_DIR": str(tmp_path)},
        ):
            servers = _scan_mounted_servers()

        assert "arc" in servers
        name, factory = servers["arc"]
        assert name == "arc"
        config, tools = factory()
        assert config["command"] == "python3"
        assert str(tmp_path / "arc_mcp_server.py") in config["args"]
        assert tools == []  # No sidecar manifest.

    def test_reads_sidecar_manifest(self, tmp_path: Path):
        (tmp_path / "arc_mcp_server.py").write_text("# server")
        (tmp_path / "arc_mcp_server.json").write_text(json.dumps({
            "name": "arc",
            "tools": ["mcp__arc__take_action", "mcp__arc__get_status"],
        }))

        with patch.dict(
            "os.environ",
            {"MCP_SERVER_MOUNT_DIR": str(tmp_path)},
        ):
            servers = _scan_mounted_servers()

        _, factory = servers["arc"]
        _, tools = factory()
        assert tools == ["mcp__arc__take_action", "mcp__arc__get_status"]

    def test_malformed_manifest_ignored(self, tmp_path: Path):
        (tmp_path / "game_mcp_server.py").write_text("# server")
        (tmp_path / "game_mcp_server.json").write_text("not json {{{")

        with patch.dict(
            "os.environ",
            {"MCP_SERVER_MOUNT_DIR": str(tmp_path)},
        ):
            servers = _scan_mounted_servers()

        assert "game" in servers
        _, factory = servers["game"]
        _, tools = factory()
        assert tools == []  # Graceful fallback.

    def test_multiple_servers(self, tmp_path: Path):
        (tmp_path / "arc_mcp_server.py").write_text("# arc")
        (tmp_path / "chess_mcp_server.py").write_text("# chess")

        with patch.dict(
            "os.environ",
            {"MCP_SERVER_MOUNT_DIR": str(tmp_path)},
        ):
            servers = _scan_mounted_servers()

        assert "arc" in servers
        assert "chess" in servers

    def test_empty_directory(self, tmp_path: Path):
        with patch.dict(
            "os.environ",
            {"MCP_SERVER_MOUNT_DIR": str(tmp_path)},
        ):
            servers = _scan_mounted_servers()

        assert servers == {}

    def test_missing_directory(self, tmp_path: Path):
        missing = tmp_path / "nonexistent"

        with patch.dict(
            "os.environ",
            {"MCP_SERVER_MOUNT_DIR": str(missing)},
        ):
            servers = _scan_mounted_servers()

        assert servers == {}

    def test_non_mcp_files_ignored(self, tmp_path: Path):
        (tmp_path / "helper.py").write_text("# not an MCP server")
        (tmp_path / "arc_mcp_server.py").write_text("# server")

        with patch.dict(
            "os.environ",
            {"MCP_SERVER_MOUNT_DIR": str(tmp_path)},
        ):
            servers = _scan_mounted_servers()

        assert list(servers.keys()) == ["arc"]

    def test_passes_all_env_vars(self, tmp_path: Path):
        (tmp_path / "test_mcp_server.py").write_text("# server")

        with patch.dict(
            "os.environ",
            {
                "MCP_SERVER_MOUNT_DIR": str(tmp_path),
                "GAME_ID": "vc33-test",
                "CUSTOM_VAR": "hello",
            },
        ):
            servers = _scan_mounted_servers()

        _, factory = servers["test"]
        config, _ = factory()
        assert config["env"]["GAME_ID"] == "vc33-test"
        assert config["env"]["CUSTOM_VAR"] == "hello"

    def test_default_mount_dir_when_env_unset(self):
        """When MCP_SERVER_MOUNT_DIR is not set, uses the default path."""
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("MCP_SERVER_MOUNT_DIR", None)
            servers = _scan_mounted_servers()

        # Default is /workspace/.mcp_servers which won't exist on
        # the host, so this should return empty — not crash.
        assert servers == {}
