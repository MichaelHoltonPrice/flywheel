"""Tests for session artifact helpers in agent_runner.py.

Tests the cwd encoding, session path construction, and session
export logic in isolation (without the Claude SDK or Docker).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch


def _import_helpers():
    """Import session helpers from agent_runner without SDK deps.

    The agent_runner module imports claude_agent_sdk at the top level.
    We mock that import so we can test the pure-Python helpers.
    """
    sdk_mock = type(sys)("claude_agent_sdk")
    sdk_mock.ClaudeAgentOptions = object
    sdk_mock.ClaudeSDKClient = object
    types_mock = type(sys)("claude_agent_sdk.types")
    types_mock.AssistantMessage = object
    types_mock.ResultMessage = object

    with patch.dict(sys.modules, {
        "claude_agent_sdk": sdk_mock,
        "claude_agent_sdk.types": types_mock,
        "anyio": type(sys)("anyio"),
        "mcp": type(sys)("mcp"),
        "mcp.server": type(sys)("mcp.server"),
        "mcp.server.fastmcp": type(sys)("mcp.server.fastmcp"),
    }):
        # Force reimport to pick up mocks.
        mod_name = "batteries.claude.agent_runner"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        # Add batteries to path if needed.
        batteries_path = str(
            Path(__file__).parent.parent / "batteries" / "claude")
        parent_path = str(Path(__file__).parent.parent / "batteries")
        root_path = str(Path(__file__).parent.parent)
        for p in [batteries_path, parent_path, root_path]:
            if p not in sys.path:
                sys.path.insert(0, p)

        spec = importlib.util.spec_from_file_location(
            "agent_runner",
            Path(__file__).parent.parent
            / "batteries" / "claude" / "agent_runner.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_runner = _import_helpers()


class TestEncodeCwd:
    def test_workspace(self):
        assert _runner._encode_cwd("/workspace") == "-workspace"

    def test_home_path(self):
        assert _runner._encode_cwd("/home/user/project") == (
            "-home-user-project"
        )

    def test_alphanumeric_preserved(self):
        assert _runner._encode_cwd("abc123") == "abc123"

    def test_dots_and_underscores(self):
        assert _runner._encode_cwd("/my.project_v2") == (
            "-my-project-v2"
        )


class TestSdkSessionPath:
    def test_default_cwd(self):
        result = _runner._sdk_session_path("abc123")
        assert result.name == "abc123.jsonl"
        assert "-workspace" in str(result)

    def test_custom_cwd(self):
        result = _runner._sdk_session_path("sess1", "/home/user/proj")
        assert "-home-user-proj" in str(result)
        assert result.name == "sess1.jsonl"


class TestExportSession:
    def test_copies_session_file(self, tmp_path: Path):
        """Session JSONL is copied to workspace output file."""
        # Create a fake SDK session file.
        sdk_dir = tmp_path / ".claude" / "projects" / "-workspace"
        sdk_dir.mkdir(parents=True)
        session_file = sdk_dir / "test-session.jsonl"
        session_file.write_text('{"role": "user"}\n')

        output_file = tmp_path / "agent_session.jsonl"

        with (
            patch.object(_runner, "SDK_PROJECTS_DIR",
                         tmp_path / ".claude" / "projects"),
            patch.object(_runner, "SESSION_OUTPUT_FILE",
                         output_file),
        ):
            _runner._export_session("test-session")

        assert output_file.exists()
        assert output_file.read_text() == '{"role": "user"}\n'

    def test_empty_session_id_is_noop(self, tmp_path: Path):
        output_file = tmp_path / "agent_session.jsonl"
        with patch.object(_runner, "SESSION_OUTPUT_FILE",
                          output_file):
            _runner._export_session("")
        assert not output_file.exists()

    def test_missing_source_is_safe(self, tmp_path: Path):
        """No error when SDK session file doesn't exist."""
        output_file = tmp_path / "agent_session.jsonl"
        with (
            patch.object(_runner, "SDK_PROJECTS_DIR",
                         tmp_path / "nonexistent"),
            patch.object(_runner, "SESSION_OUTPUT_FILE",
                         output_file),
        ):
            _runner._export_session("missing-session")
        assert not output_file.exists()
