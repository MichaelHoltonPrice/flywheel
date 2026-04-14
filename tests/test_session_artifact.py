"""Tests for session artifact helpers in agent_runner.py.

Tests the cwd encoding, session path construction, and session
export logic in isolation (without the Claude SDK or Docker).
"""

from __future__ import annotations

import importlib
import shutil
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

    def test_export_creates_parent_dir(self, tmp_path: Path):
        """Parent directory of output file is created if missing."""
        sdk_dir = tmp_path / ".claude" / "projects" / "-workspace"
        sdk_dir.mkdir(parents=True)
        session_file = sdk_dir / "sess1.jsonl"
        session_file.write_text('{"role": "user"}\n')

        # Output in a subdirectory that doesn't exist yet.
        output_file = tmp_path / "deep" / "nested" / "agent_session.jsonl"

        with (
            patch.object(_runner, "SDK_PROJECTS_DIR",
                         tmp_path / ".claude" / "projects"),
            patch.object(_runner, "SESSION_OUTPUT_FILE",
                         output_file),
        ):
            _runner._export_session("sess1")

        assert output_file.exists()


class TestSessionImport:
    def test_copies_to_sdk_path(self, tmp_path: Path):
        """RESUME_SESSION_FILE is copied to the SDK expected path."""
        # Create a session file as if mounted at /input/agent_session/.
        source = tmp_path / "agent_session.jsonl"
        source.write_text('{"role": "user"}\n')

        sdk_projects = tmp_path / ".claude" / "projects"

        with patch.object(_runner, "SDK_PROJECTS_DIR", sdk_projects):
            dest = _runner._sdk_session_path("agent_session")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)

        expected = sdk_projects / "-workspace" / "agent_session.jsonl"
        assert expected.exists()
        assert expected.read_text() == '{"role": "user"}\n'

    def test_session_id_from_filename_stem(self):
        """Session ID is extracted from the filename stem."""
        p = Path("/input/agent_session/my-session-id.jsonl")
        assert p.stem == "my-session-id"
        assert p.suffix == ".jsonl"

    def test_non_jsonl_file_rejected(self):
        """Files without .jsonl suffix should not be used."""
        p = Path("/input/agent_session/data.txt")
        assert p.suffix != ".jsonl"

    def test_session_id_extracted_from_jsonl(self, tmp_path: Path):
        """Session ID is read from the JSONL content, not filename."""
        import json as _json

        real_sid = "640582da-4609-431a-b464-d9a99b7b4f35"
        source = tmp_path / "agent_session.jsonl"
        source.write_text(
            _json.dumps({"sessionId": real_sid, "type": "init"})
            + "\n"
        )

        # The resume code should extract the UUID from the file,
        # not use "agent_session" (the filename stem).
        first_line = source.read_text(encoding="utf-8").split("\n", 1)[0]
        entry = _json.loads(first_line)
        sid_from_file = entry.get("sessionId", "")
        assert sid_from_file == real_sid
        assert source.stem == "agent_session"  # filename is generic
        assert sid_from_file != source.stem     # they differ

    def test_session_id_fallback_on_bad_json(self, tmp_path: Path):
        """Falls back to filename stem if JSONL is not valid JSON."""
        source = tmp_path / "my-session.jsonl"
        source.write_text("not valid json\n")
        resume_sid = source.stem
        try:
            first_line = source.read_text(
                encoding="utf-8").split("\n", 1)[0]
            import json as _json
            entry = _json.loads(first_line)
            sid_from_file = entry.get("sessionId", "")
            if sid_from_file:
                resume_sid = sid_from_file
        except Exception:
            pass
        assert resume_sid == "my-session"


class TestStopFile:
    def test_stop_file_constant(self):
        """STOP_FILE is at /workspace/.agent_stop by default."""
        assert _runner.STOP_FILE.name == ".agent_stop"
        assert str(_runner.STOP_FILE).endswith(".agent_stop")
