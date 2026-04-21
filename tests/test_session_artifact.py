"""Tests for session-related helpers in agent_runner.py.

Tests the cwd encoding and session path construction in
isolation (without the Claude SDK or Docker).  The
runner-side persistence flow (read/write of
``/flywheel/state/session.jsonl``) is owned by ``entrypoint.sh``
under the privilege-split model — ``agent_runner`` no longer
touches that path, so there is nothing to unit-test on the
Python side.
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
    sdk_mock.HookMatcher = object
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
    def test_scratch(self):
        assert _runner._encode_cwd("/scratch") == "-scratch"

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
        assert "-scratch" in str(result)

    def test_custom_cwd(self):
        result = _runner._sdk_session_path("sess1", "/home/user/proj")
        assert "-home-user-proj" in str(result)
        assert result.name == "sess1.jsonl"


class TestStopFile:
    def test_stop_file_matches_substrate_sentinel(self):
        """STOP_FILE is the contract's ``/scratch/.stop`` sentinel.

        The agent runner watches the same path the substrate
        writes for cooperative cancellation, so
        :meth:`ContainerExecutionHandle.stop` and a handoff-loop
        stop both terminate the agent through one channel.
        """
        assert _runner.STOP_FILE.name == ".stop"
        assert str(_runner.STOP_FILE).endswith(".stop")


class TestSessionPersistenceIsExternal:
    """The runner module no longer reaches /flywheel/state/.

    Pass 2's privilege split moves session staging + sync into
    ``entrypoint.sh`` (root).  This is a guard test: if a future
    edit reintroduces a runner-side ``SESSION_FILE`` constant or
    an ``_export_session`` function, we want a loud failure
    rather than a silent regression of the privilege boundary.
    """

    def test_no_session_file_constant(self):
        assert not hasattr(_runner, "SESSION_FILE")

    def test_no_export_session_function(self):
        assert not hasattr(_runner, "_export_session")
