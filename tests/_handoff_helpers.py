"""Shared scaffolding for ``run_agent_with_handoffs`` test files.

The handoff loop's collaborators (``launch_agent_block`` and the
agent container behind it) are heavy and Docker-dependent, so the
test files exercise the loop's contract by injecting a fake
``launch_fn`` that produces programmable :class:`_FakeAgentHandle`
objects.  Each fake handle decides what
:class:`flywheel.agent.AgentResult.exit_reason` to report and what
files to leave on disk in the agent workspace.

The :mod:`flywheel.session_splice` module is intentionally *not*
mocked here — the loop performs real splices on the synthetic
JSONL fixtures these helpers build, so a regression in either the
loop or the splice gets caught.

Files using these helpers:

- ``tests/test_agent_handoff.py`` exercises the happy paths,
  contract violations, and result-accessor surface.
- ``tests/test_handoff_resume.py`` exercises crash-resume
  semantics across the boundary's failure modes.

The leading underscore on the module name keeps pytest from
collecting it as a test file even though it lives under
``tests/``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from flywheel.agent import AgentResult
from flywheel.agent_handoff import (
    PENDING_FILE_NAME,
    SESSION_ARTIFACT_NAME,
)


@dataclass
class _FakeWorkspace:
    """Stand-in for :class:`flywheel.workspace.Workspace`.

    The handoff loop only reads ``workspace.path`` to resolve the
    agent workspace dir relative to it.  Anything else stays
    untouched.
    """

    path: Path


@dataclass
class _CycleScript:
    """How the next fake launch should behave.

    Attributes:
        exit_reason: What the launch's :class:`AgentResult` will
            report.  ``"tool_handoff"`` triggers another loop
            iteration; anything else terminates.
        pending: List of pending tool-call dicts to drop into
            ``pending_tool_calls.json`` after launch.  Ignored
            unless ``exit_reason == "tool_handoff"``.
        write_session: Whether to (re)write the session JSONL
            artifact from ``session_jsonl_text`` after launch.
            Tests set this to False to simulate a missing
            artifact and verify the loop's error path.
        session_jsonl_text: The JSONL contents to drop on disk
            (overwriting any prior splice).  Tests usually set
            this only on iteration 0; later iterations leave
            ``write_session=False`` so the spliced JSONL from
            the previous cycle remains in place.
        execution_id: Workspace-recorded ID for this launch.
            Mirrored into ``predecessor_id`` for the next
            relaunch so the chain is observable.
        agent_workspace_dir: Subpath to use; tests usually pin
            this to one value across cycles so the loop's
            relaunch logic reuses the same directory.
        pending_text: Optional override for the literal text
            written into ``pending_tool_calls.json``.  When
            provided, ``pending`` is ignored and this string is
            written verbatim, letting tests stage malformed JSON
            or alternative envelope shapes the loop's defensive
            decoder must tolerate.
    """

    exit_reason: str = "completed"
    pending: list[dict[str, Any]] = field(default_factory=list)
    write_session: bool = False
    session_jsonl_text: str = ""
    execution_id: str = "exec-0"
    agent_workspace_dir: str = "agent_workspaces/handoff_test"
    pending_text: str | None = None


@dataclass
class _LaunchCall:
    """Captures one ``launch_fn`` invocation for assertions.

    Attributes:
        kwargs: The exact kwargs the loop passed.  Tests assert
            on ``reuse_workspace``, ``predecessor_id``,
            ``agent_workspace_dir``, ``extra_env``, and
            ``prompt`` here.
    """

    kwargs: dict[str, Any]


class _FakeAgentHandle:
    """Stand-in for :class:`flywheel.agent.AgentHandle`.

    Mimics the bare ``wait()`` -> ``AgentResult`` interface the
    handoff loop relies on.  On ``wait()``, applies its script
    by writing the configured pending and session-JSONL files to
    the agent workspace, then returns the corresponding
    :class:`AgentResult`.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        script: _CycleScript,
    ) -> None:
        self._workspace_root = workspace_root
        self._script = script

    def wait(self) -> AgentResult:
        """Apply the cycle's script and return the AgentResult."""
        agent_ws = (
            self._workspace_root / self._script.agent_workspace_dir
        )
        agent_ws.mkdir(parents=True, exist_ok=True)

        if self._script.write_session:
            (agent_ws / SESSION_ARTIFACT_NAME).write_text(
                self._script.session_jsonl_text, encoding="utf-8")

        if self._script.exit_reason == "tool_handoff":
            if self._script.pending_text is not None:
                (agent_ws / PENDING_FILE_NAME).write_text(
                    self._script.pending_text, encoding="utf-8")
            else:
                envelope = {
                    "schema_version": 2,
                    "session_id": "sess-fake",
                    "pending": list(self._script.pending),
                }
                (agent_ws / PENDING_FILE_NAME).write_text(
                    json.dumps(envelope), encoding="utf-8")

        return AgentResult(
            exit_code=0,
            elapsed_s=0.1,
            evals_run=0,
            execution_id=self._script.execution_id,
            stop_reason=None,
            exit_reason=self._script.exit_reason,
            agent_workspace_dir=self._script.agent_workspace_dir,
        )


def _make_launch_fn(
    workspace_root: Path,
    scripts: list[_CycleScript],
    record: list[_LaunchCall],
) -> Callable[..., _FakeAgentHandle]:
    """Build a fake launch_fn that walks ``scripts`` cycle by cycle.

    Each call pops the next script and constructs a
    :class:`_FakeAgentHandle` honoring it.  ``record`` is appended
    to with the kwargs the loop used so the test can assert on the
    relaunch wiring (resume env var, predecessor id, reuse flag,
    prompt swap).  Calling more times than there are scripts is a
    test bug and surfaces as ``IndexError`` rather than silent
    passing.
    """
    iteration = {"i": 0}

    def _fake(**kwargs: Any) -> _FakeAgentHandle:
        record.append(_LaunchCall(kwargs=kwargs))
        idx = iteration["i"]
        iteration["i"] = idx + 1
        return _FakeAgentHandle(
            workspace_root=workspace_root,
            script=scripts[idx],
        )

    return _fake


def _build_session_jsonl(
    *,
    session_id: str,
    tool_use_ids: list[str],
    deny_marker: str = "handoff_to_flywheel",
) -> str:
    """Build a minimal SDK-shaped session JSONL.

    Mirrors the shape :mod:`flywheel.session_splice` expects:
    one ``summary`` line carrying ``sessionId``, then an assistant
    envelope with all ``tool_use_ids`` as parallel ``tool_use``
    blocks, then a user envelope with one deny ``tool_result`` per
    id.  Used by tests that exercise the real splice through the
    loop.
    """
    lines: list[dict[str, Any]] = [
        {"type": "summary", "sessionId": session_id, "version": 1},
    ]
    lines.append({
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tuid,
                    "name": "mcp__demo__handoff_me",
                    "input": {"k": tuid},
                }
                for tuid in tool_use_ids
            ],
        },
    })
    lines.append({
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tuid,
                    "content": [{
                        "type": "text",
                        "text": f"permission denied: {deny_marker}",
                    }],
                    "is_error": True,
                }
                for tuid in tool_use_ids
            ],
        },
    })
    return "\n".join(json.dumps(obj) for obj in lines) + "\n"


@pytest.fixture
def workspace(tmp_path: Path) -> _FakeWorkspace:
    """A fresh fake workspace rooted in ``tmp_path``."""
    return _FakeWorkspace(path=tmp_path)


def _base_kwargs(workspace: _FakeWorkspace) -> dict[str, Any]:
    """Minimal kwargs the loop forwards to ``launch_fn``.

    Mirrors what a real caller would pass to
    :func:`flywheel.agent.launch_agent_block`; the fake launch_fn
    ignores most of them but the loop must pass them through
    unchanged on cycle 0.
    """
    return {
        "workspace": workspace,
        "template": object(),
        "project_root": workspace.path,
        "prompt": "do the thing",
        "agent_workspace_dir": "agent_workspaces/handoff_test",
    }
