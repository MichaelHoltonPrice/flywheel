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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from flywheel.agent import AgentResult
from flywheel.agent_handoff import (
    SESSION_FILE_NAME,
)
from flywheel.artifact import ArtifactInstance, BlockExecution


@dataclass
class _FakeWorkspace:
    """Stand-in for :class:`flywheel.workspace.Workspace`.

    The handoff loop reads ``workspace.path`` to resolve paths,
    ``workspace.executions`` to locate captured state_dirs, and
    ``workspace.artifacts`` to resolve the pending-tool-calls
    output binding back to its on-disk directory.
    """

    path: Path
    executions: dict[str, BlockExecution] = field(default_factory=dict)
    artifacts: dict[str, ArtifactInstance] = field(
        default_factory=dict)


_TEST_BLOCK_NAME = "play"
"""Default block name the fakes record for their simulated
executions.  Mirrors a realistic cyberarc-shaped workload so the
state_dir path looks like what the real runtime would produce."""


def _fake_state_dir(
    workspace_root: Path,
    execution_id: str,
    block_name: str = _TEST_BLOCK_NAME,
) -> Path:
    """Return the state_dir path for a given execution.

    Mirrors what a real launch+capture would produce:
    ``<workspace>/state/<block_name>/<exec_id>/``.
    """
    return workspace_root / "state" / block_name / execution_id


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
        status: Execution status to stamp on the simulated
            :class:`BlockExecution` record.  Real runtime can
            record any of ``"succeeded"``, ``"failed"``,
            ``"interrupted"`` with a captured ``state_dir``;
            the fake defaults to ``"succeeded"`` for
            happy-path tests but lets failure-path tests set
            this explicitly so they can pin what the state
            restore logic does with non-clean captures.
    """

    exit_reason: str = "completed"
    pending: list[dict[str, Any]] = field(default_factory=list)
    write_session: bool = False
    session_jsonl_text: str = ""
    execution_id: str = "exec-0"
    agent_workspace_dir: str = "agent_workspaces/handoff_test"
    status: str = "succeeded"


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
        workspace: _FakeWorkspace | None = None,
    ) -> None:
        self._workspace_root = workspace_root
        self._script = script
        self._workspace = workspace

    def wait(self) -> AgentResult:
        """Apply the cycle's script and return the AgentResult.

        Simulates what a real launch + capture would produce:

        * Writes the session JSONL into the captured state dir
          (``<workspace>/state/<block_name>/<exec_id>/session.jsonl``).
        * On a ``tool_handoff`` exit, populates
          ``AgentResult.pending_tool_calls`` and ``exit_state``
          directly (the real launcher does this by reading control
          files from the ``/flywheel/control/`` mount after exit).
        * Adds the matching :class:`BlockExecution` record on the
          fake workspace so the handoff loop can resolve
          ``state_dir`` from the execution id.
        """
        state_dir: str | None = None
        if self._script.write_session:
            state_dir_path = _fake_state_dir(
                self._workspace_root,
                self._script.execution_id,
                block_name=_TEST_BLOCK_NAME,
            )
            state_dir_path.mkdir(parents=True, exist_ok=True)
            (state_dir_path / SESSION_FILE_NAME).write_text(
                self._script.session_jsonl_text, encoding="utf-8")
            state_dir = str(
                state_dir_path.relative_to(self._workspace_root)
            ).replace("\\", "/")

        # Register the fake execution so the loop can look up
        # state_dir by execution id.
        if self._workspace is not None:
            now = datetime.now(UTC)
            self._workspace.executions[
                self._script.execution_id
            ] = BlockExecution(
                id=self._script.execution_id,
                block_name=_TEST_BLOCK_NAME,
                started_at=now,
                finished_at=now,
                status=self._script.status,
                state_dir=state_dir,
                output_bindings={},
            )

        pending_tool_calls: list[dict[str, Any]] | None = None
        exit_state: dict[str, Any] | None = None
        if self._script.exit_reason == "tool_handoff":
            pending_tool_calls = list(self._script.pending)
            exit_state = {
                "session_id": "sess-fake",
                "status": "tool_handoff",
                "reason": "",
            }

        return AgentResult(
            exit_code=0,
            elapsed_s=0.1,
            evals_run=0,
            execution_id=self._script.execution_id,
            stop_reason=None,
            exit_reason=self._script.exit_reason,
            exit_state=exit_state,
            pending_tool_calls=pending_tool_calls,
        )


def _make_launch_fn(
    workspace_root: Path,
    scripts: list[_CycleScript],
    record: list[_LaunchCall],
    workspace: _FakeWorkspace | None = None,
) -> Callable[..., _FakeAgentHandle]:
    """Build a fake launch_fn that walks ``scripts`` cycle by cycle.

    Each call pops the next script and constructs a
    :class:`_FakeAgentHandle` honoring it.  ``record`` is appended
    to with the kwargs the loop used so the test can assert on the
    relaunch wiring (predecessor id, reuse flag, prompt swap).
    Calling more times than there are scripts is a test bug and
    surfaces as ``IndexError`` rather than silent passing.

    ``workspace`` is the fake workspace the handles should record
    simulated executions on.  Passing it lets the fake handles
    write state_dir entries that the handoff loop needs in order
    to locate the session JSONL.
    """
    iteration = {"i": 0}

    def _fake(**kwargs: Any) -> _FakeAgentHandle:
        record.append(_LaunchCall(kwargs=kwargs))
        idx = iteration["i"]
        iteration["i"] = idx + 1
        return _FakeAgentHandle(
            workspace_root=workspace_root,
            script=scripts[idx],
            workspace=workspace,
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
        "block_name": _TEST_BLOCK_NAME,
        "agent_workspace_dir": "agent_workspaces/handoff_test",
    }
