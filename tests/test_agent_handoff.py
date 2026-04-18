"""Unit tests for ``flywheel.agent_handoff.run_agent_with_handoffs``.

The loop's collaborators (``launch_agent_block`` and the agent
container behind it) are heavy and Docker-dependent.  These tests
exercise the loop's contract by injecting a fake ``launch_fn``
that produces programmable :class:`FakeAgentHandle` objects: each
fake handle decides what ``AgentResult.exit_reason`` to report and
what files to leave on disk in the agent workspace.  The
``flywheel.session_splice`` module is *not* mocked — the loop
performs real splices on synthetic JSONL fixtures, so a regression
in either the loop or the splice is caught here.

What this file covers:
  * No-handoff case: collapses to one launch (backward compat).
  * Single handoff: launch → splice → relaunch → completion.
  * Multi-tool-use handoff: one launch surfaces N pending tool
    calls; loop runs them serially and splices N times.
  * Sequence: cycles chain via ``predecessor_id``;
    ``reuse_workspace`` and ``RESUME_SESSION_FILE`` are set on
    relaunch but not on cycle 0.
  * Pending file is removed between cycles.
  * Resume prompt overrides the original on relaunch cycles.
  * Errors: missing ``block_runner``, missing pending file,
    malformed pending entries, missing session artifact, splice
    failure, ``max_iterations`` exhausted.
  * ``HandoffLoopResult`` accessors (``final_result``,
    ``total_handoffs``).
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
    DEFAULT_RESUME_PROMPT,
    PENDING_FILE_NAME,
    RESUME_ENV_VAR,
    SESSION_ARTIFACT_NAME,
    HandoffContext,
    HandoffLoopError,
    HandoffResult,
    run_agent_with_handoffs,
)

# --------------------------------------------------------------------
# Fixtures and helpers.
# --------------------------------------------------------------------


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
    """

    exit_reason: str = "completed"
    pending: list[dict[str, Any]] = field(default_factory=list)
    write_session: bool = False
    session_jsonl_text: str = ""
    execution_id: str = "exec-0"
    agent_workspace_dir: str = "agent_workspaces/handoff_test"


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


# --------------------------------------------------------------------
# Backward-compat: no handoff at all.
# --------------------------------------------------------------------


class TestNoHandoffPath:
    """Without ``HANDOFF_TOOLS`` no agent ever exits with
    ``tool_handoff``; the loop must collapse to a single launch
    and pass through everything the caller supplied unchanged.
    """

    def test_single_launch_no_block_runner_needed(
        self, workspace: _FakeWorkspace,
    ) -> None:
        scripts = [_CycleScript(exit_reason="completed")]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        out = run_agent_with_handoffs(
            block_runner=None,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )

        assert len(out.cycles) == 1
        assert out.total_handoffs == 0
        assert out.final_result.exit_reason == "completed"
        assert len(record) == 1
        only_call = record[0].kwargs
        assert only_call["prompt"] == "do the thing"
        assert "reuse_workspace" not in only_call
        assert "predecessor_id" not in only_call or (
            only_call.get("predecessor_id") is None)

    def test_terminating_exit_reasons_other_than_handoff_terminate(
        self, workspace: _FakeWorkspace,
    ) -> None:
        for term in ("crashed", "auth_failure", "max_turns",
                     "rate_limit", "stopped"):
            scripts = [_CycleScript(exit_reason=term)]
            record: list[_LaunchCall] = []
            launch = _make_launch_fn(
                workspace.path, scripts, record)
            out = run_agent_with_handoffs(
                block_runner=None,
                launch_fn=launch,
                **_base_kwargs(workspace),
            )
            assert len(out.cycles) == 1, (
                f"exit_reason={term!r} should terminate the loop, "
                f"got {len(out.cycles)} cycles"
            )
            assert out.final_result.exit_reason == term


# --------------------------------------------------------------------
# Single handoff: full round trip.
# --------------------------------------------------------------------


class TestSingleHandoffRoundTrip:
    """One handoff cycle: launch → handoff → block_runner → splice
    → relaunch → completion.  The most common production case."""

    def test_block_runner_called_once_with_correct_context(
        self, workspace: _FakeWorkspace,
    ) -> None:
        session_jsonl = _build_session_jsonl(
            session_id="sess-single",
            tool_use_ids=["toolu_solo"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_solo",
                    "tool_name": "mcp__demo__handoff_me",
                    "tool_input": {"value": 7},
                    "captured_at": "2026-04-17T15:23:01+00:00",
                }],
                write_session=True,
                session_jsonl_text=session_jsonl,
                execution_id="exec-0",
            ),
            _CycleScript(
                exit_reason="completed",
                execution_id="exec-1",
            ),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        invocations: list[HandoffContext] = []

        def _block_runner(ctx: HandoffContext) -> HandoffResult:
            invocations.append(ctx)
            return HandoffResult(
                content=f"REAL_RESULT[{ctx.tool_use_id}]",
            )

        out = run_agent_with_handoffs(
            block_runner=_block_runner,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )

        assert len(out.cycles) == 2
        assert out.total_handoffs == 1
        assert out.final_result.exit_reason == "completed"

        assert len(invocations) == 1
        ctx = invocations[0]
        assert ctx.tool_use_id == "toolu_solo"
        assert ctx.tool_name == "mcp__demo__handoff_me"
        assert ctx.tool_input == {"value": 7}
        assert ctx.iteration == 0
        assert ctx.index_in_iteration == 0
        assert ctx.siblings == 1
        assert ctx.session_id == "sess-fake"

    def test_splice_replaces_deny_with_real_result(
        self, workspace: _FakeWorkspace,
    ) -> None:
        session_jsonl = _build_session_jsonl(
            session_id="sess-splice",
            tool_use_ids=["toolu_splice"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_splice",
                    "tool_name": "mcp__demo__handoff_me",
                    "tool_input": {},
                }],
                write_session=True,
                session_jsonl_text=session_jsonl,
            ),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        def _block_runner(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="REAL_RESULT_AFTER_SPLICE")

        run_agent_with_handoffs(
            block_runner=_block_runner,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )

        agent_ws = workspace.path / "agent_workspaces/handoff_test"
        spliced = (agent_ws / SESSION_ARTIFACT_NAME).read_text(
            encoding="utf-8")
        assert "REAL_RESULT_AFTER_SPLICE" in spliced
        assert "permission denied: handoff_to_flywheel" not in spliced

    def test_relaunch_kwargs_set_resume_env_and_reuse_workspace(
        self, workspace: _FakeWorkspace,
    ) -> None:
        session_jsonl = _build_session_jsonl(
            session_id="sess-relaunch",
            tool_use_ids=["toolu_x"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_x",
                    "tool_name": "mcp__demo__handoff_me",
                    "tool_input": {},
                }],
                write_session=True,
                session_jsonl_text=session_jsonl,
                execution_id="exec-prev",
            ),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            extra_env={"PRESERVED_KEY": "preserved_value"},
            **_base_kwargs(workspace),
        )

        assert len(record) == 2

        cycle0 = record[0].kwargs
        assert cycle0["prompt"] == "do the thing"
        assert cycle0.get("reuse_workspace") is not True
        assert cycle0.get("predecessor_id") is None
        assert cycle0["extra_env"] == {
            "PRESERVED_KEY": "preserved_value",
        }

        cycle1 = record[1].kwargs
        assert cycle1["reuse_workspace"] is True
        assert cycle1["predecessor_id"] == "exec-prev"
        assert cycle1["agent_workspace_dir"] == (
            "agent_workspaces/handoff_test")
        assert cycle1["prompt"] == DEFAULT_RESUME_PROMPT
        assert cycle1["extra_env"] == {
            "PRESERVED_KEY": "preserved_value",
            RESUME_ENV_VAR: "/workspace/" + SESSION_ARTIFACT_NAME,
        }

    def test_pending_file_removed_after_handoff(
        self, workspace: _FakeWorkspace,
    ) -> None:
        session_jsonl = _build_session_jsonl(
            session_id="sess-cleanup",
            tool_use_ids=["toolu_cleanup"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_cleanup",
                    "tool_name": "mcp__demo__handoff_me",
                    "tool_input": {},
                }],
                write_session=True,
                session_jsonl_text=session_jsonl,
            ),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )

        agent_ws = workspace.path / "agent_workspaces/handoff_test"
        assert not (agent_ws / PENDING_FILE_NAME).exists(), (
            "pending file should be removed after handoff is "
            "resolved so a leftover cannot trigger a phantom cycle"
        )

    def test_custom_resume_prompt_is_used(
        self, workspace: _FakeWorkspace,
    ) -> None:
        session_jsonl = _build_session_jsonl(
            session_id="sess-prompt",
            tool_use_ids=["toolu_p"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_p",
                    "tool_name": "mcp__demo__handoff_me",
                    "tool_input": {},
                }],
                write_session=True,
                session_jsonl_text=session_jsonl,
            ),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            resume_prompt="CUSTOM RESUME INSTRUCTION",
            **_base_kwargs(workspace),
        )

        assert record[1].kwargs["prompt"] == (
            "CUSTOM RESUME INSTRUCTION")


# --------------------------------------------------------------------
# Multi-tool-use handoff.
# --------------------------------------------------------------------


class TestMultiToolUseHandoff:
    """One launch surfaces N pending tool_uses; loop must call
    ``block_runner`` N times in order, splice N times, and only
    relaunch once.  This is the multi-tool-use design (Option α
    + serial in v1)."""

    def test_three_parallel_tool_uses_one_relaunch(
        self, workspace: _FakeWorkspace,
    ) -> None:
        ids = ["toolu_p0", "toolu_p1", "toolu_p2"]
        session_jsonl = _build_session_jsonl(
            session_id="sess-multi",
            tool_use_ids=ids,
        )
        pending = [
            {
                "tool_use_id": tuid,
                "tool_name": f"mcp__demo__handoff_{i}",
                "tool_input": {"i": i},
            }
            for i, tuid in enumerate(ids)
        ]
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=pending,
                write_session=True,
                session_jsonl_text=session_jsonl,
            ),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        invocations: list[HandoffContext] = []

        def _br(ctx: HandoffContext) -> HandoffResult:
            invocations.append(ctx)
            return HandoffResult(
                content=f"RESULT[{ctx.index_in_iteration}]"
                        f":{ctx.tool_use_id}",
            )

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )

        assert len(record) == 2, (
            "multi-tool-use should still take exactly two launches "
            "(one stop/restart cycle), not N+1"
        )
        assert out.total_handoffs == 3

        assert [c.tool_use_id for c in invocations] == ids
        assert [c.index_in_iteration for c in invocations] == [
            0, 1, 2]
        assert [c.siblings for c in invocations] == [3, 3, 3]
        assert [c.iteration for c in invocations] == [0, 0, 0]

        agent_ws = workspace.path / "agent_workspaces/handoff_test"
        spliced = (agent_ws / SESSION_ARTIFACT_NAME).read_text(
            encoding="utf-8")
        for i, tuid in enumerate(ids):
            assert f"RESULT[{i}]:{tuid}" in spliced, (
                f"splice missing for {tuid} at index {i}"
            )

    def test_handoff_record_shape(
        self, workspace: _FakeWorkspace,
    ) -> None:
        ids = ["toolu_a", "toolu_b"]
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[
                    {
                        "tool_use_id": tuid,
                        "tool_name": f"mcp__demo__h_{i}",
                        "tool_input": {"i": i},
                    }
                    for i, tuid in enumerate(ids)
                ],
                write_session=True,
                session_jsonl_text=_build_session_jsonl(
                    session_id="sess-rec", tool_use_ids=ids),
            ),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content=ctx.tool_use_id)

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )

        cycle0 = out.cycles[0]
        assert len(cycle0.handoffs) == 2
        for i, rec in enumerate(cycle0.handoffs):
            assert rec["tool_use_id"] == ids[i]
            assert rec["tool_name"] == f"mcp__demo__h_{i}"
            assert rec["tool_input"] == {"i": i}
            assert rec["index_in_iteration"] == i
            assert rec["siblings"] == 2
            assert rec["is_error"] is False
            assert isinstance(rec["splice_line"], int)
            assert rec["splice_line"] >= 1


# --------------------------------------------------------------------
# Error paths.
# --------------------------------------------------------------------


class TestErrorPaths:
    """Things that should explode loudly rather than silently."""

    def test_handoff_without_block_runner_raises(
        self, workspace: _FakeWorkspace,
    ) -> None:
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_x",
                    "tool_name": "mcp__demo__h",
                    "tool_input": {},
                }],
                write_session=True,
                session_jsonl_text=_build_session_jsonl(
                    session_id="s", tool_use_ids=["toolu_x"]),
            ),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        with pytest.raises(HandoffLoopError, match="block_runner"):
            run_agent_with_handoffs(
                block_runner=None,
                launch_fn=launch,
                **_base_kwargs(workspace),
            )

    def test_pending_file_missing_raises(
        self, workspace: _FakeWorkspace,
    ) -> None:
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[],
                write_session=True,
                session_jsonl_text=_build_session_jsonl(
                    session_id="s", tool_use_ids=["toolu_x"]),
            ),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        with pytest.raises(
            HandoffLoopError, match="no tool calls",
        ):
            run_agent_with_handoffs(
                block_runner=_br,
                launch_fn=launch,
                **_base_kwargs(workspace),
            )

    def test_pending_entry_missing_tool_use_id_raises(
        self, workspace: _FakeWorkspace,
    ) -> None:
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_name": "mcp__demo__h",
                    "tool_input": {},
                }],
                write_session=True,
                session_jsonl_text=_build_session_jsonl(
                    session_id="s",
                    tool_use_ids=["whatever"]),
            ),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        with pytest.raises(
            HandoffLoopError, match="no tool_use_id",
        ):
            run_agent_with_handoffs(
                block_runner=_br,
                launch_fn=launch,
                **_base_kwargs(workspace),
            )

    def test_session_artifact_missing_raises(
        self, workspace: _FakeWorkspace,
    ) -> None:
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_x",
                    "tool_name": "mcp__demo__h",
                    "tool_input": {},
                }],
                write_session=False,
            ),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        with pytest.raises(
            HandoffLoopError, match="session artifact",
        ):
            run_agent_with_handoffs(
                block_runner=_br,
                launch_fn=launch,
                **_base_kwargs(workspace),
            )

    def test_splice_error_for_unknown_tool_use_id_raises(
        self, workspace: _FakeWorkspace,
    ) -> None:
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_does_not_match",
                    "tool_name": "mcp__demo__h",
                    "tool_input": {},
                }],
                write_session=True,
                session_jsonl_text=_build_session_jsonl(
                    session_id="s",
                    tool_use_ids=["toolu_actually_in_session"]),
            ),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        with pytest.raises(HandoffLoopError, match="splice failed"):
            run_agent_with_handoffs(
                block_runner=_br,
                launch_fn=launch,
                **_base_kwargs(workspace),
            )

    def test_max_iterations_exhausted_raises(
        self, workspace: _FakeWorkspace,
    ) -> None:
        max_iter = 3
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": f"toolu_loop_{i}",
                    "tool_name": "mcp__demo__h",
                    "tool_input": {},
                }],
                write_session=True,
                session_jsonl_text=_build_session_jsonl(
                    session_id=f"s{i}",
                    tool_use_ids=[f"toolu_loop_{i}"]),
            )
            for i in range(max_iter)
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        with pytest.raises(
            HandoffLoopError, match="max_iterations",
        ):
            run_agent_with_handoffs(
                block_runner=_br,
                launch_fn=launch,
                max_iterations=max_iter,
                **_base_kwargs(workspace),
            )
        assert len(record) == max_iter


# --------------------------------------------------------------------
# Result accessors.
# --------------------------------------------------------------------


class TestHandoffLoopResult:
    """The ``final_result`` and ``total_handoffs`` accessors are
    the public API for callers that don't want to walk cycles
    manually."""

    def test_accessors_after_multi_cycle_run(
        self, workspace: _FakeWorkspace,
    ) -> None:
        ids_per_cycle = [
            ["toolu_a", "toolu_b"],
            ["toolu_c"],
        ]
        scripts: list[_CycleScript] = []
        for i, ids in enumerate(ids_per_cycle):
            scripts.append(_CycleScript(
                exit_reason="tool_handoff",
                pending=[
                    {
                        "tool_use_id": tuid,
                        "tool_name": "mcp__demo__h",
                        "tool_input": {"i": j},
                    }
                    for j, tuid in enumerate(ids)
                ],
                write_session=True,
                session_jsonl_text=_build_session_jsonl(
                    session_id=f"s{i}", tool_use_ids=ids),
                execution_id=f"exec-{i}",
            ))
        scripts.append(_CycleScript(
            exit_reason="completed",
            execution_id="exec-final",
        ))
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content=f"r-{ctx.tool_use_id}")

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )

        assert len(out.cycles) == 3
        assert out.total_handoffs == 3
        assert out.final_result.execution_id == "exec-final"
        assert out.final_result.exit_reason == "completed"
        assert out.cycles[-1].handoffs == []
