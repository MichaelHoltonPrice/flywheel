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

from pathlib import Path
from typing import Any

import pytest

from flywheel.agent_handoff import (
    DEFAULT_RESUME_PROMPT,
    PENDING_FILE_NAME,
    RESUME_ENV_VAR,
    SESSION_ARTIFACT_NAME,
    BlockRunner,
    HandoffContext,
    HandoffLoopError,
    HandoffResult,
    make_tool_router,
    run_agent_with_handoffs,
)

from ._handoff_helpers import (
    _base_kwargs,
    _build_session_jsonl,
    _CycleScript,
    _FakeWorkspace,
    _LaunchCall,
    _make_launch_fn,
    workspace,
)

# Re-exported for use by tests in this file via fixture injection.
# Pytest discovers the fixture by import name, so the module-level
# alias keeps the test cases below unchanged after the helper
# extraction.
_ = workspace


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
# Halt source.
# --------------------------------------------------------------------


class TestHaltSource:
    """``halt_source`` lets a post-execution check stop the loop
    between cycles.  The loop drains it after resolving each
    cycle's pending tool calls and before relaunching; any
    non-empty return terminates immediately and the directives
    surface on ``HandoffLoopResult.halts``.
    """

    def test_no_halt_source_runs_to_natural_completion(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """Default ``halt_source=None`` is the no-halt path: the
        loop runs until a non-handoff exit and ``halts`` stays
        empty.  Pinned so adding ``halt_source`` cannot regress
        existing callers that never opt in.
        """
        session_jsonl = _build_session_jsonl(
            session_id="sess-no-halt",
            tool_use_ids=["toolu_a"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_a",
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

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )

        assert out.halts == []
        assert out.final_result.exit_reason == "completed"
        assert len(out.cycles) == 2

    def test_halt_source_returning_empty_does_not_stop(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """An empty drain is the steady state: the loop relaunches
        normally.  Verifies the loop checks truthiness and not
        ``is None``, so a recorder that always returns ``[]``
        when nothing is queued behaves the same as no halt
        source at all.
        """
        session_jsonl = _build_session_jsonl(
            session_id="sess-empty",
            tool_use_ids=["toolu_e"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_e",
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

        drain_calls = {"n": 0}

        def _drain() -> list[Any]:
            drain_calls["n"] += 1
            return []

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            halt_source=_drain,
            **_base_kwargs(workspace),
        )

        assert drain_calls["n"] == 1, (
            "halt_source should be drained once per cycle that "
            "produced handoffs (the terminating cycle returns "
            "early before the drain)"
        )
        assert out.halts == []
        assert len(out.cycles) == 2
        assert out.final_result.exit_reason == "completed"

    def test_halt_source_returning_directive_stops_loop(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """A non-empty drain after cycle 0 stops the loop: the
        cycle's handoffs are still recorded (the splice already
        happened), but the second launch never fires.
        """
        session_jsonl = _build_session_jsonl(
            session_id="sess-halt",
            tool_use_ids=["toolu_h"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_h",
                    "tool_name": "mcp__demo__handoff_me",
                    "tool_input": {},
                }],
                write_session=True,
                session_jsonl_text=session_jsonl,
                execution_id="exec-halted",
            ),
            _CycleScript(exit_reason="completed"),  # never reached
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        directive = {
            "scope": "run",
            "reason": "post-check rejected",
            "block": "demo_block",
            "execution_id": "exec-failed",
        }

        def _drain() -> list[Any]:
            return [directive]

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            halt_source=_drain,
            **_base_kwargs(workspace),
        )

        assert len(record) == 1, (
            "halt should suppress the relaunch; only cycle 0 ran"
        )
        assert len(out.cycles) == 1
        assert out.cycles[0].agent_result.execution_id == (
            "exec-halted")
        assert len(out.cycles[0].handoffs) == 1, (
            "the handoff that fired before the halt is still in "
            "the cycle record (its splice happened first)"
        )
        assert out.halts == [directive]
        assert out.final_result.exit_reason == "tool_handoff", (
            "natural-completion exit_reason is not synthesized; "
            "the last real launch was the handoff exit"
        )

    def test_halt_source_drain_only_once_per_handoff_cycle(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """Across two handoff cycles the drain runs once per
        cycle and any directive in cycle 1 cuts off cycle 2's
        launch.  Pins the contract that halts queued *during*
        cycle N's handoffs apply *after* cycle N's results are
        spliced in -- the agent already saw its tool results,
        we just refuse to relaunch.
        """
        session_jsonl_a = _build_session_jsonl(
            session_id="sess-multi-a",
            tool_use_ids=["toolu_a"],
        )
        session_jsonl_b = _build_session_jsonl(
            session_id="sess-multi-b",
            tool_use_ids=["toolu_b"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_a",
                    "tool_name": "mcp__demo__handoff_me",
                    "tool_input": {},
                }],
                write_session=True,
                session_jsonl_text=session_jsonl_a,
                execution_id="exec-cycle0",
            ),
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_b",
                    "tool_name": "mcp__demo__handoff_me",
                    "tool_input": {},
                }],
                write_session=True,
                session_jsonl_text=session_jsonl_b,
                execution_id="exec-cycle1",
            ),
            _CycleScript(exit_reason="completed"),  # never reached
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record)

        drained: list[list[Any]] = [[], [{"reason": "halt-now"}]]

        def _drain() -> list[Any]:
            return drained.pop(0)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            halt_source=_drain,
            **_base_kwargs(workspace),
        )

        assert len(record) == 2, (
            "first drain returned [], so cycle 1 launched; "
            "second drain returned a directive, so cycle 2 was "
            "suppressed"
        )
        assert len(out.cycles) == 2
        assert out.halts == [{"reason": "halt-now"}]


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


def _ctx(tool_name: str, **overrides: Any) -> HandoffContext:
    """Build a HandoffContext with sensible defaults for router tests."""
    fields = dict(
        tool_name=tool_name,
        tool_input={},
        session_id="sess-1",
        tool_use_id="toolu_x",
        agent_workspace=Path("/tmp/ws"),
        iteration=0,
        index_in_iteration=0,
        siblings=1,
    )
    fields.update(overrides)
    return HandoffContext(**fields)


class TestMakeToolRouter:
    """Pin the dispatch contract of ``make_tool_router``.

    The router is the seam every multi-tool handoff project (e.g.
    cyberarc, once ``predict_action`` migrates) hangs runners off
    of, so its behavior is exercised directly rather than only
    through ``run_agent_with_handoffs``.
    """

    def test_dispatches_to_matching_runner(self) -> None:
        seen: list[str] = []

        def runner_a(ctx: HandoffContext) -> HandoffResult:
            seen.append(f"a:{ctx.tool_name}")
            return HandoffResult(content="from-a")

        def runner_b(ctx: HandoffContext) -> HandoffResult:
            seen.append(f"b:{ctx.tool_name}")
            return HandoffResult(content="from-b")

        router = make_tool_router({
            "mcp__x__alpha": runner_a,
            "mcp__x__beta": runner_b,
        })

        out_a = router(_ctx("mcp__x__alpha"))
        out_b = router(_ctx("mcp__x__beta"))

        assert out_a.content == "from-a"
        assert out_a.is_error is False
        assert out_b.content == "from-b"
        assert seen == ["a:mcp__x__alpha", "b:mcp__x__beta"]

    def test_unknown_tool_returns_error_result_not_raise(self) -> None:
        def never_called(ctx: HandoffContext) -> HandoffResult:
            raise AssertionError("must not be invoked")

        router = make_tool_router({"mcp__x__alpha": never_called})

        out = router(_ctx("mcp__x__gamma"))

        assert out.is_error is True
        assert "no handoff runner" in out.content
        assert "mcp__x__gamma" in out.content
        assert "mcp__x__alpha" in out.content

    def test_routes_snapshotted_at_construction(self) -> None:
        """Mutating the input dict after construction must not affect dispatch."""
        def runner_a(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="a")

        def runner_b(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="b-late")

        routes: dict[str, Any] = {"mcp__x__alpha": runner_a}
        router = make_tool_router(routes)

        routes["mcp__x__beta"] = runner_b

        out = router(_ctx("mcp__x__beta"))
        assert out.is_error is True

    def test_runner_exception_propagates(self) -> None:
        """Router does not swallow runner errors.

        The handoff loop has its own outer try/except for that;
        the router stays a thin lookup.
        """

        def boom(ctx: HandoffContext) -> HandoffResult:
            raise RuntimeError("runner exploded")

        router = make_tool_router({"mcp__x__alpha": boom})

        with pytest.raises(RuntimeError, match="runner exploded"):
            router(_ctx("mcp__x__alpha"))

    def test_empty_routes_always_returns_error(self) -> None:
        router = make_tool_router({})
        out = router(_ctx("mcp__x__anything"))
        assert out.is_error is True
        assert "registered tools: []" in out.content

    def test_returned_callable_satisfies_block_runner_protocol(self) -> None:
        """Smoke-test that the router plugs into ``run_agent_with_handoffs``."""

        def runner_a(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        router: BlockRunner = make_tool_router({
            "mcp__x__alpha": runner_a,
        })

        assert callable(router)
        result = router(_ctx("mcp__x__alpha"))
        assert isinstance(result, HandoffResult)
