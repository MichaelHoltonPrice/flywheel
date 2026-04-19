"""Crash-resume contract tests for the handoff loop.

Drives :func:`flywheel.agent_handoff.run_agent_with_handoffs`
through every host-side failure mode in the run_agent_with_handoffs
state diagram so the recovery story for each kill point has at
least one deterministic test pinning current behavior:

==========================  =================================
Where it dies               Coverage in this file
==========================  =================================
Before step 3               ``TestStaleArtifactsTolerated``
Step 3, partial write       ``TestPendingFileShape``
Between 5 and 8             ``TestStaleArtifactsTolerated``
Between 9 and 11            existing happy-path coverage
During step 12              ``TestBlockRunnerFailure``
Step 13 partial             splice atomicity (covered in
                            ``test_session_splice``); plus
                            ``TestSpliceTmpFileTolerated``
After 14, before 16         existing happy-path coverage
During step 16              container-internal; documented in
                            ``tests/integration/README.md``
==========================  =================================

Container-internal kill points (kill mid-PreToolUse, kill
between deny-tool_result and clean exit, kill mid-resume) are
intentionally *not* exercised here — they require Docker plus a
live model, and the timing windows are too narrow to reliably
hit from a test harness.  The integration-test README documents
how to exercise them by hand when investigating a real-world
incident.

What every test in this file shares:

* No live API, no Docker, no network.  Runs in <1s.
* Real :func:`flywheel.session_splice.splice_tool_result` (not
  mocked) — partial-splice assertions verify on-disk state, not
  call records.
* The fake launcher infrastructure from
  :mod:`tests._handoff_helpers` so the loop's contract is the
  thing under test, not the harness.

These tests assume the loop's *recovery posture* is "raise loudly
on any boundary violation; leave the workspace in a state an
operator can inspect."  The contract is *not* "automatically
re-drive."  An operator (or a higher-level scheduler) is
responsible for re-running the loop after fixing whatever caused
the failure; the workspace state these tests pin tells them what
to look at.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywheel.agent_handoff import (
    PENDING_FILE_NAME,
    SESSION_FILE_NAME,
    HandoffContext,
    HandoffLoopError,
    HandoffResult,
    run_agent_with_handoffs,
)
from flywheel.session_splice import find_pending_deny_tool_use_ids

from ._handoff_helpers import (
    _base_kwargs,
    _build_session_jsonl,
    _CycleScript,
    _fake_state_dir,
    _FakeWorkspace,
    _LaunchCall,
    _make_launch_fn,
    workspace,
)

_ = workspace


# --------------------------------------------------------------------
# Helpers private to this file.
# --------------------------------------------------------------------


_AGENT_WS_SUBPATH = "agent_workspaces/handoff_test"


def _agent_ws(workspace: _FakeWorkspace) -> Path:
    """Resolve the host-side agent workspace dir the loop uses.

    Mirrors the ``agent_workspace_dir`` ``_base_kwargs`` ships,
    which the fake handle obeys verbatim.
    """
    return workspace.path / _AGENT_WS_SUBPATH


def _read_session(
    workspace: _FakeWorkspace, execution_id: str = "exec-0",
) -> str:
    """Convenience: read the spliced session JSONL from disk.

    Defaults to ``exec-0`` to match the default script
    ``execution_id``; pass explicitly when a test uses a
    non-default id.
    """
    return (
        _session_path(workspace, execution_id)
    ).read_text(encoding="utf-8")


def _pending_path(workspace: _FakeWorkspace) -> Path:
    return _agent_ws(workspace) / PENDING_FILE_NAME


def _session_path(
    workspace: _FakeWorkspace, execution_id: str = "exec-0",
) -> Path:
    """Resolve the session JSONL path inside the captured state_dir."""
    return (
        _fake_state_dir(workspace.path, execution_id)
        / SESSION_FILE_NAME
    )


def _multi_pending_script(
    *,
    session_id: str,
    tool_use_ids: list[str],
) -> _CycleScript:
    """Build a cycle that surfaces ``len(tool_use_ids)`` parallel handoffs.

    Mirrors the multi-tool-use handoff shape: one assistant turn
    with N parallel ``tool_use`` blocks, one user turn with N
    deny ``tool_result`` blocks.  Useful for any test that wants
    to exercise mid-batch behavior.
    """
    return _CycleScript(
        exit_reason="tool_handoff",
        pending=[
            {
                "tool_use_id": tuid,
                "tool_name": "mcp__demo__handoff_me",
                "tool_input": {"k": tuid},
            }
            for tuid in tool_use_ids
        ],
        write_session=True,
        session_jsonl_text=_build_session_jsonl(
            session_id=session_id, tool_use_ids=tool_use_ids),
    )


# --------------------------------------------------------------------
# Block runner raises mid-batch
# --------------------------------------------------------------------


class TestBlockRunnerFailure:
    """The runner is the seam most likely to misbehave in production:
    every nested block's actual work runs through it.

    Contract: when ``block_runner`` raises, the loop propagates
    the exception unchanged.  Anything spliced before the failure
    stays spliced (the splice is atomic per-entry); anything
    after the failure is left untouched.  The pending file is
    *not* removed, so an operator inspecting the workspace can
    see exactly which calls were resolved and which were not.
    """

    def test_first_entry_runner_exception_leaves_no_splices(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """Runner raises on entry 0 of 3: zero entries spliced,
        original deny tool_results all still in the JSONL,
        pending file intact for operator inspection.
        """
        ids = ["toolu_a", "toolu_b", "toolu_c"]
        scripts = [_multi_pending_script(
            session_id="sess-fail-first", tool_use_ids=ids)]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        def _br(ctx: HandoffContext) -> HandoffResult:
            raise RuntimeError("simulated runner crash")

        with pytest.raises(RuntimeError, match="simulated runner crash"):
            run_agent_with_handoffs(
                block_runner=_br,
                launch_fn=launch,
                **_base_kwargs(workspace),
            )

        assert _pending_path(workspace).is_file(), (
            "pending file must remain so the operator can see "
            "which tool calls were not resolved"
        )
        deny_ids = find_pending_deny_tool_use_ids(
            _session_path(workspace))
        assert deny_ids == ids, (
            f"no entries should have been spliced; expected all "
            f"{ids} still as deny markers, got {deny_ids}"
        )

    def test_mid_batch_runner_exception_keeps_partial_splices(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """Runner raises on entry 1 of 3: entry 0 stays spliced,
        entries 1+2 stay as deny markers, pending file remains.

        This is the load-bearing recovery test for multi-tool-use
        handoffs.  An operator re-driving the loop after fixing
        whatever the runner crashed on can rely on the spliced
        entry already being resolved (no double-execution risk).
        """
        ids = ["toolu_a", "toolu_b", "toolu_c"]
        scripts = [_multi_pending_script(
            session_id="sess-fail-mid", tool_use_ids=ids)]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        def _br(ctx: HandoffContext) -> HandoffResult:
            if ctx.index_in_iteration == 0:
                return HandoffResult(content="real-result-for-a")
            raise RuntimeError("simulated runner crash on entry 1")

        with pytest.raises(RuntimeError, match="entry 1"):
            run_agent_with_handoffs(
                block_runner=_br,
                launch_fn=launch,
                **_base_kwargs(workspace),
            )

        assert _pending_path(workspace).is_file()
        spliced = _read_session(workspace)
        assert "real-result-for-a" in spliced, (
            "entry 0's splice must have completed before the crash"
        )
        # Entries 1 and 2 still carry the deny marker.
        deny_ids = find_pending_deny_tool_use_ids(
            _session_path(workspace))
        assert deny_ids == ["toolu_b", "toolu_c"], (
            f"unspliced entries should still appear as deny "
            f"markers; got {deny_ids}"
        )

    def test_runner_returning_is_error_continues_loop(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """``HandoffResult(is_error=True)`` is a normal outcome, not
        a crash: the splice goes in with ``is_error=True`` set on
        the tool_result block and the loop relaunches as usual.

        The agent is expected to perceive this as a tool that
        legitimately failed (returned an error) — distinct from
        the loop refusing to proceed.
        """
        ids = ["toolu_e"]
        scripts = [
            _multi_pending_script(
                session_id="sess-iserror", tool_use_ids=ids),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(
                content="block raised: exact reason here",
                is_error=True,
            )

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )

        assert out.final_result.exit_reason == "completed"
        assert len(out.cycles) == 2
        assert len(out.cycles[0].handoffs) == 1
        assert out.cycles[0].handoffs[0]["is_error"] is True
        # Pending file removed because the batch fully resolved
        # (an error result is still a result — splice happened).
        assert not _pending_path(workspace).exists()
        spliced = _read_session(workspace)
        assert "block raised: exact reason here" in spliced
        # The spliced tool_result block's is_error flag flipped.
        for raw in spliced.splitlines():
            obj = json.loads(raw)
            content = obj.get("message", {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") == "toolu_e"
                ):
                    assert block.get("is_error") is True
                    return
        pytest.fail("spliced is_error=True block not found in JSONL")


# --------------------------------------------------------------------
# Splice mid-batch failure
# --------------------------------------------------------------------


class TestSpliceFailureMidBatch:
    """Splice-side failures during a multi-entry batch.

    Splice is atomic per call (temp file + ``os.replace`` — see
    :func:`flywheel.session_splice._atomic_write`), so an
    individual splice cannot leave the JSONL in a partial state.
    What *can* happen is one entry's splice succeeding and a
    later entry's splice failing — typically because the pending
    file's ``tool_use_id`` doesn't match anything in the session
    JSONL (a contract violation that surfaces only at splice
    time).  The loop must surface that as
    :class:`HandoffLoopError`, leave the partial splices in
    place, and leave the pending file on disk.
    """

    def test_unknown_tool_use_id_mid_batch(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """Pending lists ids ``[A, ZZZ, C]`` but session JSONL only
        carries ids ``[A, C]``.  The loop splices A, fails on ZZZ
        with :class:`HandoffLoopError`, leaves C unspliced, and
        leaves the pending file on disk.
        """
        # Build a session JSONL that knows about A and C only —
        # pending will reference B (toolu_zzz) which has no
        # corresponding deny tool_result, so its splice raises.
        session_text = _build_session_jsonl(
            session_id="sess-unknown",
            tool_use_ids=["toolu_a", "toolu_c"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[
                    {
                        "tool_use_id": "toolu_a",
                        "tool_name": "mcp__demo__handoff_me",
                        "tool_input": {},
                    },
                    {
                        "tool_use_id": "toolu_zzz",
                        "tool_name": "mcp__demo__handoff_me",
                        "tool_input": {},
                    },
                    {
                        "tool_use_id": "toolu_c",
                        "tool_name": "mcp__demo__handoff_me",
                        "tool_input": {},
                    },
                ],
                write_session=True,
                session_jsonl_text=session_text,
            ),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        runner_calls: list[str] = []

        def _br(ctx: HandoffContext) -> HandoffResult:
            runner_calls.append(ctx.tool_use_id)
            return HandoffResult(
                content=f"real-{ctx.tool_use_id}")

        with pytest.raises(HandoffLoopError, match="splice failed"):
            run_agent_with_handoffs(
                block_runner=_br,
                launch_fn=launch,
                **_base_kwargs(workspace),
            )

        # Runner saw the first two entries; the third never ran
        # because the second entry's splice crashed the cycle.
        assert runner_calls == ["toolu_a", "toolu_zzz"]
        assert _pending_path(workspace).is_file()

        # A is spliced (real result lives in the JSONL); C is not
        # (still carries the deny marker).
        spliced = _read_session(workspace)
        assert "real-toolu_a" in spliced
        deny_ids = find_pending_deny_tool_use_ids(
            _session_path(workspace))
        assert deny_ids == ["toolu_c"]


# --------------------------------------------------------------------
# Malformed pending-tool-calls file (partial write)
# --------------------------------------------------------------------


class TestPendingFileShape:
    """A truncated or malformed ``pending_tool_calls.json`` is the
    most plausible "agent crashed mid-write" surface.  The loop's
    contract is to refuse to proceed rather than silently treat
    the broken file as "no handoff" and relaunch fresh.
    """

    def test_truncated_json_treated_as_empty_pending(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """A truncated pending file decodes to nothing.  The loop's
        defensive ``_read_json`` returns ``{}`` on parse failure
        and ``_normalize_pending`` returns ``[]``; the loop then
        raises ``no tool calls`` rather than relaunching with no
        pending payload.

        This is intentional: a truncated pending file means the
        previous container exited with handoff intent but the
        evidence is corrupt; silently treating it as "no
        handoff" would lose the agent's tool calls.
        """
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                write_session=True,
                session_jsonl_text=_build_session_jsonl(
                    session_id="sess-trunc",
                    tool_use_ids=["toolu_x"]),
                pending_text='{"schema_version": 2, "pending": [{"tool_',
            ),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        def _br(ctx: HandoffContext) -> HandoffResult:  # pragma: no cover
            raise AssertionError("runner should not be called")

        with pytest.raises(HandoffLoopError, match="no tool calls"):
            run_agent_with_handoffs(
                block_runner=_br,
                launch_fn=launch,
                **_base_kwargs(workspace),
            )

    def test_garbage_pending_file_treated_as_empty_pending(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """Non-JSON garbage in the pending file decodes to nothing
        and the loop refuses to proceed for the same reason as the
        truncated case.
        """
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                write_session=True,
                session_jsonl_text=_build_session_jsonl(
                    session_id="sess-junk",
                    tool_use_ids=["toolu_x"]),
                pending_text="this is not json at all\n",
            ),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        def _br(ctx: HandoffContext) -> HandoffResult:  # pragma: no cover
            raise AssertionError("runner should not be called")

        with pytest.raises(HandoffLoopError, match="no tool calls"):
            run_agent_with_handoffs(
                block_runner=_br,
                launch_fn=launch,
                **_base_kwargs(workspace),
            )

    def test_pending_with_extra_keys_still_parses(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """Extra keys at envelope level or per entry are tolerated.

        Future schema additions (e.g., per-entry ``target_block``
        or ``params`` fields the host derived later) must not
        require a new ``schema_version`` so long as the existing
        keys keep their meaning.  This test pins forward-compat.
        """
        ids = ["toolu_p"]
        envelope = {
            "schema_version": 2,
            "session_id": "sess-extra",
            "future_field": "future-value",
            "pending": [
                {
                    "tool_use_id": "toolu_p",
                    "tool_name": "mcp__demo__handoff_me",
                    "tool_input": {"k": 1},
                    "captured_at": "2026-04-17T00:00:00Z",
                    "future_per_entry_field": ["a", "b"],
                },
            ],
        }
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                write_session=True,
                session_jsonl_text=_build_session_jsonl(
                    session_id="sess-extra", tool_use_ids=ids),
                pending_text=json.dumps(envelope),
            ),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content=f"real-{ctx.tool_use_id}")

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )
        assert len(out.cycles) == 2
        assert out.final_result.exit_reason == "completed"
        assert len(out.cycles[0].handoffs) == 1


# --------------------------------------------------------------------
# Stale workspace artifacts from a prior crashed run
# --------------------------------------------------------------------


class TestStaleArtifactsTolerated:
    """The workspace is bind-mounted and survives across host
    crashes.  Stale artifacts from a previous failed run must not
    poison a fresh run.

    The fake launcher overwrites ``.agent_state.json``,
    ``pending_tool_calls.json``, and ``agent_session.jsonl`` on
    every cycle the script asks it to, mirroring what real
    container launches do.  These tests construct workspaces with
    pre-existing leftovers before the loop starts and verify that
    the loop only consults what the *current* cycle wrote.
    """

    def test_stale_pending_file_overwritten_by_handoff_cycle(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """A pending file from a previous crashed run is sitting in
        the workspace before the loop starts.  The fresh container
        produces its own pending file (overwriting the stale
        one), and the loop processes only the fresh entries.

        Pins the contract that stale pending files cannot cause
        spurious tool calls to fire after a host-side restart.
        """
        agent_ws = _agent_ws(workspace)
        agent_ws.mkdir(parents=True, exist_ok=True)
        # Pre-seed a stale pending file pointing at a tool call
        # the loop must NOT execute.
        (agent_ws / PENDING_FILE_NAME).write_text(
            json.dumps({
                "schema_version": 2,
                "session_id": "sess-from-prior-crash",
                "pending": [{
                    "tool_use_id": "toolu_stale",
                    "tool_name": "mcp__should__never_run",
                    "tool_input": {"poison": True},
                }],
            }),
            encoding="utf-8",
        )

        ids = ["toolu_fresh"]
        scripts = [
            _multi_pending_script(
                session_id="sess-fresh", tool_use_ids=ids),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        runner_seen: list[str] = []

        def _br(ctx: HandoffContext) -> HandoffResult:
            runner_seen.append(ctx.tool_use_id)
            return HandoffResult(content="ok")

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )

        assert runner_seen == ["toolu_fresh"], (
            f"runner must only see the fresh container's pending "
            f"entries, never the stale ones; got {runner_seen}"
        )
        assert out.final_result.exit_reason == "completed"
        assert not _pending_path(workspace).exists()

    def test_stale_session_jsonl_overwritten_by_fresh_launch(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """A session JSONL from a previous run is sitting in the
        workspace before the loop starts.  The fresh container
        rewrites it, and the splice on cycle 0 operates on the
        fresh JSONL.

        Pins the contract that stale session artifacts cannot
        leak into a fresh run's splice.
        """
        agent_ws = _agent_ws(workspace)
        agent_ws.mkdir(parents=True, exist_ok=True)
        # Stale JSONL with a tool_use_id that doesn't exist in
        # the fresh run.  If the loop accidentally read this it
        # would either crash on splice or silently splice the
        # wrong file.
        (agent_ws / SESSION_FILE_NAME).write_text(
            "STALE_CONTENT_FROM_PRIOR_RUN", encoding="utf-8")

        ids = ["toolu_after"]
        scripts = [
            _multi_pending_script(
                session_id="sess-after", tool_use_ids=ids),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="real-after")

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )
        assert out.final_result.exit_reason == "completed"
        spliced = _read_session(workspace)
        assert "STALE_CONTENT_FROM_PRIOR_RUN" not in spliced
        assert "real-after" in spliced

    def test_stale_pending_ignored_when_agent_exits_clean(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """A stale pending file is sitting in the workspace and the
        fresh container exits with a non-handoff status (and does
        not write a new pending file).  The loop must respect the
        ``exit_reason``, not the on-disk pending file: a
        non-handoff exit terminates the loop cleanly without
        invoking the runner.

        This is the "Before step 3" failure-mode execution: the new
        container's PreToolUse hook never fired, so even though
        leftover state suggests pending work, the *current*
        agent reports nothing pending.
        """
        agent_ws = _agent_ws(workspace)
        agent_ws.mkdir(parents=True, exist_ok=True)
        (agent_ws / PENDING_FILE_NAME).write_text(
            json.dumps({
                "schema_version": 2,
                "session_id": "sess-stale-only",
                "pending": [{
                    "tool_use_id": "toolu_stale",
                    "tool_name": "mcp__should__never_run",
                    "tool_input": {},
                }],
            }),
            encoding="utf-8",
        )

        scripts = [
            _CycleScript(
                exit_reason="completed",
                # Don't write pending or session — emulates an
                # agent that ran fresh, did its job, and exited
                # without ever triggering PreToolUse.
            ),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        def _br(ctx: HandoffContext) -> HandoffResult:  # pragma: no cover
            raise AssertionError(
                "runner must not be called for a non-handoff exit")

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )

        assert out.final_result.exit_reason == "completed"
        assert len(out.cycles) == 1
        assert out.cycles[0].handoffs == []


# --------------------------------------------------------------------
# Splice atomic-write under a stale .tmp file from a prior crash
# --------------------------------------------------------------------


class TestSpliceTmpFileTolerated:
    """A previous failed splice may have left a ``.tmp`` file in
    the agent workspace (e.g., the host crashed between
    ``mkstemp`` and ``os.replace``).  The current splice must
    succeed regardless.

    :func:`flywheel.session_splice._atomic_write` uses
    :func:`tempfile.mkstemp` with a unique suffix per call, so
    stale tmp files never collide.  This test pins that.
    """

    def test_stale_tmp_file_does_not_block_splice(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """Pre-seed a leftover ``agent_session.jsonl.<random>.tmp``
        in the agent workspace, then run a normal handoff cycle.
        Splice succeeds; the leftover tmp file is irrelevant.
        """
        agent_ws = _agent_ws(workspace)
        agent_ws.mkdir(parents=True, exist_ok=True)
        # Use the same suffix shape as ``_atomic_write`` produces
        # so we're sure we aren't testing under an accidentally
        # different naming convention.
        leftover = (
            agent_ws / "agent_session.jsonl.PRIOR_CRASH.tmp")
        leftover.write_text(
            "leftover from prior crashed splice",
            encoding="utf-8",
        )

        ids = ["toolu_after_tmp"]
        scripts = [
            _multi_pending_script(
                session_id="sess-after-tmp", tool_use_ids=ids),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="real-after-tmp")

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )
        assert out.final_result.exit_reason == "completed"
        spliced = _read_session(workspace)
        assert "real-after-tmp" in spliced
        # Splice MUST NOT have touched the leftover file (it has a
        # different name; only the canonical session file is
        # what splice writes to).
        assert leftover.is_file()
        assert leftover.read_text(encoding="utf-8") == (
            "leftover from prior crashed splice"
        )


# --------------------------------------------------------------------
# Pending-file lifecycle invariants
# --------------------------------------------------------------------


class TestPendingFileLifecycle:
    """Operators inspect ``pending_tool_calls.json`` to know what
    work is outstanding.  These invariants pin when it is
    expected to exist on disk.
    """

    def test_pending_present_during_runner_calls(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """The pending file must be on disk while the runner is
        executing — that's how an external observer can see
        progress mid-cycle.  It is removed only after every
        entry in the batch has been spliced.
        """
        ids = ["toolu_x", "toolu_y"]
        scripts = [
            _multi_pending_script(
                session_id="sess-life", tool_use_ids=ids),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        present_during_runner: list[bool] = []

        def _br(ctx: HandoffContext) -> HandoffResult:
            present_during_runner.append(
                _pending_path(workspace).is_file())
            return HandoffResult(content=f"real-{ctx.tool_use_id}")

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )
        assert present_during_runner == [True, True]
        assert out.final_result.exit_reason == "completed"
        assert not _pending_path(workspace).exists(), (
            "pending file must be removed before the relaunch"
        )

    def test_pending_removed_even_if_loop_terminates_immediately(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """If the cycle's handoffs all resolve and the *next* cycle
        terminates the loop (non-handoff exit), the pending file
        from the resolved cycle still gets removed.  A stuck
        pending file would mislead the operator into thinking
        work was outstanding when the loop has actually finished.
        """
        ids = ["toolu_only"]
        scripts = [
            _multi_pending_script(
                session_id="sess-once", tool_use_ids=ids),
            _CycleScript(exit_reason="completed"),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )
        assert not _pending_path(workspace).exists()

    def test_pending_removed_when_halt_fires_after_splice(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """A halt directive fires after the cycle's splice but
        before the relaunch.  The pending file is *still*
        removed, because the batch fully resolved — the halt
        suppresses the next *launch*, not the splice cleanup.

        Operators reading the workspace after a halt should see
        a clean state: spliced session, no pending file, halts
        recorded in the loop result.
        """
        ids = ["toolu_h"]
        scripts = [
            _multi_pending_script(
                session_id="sess-halt-clean", tool_use_ids=ids),
        ]
        record: list[_LaunchCall] = []
        launch = _make_launch_fn(workspace.path, scripts, record, workspace=workspace)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="real-h")

        out = run_agent_with_handoffs(
            block_runner=_br,
            launch_fn=launch,
            halt_source=lambda: [{"reason": "stop"}],
            **_base_kwargs(workspace),
        )
        assert out.halts == [{"reason": "stop"}]
        assert not _pending_path(workspace).exists()
        # Splice did happen.
        assert "real-h" in _read_session(workspace)


# --------------------------------------------------------------------
# Property-style: re-running on a clean workspace after partial work
# --------------------------------------------------------------------


class TestRedriveAfterFailure:
    """An operator's recovery story: after a host-side failure
    leaves the workspace partially spliced and pending intact,
    re-running ``run_agent_with_handoffs`` against the same
    workspace must Just Work — the fresh container overwrites
    every artifact and the loop processes whatever the new
    container surfaces.

    This is the property the failure-mode table refers to as
    "host restarts cleanly; resume succeeds".
    """

    def test_redrive_after_runner_failure_completes_normally(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """First run: mid-batch runner crash.  Workspace is left
        partially spliced with pending file intact.  Second run
        (a brand new ``run_agent_with_handoffs`` call) launches
        a fresh container that does its own thing; the loop
        processes the fresh container's output and ignores the
        leftover artifacts.

        Models what an operator does after a transient failure:
        relaunch the same pattern, expect normal completion.
        """
        # --- First run: crashes mid-batch ---
        ids_1 = ["toolu_a", "toolu_b"]
        scripts_1 = [_multi_pending_script(
            session_id="sess-1", tool_use_ids=ids_1)]
        record_1: list[_LaunchCall] = []
        launch_1 = _make_launch_fn(
            workspace.path, scripts_1, record_1, workspace=workspace)

        def _br_crashy(ctx: HandoffContext) -> HandoffResult:
            if ctx.index_in_iteration == 0:
                return HandoffResult(content="real-a-from-run-1")
            raise RuntimeError("transient failure")

        with pytest.raises(RuntimeError):
            run_agent_with_handoffs(
                block_runner=_br_crashy,
                launch_fn=launch_1,
                **_base_kwargs(workspace),
            )
        # Pre-condition for the redrive: pending file present,
        # session partially spliced.
        assert _pending_path(workspace).is_file()
        assert "real-a-from-run-1" in _read_session(workspace)

        # --- Second run: fresh container, healthy runner ---
        ids_2 = ["toolu_redrive"]
        scripts_2 = [
            _multi_pending_script(
                session_id="sess-2", tool_use_ids=ids_2),
            _CycleScript(exit_reason="completed"),
        ]
        record_2: list[_LaunchCall] = []
        launch_2 = _make_launch_fn(
            workspace.path, scripts_2, record_2, workspace=workspace)

        runner_2_seen: list[str] = []

        def _br_healthy(ctx: HandoffContext) -> HandoffResult:
            runner_2_seen.append(ctx.tool_use_id)
            return HandoffResult(content=f"real-{ctx.tool_use_id}")

        out = run_agent_with_handoffs(
            block_runner=_br_healthy,
            launch_fn=launch_2,
            **_base_kwargs(workspace),
        )

        # The redrive saw only the new container's pending, never
        # the leftover ones from run 1.
        assert runner_2_seen == ["toolu_redrive"]
        assert out.final_result.exit_reason == "completed"
        spliced = _read_session(workspace)
        # The fresh splice contains the new result; the old
        # spliced content is overwritten because the second
        # container wrote its own JSONL.
        assert "real-toolu_redrive" in spliced
        assert "real-a-from-run-1" not in spliced
        assert not _pending_path(workspace).exists()
