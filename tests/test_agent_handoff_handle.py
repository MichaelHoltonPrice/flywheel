"""Unit tests for the threaded ``HandoffAgentHandle`` and
``launch_agent_with_handoffs`` factory.

Where ``test_agent_handoff.py`` exercises the synchronous loop
function directly, this file exercises the *handle* wrapper that
:class:`flywheel.pattern_runner.PatternRunner` consumes.  The
wrapper runs the loop on a background thread; tests therefore
have to be careful about timing and cleanup, but they remain
fully deterministic by injecting a fake ``launch_fn`` whose
behaviour is fixed in advance — there is no Docker, no model, no
real session JSONL on disk.

What this file covers:
  * Handle satisfies the pattern runner's minimal protocol
    (``is_alive()`` + ``wait()``).
  * ``wait()`` returns the *terminal* :class:`AgentResult` (the
    last cycle's), matching the existing one-shot
    ``AgentHandle.wait`` contract so PatternRunner consumers
    cannot tell the difference between a one-cycle and a
    multi-cycle run.
  * Background thread does in fact terminate after the loop.
  * ``loop_result`` is exposed for callers that need the per-
    cycle chain.
  * Exceptions raised inside the loop propagate to the caller
    on ``wait()``.
  * Calling ``wait()`` more than once is safe (returns or
    re-raises the same outcome).
  * The ``launch_agent_with_handoffs`` factory forwards kwargs
    to the loop unchanged and pre-binds the handoff parameters
    cleanly via :func:`functools.partial`.
  * End-to-end through PatternRunner: a one-role pattern using
    ``launch_agent_with_handoffs`` as ``launch_fn`` runs to
    completion across a real handoff cycle, the runner sees only
    the terminal :class:`AgentResult`, and the per-cycle chain
    is observable on the handle.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from flywheel.agent import AgentBlockConfig, AgentResult
from flywheel.agent_handoff import (
    PENDING_FILE_NAME,
    SESSION_ARTIFACT_NAME,
    HandoffAgentHandle,
    HandoffContext,
    HandoffLoopError,
    HandoffResult,
    launch_agent_with_handoffs,
)
from flywheel.pattern import ContinuousTrigger, Pattern, Role
from flywheel.pattern_runner import PatternRunner

# --------------------------------------------------------------------
# Reused fakes (mirroring test_agent_handoff.py's helpers; kept
# in this file so the two suites can evolve independently).
# --------------------------------------------------------------------


@dataclass
class _FakeWorkspace:
    """Stand-in for :class:`flywheel.workspace.Workspace`."""

    path: Path


@dataclass
class _CycleScript:
    """How one fake launch should behave (see test_agent_handoff)."""

    exit_reason: str = "completed"
    pending: list[dict[str, Any]] = field(default_factory=list)
    write_session: bool = False
    session_jsonl_text: str = ""
    execution_id: str = "exec-0"
    agent_workspace_dir: str = "agent_workspaces/handle_test"
    pre_wait_delay_s: float = 0.0


class _FakeAgentHandle:
    """Stand-in for :class:`flywheel.agent.AgentHandle`."""

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
        if self._script.pre_wait_delay_s > 0:
            time.sleep(self._script.pre_wait_delay_s)

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
    record: list[dict[str, Any]] | None = None,
) -> Callable[..., _FakeAgentHandle]:
    """Build a fake launch_fn that walks ``scripts`` cycle by cycle."""
    iteration = {"i": 0}

    def _fake(**kwargs: Any) -> _FakeAgentHandle:
        if record is not None:
            record.append(kwargs)
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

    See :func:`tests.test_agent_handoff._build_session_jsonl` for
    full shape notes.  Inlined here so the file does not rely on
    test-helper imports across modules.
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
    """Minimal kwargs the loop forwards to ``launch_fn``."""
    return {
        "workspace": workspace,
        "template": object(),
        "project_root": workspace.path,
        "prompt": "do the thing",
        "agent_workspace_dir": "agent_workspaces/handle_test",
    }


def _wait_for_thread_to_die(
    handle: HandoffAgentHandle,
    *,
    timeout_s: float = 5.0,
) -> None:
    """Poll until the handle's background thread exits.

    Raises an :class:`AssertionError` if the thread is still
    alive after ``timeout_s``; that's a deadlock or a runaway
    loop and the test should fail loudly rather than hang the
    suite.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not handle.is_alive():
            return
        time.sleep(0.01)
    raise AssertionError(
        f"handle thread still alive after {timeout_s}s"
    )


# --------------------------------------------------------------------
# Backward-compat: handle around a no-handoff agent.
# --------------------------------------------------------------------


class TestNoHandoffPath:
    """No ``HANDOFF_TOOLS`` configured: handle wraps a single
    launch and ``wait()`` returns its result directly."""

    def test_handle_returns_terminal_agent_result(
        self, workspace: _FakeWorkspace,
    ) -> None:
        scripts = [_CycleScript(
            exit_reason="completed", execution_id="exec-only")]
        launch = _make_launch_fn(workspace.path, scripts)

        handle = HandoffAgentHandle(
            block_runner=None,
            launch_kwargs=_base_kwargs(workspace),
            launch_fn=launch,
        )
        result = handle.wait()

        assert isinstance(result, AgentResult)
        assert result.execution_id == "exec-only"
        assert result.exit_reason == "completed"
        assert handle.loop_result is not None
        assert len(handle.loop_result.cycles) == 1

    def test_thread_terminates_after_completion(
        self, workspace: _FakeWorkspace,
    ) -> None:
        scripts = [_CycleScript(exit_reason="completed")]
        launch = _make_launch_fn(workspace.path, scripts)

        handle = HandoffAgentHandle(
            block_runner=None,
            launch_kwargs=_base_kwargs(workspace),
            launch_fn=launch,
        )
        handle.wait()
        assert not handle.is_alive(), (
            "background thread should be dead after wait() returns"
        )

    def test_is_alive_is_true_during_run(
        self, workspace: _FakeWorkspace,
    ) -> None:
        # Sleep long enough that the test can observe the
        # mid-flight state without racing.
        scripts = [_CycleScript(
            exit_reason="completed", pre_wait_delay_s=0.5)]
        launch = _make_launch_fn(workspace.path, scripts)

        handle = HandoffAgentHandle(
            block_runner=None,
            launch_kwargs=_base_kwargs(workspace),
            launch_fn=launch,
        )
        assert handle.is_alive(), (
            "handle should be alive immediately after construction "
            "while the fake launch is sleeping"
        )
        handle.wait()
        assert not handle.is_alive()


# --------------------------------------------------------------------
# Single-handoff round trip through the handle.
# --------------------------------------------------------------------


class TestHandoffRoundTripThroughHandle:
    """The handle drives a full handoff loop on its background
    thread.  ``wait()`` returns the terminal cycle's result."""

    def test_terminal_result_is_final_cycle_only(
        self, workspace: _FakeWorkspace,
    ) -> None:
        session_jsonl = _build_session_jsonl(
            session_id="sess-handle",
            tool_use_ids=["toolu_h"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_h",
                    "tool_name": "mcp__demo__handoff_me",
                    "tool_input": {"v": 1},
                }],
                write_session=True,
                session_jsonl_text=session_jsonl,
                execution_id="exec-cycle0",
            ),
            _CycleScript(
                exit_reason="completed",
                execution_id="exec-cycle1",
            ),
        ]
        launch = _make_launch_fn(workspace.path, scripts)

        invocations: list[HandoffContext] = []

        def _br(ctx: HandoffContext) -> HandoffResult:
            invocations.append(ctx)
            return HandoffResult(content="REAL_RESULT")

        handle = HandoffAgentHandle(
            block_runner=_br,
            launch_kwargs=_base_kwargs(workspace),
            launch_fn=launch,
        )
        result = handle.wait()

        assert result.execution_id == "exec-cycle1"
        assert result.exit_reason == "completed"
        assert len(invocations) == 1
        assert invocations[0].tool_use_id == "toolu_h"

    def test_loop_result_exposes_full_chain(
        self, workspace: _FakeWorkspace,
    ) -> None:
        session_jsonl = _build_session_jsonl(
            session_id="sess-chain",
            tool_use_ids=["toolu_a", "toolu_b"],
        )
        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[
                    {
                        "tool_use_id": "toolu_a",
                        "tool_name": "mcp__demo__h",
                        "tool_input": {},
                    },
                    {
                        "tool_use_id": "toolu_b",
                        "tool_name": "mcp__demo__h",
                        "tool_input": {},
                    },
                ],
                write_session=True,
                session_jsonl_text=session_jsonl,
                execution_id="exec-multi",
            ),
            _CycleScript(
                exit_reason="completed",
                execution_id="exec-final",
            ),
        ]
        launch = _make_launch_fn(workspace.path, scripts)

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content=ctx.tool_use_id)

        handle = HandoffAgentHandle(
            block_runner=_br,
            launch_kwargs=_base_kwargs(workspace),
            launch_fn=launch,
        )
        handle.wait()
        loop = handle.loop_result
        assert loop is not None
        assert len(loop.cycles) == 2
        assert loop.total_handoffs == 2
        assert loop.cycles[-1].agent_result.execution_id == (
            "exec-final")


# --------------------------------------------------------------------
# Exception propagation and idempotency.
# --------------------------------------------------------------------


class TestExceptionPropagation:
    """Errors from the loop must surface to the caller, not be
    swallowed by the daemon thread."""

    def test_handoff_loop_error_is_reraised_by_wait(
        self, workspace: _FakeWorkspace,
    ) -> None:
        # Handoff requested with no block_runner -> the loop
        # raises HandoffLoopError, which the handle must surface.
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
        launch = _make_launch_fn(workspace.path, scripts)

        handle = HandoffAgentHandle(
            block_runner=None,
            launch_kwargs=_base_kwargs(workspace),
            launch_fn=launch,
        )
        with pytest.raises(HandoffLoopError, match="block_runner"):
            handle.wait()
        assert handle.loop_error is not None
        assert isinstance(handle.loop_error, HandoffLoopError)

    def test_block_runner_exception_is_reraised(
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
        launch = _make_launch_fn(workspace.path, scripts)

        class _BoomError(RuntimeError):
            pass

        def _br(ctx: HandoffContext) -> HandoffResult:
            raise _BoomError("block runner blew up")

        handle = HandoffAgentHandle(
            block_runner=_br,
            launch_kwargs=_base_kwargs(workspace),
            launch_fn=launch,
        )
        with pytest.raises(_BoomError, match="blew up"):
            handle.wait()
        # Loop error must be the original exception, not wrapped.
        assert isinstance(handle.loop_error, _BoomError)


class TestIdempotentWait:
    """Calling ``wait()`` twice should be safe — pattern runner
    code that defensively waits more than once must not blow up."""

    def test_double_wait_returns_same_result(
        self, workspace: _FakeWorkspace,
    ) -> None:
        scripts = [_CycleScript(
            exit_reason="completed", execution_id="exec-once")]
        launch = _make_launch_fn(workspace.path, scripts)

        handle = HandoffAgentHandle(
            block_runner=None,
            launch_kwargs=_base_kwargs(workspace),
            launch_fn=launch,
        )
        first = handle.wait()
        second = handle.wait()
        assert first is second

    def test_double_wait_reraises_same_exception(
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
        launch = _make_launch_fn(workspace.path, scripts)

        handle = HandoffAgentHandle(
            block_runner=None,
            launch_kwargs=_base_kwargs(workspace),
            launch_fn=launch,
        )
        with pytest.raises(HandoffLoopError):
            handle.wait()
        with pytest.raises(HandoffLoopError):
            handle.wait()


# --------------------------------------------------------------------
# launch_agent_with_handoffs factory.
# --------------------------------------------------------------------


class TestLaunchFactory:
    """The factory just delegates to :class:`HandoffAgentHandle`,
    but it has to forward kwargs untouched and play nicely with
    ``functools.partial``-based pre-binding."""

    def test_factory_returns_handle_and_runs_loop(
        self, workspace: _FakeWorkspace,
    ) -> None:
        scripts = [_CycleScript(
            exit_reason="completed", execution_id="exec-factory")]
        launch = _make_launch_fn(workspace.path, scripts)

        handle = launch_agent_with_handoffs(
            block_runner=None,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )
        assert isinstance(handle, HandoffAgentHandle)
        result = handle.wait()
        assert result.execution_id == "exec-factory"

    def test_factory_forwards_halt_source_to_loop(
        self, workspace: _FakeWorkspace,
    ) -> None:
        """``halt_source`` must reach the loop intact: a non-empty
        drain after the first cycle suppresses the relaunch and
        the directives surface on ``loop_result.halts``.
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
                execution_id="exec-halt",
            ),
            _CycleScript(exit_reason="completed"),  # never reached
        ]
        launch = _make_launch_fn(workspace.path, scripts)

        directive = {"reason": "drain-fired"}

        def _drain() -> list[Any]:
            return [directive]

        def _br(ctx: HandoffContext) -> HandoffResult:
            return HandoffResult(content="ok")

        handle = launch_agent_with_handoffs(
            block_runner=_br,
            halt_source=_drain,
            launch_fn=launch,
            **_base_kwargs(workspace),
        )
        handle.wait()

        assert handle.loop_result is not None
        assert handle.loop_result.halts == [directive]
        assert len(handle.loop_result.cycles) == 1

    def test_factory_compatible_with_partial(
        self, workspace: _FakeWorkspace,
    ) -> None:
        # Mirror the documented production wiring pattern:
        # bind block_runner via partial, then pass the bound
        # callable as a launch_fn.
        scripts = [_CycleScript(
            exit_reason="completed", execution_id="exec-partial")]
        launch = _make_launch_fn(workspace.path, scripts)

        invocations: list[HandoffContext] = []

        def _br(ctx: HandoffContext) -> HandoffResult:
            invocations.append(ctx)
            return HandoffResult(content="x")

        bound = partial(
            launch_agent_with_handoffs,
            block_runner=_br,
            launch_fn=launch,
        )

        # Invoking the bound callable with the loop kwargs (the
        # same pattern PatternRunner uses with launch_fn) must
        # produce a handle that respects the bound block_runner.
        handle = bound(**_base_kwargs(workspace))
        result = handle.wait()
        assert result.execution_id == "exec-partial"
        assert invocations == []  # No handoff in this fixture.


# --------------------------------------------------------------------
# End-to-end through PatternRunner.
# --------------------------------------------------------------------


class TestPatternRunnerIntegration:
    """Drive a real :class:`PatternRunner` with the handoff
    factory bound as ``launch_fn``.  This is the canonical
    production wiring; the test asserts that the runner sees a
    handoff cycle as a single agent and collects the terminal
    :class:`AgentResult` into ``results_by_role``."""

    def test_pattern_runner_consumes_handoff_handle_end_to_end(
        self, tmp_path: Path,
    ) -> None:
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "p.txt").write_text(
            "do the thing", encoding="utf-8")

        @dataclass
        class _MiniWS:
            """Just enough Workspace surface for PatternRunner.

            PatternRunner only reads ``workspace.executions`` and
            ``workspace.path`` for the paths exercised by a
            single continuous role.
            """
            path: Path
            executions: dict[str, Any] = field(default_factory=dict)

            def instances_for(self, name: str) -> list[Any]:
                return []

        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        ws = _MiniWS(path=ws_path)

        cfg = AgentBlockConfig(
            workspace=ws,  # type: ignore[arg-type]
            template=object(),  # type: ignore[arg-type]
            project_root=prompts_dir,
            prompt="(unused; loaded from role)",
        )

        pattern = Pattern(
            name="handoff-pat",
            roles=[
                Role(
                    name="solo",
                    trigger=ContinuousTrigger(),
                    prompt="p.txt",
                    cardinality=1,
                ),
            ],
        )

        scripts = [
            _CycleScript(
                exit_reason="tool_handoff",
                pending=[{
                    "tool_use_id": "toolu_pr",
                    "tool_name": "mcp__demo__h",
                    "tool_input": {"v": 1},
                }],
                write_session=True,
                session_jsonl_text=_build_session_jsonl(
                    session_id="sess-pr",
                    tool_use_ids=["toolu_pr"]),
                execution_id="exec-pr-cycle0",
                agent_workspace_dir="agent_workspaces/pr_handle_test",
            ),
            _CycleScript(
                exit_reason="completed",
                execution_id="exec-pr-final",
                agent_workspace_dir="agent_workspaces/pr_handle_test",
            ),
        ]
        captured_handles: list[HandoffAgentHandle] = []
        invocations: list[HandoffContext] = []

        def _br(ctx: HandoffContext) -> HandoffResult:
            invocations.append(ctx)
            return HandoffResult(content="OK")

        # Wrap the fake launch_fn so PatternRunner kwargs flow
        # through the handle factory.  The factory keeps a
        # reference so the test can inspect the underlying
        # loop_result after run() returns.
        fake_launch = _make_launch_fn(ws_path, scripts)

        def _factory(**kwargs: Any) -> HandoffAgentHandle:
            handle = HandoffAgentHandle(
                block_runner=_br,
                launch_kwargs=kwargs,
                launch_fn=fake_launch,
            )
            captured_handles.append(handle)
            return handle

        runner = PatternRunner(
            pattern,
            base_config=cfg,
            launch_fn=_factory,
            poll_interval_s=0.05,
            max_total_runtime_s=10.0,
        )
        result = runner.run()

        assert result.cohorts_by_role == {"solo": 1}
        assert result.agents_launched == 1
        assert len(result.results_by_role["solo"]) == 1
        terminal = result.results_by_role["solo"][0]
        assert terminal.execution_id == "exec-pr-final"
        assert terminal.exit_reason == "completed"

        assert len(captured_handles) == 1
        loop = captured_handles[0].loop_result
        assert loop is not None
        assert len(loop.cycles) == 2
        assert loop.total_handoffs == 1
        assert invocations[0].tool_use_id == "toolu_pr"


# --------------------------------------------------------------------
# Sanity: handle is genuinely a daemon thread (won't block exit).
# --------------------------------------------------------------------


def test_background_thread_is_daemon(tmp_path: Path) -> None:
    """Daemon-ness matters: a non-daemon thread that hung inside
    the loop would prevent test process shutdown.  Pin it."""
    ws = _FakeWorkspace(path=tmp_path)
    scripts = [_CycleScript(exit_reason="completed")]
    launch = _make_launch_fn(tmp_path, scripts)

    handle = HandoffAgentHandle(
        block_runner=None,
        launch_kwargs=_base_kwargs(ws),
        launch_fn=launch,
    )
    try:
        # Inspect before the loop has had a chance to finish.
        thread = handle._thread  # noqa: SLF001 - intentional white-box
        assert isinstance(thread, threading.Thread)
        assert thread.daemon is True
    finally:
        handle.wait()
