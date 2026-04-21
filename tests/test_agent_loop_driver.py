"""Unit tests for ``flywheel.agent_loop_driver.run_with_handoffs``.

These tests substitute a tiny Python script for the real
``docker run`` command.  The fake "agent" reads a script of
per-iteration actions (write a session JSONL, write a pending
file, write a state file) from environment variables the test
sets, then exits.  This lets us drive the host-side loop
deterministically and assert:

- Single-handoff turns (siblings == 1) round-trip and splice once.
- Multi-handoff turns (siblings > 1) round-trip and splice once
  per pending entry, in order.
- The driver iterates ``pending_tool_calls.json`` (``schema_version``
  2) and accepts the documented backwards-compatible shapes.
- The terminating iteration produces an empty handoffs list.
- Malformed pending files raise ``HandoffLoopError`` with the
  intended messages.

A small wrapper script absorbs the ``-e KEY=VAL`` argv flags that
``run_with_handoffs`` injects (real ``docker run`` consumes them
itself; here we promote them into ``os.environ`` for the fake
agent before exec-ing it).  This keeps the unit tests honest
about the driver contract -- they exercise the same env-flag
splicing the real flow uses, just without an actual container.

No live model, no Docker, no network.  Runs in <2s.
"""

from __future__ import annotations

import json
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any

import pytest

from flywheel.agent_loop_driver import (
    HandoffContext,
    HandoffLoopError,
    HandoffResult,
    LoopResult,
    _normalize_pending,
    run_with_handoffs,
)

# --------------------------------------------------------------------
# Test scaffolding: a fake "agent" Python script that the driver
# launches in place of docker run.
# --------------------------------------------------------------------

# The fake agent reads ``FAKE_AGENT_PLAN`` (a path) for a JSON file
# describing what to do this iteration.  The plan file is rewritten
# by the test harness between iterations so we can stage per-iter
# behavior.  A plan looks like:
#
#   {
#     "iterations": [
#       {
#         "session_jsonl": "<stringified jsonl>",
#         "pending": {... pending file contents ...} | null,
#         "state": {"status": "tool_handoff", ...},
#         "echo_resume_env": true,        # (optional) drop env value
#                                         # for the test to inspect
#       },
#       ...
#     ]
#   }
#
# A counter file tracks which iteration is current.
_FAKE_AGENT_SOURCE = textwrap.dedent("""\
    import json
    import os
    import sys
    from pathlib import Path

    plan_path = Path(os.environ["FAKE_AGENT_PLAN"])
    counter_path = Path(os.environ["FAKE_AGENT_COUNTER"])
    workspace = Path(os.environ["FAKE_AGENT_WORKSPACE"])

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    iterations = plan["iterations"]

    if counter_path.is_file():
        idx = int(counter_path.read_text(encoding="utf-8").strip())
    else:
        idx = 0
    counter_path.write_text(str(idx + 1), encoding="utf-8")

    if idx >= len(iterations):
        # Past the planned end -> emit a terminating state and exit.
        (workspace / ".agent_state.json").write_text(
            json.dumps({"status": "complete", "session_id": "synth"}),
            encoding="utf-8",
        )
        sys.exit(0)

    spec = iterations[idx]

    if spec.get("echo_resume_env"):
        observed = workspace / "_observed_env.json"
        observed.write_text(
            json.dumps({
                "RESUME_SESSION_FILE": os.environ.get(
                    "RESUME_SESSION_FILE", ""),
                "iteration_index": idx,
            }),
            encoding="utf-8",
        )

    if spec.get("session_jsonl") is not None:
        (workspace / "agent_session.jsonl").write_text(
            spec["session_jsonl"], encoding="utf-8")

    if spec.get("pending") is not None:
        (workspace / "pending_tool_calls.json").write_text(
            json.dumps(spec["pending"]), encoding="utf-8")

    (workspace / ".agent_state.json").write_text(
        json.dumps(spec.get("state", {})), encoding="utf-8")

    # Consume stdin so the driver's blocking write doesn't hang.
    try:
        sys.stdin.read()
    except Exception:
        pass

    sys.exit(spec.get("exit_code", 0))
""")


# Wrapper that consumes ``-e KEY=VAL`` argv pairs the driver
# injects (mimicking ``docker run -e KEY=VAL``) and exec-s the
# trailing script with those vars promoted into the environment.
_WRAPPER_SOURCE = textwrap.dedent("""\
    import os
    import runpy
    import sys

    args = sys.argv[1:]
    out_args = []
    i = 0
    while i < len(args):
        if args[i] == "-e" and i + 1 < len(args):
            kv = args[i + 1]
            if "=" in kv:
                k, _, v = kv.partition("=")
                os.environ[k] = v
            i += 2
            continue
        out_args.append(args[i])
        i += 1

    if not out_args:
        sys.stderr.write("wrapper: no script to run\\n")
        sys.exit(2)

    target = out_args[0]
    sys.argv = out_args
    runpy.run_path(target, run_name="__main__")
""")


@pytest.fixture()
def fake_agent_script(tmp_path: Path) -> Path:
    """Materialize the fake agent script and return its path."""
    p = tmp_path / "fake_agent.py"
    p.write_text(_FAKE_AGENT_SOURCE, encoding="utf-8")
    return p


@pytest.fixture()
def wrapper_script(tmp_path: Path) -> Path:
    """Materialize the env-flag-stripping wrapper script."""
    p = tmp_path / "wrapper.py"
    p.write_text(_WRAPPER_SOURCE, encoding="utf-8")
    return p


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / f"ws-{uuid.uuid4().hex[:8]}"
    ws.mkdir()
    return ws


def _stage(
    *,
    wrapper_script: Path,
    fake_agent_script: Path,
    workspace: Path,
    plan: dict[str, Any],
) -> tuple[list[str], Path, Path]:
    """Write the plan and return (docker_command, plan_path, counter).

    The returned command obeys the driver's contract: the LAST
    element is the "image" the driver appends ``-e KEY=VAL`` flags
    in front of.  Here that "image" is our fake agent script path,
    and the wrapper preceding it strips those env flags into
    ``os.environ`` before exec-ing the script.
    """
    plan_path = workspace.parent / f"plan-{uuid.uuid4().hex[:6]}.json"
    counter_path = workspace.parent / f"counter-{uuid.uuid4().hex[:6]}.txt"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    cmd = [
        sys.executable,
        str(wrapper_script),
        str(fake_agent_script),
    ]
    return cmd, plan_path, counter_path


def _build_session_jsonl(
    *,
    session_id: str,
    tool_use_ids_in_one_turn: list[str],
    deny_marker: str = "handoff_to_flywheel",
) -> str:
    """Build a synthetic session JSONL the splice can rewrite.

    The first line carries ``sessionId`` (per the splice contract).
    Then an assistant envelope with all ``tool_use_ids_in_one_turn``
    as parallel tool_use blocks, then a user envelope with one deny
    ``tool_result`` per ``tool_use_id``.
    """
    lines: list[dict[str, Any]] = [
        {"type": "summary", "sessionId": session_id, "version": 1},
    ]
    assistant_content = [
        {
            "type": "tool_use",
            "id": tuid,
            "name": "mcp__demo__do_something",
            "input": {"k": tuid},
        }
        for tuid in tool_use_ids_in_one_turn
    ]
    lines.append({
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": assistant_content,
        },
    })
    user_content = [
        {
            "type": "tool_result",
            "tool_use_id": tuid,
            "content": [
                {
                    "type": "text",
                    "text": f"permission denied: {deny_marker}",
                }
            ],
            "is_error": True,
        }
        for tuid in tool_use_ids_in_one_turn
    ]
    lines.append({
        "type": "user",
        "message": {"role": "user", "content": user_content},
    })
    return "\n".join(json.dumps(obj) for obj in lines) + "\n"


# --------------------------------------------------------------------
# _normalize_pending: schema-shape acceptance.
# --------------------------------------------------------------------


class TestNormalizePending:
    """Unit tests for the on-disk schema decoder."""

    def test_canonical_schema_v2(self) -> None:
        doc = {
            "schema_version": 2,
            "session_id": "abc",
            "pending": [
                {"tool_use_id": "u1", "tool_name": "t1"},
                {"tool_use_id": "u2", "tool_name": "t2"},
            ],
        }
        assert _normalize_pending(doc) == doc["pending"]

    def test_bare_list_accepted(self) -> None:
        items = [
            {"tool_use_id": "u1", "tool_name": "t1"},
            {"tool_use_id": "u2", "tool_name": "t2"},
        ]
        assert _normalize_pending(items) == items  # type: ignore[arg-type]

    def test_legacy_v1_single_dict_accepted(self) -> None:
        legacy = {
            "schema_version": 1,
            "tool_use_id": "u1",
            "tool_name": "t1",
            "tool_input": {"x": 1},
        }
        assert _normalize_pending(legacy) == [legacy]

    def test_empty_doc_returns_empty(self) -> None:
        assert _normalize_pending({}) == []

    def test_pending_must_be_list(self) -> None:
        assert _normalize_pending({"pending": "not-a-list"}) == []

    def test_non_dict_entries_filtered(self) -> None:
        doc = {
            "schema_version": 2,
            "pending": [
                {"tool_use_id": "u1", "tool_name": "t1"},
                "junk",
                None,
                {"tool_use_id": "u2", "tool_name": "t2"},
            ],
        }
        out = _normalize_pending(doc)
        assert len(out) == 2
        assert [p["tool_use_id"] for p in out] == ["u1", "u2"]


# --------------------------------------------------------------------
# Single-handoff turns (siblings == 1).
# --------------------------------------------------------------------


def test_single_handoff_round_trip(
    wrapper_script: Path,
    fake_agent_script: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One pending entry: one block_runner call, one splice line."""
    session_jsonl = _build_session_jsonl(
        session_id="sess-single",
        tool_use_ids_in_one_turn=["toolu_solo"],
    )
    plan = {
        "iterations": [
            {
                "session_jsonl": session_jsonl,
                "pending": {
                    "schema_version": 2,
                    "session_id": "sess-single",
                    "pending": [
                        {
                            "tool_use_id": "toolu_solo",
                            "tool_name": "mcp__demo__do_something",
                            "tool_input": {"k": "toolu_solo"},
                        }
                    ],
                },
                "state": {
                    "status": "tool_handoff",
                    "session_id": "sess-single",
                    "reason": "mcp__demo__do_something",
                },
                "echo_resume_env": False,
            },
            {
                # Terminating iteration: no pending, no session
                # rewrite, just a clean state.
                "session_jsonl": None,
                "pending": None,
                "state": {
                    "status": "complete",
                    "session_id": "sess-single",
                },
                "echo_resume_env": True,
            },
        ]
    }
    cmd, plan_path, counter_path = _stage(
        wrapper_script=wrapper_script,
        fake_agent_script=fake_agent_script,
        workspace=workspace,
        plan=plan,
    )
    monkeypatch.setenv("FAKE_AGENT_PLAN", str(plan_path))
    monkeypatch.setenv("FAKE_AGENT_COUNTER", str(counter_path))
    monkeypatch.setenv("FAKE_AGENT_WORKSPACE", str(workspace))

    invocations: list[HandoffContext] = []

    def block_runner(ctx: HandoffContext) -> HandoffResult:
        invocations.append(ctx)
        return HandoffResult(
            content=f"REAL_RESULT_FOR_{ctx.tool_use_id}",
            is_error=False,
        )

    result: LoopResult = run_with_handoffs(
        workspace=workspace,
        docker_command=cmd,
        prompt="initial-prompt",
        block_runner=block_runner,
        max_iterations=4,
    )

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.tool_use_id == "toolu_solo"
    assert invocation.tool_name == "mcp__demo__do_something"
    assert invocation.tool_input == {"k": "toolu_solo"}
    assert invocation.iteration == 0
    assert invocation.index_in_iteration == 0
    assert invocation.siblings == 1

    assert len(result.iterations) == 2
    handoff_iter = result.iterations[0]
    assert handoff_iter.state.get("status") == "tool_handoff"
    assert len(handoff_iter.handoffs) == 1
    handoff_record = handoff_iter.handoffs[0]
    assert handoff_record["tool_use_id"] == "toolu_solo"
    assert handoff_record["siblings"] == 1
    assert handoff_record["index_in_iteration"] == 0
    assert handoff_record["splice_line"] is not None
    assert result.iterations[-1].handoffs == []
    assert result.iterations[-1].state.get("status") == "complete"

    # Pending file was cleaned up before terminating iteration.
    assert not (workspace / "pending_tool_calls.json").exists()

    # Spliced session contains the real result text and no longer
    # contains the deny marker for that tool_use_id.
    spliced = (workspace / "agent_session.jsonl").read_text(
        encoding="utf-8")
    assert "REAL_RESULT_FOR_toolu_solo" in spliced
    # The deny marker may still appear in untouched lines, but the
    # tool_use_id's line should now carry the real text.
    assert "permission denied: handoff_to_flywheel" not in [
        block["text"]
        for line in spliced.splitlines()
        for obj in [json.loads(line)]
        if isinstance(obj.get("message"), dict)
        for block in (obj["message"].get("content") or [])
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id") == "toolu_solo"
        for inner in block.get("content") or []
        if isinstance(inner, dict) and (text := inner.get("text"))
        for _ in [block]
        for block in [{"text": text}]
    ]

    # Resume env var was set on iteration 1 and pointed at the
    # spliced session inside the container's mount.
    observed = json.loads(
        (workspace / "_observed_env.json").read_text(
            encoding="utf-8"))
    assert observed["iteration_index"] == 1
    assert observed["RESUME_SESSION_FILE"] == (
        "/scratch/agent_session.jsonl")


# --------------------------------------------------------------------
# Multi-handoff turns (siblings > 1) — the path the live test can
# only exercise opportunistically.
# --------------------------------------------------------------------


def test_multi_handoff_round_trip_in_one_cycle(
    wrapper_script: Path,
    fake_agent_script: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three parallel tool_uses: three serial block_runner calls,
    three splices, one stop/restart cycle, in emission order."""
    pending_entries = [
        {
            "tool_use_id": f"toolu_p{i}",
            "tool_name": "mcp__demo__do_something",
            "tool_input": {"idx": i},
        }
        for i in range(3)
    ]
    session_jsonl = _build_session_jsonl(
        session_id="sess-multi",
        tool_use_ids_in_one_turn=[p["tool_use_id"] for p in pending_entries],
    )
    plan = {
        "iterations": [
            {
                "session_jsonl": session_jsonl,
                "pending": {
                    "schema_version": 2,
                    "session_id": "sess-multi",
                    "pending": pending_entries,
                },
                "state": {
                    "status": "tool_handoff",
                    "session_id": "sess-multi",
                    "reason": (
                        "mcp__demo__do_something (+2 more)"
                    ),
                },
            },
            {
                "session_jsonl": None,
                "pending": None,
                "state": {
                    "status": "complete",
                    "session_id": "sess-multi",
                },
            },
        ]
    }
    cmd, plan_path, counter_path = _stage(
        wrapper_script=wrapper_script,
        fake_agent_script=fake_agent_script,
        workspace=workspace,
        plan=plan,
    )
    monkeypatch.setenv("FAKE_AGENT_PLAN", str(plan_path))
    monkeypatch.setenv("FAKE_AGENT_COUNTER", str(counter_path))
    monkeypatch.setenv("FAKE_AGENT_WORKSPACE", str(workspace))

    invocations: list[HandoffContext] = []

    def block_runner(ctx: HandoffContext) -> HandoffResult:
        invocations.append(ctx)
        return HandoffResult(
            content=f"RESULT[{ctx.index_in_iteration}]:{ctx.tool_use_id}",
            is_error=False,
        )

    result = run_with_handoffs(
        workspace=workspace,
        docker_command=cmd,
        prompt="initial-prompt",
        block_runner=block_runner,
        max_iterations=4,
    )

    # block_runner called once per pending entry, in emission order.
    assert [c.tool_use_id for c in invocations] == [
        "toolu_p0", "toolu_p1", "toolu_p2",
    ]
    assert all(c.iteration == 0 for c in invocations)
    assert [c.index_in_iteration for c in invocations] == [0, 1, 2]
    assert all(c.siblings == 3 for c in invocations)
    # tool_input flowed through verbatim from the pending file.
    assert [c.tool_input["idx"] for c in invocations] == [0, 1, 2]

    # Driver record mirrors the same shape.
    assert len(result.iterations) == 2
    handoff_iter = result.iterations[0]
    assert len(handoff_iter.handoffs) == 3
    assert [h["tool_use_id"] for h in handoff_iter.handoffs] == [
        "toolu_p0", "toolu_p1", "toolu_p2",
    ]
    assert all(h["siblings"] == 3 for h in handoff_iter.handoffs)
    assert [h["index_in_iteration"] for h in handoff_iter.handoffs] == [
        0, 1, 2,
    ]
    splice_lines = [h["splice_line"] for h in handoff_iter.handoffs]
    assert all(isinstance(s, int) for s in splice_lines), splice_lines
    # All three splices target the same envelope line (since we put
    # all three deny tool_results in one user message).
    assert len(set(splice_lines)) == 1
    assert result.iterations[-1].handoffs == []

    # Each splice landed: the spliced JSONL contains every real
    # result text and no longer carries any deny marker.
    spliced = (workspace / "agent_session.jsonl").read_text(
        encoding="utf-8")
    for entry in pending_entries:
        tuid = entry["tool_use_id"]
        assert "RESULT[" in spliced
        assert tuid in spliced
    # The deny marker should be entirely gone (every tool_result
    # in the user envelope was replaced).
    assert "handoff_to_flywheel" not in spliced

    # Pending file removed before resume.
    assert not (workspace / "pending_tool_calls.json").exists()


# --------------------------------------------------------------------
# Schema decoder backwards-compat at the loop boundary.
# --------------------------------------------------------------------


def test_legacy_v1_single_dict_pending_still_runs(
    wrapper_script: Path,
    fake_agent_script: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare legacy v1 single-dict pending file still drives one handoff."""
    session_jsonl = _build_session_jsonl(
        session_id="sess-legacy",
        tool_use_ids_in_one_turn=["toolu_legacy"],
    )
    plan = {
        "iterations": [
            {
                "session_jsonl": session_jsonl,
                "pending": {
                    "schema_version": 1,
                    "tool_use_id": "toolu_legacy",
                    "tool_name": "mcp__demo__do_something",
                    "tool_input": {"k": "v"},
                    "session_id": "sess-legacy",
                    "captured_at": "2026-04-17T15:23:01+00:00",
                },
                "state": {
                    "status": "tool_handoff",
                    "session_id": "sess-legacy",
                },
            },
            {
                "session_jsonl": None,
                "pending": None,
                "state": {
                    "status": "complete",
                    "session_id": "sess-legacy",
                },
            },
        ]
    }
    cmd, plan_path, counter_path = _stage(
        wrapper_script=wrapper_script,
        fake_agent_script=fake_agent_script,
        workspace=workspace,
        plan=plan,
    )
    monkeypatch.setenv("FAKE_AGENT_PLAN", str(plan_path))
    monkeypatch.setenv("FAKE_AGENT_COUNTER", str(counter_path))
    monkeypatch.setenv("FAKE_AGENT_WORKSPACE", str(workspace))

    invocations: list[HandoffContext] = []

    def block_runner(ctx: HandoffContext) -> HandoffResult:
        invocations.append(ctx)
        return HandoffResult(content="LEGACY_RESULT", is_error=False)

    result = run_with_handoffs(
        workspace=workspace,
        docker_command=cmd,
        prompt="initial-prompt",
        block_runner=block_runner,
        max_iterations=4,
    )

    assert len(invocations) == 1
    assert invocations[0].tool_use_id == "toolu_legacy"
    assert invocations[0].siblings == 1
    assert result.iterations[0].handoffs[0]["tool_use_id"] == "toolu_legacy"


# --------------------------------------------------------------------
# Failure modes.
# --------------------------------------------------------------------


def test_tool_handoff_with_empty_pending_raises(
    wrapper_script: Path,
    fake_agent_script: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exit with status=tool_handoff but no pending entries is
    treated as a contract violation.  The driver refuses to proceed."""
    plan = {
        "iterations": [
            {
                "session_jsonl": _build_session_jsonl(
                    session_id="sess-bad",
                    tool_use_ids_in_one_turn=["toolu_x"],
                ),
                "pending": {
                    "schema_version": 2,
                    "session_id": "sess-bad",
                    "pending": [],
                },
                "state": {
                    "status": "tool_handoff",
                    "session_id": "sess-bad",
                },
            },
        ]
    }
    cmd, plan_path, counter_path = _stage(
        wrapper_script=wrapper_script,
        fake_agent_script=fake_agent_script,
        workspace=workspace,
        plan=plan,
    )
    monkeypatch.setenv("FAKE_AGENT_PLAN", str(plan_path))
    monkeypatch.setenv("FAKE_AGENT_COUNTER", str(counter_path))
    monkeypatch.setenv("FAKE_AGENT_WORKSPACE", str(workspace))

    def block_runner(ctx: HandoffContext) -> HandoffResult:  # pragma: no cover
        raise AssertionError("should never be called")

    with pytest.raises(HandoffLoopError, match="no tool calls"):
        run_with_handoffs(
            workspace=workspace,
            docker_command=cmd,
            prompt="initial",
            block_runner=block_runner,
            max_iterations=2,
        )


def test_pending_entry_missing_tool_use_id_raises(
    wrapper_script: Path,
    fake_agent_script: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {
        "iterations": [
            {
                "session_jsonl": _build_session_jsonl(
                    session_id="sess-bad",
                    tool_use_ids_in_one_turn=["toolu_x"],
                ),
                "pending": {
                    "schema_version": 2,
                    "session_id": "sess-bad",
                    "pending": [
                        {
                            "tool_name": "mcp__demo__do_something",
                            "tool_input": {},
                        },
                    ],
                },
                "state": {
                    "status": "tool_handoff",
                    "session_id": "sess-bad",
                },
            },
        ]
    }
    cmd, plan_path, counter_path = _stage(
        wrapper_script=wrapper_script,
        fake_agent_script=fake_agent_script,
        workspace=workspace,
        plan=plan,
    )
    monkeypatch.setenv("FAKE_AGENT_PLAN", str(plan_path))
    monkeypatch.setenv("FAKE_AGENT_COUNTER", str(counter_path))
    monkeypatch.setenv("FAKE_AGENT_WORKSPACE", str(workspace))

    def block_runner(ctx: HandoffContext) -> HandoffResult:  # pragma: no cover
        raise AssertionError("should never be called")

    with pytest.raises(HandoffLoopError, match="no tool_use_id"):
        run_with_handoffs(
            workspace=workspace,
            docker_command=cmd,
            prompt="initial",
            block_runner=block_runner,
            max_iterations=2,
        )


def test_max_iterations_guard_trips_on_endless_handoff(
    wrapper_script: Path,
    fake_agent_script: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every iteration re-issues a handoff, the cap fires."""
    one_iter_spec = {
        "session_jsonl": _build_session_jsonl(
            session_id="sess-loop",
            tool_use_ids_in_one_turn=["toolu_loop"],
        ),
        "pending": {
            "schema_version": 2,
            "session_id": "sess-loop",
            "pending": [
                {
                    "tool_use_id": "toolu_loop",
                    "tool_name": "mcp__demo__do_something",
                    "tool_input": {},
                },
            ],
        },
        "state": {
            "status": "tool_handoff",
            "session_id": "sess-loop",
        },
    }
    # Each iteration uses a different tool_use_id so the splice
    # always finds a deny block to rewrite (no stale-id error).
    iterations_spec = []
    for i in range(5):
        spec = json.loads(json.dumps(one_iter_spec))
        spec["session_jsonl"] = _build_session_jsonl(
            session_id="sess-loop",
            tool_use_ids_in_one_turn=[f"toolu_loop{i}"],
        )
        spec["pending"]["pending"][0]["tool_use_id"] = f"toolu_loop{i}"
        iterations_spec.append(spec)
    plan = {"iterations": iterations_spec}
    cmd, plan_path, counter_path = _stage(
        wrapper_script=wrapper_script,
        fake_agent_script=fake_agent_script,
        workspace=workspace,
        plan=plan,
    )
    monkeypatch.setenv("FAKE_AGENT_PLAN", str(plan_path))
    monkeypatch.setenv("FAKE_AGENT_COUNTER", str(counter_path))
    monkeypatch.setenv("FAKE_AGENT_WORKSPACE", str(workspace))

    def block_runner(ctx: HandoffContext) -> HandoffResult:
        return HandoffResult(content="ok", is_error=False)

    with pytest.raises(HandoffLoopError, match="max_iterations"):
        run_with_handoffs(
            workspace=workspace,
            docker_command=cmd,
            prompt="initial",
            block_runner=block_runner,
            max_iterations=3,
        )


def test_resume_without_session_artifact_raises(
    wrapper_script: Path,
    fake_agent_script: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If iteration 0 ends with tool_handoff but never wrote the
    session artifact, the driver refuses to splice."""
    plan = {
        "iterations": [
            {
                # Note: session_jsonl is None -> agent never wrote it.
                "session_jsonl": None,
                "pending": {
                    "schema_version": 2,
                    "session_id": "sess-nosession",
                    "pending": [
                        {
                            "tool_use_id": "toolu_x",
                            "tool_name": "mcp__demo__do_something",
                        },
                    ],
                },
                "state": {
                    "status": "tool_handoff",
                    "session_id": "sess-nosession",
                },
            },
        ]
    }
    cmd, plan_path, counter_path = _stage(
        wrapper_script=wrapper_script,
        fake_agent_script=fake_agent_script,
        workspace=workspace,
        plan=plan,
    )
    monkeypatch.setenv("FAKE_AGENT_PLAN", str(plan_path))
    monkeypatch.setenv("FAKE_AGENT_COUNTER", str(counter_path))
    monkeypatch.setenv("FAKE_AGENT_WORKSPACE", str(workspace))

    def block_runner(ctx: HandoffContext) -> HandoffResult:  # pragma: no cover
        raise AssertionError("should never be called")

    with pytest.raises(HandoffLoopError, match="no session"):
        run_with_handoffs(
            workspace=workspace,
            docker_command=cmd,
            prompt="initial",
            block_runner=block_runner,
            max_iterations=2,
        )
