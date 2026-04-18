"""Live-API B1 demonstration: PreToolUse → splice → resume cycle.

This is the experiment that makes or breaks the full-stop nested
block design.  It runs a real Haiku-backed Claude Agent SDK
session, intercepts a custom MCP tool call with ``PreToolUse``,
captures the tool's intent, denies the call so the SDK records a
synthetic deny ``tool_result``, then splices in a real result and
resumes a fresh SDK session from the modified JSONL.  We assert
that the model perceives the spliced result as the genuine
outcome of its original tool call.

If this test passes, the B1 go/no-go is "go" and the rest of the
campaign (B2–B8) proceeds.  If it fails, the failure mode tells
us which of the assumed JSONL invariants is wrong; we update
``plans/full-stop-state-contract.md`` and the splice module
before any other phase starts.

The full design contract this test validates:
``plans/full-stop-state-contract.md``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    create_sdk_mcp_server,
    tool,
)

from flywheel.session_splice import (
    find_pending_deny_tool_use_ids,
    splice_tool_result,
)

pytestmark = pytest.mark.live_api

# Haiku 4.5 is the cheapest current model fast enough to make
# this test bearable in a tight loop.  If the slug is wrong on
# your account, override via the FLYWHEEL_TEST_HAIKU_MODEL env
# var; the test will surface the SDK error verbatim.
DEFAULT_HAIKU = "claude-haiku-4-5"

DENY_MARKER = "handoff_to_flywheel"


@pytest.fixture()
def haiku_model() -> str:
    return os.environ.get("FLYWHEEL_TEST_HAIKU_MODEL", DEFAULT_HAIKU)


@pytest.fixture()
def isolated_workspace(tmp_path: Path) -> Path:
    """A clean cwd for the SDK session under test.

    Using a unique tmp directory keeps the SDK's session file
    out of any project's real ``~/.claude/projects/<cwd>/``
    subdirectory and makes cleanup easy.
    """
    ws = tmp_path / f"ws-{uuid.uuid4().hex[:8]}"
    ws.mkdir()
    return ws


def _sdk_session_jsonl(session_id: str, cwd: Path) -> Path:
    """Locate the SDK's session JSONL for a given session + cwd.

    Mirrors the encoding used by the agent runner
    (``_encode_cwd`` -> ``-`` for path separators) so we can find
    the file the SDK just wrote.
    """
    encoded = str(cwd).replace(os.sep, "-").replace("/", "-")
    encoded = encoded.replace(":", "-")
    return (
        Path.home() / ".claude" / "projects"
        / encoded / f"{session_id}.jsonl"
    )


def _find_session_jsonl(session_id: str) -> Path:
    """Find a session JSONL anywhere under ``~/.claude/projects``.

    Falls back to a glob search when the encoded-cwd derivation
    in ``_sdk_session_jsonl`` doesn't quite match what the SDK
    wrote (the encoding scheme has varied across SDK versions).
    """
    projects = Path.home() / ".claude" / "projects"
    if not projects.exists():
        raise FileNotFoundError(
            f"SDK projects directory missing: {projects}")
    matches = list(projects.glob(f"*/{session_id}.jsonl"))
    if not matches:
        raise FileNotFoundError(
            f"no session JSONL for session_id={session_id} "
            f"under {projects}")
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple session JSONLs for {session_id}: "
            f"{matches}")
    return matches[0]


async def _run_until_deny(
    *,
    haiku_model: str,
    workspace: Path,
    captured: dict[str, Any],
) -> str:
    """Drive an SDK session that calls one tool and gets denied.

    Returns the session_id.  Populates ``captured`` with the
    PreToolUse hook's view of the tool call (tool_use_id,
    tool_name, tool_input).
    """

    @tool(
        "double_it",
        "Doubles a number and returns the result.",
        {"value": float},
    )
    async def double_it(args: dict[str, Any]) -> dict[str, Any]:
        # The hook denies before this body runs in the happy path.
        # If it ever runs, we still return something valid so the
        # SDK doesn't throw.
        return {"content": [
            {"type": "text",
             "text": f"local result: {args['value'] * 2}"},
        ]}

    server = create_sdk_mcp_server(
        name="splicedemo", tools=[double_it])

    async def pre_tool_use_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        if (
            input_data.get("tool_name", "").endswith("double_it")
            and tool_use_id is not None
            and "tool_use_id" not in captured
        ):
            captured["tool_use_id"] = tool_use_id
            captured["tool_name"] = input_data.get("tool_name")
            captured["tool_input"] = input_data.get("tool_input")
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"permission denied: {DENY_MARKER}"
                    ),
                },
            }
        return {}

    options = ClaudeAgentOptions(
        cwd=str(workspace),
        model=haiku_model,
        permission_mode="bypassPermissions",
        mcp_servers={"splicedemo": server},
        allowed_tools=["mcp__splicedemo__double_it"],
        hooks={
            "PreToolUse": [
                HookMatcher(
                    matcher="mcp__splicedemo__double_it",
                    hooks=[pre_tool_use_hook],
                ),
            ],
        },
        max_turns=4,
    )

    session_id = ""
    async with ClaudeSDKClient(options=options) as client:
        await client.query(
            "Use the double_it tool to double the number 7. "
            "After it returns, tell me the doubled value as a "
            "single sentence."
        )
        async for message in client.receive_response():
            sid = getattr(message, "session_id", None)
            if sid:
                session_id = sid

    if not session_id:
        raise RuntimeError(
            "SDK never reported a session_id; cannot locate JSONL")

    return session_id


async def _resume_and_collect(
    *,
    haiku_model: str,
    workspace: Path,
    spliced_jsonl: Path,
    session_id: str,
) -> str:
    """Resume the spliced session and return the final assistant text.

    Copies the spliced JSONL into the SDK projects dir for the
    SAME session UUID, then opens a fresh ``ClaudeSDKClient``
    with ``options.resume = session_id`` and asks the model to
    confirm the tool's result.
    """

    @tool(
        "double_it",
        "Doubles a number and returns the result.",
        {"value": float},
    )
    async def double_it(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [
            {"type": "text",
             "text": f"local result: {args['value'] * 2}"},
        ]}

    server = create_sdk_mcp_server(
        name="splicedemo", tools=[double_it])

    sdk_path = _find_session_jsonl(session_id)
    shutil.copy2(spliced_jsonl, sdk_path)

    options = ClaudeAgentOptions(
        cwd=str(workspace),
        model=haiku_model,
        permission_mode="bypassPermissions",
        mcp_servers={"splicedemo": server},
        allowed_tools=["mcp__splicedemo__double_it"],
        resume=session_id,
        max_turns=2,
    )

    final_text = ""
    async with ClaudeSDKClient(options=options) as client:
        await client.query(
            "Based on the tool result you just received, what is "
            "the doubled value?  Reply with a single short "
            "sentence."
        )
        async for message in client.receive_response():
            cls = type(message).__name__
            if cls == "AssistantMessage":
                for block in getattr(message, "content", []) or []:
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        final_text += text + "\n"
    return final_text.strip()


def test_splice_roundtrip_with_haiku(
    haiku_model: str,
    isolated_workspace: Path,
    tmp_path: Path,
) -> None:
    """End-to-end: deny → capture → splice → resume → verify.

    The expected flow:

    1.  Haiku is asked to call ``double_it(7)``.
    2.  The PreToolUse hook denies and captures the
        ``tool_use_id``.  The SDK records a deny tool_result.
    3.  We locate the SDK session JSONL, validate the
        ``tool_use_id`` is reachable via
        ``find_pending_deny_tool_use_ids`` (the splice module's
        only schema-level assumption being live-checked here),
        and splice in a real result of ``"42"``.
    4.  We resume from the spliced JSONL and ask Haiku what the
        doubled value is.  Haiku must answer "42" — proving the
        model received the spliced result, not the deny stub.
    """
    captured: dict[str, Any] = {}

    session_id = asyncio.run(_run_until_deny(
        haiku_model=haiku_model,
        workspace=isolated_workspace,
        captured=captured,
    ))

    assert "tool_use_id" in captured, (
        "PreToolUse hook never fired with our tool name; check "
        "that Haiku actually invoked mcp__splicedemo__double_it"
    )
    tool_use_id = captured["tool_use_id"]

    sdk_jsonl = _find_session_jsonl(session_id)
    pending = find_pending_deny_tool_use_ids(
        sdk_jsonl, deny_marker=DENY_MARKER)
    assert tool_use_id in pending, (
        f"splice module did not see the deny marker for "
        f"tool_use_id={tool_use_id} in {sdk_jsonl}; the JSONL "
        f"shape may differ from "
        f"plans/full-stop-state-contract.md.  Pending IDs found: "
        f"{pending}"
    )

    spliced_path = tmp_path / "spliced_session.jsonl"
    shutil.copy2(sdk_jsonl, spliced_path)

    splice_tool_result(
        spliced_path,
        tool_use_id=tool_use_id,
        tool_result_content="The doubled value is 42.",
    )

    spliced_text = spliced_path.read_text(encoding="utf-8")
    assert "The doubled value is 42." in spliced_text
    assert (
        find_pending_deny_tool_use_ids(
            spliced_path, deny_marker=DENY_MARKER)
        == []
    ), "splice did not remove the deny marker for the target ID"

    final_text = asyncio.run(_resume_and_collect(
        haiku_model=haiku_model,
        workspace=isolated_workspace,
        spliced_jsonl=spliced_path,
        session_id=session_id,
    ))

    assert "42" in final_text, (
        f"Resumed model did not see the spliced result.  "
        f"Final assistant text: {final_text!r}"
    )


def test_session_jsonl_invariants(
    haiku_model: str,
    isolated_workspace: Path,
    tmp_path: Path,
) -> None:
    """Validate the shape assumptions that the splice depends on.

    Pure observation: runs Haiku with a denied tool call, then
    inspects the resulting JSONL and asserts every invariant
    listed in ``plans/full-stop-state-contract.md``:

    - First non-empty line carries a top-level ``sessionId``.
    - Every tool_use block has an ``id``; every tool_result
      block has a ``tool_use_id``.
    - The captured ``tool_use_id`` appears in exactly one
      tool_result block.
    - The deny tool_result lives inside a message-envelope's
      ``content`` array (located at either ``message.content``
      or bare ``content``).
    """
    captured: dict[str, Any] = {}
    session_id = asyncio.run(_run_until_deny(
        haiku_model=haiku_model,
        workspace=isolated_workspace,
        captured=captured,
    ))
    tool_use_id = captured.get("tool_use_id")
    assert tool_use_id, (
        "PreToolUse hook never fired; cannot validate invariants")

    sdk_jsonl = _find_session_jsonl(session_id)
    captured_jsonl = tmp_path / f"observed_{session_id}.jsonl"
    shutil.copy2(sdk_jsonl, captured_jsonl)

    lines = [
        line for line in captured_jsonl.read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines, "captured JSONL is empty"

    first = json.loads(lines[0])
    assert "sessionId" in first, (
        f"first JSONL line lacks top-level sessionId; keys are "
        f"{sorted(first.keys())}"
    )

    tool_use_ids: list[str] = []
    tool_result_uses_for_target: list[str] = []
    envelope_shapes_seen: set[str] = set()

    for raw in lines:
        obj = json.loads(raw)
        for shape, content in (
            ("message.content",
                obj.get("message", {}).get("content")
                if isinstance(obj.get("message"), dict)
                else None),
            ("content", obj.get("content")),
        ):
            if not isinstance(content, list):
                continue
            envelope_shapes_seen.add(shape)
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use":
                    bid = block.get("id")
                    assert isinstance(bid, str), (
                        f"tool_use block missing string id: "
                        f"{block}"
                    )
                    tool_use_ids.append(bid)
                elif btype == "tool_result":
                    bid = block.get("tool_use_id")
                    assert isinstance(bid, str), (
                        f"tool_result block missing string "
                        f"tool_use_id: {block}"
                    )
                    if bid == tool_use_id:
                        tool_result_uses_for_target.append(bid)

    assert tool_use_id in tool_use_ids, (
        f"captured tool_use_id={tool_use_id} not present in "
        f"any tool_use block; SDK may surface a different ID "
        f"than the hook does"
    )
    assert len(tool_result_uses_for_target) == 1, (
        f"expected exactly one deny tool_result for target ID; "
        f"got {len(tool_result_uses_for_target)}"
    )
    assert envelope_shapes_seen, (
        "no tool_use/tool_result envelope shapes detected; the "
        "JSONL schema may have shifted entirely")

    state_contract_path = tmp_path / "observed_envelope_shapes.txt"
    state_contract_path.write_text(
        "Envelope shapes containing tool_use/tool_result blocks:\n"
        + "\n".join(f" - {s}" for s in sorted(envelope_shapes_seen)),
        encoding="utf-8",
    )
