"""Tests for the post-execution callback protocol.

Covers four layers:

1. The :class:`flywheel.post_check.HaltDirective` and
   :func:`resolve_dotted_path` helpers (pure unit).
2. :class:`flywheel.blocks.registry.BlockRegistry` parsing
   ``post_check:`` and resolving the dotted path eagerly.
3. :class:`flywheel.execution_channel.ExecutionChannel` invoking
   the callback after ``end_execution`` (success and failure
   paths) and after begin-time rejections (synthetic rows).
4. The ``GET /halt`` endpoint and the channel-side halt queue.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml

from flywheel.blocks.registry import BlockRegistry, load_block_file
from flywheel.execution_channel import ExecutionChannel
from flywheel.post_check import (
    HaltDirective,
    PostCheckContext,
    resolve_dotted_path,
)
from flywheel.template import Template, parse_block_definition
from flywheel.tool_block import BlockChannelClient, BlockChannelError
from flywheel.workspace import Workspace


# ------------------------------------------------------------------
# 1. HaltDirective + resolve_dotted_path unit tests
# ------------------------------------------------------------------

class TestHaltDirective:
    def test_round_trip(self):
        d = HaltDirective(scope="caller", reason="stop please")
        assert HaltDirective.from_dict(d.to_dict()) == d

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValueError, match="scope"):
            HaltDirective.from_dict({"scope": "global", "reason": "x"})

    def test_non_string_reason_rejected(self):
        with pytest.raises(ValueError, match="reason"):
            HaltDirective.from_dict({"scope": "run", "reason": 42})


class TestResolveDottedPath:
    def test_resolves_real_callable(self):
        # ``json.dumps`` is conveniently importable by dotted path.
        func = resolve_dotted_path("json.dumps")
        assert func is json.dumps

    def test_missing_module_raises_clean(self):
        with pytest.raises(ValueError, match="cannot import module"):
            resolve_dotted_path("definitely_not_a_real_module.func")

    def test_missing_attribute_raises_clean(self):
        with pytest.raises(ValueError, match="no attribute"):
            resolve_dotted_path("json.no_such_attr_anywhere")

    def test_not_callable_raises_clean(self):
        with pytest.raises(ValueError, match="not callable"):
            resolve_dotted_path("json.__doc__")

    def test_empty_path_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            resolve_dotted_path("")

    def test_path_without_dot_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            resolve_dotted_path("nodot")


# ------------------------------------------------------------------
# 2. Registry parsing of post_check:
# ------------------------------------------------------------------

# Module-level callable used by registry tests below.  Lives at
# ``tests.test_post_check._noop_check`` so the dotted-path
# resolver can find it.
_received: list[PostCheckContext] = []


def _noop_check(ctx: PostCheckContext) -> HaltDirective | None:
    _received.append(ctx)
    return None


def _always_halt_caller(
    ctx: PostCheckContext,
) -> HaltDirective | None:
    return HaltDirective(
        scope="caller",
        reason=f"halt: block={ctx.block} status={ctx.status}",
    )


def _always_halt_run(
    ctx: PostCheckContext,
) -> HaltDirective | None:
    return HaltDirective(scope="run", reason="run-wide halt")


def _broken_check(ctx: PostCheckContext) -> HaltDirective | None:
    raise RuntimeError("check itself is broken")


def _wrong_return_type(
    ctx: PostCheckContext,
) -> HaltDirective | None:
    return "not a HaltDirective"  # type: ignore[return-value]


class TestRegistryPostCheckParsing:
    def test_block_yaml_with_post_check_resolves_callable(
        self, tmp_path: Path,
    ):
        block_yaml = (
            "name: noop\n"
            "runner: lifecycle\n"
            "runner_justification: test\n"
            "inputs: []\n"
            "outputs: []\n"
            "post_check: tests.test_post_check._noop_check\n"
        )
        path = tmp_path / "noop.yaml"
        path.write_text(block_yaml, encoding="utf-8")

        registry = BlockRegistry.from_files([path])
        assert "noop" in registry
        assert registry.get("noop").post_check == (
            "tests.test_post_check._noop_check")
        # Resolved eagerly:
        assert registry.post_check_for("noop") is _noop_check

    def test_block_without_post_check_has_none(
        self, tmp_path: Path,
    ):
        block_yaml = (
            "name: bare\n"
            "runner: lifecycle\n"
            "runner_justification: test\n"
            "inputs: []\n"
            "outputs: []\n"
        )
        path = tmp_path / "bare.yaml"
        path.write_text(block_yaml, encoding="utf-8")

        registry = BlockRegistry.from_files([path])
        assert registry.get("bare").post_check is None
        assert registry.post_check_for("bare") is None

    def test_invalid_dotted_path_fails_at_load(
        self, tmp_path: Path,
    ):
        block_yaml = (
            "name: bad\n"
            "runner: lifecycle\n"
            "runner_justification: test\n"
            "inputs: []\n"
            "outputs: []\n"
            "post_check: nonexistent.module.func\n"
        )
        path = tmp_path / "bad.yaml"
        path.write_text(block_yaml, encoding="utf-8")

        with pytest.raises(ValueError, match="cannot import module"):
            BlockRegistry.from_files([path])

    def test_non_string_post_check_rejected(
        self, tmp_path: Path,
    ):
        block_yaml = (
            "name: bad\n"
            "runner: lifecycle\n"
            "runner_justification: test\n"
            "inputs: []\n"
            "outputs: []\n"
            "post_check: [a, b, c]\n"
        )
        path = tmp_path / "bad.yaml"
        path.write_text(block_yaml, encoding="utf-8")

        with pytest.raises(ValueError, match="dotted Python path"):
            load_block_file(path)


# ------------------------------------------------------------------
# 3. ExecutionChannel post_check wiring (integration)
# ------------------------------------------------------------------

TEMPLATE_YAML = """\
artifacts:
  - name: out
    kind: copy

blocks:
  - lc
"""

LC_BLOCK_YAML = """\
name: lc
runner: lifecycle
runner_justification: "Tool-triggered lifecycle block for tests."
inputs: []
outputs:
  - name: out
"""


def _make_workspace_and_channel(
    tmp_path: Path,
    *,
    post_checks: dict | None = None,
):
    """Build a template, workspace, and started channel."""
    registry = BlockRegistry(blocks={
        "lc": parse_block_definition(yaml.safe_load(LC_BLOCK_YAML)),
    })
    tmpl_path = tmp_path / "test.yaml"
    tmpl_path.write_text(TEMPLATE_YAML)
    template = Template.from_yaml(tmpl_path, block_registry=registry)

    ws_path = tmp_path / "workspace"
    ws_path.mkdir()
    (ws_path / "artifacts").mkdir()
    ws = Workspace(
        name="test",
        path=ws_path,
        template_name="test",
        created_at=datetime.now(UTC),
        artifact_declarations={
            a.name: a.kind for a in template.artifacts
        },
        artifacts={},
    )
    ws.save()

    chan = ExecutionChannel(
        template=template,
        workspace=ws,
        host="127.0.0.1",
        post_checks=post_checks,
    )
    chan.start()
    return chan, ws, template


def _post_json(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


def _get_json(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5.0) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


class TestChannelPostCheck:
    def test_no_post_check_no_change(self, tmp_path):
        chan, ws, _ = _make_workspace_and_channel(tmp_path)
        try:
            client = BlockChannelClient(base_url=chan.url)
            with client.begin("lc") as ctx:
                ctx.set_output("out", {"v": 1})

            # Exactly one row, succeeded, no halt directive.
            (eid, ex), = ws.executions.items()
            assert ex.status == "succeeded"
            assert ex.halt_directive is None
            assert ex.post_check_error is None
            assert ex.synthetic is False
            assert chan.halt_directives() == []
        finally:
            chan.stop(timeout=5.0)

    def test_post_check_fires_on_success(self, tmp_path):
        _received.clear()
        chan, ws, _ = _make_workspace_and_channel(
            tmp_path,
            post_checks={"lc": _noop_check},
        )
        try:
            client = BlockChannelClient(base_url=chan.url)
            with client.begin("lc") as ctx:
                ctx.set_output("out", {"v": 42})

            assert len(_received) == 1
            ctx_in = _received[0]
            assert ctx_in.block == "lc"
            assert ctx_in.status == "succeeded"
            assert ctx_in.outputs.get("out") == {"v": 42}
            assert ctx_in.error is None
            assert ctx_in.synthetic is False
        finally:
            chan.stop(timeout=5.0)

    def test_post_check_fires_on_body_failure(self, tmp_path):
        _received.clear()
        chan, ws, _ = _make_workspace_and_channel(
            tmp_path,
            post_checks={"lc": _noop_check},
        )
        try:
            client = BlockChannelClient(base_url=chan.url)
            with pytest.raises(RuntimeError):
                with client.begin("lc"):
                    raise RuntimeError("boom")

            assert len(_received) == 1
            ctx_in = _received[0]
            assert ctx_in.status == "failed"
            assert "boom" in (ctx_in.error or "")
        finally:
            chan.stop(timeout=5.0)

    def test_halt_directive_persisted_on_row(self, tmp_path):
        chan, ws, _ = _make_workspace_and_channel(
            tmp_path,
            post_checks={"lc": _always_halt_caller},
        )
        try:
            client = BlockChannelClient(base_url=chan.url)
            with client.begin("lc") as ctx:
                ctx.set_output("out", {"v": 1})

            (eid, ex), = ws.executions.items()
            assert ex.halt_directive == {
                "scope": "caller",
                "reason": "halt: block=lc status=succeeded",
            }
            # And the channel queue picked it up.
            queue = chan.halt_directives()
            assert len(queue) == 1
            assert queue[0]["scope"] == "caller"
            assert queue[0]["execution_id"] == eid
        finally:
            chan.stop(timeout=5.0)

    def test_check_exception_recorded_not_raised(self, tmp_path):
        chan, ws, _ = _make_workspace_and_channel(
            tmp_path,
            post_checks={"lc": _broken_check},
        )
        try:
            client = BlockChannelClient(base_url=chan.url)
            # Body succeeds even though check raises.
            with client.begin("lc") as ctx:
                ctx.set_output("out", {"v": 1})

            (eid, ex), = ws.executions.items()
            assert ex.status == "succeeded"
            assert ex.halt_directive is None
            assert ex.post_check_error is not None
            assert "RuntimeError" in ex.post_check_error
            assert "check itself is broken" in ex.post_check_error
        finally:
            chan.stop(timeout=5.0)

    def test_check_wrong_return_type_recorded_not_raised(
        self, tmp_path,
    ):
        chan, ws, _ = _make_workspace_and_channel(
            tmp_path,
            post_checks={"lc": _wrong_return_type},
        )
        try:
            client = BlockChannelClient(base_url=chan.url)
            with client.begin("lc") as ctx:
                ctx.set_output("out", {"v": 1})

            (eid, ex), = ws.executions.items()
            assert ex.status == "succeeded"
            assert ex.halt_directive is None
            assert ex.post_check_error is not None
            assert "expected HaltDirective" in ex.post_check_error
        finally:
            chan.stop(timeout=5.0)


# ------------------------------------------------------------------
# 4. Synthetic rows for begin-time rejections
# ------------------------------------------------------------------

class TestSyntheticRows:
    def test_unknown_block_creates_synthetic_row(self, tmp_path):
        _received.clear()
        chan, ws, _ = _make_workspace_and_channel(
            tmp_path,
            # No post_check on the unknown block, but the channel
            # should still create the synthetic row.
            post_checks=None,
        )
        try:
            status, resp = _post_json(
                f"{chan.url}/execution/begin",
                {
                    "block": "ghost",
                    "caller": {"mcp_server": "x", "tool": "y"},
                    "params": {"k": "v"},
                },
            )
            assert status == 200
            assert resp["ok"] is False
            assert resp["error_type"] == "unknown_block"
            assert resp["synthetic"] is True
            assert resp["execution_id"].startswith("exec_")

            (eid, ex), = ws.executions.items()
            assert eid == resp["execution_id"]
            assert ex.status == "failed"
            assert ex.synthetic is True
            assert ex.block_name == "ghost"
            assert "unknown_block" in (ex.error or "")
            # Caller and params preserved on the synthetic row.
            assert ex.caller == {"mcp_server": "x", "tool": "y"}
            assert ex.params == {"k": "v"}
        finally:
            chan.stop(timeout=5.0)

    def test_synthetic_row_runs_post_check_if_configured(
        self, tmp_path,
    ):
        # A post_check on a known block name still fires when the
        # channel rejects the begin (e.g., not in allowed_blocks).
        registry = BlockRegistry(blocks={
            "lc": parse_block_definition(
                yaml.safe_load(LC_BLOCK_YAML)),
        })
        tmpl_path = tmp_path / "test.yaml"
        tmpl_path.write_text(TEMPLATE_YAML)
        template = Template.from_yaml(
            tmpl_path, block_registry=registry)

        ws_path = tmp_path / "workspace"
        ws_path.mkdir()
        (ws_path / "artifacts").mkdir()
        ws = Workspace(
            name="test", path=ws_path, template_name="test",
            created_at=datetime.now(UTC),
            artifact_declarations={
                a.name: a.kind for a in template.artifacts},
            artifacts={},
        )
        ws.save()

        chan = ExecutionChannel(
            template=template,
            workspace=ws,
            host="127.0.0.1",
            allowed_blocks=["other_block_only"],
            post_checks={"lc": _always_halt_run},
        )
        chan.start()
        try:
            status, resp = _post_json(
                f"{chan.url}/execution/begin",
                {"block": "lc"},
            )
            assert resp["ok"] is False
            assert resp["error_type"] == "block_not_allowed"
            assert resp["synthetic"] is True

            (eid, ex), = ws.executions.items()
            assert ex.synthetic is True
            assert ex.halt_directive == {
                "scope": "run", "reason": "run-wide halt",
            }
            queue = chan.halt_directives()
            assert len(queue) == 1
            assert queue[0]["scope"] == "run"
        finally:
            chan.stop(timeout=5.0)


# ------------------------------------------------------------------
# 5. /halt endpoint
# ------------------------------------------------------------------

class TestHaltEndpoint:
    def test_empty_queue_returns_no_halts(self, tmp_path):
        chan, _, _ = _make_workspace_and_channel(tmp_path)
        try:
            status, resp = _get_json(f"{chan.url}/halt")
            assert status == 200
            assert resp["ok"] is True
            assert resp["halts"] == []
        finally:
            chan.stop(timeout=5.0)

    def test_caller_halt_filtered_by_execution_id(self, tmp_path):
        chan, ws, _ = _make_workspace_and_channel(
            tmp_path,
            post_checks={"lc": _always_halt_caller},
        )
        try:
            client = BlockChannelClient(
                base_url=chan.url,
                parent_execution_id="exec_parent_xyz",
            )
            with client.begin("lc") as ctx:
                ctx.set_output("out", {"v": 1})

            # No filter: the directive shows up.
            status, resp = _get_json(f"{chan.url}/halt")
            assert len(resp["halts"]) == 1

            # Filter for the matching parent: still visible.
            status, resp = _get_json(
                f"{chan.url}/halt?execution_id=exec_parent_xyz")
            assert len(resp["halts"]) == 1
            assert resp["halts"][0]["scope"] == "caller"

            # Filter for a non-matching parent: hidden.
            status, resp = _get_json(
                f"{chan.url}/halt?execution_id=exec_someone_else")
            assert resp["halts"] == []
        finally:
            chan.stop(timeout=5.0)

    def test_run_halt_visible_to_every_runner(self, tmp_path):
        chan, ws, _ = _make_workspace_and_channel(
            tmp_path,
            post_checks={"lc": _always_halt_run},
        )
        try:
            client = BlockChannelClient(
                base_url=chan.url,
                parent_execution_id="exec_a",
            )
            with client.begin("lc") as ctx:
                ctx.set_output("out", {"v": 1})

            # Filter by an unrelated parent: a run-wide halt is
            # visible to everyone.
            status, resp = _get_json(
                f"{chan.url}/halt?execution_id=exec_b")
            assert len(resp["halts"]) == 1
            assert resp["halts"][0]["scope"] == "run"
        finally:
            chan.stop(timeout=5.0)


# ------------------------------------------------------------------
# 6. Workspace round-trip of the new BlockExecution fields
# ------------------------------------------------------------------

class TestWorkspaceRoundTrip:
    def test_new_fields_persist_across_save_load(self, tmp_path):
        chan, ws, _ = _make_workspace_and_channel(
            tmp_path,
            post_checks={"lc": _always_halt_caller},
        )
        try:
            client = BlockChannelClient(base_url=chan.url)
            with client.begin("lc") as ctx:
                ctx.set_output("out", {"v": 1})
            (eid, _), = ws.executions.items()
        finally:
            chan.stop(timeout=5.0)

        # Reload from disk.
        reloaded = Workspace.load(ws.path)
        ex = reloaded.executions[eid]
        assert ex.halt_directive == {
            "scope": "caller",
            "reason": "halt: block=lc status=succeeded",
        }
        assert ex.post_check_error is None
        assert ex.synthetic is False
