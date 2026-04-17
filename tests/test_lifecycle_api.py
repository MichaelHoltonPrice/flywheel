"""Tests for the block-execution lifecycle API.

Exercises ``POST /execution/begin`` and ``POST /execution/end/{id}``
end-to-end against a real ``ExecutionChannel`` HTTP server, and
the ``flywheel.tool_block.BlockChannelClient`` shim that wraps it.

These tests pin down:

- Begin opens a ``running`` ledger row with the right metadata.
- End atomically registers outputs and transitions to
  ``succeeded``.
- End with ``status="failed"`` records the error and registers
  no outputs.
- Optional inputs are skipped when missing; required inputs
  produce a structured ``missing_input`` error.
- Allowed-blocks gating works for the lifecycle API.
- The context manager wraps both calls and propagates body
  exceptions while still closing the ledger row.
- The output_bindings response shape matches the legacy record
  mode for callers that read ``<name>_artifact_id`` keys.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError

import pytest

from flywheel.execution_channel import ExecutionChannel
from flywheel.template import Template
from flywheel.tool_block import (
    BlockChannelClient,
    BlockChannelError,
)
from flywheel.workspace import Workspace

# Template with a couple of blocks suitable for begin/end.
TEMPLATE_YAML = """\
artifacts:
  - name: predictor
    kind: copy
  - name: game_history
    kind: copy
  - name: prediction
    kind: copy
  - name: game_step
    kind: copy
  - name: scratch
    kind: copy

blocks:
  - name: predict
    image: "__record__"
    inputs:
      - name: predictor
        container_path: /input/predictor
      - name: game_history
        container_path: /input/game_history
        optional: true
    outputs:
      - name: prediction
        container_path: /output/prediction

  - name: noop
    image: "__record__"
    inputs: []
    outputs: []
"""


def _make_workspace(tmp_path: Path) -> tuple[Template, Workspace]:
    """Create a template and workspace under tmp_path."""
    tmpl_path = tmp_path / "test.yaml"
    tmpl_path.write_text(TEMPLATE_YAML)
    template = Template.from_yaml(tmpl_path)

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
    return template, ws


def _seed_artifact(
    ws: Workspace, name: str, data: dict,
) -> str:
    """Seed an artifact instance for ``name`` and return its ID."""
    from flywheel.artifact import ArtifactInstance

    aid = ws.generate_artifact_id(name)
    art_dir = ws.path / "artifacts" / aid
    art_dir.mkdir(parents=True)
    (art_dir / f"{name}.json").write_text(json.dumps(data))
    inst = ArtifactInstance(
        id=aid, name=name, kind="copy",
        created_at=datetime.now(UTC), copy_path=aid,
    )
    ws.add_artifact(inst)
    return aid


def _post_json(url: str, body: dict) -> tuple[int, dict]:
    """POST a JSON body and return (status, parsed_response)."""
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


@pytest.fixture()
def channel(tmp_path: Path):
    """Spin up an ExecutionChannel on an ephemeral port."""
    template, ws = _make_workspace(tmp_path)
    chan = ExecutionChannel(template=template, workspace=ws)
    port = chan.start()
    try:
        yield chan, ws, template, port
    finally:
        chan.stop(timeout=5.0)


class TestLifecycleBegin:
    def test_begin_opens_running_row(self, channel):
        chan, ws, _, port = channel
        _seed_artifact(ws, "predictor", {"version": 1})

        status, resp = _post_json(
            f"http://127.0.0.1:{port}/execution/begin",
            {
                "block": "predict",
                "params": {"action_id": 6, "x": 15, "y": 20},
                "caller": {"mcp_server": "arc",
                           "tool": "predict_action"},
                "runner": "inprocess",
            },
        )

        assert status == 200, resp
        assert resp["ok"] is True
        assert resp["execution_id"].startswith("exec_")
        assert resp["input_bindings"]["predictor"]
        assert "input_paths" in resp
        assert "scratch_dir" in resp
        assert "predictor" in resp["input_paths"]

        execution = ws.executions[resp["execution_id"]]
        assert execution.status == "running"
        assert execution.block_name == "predict"
        assert execution.runner == "inprocess"
        assert execution.caller == {
            "mcp_server": "arc", "tool": "predict_action",
        }
        assert execution.params == {
            "action_id": 6, "x": 15, "y": 20,
        }
        assert execution.input_bindings["predictor"] == (
            resp["input_bindings"]["predictor"])

    def test_begin_skips_missing_optional_input(self, channel):
        chan, ws, _, port = channel
        _seed_artifact(ws, "predictor", {"v": 1})
        # game_history is optional and we don't seed it.

        status, resp = _post_json(
            f"http://127.0.0.1:{port}/execution/begin",
            {"block": "predict"},
        )

        assert status == 200, resp
        assert resp["ok"] is True
        assert "predictor" in resp["input_bindings"]
        assert "game_history" not in resp["input_bindings"]

    def test_begin_required_input_missing_returns_error(
            self, channel):
        chan, ws, _, port = channel
        # No predictor instance seeded; required input is missing.

        status, resp = _post_json(
            f"http://127.0.0.1:{port}/execution/begin",
            {"block": "predict"},
        )

        assert status == 200, resp
        assert resp["ok"] is False
        assert resp["error_type"] == "missing_input"
        assert "predictor" in resp["message"]

    def test_begin_unknown_block_returns_error(self, channel):
        _, _, _, port = channel
        status, resp = _post_json(
            f"http://127.0.0.1:{port}/execution/begin",
            {"block": "does_not_exist"},
        )

        assert status == 200, resp
        assert resp["ok"] is False
        assert resp["error_type"] == "unknown_block"

    def test_begin_records_parent_execution(self, channel):
        _, ws, _, port = channel
        _seed_artifact(ws, "predictor", {"v": 1})

        status, resp = _post_json(
            f"http://127.0.0.1:{port}/execution/begin",
            {"block": "predict",
             "parent_execution_id": "exec_parent01"},
        )

        assert resp["ok"]
        assert resp["parent_execution_id"] == "exec_parent01"
        execution = ws.executions[resp["execution_id"]]
        assert execution.parent_execution_id == "exec_parent01"


class TestLifecycleEnd:
    def test_end_succeeds_and_registers_output(self, channel):
        _, ws, _, port = channel
        _seed_artifact(ws, "predictor", {"v": 1})

        _, begin_resp = _post_json(
            f"http://127.0.0.1:{port}/execution/begin",
            {"block": "predict",
             "params": {"action_id": 1}},
        )
        eid = begin_resp["execution_id"]

        status, end_resp = _post_json(
            f"http://127.0.0.1:{port}/execution/end/{eid}",
            {"status": "ok",
             "outputs": {
                 "prediction": {
                     "predicted_state": [[0, 1], [2, 3]],
                     "rationale": "test",
                 }},
             "elapsed_s": 0.123},
        )

        assert status == 200, end_resp
        assert end_resp["ok"] is True
        assert end_resp["status"] == "succeeded"
        aid = end_resp["prediction_artifact_id"]
        assert aid in ws.artifacts

        execution = ws.executions[eid]
        assert execution.status == "succeeded"
        assert execution.elapsed_s == pytest.approx(0.123)
        assert execution.output_bindings["prediction"] == aid
        assert execution.params == {"action_id": 1}

        artifact_file = (
            ws.path / "artifacts" / aid / "prediction.json")
        data = json.loads(artifact_file.read_text())
        assert data["predicted_state"] == [[0, 1], [2, 3]]

    def test_end_failed_records_error_no_outputs(self, channel):
        _, ws, _, port = channel
        _seed_artifact(ws, "predictor", {"v": 1})

        _, begin_resp = _post_json(
            f"http://127.0.0.1:{port}/execution/begin",
            {"block": "predict"},
        )
        eid = begin_resp["execution_id"]

        artifacts_before = len(ws.artifacts)
        status, end_resp = _post_json(
            f"http://127.0.0.1:{port}/execution/end/{eid}",
            {"status": "failed",
             "error": "ValueError: bad state"},
        )

        assert status == 200, end_resp
        assert end_resp["status"] == "failed"
        assert end_resp["error"] == "ValueError: bad state"
        assert len(ws.artifacts) == artifacts_before

        execution = ws.executions[eid]
        assert execution.status == "failed"
        assert execution.error == "ValueError: bad state"
        assert execution.output_bindings == {}

    def test_end_unknown_id_404(self, channel):
        _, _, _, port = channel
        status, resp = _post_json(
            f"http://127.0.0.1:{port}/execution/end/exec_bogus",
            {"status": "ok"},
        )
        assert status == 404
        assert resp["error_type"] == "unknown_execution"

    def test_end_twice_400(self, channel):
        _, ws, _, port = channel
        _seed_artifact(ws, "predictor", {"v": 1})

        _, begin_resp = _post_json(
            f"http://127.0.0.1:{port}/execution/begin",
            {"block": "predict"},
        )
        eid = begin_resp["execution_id"]

        _post_json(
            f"http://127.0.0.1:{port}/execution/end/{eid}",
            {"status": "ok", "outputs": {}},
        )

        status, resp = _post_json(
            f"http://127.0.0.1:{port}/execution/end/{eid}",
            {"status": "ok", "outputs": {}},
        )
        assert status == 400
        assert resp["error_type"] == "not_running"


class TestAllowedBlocksGating:
    def test_block_not_in_allowed_list_rejected(self, tmp_path):
        template, ws = _make_workspace(tmp_path)
        chan = ExecutionChannel(
            template=template, workspace=ws,
            allowed_blocks=["noop"],
        )
        port = chan.start()
        try:
            _seed_artifact(ws, "predictor", {"v": 1})

            status, resp = _post_json(
                f"http://127.0.0.1:{port}/execution/begin",
                {"block": "predict"},
            )
            assert resp["ok"] is False
            assert resp["error_type"] == "block_not_allowed"
        finally:
            chan.stop(timeout=5.0)


class TestExecutionEvents:
    def test_on_execution_fires_for_succeeded(self, tmp_path):
        from flywheel.executor import ExecutionEvent
        template, ws = _make_workspace(tmp_path)

        events: list[ExecutionEvent] = []

        chan = ExecutionChannel(
            template=template, workspace=ws,
            on_execution=events.append,
        )
        port = chan.start()
        try:
            _seed_artifact(ws, "predictor", {"v": 1})

            _, begin_resp = _post_json(
                f"http://127.0.0.1:{port}/execution/begin",
                {"block": "predict", "runner": "inprocess"},
            )
            eid = begin_resp["execution_id"]

            _post_json(
                f"http://127.0.0.1:{port}/execution/end/{eid}",
                {"status": "ok",
                 "outputs": {"prediction": {"x": 1}}},
            )

            assert len(events) == 1
            event = events[0]
            assert event.block_name == "predict"
            assert event.execution_id == eid
            assert event.status == "succeeded"
            assert event.executor_type == "inprocess"
            assert "prediction" in event.output_bindings
        finally:
            chan.stop(timeout=5.0)

    def test_on_execution_fires_for_failed(self, tmp_path):
        from flywheel.executor import ExecutionEvent
        template, ws = _make_workspace(tmp_path)

        events: list[ExecutionEvent] = []

        chan = ExecutionChannel(
            template=template, workspace=ws,
            on_execution=events.append,
        )
        port = chan.start()
        try:
            _seed_artifact(ws, "predictor", {"v": 1})
            _, begin_resp = _post_json(
                f"http://127.0.0.1:{port}/execution/begin",
                {"block": "predict"},
            )
            _post_json(
                (f"http://127.0.0.1:{port}/execution/end/"
                 f"{begin_resp['execution_id']}"),
                {"status": "failed", "error": "boom"},
            )

            assert len(events) == 1
            assert events[0].status == "failed"
        finally:
            chan.stop(timeout=5.0)


class TestBlockChannelClient:
    def test_context_manager_roundtrip(self, channel):
        _, ws, _, port = channel
        _seed_artifact(ws, "predictor", {"v": 1})

        client = BlockChannelClient(
            base_url=f"http://127.0.0.1:{port}")

        with client.begin(
            block="predict",
            params={"action_id": 3, "x": 10, "y": 11},
            caller={"mcp_server": "arc",
                    "tool": "predict_action"},
            runner="inprocess",
        ) as ctx:
            assert ctx.block_name == "predict"
            assert ctx.execution_id.startswith("exec_")
            assert "predictor" in ctx.input_paths
            ctx.set_output("prediction", {"x": 42})

        # After context exit, ledger row should be succeeded.
        execution = ws.executions[ctx.execution_id]
        assert execution.status == "succeeded"
        assert "prediction" in execution.output_bindings
        assert execution.params == {
            "action_id": 3, "x": 10, "y": 11,
        }
        assert execution.runner == "inprocess"

        # The client must echo the channel's output_bindings back
        # onto the context so callers can chain artifact IDs from
        # one block execution into the next without a separate
        # workspace lookup.
        assert ctx.output_bindings == execution.output_bindings
        assert "prediction" in ctx.output_bindings

    def test_body_exception_records_failure_and_propagates(
            self, channel):
        _, ws, _, port = channel
        _seed_artifact(ws, "predictor", {"v": 1})
        client = BlockChannelClient(
            base_url=f"http://127.0.0.1:{port}")

        captured_id: list[str] = []

        with pytest.raises(RuntimeError, match="body fail"):
            with client.begin(block="predict") as ctx:
                captured_id.append(ctx.execution_id)
                raise RuntimeError("body fail")

        eid = captured_id[0]
        execution = ws.executions[eid]
        assert execution.status == "failed"
        assert "body fail" in execution.error
        # Failed body → no output_bindings populated on ctx.
        assert ctx.output_bindings == {}

    def test_set_output_duplicate_raises(self, channel):
        _, ws, _, port = channel
        _seed_artifact(ws, "predictor", {"v": 1})
        client = BlockChannelClient(
            base_url=f"http://127.0.0.1:{port}")

        with client.begin(block="predict") as ctx:
            ctx.set_output("prediction", {"x": 1})
            with pytest.raises(ValueError, match="already set"):
                ctx.set_output("prediction", {"x": 2})

    def test_begin_unknown_block_raises_block_channel_error(
            self, channel):
        _, _, _, port = channel
        client = BlockChannelClient(
            base_url=f"http://127.0.0.1:{port}")

        with pytest.raises(BlockChannelError) as excinfo:
            with client.begin(block="bogus"):
                pass
        assert excinfo.value.error_type == "unknown_block"

    def test_from_env_reads_url(self, monkeypatch):
        monkeypatch.setenv(
            "EXEC_CHANNEL_URL", "http://example:1234")
        client = BlockChannelClient.from_env()
        assert client.base_url == "http://example:1234"

    def test_from_env_falls_back_to_eval_endpoint(
            self, monkeypatch):
        monkeypatch.delenv("EXEC_CHANNEL_URL", raising=False)
        monkeypatch.setenv(
            "EVAL_ENDPOINT", "http://legacy:9999/")
        client = BlockChannelClient.from_env()
        # rstrip("/") is applied.
        assert client.base_url == "http://legacy:9999"

    def test_from_env_missing_raises(self, monkeypatch):
        monkeypatch.delenv("EXEC_CHANNEL_URL", raising=False)
        monkeypatch.delenv("EVAL_ENDPOINT", raising=False)
        with pytest.raises(RuntimeError, match="is set"):
            BlockChannelClient.from_env()


class TestRunningRowPersistence:
    def test_running_row_round_trips_through_save_load(
            self, channel):
        _, ws, _, port = channel
        _seed_artifact(ws, "predictor", {"v": 1})

        _, begin_resp = _post_json(
            f"http://127.0.0.1:{port}/execution/begin",
            {"block": "predict",
             "caller": {"mcp_server": "arc", "tool": "x"}},
        )
        eid = begin_resp["execution_id"]

        # Simulate process restart: reload the workspace.
        reloaded = Workspace.load(ws.path)
        ex = reloaded.executions[eid]
        assert ex.status == "running"
        assert ex.runner is None
        assert ex.caller == {"mcp_server": "arc", "tool": "x"}

    def test_scratch_dir_cleaned_after_success(self, channel):
        _, ws, _, port = channel
        _seed_artifact(ws, "predictor", {"v": 1})

        _, begin_resp = _post_json(
            f"http://127.0.0.1:{port}/execution/begin",
            {"block": "predict"},
        )
        eid = begin_resp["execution_id"]
        scratch = Path(begin_resp["scratch_dir"])
        assert scratch.exists()

        _post_json(
            f"http://127.0.0.1:{port}/execution/end/{eid}",
            {"status": "ok",
             "outputs": {"prediction": {"x": 1}}},
        )

        # Give the channel a moment to clean up.
        for _ in range(10):
            if not scratch.exists():
                break
            time.sleep(0.05)
        assert not scratch.exists()
