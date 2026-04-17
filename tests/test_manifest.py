"""Tests for the tool-to-block manifest loader and channel enforcement.

Covers Phase 3a of the block-execution refactor:

- ``ToolBlockManifest.from_file`` parses the documented schema.
- ``validate_against_registry`` rejects unknown block references.
- ``load_manifests`` rejects duplicate ``mcp_server`` names.
- ``build_invocation_table`` produces the channel's lookup table.
- ``ExecutionChannel`` rejects ``/execution/begin`` calls whose
  ``caller`` violates the manifest, and accepts everything else.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from flywheel.artifact import ArtifactInstance
from flywheel.blocks import (
    BlockRegistry,
    ToolBlockManifest,
    build_invocation_table,
    load_manifests,
    validate_against_registry,
)
from flywheel.execution_channel import ExecutionChannel
from flywheel.template import Template
from flywheel.workspace import Workspace


def _seed_artifact(
    ws: Workspace, name: str, contents: str,
) -> str:
    """Seed a copy artifact instance and return its ID."""
    aid = ws.generate_artifact_id(name)
    art_dir = ws.path / "artifacts" / aid
    art_dir.mkdir(parents=True)
    (art_dir / f"{name}.py").write_text(contents)
    inst = ArtifactInstance(
        id=aid,
        name=name,
        kind="copy",
        created_at=datetime.now(UTC),
        copy_path=aid,
    )
    ws.add_artifact(inst)
    ws.save()
    return aid


# A two-block registry: one container, one inprocess.
TRAIN_BLOCK = {
    "name": "train",
    "image": "train:latest",
    "inputs": [{"name": "engine", "container_path": "/input/engine"}],
    "outputs": [
        {"name": "checkpoint", "container_path": "/output/checkpoint"}
    ],
}

PREDICT_BLOCK = {
    "name": "predict",
    "runner": "inprocess",
    "runner_justification": "Lightweight Python eval.",
    "inputs": [
        {"name": "predictor", "container_path": "/input/predictor"}
    ],
    "outputs": [
        {"name": "prediction", "container_path": "/output/prediction"}
    ],
    "implementation": {
        "python_module": "mypkg.blocks.predict_block",
        "entry": "run",
    },
}


def _write_yaml(path: Path, body: dict) -> None:
    path.write_text(yaml.safe_dump(body, sort_keys=False))


@pytest.fixture()
def registry(tmp_path: Path) -> BlockRegistry:
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    _write_yaml(blocks / "train.yaml", TRAIN_BLOCK)
    _write_yaml(blocks / "predict.yaml", PREDICT_BLOCK)
    return BlockRegistry.from_directory(blocks)


# ---------------- manifest loading ---------------- #


class TestManifestParse:
    def test_minimal_valid_manifest(self, tmp_path: Path):
        f = tmp_path / "arc.yaml"
        _write_yaml(f, {
            "mcp_server": "arc",
            "tools": {
                "predict_action": {
                    "block": "predict",
                    "args": {"action": "action_id"},
                    "server_state": {"_last_frame": "source_state"},
                },
                "take_action": {"block": "take_action"},
            },
        })
        m = ToolBlockManifest.from_file(f)
        assert m.mcp_server == "arc"
        assert m.source == f
        assert sorted(m.tools) == ["predict_action", "take_action"]
        binding = m.tools["predict_action"]
        assert binding.block == "predict"
        assert binding.args == {"action": "action_id"}
        assert binding.server_state == {
            "_last_frame": "source_state"}
        # Empty maps default cleanly.
        assert m.tools["take_action"].args == {}
        assert m.tools["take_action"].server_state == {}

    def test_get_and_contains(self, tmp_path: Path):
        f = tmp_path / "arc.yaml"
        _write_yaml(f, {
            "mcp_server": "arc",
            "tools": {"take_action": {"block": "take_action"}},
        })
        m = ToolBlockManifest.from_file(f)
        assert "take_action" in m
        assert "predict_action" not in m
        assert m.get("take_action").block == "take_action"
        assert m.get("missing") is None

    def test_missing_mcp_server_rejected(self, tmp_path: Path):
        f = tmp_path / "bad.yaml"
        _write_yaml(f, {"tools": {}})
        with pytest.raises(
                ValueError, match="missing required 'mcp_server'"):
            ToolBlockManifest.from_file(f)

    def test_empty_mcp_server_rejected(self, tmp_path: Path):
        f = tmp_path / "bad.yaml"
        _write_yaml(f, {"mcp_server": "", "tools": {}})
        with pytest.raises(
                ValueError, match="must be a non-empty string"):
            ToolBlockManifest.from_file(f)

    def test_tool_missing_block_rejected(self, tmp_path: Path):
        f = tmp_path / "bad.yaml"
        _write_yaml(f, {
            "mcp_server": "arc",
            "tools": {"foo": {"args": {}}},
        })
        with pytest.raises(
                ValueError, match="missing required 'block'"):
            ToolBlockManifest.from_file(f)

    def test_args_must_be_string_to_string(self, tmp_path: Path):
        f = tmp_path / "bad.yaml"
        _write_yaml(f, {
            "mcp_server": "arc",
            "tools": {
                "foo": {"block": "predict", "args": {"a": 1}}
            },
        })
        with pytest.raises(
                ValueError, match="must be string.string"):
            ToolBlockManifest.from_file(f)

    def test_empty_file_rejected(self, tmp_path: Path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        with pytest.raises(ValueError, match="empty"):
            ToolBlockManifest.from_file(f)

    def test_top_level_list_rejected(self, tmp_path: Path):
        f = tmp_path / "bad.yaml"
        f.write_text("- a\n- b\n")
        with pytest.raises(ValueError, match="mapping"):
            ToolBlockManifest.from_file(f)


# ---------------- registry validation ---------------- #


class TestValidateAgainstRegistry:
    def test_passes_when_all_blocks_known(
            self, tmp_path: Path, registry: BlockRegistry):
        f = tmp_path / "arc.yaml"
        _write_yaml(f, {
            "mcp_server": "arc",
            "tools": {"predict_action": {"block": "predict"}},
        })
        m = ToolBlockManifest.from_file(f)
        validate_against_registry(m, registry)  # no raise

    def test_rejects_unknown_block(
            self, tmp_path: Path, registry: BlockRegistry):
        f = tmp_path / "arc.yaml"
        _write_yaml(f, {
            "mcp_server": "arc",
            "tools": {"foo": {"block": "ghost"}},
        })
        m = ToolBlockManifest.from_file(f)
        with pytest.raises(
                ValueError, match="unknown block 'ghost'"):
            validate_against_registry(m, registry)


class TestLoadManifests:
    def test_loads_multiple(
            self, tmp_path: Path, registry: BlockRegistry):
        a = tmp_path / "arc.yaml"
        b = tmp_path / "engine.yaml"
        _write_yaml(a, {
            "mcp_server": "arc",
            "tools": {"predict_action": {"block": "predict"}},
        })
        _write_yaml(b, {
            "mcp_server": "engine",
            "tools": {"do_train": {"block": "train"}},
        })
        manifests = load_manifests([a, b], registry=registry)
        assert sorted(m.mcp_server for m in manifests) == [
            "arc", "engine"]

    def test_duplicate_mcp_server_rejected(
            self, tmp_path: Path):
        a = tmp_path / "arc.yaml"
        b = tmp_path / "arc2.yaml"
        body = {"mcp_server": "arc", "tools": {}}
        _write_yaml(a, body)
        _write_yaml(b, body)
        with pytest.raises(ValueError, match="Duplicate manifest"):
            load_manifests([a, b])

    def test_validation_optional(self, tmp_path: Path):
        # No registry → no block-existence check; loads cleanly
        # even with a phantom block name.
        f = tmp_path / "arc.yaml"
        _write_yaml(f, {
            "mcp_server": "arc",
            "tools": {"foo": {"block": "ghost"}},
        })
        manifests = load_manifests([f])
        assert manifests[0].tools["foo"].block == "ghost"


class TestBuildInvocationTable:
    def test_flat_lookup_keys(self, tmp_path: Path):
        a = tmp_path / "arc.yaml"
        _write_yaml(a, {
            "mcp_server": "arc",
            "tools": {
                "predict_action": {"block": "predict"},
                "take_action": {"block": "take_action"},
            },
        })
        manifests = load_manifests([a])
        table = build_invocation_table(manifests)
        assert table[("arc", "predict_action")] == "predict"
        assert table[("arc", "take_action")] == "take_action"
        assert ("arc", "missing") not in table


# ---------------- channel enforcement ---------------- #


# Template referencing the registry blocks.
TEMPLATE_YAML = """\
artifacts:
  - name: engine
    kind: copy
  - name: checkpoint
    kind: copy
  - name: predictor
    kind: copy
  - name: prediction
    kind: copy

blocks:
  - train
  - predict
"""


@pytest.fixture()
def channel_setup(tmp_path: Path, registry: BlockRegistry):
    """Boot an ExecutionChannel with a single 'arc' manifest.

    The arc manifest binds tool ``predict_action`` to block
    ``predict``.  No binding for ``take_action`` (so a request
    naming it from the arc server is a manifest violation).

    Yields ``(channel_url, workspace, registry, manifest_file)``.
    """
    tmpl_path = tmp_path / "test_tmpl.yaml"
    tmpl_path.write_text(TEMPLATE_YAML)
    template = Template.from_yaml(
        tmpl_path, block_registry=registry)

    ws_path = tmp_path / "ws"
    ws_path.mkdir()
    (ws_path / "artifacts").mkdir()
    workspace = Workspace(
        name="test-ws",
        path=ws_path,
        template_name="test_tmpl",
        created_at=datetime.now(UTC),
        artifact_declarations={
            a.name: a.kind for a in template.artifacts},
        artifacts={},
    )
    workspace.save()

    # Pre-register a predictor instance so /execution/begin can
    # resolve the predict block's input.
    _seed_artifact(workspace, "predictor", "# stub\n")

    manifest_file = tmp_path / "arc.yaml"
    _write_yaml(manifest_file, {
        "mcp_server": "arc",
        "tools": {
            "predict_action": {
                "block": "predict",
                "args": {"action": "action_id"},
            },
        },
    })
    manifests = load_manifests(
        [manifest_file], registry=registry)

    channel = ExecutionChannel(
        template=template,
        workspace=workspace,
        host="127.0.0.1",
        manifests=manifests,
    )
    port = channel.start()
    base = f"http://127.0.0.1:{port}"
    # Give the server a beat to be ready.
    time.sleep(0.05)
    try:
        yield base, workspace, registry, manifest_file
    finally:
        channel.stop()


def _post(url: str, body: dict, *, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TestChannelManifestEnforcement:
    def test_matching_caller_succeeds(self, channel_setup):
        base, _ws, _reg, _mf = channel_setup
        resp = _post(f"{base}/execution/begin", {
            "block": "predict",
            "params": {"action_id": 2},
            "caller": {
                "mcp_server": "arc",
                "tool": "predict_action",
            },
        })
        assert resp["ok"] is True
        assert resp["execution_id"]
        # End it cleanly so the test doesn't leave a row in
        # "running".
        end = _post(
            f"{base}/execution/end/{resp['execution_id']}",
            {"status": "ok",
             "outputs": {"prediction": {"foo": "bar"}}},
        )
        assert end["ok"] is True

    def test_unknown_tool_rejected(self, channel_setup):
        base, _ws, _reg, _mf = channel_setup
        resp = _post(f"{base}/execution/begin", {
            "block": "predict",
            "caller": {
                "mcp_server": "arc",
                "tool": "ghost_tool",
            },
        })
        assert resp["ok"] is False
        assert resp["error_type"] == "manifest_violation"
        assert "ghost_tool" in resp["message"]

    def test_block_mismatch_rejected(self, channel_setup):
        base, _ws, _reg, _mf = channel_setup
        # predict_action is bound to "predict"; asking for
        # "train" via that tool is a violation.
        resp = _post(f"{base}/execution/begin", {
            "block": "train",
            "caller": {
                "mcp_server": "arc",
                "tool": "predict_action",
            },
        })
        assert resp["ok"] is False
        assert resp["error_type"] == "manifest_violation"
        assert "predict" in resp["message"]

    def test_caller_omitted_bypasses_check(self, channel_setup):
        # Calls without a caller (e.g., direct CLI usage, tests)
        # must not trip the manifest check.
        base, _ws, _reg, _mf = channel_setup
        resp = _post(f"{base}/execution/begin", {
            "block": "predict",
        })
        assert resp["ok"] is True
        _post(
            f"{base}/execution/end/{resp['execution_id']}",
            {"status": "ok", "outputs": {"prediction": {}}},
        )

    def test_unknown_mcp_server_bypasses_check(
            self, channel_setup):
        # During the rollout, MCP servers without a manifest
        # remain accepted; only servers the channel knows about
        # are policed.
        base, _ws, _reg, _mf = channel_setup
        resp = _post(f"{base}/execution/begin", {
            "block": "predict",
            "caller": {
                "mcp_server": "newserver",
                "tool": "something",
            },
        })
        assert resp["ok"] is True
        _post(
            f"{base}/execution/end/{resp['execution_id']}",
            {"status": "ok", "outputs": {"prediction": {}}},
        )


class TestChannelWithoutManifests:
    def test_no_manifest_means_no_enforcement(
            self, tmp_path: Path, registry: BlockRegistry):
        tmpl_path = tmp_path / "test_tmpl.yaml"
        tmpl_path.write_text(TEMPLATE_YAML)
        template = Template.from_yaml(
            tmpl_path, block_registry=registry)
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        (ws_path / "artifacts").mkdir()
        workspace = Workspace(
            name="test-ws",
            path=ws_path,
            template_name="test_tmpl",
            created_at=datetime.now(UTC),
            artifact_declarations={
                a.name: a.kind for a in template.artifacts},
            artifacts={},
        )
        workspace.save()
        _seed_artifact(workspace, "predictor", "# stub\n")

        channel = ExecutionChannel(
            template=template,
            workspace=workspace,
            host="127.0.0.1",
            # manifests omitted → no enforcement
        )
        port = channel.start()
        base = f"http://127.0.0.1:{port}"
        time.sleep(0.05)
        try:
            resp = _post(f"{base}/execution/begin", {
                "block": "predict",
                "caller": {
                    "mcp_server": "arc",
                    "tool": "anything_at_all",
                },
            })
            assert resp["ok"] is True
            _post(
                f"{base}/execution/end/{resp['execution_id']}",
                {"status": "ok",
                 "outputs": {"prediction": {}}},
            )
        finally:
            channel.stop()
