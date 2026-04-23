"""End-to-end tests for ``output_builder`` in the executor.

Covers the behaviour an ``output_builder`` field promises:

- runs after the container exits, before artifact collection,
- can read the raw container output and write the canonical
  artifact files in place,
- failures land on the ``BlockExecution`` record with phase
  ``output_collect`` (same phase as a broken collection pass).

Uses the same ``FakePopen`` / ``start_container`` patching
pattern as :mod:`tests.test_process_exit_executor`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from flywheel import runtime
from flywheel.executor import ProcessExitExecutor
from flywheel.template import (
    ArtifactDeclaration,
    BlockDefinition,
    InputSlot,
    OutputSlot,
    Template,
)
from flywheel.workspace import Workspace


class FakePopen:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode
        self.stdout = None
        self.stderr = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


def _fake_run_success(container_contents: dict[str, str]):
    """Patched ``start_container`` that writes fake output files."""

    def _fake(config, args=None, name=None, **_):
        for host, container, mode in config.mounts:
            if mode != "rw":
                continue
            if container in ("/scratch",
                             runtime.STATE_MOUNT_PATH):
                continue
            host_p = Path(host)
            for fname, content in container_contents.items():
                (host_p / fname).write_text(content)
        return FakePopen(returncode=0)

    return _fake


# ── Module-level helpers referenced by output_builder paths ─────


def _collapse_markdown_to_jsonl(ctx) -> None:
    """Reference builder: read two .md files, write one JSONL line.

    Matches the brainstorm use case: container writes
    ``ideas.md`` and ``next_experiment.md``; builder assembles a
    single JSON object with both and appends one line of
    ``entries.jsonl`` in place.  Stamps ``last_observed_step``
    from the workspace (so the builder exercises the workspace
    read path too).
    """
    out_dir = ctx.outputs["brainstorm_result"]
    ideas = (out_dir / "ideas.md").read_text(encoding="utf-8")
    nxt = (out_dir / "next_experiment.md").read_text(
        encoding="utf-8")
    # Read the existing incremental input (if any) to demonstrate
    # workspace-side state access.  Tests configure an empty
    # workspace so this is just 0 — the important part is that
    # the workspace is reachable.
    _ = ctx.workspace
    entry = {
        "ideas": ideas,
        "next_experiment": nxt,
        "last_observed_step": 0,
    }
    (out_dir / "ideas.md").unlink()
    (out_dir / "next_experiment.md").unlink()
    (out_dir / "entries.jsonl").write_text(
        json.dumps(entry) + "\n", encoding="utf-8")


def _builder_that_raises(ctx) -> None:
    raise RuntimeError("builder boom")


# ── Template + workspace helpers ───────────────────────────────


def _template(
    *,
    output_builder: str | None,
    output_kind: str = "incremental",
) -> Template:
    artifact = ArtifactDeclaration(
        name="brainstorm_result", kind=output_kind)
    block = BlockDefinition(
        name="brainstorm",
        image="test:latest",
        runner="container",
        inputs=[],
        outputs={"normal": [OutputSlot(
            name="brainstorm_result",
            container_path="/output/brainstorm_result",
        )]},
        output_builder=output_builder,
    )
    return Template(
        name="t", artifacts=[artifact], blocks=[block])


def _workspace(tmp_path: Path, template: Template) -> Workspace:
    ws_path = tmp_path / "ws"
    ws_path.mkdir()
    (ws_path / "artifacts").mkdir()
    ws = Workspace(
        name="ws",
        path=ws_path,
        template_name=template.name,
        created_at=datetime.now(UTC),
        artifact_declarations={
            a.name: a.kind for a in template.artifacts
        },
        artifacts={},
    )
    ws.save()
    return ws


# ── Tests ───────────────────────────────────────────────────────


class TestOutputBuilderHappyPath:
    """Builder runs, rewrites the tempdir, collection picks it up."""

    @pytest.mark.skip(reason="Test depends on incremental artifact kind, which has been excised")
    def test_rewrites_tempdir_into_canonical_artifact(
            self, tmp_path: Path):
        template = _template(
            output_builder=(
                "tests.test_executor_output_builder"
                "._collapse_markdown_to_jsonl"
            ),
            output_kind="incremental",
        )
        ws = _workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        # Fake container writes the two markdown files the
        # builder will collapse.
        fake = _fake_run_success({
            "ideas.md": "# ideas\nhypothesis one\n",
            "next_experiment.md": "do X at y=4\n",
        })
        with patch(
            "flywheel.executor.start_container",
            side_effect=fake,
        ):
            handle = executor.launch(
                "brainstorm", ws, input_bindings={})
        result = handle.wait()

        assert result.status == "succeeded"
        assert "brainstorm_result" in result.output_bindings

        # The canonical incremental artifact's entries.jsonl now
        # holds exactly what the builder assembled.
        aid = result.output_bindings["brainstorm_result"]
        entries_file = (
            ws.path / "artifacts" / aid / "entries.jsonl")
        lines = entries_file.read_text(
            encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["ideas"].startswith("# ideas")
        assert entry["next_experiment"].startswith("do X")
        assert entry["last_observed_step"] == 0


class TestOutputBuilderFailureMapsToCollectPhase:
    """A raising builder fails the execution at output_collect."""

    def test_builder_exception_marks_execution_failed(
            self, tmp_path: Path):
        template = _template(
            output_builder=(
                "tests.test_executor_output_builder"
                "._builder_that_raises"
            ),
            output_kind="copy",
        )
        ws = _workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_run_success(
                {"payload.txt": "ok"}),
        ):
            handle = executor.launch(
                "brainstorm", ws, input_bindings={})
        result = handle.wait()

        assert result.status == "failed"
        ex = ws.executions[result.execution_id]
        assert ex.failure_phase == runtime.FAILURE_OUTPUT_COLLECT
        assert "builder boom" in (ex.error or "")


class TestNoBuilderFieldSkipsTheStep:
    """Blocks without ``output_builder`` behave exactly as before."""

    def test_standard_collection_still_runs(
            self, tmp_path: Path):
        template = _template(
            output_builder=None, output_kind="copy")
        ws = _workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_run_success(
                {"payload.txt": "hi"}),
        ):
            handle = executor.launch(
                "brainstorm", ws, input_bindings={})
        result = handle.wait()

        assert result.status == "succeeded"
        aid = result.output_bindings["brainstorm_result"]
        canonical = ws.path / "artifacts" / aid
        assert (
            canonical / "payload.txt").read_text() == "hi"
