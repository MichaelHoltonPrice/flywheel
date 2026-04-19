"""Tests for :class:`flywheel.executor.ProcessExitExecutor`.

Uses mocked ``start_container`` so the tests don't need Docker.
Coverage:

- happy path (stateless and stateful blocks),
- canonical-never-directly-mounted invariant on inputs,
- state populate / capture round-trip, state_lineage_id
  isolation, missing-state_dir failing stage_in,
- every failure phase lands on the ``BlockExecution`` record
  (stage_in, invoke, state_capture, output_collect),
- incremental-output collection appends rather than creating
  fresh instances,
- ``exit_code`` sentinel on invoke-time exceptions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from flywheel import runtime
from flywheel.artifact import ArtifactInstance
from flywheel.executor import (
    INVOKE_FAILURE_EXIT_CODE,
    ProcessExitExecutor,
)
from flywheel.input_staging import StagingError
from flywheel.template import (
    ArtifactDeclaration,
    BlockDefinition,
    InputSlot,
    OutputSlot,
    Template,
)
from flywheel.workspace import Workspace


class FakePopen:
    """Minimal :class:`subprocess.Popen` stand-in.

    Holds a ``returncode`` that :meth:`poll` / :meth:`wait` can
    return.  Sufficient for tests whose simulated container has
    already finished its work synchronously by the time
    ``start_container`` returns.  Cancellation tests extend this
    with richer state transitions.
    """

    def __init__(self, returncode: int = 0):
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


def _make_template(
    *,
    block_name: str = "train",
    state: bool = False,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    output_kinds: dict[str, str] | None = None,
    stop_timeout_s: int = 30,
) -> Template:
    """Build a tiny Template with one container block.

    ``output_kinds`` overrides the kind of individual outputs;
    any unlisted output (and every input) defaults to ``copy``.
    ``stop_timeout_s`` sets the cooperative-cancel window, so
    cancellation tests can assert the configured value is
    respected.
    """
    inputs = inputs or []
    outputs = outputs or []
    output_kinds = output_kinds or {}
    artifact_names = set(inputs) | set(outputs)
    artifacts = [
        ArtifactDeclaration(
            name=n,
            kind=output_kinds.get(n, "copy"),
        )
        for n in artifact_names
    ]
    block = BlockDefinition(
        name=block_name,
        image="test:latest",
        runner="container",
        inputs=[
            InputSlot(name=n, container_path=f"/input/{n}")
            for n in inputs
        ],
        outputs=[
            OutputSlot(name=n, container_path=f"/output/{n}")
            for n in outputs
        ],
        state=state,
        stop_timeout_s=stop_timeout_s,
    )
    return Template(
        name="t", artifacts=artifacts, blocks=[block])


def _make_workspace(tmp_path: Path, template: Template) -> Workspace:
    """Build a workspace on disk rooted at ``tmp_path``."""
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


def _fake_run_success(
    container_contents: dict[str, str] | None = None,
    state_contents: dict[str, str] | None = None,
    exit_code: int = 0,
):
    """Return a ``start_container`` replacement that writes files.

    Simulates a container that has finished its work
    synchronously by the time ``start_container`` returns:
    writes the given ``container_contents`` into every rw
    output mount, plus ``state_contents`` into ``/state/`` (if
    mounted), then returns a :class:`FakePopen` already at
    ``exit_code``.  ``wait()`` on it returns immediately.
    """
    def _fake(config, args=None, name=None):
        for host, container, mode in config.mounts:
            if mode != "rw":
                continue
            host_p = Path(host)
            if container == runtime.STATE_MOUNT_PATH:
                if state_contents:
                    for fname, content in state_contents.items():
                        (host_p / fname).write_text(content)
            elif container == "/workspace":
                # The work-area mount; no files expected here
                # for the success path.
                continue
            else:
                if container_contents:
                    for fname, content in container_contents.items():
                        (host_p / fname).write_text(content)
        return FakePopen(returncode=exit_code)
    return _fake


class TestHappyPathStateless:
    def test_succeeded_with_outputs(self, tmp_path: Path):
        template = _make_template(outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_run_success(
                container_contents={"payload.txt": "ok"}),
        ):
            handle = executor.launch(
                "train", ws, input_bindings={})
        result = handle.wait()

        assert result.status == "succeeded"
        assert result.exit_code == 0
        assert "result" in result.output_bindings

        ex = ws.executions[result.execution_id]
        assert ex.failure_phase is None
        assert ex.state_dir is None
        assert ex.runner == "container"

        # Canonical artifact dir exists with the output the
        # fake container wrote.
        aid = result.output_bindings["result"]
        canonical = ws.path / "artifacts" / aid
        assert (canonical / "payload.txt").read_text() == "ok"

    def test_no_output_artifacts_when_output_dirs_empty(
        self, tmp_path: Path,
    ):
        """A block that produces no files should register no outputs."""
        template = _make_template(outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_run_success(),  # writes nothing
        ):
            handle = executor.launch(
                "train", ws, input_bindings={})
        result = handle.wait()

        assert result.status == "succeeded"
        assert result.output_bindings == {}


class TestStateRoundTrip:
    def test_first_execution_empty_state(self, tmp_path: Path):
        template = _make_template(state=True, outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_run_success(
                state_contents={"counter.txt": "1"}),
        ):
            handle = executor.launch(
                "train", ws, input_bindings={})
        result = handle.wait()

        ex = ws.executions[result.execution_id]
        assert ex.state_dir is not None
        assert ex.state_dir.startswith("state/train/")

        # State dir on disk contains what the container wrote.
        state_on_disk = ws.path / ex.state_dir
        assert (state_on_disk / "counter.txt").read_text() == "1"

    def test_second_execution_sees_prior_state(
        self, tmp_path: Path,
    ):
        """Second execution's /state/ must be populated from the
        first's captured state."""
        template = _make_template(state=True, outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        seen_state_on_second: dict[str, str] = {}

        def _fake_first(config, args=None, name=None):
            for host, container, _mode in config.mounts:
                if container == runtime.STATE_MOUNT_PATH:
                    (Path(host) / "counter.txt").write_text("1")
            return FakePopen(returncode=0)

        def _fake_second(config, args=None, name=None):
            # Record what's visible in /state/ at start, then
            # append.
            for host, container, _mode in config.mounts:
                if container == runtime.STATE_MOUNT_PATH:
                    host_p = Path(host)
                    for child in host_p.iterdir():
                        seen_state_on_second[child.name] = (
                            child.read_text())
                    (host_p / "counter.txt").write_text("2")
            return FakePopen(returncode=0)

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_first,
        ):
            executor.launch("train", ws, input_bindings={}).wait()

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_second,
        ):
            handle = executor.launch(
                "train", ws, input_bindings={})
        result = handle.wait()

        # Second container saw the first's state on mount-in.
        assert seen_state_on_second == {"counter.txt": "1"}
        # And the second's captured state reflects its write.
        ex = ws.executions[result.execution_id]
        assert (
            ws.path / ex.state_dir / "counter.txt"
        ).read_text() == "2"


class TestCanonicalNotMountedInvariant:
    def test_input_mounts_do_not_use_canonical_path(
        self, tmp_path: Path,
    ):
        """Mounts handed to ``start_container`` must not include any
        path under ``<workspace>/artifacts/`` — that would violate
        the canonical-never-directly-mounted invariant."""
        template = _make_template(
            inputs=["engine"], outputs=["result"])
        ws = _make_workspace(tmp_path, template)

        # Pre-register an input artifact with a real on-disk
        # canonical dir.
        aid = "engine@baseline"
        canonical = ws.path / "artifacts" / aid
        canonical.mkdir(parents=True)
        (canonical / "data.txt").write_text("original")
        ws.add_artifact(ArtifactInstance(
            id=aid, name="engine", kind="copy",
            created_at=datetime.now(UTC),
            copy_path=aid,
        ))

        executor = ProcessExitExecutor(template)

        captured_mounts: list[tuple[str, str, str]] = []

        def _fake(config, args=None, name=None):
            captured_mounts.extend(config.mounts)
            return FakePopen(returncode=0)

        with patch(
            "flywheel.executor.start_container", side_effect=_fake,
        ):
            executor.launch(
                "train", ws,
                input_bindings={"engine": aid},
            ).wait()

        canonical_root = str((ws.path / "artifacts").resolve())
        for host, _, _ in captured_mounts:
            host_resolved = str(Path(host).resolve())
            assert not host_resolved.startswith(canonical_root), (
                f"Mount {host} is under canonical artifacts/ — "
                f"violates canonical-never-directly-mounted"
            )


class TestFailurePhases:
    def test_stage_in_failure(self, tmp_path: Path):
        """A stage_artifact_instances error should record
        failure_phase='stage_in'."""
        template = _make_template(
            inputs=["engine"], outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        def _raise(*args, **kwargs):
            raise StagingError(
                "simulated", instance_id="engine@broken")

        with patch(
            "flywheel.executor.stage_artifact_instances",
            side_effect=_raise,
        ):
            handle = executor.launch(
                "train", ws,
                input_bindings={"engine": "engine@broken"},
            )
        result = handle.wait()

        assert result.status == "failed"
        ex = ws.executions[result.execution_id]
        assert ex.failure_phase == runtime.FAILURE_STAGE_IN
        assert "stage_in" in ex.error

    def test_invoke_failure_non_zero_exit(self, tmp_path: Path):
        template = _make_template(outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_run_success(exit_code=2),
        ):
            handle = executor.launch(
                "train", ws, input_bindings={})
        result = handle.wait()

        assert result.status == "failed"
        assert result.exit_code == 2
        ex = ws.executions[result.execution_id]
        assert ex.failure_phase == runtime.FAILURE_INVOKE
        assert "exited with code 2" in ex.error

    def test_invoke_failure_start_container_raises(
        self, tmp_path: Path,
    ):
        template = _make_template(outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        def _boom(*args, **kwargs):
            raise RuntimeError("docker daemon gone")

        with patch(
            "flywheel.executor.start_container", side_effect=_boom,
        ):
            handle = executor.launch(
                "train", ws, input_bindings={})
        result = handle.wait()

        assert result.status == "failed"
        ex = ws.executions[result.execution_id]
        assert ex.failure_phase == runtime.FAILURE_INVOKE
        assert "docker daemon gone" in ex.error


class TestIncrementalOutputs:
    """Outputs declared ``kind: incremental`` must append to the
    canonical incremental artifact, not be rejected as copy."""

    def test_incremental_output_appends_entries(
        self, tmp_path: Path,
    ):
        template = _make_template(
            outputs=["game_history"],
            output_kinds={"game_history": "incremental"},
        )
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        def _fake(config, args=None, name=None):
            for host, container, _mode in config.mounts:
                if container == "/output/game_history":
                    (Path(host) / "entries.jsonl").write_text(
                        '{"step": 1}\n{"step": 2}\n')
            return FakePopen(returncode=0)

        with patch(
            "flywheel.executor.start_container", side_effect=_fake,
        ):
            handle = executor.launch(
                "train", ws, input_bindings={})
        result = handle.wait()

        assert result.status == "succeeded"
        assert "game_history" in result.output_bindings
        instance_id = result.output_bindings["game_history"]
        instance = ws.artifacts[instance_id]
        assert instance.kind == "incremental"

    def test_incremental_second_execution_appends_to_same_instance(
        self, tmp_path: Path,
    ):
        """Two executions of a block with an incremental output
        must share one canonical instance that grows, not create
        two separate instances."""
        template = _make_template(
            outputs=["game_history"],
            output_kinds={"game_history": "incremental"},
        )
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        def _fake_step(step_id: int):
            def _inner(config, args=None, name=None):
                for host, container, _mode in config.mounts:
                    if container == "/output/game_history":
                        (Path(host) / "entries.jsonl").write_text(
                            f'{{"step": {step_id}}}\n')
                return FakePopen(returncode=0)
            return _inner

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_step(1),
        ):
            r1 = executor.launch(
                "train", ws, input_bindings={}).wait()
        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_step(2),
        ):
            r2 = executor.launch(
                "train", ws, input_bindings={}).wait()

        # Same instance id from both — the canonical grows.
        assert (
            r1.output_bindings["game_history"]
            == r2.output_bindings["game_history"]
        )
        # And the entries file holds both appends.  Re-parse via
        # json so we're not sensitive to whitespace differences
        # from the workspace's canonical serializer.
        instance = ws.artifacts[
            r1.output_bindings["game_history"]]
        entries_on_disk = (
            ws.path / "artifacts" / instance.copy_path
            / "entries.jsonl"
        ).read_text()
        parsed = [
            json.loads(line) for line in
            entries_on_disk.splitlines() if line.strip()
        ]
        assert parsed == [{"step": 1}, {"step": 2}]

    def test_incremental_without_entries_file_skipped(
        self, tmp_path: Path,
    ):
        """If the container declared an incremental output but
        wrote no entries.jsonl, we skip silently — optional-not-
        produced semantics match the copy path."""
        template = _make_template(
            outputs=["game_history"],
            output_kinds={"game_history": "incremental"},
        )
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        def _fake(config, args=None, name=None):
            # Write a file that isn't entries.jsonl.
            for host, container, _mode in config.mounts:
                if container == "/output/game_history":
                    (Path(host) / "README").write_text("debug")
            return FakePopen(returncode=0)

        with patch(
            "flywheel.executor.start_container", side_effect=_fake,
        ):
            result = executor.launch(
                "train", ws, input_bindings={}).wait()

        assert result.status == "succeeded"
        assert "game_history" not in result.output_bindings


class TestStateCaptureFailure:
    def test_state_capture_oserror_records_failure_phase(
        self, tmp_path: Path,
    ):
        template = _make_template(state=True, outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        def _fake(config, args=None, name=None):
            # Container writes state successfully.
            for host, container, _mode in config.mounts:
                if container == runtime.STATE_MOUNT_PATH:
                    (Path(host) / "counter.txt").write_text("1")
            return FakePopen(returncode=0)

        def _raise(*args, **kwargs):
            raise OSError("simulated disk-full")

        with (
            patch(
                "flywheel.executor.start_container",
                side_effect=_fake,
            ),
            patch(
                "flywheel.executor._capture_state",
                side_effect=_raise,
            ),
        ):
            result = executor.launch(
                "train", ws, input_bindings={}).wait()

        assert result.status == "failed"
        ex = ws.executions[result.execution_id]
        assert ex.failure_phase == runtime.FAILURE_STATE_CAPTURE
        assert "simulated disk-full" in ex.error


class TestOutputCollectFailure:
    def test_output_collect_exception_records_failure_phase(
        self, tmp_path: Path,
    ):
        template = _make_template(outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        def _fake(config, args=None, name=None):
            for host, container, _mode in config.mounts:
                if container == "/output/result":
                    (Path(host) / "payload.txt").write_text("x")
            return FakePopen(returncode=0)

        def _raise(*args, **kwargs):
            raise RuntimeError("simulated workspace lock failure")

        with (
            patch(
                "flywheel.executor.start_container",
                side_effect=_fake,
            ),
            patch(
                "flywheel.executor._collect_outputs",
                side_effect=_raise,
            ),
        ):
            result = executor.launch(
                "train", ws, input_bindings={}).wait()

        assert result.status == "failed"
        ex = ws.executions[result.execution_id]
        assert ex.failure_phase == runtime.FAILURE_OUTPUT_COLLECT
        assert "simulated workspace lock failure" in ex.error


class TestMissingStateDir:
    def test_missing_recorded_state_dir_fails_stage_in(
        self, tmp_path: Path,
    ):
        """Prior execution has state_dir in the ledger but the
        directory is gone on disk — stage_in must fail loudly
        rather than silently cold-starting."""
        template = _make_template(state=True, outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        # Seed a prior execution + real state dir.
        def _fake_first(config, args=None, name=None):
            for host, container, _mode in config.mounts:
                if container == runtime.STATE_MOUNT_PATH:
                    (Path(host) / "counter.txt").write_text("1")
            return FakePopen(returncode=0)

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_first,
        ):
            first = executor.launch(
                "train", ws, input_bindings={}).wait()

        # Delete the captured state_dir to simulate corruption.
        shutil.rmtree(
            ws.path / ws.executions[first.execution_id].state_dir)

        # Second execution must fail stage_in.
        def _fake_second(config, args=None, name=None):
            return FakePopen(returncode=0)

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_second,
        ):
            result = executor.launch(
                "train", ws, input_bindings={}).wait()

        assert result.status == "failed"
        ex = ws.executions[result.execution_id]
        assert ex.failure_phase == runtime.FAILURE_STAGE_IN
        assert "missing on disk" in ex.error


class TestStateLineage:
    def test_default_lineage_is_none(self, tmp_path: Path):
        template = _make_template(state=True)
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_run_success(),
        ):
            result = executor.launch(
                "train", ws, input_bindings={}).wait()

        ex = ws.executions[result.execution_id]
        assert ex.state_lineage_id is None

    def test_distinct_lineages_have_independent_state(
        self, tmp_path: Path,
    ):
        """Two executions in distinct lineages don't see each
        other's state on /state/ populate."""
        template = _make_template(state=True)
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        def _fake_write(tag: str):
            def _inner(config, args=None, name=None):
                for host, container, _mode in config.mounts:
                    if container == runtime.STATE_MOUNT_PATH:
                        (Path(host) / "tag.txt").write_text(tag)
                return FakePopen(returncode=0)
            return _inner

        # Execution in lineage A writes "A".
        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_write("A"),
        ):
            executor.launch(
                "train", ws, input_bindings={},
                state_lineage_id="A",
            ).wait()

        # Execution in lineage B must see an empty /state/ — the
        # A-lineage state is not its predecessor.
        seen: dict[str, str] = {}

        def _fake_b(config, args=None, name=None):
            for host, container, _mode in config.mounts:
                if container == runtime.STATE_MOUNT_PATH:
                    host_p = Path(host)
                    seen["empty"] = (
                        "yes" if not any(host_p.iterdir())
                        else "no"
                    )
                    (host_p / "tag.txt").write_text("B")
            return FakePopen(returncode=0)

        with patch(
            "flywheel.executor.start_container", side_effect=_fake_b,
        ):
            executor.launch(
                "train", ws, input_bindings={},
                state_lineage_id="B",
            ).wait()

        assert seen["empty"] == "yes"


class TestExitCodeOnInvokeException:
    def test_exit_code_is_sentinel_not_zero(
        self, tmp_path: Path,
    ):
        """When run_container raises before producing a result,
        the returned ExecutionResult must surface a non-zero
        exit_code so callers inspecting just the handle don't
        mistake it for success."""
        template = _make_template(outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        def _boom(*args, **kwargs):
            raise RuntimeError("docker daemon gone")

        with patch(
            "flywheel.executor.start_container", side_effect=_boom,
        ):
            result = executor.launch(
                "train", ws, input_bindings={}).wait()

        assert result.status == "failed"
        assert result.exit_code == INVOKE_FAILURE_EXIT_CODE
        assert result.exit_code != 0

        # The workspace execution record correctly shows None for
        # a non-existent exit code, though — no sentinel leaks
        # into the ledger.
        ex = ws.executions[result.execution_id]
        assert ex.exit_code is None


class _CancellableFakePopen:
    """Richer FakePopen for two-phase cancellation tests.

    Simulates a running process that can exit for one of three
    reasons:

    - The cooperative stop sentinel appeared at
      ``sentinel_path`` AND ``exits_on_sentinel`` is True.
    - A SIGTERM was signaled AND ``exits_on_term`` is True.
    - A SIGKILL was signaled.

    ``wait(timeout=...)`` returns the exit code if the process
    has exited, or raises ``subprocess.TimeoutExpired`` after
    the specified delay (simulated — does not actually sleep).
    Tests drive the "does the timeout fire?" decision by
    arranging sentinel / signal state before each ``wait``
    call.
    """

    def __init__(
        self,
        *,
        sentinel_path,
        exits_on_sentinel: bool = True,
        exits_on_term: bool = True,
        natural_exit_code: int = 0,
    ):
        self._sentinel_path = sentinel_path
        self._exits_on_sentinel = exits_on_sentinel
        self._exits_on_term = exits_on_term
        self._natural_exit_code = natural_exit_code
        self._term_signaled = False
        self._kill_signaled = False
        self.returncode: int | None = None
        # Records the timeout seen on every ``wait`` call.  Tests
        # assert these match the block's configured
        # ``stop_timeout_s``, so the executor can't silently pass
        # the wrong value.
        self.wait_timeouts: list[float | None] = []

    def receive_term(self) -> None:
        """Called by the test's docker-kill mock on SIGTERM."""
        self._term_signaled = True

    def receive_kill(self) -> None:
        """Called by the test's docker-kill mock on SIGKILL."""
        self._kill_signaled = True

    def _update_exit(self) -> None:
        if self.returncode is not None:
            return
        if self._kill_signaled:
            self.returncode = 137  # conventional SIGKILL exit code
        elif self._term_signaled and self._exits_on_term:
            self.returncode = 143  # conventional SIGTERM exit code
        elif (self._exits_on_sentinel
                and self._sentinel_path is not None
                and self._sentinel_path.exists()):
            self.returncode = self._natural_exit_code

    def poll(self) -> int | None:
        self._update_exit()
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        self._update_exit()
        if self.returncode is not None:
            return self.returncode
        if timeout is None:
            # Wouldn't terminate; treat as a broken-test signal.
            raise AssertionError(
                "_CancellableFakePopen.wait() without timeout "
                "and no exit condition — test setup is wrong"
            )
        raise subprocess.TimeoutExpired(
            cmd="fake", timeout=timeout)


def _start_cancellable(
    *,
    exits_on_sentinel: bool = True,
    exits_on_term: bool = True,
    natural_exit_code: int = 0,
):
    """Return a ``start_container`` replacement for cancel tests.

    The returned callable constructs a ``_CancellableFakePopen``
    bound to the work-area's sentinel path, and returns it —
    wiring the pipeline for cancellation exercises.  The test
    keeps a reference to the returned popen via the outer
    ``popen_ref`` dict so it can drive transitions.
    """
    popen_ref: dict = {}

    def _factory(config, args=None, name=None):
        # Locate the work-area mount (host path for /workspace).
        work_area = None
        for host, container, _mode in config.mounts:
            if container == "/workspace":
                work_area = Path(host)
                break
        sentinel_path = (
            (work_area / runtime.STOP_SENTINEL_WORKSPACE_RELATIVE)
            if work_area is not None else None
        )
        popen = _CancellableFakePopen(
            sentinel_path=sentinel_path,
            exits_on_sentinel=exits_on_sentinel,
            exits_on_term=exits_on_term,
            natural_exit_code=natural_exit_code,
        )
        popen_ref["popen"] = popen
        return popen

    return _factory, popen_ref


class TestCooperativeStop:
    """Cooperative stop: container sees the sentinel and exits."""

    def test_honoring_container_records_interrupted(
        self, tmp_path: Path,
    ):
        template = _make_template(outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        factory, popen_ref = _start_cancellable(
            exits_on_sentinel=True)

        with patch(
            "flywheel.executor.start_container",
            side_effect=factory,
        ):
            handle = executor.launch(
                "train", ws, input_bindings={})
            handle.stop(reason="test_cancel")
            result = handle.wait()

        assert result.status == "interrupted"
        ex = ws.executions[result.execution_id]
        assert ex.stop_reason == "test_cancel"
        # Forced termination wasn't needed — the container
        # exited on the sentinel, so exit_code is the natural
        # one we passed in.
        assert ex.exit_code == 0


class TestForcedTermination:
    """Container ignores the sentinel; TERM (and maybe KILL) fires."""

    def test_ignoring_sentinel_triggers_term_after_configured_timeout(
        self, tmp_path: Path,
    ):
        # Use a non-default timeout and assert it's what the
        # executor actually passes to wait().
        configured_timeout = 7
        template = _make_template(
            outputs=["result"],
            stop_timeout_s=configured_timeout,
        )
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        factory, popen_ref = _start_cancellable(
            exits_on_sentinel=False,
            exits_on_term=True,
        )

        def _fake_signal(container_name, signal):
            popen = popen_ref["popen"]
            if signal == "TERM":
                popen.receive_term()
            elif signal == "KILL":
                popen.receive_kill()

        with (
            patch(
                "flywheel.executor.start_container",
                side_effect=factory,
            ),
            patch(
                "flywheel.executor._docker_send_signal",
                side_effect=_fake_signal,
            ),
        ):
            handle = executor.launch(
                "train", ws, input_bindings={})
            handle.stop(reason="timeout")
            result = handle.wait()

        assert result.status == "interrupted"
        ex = ws.executions[result.execution_id]
        assert ex.stop_reason == "timeout"
        assert ex.exit_code == 143  # SIGTERM exit

        # Proves the configured stop_timeout_s was what the
        # executor passed to the cooperative wait.  Should be
        # the first timed wait in the sequence.
        popen = popen_ref["popen"]
        assert configured_timeout in popen.wait_timeouts, (
            f"Expected cooperative wait to use "
            f"stop_timeout_s={configured_timeout}; "
            f"saw {popen.wait_timeouts}"
        )

    def test_ignoring_term_triggers_kill(
        self, tmp_path: Path,
    ):
        """Container ignores sentinel AND SIGTERM; only SIGKILL
        ends it — proves TERM/KILL escalation works."""
        template = _make_template(
            outputs=["result"], stop_timeout_s=0)
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        factory, popen_ref = _start_cancellable(
            exits_on_sentinel=False,
            exits_on_term=False,
        )
        signals_sent: list[str] = []

        def _fake_signal(container_name, signal):
            signals_sent.append(signal)
            popen = popen_ref["popen"]
            if signal == "KILL":
                popen.receive_kill()
            # TERM is deliberately not honored by this popen.

        with (
            patch(
                "flywheel.executor.start_container",
                side_effect=factory,
            ),
            patch(
                "flywheel.executor._docker_send_signal",
                side_effect=_fake_signal,
            ),
        ):
            handle = executor.launch(
                "train", ws, input_bindings={})
            handle.stop(reason="hung")
            result = handle.wait()

        assert signals_sent == ["TERM", "KILL"], (
            f"Expected TERM then KILL; got {signals_sent}"
        )
        assert result.status == "interrupted"
        ex = ws.executions[result.execution_id]
        assert ex.exit_code == 137  # SIGKILL exit


class TestStopOnExitedContainer:
    """stop() when the container already finished must not mislabel."""

    def test_stop_after_natural_exit(self, tmp_path: Path):
        template = _make_template(outputs=["result"])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        # Happy-path fake: container already returned a popen
        # with returncode=0, so stop() should short-circuit.
        with patch(
            "flywheel.executor.start_container",
            side_effect=_fake_run_success(),
        ):
            handle = executor.launch(
                "train", ws, input_bindings={})
            # Calling stop here must not raise or alter the
            # execution record.
            handle.stop(reason="late")
            result = handle.wait()

        assert result.status == "succeeded"
        ex = ws.executions[result.execution_id]
        # Natural exit; stop_reason should be None.
        assert ex.stop_reason is None



class TestValidation:
    def test_unknown_block_raises(self, tmp_path: Path):
        template = _make_template(block_name="train")
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)
        with pytest.raises(ValueError, match="not found"):
            executor.launch("bogus", ws, input_bindings={})

    def test_not_allowed_raises(self, tmp_path: Path):
        template = _make_template(block_name="train")
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)
        with pytest.raises(ValueError, match="not in allowed list"):
            executor.launch(
                "train", ws, input_bindings={},
                allowed_blocks=["other"],
            )

    def test_non_container_runner_raises(self, tmp_path: Path):
        """ProcessExitExecutor only runs container blocks; the
        schema narrowing means ``lifecycle`` is the only other
        runner kind."""
        block = BlockDefinition(
            name="logical",
            runner="lifecycle",
            runner_justification="test fixture",
        )
        template = Template(
            name="t", artifacts=[], blocks=[block])
        ws = _make_workspace(tmp_path, template)
        executor = ProcessExitExecutor(template)

        with pytest.raises(
            ValueError, match="only runs container blocks",
        ):
            executor.launch(
                "logical", ws, input_bindings={})
