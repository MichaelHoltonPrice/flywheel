"""Tests for :class:`flywheel.executor.RequestResponseExecutor`.

The executor drives request-response blocks: persistent
containers serving many per-request executions over an
executor-owned control channel.  Tests here cover the host-side
contract — runtime lifecycle, per-request mount staging,
host-side request serialization, cancellation, attachment-key
reuse — with a fake :class:`ControlChannel` injected via the
executor's ``channel_factory`` hook so Docker is never touched.

Integration tests that do boot a real container live alongside
the ``persistent_fake/`` fixture and are gated behind
``pytest.mark.docker``.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from flywheel import runtime
from flywheel.executor import (
    ControlChannel,
    ControlChannelError,
    RequestResponseExecutor,
    _RUNTIME_LABEL_BLOCK,
    _RUNTIME_LABEL_LIFECYCLE,
    _RUNTIME_LABEL_PROTOCOL,
    _RUNTIME_LABEL_WORKSPACE,
    _RUNTIME_LABEL_WORKSPACE_PATH,
    REQUEST_RESPONSE_PROTOCOL_VERSION,
)
from flywheel.template import (
    ArtifactDeclaration,
    BlockDefinition,
    InputSlot,
    OutputSlot,
    Template,
)
from flywheel.workspace import Workspace


# ─── Fakes ──────────────────────────────────────────────────────


class FakePopen:
    """Minimal stand-in for the ``subprocess.Popen`` of a container.

    The request-response executor runs containers detached; it
    only needs ``poll`` and ``wait`` during teardown.  Tests
    that assert on teardown transitions drive this fake's
    ``returncode`` directly.
    """

    def __init__(self, returncode: int | None = None):
        self.returncode = returncode
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.returncode is None:
            # Simulate a still-live process by refusing to block
            # forever — treat as timeout-expired so tests can
            # observe the escalation path.
            import subprocess
            raise subprocess.TimeoutExpired(
                cmd="fake", timeout=timeout)
        return self.returncode


class FakeChannel:
    """Programmable :class:`ControlChannel` for unit tests.

    ``execute_handler`` is a callback that receives ``(request_id,
    block_name)`` and the runtime's work_area, and must return a
    response dict.  It can write files into the per-request
    output directory so the executor's output-collection pipeline
    exercises.
    """

    def __init__(
        self,
        *,
        work_area: Path,
        health_ready_after: int = 0,
        execute_handler: Any = None,
    ):
        self._work_area = work_area
        self._health_ready_after = health_ready_after
        self._health_calls = 0
        self._execute_handler = (
            execute_handler or (lambda req_id, block_name: {
                "status": "succeeded"})
        )
        self.executed_requests: list[tuple[str, str]] = []
        self.cancelled_requests: list[str] = []
        # Used by tests that want to pause execute() so stop()
        # can race.
        self.execute_gate: threading.Event | None = None
        self.execute_released: threading.Event = threading.Event()

    def health(self, timeout_s: float) -> bool:
        self._health_calls += 1
        return self._health_calls > self._health_ready_after

    def execute(
        self, request_id: str, block_name: str, timeout_s: float,
    ) -> dict[str, Any]:
        self.executed_requests.append((request_id, block_name))
        if self.execute_gate is not None:
            self.execute_gate.wait()
        self.execute_released.set()
        return self._execute_handler(request_id, block_name)

    def cancel(self, request_id: str, timeout_s: float) -> None:
        self.cancelled_requests.append(request_id)
        if self.execute_gate is not None:
            self.execute_gate.set()


# ─── Fixtures ──────────────────────────────────────────────────


def _make_template(
    *,
    block_name: str = "arc_engine",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    output_kinds: dict[str, str] | None = None,
    stop_timeout_s: int = 5,
) -> Template:
    """Build a minimal template with one workspace-persistent block."""
    inputs = inputs or []
    outputs = outputs or []
    output_kinds = output_kinds or {}
    artifact_names = set(inputs) | set(outputs)
    artifacts = [
        ArtifactDeclaration(
            name=n, kind=output_kinds.get(n, "copy"))
        for n in artifact_names
    ]
    block = BlockDefinition(
        name=block_name,
        image="test:latest",
        runner="container",
        lifecycle="workspace_persistent",
        inputs=[
            InputSlot(name=n, container_path=f"/input/{n}")
            for n in inputs
        ],
        outputs=[
            OutputSlot(name=n, container_path=f"/output/{n}")
            for n in outputs
        ],
        state=False,
        stop_timeout_s=stop_timeout_s,
    )
    return Template(
        name="t", artifacts=artifacts, blocks=[block])


def _make_workspace(tmp_path: Path, template: Template) -> Workspace:
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


class _ExecutorTestHarness:
    """Bundles an executor with its injected fakes.

    Keeps each test's setup to 3–4 lines and gives tests access
    to the channel factory so they can introspect what the
    executor has done.
    """

    def __init__(
        self,
        template: Template,
        *,
        work_area: Path,
        execute_handler: Any = None,
        health_ready_after: int = 0,
    ):
        self.channels: list[FakeChannel] = []
        self.popens: list[FakePopen] = []
        self._work_area = work_area
        self._execute_handler = execute_handler
        self._health_ready_after = health_ready_after

        def _factory(host, port):
            ch = FakeChannel(
                work_area=self._work_area,
                health_ready_after=self._health_ready_after,
                execute_handler=self._execute_handler,
            )
            self.channels.append(ch)
            return ch

        self.executor = RequestResponseExecutor(
            template,
            channel_factory=_factory,
            startup_timeout_s=1.0,
            execute_timeout_s=2.0,
            cancel_timeout_s=0.5,
        )


@pytest.fixture
def template() -> Template:
    return _make_template(outputs=["game_step"])


@pytest.fixture
def workspace(
    tmp_path: Path, template: Template,
) -> Workspace:
    return _make_workspace(tmp_path, template)


# ─── Tests ─────────────────────────────────────────────────────


class TestLaunchValidation:
    def test_unknown_block_raises(self, workspace: Workspace):
        template = _make_template(block_name="arc_engine")
        executor = RequestResponseExecutor(template)
        with pytest.raises(ValueError, match="not found"):
            executor.launch("bogus", workspace, input_bindings={})

    def test_one_shot_block_rejected(
        self, workspace: Workspace,
    ):
        block = BlockDefinition(
            name="step",
            image="test:latest",
            runner="container",
            lifecycle="one_shot",
            inputs=[],
            outputs=[],
        )
        template = Template(
            name="t", artifacts=[], blocks=[block])
        executor = RequestResponseExecutor(template)
        with pytest.raises(
            ValueError, match="workspace_persistent",
        ):
            executor.launch(
                "step", workspace, input_bindings={})

    def test_lifecycle_runner_rejected(
        self, workspace: Workspace,
    ):
        block = BlockDefinition(
            name="logical",
            runner="lifecycle",
            runner_justification="test fixture",
        )
        template = Template(
            name="t", artifacts=[], blocks=[block])
        executor = RequestResponseExecutor(template)
        with pytest.raises(
            ValueError, match="only runs container blocks",
        ):
            executor.launch(
                "logical", workspace, input_bindings={})


class TestRuntimeStartup:
    def test_happy_path_starts_container_and_waits_for_health(
        self, workspace: Workspace, template: Template,
    ):
        work_area = workspace.path / "runtimes" / "arc_engine"
        harness = _ExecutorTestHarness(
            template, work_area=work_area,
            health_ready_after=2,  # two failed probes then ready
        )

        start_calls: list[str] = []

        def _fake_start(cc, name):
            start_calls.append(name)
            return "cid"

        with patch(
            "flywheel.executor._run_detached_container",
            side_effect=_fake_start,
        ), patch(
            "flywheel.executor._docker_ps_find",
            return_value=None,
        ), patch(
            "flywheel.executor._docker_wait_gone",
            return_value=True,
        ):
            handle = harness.executor.launch(
                "arc_engine", workspace, input_bindings={})
            result = handle.wait()

        assert result.status == "succeeded"
        assert len(start_calls) == 1
        # The executor recorded exactly one runtime.  The
        # attachment key is derived from the workspace path
        # digest rather than the workspace name, so pin length
        # rather than a literal value.
        keys = harness.executor.attached_keys()
        assert len(keys) == 1
        assert keys[0].endswith("::arc_engine")

    def test_startup_health_timeout_raises(
        self, workspace: Workspace, template: Template,
    ):
        """A container that never reports healthy must fail
        loudly rather than leaving a dead runtime in the
        registry."""
        work_area = workspace.path / "runtimes" / "arc_engine"

        class _NeverReadyChannel:
            def health(self, timeout_s: float) -> bool:
                return False

            def execute(self, *_args, **_kw):
                raise AssertionError("should not be called")

            def cancel(self, *_args, **_kw):
                return None

        def _factory(host, port):
            return _NeverReadyChannel()

        executor = RequestResponseExecutor(
            template,
            channel_factory=_factory,
            startup_timeout_s=0.2,
        )

        term_killed: list[str] = []

        def _fake_signal(name, signal):
            term_killed.append(signal)

        with patch(
            "flywheel.executor._run_detached_container",
            return_value="cid",
        ), patch(
            "flywheel.executor._docker_ps_find",
            return_value=None,
        ), patch(
            "flywheel.executor._docker_kill",
            side_effect=_fake_signal,
        ), patch(
            "flywheel.executor._docker_wait_gone",
            return_value=True,
        ):
            with pytest.raises(
                ControlChannelError,
                match="did not report ready",
            ):
                executor.launch(
                    "arc_engine", workspace, input_bindings={})
        assert "TERM" in term_killed
        # No runtime registered for the failed startup.
        assert executor.attached_keys() == []


class TestRequestSerialization:
    def test_attachment_key_reused_across_requests(
        self, workspace: Workspace, template: Template,
    ):
        """Two launches for the same block + workspace share one
        runtime; the second request does not start a new
        container."""
        work_area = workspace.path / "runtimes" / "arc_engine"

        def _handler(req_id, block):
            # Emit a tiny output into the per-request dir so the
            # collection pipeline has something to commit.
            return {"status": "succeeded"}

        harness = _ExecutorTestHarness(
            template, work_area=work_area,
            execute_handler=_handler,
        )

        start_calls: list[str] = []

        def _fake_start(cc, name):
            start_calls.append(name)
            return "cid"

        with patch(
            "flywheel.executor._run_detached_container",
            side_effect=_fake_start,
        ), patch(
            "flywheel.executor._docker_ps_find",
            return_value=None,
        ), patch(
            "flywheel.executor._docker_wait_gone",
            return_value=True,
        ):
            h1 = harness.executor.launch(
                "arc_engine", workspace, input_bindings={})
            h1.wait()
            h2 = harness.executor.launch(
                "arc_engine", workspace, input_bindings={})
            h2.wait()

        assert len(start_calls) == 1, (
            "second launch should attach to the running runtime")
        # Both requests went through the same channel.
        assert len(harness.channels) == 1
        assert len(harness.channels[0].executed_requests) == 2

    def test_concurrent_launches_serialize_host_side(
        self, workspace: Workspace, template: Template,
    ):
        """The executor's per-runtime request lock prevents two
        concurrent requests from reaching the container at the
        same time.  Test uses a pre-configured gate so the first
        request blocks deterministically in the channel's
        ``execute()``; the second launch blocks on acquiring the
        runtime lock."""
        work_area = workspace.path / "runtimes" / "arc_engine"
        gate = threading.Event()
        order: list[str] = []
        enter_first = threading.Event()
        enter_second = threading.Event()
        seen = [False]

        def _handler(req_id, block):
            which = "first" if not seen[0] else "second"
            seen[0] = True
            order.append(f"enter-{which}")
            if which == "first":
                enter_first.set()
                gate.wait(timeout=5.0)
            else:
                enter_second.set()
            order.append(f"exit-{which}")
            return {"status": "succeeded"}

        harness = _ExecutorTestHarness(
            template, work_area=work_area,
            execute_handler=_handler,
        )

        with patch(
            "flywheel.executor._run_detached_container",
            return_value="cid",
        ), patch(
            "flywheel.executor._docker_ps_find",
            return_value=None,
        ), patch(
            "flywheel.executor._docker_wait_gone",
            return_value=True,
        ):
            # Both launches return immediately; the POST threads
            # queue on the runtime's request lock.
            h1 = harness.executor.launch(
                "arc_engine", workspace, input_bindings={})
            h2 = harness.executor.launch(
                "arc_engine", workspace, input_bindings={})

            # First handler is running; second is queued and
            # must not have entered its handler yet.
            assert enter_first.wait(timeout=1.0)
            assert not enter_second.wait(timeout=0.2)

            # Release the first; second unblocks once the lock
            # drops.
            gate.set()
            h1.wait()
            h2.wait()

        # The runtime lock enforced ordering: first fully
        # exited before second entered.
        assert order == [
            "enter-first", "exit-first",
            "enter-second", "exit-second",
        ]


class TestRequestCancellation:
    def test_stop_cancels_in_flight_request(
        self, workspace: Workspace, template: Template,
    ):
        """``stop()`` on the request handle fires ``POST /cancel``
        on the channel and unblocks ``wait()``.  The execution
        record's ``stop_reason`` reflects the cancel."""
        work_area = workspace.path / "runtimes" / "arc_engine"
        gate = threading.Event()
        started = threading.Event()

        def _handler(req_id, block):
            started.set()
            gate.wait(timeout=5.0)
            return {"status": "succeeded"}

        # The fake channel's cancel() fires the gate so the
        # blocked execute() can return, modelling a container
        # that honors cancellation.
        class _CancellingChannel:
            def __init__(self):
                self.executed: list[tuple[str, str]] = []
                self.cancelled: list[str] = []

            def health(self, timeout_s: float) -> bool:
                return True

            def execute(
                self, request_id: str, block_name: str,
                timeout_s: float,
            ) -> dict[str, Any]:
                self.executed.append((request_id, block_name))
                return _handler(request_id, block_name)

            def cancel(
                self, request_id: str, timeout_s: float,
            ) -> None:
                self.cancelled.append(request_id)
                gate.set()

        channel_holder: dict[str, _CancellingChannel] = {}

        def _factory(host, port):
            ch = _CancellingChannel()
            channel_holder["ch"] = ch
            return ch

        executor = RequestResponseExecutor(
            template,
            channel_factory=_factory,
            startup_timeout_s=1.0,
            execute_timeout_s=5.0,
            cancel_timeout_s=0.5,
        )

        with patch(
            "flywheel.executor._run_detached_container",
            return_value="cid",
        ), patch(
            "flywheel.executor._docker_ps_find",
            return_value=None,
        ), patch(
            "flywheel.executor._docker_wait_gone",
            return_value=True,
        ):
            handle = executor.launch(
                "arc_engine", workspace, input_bindings={})
            # Wait until the handler has actually entered
            # execute() before firing stop — otherwise we may
            # race the thread and never block.
            started.wait(timeout=2.0)
            handle.stop(reason="test_cancel")
            result = handle.wait()

        assert result.status == "interrupted"
        execution = workspace.executions[result.execution_id]
        assert execution.stop_reason == "test_cancel"
        assert channel_holder["ch"].cancelled, (
            "cancel() must have been called on the channel"
        )


class TestExecuteResponseShapes:
    def test_failed_status_maps_to_invoke_failure_phase(
        self, workspace: Workspace, template: Template,
    ):
        def _handler(req_id, block):
            return {
                "status": "failed",
                "error": "engine raised ValueError",
            }

        harness = _ExecutorTestHarness(
            _make_template(outputs=["game_step"]),
            work_area=workspace.path / "runtimes" / "arc_engine",
            execute_handler=_handler,
        )
        with patch(
            "flywheel.executor._run_detached_container",
            return_value="cid",
        ), patch(
            "flywheel.executor._docker_ps_find",
            return_value=None,
        ), patch(
            "flywheel.executor._docker_wait_gone",
            return_value=True,
        ):
            handle = harness.executor.launch(
                "arc_engine", workspace, input_bindings={})
            result = handle.wait()

        assert result.status == "failed"
        execution = workspace.executions[result.execution_id]
        assert execution.failure_phase == runtime.FAILURE_INVOKE
        assert "engine raised" in (execution.error or "")

    def test_transport_error_maps_to_invoke_failure(
        self, workspace: Workspace, template: Template,
    ):
        def _handler(req_id, block):
            raise ControlChannelError("connection refused")

        harness = _ExecutorTestHarness(
            _make_template(outputs=["game_step"]),
            work_area=workspace.path / "runtimes" / "arc_engine",
            execute_handler=_handler,
        )
        with patch(
            "flywheel.executor._run_detached_container",
            return_value="cid",
        ), patch(
            "flywheel.executor._docker_ps_find",
            return_value=None,
        ), patch(
            "flywheel.executor._docker_wait_gone",
            return_value=True,
        ):
            handle = harness.executor.launch(
                "arc_engine", workspace, input_bindings={})
            result = handle.wait()

        assert result.status == "failed"
        execution = workspace.executions[result.execution_id]
        assert execution.failure_phase == runtime.FAILURE_INVOKE
        assert "transport error" in (execution.error or "")


class TestOutputCollection:
    def test_output_file_committed_as_artifact(
        self, workspace: Workspace, template: Template,
    ):
        """When the handler writes a file into the per-request
        output dir, the executor commits it as an artifact
        instance bound to the execution."""
        work_area = workspace.path / "runtimes" / "arc_engine"

        def _handler_factory(harness: _ExecutorTestHarness):
            def _handler(req_id, block):
                # Locate the per-request output tree.
                out_dir = (
                    work_area / "requests" / req_id
                    / "output" / "game_step")
                (out_dir / "step.json").write_text(
                    '{"action": 6, "frame": "..."}',
                    encoding="utf-8",
                )
                return {"status": "succeeded"}
            return _handler

        harness = _ExecutorTestHarness(
            template, work_area=work_area)
        # Late-bind the handler so it can see the harness.
        harness._execute_handler = _handler_factory(harness)

        with patch(
            "flywheel.executor._run_detached_container",
            return_value="cid",
        ), patch(
            "flywheel.executor._docker_ps_find",
            return_value=None,
        ), patch(
            "flywheel.executor._docker_wait_gone",
            return_value=True,
        ):
            handle = harness.executor.launch(
                "arc_engine", workspace, input_bindings={})
            result = handle.wait()

        assert result.status == "succeeded"
        assert "game_step" in result.output_bindings
        aid = result.output_bindings["game_step"]
        artifact_dir = workspace.path / "artifacts" / aid
        assert (artifact_dir / "step.json").is_file()


class TestReattachment:
    def test_label_mismatch_refuses_attachment(
        self, workspace: Workspace, template: Template,
    ):
        """A pre-existing container with the same name but
        foreign labels must not be attached to; the executor
        raises rather than risk racing Docker with a new
        container of the same name."""
        work_area = workspace.path / "runtimes" / "arc_engine"
        harness = _ExecutorTestHarness(
            template, work_area=work_area)

        stale_info = {
            "_running": "true",
            _RUNTIME_LABEL_WORKSPACE: "other-workspace",
            _RUNTIME_LABEL_BLOCK: "arc_engine",
            _RUNTIME_LABEL_PROTOCOL: (
                REQUEST_RESPONSE_PROTOCOL_VERSION),
            _RUNTIME_LABEL_LIFECYCLE: "workspace_persistent",
            "flywheel.control_port": "65000",
        }

        with patch(
            "flywheel.executor._run_detached_container",
            return_value="cid",
        ), patch(
            "flywheel.executor._docker_ps_find",
            return_value=stale_info,
        ):
            with pytest.raises(
                ControlChannelError,
                match="foreign labels",
            ):
                harness.executor.launch(
                    "arc_engine", workspace,
                    input_bindings={})
        assert harness.executor.attached_keys() == []


class TestShutdown:
    def test_shutdown_unknown_key_is_noop(
        self, workspace: Workspace, template: Template,
    ):
        executor = RequestResponseExecutor(template)
        # Must not raise.
        executor.shutdown("unknown::key")

    def test_shutdown_fresh_process_falls_back_to_docker(
        self, workspace: Workspace, template: Template,
    ):
        """An executor constructed in a fresh process has an
        empty in-memory registry.  ``shutdown(key, block_name,
        workspace)`` must still find the container via
        ``docker inspect`` and tear it down — otherwise the CLI
        teardown path is silently a no-op."""
        executor = RequestResponseExecutor(template)

        # Pretend Docker has our container running.  Labels
        # match this workspace+block.
        our_labels = {
            "_running": "true",
            _RUNTIME_LABEL_WORKSPACE: workspace.name,
            _RUNTIME_LABEL_WORKSPACE_PATH: str(
                workspace.path.resolve()),
            _RUNTIME_LABEL_BLOCK: "arc_engine",
            _RUNTIME_LABEL_PROTOCOL: (
                REQUEST_RESPONSE_PROTOCOL_VERSION),
            _RUNTIME_LABEL_LIFECYCLE: "workspace_persistent",
            "flywheel.control_port": "65000",
        }
        kills: list[tuple[str, str]] = []

        def _fake_kill(name, signal):
            kills.append((name, signal))

        with patch(
            "flywheel.executor._docker_ps_find",
            return_value=our_labels,
        ), patch(
            "flywheel.executor._docker_kill",
            side_effect=_fake_kill,
        ), patch(
            "flywheel.executor._docker_wait_gone",
            return_value=True,
        ):
            handled = executor.shutdown(
                "any-key-the-cli-passed",
                block_name="arc_engine",
                workspace=workspace,
            )
        assert handled is True, (
            "shutdown must actually tear down the Docker "
            "container even when the registry is empty"
        )

    def test_leaked_handle_does_not_strand_runtime_lock(
        self, workspace: Workspace, template: Template,
    ):
        """A handle that never gets ``wait()``-ed must not
        permanently block subsequent requests.  Regression
        guard: the runtime lock used to be acquired on the
        calling thread in ``__init__`` and released only in
        ``wait()``'s finally, making this scenario wedge the
        whole runtime."""
        work_area = workspace.path / "runtimes" / "arc_engine"
        harness = _ExecutorTestHarness(
            template, work_area=work_area)

        with patch(
            "flywheel.executor._run_detached_container",
            return_value="cid",
        ), patch(
            "flywheel.executor._docker_ps_find",
            return_value=None,
        ), patch(
            "flywheel.executor._docker_wait_gone",
            return_value=True,
        ):
            # Leak the first handle — deliberately do not call
            # wait().  Its POST thread still runs and releases
            # the runtime lock when done.
            leaked = harness.executor.launch(
                "arc_engine", workspace, input_bindings={})
            # Give the POST thread a moment to run.
            for _ in range(40):
                if leaked._thread.is_alive() is False:
                    break
                threading.Event().wait(0.05)
            assert not leaked._thread.is_alive(), (
                "POST thread should have completed "
                "independent of wait()"
            )

            # A subsequent launch must be able to acquire the
            # runtime lock.  If the leaked handle stranded it,
            # this wait() would hang forever.
            second = harness.executor.launch(
                "arc_engine", workspace, input_bindings={})
            result = second.wait()

        assert result.status == "succeeded"

    def test_shutdown_sends_sentinel_then_term_then_kill(
        self, workspace: Workspace, template: Template,
    ):
        work_area = workspace.path / "runtimes" / "arc_engine"
        harness = _ExecutorTestHarness(
            template, work_area=work_area)

        signals: list[str] = []

        def _fake_kill(name, signal):
            signals.append(signal)

        wait_gone_calls: list[float] = []

        def _fake_wait_gone(name, timeout_s):
            wait_gone_calls.append(timeout_s)
            # Cooperative phase: report still-running so the
            # executor escalates to TERM.  Post-TERM phase:
            # report gone so teardown completes.
            return len(wait_gone_calls) > 1

        with patch(
            "flywheel.executor._run_detached_container",
            return_value="cid",
        ), patch(
            "flywheel.executor._docker_ps_find",
            return_value=None,
        ), patch(
            "flywheel.executor._docker_kill",
            side_effect=_fake_kill,
        ), patch(
            "flywheel.executor._docker_wait_gone",
            side_effect=_fake_wait_gone,
        ):
            handle = harness.executor.launch(
                "arc_engine", workspace, input_bindings={})
            handle.wait()
            key = harness.executor.attached_keys()[0]
            harness.executor.shutdown(key)

        # Cooperative wait first, then TERM.  KILL not needed
        # because the post-TERM wait_gone returns True.
        assert signals == ["TERM"]
        # work_area is torn down by shutdown().
        assert not work_area.exists()
        assert harness.executor.attached_keys() == []
