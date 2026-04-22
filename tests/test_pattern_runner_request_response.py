"""End-to-end: :class:`PatternRunner` driving a real container executor.

Earlier slices proved the runner is structurally agent-agnostic at
the :class:`flywheel.executor.BlockExecutor` seam by exercising the
dispatch path against a hand-rolled
:class:`flywheel.tests.test_pattern_runner_executor_dispatch._RecordingExecutor`.
That fake matches our *understanding* of the launch contract; this
module pins the same path against the **real**
:class:`flywheel.executor.RequestResponseExecutor` so a future
signature drift in the executor (or in the runner's launch-kwarg
wiring) trips the suite instead of silently producing a runtime
error in cyberloop.

What is real here
-----------------

* :class:`flywheel.workspace.Workspace` — created on disk.
* :class:`RequestResponseExecutor` — instantiated and called by
  the runner; runtime registration, the per-request POST thread,
  per-request directory layout, and :class:`BlockExecution`
  recording onto the workspace ledger all run for real.  The
  test block deliberately declares no input or output slots, so
  per-mount staging and ``_collect_outputs`` are exercised on
  their empty paths only — see the dedicated executor suite for
  the populated-slot variants.
* :class:`PatternRunner` — wired with a real
  :class:`flywheel.run_defaults.RunDefaults` and an
  ``executor_factory`` that returns the real executor for every
  block lookup.

What is stubbed
---------------

* Docker.  ``_run_detached_container``, ``_docker_ps_find``, and
  ``_docker_wait_gone`` are patched so no real container is
  started or torn down.
* The control channel.  A :class:`_FakeChannel` handles the
  executor's ``health`` / ``execute`` / ``cancel`` calls in-process
  so the test runs without binding to a real localhost port.
* Background threads from the executor (request POST, runtime
  health probe) are still real; only the network calls inside
  them are faked.

What is *not* in this module
----------------------------

The agent battery.  This file deliberately does not import
:mod:`flywheel.agent_executor` or :mod:`flywheel.agent_handoff`;
the whole point is to demonstrate a working
``Pattern → PatternRunner → RequestResponseExecutor`` loop with
no agent code in the call graph.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from flywheel.executor import RequestResponseExecutor
from flywheel.pattern import (
    BlockInstance,
    ContinuousTrigger,
    Pattern,
)
from flywheel.pattern_runner import PatternRunner
from flywheel.run_defaults import RunDefaults
from flywheel.template import (
    ArtifactDeclaration,
    BlockDefinition,
    InputSlot,
    OutputSlot,
    Template,
)
from flywheel.workspace import Workspace


# ── Fakes ──────────────────────────────────────────────────────


class _FakeChannel:
    """In-memory :class:`flywheel.executor.ControlChannel`.

    The executor uses the channel for three things: ``health`` to
    confirm the container is up, ``execute`` to run a request, and
    ``cancel`` to abort an in-flight request.  Tests interact via
    ``executed_requests`` (one entry per executed request) and the
    constructor's ``execute_handler`` (so tests that need richer
    behaviour can substitute a callable).
    """

    def __init__(
        self,
        *,
        execute_handler: Any = None,
    ):
        self._execute_handler = execute_handler or (
            lambda req_id, block_name: {"status": "succeeded"}
        )
        self.executed_requests: list[tuple[str, str]] = []
        self.cancelled_requests: list[str] = []

    def health(self, timeout_s: float) -> bool:
        del timeout_s
        return True

    def execute(
        self, request_id: str, block_name: str, timeout_s: float,
    ) -> dict[str, Any]:
        del timeout_s
        self.executed_requests.append((request_id, block_name))
        return self._execute_handler(request_id, block_name)

    def cancel(self, request_id: str, timeout_s: float) -> None:
        del timeout_s
        self.cancelled_requests.append(request_id)


# ── Helpers ────────────────────────────────────────────────────


def _make_template(
    *,
    block_name: str = "engine",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> Template:
    """Single workspace-persistent block; no state, no slots."""
    inputs = inputs or []
    outputs = outputs or []
    artifact_names = set(inputs) | set(outputs)
    artifacts = [
        ArtifactDeclaration(name=n, kind="copy")
        for n in artifact_names
    ]
    block = BlockDefinition(
        name=block_name,
        image="example/engine:latest",
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
    )
    return Template(name="t", artifacts=artifacts, blocks=[block])


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


def _make_executor_with_channel(
    template: Template, channel: _FakeChannel,
) -> RequestResponseExecutor:
    """RequestResponseExecutor wired to ``channel`` for every runtime.

    The factory closure ignores ``host`` / ``port`` because the
    fake channel does not actually bind anywhere — it only needs
    to satisfy the executor's protocol surface.
    """
    return RequestResponseExecutor(
        template,
        channel_factory=lambda host, port: channel,
        startup_timeout_s=1.0,
        execute_timeout_s=2.0,
        cancel_timeout_s=0.5,
    )


def _continuous_pattern(
    *, instance_name: str = "engine", block_name: str = "engine",
) -> Pattern:
    """One prompt-less continuous instance — no agent battery touched."""
    return Pattern(
        name="engine-loop",
        roles=[],
        instances=[
            BlockInstance(
                name=instance_name,
                block=block_name,
                trigger=ContinuousTrigger(),
                cardinality=1,
                prompt=None,
            ),
        ],
    )


# ── Tests ──────────────────────────────────────────────────────


@pytest.fixture
def docker_patches():
    """Stub the Docker calls the executor would otherwise make.

    Without these, the runtime startup path tries to spawn a real
    container, fails to find a docker daemon, and raises before the
    runner ever reaches its main loop.  We also stub ``ps`` (so
    "is there a leftover container?" returns "no") and
    ``wait_gone`` (so teardown does not block on a real container
    actually disappearing).
    """
    with patch(
        "flywheel.executor._run_detached_container",
        return_value="fake-container-id",
    ), patch(
        "flywheel.executor._docker_ps_find",
        return_value=None,
    ), patch(
        "flywheel.executor._docker_wait_gone",
        return_value=True,
    ):
        yield


def test_pattern_runner_drives_request_response_executor_to_success(
    tmp_path: Path, docker_patches,
) -> None:
    """A continuous prompt-less instance runs end-to-end via the RR executor.

    Pins three things at once:

    1. The runner's ``executor.launch`` call shape is accepted by
       :meth:`RequestResponseExecutor.launch` (signature match,
       not just protocol shape).
    2. The executor's ``ExecutionHandle`` cooperates with the
       runner's lifecycle expectations
       (``is_alive`` → False after the request completes,
       ``wait`` returns a succeeded :class:`ExecutionResult`).
    3. The executor records a :class:`BlockExecution` on the real
       workspace ledger, tagged with the runner's ``run_id`` so
       cadence accounting works for pure-container patterns the
       same way it does for agent patterns.
    """
    template = _make_template(block_name="engine")
    ws = _make_workspace(tmp_path, template)
    channel = _FakeChannel()
    executor = _make_executor_with_channel(template, channel)
    pattern = _continuous_pattern()

    runner = PatternRunner(
        pattern,
        defaults=RunDefaults(
            workspace=ws,
            template=template,
            project_root=tmp_path,
        ),
        executor_factory=lambda _block_def: executor,
        poll_interval_s=0.01,
        max_total_runtime_s=5.0,
    )

    result = runner.run()

    assert result.cohorts_by_role == {"engine": 1}
    assert result.agents_launched == 1
    assert len(channel.executed_requests) == 1
    request_id, block_name = channel.executed_requests[0]
    assert block_name == "engine"
    assert request_id  # non-empty UUID

    # The executor recorded one BlockExecution on the ledger,
    # tagged with the runner's run_id.  This is the critical
    # cross-component proof: cadence accounting and run grouping
    # work for pure-container patterns the same way they do for
    # agent patterns.
    assert len(ws.executions) == 1
    execution = next(iter(ws.executions.values()))
    assert execution.block_name == "engine"
    assert execution.status == "succeeded"
    assert execution.run_id == result.run_id
    assert execution.runner == "container"


def test_pattern_runner_forwards_extra_env_to_request_response_runtime(
    tmp_path: Path, docker_patches,
) -> None:
    """Container-extras kwargs land on the executor's runtime startup.

    The runner passes ``extra_env`` as an explicit
    :meth:`RequestResponseExecutor.launch` kwarg (per the
    container-extras seam).  The executor consumes them on
    runtime *startup* — so we assert against
    ``_run_detached_container``'s recorded
    :class:`flywheel.runtime.ContainerConfig`, which is the
    real attachment point cyberloop's training loop will
    eventually rely on.
    """
    template = _make_template(block_name="engine")
    ws = _make_workspace(tmp_path, template)
    channel = _FakeChannel()
    executor = _make_executor_with_channel(template, channel)
    pattern = Pattern(
        name="engine-loop",
        roles=[],
        instances=[
            BlockInstance(
                name="engine",
                block="engine",
                trigger=ContinuousTrigger(),
                cardinality=1,
                prompt=None,
                extra_env={
                    "RUN_TAG": "skeleton",
                    "SEED": "1234",
                },
            ),
        ],
    )

    captured: dict[str, Any] = {}

    def _capture_start(container_config, runtime_label):
        del runtime_label
        captured["env"] = dict(container_config.env)
        return "fake-container-id"

    with patch(
        "flywheel.executor._run_detached_container",
        side_effect=_capture_start,
    ), patch(
        "flywheel.executor._docker_ps_find",
        return_value=None,
    ), patch(
        "flywheel.executor._docker_wait_gone",
        return_value=True,
    ):
        runner = PatternRunner(
            pattern,
            defaults=RunDefaults(
                workspace=ws,
                template=template,
                project_root=tmp_path,
            ),
            executor_factory=lambda _block_def: executor,
            poll_interval_s=0.01,
            max_total_runtime_s=5.0,
        )
        runner.run()

    assert captured["env"].get("RUN_TAG") == "skeleton"
    assert captured["env"].get("SEED") == "1234"
