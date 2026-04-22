"""Block executor abstractions and concrete implementations.

Defines the protocol for executing blocks and the concrete
executor types.  A block executor takes a block definition, a
workspace, and input bindings, and produces an execution result
with output artifacts.

- **ProcessExitExecutor**: process-exit protocol.  One container
  run per execution.  Stages inputs, starts the container, waits
  for exit, captures ``/state/`` (if declared), collects outputs
  as artifact instances, records the execution.
- **ProcessExecutor**: Runs a block as a local subprocess.
  For trusted, host-local processes like game servers.

``runner: lifecycle`` blocks have no executor — they are
recorded directly by
:class:`flywheel.local_block.LocalBlockRecorder`.

All executors satisfy the ``BlockExecutor`` protocol and return
an ``ExecutionHandle`` from ``launch()``.  The external
container runtime contract these executors implement lives in
``cyber-root/substrate-contract.md``.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from flywheel import runtime
from flywheel.artifact import ArtifactInstance, BlockExecution
from flywheel.post_check import (
    HaltDirective,
    PostCheckCallable,
    PostCheckContext,
)
from flywheel.container import (
    ContainerConfig,
    start_container,
)
from flywheel.input_staging import (
    StagingError,
    cleanup_staged_inputs,
    stage_artifact_instances,
)
from flywheel.template import BlockDefinition, Template
from flywheel.workspace import Workspace

# Sentinel exit code for "the container never produced one."
# Used when container startup raises before an exit status
# exists (Docker daemon gone, container failed to start, etc.).
# ``-1`` distinguishes this case from a real successful exit in
# the returned :class:`ExecutionResult`; the
# :attr:`BlockExecution.status` field is the authoritative
# signal, but API-level callers reading just ``exit_code``
# should see a non-zero value.
INVOKE_FAILURE_EXIT_CODE: int = -1

# Grace period between SIGTERM and SIGKILL when forced
# termination is needed.  Short because well-behaved SIGTERM
# handlers wrap up in a few seconds; anything slower than this
# is treated as "unresponsive to TERM" and escalated to KILL.
_TERM_TO_KILL_GRACE_S: float = 5.0


@dataclass(frozen=True)
class ExecutionResult:
    """Result of a block execution via any executor.

    Attributes:
        exit_code: Process exit code (0 = success).
        elapsed_s: Wall-clock time in seconds.
        output_bindings: Maps output slot names to artifact
            instance IDs created by this execution.
        execution_id: The workspace execution ID recorded.
        status: Outcome: ``"succeeded"``, ``"failed"``, or
            ``"interrupted"``.
    """

    exit_code: int
    elapsed_s: float
    output_bindings: dict[str, str]
    execution_id: str
    status: str


@dataclass(frozen=True)
class ExecutionEvent:
    """Fired when an executor completes a block execution.

    Observers register via the execution channel.

    Attributes:
        executor_type: Which executor ran (``"container"``,
            ``"process"``).
        block_name: The block that was executed.
        execution_id: The workspace execution ID.
        status: Outcome (``"succeeded"``, ``"failed"``, etc.).
        output_bindings: Maps output slot names to artifact IDs.
        outputs_data: Raw output dicts when meaningful (currently
            unused; reserved for future executors).
    """

    executor_type: str
    block_name: str
    execution_id: str
    status: str
    output_bindings: dict[str, str] = field(default_factory=dict)
    outputs_data: dict[str, Any] | None = None


class ExecutionHandle:
    """Handle to a running or completed block execution.

    Returned by ``BlockExecutor.launch()``.  The caller **must**
    call ``wait()`` exactly once to collect the result.

    For asynchronous executors (Container, Process), ``wait()``
    blocks until the execution completes.  Synchronous handles
    (see :class:`SyncExecutionHandle`) return immediately.
    """

    def is_alive(self) -> bool:
        """Check if the execution is still running."""
        raise NotImplementedError

    def stop(self, reason: str = "requested") -> None:
        """Request a graceful stop."""
        raise NotImplementedError

    def wait(self) -> ExecutionResult:
        """Block until completion and return the result."""
        raise NotImplementedError


class SyncExecutionHandle(ExecutionHandle):
    """Handle for an execution that completed synchronously.

    Used by blocking executor paths that have a result in hand at
    ``launch()`` time.  ``wait()`` returns the pre-computed result
    immediately.
    """

    def __init__(self, result: ExecutionResult):
        """Initialize with a pre-computed result."""
        self._result = result
        self._waited = False

    def is_alive(self) -> bool:
        """Always False — execution already completed."""
        return False

    def stop(self, reason: str = "requested") -> None:
        """No-op — execution already completed."""

    def wait(self) -> ExecutionResult:
        """Return the pre-computed result.

        Raises:
            RuntimeError: If ``wait()`` has already been called.
        """
        if self._waited:
            raise RuntimeError(
                "wait() already called on this handle")
        self._waited = True
        return self._result


@runtime_checkable
class BlockExecutor(Protocol):
    """Protocol for executing a block definition.

    The seam every runner (the ad-hoc ``flywheel run *`` commands
    and the pattern runner) talks to.  Implementations receive a
    block definition by name, the workspace whose ledger they
    will append to, and input slot bindings; they return an
    :class:`ExecutionHandle` whose ``wait()`` produces an
    :class:`ExecutionResult` and an appended :class:`BlockExecution`
    record.

    Runners must depend only on this protocol — never on a
    concrete executor class, never on agent-specific surfaces.
    Batteries-included blocks (the agent battery being the
    canonical example) wrap their machinery as a
    ``BlockExecutor`` implementation; the runner cannot tell them
    apart from a generic container executor.

    Per-launch context falls into three buckets:

    * **Bindings** — :attr:`input_bindings`, :attr:`overrides`,
      :attr:`allowed_blocks`: what this single execution is
      processing.
    * **Workspace identity** — :attr:`execution_id`,
      :attr:`state_lineage_id`, :attr:`run_id`: how this
      execution slots into the durable ledger.
    * **Executor-specific extras** (not on the protocol):
      container-shaped executors additionally accept
      ``extra_env``, ``extra_mounts``, ``extra_docker_args``;
      callers using those kwargs implicitly assume the executor
      is container-shaped.  See ``executors.md`` for the full
      conventions.
    """

    def launch(
        self,
        block_name: str,
        workspace: Workspace,
        input_bindings: dict[str, str],
        *,
        execution_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        allowed_blocks: list[str] | None = None,
        state_lineage_id: str | None = None,
        run_id: str | None = None,
    ) -> ExecutionHandle:
        """Launch a block execution.

        Args:
            block_name: Name of the block to execute.  Must
                resolve against the executor's bound template.
            workspace: The flywheel workspace.  The execution
                record is appended to its ledger; output
                artifacts are registered against its declarations.
            input_bindings: Maps input slot names to artifact
                instance IDs already registered in ``workspace``.
            execution_id: Optional pre-assigned execution ID.
                Callers that need to thread an ID through their
                own bookkeeping (e.g. nested handoffs) supply
                one; otherwise the executor mints its own.
            overrides: Free-form per-launch overrides the
                executor may consume.  Container-style executors
                today read CLI flag substitutions; battery
                executors read battery-specific knobs (prompt,
                model, ...).  Unknown keys must be ignored
                silently — validation, if any, is the executor's
                concern.
            allowed_blocks: If set, only these block names are
                permitted.  Used by host-side handoff loops to
                fence what an agent's nested executions can
                touch.
            state_lineage_id: Optional state-chain identifier the
                executor uses to populate ``/state/`` from a
                prior execution's captured state directory.
                Executors with no state concept ignore this.
            run_id: Optional :class:`flywheel.artifact.RunRecord`
                id stamped onto the resulting execution record so
                pattern-level cadence counters can scope to a
                single run.

        Returns:
            An :class:`ExecutionHandle` for monitoring and
            collecting the result.  Callers must call
            :meth:`ExecutionHandle.wait` exactly once.
        """
        ...


def _find_block(
    template: Template, block_name: str,
) -> BlockDefinition | None:
    """Look up a block definition by name in the template."""
    for block in template.blocks:
        if block.name == block_name:
            return block
    return None


# ── ProcessExitExecutor ──────────────────────────────────────────


class _AbortExecution(Exception):
    """Internal signal: a failure phase was recorded; stop the pipeline.

    Raised by :class:`ProcessExitExecutor` when a pipeline phase
    fails.  Keeps the per-phase try/except blocks terse and lets
    the outer ``try/finally`` guarantee cleanup.  Not part of the
    public API.
    """


def _docker_send_signal(container_name: str, signal: str) -> None:
    """Send a signal to a running container by name.

    Extracted so tests can patch the Docker interaction without
    needing a real daemon.  ``check=False`` because the container
    may have already exited between the caller's decision and
    this call — that race is always possible and isn't an error
    on the caller's side.
    """
    subprocess.run(
        ["docker", "kill", "--signal", signal, container_name],
        check=False,
        capture_output=True,
    )


class ContainerExecutionHandle(ExecutionHandle):
    """Handle to a running container execution.

    Returned by :meth:`ProcessExitExecutor.launch` when the
    container has been started.  The caller must call
    :meth:`wait` exactly once to drive the post-exit pipeline
    (state capture, output collection, execution record).
    :meth:`stop` is optional; when called, it runs the two-phase
    cancellation protocol.

    The handle owns a substantial amount of pipeline state (the
    staged inputs, the state mount, the output tempdirs, the
    work-area tempdir).  Cleanup of those happens in
    :meth:`wait`'s ``finally`` block so partial exits leave no
    tempdirs behind.
    """

    def __init__(
        self,
        *,
        process: subprocess.Popen,
        workspace: Workspace,
        block_def: BlockDefinition,
        execution_id: str,
        started_at: datetime,
        start_monotonic: float,
        container_name: str,
        staged_inputs: dict[str, Path],
        state_mount: Path | None,
        output_tempdirs: dict[str, Path],
        work_area: Path,
        input_bindings: dict[str, str],
        state_lineage_id: str | None,
        run_id: str | None = None,
        log_dir: Path | None = None,
        total_timeout_s: float | None = None,
        on_stdout_line: Callable[[str], None] | None = None,
    ):
        """Capture every piece of state :meth:`wait` will need.

        The ``log_dir``, ``total_timeout_s``, and ``on_stdout_line``
        parameters opt in to three orthogonal background threads:

        * When ``log_dir`` is set, two drain threads copy the
          container's stdout and stderr into ``log_dir/stdout.log``
          and ``log_dir/stderr.log`` respectively.  Requires that
          the process was started with ``capture_output=True`` so
          ``process.stdout`` and ``process.stderr`` are real pipes.
        * When ``total_timeout_s`` is set, a watchdog thread
          polls the process and calls :meth:`stop` with reason
          ``"total_timeout"`` if wall-clock elapsed exceeds the
          budget.  Decoupled from stdout drainage: a container
          that stalls silently still hits the timeout.
        * When ``on_stdout_line`` is set and ``log_dir`` is set,
          the stdout drain thread calls the callback for each
          newline-terminated line as it arrives.  Used by callers
          that want to parse structured events from the stream
          without re-reading the log file post-exit.

        Concurrency contract: :meth:`stop` is idempotent and
        lock-protected.  Whichever caller initiates the stop
        (operator, watchdog, or natural-exit racer) wins; the
        others see a no-op.  Exactly one SIGTERM/SIGKILL
        escalation per handle.
        """
        self._process = process
        self._workspace = workspace
        self._block_def = block_def
        self._execution_id = execution_id
        self._started_at = started_at
        self._start_monotonic = start_monotonic
        self._container_name = container_name
        self._staged_inputs = staged_inputs
        self._state_mount = state_mount
        self._output_tempdirs = output_tempdirs
        self._work_area = work_area
        self._input_bindings = input_bindings
        self._state_lineage_id = state_lineage_id
        self._run_id = run_id
        self._log_dir = log_dir
        self._total_timeout_s = total_timeout_s
        self._on_stdout_line = on_stdout_line
        self._stop_reason: str | None = None
        self._stop_lock = threading.Lock()
        self._stop_initiated = False
        self._waited = False

        # Start auxiliary threads.  They run for the lifetime of
        # the handle and are joined in :meth:`wait`'s finally.
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._stdout_thread = threading.Thread(
                target=self._drain_stream,
                args=(self._process.stdout,
                      log_dir / "stdout.log",
                      on_stdout_line),
                name=f"flywheel-stdout-{container_name}",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread = threading.Thread(
                target=self._drain_stream,
                args=(self._process.stderr,
                      log_dir / "stderr.log",
                      None),
                name=f"flywheel-stderr-{container_name}",
                daemon=True,
            )
            self._stderr_thread.start()

        if total_timeout_s is not None and total_timeout_s > 0:
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_body,
                name=f"flywheel-watchdog-{container_name}",
                daemon=True,
            )
            self._watchdog_thread.start()

    @staticmethod
    def _drain_stream(
        pipe: Any,
        log_path: Path,
        on_line: Callable[[str], None] | None,
    ) -> None:
        """Copy a pipe line-by-line to a log file, optionally emitting.

        Runs in a background thread for the container's lifetime.
        Exits when the pipe returns EOF (container closed stdout/
        stderr on exit).  Errors writing the log are swallowed —
        log capture must never mask the real container exit.
        """
        if pipe is None:
            return
        try:
            with open(log_path, "w", encoding="utf-8") as sink:
                for line in pipe:
                    try:
                        sink.write(line)
                        sink.flush()
                    except OSError:
                        pass
                    if on_line is not None:
                        try:
                            on_line(line.rstrip("\n"))
                        except Exception:
                            # Callback errors are the caller's
                            # problem; don't kill the drain.
                            pass
        except Exception:
            pass

    def _watchdog_body(self) -> None:
        """Enforce ``total_timeout_s`` independent of stdout activity.

        Polls the process every second and triggers :meth:`stop`
        when the wall-clock budget is exhausted.  Exits silently
        on natural container exit.  The ``stop()`` body runs
        outside the watchdog thread's frame (via lock hand-off),
        so a slow two-phase cancellation does not pin this thread.
        """
        assert self._total_timeout_s is not None
        deadline = self._start_monotonic + self._total_timeout_s
        while True:
            if self._process.poll() is not None:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop(reason="total_timeout")
                return
            time.sleep(min(remaining, 1.0))

    def is_alive(self) -> bool:
        """True while the container process has no exit code yet."""
        return self._process.poll() is None

    def stop(self, reason: str = "requested") -> None:
        """Run the two-phase cancellation protocol.

        Phase 1 — *cooperative.*  Write the stop sentinel file at
        ``<work_area>/.stop`` (visible inside the container as
        ``/scratch/.stop``) and wait up to the block's
        ``stop_timeout_s`` for the container to exit on its own.

        Phase 2 — *forced.*  If cooperative times out, send
        SIGTERM via ``docker kill`` and wait a short grace.  If
        SIGTERM is ignored too, escalate to SIGKILL.

        Concurrency: the operator, the watchdog thread, and a
        racing natural exit can all land here concurrently.  A
        short lock guards the "has anyone initiated stop?"
        decision; the first caller wins and runs the body, while
        later callers return immediately.  ``reason`` is recorded
        from the winning caller.  The TERM/KILL escalation runs
        outside the lock; subsequent stop() calls do not block on
        it.
        """
        with self._stop_lock:
            if self._stop_initiated:
                return
            if self._process.poll() is not None:
                # Container already exited naturally; mark
                # initiated to keep subsequent stop() calls a
                # no-op but leave stop_reason unset — the status
                # is not "interrupted".
                self._stop_initiated = True
                return
            self._stop_initiated = True
            self._stop_reason = reason

        # Phase 1: cooperative stop via sentinel.
        sentinel = self._work_area / (
            runtime.STOP_SENTINEL_WORKSPACE_RELATIVE)
        # If we can't write the sentinel, skip straight to
        # forced termination; phase 2 guarantees the container
        # stops regardless.
        with contextlib.suppress(OSError):
            sentinel.touch()

        timeout = max(0, self._block_def.stop_timeout_s)
        if timeout > 0:
            try:
                self._process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                pass

        # Phase 2a: SIGTERM.
        _docker_send_signal(self._container_name, "TERM")
        try:
            self._process.wait(timeout=_TERM_TO_KILL_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass

        # Phase 2b: SIGKILL — container cannot refuse.
        _docker_send_signal(self._container_name, "KILL")
        self._process.wait()

    def wait(self) -> ExecutionResult:
        """Block for container exit, then finalize the execution.

        After the container process has exited, runs the
        post-exit pipeline: capture ``/state/``, collect outputs
        into canonical artifact dirs, write the
        :class:`BlockExecution` record with the right
        ``status``, ``failure_phase``, and (for cancelled
        executions) ``stop_reason``.

        Raises:
            RuntimeError: If called more than once on the same
                handle.
        """
        if self._waited:
            raise RuntimeError(
                "wait() already called on this handle")
        self._waited = True

        failure_phase: str | None = None
        error: str | None = None
        state_dir_rel: str | None = None
        output_bindings: dict[str, str] = {}

        try:
            self._process.wait()
            exit_code = self._process.returncode
            finished_at = datetime.now(UTC)
            elapsed = time.monotonic() - self._start_monotonic

            # If the container exited non-zero and wasn't
            # explicitly stopped, treat it as invoke failure.
            if exit_code != 0 and self._stop_reason is None:
                failure_phase = runtime.FAILURE_INVOKE
                error = (
                    f"invoke: container exited with code "
                    f"{exit_code}"
                )

            # State capture runs even on invoke failure —
            # partial state is worth preserving for debugging.
            if self._state_mount is not None:
                try:
                    state_dir_rel = capture_state(
                        self._workspace,
                        self._block_def.name,
                        self._execution_id,
                        self._state_mount,
                    )
                except OSError as e:
                    if failure_phase is None:
                        failure_phase = (
                            runtime.FAILURE_STATE_CAPTURE)
                        error = f"state_capture: {e}"

            # Host-side output builder (if any).  Runs after the
            # container exited and before flywheel's standard
            # artifact collection reads the output tempdirs,
            # giving the block author a place to collapse
            # human-readable intermediate files into the
            # canonical artifact shape.  Builder failures map to
            # ``FAILURE_OUTPUT_COLLECT`` — same phase as a
            # broken collection pass, since conceptually they
            # are both "getting the outputs into artifacts."
            if (failure_phase is None
                    and self._block_def.output_builder
                    is not None):
                try:
                    _run_output_builder(
                        self._block_def,
                        self._output_tempdirs,
                        self._workspace,
                        self._execution_id,
                    )
                except Exception as e:
                    failure_phase = runtime.FAILURE_OUTPUT_COLLECT
                    error = f"output_builder: {e}"

            # Output collection.
            try:
                output_bindings = _collect_outputs(
                    self._workspace,
                    self._block_def,
                    self._output_tempdirs,
                    self._execution_id,
                    finished_at,
                )
            except Exception as e:
                if failure_phase is None:
                    failure_phase = runtime.FAILURE_OUTPUT_COLLECT
                    error = f"output_collect: {e}"
        finally:
            # Join auxiliary threads before tearing down their
            # dependencies (pipes, work_area).  Drain threads
            # exit on pipe EOF; the watchdog exits on a non-
            # None poll().  Explicit joins guarantee log
            # flushes and watchdog-initiated stop() bodies
            # have finished before cleanup runs.
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout=30)
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=30)
            if self._watchdog_thread is not None:
                self._watchdog_thread.join(timeout=30)

            # Cleanup is best-effort; never mask a real failure.
            cleanup_staged_inputs(self._staged_inputs)
            if self._state_mount is not None:
                shutil.rmtree(
                    self._state_mount, ignore_errors=True)
            for path in self._output_tempdirs.values():
                shutil.rmtree(path, ignore_errors=True)
            shutil.rmtree(self._work_area, ignore_errors=True)

        # Status: interrupted if we explicitly cancelled; failed
        # on any failure_phase; succeeded otherwise.
        if self._stop_reason is not None:
            status = "interrupted"
        elif failure_phase is not None:
            status = "failed"
        else:
            status = "succeeded"

        execution = BlockExecution(
            id=self._execution_id,
            block_name=self._block_def.name,
            started_at=self._started_at,
            finished_at=finished_at,
            status=status,
            input_bindings=self._input_bindings,
            output_bindings=output_bindings,
            exit_code=exit_code,
            elapsed_s=elapsed,
            image=self._block_def.image,
            runner="container",
            state_dir=state_dir_rel,
            failure_phase=failure_phase,
            error=error,
            state_lineage_id=self._state_lineage_id,
            stop_reason=self._stop_reason,
            run_id=self._run_id,
        )
        self._workspace.add_execution(execution)
        self._workspace.save()

        return ExecutionResult(
            exit_code=exit_code,
            elapsed_s=elapsed,
            output_bindings=output_bindings,
            execution_id=self._execution_id,
            status=status,
        )


class ProcessExitExecutor:
    """Process-exit protocol executor — one container per execution.

    Implements the container runtime contract for blocks whose
    lifecycle is ``one_shot``:

    1. Stage declared inputs into per-mount tempdirs (never
       mount canonical artifacts).
    2. If the block declares ``state: true``, populate ``/state/``
       from the most recent prior successful execution of the
       same block.
    3. Allocate per-slot output tempdirs.
    4. Launch the container, wait for exit.
    5. Capture ``/state/`` into the workspace's state dir.
    6. Collect output tempdir contents as artifact instances.
    7. Record a :class:`BlockExecution` with the right
       ``failure_phase`` on any failure.

    Failure phases are distinguished so a reader of the ledger
    can tell a broken container body (``invoke``) from broken
    inputs (``stage_in``) from a broken artifact commit
    (``artifact_commit``).  See :mod:`flywheel.runtime`.
    """

    def __init__(
        self,
        template: Template,
        overrides: dict[str, Any] | None = None,
    ):
        """Initialize with a template and optional overrides.

        Args:
            template: The template containing block definitions.
            overrides: Default CLI flag overrides for containers;
                merged with per-launch overrides at execution
                time.
        """
        self._template = template
        self._overrides = overrides

    def launch(
        self,
        block_name: str,
        workspace: Workspace,
        input_bindings: dict[str, str],
        *,
        execution_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        allowed_blocks: list[str] | None = None,
        state_lineage_id: str | None = None,
        run_id: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_mounts: list[tuple[str, str, str]] | None = None,
        extra_docker_args: list[str] | None = None,
        log_dir: Path | None = None,
        total_timeout_s: float | None = None,
        on_stdout_line: Callable[[str], None] | None = None,
    ) -> ExecutionHandle:
        """Start a one-shot container execution.

        The returned handle is live — the container is running.
        Call :meth:`ExecutionHandle.wait` to block for exit and
        finalize the execution; call :meth:`ExecutionHandle.stop`
        first for cooperative-then-forced cancellation.
        ``wait()`` does the post-exit pipeline (state capture,
        output collection, execution record), so a caller that
        never calls ``wait`` leaks both the container and its
        pipeline state.

        On pre-container failures (staging errors, container-
        start exceptions), the returned handle is a
        :class:`SyncExecutionHandle` that's already recorded the
        failure — ``wait()`` returns immediately and ``stop()``
        is a no-op.

        Args:
            block_name: Name of the block to execute.  Must
                exist in the template and declare
                ``runner: container`` with
                ``lifecycle: one_shot`` (the default).
            workspace: The flywheel workspace.
            input_bindings: Maps input slot names to artifact
                instance IDs.
            execution_id: Pre-assigned execution ID.  Generated
                from the workspace if omitted.
            overrides: Per-launch CLI flag overrides (merged
                with the executor's defaults).
            allowed_blocks: If set, only these block names are
                permitted.  Used by the channel to restrict what
                a given caller can invoke.
            state_lineage_id: Optional name of the state lineage
                to restore from and extend.  ``None`` (the
                default) uses the workspace-wide single-lineage
                bucket — matches every stateful-block use case
                we support today.  Callers that want parallel or
                forked state chains can pass distinct IDs.
            run_id: Optional run grouping id.  When set, the
                recorded :class:`BlockExecution` is stamped with
                this run_id so callers can filter cadence
                counters to a single run.  ``None`` means
                ad-hoc (no grouping).
            extra_env: Per-launch environment variables to merge
                into ``block_def.env`` before building the
                ``ContainerConfig``.  Later writes win, so a
                launcher can override a block-declared default.
            extra_mounts: Per-launch bind mounts appended to the
                block's staged-input / output / state / workspace
                mount list.  Each entry is ``(host_path,
                container_path, mode)``.  Mount paths must not
                collide with the contract-reserved ``/input/*``,
                ``/output/*``, ``/state``, or ``/scratch`` —
                collision surfaces as a Docker error.
            extra_docker_args: Per-launch Docker CLI flags
                appended to ``block_def.docker_args``.  Used for
                launch-time policies such as ``--cap-add=NET_ADMIN``
                that the caller turns on or off per run without
                rewriting the block definition.  The merge is
                local to this launch; the block definition is
                not mutated.
            log_dir: When set, capture the container's stdout
                and stderr into ``log_dir/stdout.log`` and
                ``log_dir/stderr.log`` via background drain
                threads.  Default ``None`` lets the streams
                inherit the parent's file descriptors.
            total_timeout_s: When set, a watchdog thread enforces
                a wall-clock budget independent of stdout
                activity; on expiry it calls :meth:`stop` with
                reason ``"total_timeout"``.  Default ``None``
                disables the watchdog (natural exit only).
            on_stdout_line: When set (alongside ``log_dir``),
                the stdout drain thread calls this function for
                each line as it arrives.  Used by callers that
                want structured event processing without
                post-exit log re-reads.

        Returns:
            A live :class:`ContainerExecutionHandle` on the
            normal path, or a pre-completed
            :class:`SyncExecutionHandle` if a pre-container
            phase failed.  Both satisfy the
            :class:`ExecutionHandle` protocol.

        Raises:
            ValueError: If the block is not found or not
                permitted, or if the block's declared runner is
                not ``container``.
        """
        if allowed_blocks and block_name not in allowed_blocks:
            raise ValueError(
                f"Block {block_name!r} not in allowed list: "
                f"{allowed_blocks}")

        block_def = _find_block(self._template, block_name)
        if block_def is None:
            raise ValueError(
                f"Block {block_name!r} not found in template")
        if block_def.runner != "container":
            raise ValueError(
                f"Block {block_name!r}: ProcessExitExecutor only "
                f"runs container blocks (got runner "
                f"{block_def.runner!r})")

        exec_id = execution_id or workspace.generate_execution_id()
        started_at = datetime.now(UTC)

        # Pre-container pipeline state.  Any failure here short-
        # circuits into a pre-completed :class:`SyncExecutionHandle`
        # — the container never started, so there's nothing for
        # a caller to ``stop()``.
        staged_inputs: dict[str, Path] = {}
        state_mount: Path | None = None
        output_tempdirs: dict[str, Path] = {}
        work_area: Path | None = None

        try:
            # Phase: stage_in.
            try:
                staged_inputs = stage_artifact_instances(
                    workspace, input_bindings)
            except StagingError as e:
                return _record_pre_container_failure(
                    workspace=workspace,
                    block_def=block_def,
                    exec_id=exec_id,
                    started_at=started_at,
                    input_bindings=input_bindings,
                    state_lineage_id=state_lineage_id,

                    run_id=run_id,
                    failure_phase=runtime.FAILURE_STAGE_IN,
                    error=f"stage_in: {e}",
                    staged_inputs=staged_inputs,
                    state_mount=state_mount,
                    output_tempdirs=output_tempdirs,
                    work_area=work_area,
                )

            # Populate /state/ mount for stateful blocks.
            if block_def.state:
                try:
                    state_mount = populate_state_mount(
                        workspace, block_name, state_lineage_id)
                except OSError as e:
                    return _record_pre_container_failure(
                        workspace=workspace,
                        block_def=block_def,
                        exec_id=exec_id,
                        started_at=started_at,
                        input_bindings=input_bindings,
                        state_lineage_id=state_lineage_id,

                        run_id=run_id,
                        failure_phase=runtime.FAILURE_STAGE_IN,
                        error=f"stage_in (state): {e}",
                        staged_inputs=staged_inputs,
                        state_mount=state_mount,
                        output_tempdirs=output_tempdirs,
                        work_area=work_area,
                    )

            # Allocate per-slot output tempdirs.
            for slot in block_def.outputs:
                output_tempdirs[slot.name] = Path(
                    tempfile.mkdtemp(
                        prefix=f"flywheel-output-{slot.name}-"))

            # Allocate the /scratch/ work-area tempdir where
            # flywheel and the container communicate (stop
            # sentinel, future runtime socket).
            work_area = Path(tempfile.mkdtemp(
                prefix=f"flywheel-work-{block_name}-"))

            # Phase: invoke (start the container, non-blocking).
            mounts = _build_mounts(
                block_def, staged_inputs, output_tempdirs,
                state_mount, work_area,
            )
            if extra_mounts:
                mounts.extend(extra_mounts)
            merged_env = dict(block_def.env)
            if extra_env:
                merged_env.update(extra_env)
            merged_docker_args = list(block_def.docker_args)
            if extra_docker_args:
                merged_docker_args.extend(extra_docker_args)
            cc = ContainerConfig(
                image=block_def.image,
                docker_args=merged_docker_args,
                env=merged_env,
                mounts=mounts,
            )
            container_args = _build_container_args(
                self._overrides, overrides)
            container_name = f"flywheel-block-{exec_id}"

            start_monotonic = time.monotonic()
            capture_output = log_dir is not None
            try:
                process = start_container(
                    cc,
                    args=container_args or None,
                    name=container_name,
                    capture_output=capture_output,
                )
            except Exception as e:
                return _record_pre_container_failure(
                    workspace=workspace,
                    block_def=block_def,
                    exec_id=exec_id,
                    started_at=started_at,
                    input_bindings=input_bindings,
                    state_lineage_id=state_lineage_id,

                    run_id=run_id,
                    failure_phase=runtime.FAILURE_INVOKE,
                    error=f"invoke: {e}",
                    staged_inputs=staged_inputs,
                    state_mount=state_mount,
                    output_tempdirs=output_tempdirs,
                    work_area=work_area,
                )
        except _AbortExecution:
            # Defensive: no current path raises this anymore,
            # but keep the sentinel so future refactors don't
            # silently drop pre-container failures.
            raise

        # Container is running; hand off to the handle, which
        # owns the post-exit pipeline and cleanup.
        return ContainerExecutionHandle(
            process=process,
            workspace=workspace,
            block_def=block_def,
            execution_id=exec_id,
            started_at=started_at,
            start_monotonic=start_monotonic,
            container_name=container_name,
            staged_inputs=staged_inputs,
            state_mount=state_mount,
            output_tempdirs=output_tempdirs,
            work_area=work_area,
            input_bindings=input_bindings,
            state_lineage_id=state_lineage_id,
            run_id=run_id,
            log_dir=log_dir,
            total_timeout_s=total_timeout_s,
            on_stdout_line=on_stdout_line,
        )


_STATE_ELIGIBLE_STATUSES: frozenset[str] = frozenset(
    {"succeeded", "interrupted"}
)
"""Execution statuses whose captured state is safe to restore from.

``succeeded`` is the clean-exit case.  ``interrupted`` is the
operator-requested-stop case: the container exited cleanly via
its teardown path (e.g. an agent's ``finally:`` exports the
session) so the captured state is just as valid as a natural
exit.  ``failed`` is deliberately excluded — a non-zero exit or
body-level error may have left state mid-write, and we don't
currently have enough signal to tell partial state from complete
state.  That case would need explicit per-block opt-in if we
ever wanted to restore from it.
"""


def populate_state_mount(
    workspace: Workspace,
    block_name: str,
    state_lineage_id: str | None,
) -> Path:
    """Create a tempdir populated with the block's prior state.

    Walks the workspace's executions in reverse chronological
    order, picks the most recent execution of ``block_name``
    within the given ``state_lineage_id`` whose status is in
    :data:`_STATE_ELIGIBLE_STATUSES` and that has a captured
    ``state_dir``, and copies its captured state into a fresh
    tempdir.  The tempdir is what gets mounted at ``/state/``
    inside the container.

    Matching on ``state_lineage_id`` (including the ``None``
    default) lets callers keep independent state chains for the
    same block without colliding.  Today's default-``None``
    callers all end up in the same bucket, which matches the
    single-instance assumption for stateful blocks we actually
    have (one play agent, one predict agent, etc.).

    Returns an empty tempdir if no prior state exists in this
    lineage.

    Raises:
        OSError: If a prior execution's recorded ``state_dir``
            is missing on disk.  Silently cold-starting would
            mask workspace corruption; better to fail stage_in
            loudly so the operator notices.
    """
    staging = Path(tempfile.mkdtemp(
        prefix=f"flywheel-state-{block_name}-"))

    # Find the most recent state-eligible execution in the
    # target lineage.
    prior: BlockExecution | None = None
    for ex in sorted(
        workspace.executions.values(),
        key=lambda e: e.started_at,
        reverse=True,
    ):
        if (ex.block_name == block_name
                and ex.status in _STATE_ELIGIBLE_STATUSES
                and ex.state_dir is not None
                and ex.state_lineage_id == state_lineage_id):
            prior = ex
            break

    # Migration visibility: when a workspace has prior state
    # captured under the legacy synthetic ``__agent__`` name but
    # no state under the caller-supplied ``block_name``, warn
    # once so the operator sees the cold-start isn't a bug in
    # flywheel, it's unmigrated state.  Resolution is
    # operator-driven (rename the state_dir + rewrite the
    # execution record's block_name, or accept the cold start).
    if prior is None and block_name != "__agent__":
        legacy_present = any(
            ex.block_name == "__agent__"
            and ex.state_dir is not None
            and ex.status in _STATE_ELIGIBLE_STATUSES
            for ex in workspace.executions.values()
        )
        if legacy_present:
            print(
                f"  [flywheel] WARNING: workspace has state "
                f"captured under the legacy '__agent__' block "
                f"name but the current launch keys on "
                f"{block_name!r}.  Starting fresh; migrate by "
                f"renaming state/__agent__/ -> "
                f"state/{block_name}/ and updating "
                f"workspace.yaml execution block_name fields.",
                file=sys.stderr,
            )

    if prior is not None and prior.state_dir is not None:
        src = workspace.path / prior.state_dir
        if not src.is_dir():
            # Clean up the empty staging tempdir before raising
            # so we don't leak it on the failure path.
            shutil.rmtree(staging, ignore_errors=True)
            raise OSError(
                f"recorded state_dir {prior.state_dir!r} for "
                f"prior execution {prior.id!r} is missing on "
                f"disk; workspace state for block "
                f"{block_name!r} may be corrupted"
            )
        for child in src.iterdir():
            dest = staging / child.name
            if child.is_dir():
                shutil.copytree(child, dest)
            else:
                shutil.copy2(child, dest)

    return staging


def capture_state(
    workspace: Workspace,
    block_name: str,
    exec_id: str,
    state_mount: Path,
) -> str:
    """Copy ``/state/`` contents into the workspace's state dir.

    Creates ``<workspace>/state/<block_name>/<exec_id>/`` and
    copies everything from the container's post-exit state mount
    into it.  Returns the workspace-relative path for recording
    in ``BlockExecution.state_dir``.
    """
    rel = f"state/{block_name}/{exec_id}"
    dest = workspace.path / rel
    dest.mkdir(parents=True, exist_ok=True)
    for child in state_mount.iterdir():
        target = dest / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
    return rel


def _run_output_builder(
    block_def: BlockDefinition,
    output_tempdirs: dict[str, Path],
    workspace: Workspace,
    exec_id: str,
) -> None:
    """Invoke the block's ``output_builder`` on its output tempdirs.

    Re-resolves the dotted path at each execution rather than
    threading the registry-cached callable through every
    call site.  Python's import cache makes the re-resolution
    effectively free after the first call, and the block
    registry has already validated the path at startup — so a
    typo surfaces there, not here.

    The builder may mutate files under ``output_tempdirs`` in
    place.  It MUST NOT write outside those directories.  Any
    exception propagates to the caller, which maps it to
    ``FAILURE_OUTPUT_COLLECT``.
    """
    from flywheel.output_builder import (
        OutputBuilderContext,
        resolve_dotted_path,
    )
    path = block_def.output_builder
    assert path is not None  # caller checked
    builder = resolve_dotted_path(path)
    ctx = OutputBuilderContext(
        block=block_def.name,
        execution_id=exec_id,
        outputs={
            name: tempdir
            for name, tempdir in output_tempdirs.items()
        },
        workspace=workspace,
    )
    builder(ctx)


def _collect_outputs(
    workspace: Workspace,
    block_def: BlockDefinition,
    output_tempdirs: dict[str, Path],
    exec_id: str,
    finished_at: datetime,
) -> dict[str, str]:
    """Commit output tempdir contents per the artifact's declared kind.

    For each declared output slot whose tempdir has content,
    dispatch on the artifact's declared kind:

    - ``copy`` — allocate a fresh artifact id, copy the tempdir
      contents into ``<workspace>/artifacts/<aid>/``, register a
      new :class:`ArtifactInstance`.
    - ``incremental`` — read ``entries.jsonl`` from the tempdir,
      parse one JSON value per non-blank line, append to the
      canonical incremental artifact (creating it if this is
      the first append to that name).

    Git artifacts are never outputs; the schema rejects them.

    Empty output dirs and incremental dirs missing an
    ``entries.jsonl`` are skipped silently — a block is free to
    declare optional outputs it doesn't always produce.

    Returns the slot→artifact-instance-id map for the execution
    record's ``output_bindings``.
    """
    output_bindings: dict[str, str] = {}
    for slot in block_def.outputs:
        src = output_tempdirs.get(slot.name)
        if src is None or not any(src.iterdir()):
            continue
        kind = workspace.artifact_declarations.get(slot.name)
        if kind == "incremental":
            instance_id = _commit_incremental_output(
                workspace, slot.name, src, exec_id)
            if instance_id is not None:
                output_bindings[slot.name] = instance_id
        else:
            # Default to copy semantics.  ``kind`` being None
            # shouldn't happen for a block whose output is in
            # the template's declared artifacts (the parser
            # validates this), but if it does we still produce
            # something operable rather than crashing.
            output_bindings[slot.name] = _commit_copy_output(
                workspace, slot.name, src, exec_id, finished_at)
    return output_bindings


def _commit_copy_output(
    workspace: Workspace,
    slot_name: str,
    src: Path,
    exec_id: str,
    finished_at: datetime,
) -> str:
    """Create a fresh copy-artifact instance from a tempdir."""
    aid = workspace.generate_artifact_id(slot_name)
    canonical = workspace.path / "artifacts" / aid
    canonical.mkdir(parents=True)
    for child in src.iterdir():
        dest = canonical / child.name
        if child.is_dir():
            shutil.copytree(child, dest)
        else:
            shutil.copy2(child, dest)
    workspace.add_artifact(ArtifactInstance(
        id=aid,
        name=slot_name,
        kind="copy",
        created_at=finished_at,
        produced_by=exec_id,
        copy_path=aid,
    ))
    return aid


def _commit_incremental_output(
    workspace: Workspace,
    slot_name: str,
    src: Path,
    exec_id: str,
) -> str | None:
    """Append entries from the tempdir to the canonical instance.

    Mirrors :meth:`flywheel.local_block.LocalBlockRecorder._append_incremental_output`
    so both execution paths produce identically-shaped
    incremental artifacts.  Returns the canonical instance id,
    or ``None`` if the tempdir had no ``entries.jsonl`` to
    contribute.
    """
    entries_file = src / "entries.jsonl"
    if not entries_file.exists():
        return None
    raw = entries_file.read_text(encoding="utf-8")
    entries: list[Any] = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        entries.append(json.loads(stripped))
    if not entries:
        return None
    instance = workspace.latest_incremental_instance(slot_name)
    if instance is None:
        instance = workspace.register_incremental_artifact(
            slot_name,
            produced_by=exec_id,
            source=f"first appended by execution {exec_id}",
        )
    workspace.append_to_incremental(instance.id, entries)
    return instance.id


def _build_mounts(
    block_def: BlockDefinition,
    staged_inputs: dict[str, Path],
    output_tempdirs: dict[str, Path],
    state_mount: Path | None,
    work_area: Path,
) -> list[tuple[str, str, str]]:
    """Assemble the (host, container, mode) mount tuples.

    Never includes a canonical workspace path — every mount
    points at a per-execution staged tempdir.  The ``work_area``
    is mounted rw at ``/scratch/`` so flywheel and the
    container can exchange control signals (the stop sentinel,
    and later the runtime socket for request-response blocks).
    """
    mounts: list[tuple[str, str, str]] = []
    for slot in block_def.inputs:
        staged = staged_inputs.get(slot.name)
        if staged is not None:
            mounts.append(
                (str(staged), slot.container_path, "ro"))
    for slot in block_def.outputs:
        tempdir = output_tempdirs.get(slot.name)
        if tempdir is not None:
            mounts.append(
                (str(tempdir), slot.container_path, "rw"))
    if state_mount is not None:
        mounts.append(
            (str(state_mount), runtime.STATE_MOUNT_PATH, "rw"))
    mounts.append((str(work_area), "/scratch", "rw"))
    return mounts


def _apply_post_check(
    *,
    post_check: PostCheckCallable | None,
    execution: BlockExecution,
    output_dirs: dict[str, Path],
    workspace_path: Path,
    error: str | None,
) -> BlockExecution:
    """Invoke the block's post_check (if any) and merge the result.

    Returns ``execution`` unchanged when no check is wired.  Otherwise
    builds a :class:`PostCheckContext` in the same shape
    ``LocalBlockRecorder._run_post_check`` uses, invokes the
    callable, and returns a new :class:`BlockExecution` with
    ``halt_directive`` and / or ``post_check_error`` populated.
    Mirrors the lifecycle-block contract so the pattern runner's
    ledger-driven halt logic works uniformly across runner kinds.

    Args:
        post_check: The callable to invoke, or ``None``.
        execution: The freshly-built execution record.
        output_dirs: Map from output slot name to the per-execution
            directory the container wrote.  Passed into the
            context's ``outputs`` so checks can inspect what the
            container produced (e.g. parsing a JSON blob and
            halting on a terminal-state field).
        workspace_path: The workspace root — surfaced on the
            context for checks that want to peek at sibling
            artifacts / ledger files.
        error: The ``execution.error`` string (or ``None``), passed
            through so checks can behave differently on
            succeeded-but-with-a-note paths.  ``RequestResponseExecutor``
            does not have a ``caller`` / ``params`` / ``synthetic``
            / ``parent_execution_id`` concept today; those fields
            are left as their :class:`PostCheckContext` defaults.
    """
    if post_check is None:
        return execution

    ctx = PostCheckContext(
        block=execution.block_name,
        execution_id=execution.id,
        status=execution.status,  # type: ignore[arg-type]
        caller=None,
        params=None,
        error=error,
        outputs=output_dirs,
        parent_execution_id=None,
        synthetic=False,
        workspace_path=workspace_path,
    )

    directive: HaltDirective | None = None
    post_check_error: str | None = None
    try:
        directive = post_check(ctx)
    except Exception as exc:  # noqa: BLE001
        post_check_error = f"{type(exc).__name__}: {exc}"

    if directive is not None and not isinstance(
            directive, HaltDirective):
        post_check_error = (
            f"post_check returned "
            f"{type(directive).__name__}, expected "
            f"HaltDirective | None")
        directive = None

    if directive is None and post_check_error is None:
        return execution

    return replace(
        execution,
        halt_directive=(
            directive.to_dict() if directive is not None else None),
        post_check_error=post_check_error,
    )


def _record_pre_container_failure(
    *,
    workspace: Workspace,
    block_def: BlockDefinition,
    exec_id: str,
    started_at: datetime,
    input_bindings: dict[str, str],
    state_lineage_id: str | None,
    failure_phase: str,
    error: str,
    staged_inputs: dict[str, Path],
    state_mount: Path | None,
    output_tempdirs: dict[str, Path],
    work_area: Path | None,
    run_id: str | None = None,
) -> SyncExecutionHandle:
    """Write a failure :class:`BlockExecution` and clean up.

    Used when the pre-container pipeline fails (stage_in or the
    non-blocking container start itself).  No container ever
    ran, so there's no handle for a caller to ``stop()`` — the
    returned :class:`SyncExecutionHandle` is pre-completed.
    """
    # Best-effort cleanup of whatever tempdirs we managed to
    # allocate before the failure.
    cleanup_staged_inputs(staged_inputs)
    if state_mount is not None:
        shutil.rmtree(state_mount, ignore_errors=True)
    for path in output_tempdirs.values():
        shutil.rmtree(path, ignore_errors=True)
    if work_area is not None:
        shutil.rmtree(work_area, ignore_errors=True)

    finished_at = datetime.now(UTC)
    execution = BlockExecution(
        id=exec_id,
        block_name=block_def.name,
        started_at=started_at,
        finished_at=finished_at,
        status="failed",
        input_bindings=input_bindings,
        output_bindings={},
        exit_code=None,
        elapsed_s=0.0,
        image=block_def.image,
        runner="container",
        failure_phase=failure_phase,
        error=error,
        state_lineage_id=state_lineage_id,
        run_id=run_id,
    )
    workspace.add_execution(execution)
    workspace.save()

    return SyncExecutionHandle(ExecutionResult(
        exit_code=INVOKE_FAILURE_EXIT_CODE,
        elapsed_s=0.0,
        output_bindings={},
        execution_id=exec_id,
        status="failed",
    ))


def _build_container_args(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
) -> list[str]:
    """Translate override dicts into ``--flag value`` argv."""
    merged = dict(defaults or {})
    if overrides:
        merged.update(overrides)
    args: list[str] = []
    for key, value in merged.items():
        flag = f"--{key.replace('_', '-')}"
        args += [flag, str(value)]
    return args


# ── ProcessExecutor ──────────────────────────────────────────────


class ProcessExecutionHandle(ExecutionHandle):
    """Handle to a running local subprocess.

    Wraps a ``subprocess.Popen`` process.  ``stop()`` terminates
    the process.  ``wait()`` blocks until exit and records the
    execution in the workspace.
    """

    def __init__(
        self,
        process: subprocess.Popen,
        workspace: Workspace,
        block_name: str,
        execution_id: str,
        started_at: datetime,
        start_monotonic: float,
        run_id: str | None = None,
    ):
        """Initialize from a running subprocess."""
        self._process = process
        self._workspace = workspace
        self._block_name = block_name
        self._execution_id = execution_id
        self._started_at = started_at
        self._run_id = run_id
        self._start_monotonic = start_monotonic
        self._stop_reason: str | None = None
        self._waited = False

    def is_alive(self) -> bool:
        """Check if the subprocess is still running."""
        return self._process.poll() is None

    def stop(self, reason: str = "requested") -> None:
        """Terminate the subprocess."""
        self._stop_reason = reason
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    def wait(self) -> ExecutionResult:
        """Block until the subprocess exits and record execution.

        Raises:
            RuntimeError: If ``wait()`` has already been called.
        """
        if self._waited:
            raise RuntimeError(
                "wait() already called on this handle")
        self._waited = True

        self._process.wait()
        elapsed = time.monotonic() - self._start_monotonic
        finished_at = datetime.now(UTC)

        exit_code = self._process.returncode or 0
        if self._stop_reason:
            status = "interrupted"
        elif exit_code == 0:
            status = "succeeded"
        else:
            status = "failed"

        execution = BlockExecution(
            id=self._execution_id,
            block_name=self._block_name,
            started_at=self._started_at,
            finished_at=finished_at,
            status=status,
            exit_code=exit_code,
            elapsed_s=elapsed,
            stop_reason=self._stop_reason,
            run_id=self._run_id,
        )
        self._workspace.add_execution(execution)
        self._workspace.save()

        return ExecutionResult(
            exit_code=exit_code,
            elapsed_s=elapsed,
            output_bindings={},
            execution_id=self._execution_id,
            status=status,
        )


class ProcessExecutor:
    """Execute a block as a local subprocess.

    For trusted, host-local processes like game servers or local
    tools.  Launches via ``subprocess.Popen`` and returns a
    ``ProcessExecutionHandle`` for lifecycle control.
    """

    def launch(
        self,
        block_name: str,
        workspace: Workspace,
        input_bindings: dict[str, str],
        *,
        command: str | list[str] | None = None,
        execution_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        allowed_blocks: list[str] | None = None,
        state_lineage_id: str | None = None,
        run_id: str | None = None,
    ) -> ProcessExecutionHandle:
        """Launch a local subprocess.

        Args:
            block_name: Name of the block.
            workspace: The flywheel workspace.
            input_bindings: Input bindings (recorded but not
                mounted — local processes access files directly).
            command: Shell command string or argument list.
                Required.
            execution_id: Pre-assigned execution ID.
            overrides: Unused (protocol compat).
            allowed_blocks: If set, only these block names allowed.
            state_lineage_id: Unused (protocol compat — local
                subprocesses have no state mount to populate).
            run_id: Optional run grouping id stamped on the
                resulting :class:`BlockExecution` record.
                ``None`` means ad-hoc (no grouping).

        Returns:
            A ``ProcessExecutionHandle`` for monitoring the
            subprocess.

        Raises:
            ValueError: If command is not provided or block is
                not allowed.
        """
        if allowed_blocks and block_name not in allowed_blocks:
            raise ValueError(
                f"Block {block_name!r} not in allowed list: "
                f"{allowed_blocks}")

        if command is None:
            raise ValueError(
                "ProcessExecutor requires a command")

        exec_id = execution_id or workspace.generate_execution_id()
        started_at = datetime.now(UTC)

        if isinstance(command, str):
            process = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        return ProcessExecutionHandle(
            process=process,
            workspace=workspace,
            block_name=block_name,
            execution_id=exec_id,
            started_at=started_at,
            start_monotonic=time.monotonic(),
            run_id=run_id,
        )


# ── RequestResponseExecutor ──────────────────────────────────────

REQUEST_RESPONSE_PROTOCOL_VERSION: str = "1"
"""Bumped when the wire protocol (endpoints, request / response
shapes) changes in a way that makes older containers incompatible.
Used as a Docker label on runtimes and checked against the
running container on reattachment."""

_RUNTIME_LABEL_WORKSPACE: str = "flywheel.workspace"
_RUNTIME_LABEL_WORKSPACE_PATH: str = "flywheel.workspace_path"
_RUNTIME_LABEL_BLOCK: str = "flywheel.block"
_RUNTIME_LABEL_PROTOCOL: str = "flywheel.protocol"
_RUNTIME_LABEL_LIFECYCLE: str = "flywheel.lifecycle"

_HEALTH_POLL_INTERVAL_S: float = 0.1
_DEFAULT_STARTUP_TIMEOUT_S: float = 30.0


def _allocate_free_port() -> int:
    """Return an unused localhost TCP port.

    Binds a socket to port 0, reads the kernel-assigned port, and
    releases the socket.  There is a small window between release
    and the executor re-binding the port in Docker during which
    another process could claim it; accepted for a development-
    grade substrate.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ControlChannelError(RuntimeError):
    """Raised by a :class:`ControlChannel` when the transport fails.

    Maps to ``failure_phase=invoke`` on the execution record.
    """


@runtime_checkable
class ControlChannel(Protocol):
    """Transport between the executor and a request-response runtime.

    The contract does not mandate a wire format; today's
    implementation is HTTP over localhost TCP
    (:class:`HttpControlChannel`).  Tests inject fakes that
    satisfy this protocol without opening a real socket.
    """

    def health(self, timeout_s: float) -> bool:
        """Return ``True`` when the runtime is ready to serve."""
        ...

    def execute(
        self, request_id: str, block_name: str, timeout_s: float,
    ) -> dict[str, Any]:
        """Send ``POST /execute`` and return the parsed response.

        Raises :class:`ControlChannelError` on transport failure.
        """
        ...

    def cancel(self, request_id: str, timeout_s: float) -> None:
        """Best-effort ``POST /cancel`` for the in-flight request.

        Swallows transport errors — a cancel that can't reach the
        container still causes the executor to fall back to
        runtime teardown.
        """
        ...


class HttpControlChannel:
    """HTTP-over-localhost-TCP client for a request-response runtime."""

    def __init__(self, host: str, port: int):
        """Capture the runtime's address.

        ``host`` is usually ``"127.0.0.1"``; ``port`` is the
        host-side port allocated by :func:`_allocate_free_port`
        and passed to the container via
        :data:`flywheel.runtime.CONTROL_PORT_ENV_VAR`.
        """
        self._host = host
        self._port = port

    def _conn(self, timeout_s: float) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(
            self._host, self._port, timeout=timeout_s)

    def health(self, timeout_s: float) -> bool:
        """Return ``True`` on HTTP 200 to ``GET /health``.

        Any connection error, non-200 status, or timeout is
        treated as "not ready yet" — the caller retries.
        """
        try:
            conn = self._conn(timeout_s)
            try:
                conn.request("GET", "/health")
                resp = conn.getresponse()
                resp.read()  # drain to allow connection reuse
                return resp.status == 200
            finally:
                conn.close()
        except (OSError, http.client.HTTPException):
            return False

    def execute(
        self, request_id: str, block_name: str, timeout_s: float,
    ) -> dict[str, Any]:
        """Send ``POST /execute`` and parse the JSON response.

        Raises:
            ControlChannelError: transport or protocol-shape error.
        """
        body = json.dumps({
            "request_id": request_id,
            "block_name": block_name,
        })
        try:
            conn = self._conn(timeout_s)
            try:
                conn.request(
                    "POST", "/execute",
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                raw = resp.read().decode("utf-8")
                if resp.status != 200:
                    raise ControlChannelError(
                        f"POST /execute returned HTTP "
                        f"{resp.status}: {raw[:200]}"
                    )
            finally:
                conn.close()
        except (OSError, http.client.HTTPException) as exc:
            raise ControlChannelError(
                f"POST /execute failed: {exc}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ControlChannelError(
                f"POST /execute returned non-JSON body: "
                f"{raw[:200]}") from exc
        if not isinstance(payload, dict):
            raise ControlChannelError(
                f"POST /execute returned non-object JSON: "
                f"{type(payload).__name__}")
        return payload

    def cancel(self, request_id: str, timeout_s: float) -> None:
        """Send ``POST /cancel`` best-effort, swallowing errors."""
        body = json.dumps({"request_id": request_id})
        try:
            conn = self._conn(timeout_s)
            try:
                conn.request(
                    "POST", "/cancel",
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                resp.read()
            finally:
                conn.close()
        except (OSError, http.client.HTTPException):
            return


@dataclass
class _RuntimeHandle:
    """Host-side reference to a running request-response container.

    One per attachment key.  Carries the control channel the
    executor POSTs requests against, the lock the host uses to
    serialize requests, and the resources needed for teardown.
    """

    attachment_key: str
    container_name: str
    control_port: int
    block_def: BlockDefinition
    workspace: Workspace
    work_area: Path
    process: subprocess.Popen | None
    channel: ControlChannel
    request_lock: threading.Lock = field(default_factory=threading.Lock)


def _workspace_identity(workspace: Workspace) -> str:
    """Return a short stable identifier for a workspace.

    Used in attachment keys and container names.  The absolute
    path of the workspace directory is hashed to a fixed-length
    digest so two workspaces that happen to share a ``name`` but
    live at different paths cannot collide on the same runtime.
    The full path is also recorded as a Docker label for operator
    visibility.
    """
    path = str(workspace.path.resolve())
    digest = hashlib.sha256(
        path.encode("utf-8")).hexdigest()[:12]
    return digest


def _runtime_container_name(
    workspace: Workspace, block_name: str,
) -> str:
    """Build a stable, collision-resistant Docker container name.

    Names must be 1-64 chars from
    ``[a-zA-Z0-9][a-zA-Z0-9_.-]*``.  Built from the block name
    (for operator-legibility when listing Docker containers) and
    the workspace path digest (so two workspaces with the same
    ``workspace.name`` at different paths do not collide).
    """
    safe_block = "".join(
        c if (c.isalnum() or c in "_.-") else "-"
        for c in block_name
    )
    return (
        f"flywheel-runtime-{safe_block}-"
        f"{_workspace_identity(workspace)}"
    )[:60]


def _runtime_labels(
    block_def: BlockDefinition,
    workspace: Workspace,
) -> dict[str, str]:
    """Labels flywheel stamps on every request-response runtime.

    Lets the executor recognize its own runtimes on reattachment
    and distinguish protocol versions.  Foreign containers (same
    name but wrong labels) are treated as unsafe to attach to.
    ``workspace_path`` is the absolute workspace directory; the
    digest lives in the container name and is implicitly carried
    by the labels together.
    """
    return {
        _RUNTIME_LABEL_WORKSPACE: workspace.name,
        _RUNTIME_LABEL_WORKSPACE_PATH: str(
            workspace.path.resolve()),
        _RUNTIME_LABEL_BLOCK: block_def.name,
        _RUNTIME_LABEL_PROTOCOL: REQUEST_RESPONSE_PROTOCOL_VERSION,
        _RUNTIME_LABEL_LIFECYCLE: "workspace_persistent",
    }


def _run_detached_container(
    config: ContainerConfig,
    name: str,
) -> str:
    """Launch a container in detached mode and return its ID.

    Equivalent of ``docker run -d --rm --name <name> …`` —
    spawns the container in the background, returns the short
    container ID Docker prints on stdout, and leaves no
    ``docker run`` client process hanging around.  Needed for
    persistent request-response runtimes: the executor's host
    process must be able to exit without killing the runtime,
    and the runtime must survive to be reattached to on the
    next run.

    Raises:
        ControlChannelError: if ``docker run -d`` returns a
            non-zero exit code or prints no container id.
    """
    cmd = ["docker", "run", "-d", "--rm", "--name", name]
    cmd.extend(config.docker_args)
    for key, value in config.env.items():
        cmd.extend(["-e", f"{key}={value}"])
    for host_path, container_path, mode in config.mounts:
        normalized = host_path.replace("\\", "/")
        cmd.extend(
            ["-v", f"{normalized}:{container_path}:{mode}"])
    cmd.append(config.image)

    proc_env = os.environ.copy()
    proc_env["MSYS_NO_PATHCONV"] = "1"
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=proc_env,
    )
    if result.returncode != 0:
        raise ControlChannelError(
            f"docker run -d failed for {name!r}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    container_id = result.stdout.strip()
    if not container_id:
        raise ControlChannelError(
            f"docker run -d for {name!r} returned no "
            f"container id"
        )
    return container_id


def _docker_kill(container_name: str, signal: str) -> None:
    """Send ``signal`` to a container by name, ignoring errors.

    Parallels :func:`_docker_send_signal` but is semantically
    "tear this runtime down," not "signal this live execution."
    ``check=False`` because a container that has already exited
    between the decision and the call is not an error."""
    subprocess.run(
        ["docker", "kill", "--signal", signal, container_name],
        check=False, capture_output=True,
    )


def _docker_wait_gone(
    container_name: str, timeout_s: float,
) -> bool:
    """Block until the named container is absent from ``docker ps``.

    Returns ``True`` when the container is gone, ``False`` on
    timeout.  Polls at a modest cadence — teardown latency is
    not a hot path.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = _docker_ps_find(container_name)
        if info is None or info.get("_running") != "true":
            return True
        time.sleep(0.2)
    return False


def _docker_ps_find(
    container_name: str,
) -> dict[str, str] | None:
    """Inspect a container by name; return labels dict or ``None``.

    Uses ``docker inspect`` which returns structured JSON; this
    is tolerant of the container being stopped vs running vs
    absent.  Returns ``None`` when the container does not exist.
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", container_name],
            capture_output=True, check=False, text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not entries:
        return None
    entry = entries[0]
    labels = (
        entry.get("Config", {}).get("Labels") or {})
    state = entry.get("State", {}) or {}
    running = bool(state.get("Running", False))
    # Return the labels dict, decorated with a synthetic
    # ``_running`` key so the caller can distinguish "live" from
    # "exists but stopped".
    return {"_running": "true" if running else "false", **labels}


class _RequestExecutionHandle(ExecutionHandle):
    """Handle to one ``POST /execute`` call against a runtime.

    Runs the request on a background thread so ``stop()`` can
    fire ``POST /cancel`` from the main thread while ``wait()``
    blocks on the response.  ``wait()`` finalizes per-request
    state (output collection, :class:`BlockExecution` record)
    and cleans up per-request tempdirs.
    """

    def __init__(
        self,
        *,
        runtime: _RuntimeHandle,
        request_id: str,
        execution_id: str,
        started_at: datetime,
        start_monotonic: float,
        input_bindings: dict[str, str],
        state_lineage_id: str | None,
        run_id: str | None,
        staged_inputs: dict[str, Path],
        request_dir: Path,
        output_dirs: dict[str, Path],
        execute_timeout_s: float,
        cancel_timeout_s: float,
        post_check: PostCheckCallable | None = None,
    ):
        """Capture the per-request state and start the POST thread."""
        self._runtime = runtime
        self._request_id = request_id
        self._execution_id = execution_id
        self._started_at = started_at
        self._start_monotonic = start_monotonic
        self._input_bindings = input_bindings
        self._state_lineage_id = state_lineage_id
        self._run_id = run_id
        self._staged_inputs = staged_inputs
        self._request_dir = request_dir
        self._output_dirs = output_dirs
        self._execute_timeout_s = execute_timeout_s
        self._cancel_timeout_s = cancel_timeout_s
        self._post_check = post_check

        self._response: dict[str, Any] | None = None
        self._transport_error: ControlChannelError | None = None
        self._stop_reason: str | None = None
        self._stop_lock = threading.Lock()
        self._stop_initiated = False
        self._waited = False

        self._thread = threading.Thread(
            target=self._run,
            name=f"flywheel-rr-exec-{execution_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        """Body of the POST thread.  Populates response or error.

        The runtime's request lock is acquired around the POST
        so only one ``/execute`` flight is in progress against
        the container at a time.  Holding the lock at thread
        scope (rather than handle scope) means an abandoned
        handle cannot strand the lock — the thread always
        releases it.
        """
        with self._runtime.request_lock:
            try:
                self._response = (
                    self._runtime.channel.execute(
                        self._request_id,
                        self._runtime.block_def.name,
                        self._execute_timeout_s,
                    )
                )
            except ControlChannelError as exc:
                self._transport_error = exc

    def is_alive(self) -> bool:
        """Return whether the POST thread is still running."""
        return self._thread.is_alive()

    def stop(self, reason: str = "requested") -> None:
        """Cancel the in-flight request via ``POST /cancel``.

        Idempotent; a second call is a no-op.  Does not tear
        down the runtime — use
        :meth:`RequestResponseExecutor.shutdown` for that.
        """
        with self._stop_lock:
            if self._stop_initiated:
                return
            self._stop_initiated = True
            self._stop_reason = reason
        # Run the RPC outside the lock so a concurrent stop() is
        # immediately no-op rather than blocked on the HTTP call.
        with contextlib.suppress(Exception):
            self._runtime.channel.cancel(
                self._request_id, self._cancel_timeout_s)

    def wait(self) -> ExecutionResult:
        """Block for the request to complete and finalize the execution.

        Joins the POST thread, reads outputs from the per-request
        directory, writes a :class:`BlockExecution` record, and
        tears down per-request tempdirs.

        Raises:
            RuntimeError: If called more than once on the same handle.
        """
        if self._waited:
            raise RuntimeError(
                "wait() already called on this handle")
        self._waited = True

        failure_phase: str | None = None
        error: str | None = None
        output_bindings: dict[str, str] = {}

        try:
            self._thread.join()

            finished_at = datetime.now(UTC)
            elapsed = time.monotonic() - self._start_monotonic

            if self._transport_error is not None:
                failure_phase = runtime.FAILURE_INVOKE
                error = (
                    f"invoke: transport error — "
                    f"{self._transport_error}"
                )
            elif self._response is not None:
                status = self._response.get("status")
                if status == "failed":
                    failure_phase = runtime.FAILURE_INVOKE
                    error = (
                        f"invoke: {self._response.get('error') or ''}"
                    )
                elif status != "succeeded":
                    failure_phase = runtime.FAILURE_INVOKE
                    error = (
                        f"invoke: unexpected status {status!r}"
                    )
            else:
                failure_phase = runtime.FAILURE_INVOKE
                error = "invoke: no response from runtime"

            # Output collection runs even on invoke failure; a
            # container that emits partial outputs before
            # returning an error is still useful for debugging.
            try:
                output_bindings = _collect_outputs(
                    self._runtime.workspace,
                    self._runtime.block_def,
                    self._output_dirs,
                    self._execution_id,
                    finished_at,
                )
            except Exception as e:
                if failure_phase is None:
                    failure_phase = (
                        runtime.FAILURE_OUTPUT_COLLECT)
                    error = f"output_collect: {e}"
        finally:
            cleanup_staged_inputs(self._staged_inputs)
            shutil.rmtree(self._request_dir, ignore_errors=True)

        if self._stop_reason is not None:
            status = "interrupted"
        elif failure_phase is not None:
            status = "failed"
        else:
            status = "succeeded"

        execution = BlockExecution(
            id=self._execution_id,
            block_name=self._runtime.block_def.name,
            started_at=self._started_at,
            finished_at=finished_at,
            status=status,
            input_bindings=self._input_bindings,
            output_bindings=output_bindings,
            exit_code=None,
            elapsed_s=elapsed,
            image=self._runtime.block_def.image,
            runner="container",
            failure_phase=failure_phase,
            error=error,
            state_lineage_id=self._state_lineage_id,
            run_id=self._run_id,
            stop_reason=self._stop_reason,
        )
        # Run the block-declared post_check (if any) before the
        # execution record is persisted, so the ``halt_directive``
        # / ``post_check_error`` fields the ledger reader sees are
        # the post-check result.  Symmetric with
        # ``LocalBlockRecorder._run_post_check``: same
        # ``PostCheckContext`` shape, same merge onto the record.
        execution = _apply_post_check(
            post_check=self._post_check,
            execution=execution,
            output_dirs=self._output_dirs,
            workspace_path=self._runtime.workspace.path,
            error=error,
        )
        self._runtime.workspace.add_execution(execution)
        self._runtime.workspace.save()

        return ExecutionResult(
            exit_code=(
                0 if status == "succeeded"
                else INVOKE_FAILURE_EXIT_CODE),
            elapsed_s=elapsed,
            output_bindings=output_bindings,
            execution_id=self._execution_id,
            status=status,
        )


class RequestResponseExecutor:
    """Request-response protocol executor — one runtime, many requests.

    Implements the container runtime contract for blocks whose
    lifecycle is ``workspace_persistent``:

    1. On first ``launch()`` for a given ``(block_name, workspace)``
       attachment key, start a persistent container publishing
       its control channel on a localhost port.  Probe ``/health``
       until ready.
    2. For each request: stage declared inputs into
       ``<work_area>/requests/<req_id>/input/<slot>/``, create
       ``<work_area>/requests/<req_id>/output/<slot>/`` dirs,
       ``POST /execute``, collect outputs from the output dirs,
       write a :class:`BlockExecution`.
    3. Requests against the same runtime are serialized on the
       host side via a per-runtime :class:`threading.Lock`; the
       container may reject overlapping requests as defense-in-
       depth but is not the primary scheduler.
    4. Runtimes persist across ``launch()`` calls until an
       explicit :meth:`shutdown` (via the CLI or programmatically)
       or flywheel process exit.  Reattachment to an existing
       runtime after a host-process restart is matched by labels
       and confirmed by a ``/health`` probe before routing
       requests to it.

    Request-response blocks do not use ``/state/`` — state lives
    in the container's memory for its lifetime.  The schema
    rejects ``state: true`` with ``lifecycle: workspace_persistent``.
    """

    def __init__(
        self,
        template: Template,
        overrides: dict[str, Any] | None = None,
        *,
        channel_factory: (
            Callable[[str, int], ControlChannel] | None) = None,
        startup_timeout_s: float = _DEFAULT_STARTUP_TIMEOUT_S,
        execute_timeout_s: float = 600.0,
        cancel_timeout_s: float = 5.0,
        post_checks: (
            dict[str, "PostCheckCallable"] | None) = None,
    ):
        """Initialize with a template and optional dependencies.

        Args:
            template: The template containing block definitions.
            overrides: Default CLI flag overrides forwarded to the
                container on startup.
            channel_factory: Optional ``(host, port) ->
                ControlChannel`` factory.  Defaults to
                :class:`HttpControlChannel`.  Tests inject a fake
                to avoid opening real sockets.
            startup_timeout_s: How long to poll ``/health`` after
                launching a new runtime before giving up.
            execute_timeout_s: HTTP read timeout for
                ``POST /execute``.
            cancel_timeout_s: HTTP read timeout for
                ``POST /cancel``.
            post_checks: Map from block name to the post-execution
                callable declared in the block YAML.  When set, the
                request handle invokes the callable after each
                execution completes and stamps any returned
                :class:`~flywheel.post_check.HaltDirective` onto
                the :class:`BlockExecution` record — same ledger
                contract ``LocalBlockRecorder`` uses for
                ``runner: lifecycle`` blocks, so the executor path
                honours the same policy surface for
                ``runner: container`` blocks.  Callers pass
                ``block_registry.post_checks`` here; an empty /
                ``None`` dict disables the invocation.
        """
        self._template = template
        self._overrides = overrides
        self._channel_factory = (
            channel_factory
            or (lambda host, port: HttpControlChannel(host, port))
        )
        self._startup_timeout_s = startup_timeout_s
        self._execute_timeout_s = execute_timeout_s
        self._cancel_timeout_s = cancel_timeout_s
        self._post_checks: dict[str, PostCheckCallable] = (
            dict(post_checks) if post_checks else {})

        self._runtimes: dict[str, _RuntimeHandle] = {}
        self._registry_lock = threading.Lock()

    @staticmethod
    def _attachment_key(block_name: str, workspace: Workspace) -> str:
        """Return the registry key for a ``(block, workspace)`` pair.

        The workspace is identified by a digest of its absolute
        path, not by ``workspace.name``: two distinct workspaces
        that happen to share a name must not reattach to each
        other's runtimes.  Kept as a helper so
        :class:`RequestResponseExecutor` can evolve to per-run
        or per-session keys without every caller having to
        update.
        """
        return f"{_workspace_identity(workspace)}::{block_name}"

    def launch(
        self,
        block_name: str,
        workspace: Workspace,
        input_bindings: dict[str, str],
        *,
        execution_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        allowed_blocks: list[str] | None = None,
        state_lineage_id: str | None = None,
        run_id: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_mounts: list[tuple[str, str, str]] | None = None,
        extra_docker_args: list[str] | None = None,
    ) -> ExecutionHandle:
        """Execute one request against the block's persistent runtime.

        Starts the runtime on first call for a given
        ``(block_name, workspace)`` pair; reuses it on subsequent
        calls.  Returns a :class:`_RequestExecutionHandle` whose
        ``wait()`` blocks on the container's response.

        ``extra_env`` / ``extra_mounts`` / ``extra_docker_args``
        are consumed on runtime *startup* only — they configure
        the long-lived container, not individual requests.
        Re-passing them on a subsequent ``launch()`` for the
        same attachment key is accepted but ignored.

        Raises:
            ValueError: If the block is not found, not permitted,
                or does not declare
                ``lifecycle: workspace_persistent``.
        """
        if allowed_blocks and block_name not in allowed_blocks:
            raise ValueError(
                f"Block {block_name!r} not in allowed list: "
                f"{allowed_blocks}")

        block_def = _find_block(self._template, block_name)
        if block_def is None:
            raise ValueError(
                f"Block {block_name!r} not found in template")
        if block_def.runner != "container":
            raise ValueError(
                f"Block {block_name!r}: "
                f"RequestResponseExecutor only runs container "
                f"blocks (got runner {block_def.runner!r})")
        if block_def.lifecycle != "workspace_persistent":
            raise ValueError(
                f"Block {block_name!r}: "
                f"RequestResponseExecutor requires "
                f"'lifecycle: workspace_persistent' (got "
                f"{block_def.lifecycle!r})")

        key = self._attachment_key(block_name, workspace)
        runtime_handle = self._get_or_start_runtime(
            key=key,
            block_def=block_def,
            workspace=workspace,
            extra_env=extra_env,
            extra_mounts=extra_mounts,
            extra_docker_args=extra_docker_args,
        )

        exec_id = (
            execution_id or workspace.generate_execution_id())
        started_at = datetime.now(UTC)
        start_monotonic = time.monotonic()

        request_id = uuid.uuid4().hex

        # Stage inputs directly into the per-request input tree,
        # not into a freelance tempdir, so the container reads
        # them at the contract path.
        staged_inputs: dict[str, Path] = {}
        request_dir = (
            runtime_handle.work_area
            / runtime.REQUEST_TREE_WORKSPACE_RELATIVE
            / request_id
        )
        output_dirs: dict[str, Path] = {}
        try:
            request_dir.mkdir(parents=True)
            input_root = request_dir / "input"
            input_root.mkdir()
            for slot in block_def.inputs:
                target = input_root / slot.name
                staged_inputs[slot.name] = (
                    _stage_single_input_to_path(
                        workspace, slot.name,
                        input_bindings.get(slot.name), target)
                )
            output_root = request_dir / "output"
            output_root.mkdir()
            for slot in block_def.outputs:
                out = output_root / slot.name
                out.mkdir()
                output_dirs[slot.name] = out
        except StagingError as e:
            shutil.rmtree(request_dir, ignore_errors=True)
            cleanup_staged_inputs(staged_inputs)
            return _record_pre_container_failure(
                workspace=workspace,
                block_def=block_def,
                exec_id=exec_id,
                started_at=started_at,
                input_bindings=input_bindings,
                state_lineage_id=state_lineage_id,

                run_id=run_id,
                failure_phase=runtime.FAILURE_STAGE_IN,
                error=f"stage_in: {e}",
                staged_inputs={},
                state_mount=None,
                output_tempdirs={},
                work_area=None,
            )
        except Exception as e:
            shutil.rmtree(request_dir, ignore_errors=True)
            cleanup_staged_inputs(staged_inputs)
            return _record_pre_container_failure(
                workspace=workspace,
                block_def=block_def,
                exec_id=exec_id,
                started_at=started_at,
                input_bindings=input_bindings,
                state_lineage_id=state_lineage_id,

                run_id=run_id,
                failure_phase=runtime.FAILURE_STAGE_IN,
                error=f"stage_in: {e}",
                staged_inputs={},
                state_mount=None,
                output_tempdirs={},
                work_area=None,
            )

        return _RequestExecutionHandle(
            runtime=runtime_handle,
            request_id=request_id,
            execution_id=exec_id,
            started_at=started_at,
            start_monotonic=start_monotonic,
            input_bindings=input_bindings,
            state_lineage_id=state_lineage_id,
            run_id=run_id,
            staged_inputs=staged_inputs,
            request_dir=request_dir,
            output_dirs=output_dirs,
            execute_timeout_s=self._execute_timeout_s,
            cancel_timeout_s=self._cancel_timeout_s,
            post_check=self._post_checks.get(block_name),
        )

    def shutdown(
        self,
        attachment_key: str,
        reason: str = "requested",
        *,
        block_name: str | None = None,
        workspace: Workspace | None = None,
    ) -> bool:
        """Tear down the runtime for an attachment key.

        Checks the in-process registry first.  When the key is
        absent — the common fresh-CLI-process case — falls back
        to finding the container by name via ``docker inspect``
        and tearing it down directly, provided ``block_name``
        and ``workspace`` are supplied so the container name can
        be reconstructed.

        Writes ``/scratch/.stop`` to request cooperative
        shutdown; if the container does not exit within its
        ``stop_timeout_s``, sends SIGTERM and then SIGKILL.
        Idempotent; returns ``False`` when no runtime was found
        to tear down and ``True`` when teardown was attempted.
        """
        with self._registry_lock:
            runtime_handle = self._runtimes.pop(
                attachment_key, None)
        if runtime_handle is not None:
            self._teardown_runtime(runtime_handle, reason)
            return True

        # Fresh-process fallback: the container is live in
        # Docker but not in this process's registry.  Reconstruct
        # its name from the workspace + block and tear it down.
        if block_name is None or workspace is None:
            return False
        container_name = _runtime_container_name(
            workspace, block_name)
        info = _docker_ps_find(container_name)
        if info is None:
            return False
        # Refuse to tear down foreign containers that happen to
        # share a name — this would be a symptom of a separate
        # install or a pre-existing container.  Operator should
        # investigate via ``docker inspect``.
        if (info.get(_RUNTIME_LABEL_LIFECYCLE)
                != "workspace_persistent"):
            return False
        block_def = _find_block(self._template, block_name)
        stop_timeout = (
            block_def.stop_timeout_s if block_def is not None
            else 30
        )
        work_area = (
            workspace.path / "runtimes" / block_name)
        self._teardown_container(
            container_name,
            work_area=work_area if work_area.is_dir() else None,
            stop_timeout_s=stop_timeout,
        )
        return True

    def attached_keys(self) -> list[str]:
        """Return the attachment keys with live runtimes."""
        with self._registry_lock:
            return list(self._runtimes)

    def ensure_runtime(
        self,
        block_name: str,
        workspace: Workspace,
        *,
        extra_env: dict[str, str] | None = None,
        extra_mounts: list[tuple[str, str, str]] | None = None,
        extra_docker_args: list[str] | None = None,
    ) -> int:
        """Start (or attach to) a runtime without firing a request.

        Returns the runtime's ``control_port``, suitable for
        direct HTTP access — e.g., read-only bootstrap endpoints
        invoked by host-side pre-launch work (project hooks
        seeding incremental artifacts from the runtime's current
        state).  For request/response work the caller should
        still go through :meth:`launch`.

        Validation mirrors :meth:`launch`: the block must exist
        in the template, have ``runner: container``, and declare
        ``lifecycle: workspace_persistent``.  ``extra_env`` /
        ``extra_mounts`` / ``extra_docker_args`` are consumed on
        first-start only; later calls for the same
        ``(block_name, workspace)`` reattach to the existing
        runtime and ignore the kwargs.

        Args:
            block_name: Name of the block to ensure.
            workspace: Workspace the runtime is scoped to.
            extra_env: First-start env additions.
            extra_mounts: First-start bind mounts
                (``(host, container, mode)`` tuples).
            extra_docker_args: First-start raw docker args.

        Returns:
            The TCP port the runtime's control channel listens on
            (bound to ``host.docker.internal`` from the container
            and ``localhost`` from the host).

        Raises:
            ValueError: If the block is missing from the template,
                has a non-container runner, or is not declared
                ``lifecycle: workspace_persistent``.
        """
        block_def = _find_block(self._template, block_name)
        if block_def is None:
            raise ValueError(
                f"Block {block_name!r} not found in template")
        if block_def.runner != "container":
            raise ValueError(
                f"Block {block_name!r}: "
                f"RequestResponseExecutor only runs container "
                f"blocks (got runner {block_def.runner!r})")
        if block_def.lifecycle != "workspace_persistent":
            raise ValueError(
                f"Block {block_name!r}: "
                f"RequestResponseExecutor requires "
                f"'lifecycle: workspace_persistent' (got "
                f"{block_def.lifecycle!r})")

        key = self._attachment_key(block_name, workspace)
        handle = self._get_or_start_runtime(
            key=key,
            block_def=block_def,
            workspace=workspace,
            extra_env=extra_env,
            extra_mounts=extra_mounts,
            extra_docker_args=extra_docker_args,
        )
        return handle.control_port

    def _get_or_start_runtime(
        self,
        *,
        key: str,
        block_def: BlockDefinition,
        workspace: Workspace,
        extra_env: dict[str, str] | None,
        extra_mounts: list[tuple[str, str, str]] | None,
        extra_docker_args: list[str] | None,
    ) -> _RuntimeHandle:
        """Return the runtime for ``key``, starting or attaching if needed."""
        with self._registry_lock:
            existing = self._runtimes.get(key)
            if existing is not None:
                return existing
            # Try to reattach to a pre-existing container of the
            # same name; start a fresh one otherwise.
            handle = self._attach_existing_runtime(
                key=key,
                block_def=block_def,
                workspace=workspace,
            )
            if handle is None:
                handle = self._start_new_runtime(
                    key=key,
                    block_def=block_def,
                    workspace=workspace,
                    extra_env=extra_env,
                    extra_mounts=extra_mounts,
                    extra_docker_args=extra_docker_args,
                )
            self._runtimes[key] = handle
            return handle

    def _attach_existing_runtime(
        self,
        *,
        key: str,
        block_def: BlockDefinition,
        workspace: Workspace,
    ) -> _RuntimeHandle | None:
        """Probe for an existing container; attach if healthy.

        Returns the attached handle, ``None`` if no container of
        the expected name exists, or raises
        :class:`ControlChannelError` if a container by that name
        exists but has mismatched labels — we refuse to operate
        that container and refuse to start a new one with the
        same name, since Docker would error with a raw name-
        conflict.  Operator surfaces the problem via
        ``flywheel container list`` and removes the foreign
        container by hand.
        """
        container_name = _runtime_container_name(
            workspace, block_def.name)
        info = _docker_ps_find(container_name)
        if info is None:
            return None
        expected = _runtime_labels(block_def, workspace)
        for label, value in expected.items():
            if info.get(label) != value:
                raise ControlChannelError(
                    f"Docker container {container_name!r} "
                    f"exists with foreign labels (expected "
                    f"{label}={value!r}, got "
                    f"{info.get(label)!r}); refusing to attach "
                    f"or overwrite.  Remove the container "
                    f"manually or rename the workspace."
                )
        if info.get("_running") != "true":
            # Our container, but stopped.  Caller starts a fresh
            # one; Docker's --rm plus container-name reuse is
            # fine for the stopped-but-labeled case.
            return None
        # Rediscover the published port from docker inspect.
        port_raw = info.get("flywheel.control_port")
        if not port_raw:
            return None
        try:
            port = int(port_raw)
        except ValueError:
            return None
        channel = self._channel_factory("127.0.0.1", port)
        if not self._wait_for_health(
                channel, timeout_s=self._startup_timeout_s):
            return None
        work_area = (
            workspace.path / "runtimes" / block_def.name)
        work_area.mkdir(parents=True, exist_ok=True)
        return _RuntimeHandle(
            attachment_key=key,
            container_name=container_name,
            control_port=port,
            block_def=block_def,
            workspace=workspace,
            work_area=work_area,
            process=None,
            channel=channel,
        )

    def _start_new_runtime(
        self,
        *,
        key: str,
        block_def: BlockDefinition,
        workspace: Workspace,
        extra_env: dict[str, str] | None,
        extra_mounts: list[tuple[str, str, str]] | None,
        extra_docker_args: list[str] | None,
    ) -> _RuntimeHandle:
        """Allocate a port, start the container detached, wait for health.

        The container is started with ``docker run -d`` so the
        host process can exit without taking the runtime down
        with it.  Reattachment after a host-process restart
        requires this.
        """
        container_name = _runtime_container_name(
            workspace, block_def.name)
        port = _allocate_free_port()

        work_area = (
            workspace.path / "runtimes" / block_def.name)
        work_area.mkdir(parents=True, exist_ok=True)
        (work_area
         / runtime.REQUEST_TREE_WORKSPACE_RELATIVE).mkdir(
            parents=True, exist_ok=True)

        labels = _runtime_labels(block_def, workspace)
        labels["flywheel.control_port"] = str(port)

        env = dict(block_def.env)
        env[runtime.CONTROL_PORT_ENV_VAR] = str(port)
        if extra_env:
            env.update(extra_env)

        docker_args = list(block_def.docker_args)
        for label, value in labels.items():
            docker_args.extend(["--label", f"{label}={value}"])
        # Publish the chosen port to localhost only.
        docker_args.extend([
            "-p", f"127.0.0.1:{port}:{port}",
        ])
        if extra_docker_args:
            docker_args.extend(extra_docker_args)

        mounts: list[tuple[str, str, str]] = [
            (str(work_area.resolve()), "/scratch", "rw"),
        ]
        if extra_mounts:
            mounts.extend(extra_mounts)

        cc = ContainerConfig(
            image=block_def.image,
            docker_args=docker_args,
            env=env,
            mounts=mounts,
        )
        _run_detached_container(cc, container_name)

        channel = self._channel_factory("127.0.0.1", port)
        if not self._wait_for_health(
                channel, timeout_s=self._startup_timeout_s):
            # Health probe failed — tear down the container we
            # just started so we don't leave a stuck runtime.
            _docker_kill(container_name, "TERM")
            if not _docker_wait_gone(
                    container_name,
                    timeout_s=_TERM_TO_KILL_GRACE_S):
                _docker_kill(container_name, "KILL")
                _docker_wait_gone(
                    container_name, timeout_s=30.0)
            shutil.rmtree(work_area, ignore_errors=True)
            raise ControlChannelError(
                f"Runtime {container_name!r} did not report "
                f"ready within {self._startup_timeout_s}s"
            )

        return _RuntimeHandle(
            attachment_key=key,
            container_name=container_name,
            control_port=port,
            block_def=block_def,
            workspace=workspace,
            work_area=work_area,
            process=None,
            channel=channel,
        )

    def _wait_for_health(
        self, channel: ControlChannel, *, timeout_s: float,
    ) -> bool:
        """Poll ``channel.health()`` until ready or ``timeout_s`` elapses."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if channel.health(
                    timeout_s=_HEALTH_POLL_INTERVAL_S):
                return True
            time.sleep(_HEALTH_POLL_INTERVAL_S)
        return False

    def _teardown_runtime(
        self,
        handle: _RuntimeHandle,
        reason: str,
    ) -> None:
        """Two-phase teardown: sentinel, then TERM, then KILL.

        Mirrors :meth:`ContainerExecutionHandle.stop` semantics
        for the request-response runtime: write
        ``/scratch/.stop`` first, poll for container exit up
        to ``stop_timeout_s``, escalate if the container doesn't
        exit on its own.  Best-effort; transport or docker
        errors do not raise.  The runtime is detached so there
        is no Popen to wait on; we track liveness via
        ``docker inspect``.
        """
        self._teardown_container(
            handle.container_name,
            work_area=handle.work_area,
            stop_timeout_s=handle.block_def.stop_timeout_s,
        )

    def _teardown_container(
        self,
        container_name: str,
        *,
        work_area: Path | None,
        stop_timeout_s: int,
    ) -> None:
        """Sentinel + TERM + KILL for a named runtime container.

        Factored out so :meth:`shutdown` can tear down a
        runtime found in Docker but not in the in-process
        registry (the fresh-CLI-process case).  ``work_area``
        may be ``None`` when the caller didn't discover it
        (e.g. CLI-driven teardown of a runtime whose work_area
        layout matches the workspace conventions but the caller
        doesn't want to guess).
        """
        if work_area is not None:
            sentinel = (
                work_area
                / runtime.STOP_SENTINEL_WORKSPACE_RELATIVE
            )
            with contextlib.suppress(OSError):
                sentinel.touch()

        cooperative = max(0, stop_timeout_s)
        if cooperative > 0:
            if _docker_wait_gone(
                    container_name, timeout_s=cooperative):
                if work_area is not None:
                    shutil.rmtree(
                        work_area, ignore_errors=True)
                return

        _docker_kill(container_name, "TERM")
        if not _docker_wait_gone(
                container_name,
                timeout_s=_TERM_TO_KILL_GRACE_S):
            _docker_kill(container_name, "KILL")
            _docker_wait_gone(
                container_name, timeout_s=30.0)

        if work_area is not None:
            shutil.rmtree(work_area, ignore_errors=True)


def _stage_single_input_to_path(
    workspace: Workspace,
    slot_name: str,
    instance_id: str | None,
    target: Path,
) -> Path:
    """Stage one artifact instance into a specific target path.

    Unlike :func:`stage_artifact_instances` (which allocates
    tempdirs), this reuses a caller-supplied path — used by the
    request-response executor to stage directly under the
    per-request tree.  Returns ``target`` so callers can track
    it for cleanup.

    Raises:
        StagingError: if the instance does not exist or cannot
            be staged.
    """
    target.mkdir(parents=True, exist_ok=True)
    if instance_id is None:
        return target
    bindings = {slot_name: instance_id}
    # Delegate to the existing staging helper to keep copy /
    # incremental semantics identical; then move its output
    # into the target path.
    staged = stage_artifact_instances(workspace, bindings)
    src = staged.get(slot_name)
    if src is None:
        return target
    try:
        for child in src.iterdir():
            dest = target / child.name
            if child.is_dir():
                shutil.copytree(child, dest)
            else:
                shutil.copy2(child, dest)
    finally:
        cleanup_staged_inputs(staged)
    return target
