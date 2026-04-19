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
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from flywheel import runtime
from flywheel.artifact import ArtifactInstance, BlockExecution
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

    Implementations receive a block definition, workspace, and
    input bindings, then produce output artifacts and record the
    execution.  The ``launch()`` method returns an
    ``ExecutionHandle`` for non-blocking control.
    """

    def launch(
        self,
        block_name: str,
        workspace: Any,
        input_bindings: dict[str, str],
        *,
        outputs_data: dict[str, Any] | None = None,
        elapsed_s: float | None = None,
        execution_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        allowed_blocks: list[str] | None = None,
    ) -> ExecutionHandle:
        """Launch a block execution.

        Args:
            block_name: Name of the block to execute.
            workspace: The flywheel workspace.
            input_bindings: Maps input slot names to artifact
                instance IDs.
            outputs_data: Optional output data dicts to write as
                artifacts (used by executors that produce outputs
                from in-memory data rather than container output).
            elapsed_s: Optional wall-clock time to record on the
                execution record when known by the caller.
            execution_id: Optional pre-assigned execution ID.
            overrides: CLI flag overrides for container blocks.
            allowed_blocks: If set, only these block names are
                permitted.

        Returns:
            An ``ExecutionHandle`` for monitoring and collecting
            the result.
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
        ``/workspace/.stop``) and wait up to the block's
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
            extra_env: Per-launch environment variables to merge
                into ``block_def.env`` before building the
                ``ContainerConfig``.  Later writes win, so a
                launcher can override a block-declared default.
            extra_mounts: Per-launch bind mounts appended to the
                block's staged-input / output / state / workspace
                mount list.  Each entry is ``(host_path,
                container_path, mode)``.  Mount paths must not
                collide with the contract-reserved ``/input/*``,
                ``/output/*``, ``/state``, or ``/workspace`` —
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

            # Allocate the /workspace/ work-area tempdir where
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
    is mounted rw at ``/workspace/`` so flywheel and the
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
    mounts.append((str(work_area), "/workspace", "rw"))
    return mounts


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
    ):
        """Initialize from a running subprocess."""
        self._process = process
        self._workspace = workspace
        self._block_name = block_name
        self._execution_id = execution_id
        self._started_at = started_at
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
        outputs_data: dict[str, Any] | None = None,
        elapsed_s: float | None = None,
        execution_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        allowed_blocks: list[str] | None = None,
    ) -> ProcessExecutionHandle:
        """Launch a local subprocess.

        Args:
            block_name: Name of the block.
            workspace: The flywheel workspace.
            input_bindings: Input bindings (recorded but not
                mounted — local processes access files directly).
            command: Shell command string or argument list.
                Required.
            outputs_data: Unused (protocol compat).
            elapsed_s: Unused (protocol compat).
            execution_id: Pre-assigned execution ID.
            overrides: Unused (protocol compat).
            allowed_blocks: If set, only these block names allowed.

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
        )
