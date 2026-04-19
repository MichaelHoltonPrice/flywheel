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

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from flywheel import runtime
from flywheel.artifact import ArtifactInstance, BlockExecution
from flywheel.container import ContainerConfig, run_container
from flywheel.input_staging import (
    StagingError,
    cleanup_staged_inputs,
    stage_artifact_instances,
)
from flywheel.template import BlockDefinition, Template
from flywheel.workspace import Workspace

# Sentinel exit code for "the container never produced one."
# Used when ``run_container`` raises before an exit status exists
# (Docker daemon gone, container failed to start, etc.).  ``-1``
# distinguishes this case from a real successful exit in the
# returned :class:`ExecutionResult`; the ``BlockExecution.status``
# field is the authoritative signal, but API-level callers
# reading just ``exit_code`` should see a non-zero value.
INVOKE_FAILURE_EXIT_CODE: int = -1


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
    ) -> SyncExecutionHandle:
        """Launch a one-shot container and return when it exits.

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

        Returns:
            A :class:`SyncExecutionHandle` wrapping the
            execution's result.  ``wait()`` returns immediately
            with a pre-computed :class:`ExecutionResult`.

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

        # Pipeline state.  ``failure_phase`` is the first phase
        # that failed; subsequent phases still run (best-effort
        # state/output capture for debugging) but do not
        # overwrite it.
        failure_phase: str | None = None
        error: str | None = None
        staged_inputs: dict[str, Path] = {}
        state_mount: Path | None = None
        output_tempdirs: dict[str, Path] = {}
        output_bindings: dict[str, str] = {}
        state_dir_rel: str | None = None
        result = None
        finished_at = started_at

        try:
            # Phase: stage_in.  Any staging failure aborts
            # immediately — the container can't run without its
            # declared inputs.
            try:
                staged_inputs = stage_artifact_instances(
                    workspace, input_bindings)
            except StagingError as e:
                failure_phase = runtime.FAILURE_STAGE_IN
                error = f"stage_in: {e}"
                raise _AbortExecution from e

            # Populate /state/ mount for stateful blocks.  Counted
            # as part of stage_in — a missing prior state isn't a
            # failure (first execution has none), but an I/O
            # error copying the canonical state dir is.  A prior
            # execution whose state_dir is recorded in the ledger
            # but missing on disk is *also* a stage_in failure —
            # silently cold-starting would mask workspace
            # corruption.
            if block_def.state:
                try:
                    state_mount = _populate_state_mount(
                        workspace, block_name, state_lineage_id)
                except OSError as e:
                    failure_phase = runtime.FAILURE_STAGE_IN
                    error = f"stage_in (state): {e}"
                    raise _AbortExecution from e

            # Allocate per-slot output tempdirs.  Write-side of
            # the "canonical never directly mounted" invariant —
            # the container writes here; we move to canonical
            # after exit.
            for slot in block_def.outputs:
                output_tempdirs[slot.name] = Path(
                    tempfile.mkdtemp(
                        prefix=f"flywheel-output-{slot.name}-"))

            # Phase: invoke.  Build mounts and run.
            mounts = _build_mounts(
                block_def, staged_inputs, output_tempdirs,
                state_mount,
            )
            cc = ContainerConfig(
                image=block_def.image,
                docker_args=list(block_def.docker_args),
                env=dict(block_def.env),
                mounts=mounts,
            )
            container_args = _build_container_args(
                self._overrides, overrides)
            container_name = f"flywheel-block-{exec_id}"

            try:
                result = run_container(
                    cc,
                    args=container_args or None,
                    name=container_name,
                )
            except Exception as e:
                failure_phase = runtime.FAILURE_INVOKE
                error = f"invoke: {e}"
                finished_at = datetime.now(UTC)
                raise _AbortExecution from e

            finished_at = datetime.now(UTC)

            # Non-zero exit == block body failure.  Phase is
            # ``invoke``; we still capture state/outputs for
            # debugging.
            if result.exit_code != 0:
                failure_phase = runtime.FAILURE_INVOKE
                error = (
                    f"invoke: container exited with code "
                    f"{result.exit_code}"
                )

            # Phase: state_capture.  Runs even on invoke failure
            # — the container may have written partial state
            # worth preserving.
            if state_mount is not None:
                try:
                    state_dir_rel = _capture_state(
                        workspace, block_name, exec_id,
                        state_mount,
                    )
                except OSError as e:
                    if failure_phase is None:
                        failure_phase = runtime.FAILURE_STATE_CAPTURE
                        error = f"state_capture: {e}"

            # Phase: output_collect.  Move per-slot output
            # tempdir contents into canonical artifact dirs and
            # register.
            try:
                output_bindings = _collect_outputs(
                    workspace, block_def, output_tempdirs,
                    exec_id, finished_at,
                )
            except Exception as e:
                if failure_phase is None:
                    failure_phase = runtime.FAILURE_OUTPUT_COLLECT
                    error = f"output_collect: {e}"

        except _AbortExecution:
            # Phase already set; fall through to record-keeping.
            if finished_at == started_at:
                finished_at = datetime.now(UTC)
        finally:
            # Cleanup is best-effort and must never mask a real
            # failure with a cleanup error.
            cleanup_staged_inputs(staged_inputs)
            if state_mount is not None:
                shutil.rmtree(state_mount, ignore_errors=True)
            for path in output_tempdirs.values():
                shutil.rmtree(path, ignore_errors=True)

        # Phase: artifact_commit.  Record the execution.  If
        # add_execution/save raise, there's no recoverable way to
        # leave a phase marker behind — the record can't be
        # written — so we propagate.  Robust-commit is a future
        # concern.
        status = "failed" if failure_phase else "succeeded"
        exit_code = result.exit_code if result is not None else None
        elapsed_s = result.elapsed_s if result is not None else 0.0

        execution = BlockExecution(
            id=exec_id,
            block_name=block_name,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            input_bindings=input_bindings,
            output_bindings=output_bindings,
            exit_code=exit_code,
            elapsed_s=elapsed_s,
            image=block_def.image,
            runner="container",
            state_dir=state_dir_rel,
            failure_phase=failure_phase,
            error=error,
            state_lineage_id=state_lineage_id,
        )
        workspace.add_execution(execution)
        workspace.save()

        # ``exit_code`` may be None when ``run_container`` raised
        # before producing a result.  Surface a clear non-zero
        # sentinel so API-level callers reading just the
        # ExecutionResult see the failure.
        result_exit_code = (
            exit_code if exit_code is not None
            else INVOKE_FAILURE_EXIT_CODE
        )
        return SyncExecutionHandle(ExecutionResult(
            exit_code=result_exit_code,
            elapsed_s=elapsed_s,
            output_bindings=output_bindings,
            execution_id=exec_id,
            status=status,
        ))


def _populate_state_mount(
    workspace: Workspace,
    block_name: str,
    state_lineage_id: str | None,
) -> Path:
    """Create a tempdir populated with the block's prior state.

    Walks the workspace's executions in reverse chronological
    order, picks the most recent successful execution of
    ``block_name`` within the given ``state_lineage_id``, and
    copies its captured state into a fresh tempdir.  The tempdir
    is what gets mounted at ``/state/`` inside the container.

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

    # Find the most recent successful execution with state in
    # the target lineage.
    prior: BlockExecution | None = None
    for ex in sorted(
        workspace.executions.values(),
        key=lambda e: e.started_at,
        reverse=True,
    ):
        if (ex.block_name == block_name
                and ex.status == "succeeded"
                and ex.state_dir is not None
                and ex.state_lineage_id == state_lineage_id):
            prior = ex
            break

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


def _capture_state(
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
) -> list[tuple[str, str, str]]:
    """Assemble the (host, container, mode) mount tuples.

    Never includes a canonical workspace path — every mount
    points at a per-execution staged tempdir.
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
    return mounts


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
