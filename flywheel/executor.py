"""Block executor abstractions and concrete implementations.

Defines the protocol for executing blocks and the concrete
executor types.  A block executor takes a block definition, a
workspace, and input bindings, and produces an execution result
with output artifacts.

- **ContainerExecutor**: Runs a block in a Docker container.
- **ProcessExecutor**: Runs a block as a local subprocess.
  For trusted, host-local processes like game servers.

``runner: lifecycle`` blocks have no executor — they are
recorded directly by
:class:`flywheel.local_block.LocalBlockRecorder`.

All executors satisfy the ``BlockExecutor`` protocol and return
an ``ExecutionHandle`` from ``launch()``.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from flywheel.artifact import ArtifactInstance, BlockExecution
from flywheel.container import ContainerConfig, run_container
from flywheel.template import BlockDefinition, Template
from flywheel.workspace import Workspace


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
                execution row when known by the caller.
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


# ── ContainerExecutor ────────────────────────────────────────────


class ContainerExecutor:
    """Execute a block by launching a Docker container.

    Resolves inputs, builds a container config from the block
    definition, runs the container, and records artifacts and
    execution in the workspace.

    Args:
        template: The template containing block definitions.
        overrides: Default CLI flag overrides for containers.
    """

    def __init__(
        self,
        template: Template,
        overrides: dict[str, Any] | None = None,
    ):
        """Initialize with a template and optional overrides."""
        self._template = template
        self._overrides = overrides

    def launch(
        self,
        block_name: str,
        workspace: Workspace,
        input_bindings: dict[str, str],
        *,
        outputs_data: dict[str, Any] | None = None,
        elapsed_s: float | None = None,
        execution_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        allowed_blocks: list[str] | None = None,
        artifact_path: str | None = None,
        stopping: Any | None = None,
        active_container: list | None = None,
        agent_workspace_dir: str | None = None,
    ) -> SyncExecutionHandle:
        """Launch a block in a Docker container (blocking).

        Args:
            block_name: Name of the block to execute.
            workspace: The flywheel workspace.
            input_bindings: Maps input slot names to artifact IDs.
            outputs_data: Unused (present for protocol compat).
            elapsed_s: Unused (present for protocol compat).
            execution_id: Pre-assigned execution ID.
            overrides: CLI flag overrides (merged with defaults).
            allowed_blocks: If set, only these block names allowed.
            artifact_path: Optional path to input artifact relative
                to the agent workspace, for callers that pass an
                explicit input directory.
            stopping: Threading event for cancellation.
            active_container: Single-element list tracking the
                running container name.
            agent_workspace_dir: Subdirectory name for the agent
                workspace. Defaults to ``"agent_workspace"``.

        Returns:
            A ``SyncExecutionHandle`` with the result.

        Raises:
            ValueError: If the block is not found, not allowed,
                or inputs are invalid.
            FileNotFoundError: If the artifact path does not exist.
        """
        if allowed_blocks and block_name not in allowed_blocks:
            raise ValueError(
                f"Block {block_name!r} not in allowed list: "
                f"{allowed_blocks}")

        block_def = _find_block(self._template, block_name)
        if block_def is None:
            raise ValueError(
                f"Block {block_name!r} not found in template")

        if artifact_path is not None:
            agent_ws_name = agent_workspace_dir or "agent_workspace"
            agent_ws = workspace.path / agent_ws_name
            resolved = (agent_ws / artifact_path).resolve()

            if not resolved.is_relative_to(agent_ws.resolve()):
                raise ValueError(
                    f"Artifact path escapes workspace: "
                    f"{artifact_path}")

            # Retry briefly for Windows/WSL2 bind mount delay.
            if not resolved.exists():
                for _ in range(3):
                    time.sleep(1)
                    if resolved.exists():
                        break
                else:
                    raise FileNotFoundError(
                        f"Artifact not found: {artifact_path}")

            if not block_def.inputs:
                raise ValueError(
                    f"Block {block_name!r} has no input slots")

            input_slot = block_def.inputs[0]
            input_instance = workspace.register_artifact(
                input_slot.name, resolved,
                source="agent invocation",
            )
            input_bindings = {input_slot.name: input_instance.id}

        if stopping is not None and stopping.is_set():
            raise ValueError("Cancelled — service is shutting down")

        # Build mounts from block definition.
        mounts: list[tuple[str, str, str]] = []

        for slot in block_def.inputs:
            aid = input_bindings.get(slot.name, "")
            if not aid or aid not in workspace.artifacts:
                continue
            inst = workspace.artifacts[aid]
            if inst.kind == "copy" and inst.copy_path:
                host_path = str(
                    (workspace.path / "artifacts" / inst.copy_path
                     ).resolve()
                ).replace("\\", "/")
                mounts.append(
                    (host_path, slot.container_path, "ro"))

        exec_id = execution_id or workspace.generate_execution_id()

        # Outputs: allocate artifact directories.
        output_artifact_ids: dict[str, str] = {}
        for output_slot in block_def.outputs:
            aid = workspace.generate_artifact_id(output_slot.name)
            output_dir = workspace.path / "artifacts" / aid
            output_dir.mkdir(parents=True)
            output_host = str(
                output_dir.resolve()).replace("\\", "/")
            mounts.append(
                (output_host, output_slot.container_path, "rw"))
            output_artifact_ids[output_slot.name] = aid

        container_name = f"flywheel-block-{exec_id}"

        cc = ContainerConfig(
            image=block_def.image,
            docker_args=list(block_def.docker_args),
            env=dict(block_def.env),
            mounts=mounts,
        )

        # Build args from overrides.
        merged_overrides = dict(self._overrides or {})
        if overrides:
            merged_overrides.update(overrides)
        container_args: list[str] = []
        for key, value in merged_overrides.items():
            flag = f"--{key.replace('_', '-')}"
            container_args += [flag, str(value)]

        # Run container.
        if active_container is not None:
            active_container[0] = container_name

        started_at = datetime.now(UTC)
        try:
            result = run_container(
                cc,
                args=container_args or None,
                name=container_name,
            )
        finally:
            if active_container is not None:
                active_container[0] = None
        finished_at = datetime.now(UTC)

        # Record results.
        status = (
            "succeeded" if result.exit_code == 0 else "failed")
        output_bindings: dict[str, str] = {}

        for output_slot in block_def.outputs:
            aid = output_artifact_ids[output_slot.name]
            output_dir_path = workspace.path / "artifacts" / aid
            if (output_dir_path.exists()
                    and any(output_dir_path.iterdir())):
                output_instance = ArtifactInstance(
                    id=aid,
                    name=output_slot.name,
                    kind="copy",
                    created_at=finished_at,
                    produced_by=exec_id,
                    copy_path=aid,
                )
                workspace.add_artifact(output_instance)
                output_bindings[output_slot.name] = aid
            else:
                shutil.rmtree(output_dir_path, ignore_errors=True)

        execution = BlockExecution(
            id=exec_id,
            block_name=block_name,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            input_bindings=input_bindings,
            output_bindings=output_bindings,
            exit_code=result.exit_code,
            elapsed_s=result.elapsed_s,
            image=block_def.image,
        )
        workspace.add_execution(execution)
        workspace.save()

        return SyncExecutionHandle(ExecutionResult(
            exit_code=result.exit_code,
            elapsed_s=result.elapsed_s,
            output_bindings=output_bindings,
            execution_id=exec_id,
            status=status,
        ))


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
