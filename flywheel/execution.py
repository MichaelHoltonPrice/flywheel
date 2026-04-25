"""Block execution orchestration for flywheel.

The public entry point is :func:`run_block`, which is a thin
``prepare -> run -> commit`` orchestration over the canonical
block-execution model:

* :func:`prepare_block_execution` — resolves inputs, allocates
  per-slot proposal directories, builds the container mount set,
  and mints the execution ID.  Returns an :class:`ExecutionPlan`.
* :func:`run_one_shot_container` — runs a fresh container
  for this execution, waits for it to exit, and parses the
  termination announcement from the sidecar at
  ``/flywheel/termination``.  This is the one-shot container
  runtime strategy.  Container-specific mechanics are hidden
  behind :class:`OneShotContainerRunner`.
* :func:`commit_block_execution` — forges validated proposals
  through ``Workspace.register_artifact`` (proposal-then-forge),
  quarantines rejections, and records the :class:`BlockExecution`
  via ``Workspace.record_execution``.

``flywheel run block`` is the canonical happy-path call site
today.  Container-specific code plugs into the runtime phase as a
narrow runner: it runs work described by an :class:`ExecutionPlan`
and reports an :class:`RuntimeResult`; it does not own input
resolution, artifact forging, quarantine, or ledger semantics.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from flywheel import runtime
from flywheel.artifact import (
    ArtifactInstance,
    BlockExecution,
    RejectedOutput,
)
from flywheel.artifact_validator import (
    ArtifactValidationError,
    ArtifactValidatorRegistry,
)
from flywheel.container import ContainerConfig, ContainerResult, run_container
from flywheel.quarantine import quarantine_slot
from flywheel.state import (
    StateMode,
    state_compatibility_identity,
)
from flywheel.template import (
    ArtifactDeclaration,
    BlockDefinition,
    Template,
)
from flywheel.termination import (
    derive_status,
    normalize_termination_reason,
    read_termination_sidecar,
)
from flywheel.workspace import Workspace

# ── Plan & runtime result ─────────────────────────────────────


@dataclass
class ExecutionPlan:
    """Output of the prepare phase — everything runtime/commit need.

    A plan is constructed once per execution and threaded through
    :func:`run_one_shot_container` and
    :func:`commit_block_execution`.  No
    field is mutated after construction; downstream phases only
    read.

    Attributes:
        execution_id: The minted execution ID.
        block_def: The resolved block definition.
        block_name: Convenience copy of ``block_def.name``.
        started_at: Wall-clock timestamp captured at prepare time.
        resolved_bindings: ``{slot_name: artifact_id}`` for every
            input the block was given.
        mounts: Container mount tuples ``(host, container, mode)``.
        proposal_dirs: ``{slot_name: host_path}`` for the per-slot
            proposal directories the block writes to.
        proposals_root: Parent directory of all per-slot proposal
            dirs and the termination sidecar dir.  Removed after
            commit completes.
        termination_file: Path the block writes its
            termination-reason announcement to (sidecar at
            ``/flywheel/termination`` inside the container).
        state_mode: Substrate-visible state mode for this execution.
        state_lineage_key: Managed state lineage key, when applicable.
        state_mount_dir: Host directory mounted at ``/flywheel/state``.
        restored_state_snapshot_id: Snapshot restored into the state
            mount, when one existed.
        state_compatibility: Compatibility identity used to reject
            restoring an incompatible snapshot.
    """

    execution_id: str
    block_def: BlockDefinition
    block_name: str
    started_at: datetime
    resolved_bindings: dict[str, str]
    mounts: list[tuple[str, str, str]]
    proposal_dirs: dict[str, Path]
    proposals_root: Path
    termination_file: Path
    state_mode: StateMode = "none"
    state_lineage_key: str | None = None
    state_mount_dir: Path | None = None
    restored_state_snapshot_id: str | None = None
    state_compatibility: dict[str, str] | None = None


@dataclass
class RuntimeResult:
    """Output of the runtime phase -- what the runtime observed.

    Constructed by a runtime-specific runner or adapter and consumed
    by :func:`commit_block_execution`. Carries pure observation; no
    policy decisions live here. Commit derives status, failure phase,
    output handling, and user-facing exception text from these fields.

    Attributes:
        termination_reason: The normalized termination reason.
            Either a substrate-reserved value (``crash``,
            ``interrupted``, ``timeout``, ``protocol_violation``)
            or a project-defined value the block declared. Always
            populated. This is the only field commit branches on.
        container_result: The raw ``ContainerResult`` from the
            runtime, or ``None`` when no container exit metadata is
            available.
        announcement: The raw, stripped string the block wrote to
            the termination sidecar, or ``None`` if no sidecar was
            readable. This is an evidence field, preserved regardless
            of how ``termination_reason`` normalized. Commit reads it
            only when composing the canonical ``protocol_violation``
            fallback message. Do not parse it for control flow.
        error: Optional human-readable diagnostic from the runtime
            layer. Consulted by commit only for substrate-reserved
            failure-shaped reasons. Ignored for project-defined
            reasons; commit owns per-slot rejection text on those
            branches. When supplied it must be non-empty after
            stripping whitespace. ``None`` or whitespace-only text
            uses the canonical fallback for the termination reason.
            The same text commit chooses is written to
            ``BlockExecution.error`` and used in the raised exception
            message. This field is intentionally unstructured and
            must not be parsed for control flow.
    """

    termination_reason: str
    container_result: ContainerResult | None
    announcement: str | None = None
    error: str | None = None


class OneShotContainerRunner(Protocol):
    """Container-specific runner for an already-prepared block body.

    Implementations consume a fully prepared :class:`ExecutionPlan`
    and return an :class:`RuntimeResult`. They must not resolve
    inputs, commit artifacts, quarantine outputs, or record ledger
    entries; those are owned by prepare/commit.

    Interrupt handling supports both local-process and scheduler-style
    runners. A runner may raise ``KeyboardInterrupt`` from ``run``;
    :func:`run_block` records the execution as interrupted. Or it
    may return ``RuntimeResult(termination_reason="interrupted",
    error=...)``; :func:`commit_block_execution` records the same
    canonical interrupted outcome. Other reserved failure reasons
    (``crash``, ``timeout``, ``protocol_violation``) are reported by
    returning an :class:`RuntimeResult`. Exceptions other than
    ``KeyboardInterrupt`` raised from ``run`` are recorded as
    ``crash`` by :func:`run_block`.
    """

    def run(
        self, plan: ExecutionPlan, args: list[str] | None = None,
    ) -> RuntimeResult:
        """Run the prepared container body and report how it ended."""
        ...


# ── Helpers ──────────────────────────────────────────────────────


def _find_artifact_declaration(
    template: Template, name: str
) -> ArtifactDeclaration | None:
    """Find an artifact declaration by name in a template."""
    for decl in template.artifacts:
        if decl.name == name:
            return decl
    return None


def _mount_artifact_instance(
    instance: ArtifactInstance, container_path: str,
    workspace: Workspace,
) -> tuple[str, str, str]:
    """Build a Docker volume mount tuple for an artifact instance.

    Args:
        instance: The artifact instance to mount.
        container_path: Where to mount it inside the container.
        workspace: The workspace containing the artifact.

    Returns:
        A (host_path, container_path, mode) tuple.

    Raises:
        ValueError: If the artifact kind is not supported for mounting,
            or required fields are missing.
    """
    if instance.kind == "git":
        if instance.repo is None or instance.git_path is None:
            raise ValueError(
                f"Git artifact {instance.id!r} missing repo or git_path"
            )
        host_path = str(
            (Path(instance.repo) / instance.git_path).resolve()
        )
    elif instance.kind == "copy":
        if instance.copy_path is None:
            raise ValueError(
                f"Copy artifact {instance.id!r} missing copy_path"
            )
        host_path = str(
            (workspace.path / "artifacts" / instance.copy_path).resolve()
        )
    else:
        raise ValueError(
            f"Artifact {instance.id!r} has unsupported kind "
            f"{instance.kind!r}"
        )
    return (host_path, container_path, "ro")


def _copy_dir_contents(source: Path, target: Path) -> None:
    """Copy a directory's contents into an existing target directory."""
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


def _record_execution(
    workspace: Workspace, execution_id: str, block_name: str,
    started_at: datetime,
    termination_reason: str,
    input_bindings: dict[str, str],
    output_bindings: dict[str, str],
    exit_code: int | None, elapsed_s: float | None,
    image: str,
    *,
    all_expected_committed: bool,
    failure_phase: str | None = None,
    error: str | None = None,
    rejected_outputs: dict[str, RejectedOutput] | None = None,
    state_mode: StateMode = "none",
    state_snapshot_id: str | None = None,
) -> None:
    """Record a block execution and persist the workspace.

    Status is derived from ``termination_reason`` via the canonical
    status-mapping rule (see
    :func:`flywheel.termination.derive_status`).  Callers do not
    pass ``status`` directly.
    """
    status = derive_status(
        termination_reason,
        all_expected_committed=all_expected_committed,
    )
    execution = BlockExecution(
        id=execution_id,
        block_name=block_name,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        status=status,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        exit_code=exit_code,
        elapsed_s=elapsed_s,
        image=image,
        runner="container_one_shot",
        failure_phase=failure_phase,
        error=error,
        rejected_outputs=rejected_outputs or {},
        termination_reason=termination_reason,
        state_mode=state_mode,
        state_snapshot_id=state_snapshot_id,
    )
    workspace.record_execution(execution)


# ── Failure recording helper (prepare/runtime phases) ─────────────


def commit_failure(
    workspace: Workspace,
    *,
    execution_id: str,
    block_name: str,
    started_at: datetime,
    image: str,
    phase: str,
    error: str,
    termination_reason: str = runtime.TERMINATION_REASON_CRASH,
    resolved_bindings: dict[str, str] | None = None,
    proposals_root: Path | None = None,
    state_mode: StateMode = "none",
) -> None:
    """Record a failure that happened upstream of the commit phase.

    Used by :func:`run_block` for failures that occur after
    ``execution_id`` is minted but before the commit phase runs:
    proposal-allocation failures, input-resolution failures, and
    runtime exceptions that bubble out.

    The ``status`` is derived from ``termination_reason`` via the
    canonical mapping; callers do not pass it.  ``output_bindings``
    is forced empty (no slot reached the forge).  ``rejected_outputs``
    is empty (the substrate's own pre-commit failures are not
    per-slot rejections).

    If ``proposals_root`` is supplied and exists on disk, it is
    removed.

    Args:
        workspace: Workspace to record into.
        execution_id: Pre-minted execution ID.
        block_name: Block that was being executed.
        started_at: When the execution attempt began.
        image: Container image declared by the block.
        phase: Failure phase from :data:`flywheel.runtime.FAILURE_PHASES`.
        error: Human-readable error message.
        termination_reason: Substrate-reserved reason describing
            the failure.  Defaults to
            :data:`flywheel.runtime.TERMINATION_REASON_CRASH`.
        resolved_bindings: Any input bindings that were already
            resolved before the failure.  ``None`` records an
            empty mapping.
        proposals_root: Per-execution proposals tree to clean up,
            if one was allocated before the failure.
    """
    if proposals_root is not None:
        shutil.rmtree(proposals_root, ignore_errors=True)
    _record_execution(
        workspace, execution_id, block_name, started_at,
        termination_reason,
        resolved_bindings or {}, {},
        None, None, image,
        all_expected_committed=True,
        failure_phase=phase,
        error=error,
        state_mode=state_mode,
    )


# ── Phase 1: prepare ─────────────────────────────────────────────


def prepare_block_execution(
    workspace: Workspace,
    block_def: BlockDefinition,
    template: Template,
    project_root: Path,
    input_bindings: dict[str, str],
    *,
    execution_id: str,
    started_at: datetime,
    state_lineage_key: str | None = None,
) -> ExecutionPlan:
    """Allocate proposal dirs, resolve inputs, build mounts.

    Spec ordering: proposal allocation happens before input
    resolution so the per-execution proposals tree exists for
    cleanup before any input-resolution failure.  Identity
    (``execution_id``, ``started_at``) is minted by the caller so
    the caller can record a failed :class:`BlockExecution` via
    :func:`commit_failure` if anything in this function raises.

    Input binding resolution order:

    1. Explicit binding wins.
    2. Git-kind slots are re-resolved against the working tree.
    3. Otherwise the most recent ``copy`` instance is used.
    4. If no instance is found and the slot is optional, skip it.
    5. Required-with-no-instance raises ``ValueError``.

    Returns:
        An :class:`ExecutionPlan` with the resolved inputs,
        mounts, and proposal directories.

    Raises:
        ValueError: If a required input cannot be resolved or a
            binding is incompatible with its slot.  Caller is
            responsible for routing this through
            :func:`commit_failure`.
        OSError: If proposal-directory allocation fails.  Same
            failure-recording responsibility applies.
    """
    mounts: list[tuple[str, str, str]] = []
    resolved_bindings: dict[str, str] = {}

    # Per-slot proposal dirs (proposal-then-forge).  Allocated
    # first so any subsequent input-resolution failure has a known
    # proposals tree to clean up via commit_failure.
    proposals_root = workspace.path / "proposals" / execution_id
    proposal_dirs: dict[str, Path] = {}
    for slot in block_def.all_output_slots():
        proposal_dir = proposals_root / slot.name
        proposal_dir.mkdir(parents=True, exist_ok=True)
        proposal_dirs[slot.name] = proposal_dir
        mounts.append(
            (str(proposal_dir.resolve()),
             slot.container_path, "rw")
        )

    # Termination-channel sidecar.  The block writes one line to
    # /flywheel/termination announcing why it ended.
    termination_dir = proposals_root / "_flywheel"
    termination_dir.mkdir(parents=True, exist_ok=True)
    termination_file = termination_dir / "termination"
    mounts.append(
        (str(termination_dir.resolve()), "/flywheel", "rw")
    )

    state_mode = block_def.state
    state_mount_dir: Path | None = None
    restored_state_snapshot_id: str | None = None
    state_compatibility: dict[str, str] | None = None
    if state_mode == "managed":
        if state_lineage_key is None or not state_lineage_key.strip():
            raise ValueError(
                f"Block {block_def.name!r} declares managed state; "
                "a state lineage key is required"
            )
        state_lineage_key = state_lineage_key.strip()
        state_compatibility = state_compatibility_identity(block_def)
        latest_snapshot = workspace.latest_state_snapshot(
            state_lineage_key)
        state_mount_dir = termination_dir / "state"
        state_mount_dir.mkdir(parents=True, exist_ok=True)
        if latest_snapshot is not None:
            if latest_snapshot.compatibility != state_compatibility:
                raise ValueError(
                    f"Latest state snapshot {latest_snapshot.id!r} "
                    f"for lineage {state_lineage_key!r} is not "
                    f"compatible with block {block_def.name!r}"
                )
            _copy_dir_contents(
                workspace.state_snapshot_path(latest_snapshot.id),
                state_mount_dir,
            )
            restored_state_snapshot_id = latest_snapshot.id

    for slot in block_def.inputs:
        decl = _find_artifact_declaration(template, slot.name)

        if slot.name in input_bindings:
            artifact_id = input_bindings[slot.name]
            if artifact_id not in workspace.artifacts:
                raise ValueError(
                    f"Binding {artifact_id!r} for input "
                    f"{slot.name!r} not found in workspace"
                )
            instance = workspace.artifacts[artifact_id]
            if instance.name != slot.name:
                raise ValueError(
                    f"Artifact {artifact_id!r} belongs to "
                    f"{instance.name!r}, not {slot.name!r}"
                )
            resolved_bindings[slot.name] = artifact_id
            mounts.append(
                _mount_artifact_instance(
                    instance, slot.container_path, workspace)
            )

        elif decl is not None and decl.kind == "git":
            git_instance = workspace.register_git_artifact(
                slot.name, decl, project_root,
            )
            resolved_bindings[slot.name] = git_instance.id
            mounts.append(
                _mount_artifact_instance(
                    git_instance, slot.container_path, workspace)
            )

        else:
            instances = workspace.instances_for(slot.name)
            copy_instances = [
                i for i in instances if i.kind == "copy"]
            if copy_instances:
                instance = copy_instances[-1]
                resolved_bindings[slot.name] = instance.id
                mounts.append(
                    _mount_artifact_instance(
                        instance, slot.container_path, workspace)
                )
            elif slot.optional:
                continue
            else:
                raise ValueError(
                    f"Required input artifact {slot.name!r} for "
                    f"block {block_def.name!r} is not available"
                )

    return ExecutionPlan(
        execution_id=execution_id,
        block_def=block_def,
        block_name=block_def.name,
        started_at=started_at,
        resolved_bindings=resolved_bindings,
        mounts=mounts,
        proposal_dirs=proposal_dirs,
        proposals_root=proposals_root,
        termination_file=termination_file,
        state_mode=state_mode,
        state_lineage_key=state_lineage_key,
        state_mount_dir=state_mount_dir,
        restored_state_snapshot_id=restored_state_snapshot_id,
        state_compatibility=state_compatibility,
    )


# ── Phase 2: run ──────────────────────────────────────────────


def _container_config_for_plan(plan: ExecutionPlan) -> ContainerConfig:
    """Build the runtime container config from a prepared plan."""
    return ContainerConfig(
        image=plan.block_def.image,
        docker_args=plan.block_def.docker_args,
        env=plan.block_def.env,
        mounts=plan.mounts,
    )


def observe_one_shot_container_exit(
    plan: ExecutionPlan,
    result: ContainerResult,
) -> RuntimeResult:
    """Translate an exited one-shot container into runtime facts.

    Non-zero exit is a runtime crash.  Zero exit must announce a
    project-defined termination reason through the sidecar mounted
    at ``/flywheel/termination``; malformed or undeclared reasons
    are normalized to ``protocol_violation``.
    """
    if result.exit_code != 0:
        return RuntimeResult(
            termination_reason=runtime.TERMINATION_REASON_CRASH,
            container_result=result,
        )

    announcement = read_termination_sidecar(plan.termination_file)
    termination_reason = normalize_termination_reason(
        announcement=announcement,
        declared_reasons=set(plan.block_def.outputs.keys()),
    )
    return RuntimeResult(
        termination_reason=termination_reason,
        container_result=result,
        announcement=announcement,
    )


@dataclass(frozen=True)
class DockerOneShotContainerRunner:
    """Default runner for one-shot Docker container bodies."""

    def run(
        self, plan: ExecutionPlan, args: list[str] | None = None,
    ) -> RuntimeResult:
        """Run the prepared plan in a fresh container."""
        result = run_container(_container_config_for_plan(plan), args)
        return observe_one_shot_container_exit(plan, result)


def run_one_shot_container(
    plan: ExecutionPlan,
    args: list[str] | None = None,
    *,
    container_runner: OneShotContainerRunner | None = None,
) -> RuntimeResult:
    """Run the block body once through a one-shot container runner.

    Returns:
        An :class:`RuntimeResult` with the normalized
        termination reason and the raw container result.

    Raises:
        KeyboardInterrupt: Propagated from the container runtime
            on operator stop.  Recording the interrupted
            execution is the caller's responsibility (see
            :func:`run_block`).
    """
    runner = container_runner or DockerOneShotContainerRunner()
    return runner.run(plan, args)


# ── Phase 3: commit ──────────────────────────────────────────────


_RESERVED_FAILURE_PHASES: dict[str, str] = {
    runtime.TERMINATION_REASON_CRASH: runtime.FAILURE_INVOKE,
    runtime.TERMINATION_REASON_TIMEOUT: runtime.FAILURE_INVOKE,
    runtime.TERMINATION_REASON_INTERRUPTED: runtime.FAILURE_INVOKE,
    runtime.TERMINATION_REASON_PROTOCOL_VIOLATION: (
        runtime.FAILURE_OUTPUT_PROTOCOL
    ),
}


def _runtime_failure_error(
    runtime_result: RuntimeResult,
    block_def: BlockDefinition,
    exit_code: int | None,
) -> str:
    """Return canonical diagnostic text for reserved runtime failures.

    Runner-supplied non-empty text wins.  Otherwise commit owns the
    fallback so the ledger error and raised exception text cannot drift.
    """
    if runtime_result.error is not None and runtime_result.error.strip():
        return runtime_result.error.strip()

    reason = runtime_result.termination_reason
    if reason == runtime.TERMINATION_REASON_CRASH:
        if exit_code is None:
            return "container crashed without exit metadata"
        return f"container exited with code {exit_code}"
    if reason == runtime.TERMINATION_REASON_TIMEOUT:
        return "container exceeded its execution deadline"
    if reason == runtime.TERMINATION_REASON_INTERRUPTED:
        return "container was interrupted"
    if reason == runtime.TERMINATION_REASON_PROTOCOL_VIOLATION:
        declared = sorted(block_def.outputs.keys())
        return (
            f"protocol violation: termination announcement "
            f"{runtime_result.announcement!r} did not match any "
            f"declared reason {declared!r}"
        )
    raise ValueError(
        f"_runtime_failure_error called for non-reserved failure "
        f"reason {reason!r}"
    )


def commit_block_execution(
    workspace: Workspace,
    plan: ExecutionPlan,
    runtime_result: RuntimeResult,
    template: Template,
    *,
    validator_registry: ArtifactValidatorRegistry | None = None,
) -> ContainerResult | None:
    """Forge proposals, record the execution, and clean up.

    Implements the canonical commit pipeline:

    1. If the runtime announced a substrate-reserved
       failure-shaped reason (``crash``, ``interrupted``,
       ``protocol_violation``, ...), record the execution as
       ``failed`` with the appropriate ``failure_phase`` and
       raise.
    2. For every output slot the announced termination reason
       maps to: empty proposal → ``output_collect`` rejection;
       validator failure → quarantined ``output_validate``
       rejection; other forge failure → quarantined
       ``artifact_commit`` rejection; success → register through
       ``Workspace.register_artifact``.
    3. Record the :class:`BlockExecution` via
       ``Workspace.record_execution``.

    Returns:
        The raw ``ContainerResult`` from the runtime result on the
        happy path.  ``None`` is never returned for happy-path
        one-shot runs; the field exists so the persistent-runtime
        adapter can return ``None`` for handle metadata it doesn't
        own.

    Raises:
        RuntimeError: For substrate-reserved failure reasons and
            for non-validation commit failures.
        flywheel.artifact_validator.ArtifactValidationError: When
            ``output_validate`` is the most-downstream rejection
            phase.
    """
    block_name = plan.block_name
    block_def = plan.block_def
    image = block_def.image
    container_result = runtime_result.container_result
    exit_code = container_result.exit_code if container_result else None
    elapsed_s = container_result.elapsed_s if container_result else None
    termination_reason = runtime_result.termination_reason

    # -- Substrate-reserved failure-shaped reasons -----------------
    if termination_reason in _RESERVED_FAILURE_PHASES:
        failure_phase = _RESERVED_FAILURE_PHASES[termination_reason]
        error = _runtime_failure_error(runtime_result, block_def, exit_code)
        shutil.rmtree(plan.proposals_root, ignore_errors=True)
        _record_execution(
            workspace, plan.execution_id, block_name, plan.started_at,
            termination_reason,
            plan.resolved_bindings, {},
            exit_code, elapsed_s, image,
            all_expected_committed=True,
            failure_phase=failure_phase,
            error=error,
            state_mode=plan.state_mode,
        )
        message = f"Block {block_name!r} runtime failed: {error}"
        if termination_reason == runtime.TERMINATION_REASON_INTERRUPTED:
            raise KeyboardInterrupt(message)
        raise RuntimeError(message)

    # ── Forge proposals into artifact instances ──────────────────
    state_snapshot_id: str | None = None
    if plan.state_mode == "managed":
        assert plan.state_lineage_key is not None
        assert plan.state_mount_dir is not None
        assert plan.state_compatibility is not None
        try:
            snapshot = workspace.register_state_snapshot(
                lineage_key=plan.state_lineage_key,
                source_path=plan.state_mount_dir,
                produced_by=plan.execution_id,
                compatibility=plan.state_compatibility,
                predecessor_snapshot_id=plan.restored_state_snapshot_id,
            )
            state_snapshot_id = snapshot.id
        except Exception as exc:
            recovery_path = workspace.preserve_state_recovery(
                execution_id=plan.execution_id,
                source_path=plan.state_mount_dir,
            )
            shutil.rmtree(plan.proposals_root, ignore_errors=True)
            error = f"state_capture: {type(exc).__name__}: {exc}"
            if recovery_path is not None:
                error = f"{error} (recovery: {recovery_path})"
            _record_execution(
                workspace, plan.execution_id, block_name, plan.started_at,
                termination_reason,
                plan.resolved_bindings, {},
                exit_code, elapsed_s, image,
                all_expected_committed=False,
                failure_phase=runtime.FAILURE_STATE_CAPTURE,
                error=error,
                state_mode=plan.state_mode,
            )
            raise RuntimeError(error) from exc

    declarations = {a.name: a for a in template.artifacts}
    output_bindings: dict[str, str] = {}
    rejected_outputs: dict[str, RejectedOutput] = {}
    rejection_messages: list[str] = []
    expected_slots = block_def.outputs_for(termination_reason)

    for slot in expected_slots:
        proposal_dir = plan.proposal_dirs[slot.name]
        if not proposal_dir.exists() or not any(
                proposal_dir.iterdir()):
            rejection_messages.append(
                f"{slot.name}: no output bytes written"
            )
            rejected_outputs[slot.name] = RejectedOutput(
                reason="no output bytes written",
                quarantine_path=None,
                phase=runtime.FAILURE_OUTPUT_COLLECT,
            )
            continue
        try:
            instance = workspace.register_artifact(
                slot.name,
                source_path=proposal_dir,
                source=None,
                validator_registry=validator_registry,
                declaration=declarations.get(slot.name),
                produced_by=plan.execution_id,
            )
        except ArtifactValidationError as exc:
            reason = str(exc)
            rejection_messages.append(f"{slot.name}: {reason}")
            qpath = quarantine_slot(
                workspace.path, plan.execution_id, slot.name,
                proposal_dir,
            )
            rejected_outputs[slot.name] = RejectedOutput(
                reason=reason, quarantine_path=qpath,
                phase=runtime.FAILURE_OUTPUT_VALIDATE,
            )
            continue
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            rejection_messages.append(f"{slot.name}: {reason}")
            qpath = quarantine_slot(
                workspace.path, plan.execution_id, slot.name,
                proposal_dir,
            )
            rejected_outputs[slot.name] = RejectedOutput(
                reason=reason, quarantine_path=qpath,
                phase=runtime.FAILURE_ARTIFACT_COMMIT,
            )
            continue
        output_bindings[slot.name] = instance.id

    shutil.rmtree(plan.proposals_root, ignore_errors=True)

    all_expected_committed = not rejected_outputs

    if rejection_messages:
        # Most-downstream phase across all rejected slots is the
        # execution-level failure_phase.
        phase_order = {
            runtime.FAILURE_OUTPUT_COLLECT: 0,
            runtime.FAILURE_OUTPUT_VALIDATE: 1,
            runtime.FAILURE_ARTIFACT_COMMIT: 2,
        }
        execution_phase = max(
            (rej.phase for rej in rejected_outputs.values()),
            key=lambda p: phase_order.get(p, 0),
        )
        error_msg = (
            f"{execution_phase}: "
            f"{'; '.join(rejection_messages)}"
        )
        _record_execution(
            workspace, plan.execution_id, block_name, plan.started_at,
            termination_reason,
            plan.resolved_bindings, output_bindings,
            exit_code, elapsed_s, image,
            all_expected_committed=all_expected_committed,
            failure_phase=execution_phase,
            error=error_msg,
            rejected_outputs=rejected_outputs,
            state_mode=plan.state_mode,
            state_snapshot_id=state_snapshot_id,
        )
        if execution_phase == runtime.FAILURE_OUTPUT_VALIDATE:
            raise ArtifactValidationError(error_msg)
        raise RuntimeError(error_msg)

    _record_execution(
        workspace, plan.execution_id, block_name, plan.started_at,
        termination_reason,
        plan.resolved_bindings, output_bindings,
        exit_code, elapsed_s, image,
        all_expected_committed=all_expected_committed,
        state_mode=plan.state_mode,
        state_snapshot_id=state_snapshot_id,
    )

    return container_result


# ── Top-level entry point ────────────────────────────────────────


def run_block(
    workspace: Workspace,
    block_name: str,
    template: Template,
    project_root: Path,
    input_bindings: dict[str, str] | None = None,
    args: list[str] | None = None,
    *,
    validator_registry: ArtifactValidatorRegistry | None = None,
    container_runner: OneShotContainerRunner | None = None,
    state_lineage_key: str | None = None,
) -> ContainerResult:
    """Execute a block within a workspace.

    Thin orchestration over the canonical
    ``prepare -> run -> commit`` model.  See
    :func:`prepare_block_execution`,
    :func:`run_one_shot_container`, and
    :func:`commit_block_execution` for the per-phase contracts.

    Args:
        workspace: The workspace to execute the block in.
        block_name: Name of the block to execute.
        template: The template defining blocks and artifacts.
        project_root: The project root for resolving relative paths.
        input_bindings: Optional mapping of input slot names to
            specific artifact instance IDs.
        args: Optional extra arguments for the container entrypoint.
        validator_registry: Project-supplied artifact validator
            registry consulted before each output slot is committed.
        container_runner: Optional runner for the prepared one-shot
            container body.  Defaults to the Docker-backed runner.
        state_lineage_key: Required for blocks that declare
            ``state: managed``; identifies the execution-lineage
            state chain to restore from and capture into.

    Returns:
        A ContainerResult with exit code and wall-clock elapsed
        seconds.

    Raises:
        KeyError: If block_name is not found in the template.
        ValueError: If template/workspace mismatch or an input
            cannot be resolved.
        RuntimeError: For container crashes, protocol violations,
            or non-validation commit failures.
        flywheel.artifact_validator.ArtifactValidationError: If
            output validation rejects one or more slots.
    """
    if input_bindings is None:
        input_bindings = {}

    if template.name != workspace.template_name:
        raise ValueError(
            f"Template {template.name!r} does not match workspace "
            f"template {workspace.template_name!r}"
        )

    block_def = None
    for block in template.blocks:
        if block.name == block_name:
            block_def = block
            break
    if block_def is None:
        raise KeyError(
            f"Block {block_name!r} not found in template "
            f"{template.name!r}"
        )
    if block_def.runner != "container":
        raise ValueError(
            f"flywheel run block uses the one-shot container "
            f"pipeline and requires runner 'container'; block "
            f"{block_name!r} has runner {block_def.runner!r}"
        )
    if block_def.lifecycle != "one_shot":
        raise ValueError(
            f"flywheel run block uses the one-shot container "
            f"pipeline and requires lifecycle 'one_shot'; block "
            f"{block_name!r} has lifecycle {block_def.lifecycle!r}"
        )

    # ── identity ─────────────────────────────────────────────────
    # Minted before prepare so any prepare-time failure has an
    # execution_id to record itself against.
    execution_id = workspace.generate_execution_id()
    started_at = datetime.now(UTC)
    proposals_root = workspace.path / "proposals" / execution_id

    # ── prepare ──────────────────────────────────────────────────
    try:
        plan = prepare_block_execution(
            workspace, block_def, template, project_root,
            input_bindings,
            execution_id=execution_id,
            started_at=started_at,
            state_lineage_key=state_lineage_key,
        )
    except Exception as exc:
        commit_failure(
            workspace,
            execution_id=execution_id,
            block_name=block_name,
            started_at=started_at,
            image=block_def.image,
            phase=runtime.FAILURE_STAGE_IN,
            error=f"{type(exc).__name__}: {exc}",
            proposals_root=proposals_root,
            state_mode=block_def.state,
        )
        raise

    # -- run -----------------------------------------------------
    try:
        runtime_result = run_one_shot_container(
            plan, args, container_runner=container_runner,
        )
    except KeyboardInterrupt as exc:
        error = str(exc) or "operator interrupt"
        commit_failure(
            workspace,
            execution_id=plan.execution_id,
            block_name=plan.block_name,
            started_at=plan.started_at,
            image=block_def.image,
            phase=runtime.FAILURE_INVOKE,
            error=error,
            termination_reason=runtime.TERMINATION_REASON_INTERRUPTED,
            resolved_bindings=plan.resolved_bindings,
            proposals_root=plan.proposals_root,
            state_mode=plan.state_mode,
        )
        raise
    except Exception as exc:
        commit_failure(
            workspace,
            execution_id=plan.execution_id,
            block_name=plan.block_name,
            started_at=plan.started_at,
            image=block_def.image,
            phase=runtime.FAILURE_INVOKE,
            error=f"{type(exc).__name__}: {exc}",
            resolved_bindings=plan.resolved_bindings,
            proposals_root=plan.proposals_root,
            state_mode=plan.state_mode,
        )
        raise

    # ── commit ───────────────────────────────────────────────────
    result = commit_block_execution(
        workspace, plan, runtime_result, template,
        validator_registry=validator_registry,
    )
    # commit_block_execution returns the container_result on the
    # happy path; for completeness, fall back to the runtime_result's
    # raw container result if we somehow get None.
    return result or runtime_result.container_result  # type: ignore[return-value]
