"""Block execution orchestration for flywheel.

Ties together workspace state, container launching, and
convention-based output recording. The run_block function is the
main entry point for executing a single block within a workspace.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

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
from flywheel.template import ArtifactDeclaration, Template
from flywheel.workspace import Workspace


def _find_artifact_declaration(
    template: Template, name: str
) -> ArtifactDeclaration | None:
    """Find an artifact declaration by name in a template.

    Args:
        template: The template to search.
        name: The artifact name to find.

    Returns:
        The matching ArtifactDeclaration, or None if not found.
    """
    for decl in template.artifacts:
        if decl.name == name:
            return decl
    return None


def _resolve_git_input(
    name: str, decl: ArtifactDeclaration, project_root: Path,
    workspace: Workspace,
) -> ArtifactInstance:
    """Re-resolve a git artifact to the latest committed state.

    Checks that the repo working tree is clean, verifies the declared
    path exists at the current commit, and creates a new ArtifactInstance
    pinned to HEAD.

    Args:
        name: The artifact declaration name.
        decl: The artifact declaration from the template.
        project_root: The project root for resolving relative repo paths.
        workspace: The workspace, used to generate the next artifact ID.

    Returns:
        A new git ArtifactInstance pinned to HEAD.

    Raises:
        RuntimeError: If the git repo has a dirty working tree.
        ValueError: If the declaration is missing repo or path fields.
        FileNotFoundError: If the declared path does not exist at HEAD.
    """
    if decl.repo is None or decl.path is None:
        raise ValueError(
            f"Git artifact {name!r} missing repo or path in declaration"
        )

    repo_path = (project_root / decl.repo).resolve()

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout.strip():
        raise RuntimeError(
            f"Git repo {repo_path} has uncommitted changes. "
            f"Commit or stash before running a block."
        )

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = result.stdout.strip()

    artifact_path = repo_path / decl.path
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Git artifact {name!r} path {decl.path!r} "
            f"does not exist in repo {repo_path}"
        )

    artifact_id = workspace.generate_artifact_id(name)
    return ArtifactInstance(
        id=artifact_id,
        name=name,
        kind="git",
        created_at=datetime.now(UTC),
        produced_by=None,
        repo=str(repo_path),
        commit=commit,
        git_path=decl.path,
    )


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


def _cleanup_output_dirs(
    workspace: Workspace, output_artifact_ids: dict[str, str],
) -> None:
    """Remove output directories that were allocated but not finalized.

    Args:
        workspace: The workspace containing the directories.
        output_artifact_ids: Mapping of slot names to artifact IDs whose
            directories should be cleaned up.
    """
    for artifact_id in output_artifact_ids.values():
        output_dir = workspace.path / "artifacts" / artifact_id
        if output_dir.exists():
            shutil.rmtree(output_dir)


def _record_execution(
    workspace: Workspace, execution_id: str, block_name: str,
    started_at: datetime, status: str,
    input_bindings: dict[str, str],
    output_bindings: dict[str, str],
    exit_code: int | None, elapsed_s: float | None,
    image: str,
    failure_phase: str | None = None,
    error: str | None = None,
    rejected_outputs: dict[str, RejectedOutput] | None = None,
) -> None:
    """Record a block execution and persist the workspace.

    Args:
        workspace: The workspace to record in.
        execution_id: The execution's unique ID.
        block_name: The block that was executed.
        started_at: When execution began.
        status: The outcome (succeeded, failed, or interrupted).
        input_bindings: Which artifact instances were consumed.
        output_bindings: Which artifact instances were produced.
        exit_code: The container's exit code, if available.
        elapsed_s: Wall-clock time in seconds, if available.
        image: The Docker image that was used.
        failure_phase: Which step of the pipeline failed.  Set
            only when status is ``"failed"``.  See
            :mod:`flywheel.runtime` for the canonical constants.
        error: Human-readable error message recorded alongside
            ``failure_phase`` for failed executions.
        rejected_outputs: Per-slot rejection records for an
            ``output_validate`` failure, keyed by slot name.
            ``None`` (the default) is recorded as an empty
            mapping.
    """
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
        failure_phase=failure_phase,
        error=error,
        rejected_outputs=rejected_outputs or {},
    )
    workspace.add_execution(execution)
    workspace.save()


def run_block(
    workspace: Workspace,
    block_name: str,
    template: Template,
    project_root: Path,
    input_bindings: dict[str, str] | None = None,
    args: list[str] | None = None,
    *,
    validator_registry: ArtifactValidatorRegistry | None = None,
) -> ContainerResult:
    """Execute a block within a workspace.

    Resolves input artifact instances, launches a Docker container,
    and records produced artifact instances and the execution record
    using convention-based output directories.

    Input binding resolution follows this order for each input slot:
    1. If an explicit binding is provided, use it.
    2. If the slot is a git artifact, re-resolve to the latest commit.
    3. Otherwise, use the most recent copy instance for the slot.
       This implicit "latest wins" policy is a provisional default.
    4. If no instance exists and the slot is optional, skip it.
    5. If no instance exists and the slot is required, raise an error.

    Args:
        workspace: The workspace to execute the block in.
        block_name: Name of the block to execute.
        template: The template defining blocks and artifacts.
        project_root: The project root for resolving relative paths.
        input_bindings: Optional mapping of input slot names to specific
            artifact instance IDs. Each bound artifact must belong to
            the same declaration as the slot it is bound to.
        args: Optional extra arguments for the container entrypoint.
        validator_registry: Project-supplied artifact validator
            registry consulted before each output slot is committed.
            See :mod:`flywheel.artifact_validator` for semantics.
            Optional: when omitted, outputs are committed without
            validation.

    Returns:
        A ContainerResult with exit code and wall-clock elapsed seconds.

    Raises:
        KeyError: If block_name is not found in the template.
        ValueError: If the template does not match the workspace, a
            required input artifact is not available, or a binding
            references an artifact that does not match the slot.
        RuntimeError: If the container exits with non-zero code or
            a git repo has uncommitted changes.
        FileNotFoundError: If a git artifact path does not exist.
        flywheel.artifact_validator.ArtifactValidationError: If a
            project-declared validator rejects one or more output
            slots.  The execution is recorded ``failed`` with
            ``failure_phase=output_validate`` and the slots that
            passed validation are still committed (commit-A).
    """
    if input_bindings is None:
        input_bindings = {}

    # 1. Validate template matches workspace
    if template.name != workspace.template_name:
        raise ValueError(
            f"Template {template.name!r} does not match workspace "
            f"template {workspace.template_name!r}"
        )

    # 2. Look up block definition
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

    # 3. Create execution ID
    execution_id = workspace.generate_execution_id()
    started_at = datetime.now(UTC)

    # 4. Resolve inputs and build mounts.
    #    Explicit bindings take precedence over all other resolution,
    #    including git re-resolution.
    mounts: list[tuple[str, str, str]] = []
    resolved_bindings: dict[str, str] = {}

    for slot in block_def.inputs:
        decl = _find_artifact_declaration(template, slot.name)

        if slot.name in input_bindings:
            # Explicit binding — validate and use the specified instance
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
                _mount_artifact_instance(instance, slot.container_path,
                                         workspace)
            )

        elif decl is not None and decl.kind == "git":
            # Re-resolve git artifact to latest committed state.
            # Commit immediately — git instances are inputs that
            # exist regardless of whether the execution succeeds.
            git_instance = _resolve_git_input(
                slot.name, decl, project_root, workspace,
            )
            workspace.add_artifact(git_instance)
            resolved_bindings[slot.name] = git_instance.id
            mounts.append(
                _mount_artifact_instance(git_instance, slot.container_path,
                                         workspace)
            )

        else:
            # Implicit binding: use the most recent copy instance.
            # This is a provisional "latest wins" default.
            instances = workspace.instances_for(slot.name)
            copy_instances = [i for i in instances if i.kind == "copy"]
            if copy_instances:
                instance = copy_instances[-1]
                resolved_bindings[slot.name] = instance.id
                mounts.append(
                    _mount_artifact_instance(instance, slot.container_path,
                                             workspace)
                )
            elif slot.optional:
                continue
            else:
                raise ValueError(
                    f"Required input artifact {slot.name!r} for block "
                    f"{block_name!r} is not available"
                )

    # 5. Allocate fresh output directories with new artifact IDs
    output_artifact_ids: dict[str, str] = {}
    for slot in block_def.outputs:
        artifact_id = workspace.generate_artifact_id(slot.name)
        output_artifact_ids[slot.name] = artifact_id

        output_dir = workspace.path / "artifacts" / artifact_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        mounts.append((str(output_dir.resolve()), slot.container_path, "rw"))

    # 6. Build ContainerConfig and run.
    #    Wrap in try/except to handle both non-zero exit and
    #    KeyboardInterrupt (Ctrl+C).
    config = ContainerConfig(
        image=block_def.image,
        docker_args=block_def.docker_args,
        env=block_def.env,
        mounts=mounts,
    )

    try:
        result = run_container(config, args)
    except KeyboardInterrupt:
        _cleanup_output_dirs(workspace, output_artifact_ids)
        _record_execution(
            workspace, execution_id, block_name, started_at,
            "interrupted", resolved_bindings, {}, None, None,
            block_def.image,
        )
        raise

    finished_at = datetime.now(UTC)

    if result.exit_code != 0:
        _cleanup_output_dirs(workspace, output_artifact_ids)
        _record_execution(
            workspace, execution_id, block_name, started_at,
            "failed", resolved_bindings, {},
            result.exit_code, result.elapsed_s, block_def.image,
        )
        raise RuntimeError(
            f"Block {block_name!r} container exited with code "
            f"{result.exit_code}"
        )

    # 7. Record output artifact instances (convention-based).
    #    Validate per-slot before commit.  Commit-passing-slots:
    #    slots that pass land in ``output_bindings`` and the
    #    execution is recorded ``succeeded``; slots that fail
    #    are not committed, their bytes are quarantined under
    #    ``<workspace>/quarantine/<exec_id>/<slot>/``, and the
    #    execution is recorded ``failed`` with
    #    ``failure_phase=output_validate``.
    declarations = {a.name: a for a in template.artifacts}
    output_bindings: dict[str, str] = {}
    rejected_outputs: dict[str, RejectedOutput] = {}
    rejection_messages: list[str] = []
    for slot in block_def.outputs:
        artifact_id = output_artifact_ids[slot.name]
        output_dir = workspace.path / "artifacts" / artifact_id
        if not any(output_dir.iterdir()):
            continue
        if validator_registry is not None:
            try:
                validator_registry.validate(
                    slot.name,
                    declarations.get(slot.name),
                    output_dir,
                )
            except ArtifactValidationError as exc:
                reason = str(exc)
                rejection_messages.append(
                    f"{slot.name}: {reason}",
                )
                # Preserve the rejected bytes before tearing
                # down the pre-allocated artifact dir; the
                # validation failure is the primary signal
                # either way.
                qpath = quarantine_slot(
                    workspace.path, execution_id, slot.name,
                    output_dir,
                )
                rejected_outputs[slot.name] = RejectedOutput(
                    reason=reason, quarantine_path=qpath,
                )
                # Reject this slot's pre-allocated dir so we
                # don't leave half-committed artifact storage
                # behind.
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                continue
        instance = ArtifactInstance(
            id=artifact_id,
            name=slot.name,
            kind="copy",
            created_at=finished_at,
            produced_by=execution_id,
            copy_path=artifact_id,
        )
        workspace.add_artifact(instance)
        output_bindings[slot.name] = artifact_id

    # Clean up any pre-allocated dirs that never received
    # content (matches the prior behavior).
    for slot in block_def.outputs:
        artifact_id = output_artifact_ids[slot.name]
        if artifact_id in output_bindings:
            continue
        leftover = workspace.path / "artifacts" / artifact_id
        if leftover.exists() and not any(leftover.iterdir()):
            shutil.rmtree(leftover)

    # 8. Record execution and save.
    if rejection_messages:
        error_msg = (
            f"output_validate: {'; '.join(rejection_messages)}"
        )
        _record_execution(
            workspace, execution_id, block_name, started_at,
            "failed", resolved_bindings, output_bindings,
            result.exit_code, result.elapsed_s, block_def.image,
            failure_phase=runtime.FAILURE_OUTPUT_VALIDATE,
            error=error_msg,
            rejected_outputs=rejected_outputs,
        )
        raise ArtifactValidationError(error_msg)

    _record_execution(
        workspace, execution_id, block_name, started_at,
        "succeeded", resolved_bindings, output_bindings,
        result.exit_code, result.elapsed_s, block_def.image,
    )

    return result
