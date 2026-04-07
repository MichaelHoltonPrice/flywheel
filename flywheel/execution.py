"""Block execution orchestration for flywheel.

Ties together workspace state, container launching, and
convention-based output recording. The run_block function is the
main entry point for executing a single block within a workspace.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from flywheel.artifact import CopyArtifact, GitArtifact, GitRef
from flywheel.container import ContainerConfig, ContainerResult, run_container
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


def _resolve_git_artifact(
    name: str, decl: ArtifactDeclaration, project_root: Path
) -> GitArtifact:
    """Re-resolve a git artifact to the latest committed state.

    Checks that the repo working tree is clean, verifies the declared
    path exists at the current commit, and returns a GitArtifact pinned
    to HEAD.

    Args:
        name: The artifact slot name.
        decl: The artifact declaration from the template.
        project_root: The project root for resolving relative repo paths.

    Returns:
        A GitArtifact pinned to the current HEAD.

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

    return GitArtifact(
        name=name,
        ref=GitRef(repo=str(repo_path), commit=commit, path=decl.path),
    )


def run_block(
    workspace: Workspace,
    block_name: str,
    template: Template,
    project_root: Path,
    args: list[str] | None = None,
) -> ContainerResult:
    """Execute a block within a workspace.

    Resolves input artifacts, launches a Docker container, and records
    produced artifacts using convention-based output directories.

    Args:
        workspace: The workspace to execute the block in.
        block_name: Name of the block to execute.
        template: The template defining blocks and artifacts.
        project_root: The project root for resolving relative paths.
        args: Optional extra arguments for the container entrypoint.

    Returns:
        A ContainerResult with exit code and wall-clock elapsed seconds.

    Raises:
        KeyError: If block_name not found in template.
        ValueError: If a required input artifact is not available,
            an output artifact is already recorded, or the template
            does not match the workspace.
        RuntimeError: If the container exits with non-zero code or
            a git repo has uncommitted changes.
        FileNotFoundError: If a git artifact path does not exist.
    """
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
            f"Block {block_name!r} not found in template {template.name!r}"
        )

    # 3. Fail fast if any output artifact is already recorded
    for slot in block_def.outputs:
        existing = workspace.artifacts.get(slot.name)
        if existing is not None:
            raise ValueError(
                f"Output artifact {slot.name!r} for block {block_name!r} "
                f"is already recorded. Use a new workspace to re-run."
            )

    # 4. Resolve inputs and build mounts
    mounts: list[tuple[str, str, str]] = []

    for slot in block_def.inputs:
        decl = _find_artifact_declaration(template, slot.name)
        artifact = workspace.artifacts.get(slot.name)

        if decl is not None and decl.kind == "git":
            # Re-resolve to latest committed state for mounting.
            # The workspace retains the baseline snapshot from creation;
            # execution-time resolution is used only for the mount.
            git_artifact = _resolve_git_artifact(
                slot.name, decl, project_root
            )
            host_path = str(
                Path(git_artifact.ref.repo) / git_artifact.ref.path
            )
            mounts.append((host_path, slot.container_path, "ro"))
        elif artifact is not None and isinstance(artifact, CopyArtifact):
            host_path = str(
                workspace.path / "artifacts" / artifact.name
            )
            if artifact.path != Path("."):
                host_path = str(
                    workspace.path / "artifacts" / artifact.name
                    / artifact.path
                )
            mounts.append((host_path, slot.container_path, "ro"))
        elif slot.optional:
            continue
        else:
            raise ValueError(
                f"Required input artifact {slot.name!r} for block "
                f"{block_name!r} is not available"
            )

    # 5. Create clean output directories and mount them
    for slot in block_def.outputs:
        output_dir = workspace.path / "artifacts" / slot.name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        mounts.append((str(output_dir), slot.container_path, "rw"))

    # 6. Build ContainerConfig with resource settings and run
    config = ContainerConfig(
        image=block_def.image,
        gpus=block_def.gpus,
        shm_size=block_def.shm_size,
        env=block_def.env,
        mounts=mounts,
    )
    result = run_container(config, args)

    if result.exit_code != 0:
        raise RuntimeError(
            f"Block {block_name!r} container exited with code "
            f"{result.exit_code}"
        )

    # 7. Record output artifacts (convention-based)
    for slot in block_def.outputs:
        output_dir = workspace.path / "artifacts" / slot.name
        if any(output_dir.iterdir()):
            workspace.record_artifact(
                slot.name,
                CopyArtifact(name=slot.name, path=Path(".")),
            )

    # 8. Save workspace
    workspace.save()

    return result
