"""Per-mount input staging for artifact instances.

Invariant: **canonical artifacts are never directly read,
mounted, or otherwise exposed to a downstream block or
container**.  Every input slot is satisfied by a fresh
per-mount copy of the artifact's contents, living in an
temporary system tempdir for the lifetime of the consuming
execution.

Two motivations:

1. **Mutation isolation.**  A misbehaving in-container script
   (or a buggy in-process block) cannot corrupt the canonical
   workspace store by writing through its mount, because the
   mount points at a copy and only at a copy.
2. **Snapshot semantics.**  When a long-running incremental
   artifact is mounted, the consumer sees the snapshot taken at
   mount time.  Late appends by other executions don't surprise
   the reader mid-run.

This module is the staging primitive both
:mod:`flywheel.execution` (for container input mounts) and
:mod:`flywheel.local_block` (for in-process input bindings) use.
The trade-off is explicit: a full copy on every mount is the
price of guaranteed isolation.  The user reviewed and accepted
this; do not regress to hardlinks or shared inodes without
re-litigating that decision.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from flywheel.artifact import ArtifactInstance
from flywheel.workspace import Workspace

STAGING_PREFIX = "flywheel-mount-"
"""Default prefix passed to :func:`tempfile.mkdtemp`.

Stamped onto every staging tempdir so an operator inspecting
``$TMPDIR`` can spot orphaned mounts left behind by a crashed
runner.  Production cleanup runs in the same process; orphans
indicate a bug rather than expected wear.
"""


class StagingError(RuntimeError):
    """Raised when an artifact instance cannot be staged.

    Distinct from a generic :class:`OSError` so callers (the agent
    launcher, the in-process recorder) can decide whether to
    skip the slot or fail the whole launch.  Carries the offending
    artifact id in :attr:`instance_id` for log context.
    """

    def __init__(self, message: str, *, instance_id: str) -> None:
        """Initialize with the artifact id that failed to stage."""
        super().__init__(message)
        self.instance_id = instance_id


def stage_artifact_instance(
    workspace: Workspace,
    instance: ArtifactInstance,
    *,
    prefix: str = STAGING_PREFIX,
) -> Path:
    """Copy *instance*'s on-disk contents into a fresh tempdir.

    Allocates a new tempdir via :func:`tempfile.mkdtemp` and
    copies every entry under the canonical artifact directory
    into it.  The returned path is what callers mount at
    ``/input/<slot>/`` (containers) or hand to the block as
    its read-only input directory (in-process blocks).

    Always a full copy; never a hardlink, never a symlink.
    The user explicitly rejected the symlink/hardlink optimization
    in favor of mutation isolation.

    Args:
        workspace: The workspace whose canonical artifact store
            holds *instance*.
        instance: The artifact instance to stage.  Must be of
            kind ``"copy"`` or ``"incremental"``; ``"git"``
            artifacts have no workspace-local directory and
            should be handled by callers separately (today's
            mount path skips them silently, preserving the
            historical no-op behavior).
        prefix: Tempdir prefix.  Defaults to
            :data:`STAGING_PREFIX`; tests may override to make
            tempdir provenance obvious in error messages.

    Returns:
        Absolute :class:`Path` to the freshly populated
        tempdir.  Caller owns cleanup — call
        :func:`cleanup_staged_inputs` (or :func:`shutil.rmtree`)
        once the consuming execution exits.

    Raises:
        StagingError: If the instance kind is ``"git"`` (not
            stageable through this path), the instance has no
            ``copy_path``, or the canonical artifact directory
            is missing on disk.
    """
    if instance.kind == "git":
        raise StagingError(
            f"git artifact {instance.id!r} cannot be staged "
            f"through stage_artifact_instance; git mounts are "
            f"out of scope for the input-staging path",
            instance_id=instance.id,
        )
    if instance.copy_path is None:
        raise StagingError(
            f"artifact {instance.id!r} has no copy_path; "
            f"cannot stage",
            instance_id=instance.id,
        )

    src = workspace.path / "artifacts" / instance.copy_path
    if not src.exists() or not src.is_dir():
        raise StagingError(
            f"artifact {instance.id!r} canonical directory "
            f"{src} is missing; cannot stage",
            instance_id=instance.id,
        )

    staging = Path(tempfile.mkdtemp(
        prefix=f"{prefix}{instance.name}-"))

    try:
        for child in src.iterdir():
            dest = staging / child.name
            if child.is_dir():
                shutil.copytree(child, dest)
            else:
                shutil.copy2(child, dest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return staging


def stage_artifact_instances(
    workspace: Workspace,
    bindings: dict[str, str],
    *,
    prefix: str = STAGING_PREFIX,
) -> dict[str, Path]:
    """Stage a batch of input artifacts; return slot -> staging-path.

    Convenience wrapper around :func:`stage_artifact_instance`
    that walks a ``{slot_name: artifact_id}`` mapping (the same
    shape :class:`flywheel.local_block.LocalExecutionContext`
    and battery adapters consume) and
    returns ``{slot_name: staging_path}``.

    Bindings whose artifact id is unknown to the workspace are
    skipped silently — the caller already validated existence
    upstream and a missing id at staging time means "not
    stageable through this path", same semantic as a git
    artifact.  Bindings to git artifacts are also skipped
    silently (matching today's mount path).

    Returns:
        ``{slot_name: staging_path}`` for the slots that were
        successfully staged.  Slots whose artifacts were
        skipped (git or unknown id) are absent from the result
        rather than mapped to ``None``, so callers iterate
        without nullability paperwork.

    Raises:
        StagingError: Propagated from
            :func:`stage_artifact_instance` for the first slot
            whose copy/incremental staging actually failed.
            Already-staged tempdirs from prior slots in this
            batch are cleaned up before the exception propagates,
            so a partial failure leaves no orphaned dirs behind.
    """
    staged: dict[str, Path] = {}
    try:
        for slot_name, artifact_id in bindings.items():
            inst = workspace.artifacts.get(artifact_id)
            if inst is None:
                continue
            if inst.kind == "git":
                continue
            staged[slot_name] = stage_artifact_instance(
                workspace, inst, prefix=prefix)
    except Exception:
        cleanup_staged_inputs(staged)
        raise
    return staged


def cleanup_staged_inputs(staged: dict[str, Path]) -> None:
    """Remove staging tempdirs allocated by :func:`stage_artifact_instances`.

    Errors are swallowed (``ignore_errors=True``); cleanup is
    best-effort and a leaked tempdir is preferable to a noisy
    crash on the way out of an otherwise successful execution.
    Call this from the consumer's ``finally`` block once the
    container or in-process body has exited.
    """
    for path in list(staged.values()):
        shutil.rmtree(path, ignore_errors=True)
