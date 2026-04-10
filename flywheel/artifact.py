"""Core data models for flywheel: artifact instances and block executions.

An artifact instance is a concrete, immutable record of data produced
by a block execution or captured at workspace creation. A block
execution is the record of running a single block within a workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class ArtifactInstance:
    """A concrete, immutable artifact record within a workspace.

    Each instance has a unique ID in the form ``name@uuid``
    (e.g., ``checkpoint@a3f7b2``, ``engine@baseline``). Copy artifacts
    store files in the workspace; git artifacts reference
    version-controlled code.

    Attributes:
        id: Unique identifier within the workspace (e.g., ``checkpoint@a3f7b2``).
        name: The artifact declaration name this instance belongs to.
        kind: Storage kind, either ``"copy"`` or ``"git"``.
        created_at: When this instance was created.
        produced_by: The execution ID that produced this instance,
            or None for artifacts not produced by a block execution
            (e.g., baseline git snapshots created at workspace setup).
        copy_path: For copy artifacts, the directory name under
            ``workspace/artifacts/`` where files are stored.
        repo: For git artifacts, the absolute path to the repo root.
        commit: For git artifacts, the full commit SHA.
        git_path: For git artifacts, the path within the repo.
    """

    id: str
    name: str
    kind: Literal["copy", "git"]
    created_at: datetime
    produced_by: str | None = None
    copy_path: str | None = None
    repo: str | None = None
    commit: str | None = None
    git_path: str | None = None


@dataclass(frozen=True)
class BlockExecution:
    """The record of a single block execution within a workspace.

    Captures the full provenance: which artifact instances were consumed,
    which were produced, and the outcome.

    Attributes:
        id: Unique identifier within the workspace (e.g., ``exec1``).
        block_name: The name of the block that was executed.
        started_at: When execution began.
        finished_at: When execution ended, or None if not yet finished.
        status: The outcome, either ``"succeeded"`` or ``"failed"``.
        input_bindings: Maps each input artifact name to the artifact
            instance ID that was consumed.
        output_bindings: Maps each output artifact name to the artifact
            instance ID that was produced.
        exit_code: The container's exit code, if available.
        elapsed_s: Wall-clock time in seconds, if available.
        image: The Docker image that was used.
    """

    id: str
    block_name: str
    started_at: datetime
    finished_at: datetime | None = None
    status: Literal["succeeded", "failed", "interrupted"] = "failed"
    input_bindings: dict[str, str] = field(default_factory=dict)
    output_bindings: dict[str, str] = field(default_factory=dict)
    exit_code: int | None = None
    elapsed_s: float | None = None
    image: str | None = None
