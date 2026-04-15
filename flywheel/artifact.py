"""Core data models for flywheel artifacts, executions, and events.

An artifact instance is a concrete, immutable record of data produced
by a block execution or captured at workspace creation. A block
execution is the record of running a single block within a workspace.
A lifecycle event records an operational occurrence (e.g., agent
stopped, group completed) that is not itself a block execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


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
            (e.g., baseline git snapshots or imported artifacts).
        source: For imported artifacts, a description of where the
            artifact came from (e.g., a file path). None for
            block-produced and baseline artifacts.
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
    source: str | None = None
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
        status: The outcome: ``"succeeded"``, ``"failed"``, or
            ``"interrupted"``.
        input_bindings: Maps each input artifact name to the artifact
            instance ID that was consumed.
        output_bindings: Maps each output artifact name to the artifact
            instance ID that was produced.
        exit_code: The container's exit code, if available.
        elapsed_s: Wall-clock time in seconds, if available.
        image: The Docker image that was used.
        stop_reason: Why the execution ended, if it was stopped
            externally (e.g., ``"exploration_request"``,
            ``"prediction_mismatch"``, ``"timeout"``). None for
            normal completion.
        predecessor_id: Execution ID that this block resumes from,
            enabling resume chains. None if this is a fresh start.
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
    stop_reason: str | None = None
    predecessor_id: str | None = None


@dataclass(frozen=True)
class LifecycleEvent:
    """An operational event recorded in the workspace.

    Captures orchestration-level events that are not block
    executions: agent stops, group completions, mode transitions,
    and other operational milestones.

    Attributes:
        id: Unique identifier (e.g., ``"evt_a3f7b2"``).
        kind: Event type (e.g., ``"agent_stopped"``,
            ``"group_completed"``, ``"exploration_started"``).
        timestamp: When the event occurred.
        execution_id: Related execution ID, if any.
        detail: Free-form metadata dict.
    """

    id: str
    kind: str
    timestamp: datetime
    execution_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
