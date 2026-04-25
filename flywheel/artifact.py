"""Core data models for flywheel artifacts, executions, and events.

An artifact instance is a concrete, immutable record of data
registered in a workspace.  Instances enter the workspace through
several paths: produced by a block execution, imported directly
via ``flywheel import artifact``, or created by the ``flywheel
fix execution`` and ``flywheel amend artifact`` flows. A block
execution is the record of running a single block within a workspace.
A lifecycle event records an operational occurrence (e.g., agent
stopped, group completed) that is not itself a block execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from flywheel.state import StateMode


@dataclass(frozen=True)
class RejectionRef:
    """Reference to a rejected output slot from a failed execution.

    Used as the predecessor type when an :class:`ArtifactInstance`
    supersedes bytes that were quarantined after their producing
    execution's validator rejected them.

    Attributes:
        execution_id: The :class:`BlockExecution` whose output
            slot was rejected.
        slot: The output slot name within that execution.

    The pair ``(execution_id, slot)`` is the durable ledger
    handle for a rejected slot — the bytes themselves live under
    ``<workspace>/quarantine/<execution_id>/<slot>/`` (when
    quarantine succeeded), but lineage references the ledger
    pair, not the path, so the storage location is free to move.
    """

    execution_id: str
    slot: str


@dataclass(frozen=True)
class SupersedesRef:
    """Backward lineage pointer for a superseding ArtifactInstance.

    An :class:`ArtifactInstance` carrying a ``SupersedesRef``
    declares "this instance supersedes that predecessor."
    Exactly one of ``artifact_id`` or ``rejection`` is set;
    setting both or neither is a programmer error and raises
    ``ValueError``.

    The two flavours match the two operator workflows:

    * ``artifact_id``: amend an accepted instance — the operator
      has better bytes for the same artifact name and is
      recording a corrective successor (``flywheel amend
      artifact``).
    * ``rejection``: fix a quarantined slot from a failed
      execution — the operator corrected the bytes that the
      validator rejected and is registering them as a fresh
      instance (``flywheel fix execution``).

    The substrate treats the two flavours uniformly — the
    distinction matters for CLI ergonomics, not for storage.
    Lineage is **backward-only and kind-agnostic**: the
    predecessor is identified by a stable ledger handle, never
    by a filesystem path; multiple successors may point at the
    same predecessor (forking is allowed). The pointer is
    provenance / intent, not a resolution rule: consumers still
    resolve by ``latest created_at``.

    Attributes:
        artifact_id: The id of an accepted predecessor
            :class:`ArtifactInstance`. ``None`` when superseding
            a rejected slot.
        rejection: The :class:`RejectionRef` of a quarantined
            predecessor slot. ``None`` when superseding an
            accepted artifact.
    """

    artifact_id: str | None = None
    rejection: RejectionRef | None = None

    def __post_init__(self) -> None:
        """Enforce the one-of invariant at construction time."""
        if (self.artifact_id is None) == (self.rejection is None):
            raise ValueError(
                "SupersedesRef must have exactly one of "
                "'artifact_id' or 'rejection' set; got "
                f"artifact_id={self.artifact_id!r}, "
                f"rejection={self.rejection!r}."
            )


@dataclass(frozen=True)
class RejectedOutput:
    """Per-slot record of an output rejected by its validator.

    Lives on :attr:`BlockExecution.rejected_outputs` keyed by
    slot name.  When present, the producing execution is
    recorded as ``failed`` with
    :data:`flywheel.runtime.FAILURE_OUTPUT_VALIDATE` and the
    rejected bytes are preserved under
    ``<workspace>/quarantine/<execution_id>/<slot>/`` whenever
    the quarantine I/O succeeded.

    Attributes:
        reason: The validator's rejection message, surfaced
            verbatim from
            :class:`flywheel.artifact_validator.ArtifactValidationError`.
            Stable enough for operators (and AI agents) to grep
            and act on.
        quarantine_path: Workspace-relative path to the
            preserved bytes (e.g.,
            ``"quarantine/exec_a3f7b2/checkpoint"``), or
            ``None`` when quarantine I/O failed.  The
            validation failure is the primary signal either
            way; the missing path just means the bytes were
            not preserved this time.
        phase: The commit-pipeline phase at which the slot was
            rejected — one of
            :data:`flywheel.runtime.FAILURE_OUTPUT_COLLECT`,
            :data:`flywheel.runtime.FAILURE_OUTPUT_VALIDATE`, or
            :data:`flywheel.runtime.FAILURE_ARTIFACT_COMMIT`.
            Per-slot resolution lives here; the execution-level
            ``failure_phase`` is the most-downstream phase across
            all rejected slots.  Defaults to
            ``"output_validate"`` so legacy rejections (which
            only occurred at validation) deserialize unchanged.
    """

    reason: str
    quarantine_path: str | None = None
    phase: str = "output_validate"


@dataclass(frozen=True)
class ArtifactInstance:
    """A concrete artifact record within a workspace.

    Each instance has a unique ID in the form ``name@uuid``
    (e.g., ``checkpoint@a3f7b2``, ``engine@baseline``). Two storage
    kinds:

    * ``copy`` — directory of files written once; immutable after
      registration.
    * ``git`` — reference to a commit in a git repo.

    Attributes:
        id: Unique identifier within the workspace (e.g., ``checkpoint@a3f7b2``).
        name: The artifact declaration name this instance belongs to.
        kind: Storage kind, ``"copy"`` or ``"git"``.
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
        supersedes: Backward lineage pointer when this instance
            was registered as a superseding successor of a
            predecessor (either an accepted artifact id or a
            quarantined rejection ref).  ``None`` for plain
            registrations.  See :class:`SupersedesRef`.
        supersedes_reason: Human-readable description of *why*
            this successor was registered, captured at
            registration time so the audit trail can be
            reconstructed without out-of-band notes.  ``None``
            when ``supersedes`` is ``None``; recommended (but
            not required) when it is set.
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
    supersedes: SupersedesRef | None = None
    supersedes_reason: str | None = None


@dataclass(frozen=True)
class BlockExecution:
    """The record of a single block execution within a workspace.

    Captures the full provenance: which artifact instances were consumed,
    which were produced, and the outcome.

    Attributes:
        id: Unique identifier within the workspace (e.g., ``exec_a3f7b2e1``).
        block_name: The name of the block that was executed.
        started_at: When execution began.
        finished_at: When execution ended, or None if not yet finished.
        status: The outcome: ``"succeeded"``, ``"failed"``,
            ``"interrupted"``, or ``"running"`` for executions that
            have begun but not yet ended.  Derived from
            ``termination_reason`` via
            :func:`flywheel.termination.derive_status`.
        input_bindings: Maps each input artifact name to the artifact
            instance ID that was consumed.
        output_bindings: Maps each output artifact name to the artifact
            instance ID that was produced.
        exit_code: The container's exit code, if available.
        elapsed_s: Wall-clock time in seconds, if available.
        image: The Docker image that was used.
        runner: How this execution was physically performed:
            ``"container_one_shot"`` (one-shot container) or
            ``"container_persistent"`` (long-lived container,
            multiple invocations over an HTTP loop).
        error: Error message if status is ``"failed"`` and the
            failure produced a string description. None otherwise.
        failure_phase: When ``status == "failed"``, identifies
            *which* step of the execution pipeline failed.  Values
            from :data:`flywheel.runtime.FAILURE_PHASES`.
            ``None`` for succeeded / interrupted / running
            executions.
        rejected_outputs: Per-slot record of output slots that
            failed at any commit-time phase, keyed by slot name.
            Each entry carries the rejection ``reason``, the
            ``phase`` of failure (``output_collect``,
            ``output_validate``, or ``artifact_commit``), and the
            workspace-relative ``quarantine_path`` (when quarantine
            I/O succeeded).  Empty dict by default; absent on disk
            for executions with no rejected slots.  See
            :class:`RejectedOutput`.
        termination_reason: How the execution ended, as a string
            label.  Either a substrate-reserved value
            (``"crash"``, ``"interrupted"``, ``"timeout"``,
            ``"protocol_violation"`` — see
            :data:`flywheel.runtime.RESERVED_TERMINATION_REASONS`)
            describing what the substrate observed, or a
            project-defined value the block announced via the
            per-runtime termination channel.  ``None`` for legacy
            executions that predate the field.
        state_mode: Whether this execution had no substrate-visible
            state, managed state snapshots, or unmanaged runtime
            state.
        state_snapshot_id: Managed state snapshot captured by this
            execution, when one was captured.  ``None`` for
            ``state_mode`` values other than ``"managed"`` and for
            managed executions that did not capture a snapshot.

    Fields removed in the block-execution refactor (see
    ``flywheel/docs/specs/block-execution.md``):

    * ``caller`` and ``agent_workspace_dir`` — agent/MCP-shaped,
      removed entirely from the substrate schema.  Will live in
      whatever batteries-layer record the agent runtime owns.
    * ``parent_execution_id`` — handoff-shaped; will return as
      ``invoking_execution_id`` when the handoff primitive lands.
    * ``params``, ``synthetic``, ``halt_directive``,
      ``post_check_error``, ``state_dir``, ``state_lineage_id``,
      ``run_id``, ``predecessor_id``, ``stop_reason`` — out of
      happy-path scope; their fates are decided by the follow-on
      specs (state, handoffs, patterns, etc.) that own those
      concepts.
    """

    id: str
    block_name: str
    started_at: datetime
    finished_at: datetime | None = None
    status: Literal[
        "succeeded", "failed", "interrupted", "running"
    ] = "failed"
    input_bindings: dict[str, str] = field(default_factory=dict)
    output_bindings: dict[str, str] = field(default_factory=dict)
    exit_code: int | None = None
    elapsed_s: float | None = None
    image: str | None = None
    runner: Literal[
        "container_one_shot", "container_persistent"
    ] = "container_one_shot"
    error: str | None = None
    failure_phase: str | None = None
    rejected_outputs: dict[str, RejectedOutput] = field(
        default_factory=dict)
    termination_reason: str | None = None
    state_mode: StateMode = "none"
    state_snapshot_id: str | None = None


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
