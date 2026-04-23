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
    """

    reason: str
    quarantine_path: str | None = None


@dataclass(frozen=True)
class ArtifactInstance:
    """A concrete artifact record within a workspace.

    Each instance has a unique ID in the form ``name@uuid``
    (e.g., ``checkpoint@a3f7b2``, ``engine@baseline``). Three storage
    kinds:

    * ``copy`` — directory of files written once; immutable after
      registration.
    * ``git`` — reference to a commit in a git repo.
    * ``incremental`` — directory containing exactly one
      ``entries.jsonl`` file, append-only.  Each appended entry is
      itself immutable (no rewrites of earlier entries); the file as
      a whole grows over time as block executions add new entries.
      The "logical instance" is a sequence of immutable entries
      stored in one growing file.

    Attributes:
        id: Unique identifier within the workspace (e.g., ``checkpoint@a3f7b2``).
        name: The artifact declaration name this instance belongs to.
        kind: Storage kind, ``"copy"``, ``"git"``, or ``"incremental"``.
        created_at: When this instance was created.
        produced_by: The execution ID that produced this instance,
            or None for artifacts not produced by a block execution
            (e.g., baseline git snapshots or imported artifacts).
            For incremental artifacts, this is the *first* execution
            that appended an entry; subsequent appenders are recorded
            in their own ``BlockExecution.output_bindings``.
        source: For imported artifacts, a description of where the
            artifact came from (e.g., a file path). None for
            block-produced and baseline artifacts.
        copy_path: For copy and incremental artifacts, the directory
            name under ``workspace/artifacts/`` where files are
            stored.  Copy artifacts hold arbitrary files; incremental
            artifacts hold exactly one ``entries.jsonl``.
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
    kind: Literal["copy", "git", "incremental"]
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

    Two distinct linkage fields exist:

    - ``predecessor_id`` is the *resume chain*: this execution
      continues a previous one (same logical agent step, fresh
      container).
    - ``parent_execution_id`` is the *control-flow tree*: this
      execution was launched from inside another one (e.g., a tool
      call inside an agent container that triggers a logical block).

    Attributes:
        id: Unique identifier within the workspace (e.g., ``exec_a3f7b2e1``).
        block_name: The name of the block that was executed.
        started_at: When execution began.
        finished_at: When execution ended, or None if not yet finished.
        status: The outcome: ``"succeeded"``, ``"failed"``,
            ``"interrupted"``, or ``"running"`` for executions that
            have begun but not yet ended (used by the lifecycle API
            between begin and end).
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
        parent_execution_id: Execution ID of the execution that
            launched this one (e.g., the agent container hosting a
            tool-triggered logical block execution). None for
            top-level executions. Distinct from ``predecessor_id``:
            parent is "who launched me," predecessor is "who am I
            resuming."
        runner: How this execution was physically performed:
            ``"container"`` (Docker), ``"record"`` (legacy
            record-mode, no work performed), or ``"agent"``
            (long-lived agent container).  None for legacy
            executions that predate the runner concept.
        caller: For tool-triggered logical executions, identifies
            which MCP server and tool invoked this block. Shape is
            ``{"mcp_server": str, "tool": str}``. None for executions
            not triggered by a tool call.
        params: Function-argument parameters supplied by the caller
            (e.g., ``{"action_id": 6, "x": 15, "y": 20}``).  These
            are not artifacts but are recorded for lineage so the
            full call can be reconstructed.  None if the block has
            no params or for legacy executions.
        error: Error message if status is ``"failed"`` and the
            failure produced a string description. None otherwise.
        synthetic: ``True`` when the channel created this execution record in
        place of a request that never made it past
        ``/execution/begin`` (e.g., manifest mismatch,
        unknown block, missing input).  Synthetic executions let
            post-execution checks see infrastructure failures the
            same way they see body failures.  Default ``False``.
        halt_directive: Persisted form of a
            :class:`flywheel.post_check.HaltDirective` returned by
            this block's post-execution callback.  Shape is
            ``{"scope": "caller"|"run", "reason": str}``.  Means
            "the post-check asked one or more runners to stop
            after this execution landed."  ``None`` when no directive
            was issued.  Distinct from ``stop_reason``, which
            describes why an *individual* execution ended.
        post_check_error: Error string if the post-execution
            callback itself raised.  The execution record is not retroactively
            marked failed; the field exists so operators can see
            the check is broken without it taking the run down.
            ``None`` when no callback was configured or it
            completed normally.
        agent_workspace_dir: Workspace-relative path to the
            ``/scratch`` bind mount used by this agent (e.g.,
            ``"agent_workspaces/abc12345"``).  Recorded so an
            operator looking at a workspace directory can find
            the execution that produced it (and vice versa) without
            having to grep timestamps.  ``None`` for non-agent
            executions and for agent executions that predate this
            field.
        state_dir: Workspace-relative path to the directory where
            this execution's private state was captured (e.g.,
            ``"state/play/exec_a3f7b2e1"``).  Set only for blocks
            that declare ``state: true`` and whose state-capture
            step succeeded.  Not an artifact — never in the
            artifact graph, never consumed by other blocks.
            ``None`` for stateless blocks, for executions that
            failed before state could be captured, and for legacy
            executions that predate this field.
        failure_phase: When ``status == "failed"``, identifies
            *which* step of the execution pipeline failed.  One
            of ``"stage_in"``, ``"invoke"``, ``"state_capture"``,
            ``"output_collect"``, ``"artifact_commit"`` — see
            :mod:`flywheel.runtime` for the canonical constants.
            ``None`` for succeeded / interrupted / running
            executions, and for failed executions that predate
            this field.
        state_lineage_id: Names the state lineage this execution
            belongs to.  The state loader matches on
            ``(block_name, state_lineage_id)`` to find the prior
            state to restore.  ``None`` means the default
            (single-lineage) bucket — today's executor always
            passes ``None``, which gives every block one
            evolving state chain per workspace.  When multiple
            lineages become needed (parallel instances, forks),
            callers can pass distinct IDs without a schema
            migration.
        run_id: Names the :class:`RunRecord` this execution
            belongs to.  ``None`` means "ad-hoc" — this
            execution was not started inside a durable grouped
            run (direct ``flywheel run block`` / ``flywheel run
            agent`` calls, for example).  ``PatternRunner``
            opens a run and tags every ``BlockExecution`` it
            drives with that id; nested executions triggered
            through the host-side handoff loop inherit the
            same id from the agent that triggered them.
            Cadence counters scope by this field so executions
            from a prior run do not pollute a fresh run's
            counter.
        rejected_outputs: Per-slot record of output slots that
            failed validation, keyed by slot name.  Present only
            on executions whose ``failure_phase`` is
            ``"output_validate"``.  Each entry carries the
            validator's rejection ``reason`` and the
            workspace-relative ``quarantine_path`` (when
            quarantine I/O succeeded).  Empty dict by default;
            absent on disk for executions with no rejected
            slots.  See :class:`RejectedOutput`.
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
    stop_reason: str | None = None
    predecessor_id: str | None = None
    parent_execution_id: str | None = None
    runner: str | None = None
    caller: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    error: str | None = None
    synthetic: bool = False
    halt_directive: dict[str, Any] | None = None
    post_check_error: str | None = None
    agent_workspace_dir: str | None = None
    state_dir: str | None = None
    failure_phase: str | None = None
    state_lineage_id: str | None = None
    run_id: str | None = None
    rejected_outputs: dict[str, RejectedOutput] = field(
        default_factory=dict)


@dataclass(frozen=True)
class RunRecord:
    """A durable grouping of related block executions.

    Sits between :class:`flywheel.workspace.Workspace` and
    :class:`BlockExecution`.  Each ``PatternRunner`` invocation
    opens one run record and tags every execution it drives
    (including nested executions triggered through the host-side
    handoff loop) with the run's id.  Cadence triggers like
    ``every_n_executions`` scope their counters to
    ``run_id == self._run_id`` so re-running a pattern in an
    existing workspace does not fire a backlog of cohorts
    against the prior run's execution count.

    A workspace can hold many runs.  Ad-hoc work (direct
    ``flywheel run block`` / ``flywheel run agent`` calls, or
    tests exercising the executor in isolation) produces
    ``BlockExecution`` records with ``run_id=None``; no
    ``RunRecord`` is created for them.

    Attributes:
        id: Unique identifier within the workspace
            (e.g., ``"run_a3f7b2e1"``).
        kind: What opened this run.  Convention:
            ``"pattern:<name>"`` for pattern-driven runs
            (e.g., ``"pattern:play-brainstorm"``).  Ad-hoc
            CLI invocations do not open a run, so there is
            no ``"ad_hoc"`` kind today; future operator-
            initiated groupings may add their own kinds.
        started_at: When the run opened.
        finished_at: When the run closed, or ``None`` if still
            running.  An interrupted host process leaves the
            record as ``"running"`` with ``finished_at=None``
            until a reconciliation or resume command touches
            it.
        status: ``"running"``, ``"succeeded"``, ``"failed"``,
            ``"stopped"``.  Set to ``"running"`` on open; the
            pattern runner updates it in its outer
            ``try/finally``.
        config_snapshot: Optional free-form mapping the caller
            records alongside the run for later inspection
            (model names, budgets, pattern overrides, etc.).
            ``None`` when the caller had nothing worth
            capturing.  Not inspected by the runner.
    """

    id: str
    kind: str
    started_at: datetime
    finished_at: datetime | None = None
    status: Literal[
        "running", "succeeded", "failed", "stopped"
    ] = "running"
    config_snapshot: dict[str, Any] | None = None


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
