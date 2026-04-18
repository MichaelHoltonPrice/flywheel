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
            ``"container"`` (Docker), ``"inprocess"`` (function call
            in the calling container), ``"subprocess"`` (local
            process), ``"record"`` (legacy record-mode, no work
            performed), or ``"agent"`` (long-lived agent container).
            None for legacy rows that predate the runner concept.
        caller: For tool-triggered logical executions, identifies
            which MCP server and tool invoked this block. Shape is
            ``{"mcp_server": str, "tool": str}``. None for executions
            not triggered by a tool call.
        params: Function-argument parameters supplied by the caller
            (e.g., ``{"action_id": 6, "x": 15, "y": 20}``).  These
            are not artifacts but are recorded for lineage so the
            full call can be reconstructed.  None if the block has
            no params or for legacy rows.
        error: Error message if status is ``"failed"`` and the
            failure produced a string description. None otherwise.
        synthetic: ``True`` when the channel created this row in
            place of a request that never made it past
            ``/execution/begin`` (e.g., manifest mismatch,
            unknown block, missing input).  Synthetic rows let
            post-execution checks see infrastructure failures the
            same way they see body failures.  Default ``False``.
        halt_directive: Persisted form of a
            :class:`flywheel.post_check.HaltDirective` returned by
            this block's post-execution callback.  Shape is
            ``{"scope": "caller"|"run", "reason": str}``.  Means
            "the post-check asked one or more runners to stop
            after this row landed."  ``None`` when no directive
            was issued.  Distinct from ``stop_reason``, which
            describes why an *individual* execution ended.
        post_check_error: Error string if the post-execution
            callback itself raised.  The row is not retroactively
            marked failed; the field exists so operators can see
            the check is broken without it taking the run down.
            ``None`` when no callback was configured or it
            completed normally.
        agent_workspace_dir: Workspace-relative path to the
            ``/workspace`` bind mount used by this agent (e.g.,
            ``"agent_workspaces/abc12345"``).  Recorded so an
            operator looking at a workspace directory can find
            the row that produced it (and vice versa) without
            having to grep timestamps.  ``None`` for non-agent
            executions and for agent rows that predate this
            field.
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
