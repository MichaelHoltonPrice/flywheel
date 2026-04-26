"""Workspace creation, persistence, and artifact/execution tracking.

A workspace is a directory inside a project's foundry folder,
created from a template. It accumulates artifact instances and
block execution records over its lifetime, forming a complete
provenance graph.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from flywheel.artifact import (
    ArtifactInstance,
    BlockExecution,
    BlockInvocation,
    LifecycleEvent,
    RejectedOutput,
    RejectionRef,
    SupersedesRef,
)
from flywheel.artifact_validator import ArtifactValidatorRegistry
from flywheel.pattern_lanes import DEFAULT_LANE
from flywheel.run_record import (
    RunFixtureRecord,
    RunMemberRecord,
    RunRecord,
    RunStepRecord,
)
from flywheel.runtime import FAILURE_PHASES
from flywheel.state import StateSnapshot
from flywheel.template import ArtifactDeclaration, Template
from flywheel.termination import derive_status
from flywheel.validation import validate_name as _validate_name

_PREDECESSOR_UNSPECIFIED = object()


def _supersedes_to_yaml(ref: SupersedesRef) -> dict:
    """Serialize a :class:`SupersedesRef` for ``workspace.yaml``.

    Output shape is one of two discriminated maps:
    ``{"artifact": "<id>"}`` or
    ``{"rejected": {"execution": "<id>", "slot": "<slot>"}}``.
    The shape is stable on purpose so operators (and any other
    tooling reading the ledger by hand) can spot the predecessor
    flavour without consulting code.
    """
    if ref.artifact_id is not None:
        return {"artifact": ref.artifact_id}
    assert ref.rejection is not None  # guaranteed by SupersedesRef
    return {
        "rejected": {
            "execution": ref.rejection.execution_id,
            "slot": ref.rejection.slot,
        }
    }


def _supersedes_from_yaml(entry: object) -> SupersedesRef | None:
    """Inverse of :func:`_supersedes_to_yaml`; tolerates ``None``."""
    if entry is None:
        return None
    if not isinstance(entry, dict):
        raise ValueError(
            f"supersedes entry must be a mapping, got "
            f"{type(entry).__name__}"
        )
    if "artifact" in entry:
        return SupersedesRef(artifact_id=entry["artifact"])
    if "rejected" in entry:
        rej = entry["rejected"]
        return SupersedesRef(rejection=RejectionRef(
            execution_id=rej["execution"], slot=rej["slot"],
        ))
    raise ValueError(
        f"supersedes entry must have an 'artifact' or 'rejected' "
        f"key; got keys {sorted(entry)}"
    )


def _rejected_outputs_to_yaml(
    outputs: dict[str, RejectedOutput],
) -> dict:
    """Serialize ``BlockExecution.rejected_outputs`` for YAML.

    Per-slot entries omit ``quarantine_path`` when ``None`` so
    failed-quarantine cases serialize as a single ``reason``
    field rather than a stale ``quarantine_path: null``.
    """
    out: dict = {}
    for slot, rec in outputs.items():
        entry: dict = {"reason": rec.reason}
        if rec.quarantine_path is not None:
            entry["quarantine_path"] = rec.quarantine_path
        # ``phase`` is always populated on the dataclass (default
        # ``"output_validate"`` for legacy rejections) so it
        # always serializes — keeps the on-disk record
        # self-describing without a back-compat sniff at read
        # time.
        entry["phase"] = rec.phase
        out[slot] = entry
    return out


def _rejected_outputs_from_yaml(
    entry: object,
) -> dict[str, RejectedOutput]:
    """Inverse of :func:`_rejected_outputs_to_yaml`."""
    if entry is None:
        return {}
    if not isinstance(entry, dict):
        raise ValueError(
            f"rejected_outputs must be a mapping, got "
            f"{type(entry).__name__}"
        )
    out: dict[str, RejectedOutput] = {}
    for slot, rec in entry.items():
        if not isinstance(rec, dict):
            raise ValueError(
                f"rejected_outputs[{slot!r}] must be a mapping, "
                f"got {type(rec).__name__}"
            )
        out[slot] = RejectedOutput(
            reason=rec["reason"],
            quarantine_path=rec.get("quarantine_path"),
            phase=rec.get("phase", "output_validate"),
        )
    return out


def _run_member_to_yaml(member: RunMemberRecord) -> dict:
    """Serialize a run member result for ``workspace.yaml``."""
    entry: dict[str, Any] = {
        "name": member.name,
        "block_name": member.block_name,
        "status": member.status,
        "lane": member.lane,
    }
    if member.execution_id is not None:
        entry["execution_id"] = member.execution_id
    if member.output_bindings:
        entry["output_bindings"] = dict(member.output_bindings)
    if member.invocation_ids:
        entry["invocation_ids"] = list(member.invocation_ids)
    if member.error is not None:
        entry["error"] = member.error
    return entry


def _run_step_to_yaml(step: RunStepRecord) -> dict:
    """Serialize a run step result for ``workspace.yaml``."""
    return {
        "name": step.name,
        "min_successes": step.min_successes,
        "status": step.status,
        "members": [
            _run_member_to_yaml(member) for member in step.members
        ],
    }


def _run_steps_from_yaml(entry: object) -> list[RunStepRecord]:
    """Deserialize run step results from ``workspace.yaml``."""
    if entry is None:
        return []
    if not isinstance(entry, list):
        raise ValueError(
            f"Run steps must be a list, got {type(entry).__name__}"
        )
    steps: list[RunStepRecord] = []
    for raw_step in entry:
        if not isinstance(raw_step, dict):
            raise ValueError(
                "Run step entries must be mappings, got "
                f"{type(raw_step).__name__}"
            )
        members: list[RunMemberRecord] = []
        for raw_member in raw_step.get("members", []):
            if not isinstance(raw_member, dict):
                raise ValueError(
                    "Run member entries must be mappings, got "
                    f"{type(raw_member).__name__}"
                )
            members.append(RunMemberRecord(
                name=raw_member["name"],
                block_name=raw_member["block_name"],
                status=raw_member["status"],
                lane=raw_member.get("lane", DEFAULT_LANE),
                execution_id=raw_member.get("execution_id"),
                output_bindings=dict(
                    raw_member.get("output_bindings", {})),
                invocation_ids=list(
                    raw_member.get("invocation_ids", [])),
                error=raw_member.get("error"),
            ))
        steps.append(RunStepRecord(
            name=raw_step["name"],
            min_successes=raw_step["min_successes"],
            status=raw_step["status"],
            members=members,
        ))
    return steps


def _run_fixtures_to_yaml(fixtures: list[RunFixtureRecord]) -> list[dict]:
    """Serialize run fixture materializations."""
    return [
        {
            "id": fixture.id,
            "lane": fixture.lane,
            "name": fixture.name,
            "artifact_id": fixture.artifact_id,
            "source": fixture.source,
        }
        for fixture in fixtures
    ]


def _run_fixtures_from_yaml(entry: object) -> list[RunFixtureRecord]:
    """Deserialize run fixture materializations."""
    if entry is None:
        return []
    if not isinstance(entry, list):
        raise ValueError(
            f"Run fixtures must be a list, got {type(entry).__name__}"
        )
    fixtures: list[RunFixtureRecord] = []
    for raw_fixture in entry:
        if not isinstance(raw_fixture, dict):
            raise ValueError(
                "Run fixture entries must be mappings, got "
                f"{type(raw_fixture).__name__}"
            )
        fixtures.append(RunFixtureRecord(
            id=raw_fixture["id"],
            lane=raw_fixture["lane"],
            name=raw_fixture["name"],
            artifact_id=raw_fixture["artifact_id"],
            source=raw_fixture["source"],
        ))
    return fixtures


def _state_snapshot_to_yaml(snapshot: StateSnapshot) -> dict:
    """Serialize a managed state snapshot for ``workspace.yaml``."""
    return {
        "lineage_key": snapshot.lineage_key,
        "created_at": snapshot.created_at.isoformat(),
        "produced_by": snapshot.produced_by,
        "predecessor_snapshot_id": snapshot.predecessor_snapshot_id,
        "compatibility": dict(snapshot.compatibility),
        "state_path": snapshot.state_path,
    }


def _state_snapshots_from_yaml(entry: object) -> dict[str, StateSnapshot]:
    """Deserialize managed state snapshots from ``workspace.yaml``."""
    if entry is None:
        return {}
    if not isinstance(entry, dict):
        raise ValueError(
            f"State snapshots must be a mapping, got "
            f"{type(entry).__name__}"
        )
    snapshots: dict[str, StateSnapshot] = {}
    for snapshot_id, raw in entry.items():
        if not isinstance(raw, dict):
            raise ValueError(
                f"State snapshot {snapshot_id!r} must be a mapping, "
                f"got {type(raw).__name__}"
            )
        snapshots[snapshot_id] = StateSnapshot(
            id=snapshot_id,
            lineage_key=raw["lineage_key"],
            created_at=datetime.fromisoformat(raw["created_at"]),
            produced_by=raw["produced_by"],
            predecessor_snapshot_id=raw.get("predecessor_snapshot_id"),
            compatibility=dict(raw.get("compatibility", {})),
            state_path=raw["state_path"],
        )
    return snapshots


@dataclass
class Workspace:
    """A workspace instance created from a template.

    Accumulates artifact instances and block execution records.
    Artifact instances are immutable and keyed by ID. Execution
    records are append-only.
    """

    name: str
    path: Path
    template_name: str
    created_at: datetime
    artifact_declarations: dict[str, str]  # declaration name -> kind
    artifacts: dict[str, ArtifactInstance]  # id -> instance
    executions: dict[str, BlockExecution] = field(default_factory=dict)
    invocations: dict[str, BlockInvocation] = field(default_factory=dict)
    events: dict[str, LifecycleEvent] = field(default_factory=dict)
    runs: dict[str, RunRecord] = field(default_factory=dict)
    state_snapshots: dict[str, StateSnapshot] = field(default_factory=dict)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False,
    )

    @staticmethod
    def _short_uuid() -> str:
        """Generate a short unique identifier (8 hex characters)."""
        return uuid.uuid4().hex[:8]

    def generate_artifact_id(self, artifact_name: str) -> str:
        """Generate a unique artifact ID for a declaration.

        Args:
            artifact_name: The artifact declaration name.

        Returns:
            An ID in the form ``name@hexstring``, guaranteed unique
            within this workspace.
        """
        while True:
            candidate = f"{artifact_name}@{self._short_uuid()}"
            if candidate not in self.artifacts:
                return candidate

    def generate_execution_id(self) -> str:
        """Generate a unique execution ID.

        Returns:
            An ID in the form ``exec_hexstring``, guaranteed unique
            within this workspace.
        """
        while True:
            candidate = f"exec_{self._short_uuid()}"
            if candidate not in self.executions:
                return candidate

    def generate_invocation_id(self) -> str:
        """Generate a unique block invocation ID."""
        while True:
            candidate = f"inv_{self._short_uuid()}"
            if candidate not in self.invocations:
                return candidate

    def generate_state_snapshot_id(self) -> str:
        """Generate a unique managed state snapshot ID."""
        while True:
            candidate = f"state_{self._short_uuid()}"
            if candidate not in self.state_snapshots:
                return candidate

    def generate_run_fixture_id(self, run_id: str) -> str:
        """Generate a unique fixture ID for one run."""
        run = self.runs.get(run_id)
        existing = {fixture.id for fixture in run.fixtures} if run else set()
        while True:
            candidate = f"fix_{self._short_uuid()}"
            if candidate not in existing:
                return candidate

    def _add_artifact(self, instance: ArtifactInstance) -> None:
        """Add an artifact instance to the workspace ledger.

        Private. The sanctioned write paths for artifacts are
        :meth:`register_artifact` and (forthcoming)
        ``register_git_artifact``. This method exists only as the
        shared low-level write inside those canonical paths and is
        not meant to be called from outside the workspace, callers
        outside ``Workspace`` should use a canonical path. Tests may
        call it for fixture setup but should treat that as a
        deliberate internal-API access.

        Thread-safe: acquires the workspace lock.

        Args:
            instance: The artifact instance to add.

        Raises:
            ValueError: If an artifact with this ID already exists,
                the artifact name is not declared, the kind does not
                match, or required fields for the kind are missing.
        """
        with self._lock:
            if instance.id in self.artifacts:
                raise ValueError(
                    f"Artifact {instance.id!r} already exists "
                    f"in workspace"
                )
            if instance.name not in self.artifact_declarations:
                raise ValueError(
                    f"Artifact {instance.name!r} not declared "
                    f"in this workspace"
                )
            expected_kind = self.artifact_declarations[instance.name]
            if instance.kind != expected_kind:
                raise ValueError(
                    f"Artifact {instance.id!r} has kind "
                    f"{instance.kind!r} but {instance.name!r} "
                    f"expects {expected_kind!r}"
                )
            if instance.kind == "copy" and instance.copy_path is None:
                raise ValueError(
                    f"Copy artifact {instance.id!r} is missing "
                    f"copy_path"
                )
            if instance.kind == "git":
                missing = [
                    f for f in ("repo", "commit", "git_path")
                    if getattr(instance, f) is None
                ]
                if missing:
                    raise ValueError(
                        f"Git artifact {instance.id!r} is missing "
                        f"required fields: {', '.join(missing)}"
                    )
            if instance.produced_by is not None and instance.fixture_id is not None:
                raise ValueError(
                    f"Artifact {instance.id!r} cannot have both "
                    "produced_by and fixture_id"
                )
            self.artifacts[instance.id] = instance

    def _add_execution(self, execution: BlockExecution) -> None:
        """Add a block execution record to the workspace ledger.

        Private. The sanctioned write path for block executions is
        :meth:`record_execution`. This method exists only as the
        shared low-level write inside that canonical path and is
        not meant to be called from outside the workspace; callers
        outside ``Workspace`` should use ``record_execution``.
        Tests may call it for fixture setup but should treat that
        as a deliberate internal-API access.

        Thread-safe: acquires the workspace lock.

        Args:
            execution: The execution record to add.

        Raises:
            ValueError: If an execution with this ID already exists.
        """
        with self._lock:
            if execution.id in self.executions:
                raise ValueError(
                    f"Execution {execution.id!r} already exists "
                    f"in workspace"
                )
            self.executions[execution.id] = execution

    def _add_state_snapshot(self, snapshot: StateSnapshot) -> None:
        """Add a managed state snapshot record to the workspace."""
        with self._lock:
            if snapshot.id in self.state_snapshots:
                raise ValueError(
                    f"State snapshot {snapshot.id!r} already "
                    f"exists in workspace"
                )
            self.state_snapshots[snapshot.id] = snapshot

    def record_execution(
        self, execution: BlockExecution,
    ) -> BlockExecution:
        """Record a block execution and persist the workspace.

        The single canonical write path for ``BlockExecution``
        records.  Validates the substrate invariants, derives the
        canonical ``status`` from ``termination_reason`` (when
        provided), then adds the execution to the in-memory ledger
        and immediately persists the workspace to disk.

        Canonical-mode invariants (enforced when
        ``execution.termination_reason`` is set):

        * ``runner`` must be one of ``"container_one_shot"`` or
          ``"container_persistent"``.
        * ``failure_phase``, when set, must be a known phase from
          :data:`flywheel.runtime.FAILURE_PHASES`.
        * ``status`` is recomputed from ``termination_reason`` via
          :func:`flywheel.termination.derive_status`; any value the
          caller passed is overwritten.
        * Successful executions (``status == "succeeded"``) must
          have no ``failure_phase`` and no ``rejected_outputs``.
        * Non-successful executions must have a ``failure_phase``.

        Deferred callers that have not yet adopted
        ``termination_reason`` may pass ``None``; the record is
        appended as-is and these invariants are skipped.  This
        leniency is temporary while the remaining batteries and
        persistent-runtime specs are rebuilt.

        Args:
            execution: The execution record to record.

        Returns:
            The (possibly status-corrected) execution record, for
            caller convenience.

        Raises:
            ValueError: If an execution with this ID already
                exists, or if a canonical-mode invariant fails.
        """
        if execution.termination_reason is not None:
            execution = self._validate_and_normalize_execution(execution)
        self._add_execution(execution)
        self.save()
        return execution

    def record_invocation(
        self, invocation: BlockInvocation,
    ) -> BlockInvocation:
        """Record a completed or skipped block invocation."""
        with self._lock:
            if invocation.id in self.invocations:
                raise ValueError(
                    f"Invocation {invocation.id!r} already exists "
                    "in workspace"
                )
            if invocation.status not in ("succeeded", "failed"):
                raise ValueError(
                    f"Unknown invocation status {invocation.status!r}"
                )
            self.invocations[invocation.id] = invocation
        self.save()
        return invocation

    @staticmethod
    def _validate_and_normalize_execution(
        execution: BlockExecution,
    ) -> BlockExecution:
        """Apply canonical invariants and derive status."""
        if execution.runner not in (
            "container_one_shot", "container_persistent",
        ):
            raise ValueError(
                f"BlockExecution {execution.id!r}: runner must be "
                f"'container_one_shot' or 'container_persistent'; "
                f"got {execution.runner!r}"
            )
        if execution.state_mode not in ("none", "managed", "unmanaged"):
            raise ValueError(
                f"BlockExecution {execution.id!r}: state_mode must "
                f"be 'none', 'managed', or 'unmanaged'; got "
                f"{execution.state_mode!r}"
            )
        if (
            execution.state_snapshot_id is not None
            and execution.state_mode != "managed"
        ):
            raise ValueError(
                f"BlockExecution {execution.id!r}: state_snapshot_id "
                "requires state_mode 'managed'"
            )
        if (execution.failure_phase is not None
                and execution.failure_phase not in FAILURE_PHASES):
            raise ValueError(
                f"BlockExecution {execution.id!r}: failure_phase "
                f"{execution.failure_phase!r} is not in "
                f"runtime.FAILURE_PHASES"
            )

        all_expected_committed = (
            not execution.rejected_outputs
            and execution.failure_phase is None
        )
        derived = derive_status(
            execution.termination_reason,  # type: ignore[arg-type]
            all_expected_committed=all_expected_committed,
        )
        if execution.status != derived:
            execution = replace(execution, status=derived)

        if derived == "succeeded":
            if execution.failure_phase is not None:
                raise ValueError(
                    f"BlockExecution {execution.id!r}: status "
                    f"'succeeded' must not carry a failure_phase "
                    f"(got {execution.failure_phase!r})"
                )
            if execution.rejected_outputs:
                raise ValueError(
                    f"BlockExecution {execution.id!r}: status "
                    f"'succeeded' must not carry rejected_outputs"
                )
        else:
            if execution.failure_phase is None:
                raise ValueError(
                    f"BlockExecution {execution.id!r}: status "
                    f"{derived!r} requires a failure_phase"
                )
        return execution

    def generate_event_id(self) -> str:
        """Generate a unique lifecycle event ID.

        Returns:
            An ID in the form ``evt_hexstring``, guaranteed unique
            within this workspace.
        """
        while True:
            candidate = f"evt_{self._short_uuid()}"
            if candidate not in self.events:
                return candidate

    def add_event(self, event: LifecycleEvent) -> None:
        """Add a lifecycle event to the workspace.

        Thread-safe: acquires the workspace lock.

        Args:
            event: The lifecycle event to add.

        Raises:
            ValueError: If an event with this ID already exists.
        """
        with self._lock:
            if event.id in self.events:
                raise ValueError(
                    f"Event {event.id!r} already exists in workspace"
                )
            self.events[event.id] = event

    def begin_run(
        self,
        kind: str,
        config_snapshot: dict[str, Any] | None = None,
        lanes: list[str] | None = None,
    ) -> RunRecord:
        """Open a new run and append it to the workspace.

        Thread-safe. Returns the new :class:`RunRecord`.
        :class:`flywheel.artifact.BlockExecution` does not store
        run membership. Status starts as ``"running"``; close
        with :meth:`end_run`.

        Args:
            kind: What opened the run.  Convention:
                ``"pattern:<name>"``.
            config_snapshot: Optional config mapping recorded
                for later inspection (model names, budgets,
                etc.).
            lanes: Optional list of run-scoped artifact lanes.
                Defaults to the implicit default lane.

        Returns:
            The newly-created :class:`RunRecord`.
        """
        with self._lock:
            while True:
                candidate = f"run_{self._short_uuid()}"
                if candidate not in self.runs:
                    break
            record = RunRecord(
                id=candidate,
                kind=kind,
                started_at=datetime.now(UTC),
                status="running",
                config_snapshot=(
                    dict(config_snapshot)
                    if config_snapshot else None),
                lanes=(
                    list(lanes) if lanes is not None else [DEFAULT_LANE]
                ),
            )
            self.runs[candidate] = record
        self.save()
        return record

    def record_run_fixture(
        self,
        run_id: str,
        fixture: RunFixtureRecord,
    ) -> RunRecord:
        """Append one fixture materialization to a running run."""
        with self._lock:
            current = self.runs.get(run_id)
            if current is None:
                raise KeyError(
                    f"run {run_id!r} is not known to this workspace"
                )
            if current.status != "running":
                raise ValueError(
                    f"record_run_fixture: run {run_id!r} is already "
                    f"terminal (status={current.status!r})"
                )
            if fixture.lane not in current.lanes:
                raise ValueError(
                    f"record_run_fixture: lane {fixture.lane!r} is "
                    f"not declared for run {run_id!r}"
                )
            if fixture.artifact_id not in self.artifacts:
                raise ValueError(
                    f"record_run_fixture: artifact "
                    f"{fixture.artifact_id!r} is not known"
                )
            artifact = self.artifacts[fixture.artifact_id]
            if artifact.fixture_id != fixture.id:
                raise ValueError(
                    f"record_run_fixture: artifact "
                    f"{fixture.artifact_id!r} points at fixture "
                    f"{artifact.fixture_id!r}, not {fixture.id!r}"
                )
            if any(
                existing.id == fixture.id
                for existing in current.fixtures
            ):
                raise ValueError(
                    f"record_run_fixture: run {run_id!r} already has "
                    f"a fixture with id {fixture.id!r}"
                )
            if any(
                existing.lane == fixture.lane
                and existing.name == fixture.name
                for existing in current.fixtures
            ):
                raise ValueError(
                    f"record_run_fixture: run {run_id!r} already has "
                    f"a fixture for lane {fixture.lane!r}, "
                    f"name {fixture.name!r}"
                )
            updated = replace(
                current,
                fixtures=[*current.fixtures, fixture],
            )
            self.runs[run_id] = updated
        self.save()
        return updated

    def record_run_step(
        self,
        run_id: str,
        step: RunStepRecord,
    ) -> RunRecord:
        """Append one step result to a running run."""
        with self._lock:
            current = self.runs.get(run_id)
            if current is None:
                raise KeyError(
                    f"run {run_id!r} is not known to this "
                    f"workspace"
                )
            if current.status != "running":
                raise ValueError(
                    f"record_run_step: run {run_id!r} is already "
                    f"terminal (status={current.status!r})"
                )
            if any(existing.name == step.name for existing in current.steps):
                raise ValueError(
                    f"record_run_step: run {run_id!r} already has "
                    f"a step named {step.name!r}"
                )
            updated = replace(
                current,
                steps=[*current.steps, step],
            )
            self.runs[run_id] = updated
        self.save()
        return updated

    def end_run(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
    ) -> RunRecord:
        """Close an open run and update its status.

        Thread-safe.  Replaces the existing :class:`RunRecord`
        with a copy that has ``finished_at`` set to now and
        ``status`` set to the given value.  Typical statuses:
        ``"succeeded"``, ``"failed"``, ``"stopped"``.

        Args:
            run_id: The run to close.
            status: Terminal status.
            error: Optional run-level error summary.

        Returns:
            The updated :class:`RunRecord`.

        Raises:
            KeyError: If ``run_id`` is not known.
            ValueError: If ``status`` is not a terminal value, or
                if the run is already in a terminal state
                (double-close).
        """
        if status not in (
                "succeeded", "failed", "stopped", "interrupted"):
            raise ValueError(
                f"end_run: status {status!r} is not terminal; "
                f"expected one of "
                f"('succeeded', 'failed', 'stopped', 'interrupted')"
            )
        with self._lock:
            current = self.runs.get(run_id)
            if current is None:
                raise KeyError(
                    f"run {run_id!r} is not known to this "
                    f"workspace"
                )
            if current.status != "running":
                raise ValueError(
                    f"end_run: run {run_id!r} is already "
                    f"terminal (status={current.status!r}); "
                    f"double-close rejected"
                )
            updated = replace(
                current,
                finished_at=datetime.now(UTC),
                status=status,
                error=error,
            )
            self.runs[run_id] = updated
        self.save()
        return updated

    def events_for(self, kind: str) -> list[LifecycleEvent]:
        """Return all lifecycle events of a given kind, ordered by timestamp.

        Args:
            kind: The event kind to filter by.

        Returns:
            A list of matching events, oldest first.
        """
        return sorted(
            [e for e in self.events.values() if e.kind == kind],
            key=lambda e: e.timestamp,
        )

    def instances_for(self, artifact_name: str) -> list[ArtifactInstance]:
        """Return all artifact instances for a declaration, ordered by creation time.

        Args:
            artifact_name: The artifact declaration name.

        Returns:
            A list of artifact instances for this declaration, oldest first.
        """
        return sorted(
            [a for a in self.artifacts.values() if a.name == artifact_name],
            key=lambda a: a.created_at,
        )

    def latest_state_snapshot(
        self, lineage_key: str,
    ) -> StateSnapshot | None:
        """Return the latest snapshot by timestamp, then id tie-break."""
        return self._latest_state_snapshot_unlocked(lineage_key)

    def _latest_state_snapshot_unlocked(
        self, lineage_key: str,
    ) -> StateSnapshot | None:
        """Return the latest snapshot; caller owns synchronization."""
        snapshots = [
            snapshot for snapshot in self.state_snapshots.values()
            if snapshot.lineage_key == lineage_key
        ]
        if not snapshots:
            return None
        return max(snapshots, key=lambda s: (s.created_at, s.id))

    def state_snapshot_path(self, snapshot_id: str) -> Path:
        """Return the on-disk directory for a managed state snapshot."""
        snapshot = self.state_snapshots.get(snapshot_id)
        if snapshot is None:
            raise KeyError(snapshot_id)
        return self.path / snapshot.state_path

    def register_state_snapshot(
        self,
        *,
        lineage_key: str,
        source_path: Path,
        produced_by: str,
        compatibility: dict[str, str],
        predecessor_snapshot_id: str | None | object = (
            _PREDECESSOR_UNSPECIFIED
        ),
        persist: bool = True,
    ) -> StateSnapshot:
        """Capture a managed state snapshot into workspace storage.

        State snapshots use a write path separate from artifacts:
        they are not declared artifacts, are not artifact-validator
        inputs, and are not eligible as block input bindings.
        When ``persist`` is ``False``, the snapshot is added to
        the in-memory ledger but not written to ``workspace.yaml``;
        canonical block commit uses this to save the snapshot and
        producing execution row atomically.
        """
        if not lineage_key:
            raise ValueError("State lineage key must be non-empty")
        if not source_path.exists():
            raise FileNotFoundError(
                f"State source path does not exist: {source_path}"
            )
        if not source_path.is_dir():
            raise ValueError(
                f"State source must be a directory; got {source_path}"
            )

        latest = self.latest_state_snapshot(lineage_key)
        expected_predecessor = latest.id if latest is not None else None
        if predecessor_snapshot_id is _PREDECESSOR_UNSPECIFIED:
            predecessor_id = expected_predecessor
        elif predecessor_snapshot_id != expected_predecessor:
            raise ValueError(
                f"State lineage {lineage_key!r} latest snapshot is "
                f"{expected_predecessor!r}, not "
                f"{predecessor_snapshot_id!r}"
            )
        else:
            predecessor_id = predecessor_snapshot_id

        states_dir = self.path / "states"
        states_dir.mkdir(parents=True, exist_ok=True)
        snapshot_id = self.generate_state_snapshot_id()
        staging_dir = Path(tempfile.mkdtemp(
            prefix=f"_staging-{snapshot_id}-", dir=states_dir,
        ))
        try:
            shutil.copytree(
                source_path, staging_dir, dirs_exist_ok=True,
            )
            target_dir = states_dir / snapshot_id
            staging_dir.rename(target_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        snapshot = StateSnapshot(
            id=snapshot_id,
            lineage_key=lineage_key,
            created_at=datetime.now(UTC),
            produced_by=produced_by,
            predecessor_snapshot_id=predecessor_id,
            compatibility=dict(compatibility),
            state_path=f"states/{snapshot_id}",
        )
        try:
            with self._lock:
                latest = self._latest_state_snapshot_unlocked(lineage_key)
                expected_predecessor = (
                    latest.id if latest is not None else None
                )
                if snapshot.predecessor_snapshot_id != expected_predecessor:
                    raise ValueError(
                        f"State lineage {lineage_key!r} latest "
                        f"snapshot is {expected_predecessor!r}, "
                        f"not {snapshot.predecessor_snapshot_id!r}"
                    )
                if snapshot.id in self.state_snapshots:
                    raise ValueError(
                        f"State snapshot {snapshot.id!r} already "
                        "exists in workspace"
                    )
                self.state_snapshots[snapshot.id] = snapshot
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        if persist:
            self.save()
        return snapshot

    def preserve_state_recovery(
        self,
        *,
        execution_id: str,
        source_path: Path,
    ) -> str | None:
        """Best-effort preservation for state bytes that failed capture."""
        if not source_path.exists() or not source_path.is_dir():
            return None
        recovery_dir = self.path / "state_recovery" / execution_id
        target = recovery_dir / "state"
        try:
            recovery_dir.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source_path, target)
        except Exception:
            return None
        return target.relative_to(self.path).as_posix()

    def instance_path(self, instance_id: str) -> Path:
        """Return the on-disk directory for a copy artifact.

        For ``copy`` artifacts the returned path is
        ``<workspace>/artifacts/<copy_path>/``.  For ``git`` artifacts
        this raises — git artifacts are not directory-stored within
        the workspace.

        Args:
            instance_id: The artifact instance ID.

        Raises:
            KeyError: If no instance exists with that ID.
            ValueError: If the instance is a git artifact.
        """
        if instance_id not in self.artifacts:
            raise KeyError(instance_id)
        inst = self.artifacts[instance_id]
        if inst.kind == "git":
            raise ValueError(
                f"Artifact {instance_id!r} is a git artifact and "
                f"has no workspace-local directory"
            )
        if inst.copy_path is None:
            raise ValueError(
                f"Artifact {instance_id!r} is missing copy_path"
            )
        return self.path / "artifacts" / inst.copy_path

    def register_artifact(
        self,
        name: str,
        source_path: Path,
        source: str | None = None,
        *,
        validator_registry: ArtifactValidatorRegistry | None = None,
        declaration: ArtifactDeclaration | None = None,
        supersedes: SupersedesRef | None = None,
        supersedes_reason: str | None = None,
        produced_by: str | None = None,
        fixture_id: str | None = None,
        persist: bool = True,
    ) -> ArtifactInstance:
        """Import an external directory as an artifact instance.

        Stages the source's contents into a flywheel-owned
        directory shaped like the canonical artifact directory
        will be, runs the registered validator (if any) against
        that staging directory, and on success atomically renames
        it into the workspace's artifact storage.  The imported
        artifact binds identically to block-produced artifacts.

        Artifact instances are always directory-shaped, mirroring
        the per-slot output directories block executions write
        into.  ``source_path`` must be a directory; callers that
        want to import a single file must wrap it in a directory
        first (the artifact's structure is then explicit at the
        call site).

        The validator never sees the operator's source path — it
        sees the staged candidate, which is structurally
        identical to what the committed artifact will be.  This
        matches the validator contract used at every other
        finalization site (see :mod:`flywheel.artifact_validator`).

        Undeclared artifacts are rejected: the workspace's
        :attr:`artifact_declarations` is the single source of
        truth for what shapes a workspace knows how to track.
        See ``docs/architecture.md`` § "Undeclared artifacts are
        rejected".

        When ``supersedes`` is provided, the resulting instance
        records a backward lineage pointer at registration time;
        the predecessor must already exist in the workspace
        ledger.  See :class:`flywheel.artifact.SupersedesRef` for
        the two predecessor flavours (accepted artifact vs.
        quarantined slot).  Lineage is provenance / intent, not
        a resolution rule: consumers still pick the latest
        instance by ``created_at``.  Predecessor existence is
        verified before any bytes are staged so a bad reference
        fails cheaply.

        Args:
            name: The artifact declaration name (must be a declared
                copy artifact).
            source_path: Path to the directory whose contents
                become the new artifact instance.  Must be a
                directory; single-file imports are not supported
                (wrap the file in a directory first).
            source: Free-text description of where the artifact came
                from. Defaults to the resolved source path.
            validator_registry: Project-supplied validator registry
                consulted against the staged candidate before
                commit.  When the registered validator (if any) for
                ``name`` raises
                :class:`flywheel.artifact_validator.ArtifactValidationError`,
                the staging directory is removed, no artifact is
                written, and the exception propagates to the
                caller.  Optional: omitted (or ``None``) means
                "no validation".
            declaration: The full artifact declaration the registry
                was registered against.  Forwarded to the
                validator so it can branch on declaration metadata.
                Optional — validators that ignore the declaration
                tolerate ``None``.
            supersedes: Optional backward lineage pointer recorded
                on the resulting instance.  When set, the
                predecessor it references must exist in this
                workspace; same-name lineage is enforced for the
                ``artifact_id`` flavour (the predecessor's
                declaration name must equal ``name``).  ``None``
                for plain registrations.
            supersedes_reason: Optional human-readable description
                of *why* this successor is being registered.
                Recorded verbatim on the instance.  Must be
                ``None`` when ``supersedes`` is ``None``.
            produced_by: Optional execution ID that produced this
                artifact.  Set when the proposal directory came
                from a block execution; ``None`` (the default) for
                operator-driven imports.  Recorded verbatim on
                the instance.
            fixture_id: Optional run fixture ID that materialized
                this artifact.  Set for pattern fixture artifacts.
            persist: Whether to write ``workspace.yaml`` before
                returning.  Canonical block commit passes
                ``False`` so produced artifacts and the producing
                execution row are persisted by the same final save.

        Returns:
            The created ArtifactInstance.

        Raises:
            ValueError: If the name is not declared, is not a copy
                artifact, ``source_path`` is not a directory,
                ``supersedes_reason`` is set without
                ``supersedes``, or a ``supersedes`` predecessor
                cannot be resolved (missing artifact, name
                mismatch, missing execution, or slot not present
                in that execution's ``rejected_outputs``).
            FileNotFoundError: If source_path does not exist.
            flywheel.artifact_validator.ArtifactValidationError:
                If a validator is registered for ``name`` and rejects
                the staged candidate.
        """
        if name not in self.artifact_declarations:
            raise ValueError(
                f"Artifact {name!r} not declared in this workspace"
            )
        if self.artifact_declarations[name] != "copy":
            raise ValueError(
                f"Only copy artifacts can be imported; "
                f"{name!r} is {self.artifact_declarations[name]!r}"
            )
        if not source_path.exists():
            raise FileNotFoundError(
                f"Source path does not exist: {source_path}"
            )
        if not source_path.is_dir():
            raise ValueError(
                f"Artifact source must be a directory; "
                f"{source_path} is not.  Wrap a single file in a "
                f"directory and pass that — artifacts are always "
                f"directory-shaped, mirroring block output slots."
            )
        self._check_supersedes(name, supersedes, supersedes_reason)

        # Stage the source contents into a flywheel-owned
        # directory shaped exactly like the canonical artifact
        # directory will be.  Staging lives under
        # <workspace>/artifacts/ so the eventual move into place
        # is a same-filesystem atomic rename, and the validator
        # (if any) sees the candidate artifact in its final shape
        # — never the operator's source path.
        artifacts_dir = self.path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(
            prefix=f"_staging-{name}-", dir=artifacts_dir,
        ))
        try:
            shutil.copytree(
                source_path, staging_dir, dirs_exist_ok=True,
            )

            if validator_registry is not None:
                validator_registry.validate(
                    name, declaration, staging_dir,
                )

            artifact_id = self.generate_artifact_id(name)
            target_dir = artifacts_dir / artifact_id
            staging_dir.rename(target_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        # Only synthesise an "imported from ..." source for
        # operator-driven imports.  Block-produced artifacts
        # already carry provenance via ``produced_by`` and should
        # not have a synthetic source pointing at the transient
        # proposal directory.
        if source is None and produced_by is None and fixture_id is None:
            source = f"imported from {source_path.resolve()}"

        instance = ArtifactInstance(
            id=artifact_id,
            name=name,
            kind="copy",
            created_at=datetime.now(UTC),
            produced_by=produced_by,
            fixture_id=fixture_id,
            source=source,
            copy_path=artifact_id,
            supersedes=supersedes,
            supersedes_reason=supersedes_reason,
        )
        self._add_artifact(instance)
        if persist:
            self.save()
        return instance

    def register_git_artifact(
        self,
        name: str,
        declaration: ArtifactDeclaration,
        project_root: Path,
    ) -> ArtifactInstance:
        """Resolve a declared git input and register it.

        The canonical write path for git-kind artifact instances.
        Resolves the declaration's repo + path against ``project_root``
        at the current ``HEAD`` commit, constructs an
        :class:`ArtifactInstance` of kind ``git``, persists it via
        :meth:`_add_artifact`, and returns it.

        Git artifacts are project source materials — references to
        committed bytes, not bytes the substrate produces. The
        method therefore runs no validators and never touches the
        quarantine: there is no produced output to reject. The only
        thing being "registered" is the resolved snapshot reference.

        The repo working tree must be clean: a dirty working tree
        means HEAD does not exhaustively describe the bytes the
        block would see, which would silently break provenance.
        Operators must commit (or stash) before running a block
        that consumes a git artifact.

        Args:
            name: The artifact declaration name (must be a declared
                ``git`` artifact in the workspace's template).
            declaration: The matching declaration. Carries the
                ``repo`` (relative to ``project_root``) and ``path``
                (relative to that repo's root) being snapshotted.
            project_root: The project root used to resolve
                ``declaration.repo`` to a concrete repo directory.

        Returns:
            The registered :class:`ArtifactInstance` of kind
            ``git`` pinned to ``HEAD``.

        Raises:
            ValueError: If ``declaration`` is missing ``repo`` or
                ``path`` fields.
            RuntimeError: If the resolved repo has uncommitted
                changes.
            FileNotFoundError: If ``declaration.path`` does not
                exist at ``HEAD`` of the resolved repo.
        """
        if declaration.repo is None or declaration.path is None:
            raise ValueError(
                f"Git artifact {name!r} missing repo or path "
                f"in declaration"
            )

        repo_path = (project_root / declaration.repo).resolve()

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

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = head.stdout.strip()

        artifact_path = repo_path / declaration.path
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Git artifact {name!r} path "
                f"{declaration.path!r} does not exist in repo "
                f"{repo_path}"
            )

        artifact_id = self.generate_artifact_id(name)
        instance = ArtifactInstance(
            id=artifact_id,
            name=name,
            kind="git",
            created_at=datetime.now(UTC),
            produced_by=None,
            repo=str(repo_path),
            commit=commit,
            git_path=declaration.path,
        )
        self._add_artifact(instance)
        self.save()
        return instance

    def _check_supersedes(
        self,
        name: str,
        supersedes: SupersedesRef | None,
        supersedes_reason: str | None,
    ) -> None:
        """Verify that a ``supersedes`` predecessor exists.

        Called from :meth:`register_artifact` before any bytes
        are staged so a bad reference fails cheaply and leaves
        no debris.  The substrate trusts
        :class:`SupersedesRef`'s own one-of invariant; this
        helper only checks ledger-level existence and the
        same-name lineage rule.

        Args:
            name: The new artifact's declaration name (used to
                enforce same-name lineage for the ``artifact_id``
                flavour).
            supersedes: The lineage pointer to verify, or
                ``None`` for plain registrations.
            supersedes_reason: The accompanying reason string;
                checked here only to reject the
                "reason without lineage" combination.

        Raises:
            ValueError: If ``supersedes_reason`` is set without
                ``supersedes``, or if the predecessor referenced
                by ``supersedes`` does not exist (missing
                artifact, name mismatch, missing execution, or
                slot not present in that execution's
                ``rejected_outputs``).
        """
        if supersedes is None:
            if supersedes_reason is not None:
                raise ValueError(
                    "supersedes_reason was provided without a "
                    "supersedes lineage pointer; reason text is "
                    "only meaningful when superseding a "
                    "predecessor."
                )
            return

        if supersedes.artifact_id is not None:
            predecessor = self.artifacts.get(supersedes.artifact_id)
            if predecessor is None:
                raise ValueError(
                    f"supersedes.artifact_id "
                    f"{supersedes.artifact_id!r} does not exist "
                    f"in this workspace"
                )
            if predecessor.name != name:
                raise ValueError(
                    f"supersedes.artifact_id "
                    f"{supersedes.artifact_id!r} is declared "
                    f"{predecessor.name!r}; cannot supersede "
                    f"across artifact names (new instance is "
                    f"{name!r})"
                )
            return

        assert supersedes.rejection is not None  # SupersedesRef invariant
        rej = supersedes.rejection
        execution = self.executions.get(rej.execution_id)
        if execution is None:
            raise ValueError(
                f"supersedes.rejection.execution_id "
                f"{rej.execution_id!r} does not exist in this "
                f"workspace"
            )
        if rej.slot not in execution.rejected_outputs:
            raise ValueError(
                f"execution {rej.execution_id!r} has no rejected "
                f"output for slot {rej.slot!r}; available "
                f"rejected slots: "
                f"{sorted(execution.rejected_outputs)!r}"
            )

    @classmethod
    def create(cls, name: str, template: Template, foundry_dir: Path) -> Workspace:
        """Create a new workspace directory from a template.

        Creates the workspace directory, resolves git artifact baselines,
        and writes workspace.yaml metadata.

        Args:
            name: Workspace name (letters, digits, hyphens, underscores).
            template: The template defining artifacts and blocks.
            foundry_dir: Path to the foundry directory.

        Returns:
            The created Workspace instance.

        Raises:
            FileExistsError: If the workspace already exists.
            ValueError: If the name is invalid or git artifact
                declarations are incomplete.
            RuntimeError: If a git repo has uncommitted changes.
            FileNotFoundError: If a git artifact path does not exist
                in the repo.
        """
        _validate_name(name, "Workspace")
        ws_path = foundry_dir / "workspaces" / name
        if ws_path.exists():
            raise FileExistsError(f"Workspace already exists: {ws_path}")

        ws_path.mkdir(parents=True)
        (ws_path / "artifacts").mkdir()
        (ws_path / "states").mkdir()

        try:
            project_root = foundry_dir.parent
            declarations: dict[str, str] = {}
            artifacts: dict[str, ArtifactInstance] = {}
            now = datetime.now(UTC)

            for decl in template.artifacts:
                declarations[decl.name] = decl.kind

                if decl.kind == "git":
                    if decl.repo is None:
                        raise ValueError(
                            f"Git artifact {decl.name!r} missing repo"
                        )
                    if decl.path is None:
                        raise ValueError(
                            f"Git artifact {decl.name!r} missing path"
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
                            f"Git repo {repo_path} has uncommitted "
                            f"changes. Commit or stash before creating "
                            f"a workspace."
                        )

                    artifact_path = repo_path / decl.path
                    if not artifact_path.exists():
                        raise FileNotFoundError(
                            f"Git artifact {decl.name!r} path "
                            f"{decl.path!r} does not exist in "
                            f"repo {repo_path}"
                        )

                    result = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    commit = result.stdout.strip()

                    baseline_id = f"{decl.name}@baseline"
                    artifacts[baseline_id] = ArtifactInstance(
                        id=baseline_id,
                        name=decl.name,
                        kind="git",
                        created_at=now,
                        produced_by=None,
                        repo=str(repo_path),
                        commit=commit,
                        git_path=decl.path,
                    )

            ws = cls(
                name=name,
                path=ws_path,
                template_name=template.name,
                created_at=now,
                artifact_declarations=declarations,
                artifacts=artifacts,
            )
            ws.save()
            return ws
        except Exception:
            shutil.rmtree(ws_path, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path: Path) -> Workspace:
        """Load an existing workspace from its workspace.yaml.

        Args:
            path: Path to the workspace directory.

        Returns:
            The loaded Workspace instance.

        Raises:
            ValueError: If the YAML contains invalid data.
        """
        yaml_path = path / "workspace.yaml"
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        declarations = data.get("artifact_declarations", {})

        artifacts: dict[str, ArtifactInstance] = {}
        for aid, entry in data.get("artifacts", {}).items():
            artifacts[aid] = ArtifactInstance(
                id=aid,
                name=entry["name"],
                kind=entry["kind"],
                created_at=datetime.fromisoformat(entry["created_at"]),
                produced_by=entry.get("produced_by"),
                fixture_id=entry.get("fixture_id"),
                source=entry.get("source"),
                copy_path=entry.get("copy_path"),
                repo=entry.get("repo"),
                commit=entry.get("commit"),
                git_path=entry.get("git_path"),
                supersedes=_supersedes_from_yaml(entry.get("supersedes")),
                supersedes_reason=entry.get("supersedes_reason"),
            )

        executions: dict[str, BlockExecution] = {}
        for eid, entry in data.get("executions", {}).items():
            finished = entry.get("finished_at")
            executions[eid] = BlockExecution(
                id=eid,
                block_name=entry["block_name"],
                started_at=datetime.fromisoformat(entry["started_at"]),
                finished_at=(
                    datetime.fromisoformat(finished) if finished else None
                ),
                status=entry.get("status", "failed"),
                input_bindings=entry.get("input_bindings", {}),
                output_bindings=entry.get("output_bindings", {}),
                exit_code=entry.get("exit_code"),
                elapsed_s=entry.get("elapsed_s"),
                image=entry.get("image"),
                runner=entry["runner"],
                error=entry.get("error"),
                failure_phase=entry.get("failure_phase"),
                rejected_outputs=_rejected_outputs_from_yaml(
                    entry.get("rejected_outputs")),
                termination_reason=entry.get("termination_reason"),
                state_mode=entry.get("state_mode", "none"),
                state_snapshot_id=entry.get("state_snapshot_id"),
                invoking_execution_id=entry.get("invoking_execution_id"),
            )

        invocations: dict[str, BlockInvocation] = {}
        for iid, entry in data.get("invocations", {}).items():
            invocations[iid] = BlockInvocation(
                id=iid,
                invoking_execution_id=entry["invoking_execution_id"],
                termination_reason=entry["termination_reason"],
                invoked_block_name=entry["invoked_block_name"],
                invoked_at=datetime.fromisoformat(entry["invoked_at"]),
                status=entry["status"],
                invoked_execution_id=entry.get("invoked_execution_id"),
                input_bindings=entry.get("input_bindings", {}),
                args=entry.get("args", []),
                error=entry.get("error"),
            )

        events: dict[str, LifecycleEvent] = {}
        for evid, entry in data.get("events", {}).items():
            events[evid] = LifecycleEvent(
                id=evid,
                kind=entry["kind"],
                timestamp=datetime.fromisoformat(entry["timestamp"]),
                execution_id=entry.get("execution_id"),
                detail=entry.get("detail", {}),
            )

        runs: dict[str, RunRecord] = {}
        for rid, entry in data.get("runs", {}).items():
            finished = entry.get("finished_at")
            runs[rid] = RunRecord(
                id=rid,
                kind=entry["kind"],
                started_at=datetime.fromisoformat(
                    entry["started_at"]),
                finished_at=(
                    datetime.fromisoformat(finished)
                    if finished else None),
                status=entry.get("status", "running"),
                config_snapshot=entry.get("config_snapshot"),
                lanes=list(entry.get("lanes", [DEFAULT_LANE])),
                fixtures=_run_fixtures_from_yaml(
                    entry.get("fixtures")),
                steps=_run_steps_from_yaml(entry.get("steps")),
                error=entry.get("error"),
            )

        return cls(
            name=data["name"],
            path=path,
            template_name=data["template_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            artifact_declarations=declarations,
            artifacts=artifacts,
            executions=executions,
            invocations=invocations,
            events=events,
            runs=runs,
            state_snapshots=_state_snapshots_from_yaml(
                data.get("state_snapshots")),
        )

    def save(self) -> None:
        """Write workspace.yaml to the workspace directory.

        Thread-safe: acquires the workspace lock.  Uses atomic
        write (write to temp file, then rename) to prevent torn
        YAML from concurrent saves or crashes.
        """
        with self._lock:
            serialized_artifacts = {}
            for aid, inst in self.artifacts.items():
                entry: dict = {
                    "name": inst.name,
                    "kind": inst.kind,
                    "created_at": inst.created_at.isoformat(),
                }
                if inst.produced_by is not None:
                    entry["produced_by"] = inst.produced_by
                if inst.fixture_id is not None:
                    entry["fixture_id"] = inst.fixture_id
                if inst.source is not None:
                    entry["source"] = inst.source
                if inst.kind == "copy" and inst.copy_path is not None:
                    entry["copy_path"] = inst.copy_path
                if inst.kind == "git":
                    entry["repo"] = inst.repo
                    entry["commit"] = inst.commit
                    entry["git_path"] = inst.git_path
                if inst.supersedes is not None:
                    entry["supersedes"] = _supersedes_to_yaml(inst.supersedes)
                if inst.supersedes_reason is not None:
                    entry["supersedes_reason"] = inst.supersedes_reason
                serialized_artifacts[aid] = entry

            serialized_executions = {}
            for eid, ex in self.executions.items():
                entry = {
                    "block_name": ex.block_name,
                    "started_at": ex.started_at.isoformat(),
                    "status": ex.status,
                    "input_bindings": ex.input_bindings,
                    "output_bindings": ex.output_bindings,
                }
                if ex.finished_at is not None:
                    entry["finished_at"] = ex.finished_at.isoformat()
                if ex.exit_code is not None:
                    entry["exit_code"] = ex.exit_code
                if ex.elapsed_s is not None:
                    entry["elapsed_s"] = ex.elapsed_s
                if ex.image is not None:
                    entry["image"] = ex.image
                entry["runner"] = ex.runner
                if ex.error is not None:
                    entry["error"] = ex.error
                if ex.failure_phase is not None:
                    entry["failure_phase"] = ex.failure_phase
                if ex.rejected_outputs:
                    entry["rejected_outputs"] = (
                        _rejected_outputs_to_yaml(
                            ex.rejected_outputs))
                if ex.termination_reason is not None:
                    entry["termination_reason"] = ex.termination_reason
                if ex.state_mode != "none":
                    entry["state_mode"] = ex.state_mode
                if ex.state_snapshot_id is not None:
                    entry["state_snapshot_id"] = ex.state_snapshot_id
                if ex.invoking_execution_id is not None:
                    entry["invoking_execution_id"] = ex.invoking_execution_id
                serialized_executions[eid] = entry

            serialized_invocations = {}
            for iid, inv in self.invocations.items():
                entry = {
                    "invoking_execution_id": inv.invoking_execution_id,
                    "termination_reason": inv.termination_reason,
                    "invoked_block_name": inv.invoked_block_name,
                    "invoked_at": inv.invoked_at.isoformat(),
                    "status": inv.status,
                    "input_bindings": inv.input_bindings,
                    "args": inv.args,
                }
                if inv.invoked_execution_id is not None:
                    entry["invoked_execution_id"] = inv.invoked_execution_id
                if inv.error is not None:
                    entry["error"] = inv.error
                serialized_invocations[iid] = entry

            serialized_events = {}
            for evid, ev in self.events.items():
                entry = {
                    "kind": ev.kind,
                    "timestamp": ev.timestamp.isoformat(),
                }
                if ev.execution_id is not None:
                    entry["execution_id"] = ev.execution_id
                if ev.detail:
                    entry["detail"] = ev.detail
                serialized_events[evid] = entry

            serialized_runs = {}
            for rid, run in self.runs.items():
                entry = {
                    "kind": run.kind,
                    "started_at": run.started_at.isoformat(),
                    "status": run.status,
                }
                if run.finished_at is not None:
                    entry["finished_at"] = (
                        run.finished_at.isoformat())
                if run.config_snapshot is not None:
                    entry["config_snapshot"] = run.config_snapshot
                if run.lanes != [DEFAULT_LANE]:
                    entry["lanes"] = list(run.lanes)
                if run.fixtures:
                    entry["fixtures"] = _run_fixtures_to_yaml(
                        run.fixtures)
                if run.steps:
                    entry["steps"] = [
                        _run_step_to_yaml(step) for step in run.steps
                    ]
                if run.error is not None:
                    entry["error"] = run.error
                serialized_runs[rid] = entry

            data: dict = {
                "name": self.name,
                "template_name": self.template_name,
                "created_at": self.created_at.isoformat(),
                "artifact_declarations": self.artifact_declarations,
                "artifacts": serialized_artifacts,
                "executions": serialized_executions,
            }
            if serialized_invocations:
                data["invocations"] = serialized_invocations
            if serialized_events:
                data["events"] = serialized_events
            if serialized_runs:
                data["runs"] = serialized_runs
            if self.state_snapshots:
                data["state_snapshots"] = {
                    sid: _state_snapshot_to_yaml(snapshot)
                    for sid, snapshot in self.state_snapshots.items()
                }

            yaml_path = self.path / "workspace.yaml"
            tmp_path = yaml_path.with_suffix(".yaml.tmp")
            with open(tmp_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            os.replace(tmp_path, yaml_path)
