"""Workspace creation, persistence, and artifact/execution tracking.

A workspace is a directory inside a project's foundry folder,
created from a template. It accumulates artifact instances and
block execution records over its lifetime, forming a complete
provenance graph.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
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
    LifecycleEvent,
    RunRecord,
)
from flywheel.template import Template
from flywheel.validation import validate_name as _validate_name


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
    events: dict[str, LifecycleEvent] = field(default_factory=dict)
    runs: dict[str, RunRecord] = field(default_factory=dict)
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

    def add_artifact(self, instance: ArtifactInstance) -> None:
        """Add an artifact instance to the workspace.

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
            if (instance.kind == "incremental"
                    and instance.copy_path is None):
                raise ValueError(
                    f"Incremental artifact {instance.id!r} is "
                    f"missing copy_path"
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
            self.artifacts[instance.id] = instance

    def add_execution(self, execution: BlockExecution) -> None:
        """Add a block execution record to the workspace.

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
    ) -> RunRecord:
        """Open a new run and append it to the workspace.

        Thread-safe.  Returns the new :class:`RunRecord`; the
        caller stores the id and tags executions it drives.
        Status starts as ``"running"``; close with
        :meth:`end_run`.

        Args:
            kind: What opened the run.  Convention:
                ``"pattern:<name>"``.
            config_snapshot: Optional config mapping recorded
                for later inspection (model names, budgets,
                etc.).

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
            )
            self.runs[candidate] = record
            return record

    def end_run(self, run_id: str, status: str) -> RunRecord:
        """Close an open run and update its status.

        Thread-safe.  Replaces the existing :class:`RunRecord`
        with a copy that has ``finished_at`` set to now and
        ``status`` set to the given value.  Typical statuses:
        ``"succeeded"``, ``"failed"``, ``"stopped"``.

        Args:
            run_id: The run to close.
            status: Terminal status.

        Returns:
            The updated :class:`RunRecord`.

        Raises:
            KeyError: If ``run_id`` is not known.
            ValueError: If ``status`` is not a terminal value, or
                if the run is already in a terminal state
                (double-close).
        """
        if status not in (
                "succeeded", "failed", "stopped"):
            raise ValueError(
                f"end_run: status {status!r} is not terminal; "
                f"expected one of "
                f"('succeeded', 'failed', 'stopped')"
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
            )
            self.runs[run_id] = updated
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

    def instance_path(self, instance_id: str) -> Path:
        """Return the on-disk directory for a copy or incremental artifact.

        For ``copy`` and ``incremental`` artifacts the returned path is
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

    def register_incremental_artifact(
        self,
        name: str,
        *,
        produced_by: str | None = None,
        source: str | None = None,
    ) -> ArtifactInstance:
        """Allocate a new incremental artifact instance.

        Creates the artifact directory, an empty ``entries.jsonl``
        file, and a registered :class:`ArtifactInstance` of kind
        ``incremental``.  Subsequent appenders should use
        :meth:`append_to_incremental` to add entries.

        Args:
            name: The artifact declaration name (must be declared
                with ``kind: incremental``).
            produced_by: Execution ID of the first appender, if
                known.  May be ``None`` if the instance is created
                ahead of any appends (e.g., during workspace
                bootstrap).
            source: Free-text provenance description.

        Returns:
            The created :class:`ArtifactInstance`.

        Raises:
            ValueError: If the name is not declared as an
                incremental artifact.
        """
        if name not in self.artifact_declarations:
            raise ValueError(
                f"Artifact {name!r} not declared in this workspace"
            )
        if self.artifact_declarations[name] != "incremental":
            raise ValueError(
                f"Artifact {name!r} is declared as "
                f"{self.artifact_declarations[name]!r}; "
                f"register_incremental_artifact requires "
                f"'incremental'"
            )

        artifact_id = self.generate_artifact_id(name)
        target_dir = self.path / "artifacts" / artifact_id
        target_dir.mkdir(parents=True)
        try:
            (target_dir / "entries.jsonl").touch()
            instance = ArtifactInstance(
                id=artifact_id,
                name=name,
                kind="incremental",
                created_at=datetime.now(UTC),
                produced_by=produced_by,
                source=source,
                copy_path=artifact_id,
            )
            self.add_artifact(instance)
            self.save()
            return instance
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise

    def latest_incremental_instance(
        self, name: str,
    ) -> ArtifactInstance | None:
        """Return the most recent incremental instance of *name*, or None.

        v1 expects at most one incremental instance per declaration
        per workspace; this helper exists so callers don't have to
        enforce that themselves and so a future v2 with multiple
        instances can change the resolution policy here.

        Args:
            name: The artifact declaration name.

        Returns:
            The newest incremental instance, or ``None`` if no
            incremental instances exist for this name.
        """
        if name not in self.artifact_declarations:
            raise ValueError(
                f"Artifact {name!r} not declared in this workspace"
            )
        if self.artifact_declarations[name] != "incremental":
            raise ValueError(
                f"Artifact {name!r} is declared as "
                f"{self.artifact_declarations[name]!r}; "
                f"latest_incremental_instance requires 'incremental'"
            )
        candidates = [
            a for a in self.artifacts.values()
            if a.name == name and a.kind == "incremental"
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.created_at)

    def append_to_incremental(
        self,
        instance_id: str,
        entries: list[Any],
    ) -> None:
        """Append one or more entries to an incremental artifact.

        Each entry is JSON-encoded and written as one line to
        ``entries.jsonl``.  Holds the workspace lock for the
        duration of the append; concurrent appenders within the
        same process serialize against each other.  v1 assumes
        cross-process appenders will not collide (block executions
        producing an incremental are expected to be serialized by
        the workspace owner).

        Args:
            instance_id: The incremental artifact instance ID.
            entries: List of JSON-encodable values to append.
                Empty list is a no-op.

        Raises:
            KeyError: If no instance exists with that ID.
            ValueError: If the instance is not an incremental
                artifact.
        """
        if not entries:
            return
        if instance_id not in self.artifacts:
            raise KeyError(instance_id)
        inst = self.artifacts[instance_id]
        if inst.kind != "incremental":
            raise ValueError(
                f"Artifact {instance_id!r} has kind {inst.kind!r}; "
                f"append_to_incremental requires 'incremental'"
            )
        if inst.copy_path is None:
            raise ValueError(
                f"Incremental artifact {instance_id!r} is missing "
                f"copy_path"
            )
        entries_path = (
            self.path / "artifacts" / inst.copy_path / "entries.jsonl"
        )
        with self._lock, open(entries_path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(
                    json.dumps(entry, separators=(",", ":")) + "\n"
                )

    def read_incremental_entries(
        self, instance_id: str,
    ) -> list[Any]:
        """Read all entries from an incremental artifact in append order.

        Convenience reader for in-process callers; container blocks
        read ``/input/<name>/entries.jsonl`` directly.

        Args:
            instance_id: The incremental artifact instance ID.

        Returns:
            List of JSON values in append order.

        Raises:
            KeyError: If no instance exists with that ID.
            ValueError: If the instance is not an incremental
                artifact.
        """
        if instance_id not in self.artifacts:
            raise KeyError(instance_id)
        inst = self.artifacts[instance_id]
        if inst.kind != "incremental":
            raise ValueError(
                f"Artifact {instance_id!r} has kind {inst.kind!r}; "
                f"read_incremental_entries requires 'incremental'"
            )
        if inst.copy_path is None:
            raise ValueError(
                f"Incremental artifact {instance_id!r} is missing "
                f"copy_path"
            )
        entries_path = (
            self.path / "artifacts" / inst.copy_path / "entries.jsonl"
        )
        if not entries_path.exists():
            return []
        with open(entries_path, encoding="utf-8") as f:
            return [
                json.loads(line)
                for line in f
                if line.strip()
            ]

    def register_artifact(
        self,
        name: str,
        source_path: Path,
        source: str | None = None,
    ) -> ArtifactInstance:
        """Import an external file or directory as an artifact instance.

        Copies the source into the workspace's artifact storage and
        records it as a new artifact instance with provenance metadata.
        The imported artifact binds identically to block-produced
        artifacts.

        Args:
            name: The artifact declaration name (must be a declared
                copy artifact).
            source_path: Path to the file or directory to import.
            source: Free-text description of where the artifact came
                from. Defaults to the resolved source path.

        Returns:
            The created ArtifactInstance.

        Raises:
            ValueError: If the name is not declared or is not a copy
                artifact.
            FileNotFoundError: If source_path does not exist.
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

        artifact_id = self.generate_artifact_id(name)
        target_dir = self.path / "artifacts" / artifact_id
        target_dir.mkdir(parents=True)

        try:
            if source_path.is_file():
                shutil.copy2(source_path, target_dir / source_path.name)
            else:
                shutil.copytree(
                    source_path, target_dir, dirs_exist_ok=True
                )

            if source is None:
                source = f"imported from {source_path.resolve()}"

            instance = ArtifactInstance(
                id=artifact_id,
                name=name,
                kind="copy",
                created_at=datetime.now(UTC),
                produced_by=None,
                source=source,
                copy_path=artifact_id,
            )
            self.add_artifact(instance)
            self.save()
            return instance
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise

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
                source=entry.get("source"),
                copy_path=entry.get("copy_path"),
                repo=entry.get("repo"),
                commit=entry.get("commit"),
                git_path=entry.get("git_path"),
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
                stop_reason=entry.get("stop_reason"),
                predecessor_id=entry.get("predecessor_id"),
                parent_execution_id=entry.get("parent_execution_id"),
                runner=entry.get("runner"),
                caller=entry.get("caller"),
                params=entry.get("params"),
                error=entry.get("error"),
                synthetic=entry.get("synthetic", False),
                halt_directive=entry.get("halt_directive"),
                post_check_error=entry.get("post_check_error"),
                agent_workspace_dir=entry.get(
                    "agent_workspace_dir"),
                state_dir=entry.get("state_dir"),
                failure_phase=entry.get("failure_phase"),
                state_lineage_id=entry.get("state_lineage_id"),
                run_id=entry.get("run_id"),
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
            )

        return cls(
            name=data["name"],
            path=path,
            template_name=data["template_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            artifact_declarations=declarations,
            artifacts=artifacts,
            executions=executions,
            events=events,
            runs=runs,
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
                if inst.source is not None:
                    entry["source"] = inst.source
                if (inst.kind in ("copy", "incremental")
                        and inst.copy_path is not None):
                    entry["copy_path"] = inst.copy_path
                if inst.kind == "git":
                    entry["repo"] = inst.repo
                    entry["commit"] = inst.commit
                    entry["git_path"] = inst.git_path
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
                if ex.stop_reason is not None:
                    entry["stop_reason"] = ex.stop_reason
                if ex.predecessor_id is not None:
                    entry["predecessor_id"] = ex.predecessor_id
                if ex.parent_execution_id is not None:
                    entry["parent_execution_id"] = ex.parent_execution_id
                if ex.runner is not None:
                    entry["runner"] = ex.runner
                if ex.caller is not None:
                    entry["caller"] = ex.caller
                if ex.params is not None:
                    entry["params"] = ex.params
                if ex.error is not None:
                    entry["error"] = ex.error
                if ex.synthetic:
                    entry["synthetic"] = True
                if ex.halt_directive is not None:
                    entry["halt_directive"] = ex.halt_directive
                if ex.post_check_error is not None:
                    entry["post_check_error"] = ex.post_check_error
                if ex.agent_workspace_dir is not None:
                    entry["agent_workspace_dir"] = (
                        ex.agent_workspace_dir)
                if ex.state_dir is not None:
                    entry["state_dir"] = ex.state_dir
                if ex.failure_phase is not None:
                    entry["failure_phase"] = ex.failure_phase
                if ex.state_lineage_id is not None:
                    entry["state_lineage_id"] = ex.state_lineage_id
                if ex.run_id is not None:
                    entry["run_id"] = ex.run_id
                serialized_executions[eid] = entry

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
                serialized_runs[rid] = entry

            data: dict = {
                "name": self.name,
                "template_name": self.template_name,
                "created_at": self.created_at.isoformat(),
                "artifact_declarations": self.artifact_declarations,
                "artifacts": serialized_artifacts,
                "executions": serialized_executions,
            }
            if serialized_events:
                data["events"] = serialized_events
            if serialized_runs:
                data["runs"] = serialized_runs

            yaml_path = self.path / "workspace.yaml"
            tmp_path = yaml_path.with_suffix(".yaml.tmp")
            with open(tmp_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            os.replace(tmp_path, yaml_path)
