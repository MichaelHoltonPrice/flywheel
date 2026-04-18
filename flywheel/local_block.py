"""In-process recorder for full-stop block executions.

Under the full-stop nested-block model
(``flywheel.docs.architecture.md`` §"Nested block executions from
agents"), every nested block invocation is a real execution
boundary.  The agent stops, the host runs the block synchronously
in its own Python process, then the agent restarts with the
block's output spliced into its session.

This module is the recorder side of that handoff.  Its three
defining structural properties:

1. **No HTTP round-trip.**  The host-side block runner shares a
   Python process with the workspace, so it writes artifacts and
   ledger rows directly under the workspace's own lock.  No
   loopback HTTP service runs.
2. **No ``"running"`` row.**  Nothing needs to remember anything
   across an in-process boundary, so recorded executions go
   straight from non-existent → ``"succeeded"`` or ``"failed"``
   and the orphaned-``running`` class of bug becomes structurally
   impossible.
3. **Blocks write files; the recorder registers artifacts.**
   Block bodies write files under
   :meth:`LocalExecutionContext.output_dir` (a per-execution
   ephemeral system tempdir).  After the body exits, the
   recorder walks each declared output directory and registers
   artifacts from what's on disk.  Blocks never call an
   "emit-artifact" API, and they never write directly into the
   workspace's canonical artifact store.

Halt directives produced by post-execution checks are queued on
the recorder rather than served over ``GET /halt``; the
host-side handoff loop calls :meth:`LocalBlockRecorder.drain_halts`
between handoff cycles and refuses to relaunch the agent when a
relevant directive is present.

The class is **single-process, single-thread within a recorder
instance**: the recorder mutates the workspace directly under
the workspace's own lock, and one recorder is owned by one
host-side block runner per agent run.  Reuse across runs is fine
as long as the workspace is the same.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flywheel.artifact import ArtifactInstance, BlockExecution
from flywheel.input_staging import (
    cleanup_staged_inputs,
    stage_artifact_instances,
)
from flywheel.post_check import (
    HaltDirective,
    PostCheckCallable,
    PostCheckContext,
)
from flywheel.template import BlockDefinition, Template
from flywheel.workspace import Workspace


class LocalBlockError(RuntimeError):
    """Raised when the recorder rejects a ``begin`` request.

    Carries an ``error_type`` string identifying the rejection
    class plus a human-readable message.  Used for pre-body
    failures (unknown block, missing required input).

    Body failures (exceptions raised by the wrapped block code)
    propagate out of the ``begin`` context manager unchanged —
    those are the caller's domain, not the recorder's.

    Attributes:
        error_type: One of ``"unknown_block"``, ``"missing_input"``.
        execution_id: ``None`` because no row is written for
            rejected requests.  Present on the exception only so
            callers that want to log a stable shape can do so.
    """

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        execution_id: str | None = None,
    ) -> None:
        """Initialize the error with classification metadata."""
        super().__init__(message)
        self.error_type = error_type
        self.execution_id = execution_id


@dataclass
class LocalExecutionContext:
    """Per-execution context yielded by :meth:`LocalBlockRecorder.begin`.

    Carries everything a block body needs to read its inputs,
    write its outputs, and stage scratch state.  The recorder
    walks the per-output directories on the way out and registers
    one artifact per declared output that has any contents.

    Attributes:
        execution_id: Workspace-allocated ledger ID for this
            execution.  Stable for the life of the ``with`` block
            and persists on the resulting row.
        block_name: The block this execution corresponds to.
        input_bindings: ``{slot_name: artifact_id}`` resolved at
            ``begin`` time from the workspace's latest instances.
        input_paths: ``{slot_name: absolute path on host}`` to the
            artifact directory.  These paths point at fresh
            per-execution staging tempdirs populated by
            :func:`flywheel.input_staging.stage_artifact_instances`,
            *not* at the workspace's canonical artifact store.
            Valid for read-only use during the body; the
            recorder cleans them up after the body exits.
        scratch_dir: Per-execution scratch directory under
            ``<workspace>/execution_scratch/<execution_id>``.
            Cleaned up by the recorder on exit (success or
            failure).
        params: The params dict the caller supplied to ``begin``,
            echoed back verbatim.
        parent_execution_id: ID of the runner that invoked this
            execution, when supplied to ``begin``.
        output_root: Per-execution ephemeral system tempdir.
            Each declared output slot has a pre-created
            subdirectory at ``<output_root>/<slot_name>/`` that
            the body writes files into.  Use
            :meth:`output_dir` rather than constructing the path
            by hand.  Cleaned up after successful registration;
            preserved on failure for debugging (the path is
            stamped into the ``BlockExecution.error`` text).
        output_bindings: Populated by the recorder *after* the
            ``with`` block exits successfully, with the
            ``{slot_name: artifact_id}`` map of newly-registered
            artifacts.  Empty if the body raised or no outputs
            were registered.
    """

    execution_id: str
    block_name: str
    input_bindings: dict[str, str]
    input_paths: dict[str, str]
    scratch_dir: str
    params: dict[str, Any]
    parent_execution_id: str | None
    output_root: Path | None = None
    _output_dirs: dict[str, Path] = field(default_factory=dict)
    output_bindings: dict[str, str] = field(default_factory=dict)

    def output_dir(self, name: str) -> Path:
        """Return the per-output directory for slot *name*.

        The directory is created by the recorder before the body
        runs, so callers can write into it freely.  For copy
        artifacts, the body writes one or more files; on
        successful exit, the recorder copies the directory's
        contents into ``<workspace>/artifacts/<artifact_id>/``
        and registers a copy artifact.  For incremental
        artifacts, the body writes a file named ``entries.jsonl``
        with one JSON value per line; the recorder appends those
        entries to the workspace's canonical incremental
        artifact instance for *name* (creating the instance if
        none exists yet).

        Args:
            name: The output slot name as declared by the block.

        Raises:
            KeyError: If *name* is not a declared output of this
                block.
        """
        if name not in self._output_dirs:
            raise KeyError(
                f"{name!r} is not a declared output of block "
                f"{self.block_name!r}; declared: "
                f"{sorted(self._output_dirs)}"
            )
        return self._output_dirs[name]


@dataclass
class _QueuedHalt:
    """Internal record of a halt directive queued by a post-check.

    Stored in the recorder's halt queue and drained by the
    host-side handoff loop between cycles.  Carries enough
    context for the loop to decide whether the directive applies
    to the agent run currently in progress.
    """

    scope: str
    reason: str
    block: str
    execution_id: str
    parent_execution_id: str | None


class LocalBlockRecorder:
    """In-process recorder for tool-triggered block executions.

    One instance is owned by the host-side block runner that
    services the agent's handoffs for one run; reuse across runs
    is fine when the workspace is shared.  Operationally a thin
    state machine:

    1. :meth:`begin` resolves the block's inputs against the
       workspace's latest instances, stages each one into a
       per-mount tempdir (see
       :func:`flywheel.input_staging.stage_artifact_instances`),
       allocates an ephemeral output root with one
       sub-directory per declared output, and yields a context.
    2. The block runner runs the side-effecting work inside the
       ``with`` block, writing files into
       :meth:`LocalExecutionContext.output_dir`.  For
       ``incremental`` outputs the runner appends one or more
       JSON entries to ``<output_dir>/entries.jsonl``.
    3. On clean exit the recorder registers a fresh ``copy``
       artifact for each non-empty copy output directory and
       appends each ``incremental`` output's entries to the
       canonical incremental instance, then builds a
       ``"succeeded"`` :class:`flywheel.artifact.BlockExecution`
       and runs the block's post-execution check (if
       configured).
    4. On exception the recorder writes no artifacts, builds a
       ``"failed"`` row, runs the post-check with that row as
       context, and re-raises.

    Halts produced by post-checks are appended to
    :attr:`halt_queue`; the host-side handoff loop calls
    :meth:`drain_halts` between cycles to decide whether to
    refuse the next relaunch.

    Attributes:
        workspace: The flywheel workspace that owns the artifacts
            and ledger this recorder writes to.
        template: Template with the block definitions used to
            resolve inputs and validate outputs.
        post_checks: Mapping from block name to the post-execution
            callable to invoke after each row finalizes.  Empty
            mapping means no post-checks are wired up.
        halt_queue: Mutable list of queued halt directives.
            Drained by :meth:`drain_halts`; the recorder never
            removes entries on its own.
    """

    def __init__(
        self,
        *,
        workspace: Workspace,
        template: Template,
        post_checks: dict[str, PostCheckCallable] | None = None,
    ) -> None:
        """Bind the recorder to a workspace, template, and post-checks."""
        self.workspace = workspace
        self.template = template
        self.post_checks: dict[str, PostCheckCallable] = (
            dict(post_checks) if post_checks else {})
        self.halt_queue: list[_QueuedHalt] = []

    @contextmanager
    def begin(
        self,
        block: str,
        params: dict[str, Any] | None = None,
        caller: dict[str, Any] | None = None,
        runner: str | None = None,
        parent_execution_id: str | None = None,
    ) -> Iterator[LocalExecutionContext]:
        """Open a logical block execution as a context manager.

        Args:
            block: Block name as declared in the template.
            params: Function-argument parameters for ledger
                provenance (e.g., ``{"action_id": 6}``).  Echoed
                back through the yielded context.
            caller: Identifies the source of the call (e.g.,
                ``{"mcp_server": "arc", "tool": "take_action"}``).
                Recorded on the ledger row for downstream
                analyses keyed on caller.
            runner: Caller-declared physical runner type (e.g.,
                ``"lifecycle"``).  Recorded as-is.
            parent_execution_id: Execution ID of the runner that
                invoked this block.  For agent-triggered blocks
                this is the agent's own execution row.

        Yields:
            A :class:`LocalExecutionContext` for populating
            outputs.

        Raises:
            LocalBlockError: When the block is not in the
                template (``"unknown_block"``) or a required
                input slot has no registered instances
                (``"missing_input"``).
            Exception: Any exception raised by the body propagates
                out of the context manager after the recorder
                writes a ``"failed"`` row and runs the post-check.
        """
        block_def = self._find_block(block)
        if block_def is None:
            raise LocalBlockError(
                f"Block {block!r} not found in template",
                error_type="unknown_block",
            )

        try:
            input_bindings = self._resolve_input_bindings(block_def)
        except ValueError as exc:
            raise LocalBlockError(
                str(exc), error_type="missing_input") from exc

        execution_id = self.workspace.generate_execution_id()
        scratch_dir = (
            self.workspace.path / "execution_scratch" / execution_id
        )
        scratch_dir.mkdir(parents=True, exist_ok=True)

        output_root = Path(tempfile.mkdtemp(
            prefix=f"flywheel-exec-{execution_id}-",
        ))
        output_dirs: dict[str, Path] = {}
        for slot in block_def.outputs:
            slot_dir = output_root / slot.name
            slot_dir.mkdir(parents=True)
            output_dirs[slot.name] = slot_dir

        # Per-mount staging: copy each input artifact's canonical
        # directory contents into a fresh tempdir so the body
        # never reads from <workspace>/artifacts/ directly.  See
        # :mod:`flywheel.input_staging` for the invariant.
        staged_inputs = stage_artifact_instances(
            self.workspace, input_bindings)
        input_paths = {
            slot: str(path) for slot, path in staged_inputs.items()
        }

        ctx = LocalExecutionContext(
            execution_id=execution_id,
            block_name=block,
            input_bindings=input_bindings,
            input_paths=input_paths,
            scratch_dir=str(scratch_dir),
            params=params or {},
            parent_execution_id=parent_execution_id,
            output_root=output_root,
            _output_dirs=output_dirs,
        )
        started_at = datetime.now(UTC)
        t0 = time.monotonic()

        body_failed: BaseException | None = None
        preserved_output_root: Path | None = None
        try:
            yield ctx
        except BaseException as exc:
            body_failed = exc
            raise
        finally:
            finished_at = datetime.now(UTC)
            elapsed = time.monotonic() - t0

            if body_failed is None:
                self._finalize_succeeded(
                    ctx=ctx,
                    block_def=block_def,
                    started_at=started_at,
                    finished_at=finished_at,
                    elapsed_s=elapsed,
                    caller=caller,
                    runner=runner,
                )
            else:
                self._finalize_failed(
                    ctx=ctx,
                    block_def=block_def,
                    started_at=started_at,
                    finished_at=finished_at,
                    elapsed_s=elapsed,
                    caller=caller,
                    runner=runner,
                    error=body_failed,
                )
                if _retain_failed_output_dirs():
                    preserved_output_root = output_root

            cleanup_staged_inputs(staged_inputs)
            self._cleanup_scratch(execution_id)
            if preserved_output_root is None:
                shutil.rmtree(output_root, ignore_errors=True)

    def drain_halts(self) -> list[_QueuedHalt]:
        """Pop and return all queued halt directives.

        Called by the host-side handoff loop between cycles.
        Returning a copy and clearing the queue means the loop
        sees each directive exactly once; if it decides not to
        act on one, that's its responsibility, not the recorder's.
        """
        drained = list(self.halt_queue)
        self.halt_queue.clear()
        return drained

    def _find_block(
        self, block_name: str,
    ) -> BlockDefinition | None:
        """Look up a block definition by name in the template."""
        for block in self.template.blocks:
            if block.name == block_name:
                return block
        return None

    def _resolve_input_bindings(
        self, block_def: BlockDefinition,
    ) -> dict[str, str]:
        """Resolve declared block inputs to latest registered instances.

        Optional slots with no instance are silently skipped;
        required slots with no instance raise ``ValueError``.
        Returns ``{slot_name: artifact_id}``; host paths come
        later from
        :func:`flywheel.input_staging.stage_artifact_instances`,
        which knows how to copy ``copy`` and ``incremental``
        artifacts into per-mount staging tempdirs.
        """
        bindings: dict[str, str] = {}

        for slot in block_def.inputs:
            instances = self.workspace.instances_for(slot.name)
            if not instances:
                if slot.optional:
                    continue
                raise ValueError(
                    f"Block {block_def.name!r} input slot "
                    f"{slot.name!r} has no registered "
                    f"instances in workspace"
                )
            instance = instances[-1]
            bindings[slot.name] = instance.id

        return bindings

    def _finalize_succeeded(
        self,
        *,
        ctx: LocalExecutionContext,
        block_def: BlockDefinition,
        started_at: datetime,
        finished_at: datetime,
        elapsed_s: float,
        caller: dict[str, Any] | None,
        runner: str | None,
    ) -> None:
        """Register artifacts from per-output dirs and record a succeeded row.

        For each declared output slot, looks up the artifact's
        declared kind, then either:

        * **copy** — if the per-output directory is non-empty,
          allocates a fresh artifact ID, copies the directory's
          contents into ``<workspace>/artifacts/<aid>/``, and
          registers a copy :class:`ArtifactInstance`.  Empty
          directories are skipped silently (the block chose not
          to emit this output this time).
        * **incremental** — if the per-output directory contains a
          file named ``entries.jsonl``, parses one JSON value per
          non-blank line and appends them to the workspace's
          canonical incremental instance for this name.  Creates
          the canonical instance on first append.  An empty
          directory or missing/empty ``entries.jsonl`` is a
          silent no-op.

        Atomic with respect to artifact writes: if any output
        registration fails partway through, partial *new* copy
        artifacts are rolled back; previously-completed copy
        artifacts in this same call are kept (each completes
        atomically once it lands in the ledger).  Incremental
        appends are not rolled back — they are append-only by
        design and partial appends remain visible to subsequent
        readers.  A ``"failed"`` row is written with the
        underlying error message so operators can diagnose
        without grepping logs.
        """
        registered: list[tuple[str, str]] = []
        output_payloads: dict[str, Path] = {}
        try:
            for slot in block_def.outputs:
                slot_dir = ctx.output_dir(slot.name)
                output_payloads[slot.name] = slot_dir

                kind = self.workspace.artifact_declarations.get(
                    slot.name)
                if kind is None:
                    raise ValueError(
                        f"Block {block_def.name!r} declares output "
                        f"{slot.name!r} but the workspace has no "
                        f"such artifact declaration"
                    )

                if kind == "copy":
                    aid = self._register_copy_output(
                        slot_name=slot.name,
                        slot_dir=slot_dir,
                        execution_id=ctx.execution_id,
                        finished_at=finished_at,
                    )
                    if aid is not None:
                        registered.append((slot.name, aid))
                elif kind == "incremental":
                    aid = self._append_incremental_output(
                        slot_name=slot.name,
                        slot_dir=slot_dir,
                        execution_id=ctx.execution_id,
                        finished_at=finished_at,
                    )
                    if aid is not None:
                        registered.append((slot.name, aid))
                else:
                    raise ValueError(
                        f"Block {block_def.name!r} output "
                        f"{slot.name!r} declared as kind "
                        f"{kind!r}; only 'copy' and 'incremental' "
                        f"are valid block-output kinds"
                    )
        except Exception as exc:
            for _, aid in registered:
                inst = self.workspace.artifacts.get(aid)
                if inst is None or inst.kind != "copy":
                    continue
                aid_dir = self.workspace.path / "artifacts" / aid
                shutil.rmtree(aid_dir, ignore_errors=True)
                self.workspace.artifacts.pop(aid, None)
            failed_error = f"output_write_failed: {exc}"
            self._write_row(BlockExecution(
                id=ctx.execution_id,
                block_name=ctx.block_name,
                started_at=started_at,
                finished_at=finished_at,
                status="failed",
                input_bindings=ctx.input_bindings,
                output_bindings={},
                elapsed_s=elapsed_s,
                image=block_def.image,
                parent_execution_id=ctx.parent_execution_id,
                runner=runner,
                caller=caller,
                params=ctx.params or None,
                error=failed_error,
            ), outputs={}, error=failed_error)
            return

        output_bindings = {name: aid for name, aid in registered}
        ctx.output_bindings = dict(output_bindings)

        self._write_row(BlockExecution(
            id=ctx.execution_id,
            block_name=ctx.block_name,
            started_at=started_at,
            finished_at=finished_at,
            status="succeeded",
            input_bindings=ctx.input_bindings,
            output_bindings=output_bindings,
            elapsed_s=elapsed_s,
            image=block_def.image,
            parent_execution_id=ctx.parent_execution_id,
            runner=runner,
            caller=caller,
            params=ctx.params or None,
        ), outputs=output_payloads, error=None)

    def _register_copy_output(
        self,
        *,
        slot_name: str,
        slot_dir: Path,
        execution_id: str,
        finished_at: datetime,
    ) -> str | None:
        """Copy the per-output directory into the canonical artifact store.

        Returns the new artifact ID, or ``None`` if the per-output
        directory is empty (treated as "block chose not to emit
        this output").
        """
        if not slot_dir.exists() or not any(slot_dir.iterdir()):
            return None
        aid = self.workspace.generate_artifact_id(slot_name)
        target_dir = self.workspace.path / "artifacts" / aid
        try:
            shutil.copytree(slot_dir, target_dir)
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        instance = ArtifactInstance(
            id=aid,
            name=slot_name,
            kind="copy",
            created_at=finished_at,
            produced_by=execution_id,
            copy_path=aid,
        )
        self.workspace.add_artifact(instance)
        return aid

    def _append_incremental_output(
        self,
        *,
        slot_name: str,
        slot_dir: Path,
        execution_id: str,
        finished_at: datetime,
    ) -> str | None:
        """Append entries from ``<slot_dir>/entries.jsonl`` to the canonical instance.

        Returns the (possibly newly-created) canonical incremental
        artifact ID for *slot_name*, or ``None`` if the slot
        directory had nothing to contribute (no ``entries.jsonl``
        or the file was empty).
        """
        entries_file = slot_dir / "entries.jsonl"
        if not entries_file.exists():
            return None
        raw = entries_file.read_text(encoding="utf-8")
        entries: list[Any] = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            entries.append(json.loads(stripped))
        if not entries:
            return None

        instance = self.workspace.latest_incremental_instance(
            slot_name)
        if instance is None:
            instance = self.workspace.register_incremental_artifact(
                slot_name,
                produced_by=execution_id,
                source=(
                    f"first appended by execution {execution_id}"),
            )
        self.workspace.append_to_incremental(instance.id, entries)
        return instance.id

    def _finalize_failed(
        self,
        *,
        ctx: LocalExecutionContext,
        block_def: BlockDefinition,
        started_at: datetime,
        finished_at: datetime,
        elapsed_s: float,
        caller: dict[str, Any] | None,
        runner: str | None,
        error: BaseException,
    ) -> None:
        """Record a failed row when the body raised."""
        error_str = f"{type(error).__name__}: {error}"
        self._write_row(BlockExecution(
            id=ctx.execution_id,
            block_name=ctx.block_name,
            started_at=started_at,
            finished_at=finished_at,
            status="failed",
            input_bindings=ctx.input_bindings,
            output_bindings={},
            elapsed_s=elapsed_s,
            image=block_def.image,
            parent_execution_id=ctx.parent_execution_id,
            runner=runner,
            caller=caller,
            params=ctx.params or None,
            error=error_str,
        ), outputs={}, error=error_str)

    def _write_row(
        self,
        execution: BlockExecution,
        *,
        outputs: dict[str, Path],
        error: str | None,
    ) -> None:
        """Persist a finalized execution row, then run post-check.

        The row is written atomically with the workspace save, so
        a crash between adding the row and saving leaves the
        workspace consistent (no orphans, no half-state).  After
        the row is durable, the post-check runs against the
        finalized state and may rewrite the row with a halt
        directive or post-check error.
        """
        self.workspace.add_execution(execution)
        self.workspace.save()

        updated = self._run_post_check(
            execution=execution,
            outputs=outputs,
            error=error,
        )
        if updated is not execution:
            with self.workspace._lock:  # noqa: SLF001
                self.workspace.executions[updated.id] = updated
            self.workspace.save()

    def _run_post_check(
        self,
        *,
        execution: BlockExecution,
        outputs: dict[str, Path],
        error: str | None,
    ) -> BlockExecution:
        """Invoke the configured post-check; return updated row.

        Returns ``execution`` unchanged when no check is wired up
        for this block name or the check returned ``None`` and
        didn't raise.  Otherwise returns a new
        :class:`BlockExecution` with ``halt_directive`` and / or
        ``post_check_error`` populated.  Halt directives are
        also appended to :attr:`halt_queue` so the host-side
        handoff loop can find them.
        """
        callable_ = self.post_checks.get(execution.block_name)
        if callable_ is None:
            return execution

        ctx = PostCheckContext(
            block=execution.block_name,
            execution_id=execution.id,
            status=execution.status,  # type: ignore[arg-type]
            caller=execution.caller,
            params=execution.params,
            error=error,
            outputs=outputs,
            parent_execution_id=execution.parent_execution_id,
            synthetic=execution.synthetic,
            workspace_path=self.workspace.path,
        )

        directive: HaltDirective | None = None
        post_check_error: str | None = None
        try:
            directive = callable_(ctx)
        except Exception as exc:  # noqa: BLE001
            post_check_error = f"{type(exc).__name__}: {exc}"

        if directive is not None and not isinstance(
                directive, HaltDirective):
            post_check_error = (
                f"post_check returned "
                f"{type(directive).__name__}, expected "
                f"HaltDirective | None")
            directive = None

        halt_dict: dict | None = None
        if directive is not None:
            halt_dict = directive.to_dict()
            self.halt_queue.append(_QueuedHalt(
                scope=directive.scope,
                reason=directive.reason,
                block=execution.block_name,
                execution_id=execution.id,
                parent_execution_id=execution.parent_execution_id,
            ))

        if halt_dict is None and post_check_error is None:
            return execution

        return replace(
            execution,
            halt_directive=halt_dict,
            post_check_error=post_check_error,
        )

    def _cleanup_scratch(self, execution_id: str) -> None:
        """Remove the per-execution scratch directory."""
        scratch_dir = (
            self.workspace.path
            / "execution_scratch"
            / execution_id
        )
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(scratch_dir, ignore_errors=True)


def _retain_failed_output_dirs() -> bool:
    """Whether to keep per-execution output dirs after a body failure.

    Reads the ``FLYWHEEL_RETAIN_FAILED_OUTPUT_DIRS`` environment
    variable.  Default: retain.  Tests that want deterministic
    cleanup set this to ``"0"`` (or ``"false"`` / ``"no"``).
    """
    raw = os.environ.get("FLYWHEEL_RETAIN_FAILED_OUTPUT_DIRS")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", ""}
