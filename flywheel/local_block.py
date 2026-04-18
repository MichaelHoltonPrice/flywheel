"""In-process recorder for full-stop block executions.

Under the full-stop nested-block model
(``flywheel.docs.architecture.md`` §"Nested block executions from
agents"), every nested block invocation is a real execution
boundary.  The agent stops, the host runs the block synchronously
in its own Python process, then the agent restarts with the
block's output spliced into its session.

This module is the recorder side of that handoff.  It plays the
role :class:`flywheel.tool_block.BlockChannelClient` used to play
when the bridge was alive, but with two structural differences
that the campaign was built around:

1. **No HTTP round-trip.**  The host-side block runner shares a
   Python process with the workspace, so it writes artifacts and
   ledger rows directly instead of POSTing into a loopback
   :class:`flywheel.execution_channel.ExecutionChannel`.  The
   loopback was the last consumer of the lifecycle API; deleting
   it lets B7 also delete the API.
2. **No ``"running"`` row.**  ``BlockChannelClient.begin`` opened
   a row with ``status="running"`` so the channel could remember
   what was in flight across the HTTP boundary.  Nothing needs
   to remember anything across an in-process boundary, so
   recorded executions go straight from non-existent →
   ``"succeeded"`` or ``"failed"`` and the orphaned-``running``
   class of bug becomes structurally impossible.

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
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from flywheel.artifact import ArtifactInstance, BlockExecution
from flywheel.post_check import (
    HaltDirective,
    PostCheckCallable,
    PostCheckContext,
)
from flywheel.template import BlockDefinition, Template
from flywheel.workspace import Workspace


class LocalBlockError(RuntimeError):
    """Raised when the recorder rejects a ``begin`` request.

    Mirrors the
    :class:`flywheel.tool_block.BlockChannelError` shape callers
    used to handle: an ``error_type`` string identifying the
    rejection class plus a human-readable message.  Used for
    pre-body failures (unknown block, missing required input)
    that the legacy bridge would have surfaced as
    ``ok=false`` HTTP responses.

    Body failures (exceptions raised by the wrapped block code)
    propagate out of the ``begin`` context manager unchanged —
    those are the caller's domain, not the recorder's.

    Attributes:
        error_type: One of ``"unknown_block"``, ``"missing_input"``.
        execution_id: ``None`` because no row was written for
            rejected requests under the post-bridge model
            (synthetic rows died with the bridge).  Kept on the
            exception only so callers that want to log a stable
            shape can do so.
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

    Mirrors the public surface of the old
    :class:`flywheel.tool_block.ExecutionContext` so host-side
    block runners can treat the two interchangeably during the
    migration window — but with the HTTP-specific fields
    (``input_hashes``, the wire-shape ``output_bindings``)
    removed because the recorder writes outputs directly and
    nobody needs the intermediate forms.

    Attributes:
        execution_id: Workspace-allocated ledger ID for this
            execution.  Stable for the life of the ``with`` block
            and persists on the resulting row.
        block_name: The block this execution corresponds to.
        input_bindings: ``{slot_name: artifact_id}`` resolved at
            ``begin`` time from the workspace's latest instances.
        input_paths: ``{slot_name: absolute path on host}`` to the
            artifact directory.  Valid for read-only use during
            the body; the recorder writes outputs to a different
            location.
        scratch_dir: Per-execution scratch directory under
            ``<workspace>/execution_scratch/<execution_id>``.
            Cleaned up by the recorder on exit (success or
            failure).
        params: The params dict the caller supplied to ``begin``,
            echoed back verbatim.
        parent_execution_id: ID of the runner that invoked this
            execution, when supplied to ``begin``.
        outputs: Buffer of slot_name → JSON-serializable data for
            registration on successful exit.  Populated via
            :meth:`set_output`.
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
    outputs: dict[str, Any] = field(default_factory=dict)
    output_bindings: dict[str, str] = field(default_factory=dict)

    def set_output(self, name: str, data: Any) -> None:
        """Buffer an output for atomic registration on exit.

        Args:
            name: The output slot name as declared by the block.
            data: The artifact contents.  JSON-encoded by the
                recorder when the body exits successfully.

        Raises:
            ValueError: If an output was already buffered for
                this slot name on this context.  The block's
                output slots are single-shot by design — a
                second ``set_output("game_step", ...)`` is
                always a programming error in the runner.
        """
        if name in self.outputs:
            raise ValueError(
                f"Output {name!r} already set on execution "
                f"{self.execution_id}"
            )
        self.outputs[name] = data


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
       workspace's latest instances, allocates a scratch
       directory, and yields a context.
    2. The block runner runs the side-effecting work inside the
       ``with`` block, populating outputs via
       :meth:`LocalExecutionContext.set_output`.
    3. On clean exit the recorder writes each buffered output as
       a copy artifact, registers the resulting
       :class:`flywheel.artifact.ArtifactInstance` on the
       workspace, builds a ``"succeeded"``
       :class:`flywheel.artifact.BlockExecution`, and runs the
       block's post-execution check (if configured).
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
                (``"missing_input"``).  Callers that want
                bridge-compatible error handling can catch this
                in place of the old ``BlockChannelError``.
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
            input_bindings, input_paths = (
                self._resolve_inputs(block_def))
        except ValueError as exc:
            raise LocalBlockError(
                str(exc), error_type="missing_input") from exc

        execution_id = self.workspace.generate_execution_id()
        scratch_dir = (
            self.workspace.path / "execution_scratch" / execution_id
        )
        scratch_dir.mkdir(parents=True, exist_ok=True)

        ctx = LocalExecutionContext(
            execution_id=execution_id,
            block_name=block,
            input_bindings=input_bindings,
            input_paths=input_paths,
            scratch_dir=str(scratch_dir),
            params=params or {},
            parent_execution_id=parent_execution_id,
        )
        started_at = datetime.now(UTC)
        t0 = time.monotonic()

        body_failed: BaseException | None = None
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

            self._cleanup_scratch(execution_id)

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

    def _resolve_inputs(
        self, block_def: BlockDefinition,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Resolve declared block inputs to latest registered instances.

        Same semantics as the legacy
        ``execution_channel._resolve_inputs_for_begin``: derived
        slots are rebuilt every begin, non-derived optional slots
        with no instance are silently skipped, non-derived
        required slots with no instance raise ``ValueError``.
        Returns ``(input_bindings, input_paths)`` as absolute
        host paths (the in-process caller can read them
        directly).
        """
        bindings: dict[str, str] = {}
        paths: dict[str, str] = {}

        for slot in block_def.inputs:
            if slot.derive_from is not None:
                instance = self._materialize_derived_input(slot)
                if instance is None:
                    continue
            else:
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
            if instance.copy_path:
                paths[slot.name] = str(
                    self.workspace.path / "artifacts"
                    / instance.copy_path
                )
            else:
                paths[slot.name] = ""

        return bindings, paths

    def _materialize_derived_input(self, slot):
        """Rebuild a derived input's rollup; return the new instance.

        Returns ``None`` when the source has no instances yet so
        the caller treats the slot as absent.  Raises
        ``ValueError`` on unsupported ``derive_kind`` (defense in
        depth — the parser also catches this).
        """
        if slot.derive_kind == "jsonl_concat":
            if not self.workspace.instances_for(slot.derive_from):
                return None
            return self.workspace.materialize_sequence(
                source_name=slot.derive_from,
                target_name=slot.name,
            )
        raise ValueError(
            f"Input slot {slot.name!r}: unsupported derive_kind "
            f"{slot.derive_kind!r}"
        )

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
        """Write outputs as artifacts and record a succeeded row.

        Atomic with respect to artifact writes: if any output
        registration fails, partial writes are rolled back and a
        ``"failed"`` row is written instead.  The row's ``error``
        field carries the underlying message so operators can
        diagnose without grepping logs.
        """
        registered: list[tuple[str, str]] = []
        try:
            for slot in block_def.outputs:
                data = ctx.outputs.get(slot.name)
                if data is None:
                    continue

                aid = self.workspace.generate_artifact_id(
                    slot.name)
                output_dir = (
                    self.workspace.path / "artifacts" / aid)
                output_dir.mkdir(parents=True)

                output_file = output_dir / f"{slot.name}.json"
                output_file.write_text(
                    json.dumps(data, separators=(",", ":")),
                    encoding="utf-8",
                )

                instance = ArtifactInstance(
                    id=aid,
                    name=slot.name,
                    kind="copy",
                    created_at=finished_at,
                    produced_by=ctx.execution_id,
                    copy_path=aid,
                )
                self.workspace.add_artifact(instance)
                registered.append((slot.name, aid))
        except Exception as exc:
            for _, aid in registered:
                aid_dir = self.workspace.path / "artifacts" / aid
                shutil.rmtree(aid_dir, ignore_errors=True)
                self.workspace.artifacts.pop(aid, None)
            failed_error = (
                f"output_write_failed: {exc}")
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
        ), outputs=ctx.outputs, error=None)

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
        outputs: dict[str, Any],
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
        outputs: dict[str, Any],
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

        Mirrors the channel's behaviour exactly so projects that
        ship post-checks against the bridge keep working without
        modification.
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
