"""Agent-battery glue between the pattern runner and the handoff loop.

The pattern runner is generic over executors and does not need
to know about the handoff loop's tool-routing or post-dispatch
hook protocol.  This module owns the wiring that turns a
pattern's ``on_tool`` instances into a :class:`ToolRouter`, the
merge with any project-supplied fallback router, and the small
adapter that lets the runner expose a generic
``post_dispatch_fn(tool_name)`` callback while the handoff loop
calls back with a :class:`HandoffContext`.

This file is the only place inside :mod:`flywheel` outside of
:mod:`flywheel.agent` and :mod:`flywheel.agent_handoff` itself
that imports from the agent battery.  Other flywheel modules —
including the runner — go through these helpers so the agent
remains an opt-in wiring concern, not a runner-level
dependency.
"""

from __future__ import annotations

import functools
import inspect
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from flywheel.agent_handoff import (
    BlockRunner,
    HandoffContext,
    HandoffResult,
    ToolRouter,
)
from flywheel.instance_runtime import (
    ExecutorFactory,
    InstanceRuntimeConfig,
)
from flywheel.pattern import BlockInstance, OnToolTrigger
from flywheel.template import BlockDefinition, Template
from flywheel.workspace import Workspace


def build_pattern_tool_router(
    *,
    pattern_name: str,
    on_tool_instances: list[BlockInstance],
    workspace: Workspace,
    template: Template,
    executor_factory: ExecutorFactory | None,
    per_instance_runtime_config: dict[str, InstanceRuntimeConfig],
    run_id_provider: Callable[[], str | None],
) -> ToolRouter | None:
    """Turn ``on_tool`` instances into a :class:`ToolRouter`.

    Returns ``None`` when the pattern declares no ``on_tool``
    instances (the runner then leaves the caller's existing
    block_runner unchanged).  Raises at construction time on
    any declared-but-unserviceable instance:

    * Missing ``executor_factory``: the caller must provide
      one whenever the pattern has at least one ``on_tool``
      trigger.
    * Target block not in the template: the pattern
      references a block the project's registry doesn't know.
    * Target block has an ``inputs:`` list with count != 1:
      the tool-input serialisation convention (JSON into the
      single declared input slot) only works for 1-input
      blocks.  Multi-input blocks must wait for an explicit
      ``input_slot`` or ``input_map`` field on the trigger.
    * Input slot is not declared as a ``copy`` artifact: the
      tool-input bridge writes a JSON file and registers it
      via ``workspace.register_artifact``, which only handles
      copy artifacts today.

    ``run_id_provider`` is invoked lazily inside each per-tool
    runner so the dispatched executions are tagged with the
    runner's *current* run id (set in ``PatternRunner.run``,
    not at construction time).
    """
    if not on_tool_instances:
        return None
    if executor_factory is None:
        raise ValueError(
            f"Pattern {pattern_name!r} declares "
            f"{len(on_tool_instances)} on_tool "
            f"instance(s) but no ``executor_factory`` was "
            f"supplied to PatternRunner; cannot dispatch."
        )
    routes: dict[str, BlockRunner] = {}
    for inst in on_tool_instances:
        trigger = inst.trigger
        assert isinstance(trigger, OnToolTrigger)
        block_def = _find_block(template, inst.block)
        if block_def is None:
            raise ValueError(
                f"Pattern {pattern_name!r} on_tool "
                f"instance {inst.name!r} references "
                f"unknown block {inst.block!r}"
            )
        if len(block_def.inputs) != 1:
            raise ValueError(
                f"Pattern {pattern_name!r} on_tool "
                f"dispatch to block {block_def.name!r} "
                f"requires exactly one declared input slot "
                f"(tool_input is serialised as JSON into "
                f"that slot); block has "
                f"{len(block_def.inputs)} inputs.  "
                f"Declare a wrapper block or extend the "
                f"trigger with an explicit input mapping."
            )
        slot_name = block_def.inputs[0].name
        declared_kind = (
            workspace.artifact_declarations.get(slot_name))
        if declared_kind not in (None, "copy"):
            raise ValueError(
                f"Pattern {pattern_name!r} on_tool "
                f"dispatch to block {block_def.name!r}: "
                f"input slot {slot_name!r} is declared as "
                f"{declared_kind!r} but the tool-input "
                f"bridge only supports ``copy`` artifacts "
                f"(it serialises tool_input to a single "
                f"JSON file and registers it as a fresh "
                f"instance).  Declare the artifact as "
                f"``kind: copy`` or extend the trigger "
                f"with an explicit input mapping."
            )
        runtime_cfg = per_instance_runtime_config.get(
            inst.name, InstanceRuntimeConfig())
        runner = _make_on_tool_runner(
            block_def=block_def,
            workspace=workspace,
            executor_factory=executor_factory,
            runtime_cfg=runtime_cfg,
            run_id_provider=run_id_provider,
        )
        if trigger.tool in routes:
            raise ValueError(
                f"Pattern {pattern_name!r}: tool "
                f"{trigger.tool!r} is declared by multiple "
                f"on_tool instances"
            )
        routes[trigger.tool] = runner
    return ToolRouter(routes)


def wrap_launch_fn_with_router(
    launch_fn: Callable[..., Any],
    *,
    pattern_name: str,
    on_tool_instance_count: int,
    pattern_router: ToolRouter,
    post_dispatch_fn: Callable[[str], None],
) -> Callable[..., Any]:
    """Produce a launch_fn that forwards a merged block_runner.

    Construction-time invariant: when the pattern declares
    ``on_tool`` triggers, any pre-bound ``block_runner`` on
    ``launch_fn`` must be a :class:`ToolRouter` (or absent).
    Opaque callables are rejected — the whole point of the
    "pattern is authoritative" guarantee is that we can see
    every tool both sides will dispatch; an opaque fallback
    could silently handle a tool the pattern also claims,
    producing exactly the split-brain dispatch this check
    exists to prevent.

    Once both sides are :class:`ToolRouter`s the merge
    unions the routes; any tool name declared by both
    raises immediately.

    ``post_dispatch_fn`` is the runner's generic cadence hook:
    it accepts a tool name string.  This wrapper adapts the
    handoff loop's :class:`HandoffContext`-shaped callback to
    that signature so the runner stays free of any agent-
    battery types.
    """
    existing_router: Any = None
    if isinstance(launch_fn, functools.partial):
        existing_router = launch_fn.keywords.get("block_runner")
    if (existing_router is not None
            and not isinstance(existing_router, ToolRouter)):
        raise ValueError(
            f"Pattern {pattern_name!r}: declares "
            f"{on_tool_instance_count} on_tool instance(s) "
            f"but the caller's ``launch_fn`` has a pre-bound "
            f"``block_runner`` of type "
            f"{type(existing_router).__name__!r} that is not "
            f"a ``ToolRouter``.  Wrap the caller's routing in "
            f"``make_tool_router(...)`` so the pattern runner "
            f"can statically detect tool-name collisions and "
            f"honour the 'pattern is authoritative' guarantee."
        )
    if isinstance(existing_router, ToolRouter):
        overlap = (
            pattern_router.tools()
            & existing_router.tools()
        )
        if overlap:
            raise ValueError(
                f"Pattern {pattern_name!r}: tool(s) "
                f"{sorted(overlap)} are declared by both the "
                f"pattern's on_tool instances and the caller-"
                f"supplied block_runner; refusing to dispatch "
                f"one silently over the other"
            )
    merged = _MergedRouter(
        pattern_router=pattern_router,
        fallback=existing_router,
    )

    def _adapter(ctx: HandoffContext) -> None:
        post_dispatch_fn(ctx.tool_name)

    def _wrapped(**kwargs: Any) -> Any:
        kwargs["block_runner"] = merged
        # Only forward post_dispatch_fn when the callee's
        # launcher accepts it.  ``launch_agent_with_handoffs``
        # (production) does; bare ``launch_agent_block`` does
        # not.  The pattern runner's default ``launch_fn`` is
        # the latter so tests that don't need handoff loop
        # semantics keep working; production wires the former
        # via ``functools.partial`` in
        # :func:`cyberarc.project.ProjectHooks.init`.
        try:
            forward_hook = launch_fn_accepts_post_dispatch(
                launch_fn)
        except (TypeError, ValueError):
            forward_hook = False
        if forward_hook:
            kwargs.setdefault("post_dispatch_fn", _adapter)
        return launch_fn(**kwargs)

    return _wrapped


def launch_fn_accepts_post_dispatch(launch_fn: Any) -> bool:
    """Return True if ``launch_fn`` accepts a ``post_dispatch_fn`` kwarg.

    Handles :class:`functools.partial` by unwrapping to the inner
    callable.  Inspecting ``inspect.signature`` is safe for
    production callers (``launch_agent_with_handoffs``,
    ``launch_agent_block``) and for tests passing plain
    ``lambda``s with ``**kwargs`` — the latter returns ``True`` on
    the ``**kwargs`` varkeyword entry.  Unknown callable kinds
    return ``False`` so the router silently declines to forward
    the hook rather than crashing at launch time.
    """
    target = launch_fn
    while isinstance(target, functools.partial):
        target = target.func
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.name == "post_dispatch_fn":
            return True
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _make_on_tool_runner(
    *,
    block_def: BlockDefinition,
    workspace: Workspace,
    executor_factory: ExecutorFactory,
    runtime_cfg: InstanceRuntimeConfig,
    run_id_provider: Callable[[], str | None],
) -> BlockRunner:
    """Build the callable that dispatches one tool → block.

    Called once per ``on_tool`` instance from
    :func:`build_pattern_tool_router`.  The resulting callable
    writes ``ctx.tool_input`` as JSON into the block's single
    declared input slot, registers it as a fresh artifact
    instance, and invokes the executor with the per-instance
    runtime config forwarded as ``extra_env`` /
    ``extra_mounts``.
    """
    executor = executor_factory(block_def)
    slot_name = block_def.inputs[0].name

    def _runner(ctx: HandoffContext) -> HandoffResult:
        # Every failure from here to the end is mapped to a
        # ``HandoffResult(is_error=True)`` so no exception
        # can escape into the handoff loop's uncaught path.
        # Earlier versions wrapped only the ``executor.launch``
        # call, letting ``json.dumps`` or
        # ``register_artifact`` failures blow up the loop.
        try:
            try:
                payload = json.dumps(ctx.tool_input)
            except (TypeError, ValueError) as exc:
                return HandoffResult(
                    content=(
                        f"ERROR: on_tool dispatch to "
                        f"{block_def.name!r} cannot "
                        f"serialize tool_input: {exc}"),
                    is_error=True,
                )
            tmp = Path(tempfile.mkdtemp(prefix="on-tool-"))
            try:
                (tmp / f"{slot_name}.json").write_text(
                    payload, encoding="utf-8")
                instance = workspace.register_artifact(
                    slot_name, tmp,
                    source=(
                        f"on_tool {ctx.tool_name!r} -> "
                        f"{block_def.name}"),
                )
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            handle = executor.launch(
                block_name=block_def.name,
                workspace=workspace,
                input_bindings={slot_name: instance.id},
                run_id=run_id_provider(),
                extra_env=(
                    dict(runtime_cfg.extra_env)
                    if runtime_cfg.extra_env else None),
                extra_mounts=(
                    list(runtime_cfg.extra_mounts)
                    if runtime_cfg.extra_mounts else None),
            )
            result = handle.wait()
        except Exception as exc:  # noqa: BLE001
            return HandoffResult(
                content=(
                    f"ERROR: on_tool dispatch to "
                    f"{block_def.name!r} failed: {exc}"
                ),
                is_error=True,
            )
        if result.status == "succeeded":
            return HandoffResult(content="OK")
        execution = workspace.executions.get(
            result.execution_id)
        err = (
            execution.error if execution is not None
            else result.status
        )
        return HandoffResult(
            content=f"ERROR: {err}", is_error=True,
        )

    return _runner


def _find_block(
    template: Template, block_name: str,
) -> BlockDefinition | None:
    """Return the block definition with this name, or ``None``."""
    for b in template.blocks:
        if b.name == block_name:
            return b
    return None


class _MergedRouter:
    """Layer a pattern-owned :class:`ToolRouter` over a fallback.

    The pattern router handles every tool it declares; anything
    else is forwarded to the ``fallback`` block runner (usually
    a project-hooks-provided router for the handoff tools the
    pattern has not yet migrated to ``on_tool``).  Overlap is
    detected and rejected at construction time in
    :func:`wrap_launch_fn_with_router`; this class assumes the
    merge is disjoint and trusts its builder.
    """

    def __init__(
        self,
        *,
        pattern_router: ToolRouter,
        fallback: BlockRunner | None,
    ) -> None:
        """Capture both routers."""
        self._pattern = pattern_router
        self._fallback = fallback

    def __call__(
        self, ctx: HandoffContext,
    ) -> HandoffResult:
        """Dispatch to the pattern router's tools first."""
        if ctx.tool_name in self._pattern.tools():
            return self._pattern(ctx)
        if self._fallback is not None:
            return self._fallback(ctx)
        return HandoffResult(
            content=(
                f"ERROR: no runner for tool "
                f"{ctx.tool_name!r}"
            ),
            is_error=True,
        )
