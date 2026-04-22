"""Direct tests for the pattern→handoff wiring helpers.

These cover the public surface of :mod:`flywheel.pattern_handoff`
without going through the pattern runner.  The runner-level
tests in ``test_pattern_runner.py`` exercise the same code via
the real ``PatternRunner`` construction path; this file pins
the module's free functions in isolation so future drift in
either caller is caught locally.
"""

from __future__ import annotations

import functools

from flywheel.agent_handoff import (
    HandoffContext,
    HandoffResult,
    ToolRouter,
)
from flywheel.pattern_handoff import (
    adapt_post_dispatch_for_handoff,
    build_pattern_tool_router,
    launch_fn_accepts_post_dispatch,
    make_pattern_executor_factory,
    merge_block_runners,
    wrap_launch_fn_with_router,
)


class TestBuildPatternToolRouter:
    """Surface-level invariants of :func:`build_pattern_tool_router`."""

    def test_empty_on_tool_instances_returns_none(self) -> None:
        """No on_tool instances → None (caller's runner stays put)."""
        router = build_pattern_tool_router(
            pattern_name="p",
            on_tool_instances=[],
            workspace=object(),  # type: ignore[arg-type]
            template=object(),  # type: ignore[arg-type]
            executor_factory=None,
            per_instance_runtime_config={},
            run_id_provider=lambda: None,
        )
        assert router is None


class TestWrapLaunchFnWithRouter:
    """Surface-level invariants of :func:`wrap_launch_fn_with_router`."""

    def test_forwards_merged_block_runner(self) -> None:
        """Wrapped launch_fn injects the merged router as block_runner."""
        captured: dict[str, object] = {}

        def fake_launch(**kwargs: object) -> str:
            captured.update(kwargs)
            return "handle"

        pattern_router = ToolRouter({
            "tool_a": lambda ctx: HandoffResult(content="OK"),
        })
        wrapped = wrap_launch_fn_with_router(
            fake_launch,
            pattern_name="p",
            on_tool_instance_count=1,
            pattern_router=pattern_router,
            post_dispatch_fn=lambda _name: None,
        )
        result = wrapped(prompt="x")
        assert result == "handle"
        assert "block_runner" in captured

    def test_post_dispatch_adapter_unwraps_handoff_context(
        self,
    ) -> None:
        """The wiring's adapter calls post_dispatch_fn(tool_name).

        Pattern runner provides a ``str -> None`` cadence hook;
        the agent handoff loop calls back with a
        :class:`HandoffContext`.  The adapter built by
        :func:`wrap_launch_fn_with_router` is what bridges them.
        """
        seen_names: list[str] = []
        captured: dict[str, object] = {}

        def fake_launch(*, post_dispatch_fn=None, **_: object) -> str:
            captured["post_dispatch_fn"] = post_dispatch_fn
            return "handle"

        pattern_router = ToolRouter({
            "tool_a": lambda ctx: HandoffResult(content="OK"),
        })
        wrapped = wrap_launch_fn_with_router(
            fake_launch,
            pattern_name="p",
            on_tool_instance_count=1,
            pattern_router=pattern_router,
            post_dispatch_fn=seen_names.append,
        )
        wrapped(prompt="x")
        adapter = captured["post_dispatch_fn"]
        assert callable(adapter)
        adapter(HandoffContext(
            tool_name="tool_a",
            tool_input={},
            session_id="sess",
            tool_use_id="use",
            iteration=0,
        ))
        assert seen_names == ["tool_a"]

    def test_opaque_pre_bound_block_runner_rejected(self) -> None:
        """A non-ToolRouter pre-bound block_runner raises."""
        def fake_launch(*, block_runner=None, **_: object) -> str:
            return "handle"

        opaque = lambda ctx: HandoffResult(content="X")  # noqa: E731
        partial_launch = functools.partial(
            fake_launch, block_runner=opaque)
        pattern_router = ToolRouter({
            "tool_a": lambda ctx: HandoffResult(content="OK"),
        })
        try:
            wrap_launch_fn_with_router(
                partial_launch,
                pattern_name="p",
                on_tool_instance_count=1,
                pattern_router=pattern_router,
                post_dispatch_fn=lambda _name: None,
            )
        except ValueError as exc:
            assert "not a ``ToolRouter``" in str(exc)
        else:
            raise AssertionError(
                "wrap_launch_fn_with_router should reject an "
                "opaque pre-bound block_runner")


class TestLaunchFnAcceptsPostDispatch:
    """Signature-introspection helper used by the wrapper."""

    def test_named_kwarg_accepted(self) -> None:
        """A function declaring ``post_dispatch_fn`` returns True."""
        def fn(*, post_dispatch_fn=None) -> None:
            pass

        assert launch_fn_accepts_post_dispatch(fn) is True

    def test_var_keyword_accepted(self) -> None:
        """A function with ``**kwargs`` returns True (best-effort)."""
        def fn(**_: object) -> None:
            pass

        assert launch_fn_accepts_post_dispatch(fn) is True

    def test_no_var_keyword_no_named_returns_false(self) -> None:
        """A function with neither returns False."""
        def fn(*, prompt: str = "") -> None:
            pass

        assert launch_fn_accepts_post_dispatch(fn) is False

    def test_partial_unwraps_to_inner(self) -> None:
        """``functools.partial`` is unwrapped before introspection."""
        def fn(*, post_dispatch_fn=None) -> None:
            pass

        wrapped = functools.partial(fn)
        assert launch_fn_accepts_post_dispatch(wrapped) is True


class TestMergeBlockRunners:
    """Direct invariants for :func:`merge_block_runners`."""

    def test_no_fallback_returns_pattern_only_runner(self) -> None:
        """With ``fallback=None`` every tool routes to the pattern."""
        pattern_router = ToolRouter({
            "tool_a": lambda ctx: HandoffResult(content="A"),
        })
        merged = merge_block_runners(
            pattern_router=pattern_router,
            fallback=None,
        )
        result = merged(HandoffContext(
            tool_name="tool_a",
            tool_input={},
            session_id="s",
            tool_use_id="u",
            iteration=0,
        ))
        assert result.content == "A"
        # Tool the pattern doesn't claim and no fallback present:
        # the merge produces a self-contained error result.
        result = merged(HandoffContext(
            tool_name="other",
            tool_input={},
            session_id="s",
            tool_use_id="u",
            iteration=0,
        ))
        assert result.is_error
        assert "no runner for tool" in result.content

    def test_fallback_router_is_layered_below(self) -> None:
        """Tools the pattern doesn't claim fall through to the fallback."""
        pattern_router = ToolRouter({
            "tool_a": lambda ctx: HandoffResult(content="A"),
        })
        fallback = ToolRouter({
            "tool_b": lambda ctx: HandoffResult(content="B"),
        })
        merged = merge_block_runners(
            pattern_router=pattern_router,
            fallback=fallback,
        )
        result = merged(HandoffContext(
            tool_name="tool_b",
            tool_input={},
            session_id="s",
            tool_use_id="u",
            iteration=0,
        ))
        assert result.content == "B"

    def test_collision_rejected(self) -> None:
        """Tool name claimed by both raises with the collision label."""
        pattern_router = ToolRouter({
            "tool_a": lambda ctx: HandoffResult(content="A"),
        })
        fallback = ToolRouter({
            "tool_a": lambda ctx: HandoffResult(content="B"),
        })
        try:
            merge_block_runners(
                pattern_router=pattern_router,
                fallback=fallback,
                collision_label="MyContext",
            )
        except ValueError as exc:
            assert "MyContext" in str(exc)
            assert "tool_a" in str(exc)
        else:
            raise AssertionError(
                "merge_block_runners should reject overlapping "
                "tool names")

    def test_opaque_fallback_rejected(self) -> None:
        """Non-ToolRouter fallback raises with the context label."""
        pattern_router = ToolRouter({
            "tool_a": lambda ctx: HandoffResult(content="A"),
        })
        opaque = lambda ctx: HandoffResult(content="X")  # noqa: E731
        try:
            merge_block_runners(
                pattern_router=pattern_router,
                fallback=opaque,
                context_label="MyCtx",
            )
        except ValueError as exc:
            assert "MyCtx" in str(exc)
            assert "ToolRouter" in str(exc)
        else:
            raise AssertionError(
                "merge_block_runners should reject opaque "
                "fallback BlockRunners")


class TestAdaptPostDispatchForHandoff:
    """Direct invariants for :func:`adapt_post_dispatch_for_handoff`."""

    def test_extracts_tool_name_and_forwards(self) -> None:
        """The adapter strips the HandoffContext to ``tool_name``."""
        seen: list[str] = []
        adapter = adapt_post_dispatch_for_handoff(seen.append)
        adapter(HandoffContext(
            tool_name="t",
            tool_input={},
            session_id="s",
            tool_use_id="u",
            iteration=0,
        ))
        assert seen == ["t"]


class TestMakePatternExecutorFactory:
    """Direct invariants for :func:`make_pattern_executor_factory`."""

    def test_executors_without_for_pattern_pass_through(self) -> None:
        """Executors that don't expose ``for_pattern`` are returned as-is."""
        sentinel = object()

        def project_factory(_block_def: object) -> object:
            return sentinel

        pattern_router = ToolRouter({
            "tool_a": lambda ctx: HandoffResult(content="OK"),
        })
        wrapped = make_pattern_executor_factory(
            project_factory,  # type: ignore[arg-type]
            pattern_router=pattern_router,
            post_dispatch_fn=lambda _name: None,
        )

        class _Block:
            name = "b"

        assert wrapped(_Block()) is sentinel  # type: ignore[arg-type]

    def test_executors_with_for_pattern_get_sibling(self) -> None:
        """``for_pattern`` is invoked with router + post_dispatch_fn.

        The wrapped factory returns whatever ``for_pattern`` returns,
        not the original executor.  This is what lets the runner give
        each agent executor a sibling that knows about the pattern's
        ``on_tool`` router without leaking router knowledge into the
        runner itself.
        """
        forwarded: dict[str, object] = {}
        sibling = object()

        class _Executor:
            def for_pattern(
                self,
                *,
                pattern_router: object,
                post_dispatch_fn: object,
            ) -> object:
                forwarded["pattern_router"] = pattern_router
                forwarded["post_dispatch_fn"] = post_dispatch_fn
                return sibling

        def project_factory(_block_def: object) -> object:
            return _Executor()

        pattern_router = ToolRouter({
            "tool_a": lambda ctx: HandoffResult(content="OK"),
        })
        post_dispatch = lambda _name: None  # noqa: E731
        wrapped = make_pattern_executor_factory(
            project_factory,  # type: ignore[arg-type]
            pattern_router=pattern_router,
            post_dispatch_fn=post_dispatch,
        )

        class _Block:
            name = "b"

        result = wrapped(_Block())  # type: ignore[arg-type]
        assert result is sibling
        assert forwarded["pattern_router"] is pattern_router
        assert forwarded["post_dispatch_fn"] is post_dispatch

    def test_results_are_cached_per_block_name(self) -> None:
        """Calling the wrapped factory twice for the same block reuses.

        The cache prevents an AgentExecutor (or any other for_pattern
        executor) from being rebuilt on every launch.  ``project_factory``
        is consulted exactly once per block name.
        """
        call_count = {"n": 0}

        class _Executor:
            def for_pattern(
                self,
                *,
                pattern_router: object,
                post_dispatch_fn: object,
            ) -> object:
                return self

        def project_factory(_block_def: object) -> object:
            call_count["n"] += 1
            return _Executor()

        pattern_router = ToolRouter({
            "tool_a": lambda ctx: HandoffResult(content="OK"),
        })
        wrapped = make_pattern_executor_factory(
            project_factory,  # type: ignore[arg-type]
            pattern_router=pattern_router,
            post_dispatch_fn=lambda _name: None,
        )

        class _Block:
            name = "b"

        first = wrapped(_Block())  # type: ignore[arg-type]
        second = wrapped(_Block())  # type: ignore[arg-type]
        assert first is second
        assert call_count["n"] == 1
