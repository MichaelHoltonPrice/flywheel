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
    build_pattern_tool_router,
    launch_fn_accepts_post_dispatch,
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
