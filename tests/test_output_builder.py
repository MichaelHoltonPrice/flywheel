"""Tests for :mod:`flywheel.output_builder`.

Covers the dotted-path resolver contract that ``BlockRegistry``
uses at registry-load time.  End-to-end coverage (the executor
actually runs the builder between container exit and artifact
collection) lives in
:mod:`tests.test_executor_output_builder`.
"""

from __future__ import annotations

import pytest

from flywheel.output_builder import (
    OutputBuilderContext,
    resolve_dotted_path,
)


class TestContextShape:
    """``OutputBuilderContext`` is a frozen read-only record.

    The lock-down test here prevents accidental mutation of the
    context from inside a builder — the contract is explicit
    that builders may write only into their own output tempdirs.
    """

    def test_is_frozen(self, tmp_path):
        ctx = OutputBuilderContext(
            block="brainstorm",
            execution_id="exec_00000001",
            outputs={"brainstorm_result": tmp_path},
            workspace=object(),  # type: ignore[arg-type]
        )
        with pytest.raises(Exception):
            ctx.block = "changed"  # type: ignore[misc]


class TestResolveDottedPath:
    """``resolve_dotted_path`` imports a callable by dotted path.

    The resolver is deliberately noisy: every failure mode
    raises :class:`ValueError` with an actionable message, so
    registry-load-time validation can surface a human-readable
    "fix the typo in ``output_builder:`` here" error.
    """

    def test_resolves_valid_path(self):
        # An obvious importable callable in the stdlib works.
        func = resolve_dotted_path("json.dumps")
        assert callable(func)

    def test_non_string_raises(self):
        with pytest.raises(
            ValueError, match="non-empty string",
        ):
            resolve_dotted_path(None)  # type: ignore[arg-type]

    def test_empty_string_raises(self):
        with pytest.raises(
            ValueError, match="non-empty string",
        ):
            resolve_dotted_path("")

    def test_no_dot_raises(self):
        with pytest.raises(
            ValueError, match="at least one '.'",
        ):
            resolve_dotted_path("no_dot_here")

    def test_missing_module_raises(self):
        with pytest.raises(
            ValueError, match="cannot import module",
        ):
            resolve_dotted_path(
                "definitely_not_a_real_module.func")

    def test_missing_attribute_raises(self):
        with pytest.raises(
            ValueError,
            match="has no attribute",
        ):
            resolve_dotted_path("json.definitely_not_a_func")

    def test_non_callable_raises(self):
        with pytest.raises(
            ValueError, match="is not\\s*callable",
        ):
            # ``json.__name__`` is a module attribute (str), not
            # callable — resolves to the str object.
            resolve_dotted_path("json.__name__")
