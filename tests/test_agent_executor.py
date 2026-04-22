"""Unit tests for :class:`flywheel.agent_executor.AgentExecutor`.

The executor is purely additive in slice 2: no production caller
has been moved onto it yet, so these tests exercise it in
isolation by monkeypatching ``launch_agent_block`` and
``launch_agent_with_handoffs`` to return controlled fake handles.
There is no Docker, no real container, no real session JSONL on
disk.

Coverage:

* Defaults from the constructor flow into the underlying launcher
  unchanged when ``overrides`` is empty (besides ``prompt``).
* Per-launch ``overrides`` win over constructor defaults for
  scalar keys.
* ``extra_env`` merges (per-launch wins on collision);
  ``extra_mounts`` appends (constructor first, per-launch after).
* ``prompt_substitutions`` mutate the ``prompt`` before launch.
* Missing / empty ``prompt`` raises ``ValueError`` at launch time.
* The returned handle satisfies ``ExecutionHandle``: ``is_alive``
  / ``stop`` forward; ``wait()`` returns an ``ExecutionResult``
  whose ``status`` / ``output_bindings`` come from the durable
  :class:`BlockExecution` record; ``agent_result()`` exposes the
  raw :class:`AgentResult` for agent-aware callers.
* The pre-container failure path (no record in the workspace)
  yields a ``status="failed"`` result without crashing.
* When ``block_runner`` is configured the executor routes through
  ``launch_agent_with_handoffs`` instead of
  ``launch_agent_block`` and forwards the loop-level constructor
  defaults (``halt_source``, ``post_dispatch_fn``,
  ``resume_prompt``, ``max_iterations``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from flywheel import agent_executor as agent_executor_mod
from flywheel.agent import AgentResult
from flywheel.agent_executor import (
    AgentExecutionHandle,
    AgentExecutor,
)
from flywheel.agent_handoff import HandoffContext, HandoffResult
from flywheel.artifact import BlockExecution
from flywheel.executor import ExecutionResult


# --------------------------------------------------------------------
# Fakes.  The executor only touches ``workspace.executions`` to
# look up the durable record after wait(), so a tiny stand-in is
# enough.
# --------------------------------------------------------------------


@dataclass
class _FakeWorkspace:
    """Minimal stand-in: only ``executions`` is read by the SUT."""

    path: Path
    executions: dict[str, BlockExecution] = field(
        default_factory=dict)


class _FakeInnerHandle:
    """Stand-in for AgentHandle / HandoffAgentHandle.

    Records ``stop`` calls and lets the test pre-load the
    :class:`AgentResult` that ``wait()`` returns.  Optionally
    writes a :class:`BlockExecution` into the workspace before
    returning, mirroring how the real agent path produces a
    durable record during ``wait``.
    """

    def __init__(
        self,
        *,
        result: AgentResult,
        workspace: _FakeWorkspace | None = None,
        write_execution: BlockExecution | None = None,
    ) -> None:
        self._result = result
        self._workspace = workspace
        self._write_execution = write_execution
        self.stop_calls: list[str] = []
        self._waited = False
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def stop(self, reason: str = "requested") -> None:
        self.stop_calls.append(reason)

    def wait(self) -> AgentResult:
        if self._waited:
            raise RuntimeError("wait() called twice on fake")
        self._waited = True
        self._alive = False
        if (
            self._write_execution is not None
            and self._workspace is not None
        ):
            self._workspace.executions[
                self._write_execution.id
            ] = self._write_execution
        return self._result


def _captured_kwargs() -> dict[str, Any]:
    """Container the patched launcher writes its kwargs into."""
    return {}


def _patch_block_launcher(
    monkeypatch: pytest.MonkeyPatch,
    *,
    inner: _FakeInnerHandle,
    captured: dict[str, Any],
) -> None:
    """Replace ``launch_agent_block`` with a kwargs recorder."""

    def _fake(**kwargs: Any) -> _FakeInnerHandle:
        captured.clear()
        captured.update(kwargs)
        return inner

    monkeypatch.setattr(
        agent_executor_mod, "launch_agent_block", _fake)


def _patch_handoff_launcher(
    monkeypatch: pytest.MonkeyPatch,
    *,
    inner: _FakeInnerHandle,
    captured: dict[str, Any],
) -> None:
    """Replace ``launch_agent_with_handoffs`` with a recorder."""

    def _fake(**kwargs: Any) -> _FakeInnerHandle:
        captured.clear()
        captured.update(kwargs)
        return inner

    monkeypatch.setattr(
        agent_executor_mod,
        "launch_agent_with_handoffs",
        _fake,
    )


def _agent_result(
    *,
    execution_id: str | None = "exec-1",
    exit_code: int = 0,
    elapsed_s: float = 0.5,
    exit_reason: str = "completed",
) -> AgentResult:
    """Build a minimal :class:`AgentResult` for fakes."""
    return AgentResult(
        exit_code=exit_code,
        elapsed_s=elapsed_s,
        evals_run=0,
        execution_id=execution_id,
        stop_reason=None,
        exit_reason=exit_reason,
        exit_state=None,
        pending_tool_calls=None,
    )


def _block_execution(
    *,
    execution_id: str = "exec-1",
    status: str = "succeeded",
    output_bindings: dict[str, str] | None = None,
    exit_code: int = 0,
) -> BlockExecution:
    """Build a minimal :class:`BlockExecution` for fakes."""
    now = datetime.now(UTC)
    return BlockExecution(
        id=execution_id,
        block_name="play",
        started_at=now,
        finished_at=now,
        status=status,  # type: ignore[arg-type]
        output_bindings=dict(output_bindings or {}),
        exit_code=exit_code,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> _FakeWorkspace:
    """Fresh workspace rooted in ``tmp_path``."""
    return _FakeWorkspace(path=tmp_path)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """A throwaway project-root path; never actually read."""
    return tmp_path


@pytest.fixture
def template() -> Any:
    """Opaque stand-in: AgentExecutor only forwards it."""
    return object()


# --------------------------------------------------------------------
# launch() default path: routes through launch_agent_block and
# forwards constructor defaults.
# --------------------------------------------------------------------


class TestDefaultsPassThrough:
    """With no ``block_runner`` and an empty overrides dict (apart
    from the required ``prompt``), constructor defaults flow into
    ``launch_agent_block`` unchanged."""

    def test_constructor_defaults_become_launch_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
    ) -> None:
        captured = _captured_kwargs()
        inner = _FakeInnerHandle(
            result=_agent_result(),
            workspace=workspace,
            write_execution=_block_execution(),
        )
        _patch_block_launcher(
            monkeypatch, inner=inner, captured=captured)

        executor = AgentExecutor(
            template=template,
            project_root=project_root,
            agent_image="img:test",
            auth_volume="auth-vol",
            model="claude-test",
            max_turns=42,
            total_timeout=900,
            source_dirs=["src"],
            mcp_servers="demo",
            allowed_tools="Bash",
            extra_env={"K": "v"},
            extra_mounts=[("/host", "/container", "ro")],
            isolated_network=True,
        )

        executor.launch(
            block_name="play",
            workspace=workspace,
            input_bindings={"prompt_in": "art-1"},
            overrides={"prompt": "do the thing"},
            run_id="run-1",
        )

        assert captured["block_name"] == "play"
        assert captured["workspace"] is workspace
        assert captured["template"] is template
        assert captured["project_root"] == project_root
        assert captured["prompt"] == "do the thing"
        assert captured["agent_image"] == "img:test"
        assert captured["auth_volume"] == "auth-vol"
        assert captured["model"] == "claude-test"
        assert captured["max_turns"] == 42
        assert captured["total_timeout"] == 900
        assert captured["source_dirs"] == ["src"]
        assert captured["input_artifacts"] == {"prompt_in": "art-1"}
        assert captured["mcp_servers"] == "demo"
        assert captured["allowed_tools"] == "Bash"
        assert captured["extra_env"] == {"K": "v"}
        assert captured["extra_mounts"] == [
            ("/host", "/container", "ro")]
        assert captured["isolated_network"] is True
        assert captured["predecessor_id"] is None
        assert captured["run_id"] == "run-1"


# --------------------------------------------------------------------
# overrides layering
# --------------------------------------------------------------------


class TestOverridesLayering:
    """Per-launch ``overrides`` win over constructor defaults."""

    def test_scalar_overrides_win(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
    ) -> None:
        captured = _captured_kwargs()
        inner = _FakeInnerHandle(
            result=_agent_result(),
            workspace=workspace,
            write_execution=_block_execution(),
        )
        _patch_block_launcher(
            monkeypatch, inner=inner, captured=captured)

        executor = AgentExecutor(
            template=template,
            project_root=project_root,
            model="default-model",
            max_turns=10,
            total_timeout=60,
            mcp_servers="default-mcp",
            allowed_tools="Read",
            source_dirs=["src"],
            agent_image="img:default",
            auth_volume="default-vol",
            isolated_network=False,
        )

        executor.launch(
            block_name="play",
            workspace=workspace,
            input_bindings={},
            overrides={
                "prompt": "p",
                "model": "override-model",
                "max_turns": 5,
                "total_timeout": 30,
                "mcp_servers": "override-mcp",
                "allowed_tools": "Bash",
                "source_dirs": ["other-src"],
                "agent_image": "img:override",
                "auth_volume": "override-vol",
                "isolated_network": True,
                "predecessor_id": "prev-exec",
            },
        )

        assert captured["model"] == "override-model"
        assert captured["max_turns"] == 5
        assert captured["total_timeout"] == 30
        assert captured["mcp_servers"] == "override-mcp"
        assert captured["allowed_tools"] == "Bash"
        assert captured["source_dirs"] == ["other-src"]
        assert captured["agent_image"] == "img:override"
        assert captured["auth_volume"] == "override-vol"
        assert captured["isolated_network"] is True
        assert captured["predecessor_id"] == "prev-exec"

    def test_extra_env_merges_per_launch_wins(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
    ) -> None:
        captured = _captured_kwargs()
        inner = _FakeInnerHandle(
            result=_agent_result(),
            workspace=workspace,
            write_execution=_block_execution(),
        )
        _patch_block_launcher(
            monkeypatch, inner=inner, captured=captured)

        executor = AgentExecutor(
            template=template,
            project_root=project_root,
            extra_env={"A": "1", "B": "2"},
        )

        executor.launch(
            block_name="play",
            workspace=workspace,
            input_bindings={},
            overrides={
                "prompt": "p",
                "extra_env": {"B": "OVER", "C": "3"},
            },
        )

        # Per-launch ``B`` overwrites constructor ``B``; ``A``
        # survives; ``C`` is added.
        assert captured["extra_env"] == {
            "A": "1", "B": "OVER", "C": "3"}

    def test_extra_mounts_appends_per_launch_after(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
    ) -> None:
        captured = _captured_kwargs()
        inner = _FakeInnerHandle(
            result=_agent_result(),
            workspace=workspace,
            write_execution=_block_execution(),
        )
        _patch_block_launcher(
            monkeypatch, inner=inner, captured=captured)

        executor = AgentExecutor(
            template=template,
            project_root=project_root,
            extra_mounts=[("/host/a", "/cont/a", "ro")],
        )

        executor.launch(
            block_name="play",
            workspace=workspace,
            input_bindings={},
            overrides={
                "prompt": "p",
                "extra_mounts": [
                    ("/host/b", "/cont/b", "rw"),
                ],
            },
        )

        assert captured["extra_mounts"] == [
            ("/host/a", "/cont/a", "ro"),
            ("/host/b", "/cont/b", "rw"),
        ]

    def test_prompt_substitutions_applied(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
    ) -> None:
        captured = _captured_kwargs()
        inner = _FakeInnerHandle(
            result=_agent_result(),
            workspace=workspace,
            write_execution=_block_execution(),
        )
        _patch_block_launcher(
            monkeypatch, inner=inner, captured=captured)

        executor = AgentExecutor(
            template=template, project_root=project_root)

        executor.launch(
            block_name="play",
            workspace=workspace,
            input_bindings={},
            overrides={
                "prompt": "hello {{NAME}}, run {{N}} times",
                "prompt_substitutions": {
                    "NAME": "world", "N": 3},
            },
        )

        assert captured["prompt"] == "hello world, run 3 times"


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


class TestValidation:
    """Missing / empty prompt fails fast at launch time."""

    @pytest.mark.parametrize("bad", [None, "", 0, 42, []])
    def test_missing_prompt_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
        bad: Any,
    ) -> None:
        captured = _captured_kwargs()
        inner = _FakeInnerHandle(result=_agent_result())
        _patch_block_launcher(
            monkeypatch, inner=inner, captured=captured)

        executor = AgentExecutor(
            template=template, project_root=project_root)

        overrides: dict[str, Any] = (
            {} if bad is None else {"prompt": bad})

        with pytest.raises(ValueError):
            executor.launch(
                block_name="play",
                workspace=workspace,
                input_bindings={},
                overrides=overrides or None,
            )

        # The launcher must not have been invoked.
        assert captured == {}


# --------------------------------------------------------------------
# Returned handle satisfies the ExecutionHandle protocol.
# --------------------------------------------------------------------


class TestExecutionHandleSemantics:
    """``AgentExecutionHandle`` adapts the inner handle properly."""

    def test_is_alive_and_stop_forward(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
    ) -> None:
        captured = _captured_kwargs()
        inner = _FakeInnerHandle(
            result=_agent_result(),
            workspace=workspace,
            write_execution=_block_execution(),
        )
        _patch_block_launcher(
            monkeypatch, inner=inner, captured=captured)

        executor = AgentExecutor(
            template=template, project_root=project_root)
        handle = executor.launch(
            block_name="play",
            workspace=workspace,
            input_bindings={},
            overrides={"prompt": "p"},
        )

        assert isinstance(handle, AgentExecutionHandle)
        assert handle.is_alive() is True

        handle.stop(reason="custom-reason")
        assert inner.stop_calls == ["custom-reason"]

    def test_wait_uses_block_execution_record(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
    ) -> None:
        captured = _captured_kwargs()
        execution = _block_execution(
            execution_id="exec-real",
            status="succeeded",
            output_bindings={"out_slot": "art-out-1"},
            exit_code=0,
        )
        inner = _FakeInnerHandle(
            result=_agent_result(execution_id="exec-real"),
            workspace=workspace,
            write_execution=execution,
        )
        _patch_block_launcher(
            monkeypatch, inner=inner, captured=captured)

        executor = AgentExecutor(
            template=template, project_root=project_root)
        handle = executor.launch(
            block_name="play",
            workspace=workspace,
            input_bindings={},
            overrides={"prompt": "p"},
        )
        result = handle.wait()

        assert isinstance(result, ExecutionResult)
        assert result.execution_id == "exec-real"
        assert result.status == "succeeded"
        assert result.output_bindings == {"out_slot": "art-out-1"}
        assert result.exit_code == 0
        assert result.elapsed_s == pytest.approx(0.5)
        assert handle.is_alive() is False

    def test_wait_is_idempotent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
    ) -> None:
        captured = _captured_kwargs()
        inner = _FakeInnerHandle(
            result=_agent_result(execution_id="exec-x"),
            workspace=workspace,
            write_execution=_block_execution(
                execution_id="exec-x"),
        )
        _patch_block_launcher(
            monkeypatch, inner=inner, captured=captured)

        executor = AgentExecutor(
            template=template, project_root=project_root)
        handle = executor.launch(
            block_name="play",
            workspace=workspace,
            input_bindings={},
            overrides={"prompt": "p"},
        )
        first = handle.wait()
        second = handle.wait()

        assert first is second

    def test_agent_result_exposed_after_wait(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
    ) -> None:
        captured = _captured_kwargs()
        ar = _agent_result(
            execution_id="exec-ar", exit_reason="max_turns")
        inner = _FakeInnerHandle(
            result=ar,
            workspace=workspace,
            write_execution=_block_execution(
                execution_id="exec-ar"),
        )
        _patch_block_launcher(
            monkeypatch, inner=inner, captured=captured)

        executor = AgentExecutor(
            template=template, project_root=project_root)
        handle = executor.launch(
            block_name="play",
            workspace=workspace,
            input_bindings={},
            overrides={"prompt": "p"},
        )

        with pytest.raises(RuntimeError):
            handle.agent_result()

        handle.wait()
        assert handle.agent_result() is ar
        assert handle.agent_result().exit_reason == "max_turns"

    def test_wait_handles_missing_execution_record(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
    ) -> None:
        # Pre-container failure: the agent never produced a
        # durable :class:`BlockExecution`.  AgentExecutionHandle
        # must not crash; status is "failed", output_bindings
        # is empty.
        captured = _captured_kwargs()
        inner = _FakeInnerHandle(
            result=_agent_result(
                execution_id=None, exit_code=42,
                exit_reason="crashed"),
            workspace=workspace,
            write_execution=None,
        )
        _patch_block_launcher(
            monkeypatch, inner=inner, captured=captured)

        executor = AgentExecutor(
            template=template, project_root=project_root)
        handle = executor.launch(
            block_name="play",
            workspace=workspace,
            input_bindings={},
            overrides={"prompt": "p"},
        )
        result = handle.wait()

        assert result.status == "failed"
        assert result.output_bindings == {}
        assert result.exit_code == 42
        assert result.execution_id == ""


# --------------------------------------------------------------------
# Handoff path
# --------------------------------------------------------------------


class TestHandoffPath:
    """When ``block_runner`` is set, the executor routes through
    :func:`launch_agent_with_handoffs` and forwards the loop-level
    constructor defaults."""

    def test_handoff_launcher_receives_loop_defaults(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workspace: _FakeWorkspace,
        project_root: Path,
        template: Any,
    ) -> None:
        captured = _captured_kwargs()
        inner = _FakeInnerHandle(
            result=_agent_result(),
            workspace=workspace,
            write_execution=_block_execution(),
        )
        _patch_handoff_launcher(
            monkeypatch, inner=inner, captured=captured)

        # Block-launcher path must NOT be taken on this branch.
        def _explode(**_: Any) -> Any:
            raise AssertionError(
                "launch_agent_block called when handoff path "
                "should have been taken"
            )

        monkeypatch.setattr(
            agent_executor_mod, "launch_agent_block", _explode)

        def _runner(_: HandoffContext) -> HandoffResult:
            raise AssertionError("not invoked in this test")

        halts: list[Any] = []

        def _halt_source() -> list[Any]:
            return list(halts)

        def _post(_: HandoffContext) -> None:
            return None

        executor = AgentExecutor(
            template=template,
            project_root=project_root,
            block_runner=_runner,
            halt_source=_halt_source,
            post_dispatch_fn=_post,
            resume_prompt="resume!",
            max_iterations=7,
        )

        executor.launch(
            block_name="play",
            workspace=workspace,
            input_bindings={},
            overrides={"prompt": "p"},
        )

        assert captured["block_runner"] is _runner
        assert captured["halt_source"] is _halt_source
        assert captured["post_dispatch_fn"] is _post
        assert captured["resume_prompt"] == "resume!"
        assert captured["max_iterations"] == 7
        # The agent-block kwargs must still be present.
        assert captured["block_name"] == "play"
        assert captured["prompt"] == "p"
