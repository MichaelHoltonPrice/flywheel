"""Client-side helpers for tool-triggered logical block executions.

Tool authors (e.g., MCP servers running inside agent containers)
use this module to wrap their side-effecting work in the
block-execution lifecycle: a ``begin`` HTTP call to the
``ExecutionChannel`` opens a ledger row with declared inputs;
the work runs; an ``end`` HTTP call closes the ledger row and
atomically registers any produced output artifacts.

The primary entry point is the :class:`BlockChannelClient` plus
its :meth:`BlockChannelClient.begin` context manager:

.. code-block:: python

    from flywheel.tool_block import BlockChannelClient

    client = BlockChannelClient.from_env()

    with client.begin(
        block="predict",
        caller={"mcp_server": "arc", "tool": "predict_action"},
        params={"action_id": action, "x": x, "y": y},
    ) as ctx:
        # ctx.input_paths gives freshly-resolved artifact paths
        # ctx.params echoes back what we sent
        result = run_the_work(ctx.input_paths, ctx.params)
        ctx.set_output("prediction", {"predicted_state": result})

If the body raises, the context manager calls ``end`` with
``status="failed"`` and re-raises.  No partial outputs are
registered.

Environment contract for ``from_env``:

- ``EXEC_CHANNEL_URL``: base URL of the execution channel.  Falls
  back to ``EVAL_ENDPOINT`` for compatibility with the legacy
  bridge env name.
- ``EXEC_PARENT_ID`` (optional): parent execution ID set by the
  agent runner for tool calls launched inside an agent container.

The client does not import any flywheel internals beyond the
standard library, so it is safe to vendor or use from minimal
container images.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError


class BlockChannelError(RuntimeError):
    """Raised when the execution channel returns a structured error.

    Attributes:
        error_type: The ``error_type`` field from the channel
            response (e.g., ``"unknown_block"``,
            ``"missing_input"``).
        retryable: Whether the channel marked the failure as
            retryable.
        payload: The full response payload from the channel.
    """

    def __init__(
        self,
        message: str,
        error_type: str,
        retryable: bool = False,
        payload: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
        self.payload = payload or {}


@dataclass
class ExecutionContext:
    """Per-execution context handed to a block body.

    Created by ``BlockChannelClient.begin`` after the channel
    accepts the begin request.  Body code reads inputs, sets
    outputs, and optionally records error state via this object.

    Attributes:
        execution_id: The ledger ID assigned by the channel.
        block_name: The block this execution corresponds to.
        input_bindings: ``{slot_name: artifact_id}`` resolved by
            the channel at begin time.
        input_paths: ``{slot_name: absolute path on host}`` to the
            artifact directory.  These paths are valid on the
            *host* filesystem; tool authors running inside a
            container will need a separate mechanism (e.g., a
            shared mount or the channel's GET endpoint) to read
            them.  Phase 1 surfaces the paths for ledger debugging
            and for inprocess runners that share the filesystem.
        input_hashes: Reserved for future use.  Currently always
            ``None`` per slot.
        scratch_dir: Per-execution scratch directory under
            ``<workspace>/execution_scratch/<execution_id>``.
            Cleaned up by the channel after end.
        params: The params dict echoed back from the begin call.
        parent_execution_id: The parent execution if any.
        outputs: Buffer of name → JSON-serializable data to send
            on end.  Use ``set_output`` to populate.
        started_at: Monotonic wall-clock at begin (for elapsed_s).
    """

    execution_id: str
    block_name: str
    input_bindings: dict[str, str]
    input_paths: dict[str, str]
    input_hashes: dict[str, str | None]
    scratch_dir: str
    params: dict[str, Any]
    parent_execution_id: str | None
    started_at: float
    outputs: dict[str, Any] = field(default_factory=dict)

    def set_output(self, name: str, data: Any) -> None:
        """Record a JSON-serializable artifact for later registration.

        Args:
            name: The output slot name as declared by the block.
            data: The artifact contents.  Will be JSON-encoded
                when ``end`` is called.

        Raises:
            ValueError: If an output for this name was already
                set on this context.
        """
        if name in self.outputs:
            raise ValueError(
                f"Output {name!r} already set on execution "
                f"{self.execution_id}"
            )
        self.outputs[name] = data


@dataclass
class BlockChannelClient:
    """HTTP client for the execution channel's lifecycle API.

    Construct via :meth:`from_env` to pick up the channel URL
    from container environment variables, or pass ``base_url``
    directly for tests.

    Attributes:
        base_url: Base URL of the execution channel
            (e.g., ``http://host.docker.internal:9123``).
        parent_execution_id: Default parent execution to pass to
            ``begin`` calls.  Inherited from the
            ``EXEC_PARENT_ID`` env var when constructed via
            :meth:`from_env`.
        timeout: HTTP timeout in seconds.
    """

    base_url: str
    parent_execution_id: str | None = None
    timeout: float = 30.0

    @classmethod
    def from_env(
        cls,
        url_var: str = "EXEC_CHANNEL_URL",
        fallback_var: str = "EVAL_ENDPOINT",
        parent_var: str = "EXEC_PARENT_ID",
        timeout: float = 30.0,
    ) -> BlockChannelClient:
        """Build a client from environment variables.

        Args:
            url_var: Primary env var for the channel URL.
            fallback_var: Fallback env var for compatibility with
                the legacy bridge env name.
            parent_var: Env var for the parent execution ID, if
                this client is being constructed inside a child
                process whose parent is an agent execution.
            timeout: HTTP timeout in seconds.

        Returns:
            A configured client.

        Raises:
            RuntimeError: If neither env var is set.
        """
        url = os.environ.get(url_var) or os.environ.get(fallback_var)
        if not url:
            raise RuntimeError(
                f"BlockChannelClient.from_env: neither "
                f"{url_var} nor {fallback_var} is set"
            )
        return cls(
            base_url=url.rstrip("/"),
            parent_execution_id=os.environ.get(parent_var) or None,
            timeout=timeout,
        )

    @contextmanager
    def begin(
        self,
        block: str,
        params: dict[str, Any] | None = None,
        caller: dict[str, Any] | None = None,
        runner: str | None = None,
        parent_execution_id: str | None = None,
    ):
        """Open a logical block execution as a context manager.

        Calls ``POST /execution/begin``, yields an
        :class:`ExecutionContext`, and on exit calls
        ``POST /execution/end/{id}`` with the buffered outputs.

        Args:
            block: The block name as declared in the template.
            params: Function-argument parameters for ledger
                provenance.  Should be JSON-serializable.
            caller: Identifies the source (e.g.,
                ``{"mcp_server": "arc", "tool": "predict_action"}``).
            runner: Caller-declared runner type.  Optional in
                Phase 1; will be enforced via the manifest in
                Phase 3.
            parent_execution_id: Override the client's default
                parent execution ID for this call only.

        Yields:
            ExecutionContext for populating outputs.

        Raises:
            BlockChannelError: If the channel rejects begin or end.
            URLError, HTTPError: For transport-level failures.
        """
        body = {
            "block": block,
            "params": params or {},
            "caller": caller,
            "runner": runner,
            "parent_execution_id": (
                parent_execution_id or self.parent_execution_id),
        }
        response = self._post_json("/execution/begin", body)

        if not response.get("ok"):
            raise BlockChannelError(
                message=response.get(
                    "message", "execution channel rejected begin"),
                error_type=response.get(
                    "error_type", "unknown_error"),
                retryable=response.get("retryable", False),
                payload=response,
            )

        ctx = ExecutionContext(
            execution_id=response["execution_id"],
            block_name=block,
            input_bindings=response.get("input_bindings", {}),
            input_paths=response.get("input_paths", {}),
            input_hashes=response.get("input_hashes", {}),
            scratch_dir=response.get("scratch_dir", ""),
            params=params or {},
            parent_execution_id=response.get("parent_execution_id"),
            started_at=time.monotonic(),
        )

        body_failed: BaseException | None = None
        try:
            yield ctx
        except BaseException as exc:  # includes KeyboardInterrupt
            body_failed = exc
            raise
        finally:
            elapsed = time.monotonic() - ctx.started_at
            if body_failed is None:
                end_body = {
                    "status": "ok",
                    "outputs": ctx.outputs,
                    "elapsed_s": elapsed,
                }
            else:
                end_body = {
                    "status": "failed",
                    "error": (
                        f"{type(body_failed).__name__}: "
                        f"{body_failed}"),
                    "elapsed_s": elapsed,
                }
            try:
                self._post_json(
                    f"/execution/end/{ctx.execution_id}", end_body)
            except Exception:
                # Don't mask the original exception; swallow the
                # secondary end failure.  The channel's row will
                # remain in "running" state, surfacing as a
                # ledger anomaly.
                if body_failed is None:
                    raise

    def _post_json(
        self, path: str, body: dict[str, Any],
    ) -> dict[str, Any]:
        """POST a JSON body and return the parsed JSON response."""
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                    req, timeout=self.timeout) as resp:
                raw = resp.read()
        except HTTPError as e:
            # The channel uses HTTP status codes for transport
            # errors; structured errors are returned as 200 with
            # ok=false.  For HTTPError, try to parse body as JSON.
            try:
                raw = e.read()
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                raise
            raise BlockChannelError(
                message=payload.get(
                    "message", f"HTTP {e.code} from channel"),
                error_type=payload.get(
                    "error_type", f"http_{e.code}"),
                retryable=payload.get("retryable", False),
                payload=payload,
            ) from e
        except URLError:
            raise

        return json.loads(raw.decode("utf-8"))
