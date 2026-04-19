"""Shared constants for the container runtime contract.

The contract is specified in ``cyber-root/substrate-contract.md``
(migrating to ``flywheel/docs/runtime-contract.md`` when the
reshape lands).  This module is the single source of truth for
the constants the contract names — paths, failure phases, stop
timeouts.  Executors import them; block-side code inside a
container should not depend on this module (it's flywheel-only).
"""

from __future__ import annotations

from typing import Final

# ── Filesystem conventions ──────────────────────────────────────

STATE_MOUNT_PATH: Final[str] = "/state"
"""Container-side path where flywheel mounts `/state/` for blocks
that declare ``state: true``.  Populated at container start from
the prior execution's state dir; captured at container exit."""

STOP_SENTINEL_WORKSPACE_RELATIVE: Final[str] = ".stop"
"""Workspace-relative path of the cooperative cancellation
sentinel.  Flywheel writes this file to request a clean shutdown;
the container polls for it and exits when it appears.  The full
container-side path depends on where the workspace is mounted
(typically ``/workspace/.stop``)."""

RUNTIME_SOCKET_WORKSPACE_RELATIVE: Final[str] = ".runtime.sock"
"""Workspace-relative path of the Unix domain socket used by the
request-response protocol.  Persistent containers bind this
socket; flywheel connects to it to issue per-execution requests.
Unused by process-exit blocks."""

# ── Failure phases ──────────────────────────────────────────────
#
# Set on ``BlockExecution.failure_phase`` when an execution ends
# with ``status="failed"``.  ``None`` on successful executions.
# The phase names are stable ledger values; prefer the constants
# below over string literals in executor code.

FAILURE_STAGE_IN: Final[str] = "stage_in"
"""Staging an input into its per-mount dir failed.  Container
never started."""

FAILURE_INVOKE: Final[str] = "invoke"
"""The container body itself signaled failure: non-zero exit for
process-exit, error response for request-response."""

FAILURE_STATE_CAPTURE: Final[str] = "state_capture"
"""Container exited but ``/state/`` could not be captured into
the workspace state dir.  The body may have succeeded; the
execution is still recorded as failed."""

FAILURE_OUTPUT_COLLECT: Final[str] = "output_collect"
"""Container exited but one or more declared output dirs could
not be read or registered."""

FAILURE_ARTIFACT_COMMIT: Final[str] = "artifact_commit"
"""Outputs were read but the workspace commit failed (e.g., disk
full, concurrent write race, invalid instance shape)."""

FAILURE_PHASES: Final[frozenset[str]] = frozenset({
    FAILURE_STAGE_IN,
    FAILURE_INVOKE,
    FAILURE_STATE_CAPTURE,
    FAILURE_OUTPUT_COLLECT,
    FAILURE_ARTIFACT_COMMIT,
})
"""All valid non-``None`` ``BlockExecution.failure_phase`` values.

Used by validators and tests.  An executor that sets a phase not
in this set is a bug."""
