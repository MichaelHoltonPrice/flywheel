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

REQUEST_TREE_WORKSPACE_RELATIVE: Final[str] = "requests"
"""Workspace-relative parent directory for request-response
per-request I/O trees.  For request ``id``, flywheel creates
``/workspace/requests/<id>/input/<slot>/`` (read-only staged
inputs) and ``/workspace/requests/<id>/output/<slot>/`` (empty
write targets) before each ``POST /execute`` call.  Unused by
process-exit blocks."""

CONTROL_PORT_ENV_VAR: Final[str] = "FLYWHEEL_CONTROL_PORT"
"""Environment variable used to pass the host-allocated control-
channel port into a request-response container.  The executor
picks a free localhost port, binds the container to publish on
it, and sets this env var so the container's in-process HTTP
server knows which port to listen on.  Transport choice (today
TCP; could be a socket path in a future implementation) is an
executor concern — block-side code reads the relevant env var
and binds accordingly."""

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
