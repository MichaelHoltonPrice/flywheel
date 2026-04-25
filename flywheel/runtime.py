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

STATE_MOUNT_PATH: Final[str] = "/flywheel/state"
"""Container-side path where flywheel mounts state for blocks that
declare ``state: managed``.  Populated at container start from the
latest compatible state snapshot; captured after clean exit.  Lives
under ``/flywheel/`` so every flywheel-owned mount is under a
single namespace distinct from the block body's inputs/outputs."""

FLYWHEEL_CONTROL_MOUNT: Final[str] = "/flywheel/control"
"""Container-side path where agent-style blocks receive
framework-owned control files (e.g. ``pending_tool_calls.json``,
``agent_exit_state.json``, ``.agent_resume``).  Agent-specific —
not written by the generic executor.  The agent launcher mounts
a host tempdir here and reads any control files the block wrote
after exit.  Not in the artifact graph."""

FLYWHEEL_MCP_SERVERS_MOUNT: Final[str] = "/flywheel/mcp_servers"
"""Container-side path where projects mount MCP server code.
Agent-specific; read-only.  The agent runner discovers
``*_mcp_server.py`` files here at startup."""

STOP_SENTINEL_WORKSPACE_RELATIVE: Final[str] = ".stop"
"""Workspace-relative path of the cooperative cancellation
sentinel.  Flywheel writes this file to request a clean shutdown;
the container polls for it and exits when it appears.  The full
container-side path depends on where the workspace is mounted
(typically ``/scratch/.stop``)."""

REQUEST_TREE_WORKSPACE_RELATIVE: Final[str] = "requests"
"""Workspace-relative parent directory for request-response
per-request I/O trees.  For request ``id``, flywheel creates
``/scratch/requests/<id>/input/<slot>/`` (read-only staged
inputs) and ``/scratch/requests/<id>/output/<slot>/`` (empty
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
"""Container exited but ``/flywheel/state`` could not be captured
into a managed state snapshot.  The body may have succeeded; the
execution is still recorded as failed."""

FAILURE_STATE_VALIDATE: Final[str] = "state_validate"
"""Container exited cleanly but a project state validator rejected
the candidate state before it was registered as a snapshot."""

FAILURE_OUTPUT_COLLECT: Final[str] = "output_collect"
"""Container exited but one or more declared output dirs could
not be read or registered."""

FAILURE_OUTPUT_VALIDATE: Final[str] = "output_validate"
"""Container exited and outputs were read, but a project-declared
validator (see :mod:`flywheel.artifact_validator`) rejected one
or more declared output slots.  Slots whose validator passed are
still committed; rejected slots are not.  The ledger
``error`` field carries the per-slot rejection reasons."""

FAILURE_ARTIFACT_COMMIT: Final[str] = "artifact_commit"
"""Outputs were read but the workspace commit failed (e.g., disk
full, concurrent write race, invalid instance shape)."""

FAILURE_OUTPUT_PROTOCOL: Final[str] = "output_protocol"
"""Block exited cleanly but failed to follow the termination-channel
protocol — i.e., the substrate normalized the announced reason to
``protocol_violation``.  See
:data:`TERMINATION_REASON_PROTOCOL_VIOLATION` and the
``block-execution`` spec."""

FAILURE_PHASES: Final[frozenset[str]] = frozenset({
    FAILURE_STAGE_IN,
    FAILURE_INVOKE,
    FAILURE_STATE_CAPTURE,
    FAILURE_STATE_VALIDATE,
    FAILURE_OUTPUT_COLLECT,
    FAILURE_OUTPUT_VALIDATE,
    FAILURE_ARTIFACT_COMMIT,
    FAILURE_OUTPUT_PROTOCOL,
})
"""All valid non-``None`` ``BlockExecution.failure_phase`` values.

Used by validators and tests.  An executor that sets a phase not
in this set is a bug."""

# ── Termination reasons ─────────────────────────────────────────
#
# A first-class field on :class:`flywheel.artifact.BlockExecution`
# (``termination_reason``).  Always populated.  Either a
# substrate-reserved value the substrate set based on what it
# observed, or a project-defined value the block announced via
# the per-runtime termination channel.  The ledger value is a
# plain string; these constants are the substrate-reserved names
# only.  See ``flywheel/docs/specs/block-execution.md`` for the
# full contract.

TERMINATION_REASON_CRASH: Final[str] = "crash"
"""Block process exited non-zero, the container died unexpectedly,
the persistent server raised, or the connection dropped."""

TERMINATION_REASON_INTERRUPTED: Final[str] = "interrupted"
"""Block was killed by stop signal or operator interrupt."""

TERMINATION_REASON_TIMEOUT: Final[str] = "timeout"
"""Block exceeded a substrate-enforced deadline.  The constant is
reserved now so the vocabulary is stable; deadline enforcement
itself lands later."""

TERMINATION_REASON_PROTOCOL_VIOLATION: Final[str] = "protocol_violation"
"""Block exited cleanly but failed to follow the
termination-channel protocol — no announcement, an announcement
that collides with a reserved name, or an announcement that does
not match any declared project reason.  The substrate substitutes
this value before recording."""

RESERVED_TERMINATION_REASONS: Final[frozenset[str]] = frozenset({
    TERMINATION_REASON_CRASH,
    TERMINATION_REASON_INTERRUPTED,
    TERMINATION_REASON_TIMEOUT,
    TERMINATION_REASON_PROTOCOL_VIOLATION,
})
"""All substrate-reserved ``termination_reason`` values.  A block
that announces any of these via the termination channel is a
protocol violation."""

DEFAULT_TERMINATION_REASON: Final[str] = "normal"
"""Convention default termination-reason label used by happy-path
blocks that declare a single clean-exit pathway.  Not reserved —
projects may use a different label.  The block-declaration
parser maps a legacy flat ``outputs:`` list to this label so
existing single-reason blocks parse without YAML migration."""

TERMINATION_PATH: Final[str] = "/flywheel/termination"
"""Container-side path for the one-shot runtime's termination
channel.  The block writes a single line of UTF-8 with the
termination reason; the executor reads it after the container
exits.  See ``flywheel/docs/specs/block-execution.md`` §
"Runtime mechanism for termination reasons"."""

TERMINATION_REASON_RESPONSE_FIELD: Final[str] = "termination_reason"
"""JSON field name in the persistent runtime's ``/execute``
response body carrying the block's announced termination reason.
Top-level string field; absent / non-string / missing is
normalized to ``crash`` (transport failures) or ignored when the
substrate observed something stronger."""
