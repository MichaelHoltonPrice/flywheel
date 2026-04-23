"""Tests for the shared runtime-contract constants.

These constants are the single source of truth for the paths and
failure-phase identifiers named in the container runtime
contract.  Tests here lock in the names against accidental
rename / drift — executors and container authors both depend on
them, so a silent change would break either side.
"""

from __future__ import annotations

from flywheel import runtime


class TestPathConstants:
    def test_state_mount_path(self):
        # Block authors look for this path; the executor mounts
        # there when a block declares ``state: true``.
        assert runtime.STATE_MOUNT_PATH == "/flywheel/state"

    def test_flywheel_control_mount(self):
        # Agent-specific framework-owned control files; the agent
        # launcher mounts a host tempdir here.
        assert (
            runtime.FLYWHEEL_CONTROL_MOUNT == "/flywheel/control"
        )

    def test_flywheel_mcp_servers_mount(self):
        # Agent-specific MCP server code mount path.
        assert (
            runtime.FLYWHEEL_MCP_SERVERS_MOUNT
            == "/flywheel/mcp_servers"
        )

    def test_stop_sentinel_relative(self):
        # Workspace-relative path; absolute location depends on
        # where the workspace is mounted in the container.
        assert runtime.STOP_SENTINEL_WORKSPACE_RELATIVE == ".stop"

    def test_request_tree_relative(self):
        # Workspace-relative parent directory for request-response
        # per-request I/O trees.
        assert (
            runtime.REQUEST_TREE_WORKSPACE_RELATIVE
            == "requests"
        )

    def test_control_port_env_var(self):
        # Executor passes the allocated localhost port via this
        # env var to the request-response container.
        assert (
            runtime.CONTROL_PORT_ENV_VAR
            == "FLYWHEEL_CONTROL_PORT"
        )


class TestFailurePhases:
    def test_named_phase_constants(self):
        # These string values end up in the workspace ledger.
        # They are stable public identifiers; changing one is a
        # breaking change for any tooling that reads the ledger.
        assert runtime.FAILURE_STAGE_IN == "stage_in"
        assert runtime.FAILURE_INVOKE == "invoke"
        assert runtime.FAILURE_STATE_CAPTURE == "state_capture"
        assert runtime.FAILURE_OUTPUT_COLLECT == "output_collect"
        assert runtime.FAILURE_OUTPUT_VALIDATE == "output_validate"
        assert runtime.FAILURE_ARTIFACT_COMMIT == "artifact_commit"

    def test_failure_phases_set_contains_all_named(self):
        # The frozenset used by validators must include every
        # named constant above — adding a new phase without
        # updating the set is the kind of drift the test catches.
        expected = {
            runtime.FAILURE_STAGE_IN,
            runtime.FAILURE_INVOKE,
            runtime.FAILURE_STATE_CAPTURE,
            runtime.FAILURE_OUTPUT_COLLECT,
            runtime.FAILURE_OUTPUT_VALIDATE,
            runtime.FAILURE_ARTIFACT_COMMIT,
        }
        assert expected == runtime.FAILURE_PHASES

    def test_failure_phases_set_is_frozen(self):
        # Frozen so callers can safely share the reference.
        assert isinstance(runtime.FAILURE_PHASES, frozenset)
