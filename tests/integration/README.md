# Integration tests

Tests in this directory talk to real external services (the
Anthropic API, Docker daemons, etc.).  They are kept out of the
default test run because they:

- need network access,
- need credentials,
- are slow,
- and cost money / API quota each time they run.

Every test here is decorated with `@pytest.mark.live_api` and
the top-level `tests/conftest.py` skips marked tests unless
the opt-in flag `--run-live-api` is passed on the command line.

## Running them

From the `flywheel/` directory:

```
pytest tests/integration --run-live-api -v
```

To run a single file:

```
pytest tests/integration/test_full_stop_b1_splice_roundtrip.py --run-live-api -v
```

To run a single test by name:

```
pytest tests/integration --run-live-api -v -k splice_roundtrip
```

## Authentication

The Claude Agent SDK picks up authentication from the same
sources Claude Code itself uses:

- An active Claude Code login (preferred when running this
  inside a Claude Code session).
- The `ANTHROPIC_API_KEY` environment variable.
- The standard credential file at `~/.claude/credentials.json`.

If no auth is configured, the test fails fast with a clear
error rather than a long timeout.

## What lives here

- `test_full_stop_b1_splice_roundtrip.py` — splice round-trip
  with Haiku (no container).
- `test_full_stop_b1_container_roundtrip.py` — agent-in-Docker
  handoff cycle with Haiku.
- `test_full_stop_b1_two_container_roundtrip.py` — agent and
  block runner each in their own Docker container.

## Crash-resume coverage

Every host-side failure mode of the handoff loop has a
deterministic regression test in
`flywheel/tests/test_handoff_resume.py` (no live API, no
Docker, runs in <1s).  That file is the source of truth for
the loop's recovery contract; integration tests here only
add evidence that real containers obey the same contract.

Container-internal kill points are intentionally NOT
automated.  These boundaries live inside the agent runner
process (kill mid-PreToolUse hook, kill between deny
tool_result and clean exit, kill mid-resume), and the timing
windows are too narrow to hit reliably from a test harness.
When investigating a real-world incident at one of these
boundaries, exercise it manually:

1. Run `test_full_stop_b1_container_roundtrip.py` with
   `--run-live-api` and a deliberately throttled prompt
   (e.g. add `time.sleep` to the test MCP server).
2. From a second shell, `docker kill <container-name>` at
   the suspected crash point.  The test container name is
   printed at startup; grep stderr for it.
3. Inspect the bind-mounted workspace under `tmp_path`:
   `pending_tool_calls.json`, `agent_session.jsonl`, and
   `.agent_state.json` are the three artifacts the host
   loop consults on the next launch.  The
   `find_pending_deny_tool_use_ids` helper in
   `flywheel.session_splice` enumerates unspliced
   tool_use_ids from a saved JSONL.
4. Re-invoke `run_agent_with_handoffs` against the same
   workspace and observe whether the recovery matches what
   `test_handoff_resume.py::TestRedriveAfterFailure`
   asserts for the equivalent host-side scenario.

If you discover a container-internal failure mode whose
recovery doesn't match the unit-test contract, that's a real
bug — open an issue with the workspace state captured.
