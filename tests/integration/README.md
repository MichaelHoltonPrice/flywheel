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

Currently only the B1 splice round-trip lives here.  As the
campaign progresses, B8 will populate this directory with
crash-resume integration tests covering every boundary state
described in `plans/full-stop-state-contract.md`.
