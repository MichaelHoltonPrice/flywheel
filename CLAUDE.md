# Flywheel

Python orchestration framework for measurable AI improvement loops. Wires Docker containers together, tracks artifacts, and provides visibility into hybrid human+agent workflows.

## Key concepts

- **Blocks with I/O contracts** are the fundamental unit. See `docs/vision.md`.
- **All project logic runs in Docker containers.** Flywheel on the host is pure orchestration.
- **Workspaces** compartmentalize work within a run.
- **Artifacts** are either dumb (file/directory copies stored in a workspace) or git-based (references to version-controlled code).
- **Patterns**: container, sequential_pipeline, sweep.

## Batteries (solve-once capabilities)

- Computer use agent (CUA) — X11 session management, screenshot loop
- Claude Code / Codex agent wrappers — container launch, auth, MCP tool exposure

## Build and test

```bash
pip install -e ".[dev]"
bash scripts/verify.sh
```

Run `scripts/verify.sh` before considering any work complete. It runs ruff and pytest.

Before committing, also run a review agent to check for:
- Test quality (coverage gaps, edge cases, phantom tests)
- Docstring completeness (Args/Returns/Raises sections for public functions with parameters — ruff does not enforce this)

## Project layout

Flywheel expects a `flywheel.yaml` in the target project root defining `harness_dir` (the workforce directory). Templates and workspaces live under `harness_dir`.
