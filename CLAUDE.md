# Flywheel

Python orchestration framework for measurable AI improvement loops. Wires Docker containers together, tracks artifacts, and provides visibility into hybrid human+agent workflows.

## Key concepts

- **Block execution** is the atomic operation. One block, one Docker container, one set of inputs producing outputs. See `docs/vision.md` (aspirational) and `docs/architecture.md` (implemented or agreed-upon decisions; future work is noted separately at the end).
- **All project logic runs in Docker containers.** Flywheel on the host is pure orchestration.
- **Artifacts** have two levels: declarations (template-level types like "checkpoint" or "score") and instances (concrete, immutable records produced by block executions). Two storage kinds exist: copy artifacts (files in the workspace) and git artifacts (references to version-controlled code).
- **Workspaces** are created from templates and accumulate artifact instances and execution records over their lifetime. The workspace is a history, not a snapshot.
- **The foundry** is the flywheel-managed directory within a project, holding templates and workspaces. It is a peer to the project source code.
- **Agent blocks** are a special block execution variant where an AI agent can trigger nested block executions via a block bridge HTTP service. The bridge reads block definitions from the template -- what those blocks do (evaluation, validation, etc.) is project-specific, not a flywheel concern. See `flywheel/agent.py` and `flywheel/block_bridge.py`.

## Batteries (solve-once capabilities)

- Computer use agent (CUA) -- X11 session management, screenshot loop
- Claude Code agent wrapper (`batteries/claude/`) -- Dockerfile, agent runner with pause/resume support, MCP server for nested block invocation. See `docs/architecture.md` "Agent pause and resume" for the mechanism.

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

Flywheel expects a `flywheel.yaml` in the target project root defining `foundry_dir` (the foundry directory). Templates and workspaces live under `foundry_dir`.
