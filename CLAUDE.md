# Flywheel

Python orchestration framework for measurable AI improvement loops. Wires Docker containers together, tracks artifacts, and provides visibility into hybrid human+agent workflows.

## Key concepts

- **Block execution** is the atomic operation. One block, one Docker container, one set of inputs producing outputs. See `docs/vision.md` (aspirational) and `docs/architecture.md` (implemented or agreed-upon decisions; future work is noted separately at the end).
- **All project logic runs in Docker containers.** Flywheel on the host is pure orchestration.
- **Artifacts** have two levels: declarations (template-level types like "checkpoint" or "score") and instances (concrete, immutable records produced by block executions). Two storage kinds exist: copy artifacts (files in the workspace) and git artifacts (references to version-controlled code).
- **Workspaces** are created from templates and accumulate artifact instances and execution records over their lifetime. The workspace is a history, not a snapshot.
- **The foundry** is the flywheel-managed directory within a project, holding templates and workspaces. It is a peer to the project source code.
- **Agent blocks** are a special block execution variant where an AI agent can trigger nested block executions via a block bridge HTTP service. The bridge reads block definitions from the template -- what those blocks do (evaluation, validation, etc.) is project-specific, not a flywheel concern. The bridge supports two modes: **invoke** (launches a Docker container) and **record** (creates artifacts without containers, for provenance tracking of actions that already happened). Record-mode blocks use the `__record__` sentinel as their image. See `flywheel/agent.py` and `flywheel/block_bridge.py`.
- **Non-blocking agent handle**: `launch_agent_block()` returns an `AgentHandle` with `kill()`, `wait()`, and `is_alive()`. Enables system-controlled agent interruption (e.g., killing an agent from a bridge callback). `run_agent_block()` is a blocking wrapper.
- **Bridge record callback**: `BlockBridgeService` accepts `on_record` — a callback fired after each successful record-mode invocation. Runs in the bridge HTTP thread.
- **Session artifacts**: The agent runner exports the SDK session JSONL to the workspace on exit and can resume from a session artifact via `RESUME_SESSION_FILE`. Sessions are regular copy artifacts with standard provenance.
- **Lifecycle tracking**: `BlockExecution` has `stop_reason` and `predecessor_id` for tracking why agents were stopped and linking resume chains. `AgentHandle.wait()` records the agent itself as a `BlockExecution` (block_name=`__agent__`). `LifecycleEvent` captures operational events (agent stops, group completions). See `docs/architecture.md` "Lifecycle tracking".
- **Parallel agent groups**: `AgentGroup` (`flywheel/agent_group.py`) launches multiple agents with distinct workspace dirs, waits sequentially, and collects artifacts with optional fallbacks. `AgentBlockConfig` groups launch parameters. `prepare_agent_workspace()` is a standalone function for workspace setup.
- **Service dependencies**: Templates declare external services via a `services` key. `ServiceDependency` records name, `url_env`, and description. `check_service_dependencies()` warns about unset env vars.

## Batteries (solve-once capabilities)

- Computer use agent (CUA) -- X11 session management, screenshot loop
- Claude Code agent wrapper (`batteries/claude/`) -- Dockerfile, agent runner with pause/resume support, MCP server for nested block invocation. Projects can mount additional MCP servers at `/workspace/.mcp_servers/`. See `docs/architecture.md` "Agent pause and resume" for the mechanism.

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
