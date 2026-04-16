# Flywheel

Python orchestration framework for measurable AI improvement loops. Wires Docker containers together, tracks artifacts, and provides visibility into hybrid human+agent workflows.

## Key concepts

- **Block execution** is the atomic operation. One block, one Docker container, one set of inputs producing outputs. See `docs/vision.md` (aspirational) and `docs/architecture.md` (implemented or agreed-upon decisions; future work is noted separately at the end).
- **All project logic runs in Docker containers.** Flywheel on the host is pure orchestration.
- **Artifacts** have two levels: declarations (template-level types like "checkpoint" or "score") and instances (concrete, immutable records produced by block executions). Two storage kinds exist: copy artifacts (files in the workspace) and git artifacts (references to version-controlled code).
- **Workspaces** are created from templates and accumulate artifact instances and execution records over their lifetime. The workspace is a history, not a snapshot.
- **The foundry** is the flywheel-managed directory within a project, holding templates and workspaces. It is a peer to the project source code.
- **Block executors** (`flywheel/executor.py`): The `BlockExecutor` protocol decouples block execution from Docker. Three implementations: `ContainerExecutor` (Docker container), `RecordExecutor` (artifact creation without containers, replaces the `__record__` sentinel), `ProcessExecutor` (local subprocess for trusted host-side processes like game servers). All return an `ExecutionHandle` with `stop()`, `wait()`, `is_alive()`.
- **Execution channel** (`flywheel/execution_channel.py`): HTTP service replacing `BlockBridgeService`. Routes requests to executors: `mode=record` → RecordExecutor, default → ContainerExecutor. Same HTTP protocol — MCP servers inside containers need zero changes. Fires typed `ExecutionEvent` callbacks (replaces the former `on_record`). `on_record` still accepted as a backward-compat alias.
- **Agent blocks** are a special block execution variant where an AI agent triggers nested block executions via the execution channel. See `flywheel/agent.py`.
- **Non-blocking agent handle**: `launch_agent_block()` returns an `AgentHandle` with `stop(reason)`, `wait()`, and `is_alive()`. `run_agent_block()` is a blocking wrapper.
- **Session artifacts**: The agent runner exports the SDK session JSONL to the workspace on exit and can resume from a session artifact via `RESUME_SESSION_FILE`. Sessions are regular copy artifacts with standard provenance.
- **Lifecycle tracking**: `BlockExecution` has `stop_reason` and `predecessor_id` for tracking why agents were stopped and linking resume chains. `AgentHandle.wait()` records the agent itself as a `BlockExecution` (block_name=`__agent__`). `LifecycleEvent` captures operational events (agent stops, group completions). See `docs/architecture.md` "Lifecycle tracking".
- **Parallel agent groups**: `AgentGroup` (`flywheel/agent_group.py`) launches multiple agents with distinct workspace dirs, waits sequentially, and collects artifacts with optional fallbacks. `AgentBlockConfig` groups launch parameters. `prepare_agent_workspace()` is a standalone function for workspace setup.
- **Agent loop** (`flywheel/agent_loop.py`): Flywheel-owned orchestration loop. Projects provide hooks implementing `AgentLoopHooks`: `init(workspace, template, project_root, args)` for one-time setup, `decide(state) -> Action` for round decisions, and `build_prompt(action, state) -> str` for prompt construction. The loop manages round counting, session resume, circuit breaker (consecutive auth/rate-limit failures), and lifecycle events. Actions: `Continue`, `SpawnGroup`, `Stop`, `Finished`. See `docs/architecture.md` "Agent loop".
- **Hooks discovery**: Projects declare their hooks class in `flywheel.yaml` via `hooks: module.path:ClassName`. `flywheel run loop` loads the class, calls `init()` with project-specific args (after `--`), and runs the loop. `load_hooks_class()` resolves the import path.
- **Exit reason classification**: `AgentResult.exit_reason` classifies why the agent exited: `"completed"`, `"auth_failure"`, `"rate_limit"`, `"max_turns"`, `"stopped"`, `"crashed"`. Reads `.agent_state.json` written by the agent_runner.
- **Block groups** (`flywheel/block_group.py`): Generic parallel execution for any `BlockExecutor` type. `BlockGroup` launches N blocks via an executor, waits sequentially, records lifecycle events. `AgentGroup` remains for agent-specific parallel launches.
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

Flywheel expects a `flywheel.yaml` in the target project root defining `foundry_dir` (the foundry directory) and optionally `hooks` (a `module:Class` import path for the project's `AgentLoopHooks` implementation). Templates and workspaces live under `foundry_dir`.
