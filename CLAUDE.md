# Flywheel

Python orchestration framework for measurable AI improvement loops. Wires Docker containers together, tracks artifacts, and provides visibility into hybrid human+agent workflows.

## Key concepts

- **Block execution** is the atomic operation. One block, one Docker container, one set of inputs producing outputs. See `docs/vision.md` (aspirational) and `docs/architecture.md` (implemented or agreed-upon decisions; future work is noted separately at the end).
- **All project logic runs in Docker containers.** Flywheel on the host is pure orchestration.
- **Artifacts** have two levels: declarations (template-level types like "checkpoint" or "score") and instances (concrete, immutable records produced by block executions). Three storage kinds exist: copy artifacts (files in the workspace), git artifacts (references to version-controlled code), and incremental artifacts (one growing instance per name per workspace, stored as an append-only `entries.jsonl`). Block bodies never call an emit-artifact API — they write files into a per-execution ephemeral output directory and flywheel registers from disk after exit. Canonical artifact directories are never bind-mounted directly; flywheel always copies into a per-mount staging tempdir first. See `docs/architecture.md` "Storage kinds", "Per-execution output directories", and "Per-mount input staging".
- **Workspaces** are created from templates and accumulate artifact instances and execution records over their lifetime. The workspace is a history, not a snapshot.
- **The foundry** is the flywheel-managed directory within a project, holding templates and workspaces. It is a peer to the project source code.
- **Block executors** (`flywheel/executor.py`): The `BlockExecutor` protocol decouples block execution from Docker. Two implementations: `ContainerExecutor` (Docker container) and `ProcessExecutor` (local subprocess for trusted host-side processes like game servers). Both return an `ExecutionHandle` with `stop()`, `wait()`, `is_alive()`. Logical block executions that have no container body use `runner: lifecycle` and are recorded directly by `flywheel.local_block.LocalBlockRecorder` rather than going through an executor.
- **Local block recorder** (`flywheel/local_block.py`): Host-side, in-process write surface for `runner: lifecycle` blocks invoked during a full-stop handoff. Stages each declared input into a per-mount tempdir (snapshot for incremental inputs), allocates an execution id and a per-execution output tempdir exposing `ctx.output_dir(name)`, lets the block body run, and registers each declared output from disk before writing a single `succeeded`/`failed` execution record. There is no `running` execution record and no HTTP. Post-execution checks fire after the execution record is durable; halt directives are queued for the handoff loop to drain.
- **Block templates and registry**: Templates declare blocks by string reference (e.g., `blocks: [predict, game_step]`). Block bodies live in `workforce/blocks/<name>.yaml` and are loaded into a `BlockRegistry`. Inline-dict block definitions in templates are rejected.
- **Agent blocks** are a special block execution variant where an AI agent triggers nested block executions via the host-side full-stop handoff loop (`flywheel/agent_handoff.py`). See `flywheel/agent.py`.
- **Non-blocking agent handle**: `launch_agent_block()` returns an `AgentHandle` with `stop(reason)`, `wait()`, and `is_alive()`. `run_agent_block()` is a blocking wrapper.
- **Session artifacts**: The agent runner exports the SDK session JSONL to the workspace on exit and can resume from a session artifact via `RESUME_SESSION_FILE`. Sessions are regular copy artifacts with standard provenance.
- **Lifecycle tracking**: `BlockExecution` has `stop_reason` and `predecessor_id` for tracking why agents were stopped and linking resume chains. `AgentHandle.wait()` records the agent itself as a `BlockExecution` (block_name=`__agent__`). `LifecycleEvent` captures operational events (agent stops, group completions). See `docs/architecture.md` "Lifecycle tracking".
- **Parallel agent groups**: `AgentGroup` (`flywheel/agent_group.py`) launches multiple agents with distinct workspace dirs, waits sequentially, and collects artifacts with optional fallbacks. `AgentBlockConfig` groups launch parameters. `prepare_agent_workspace()` is a standalone function for workspace setup.
- **Patterns** (`flywheel/pattern.py`, `flywheel/pattern_runner.py`): Declarative multi-agent topology. A pattern YAML lists roles (each with a prompt, model, cardinality, and trigger) and the runner translates that into `launch_agent_block` calls. Triggers: `continuous` (one cohort at run start), `every_n_executions` (cohort every N successful executions of a referenced block), `on_request` / `on_event` (parsed but not yet implemented). The runner resolves per-role `inputs` to the latest registered instance (incremental inputs are re-staged on every launch so each cycle sees newly appended entries). Per-role `materialize` rollups and `derive_from` / `derive_kind` input slots are not supported — declare a first-class `incremental` artifact instead. See `docs/architecture.md` "Patterns".
- **Project hooks** (`flywheel/project_hooks.py`): The shrunken project-side surface consumed by `flywheel run pattern`. `ProjectHooks.init(workspace, template, project_root, args) -> dict` parses project-specific CLI args (after `--`) and returns `AgentBlockConfig` overrides (env vars, mounts, pre-launch hook, etc.). Optional `teardown()` releases resources after the run. Discovered via `project_hooks: module.path:ClassName` in `flywheel.yaml` (or the `--project-hooks` CLI flag); `load_project_hooks_class()` resolves the import path.
- **Exit reason classification**: `AgentResult.exit_reason` classifies why the agent exited: `"completed"`, `"auth_failure"`, `"rate_limit"`, `"max_turns"`, `"stopped"`, `"crashed"`. Reads `.agent_state.json` written by the agent_runner.
- **Block groups** (`flywheel/block_group.py`): Generic parallel execution for any `BlockExecutor` type. `BlockGroup` launches N blocks via an executor, waits sequentially, records lifecycle events. `AgentGroup` remains for agent-specific parallel launches.

## Batteries (solve-once capabilities)

- Computer use agent (CUA) -- X11 session management, screenshot loop
- Claude Code agent wrapper (`batteries/claude/`) -- Dockerfile and agent runner with pause/resume support. Nested block invocations are handled by a `PreToolUse` hook that intercepts handoff tools, exits the container, and lets the host-side handoff loop run the blocks before relaunching with the spliced session. Projects mount MCP servers at `/workspace/.mcp_servers/`. See `docs/architecture.md` "Agent pause and resume" for the mechanism.

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

Flywheel expects a `flywheel.yaml` in the target project root defining `foundry_dir` (the foundry directory) and optionally `project_hooks` (a `module:Class` import path for the project's `ProjectHooks` implementation, consumed by `flywheel run pattern`). Templates and workspaces live under `foundry_dir`; patterns live under `<project>/patterns/`.
