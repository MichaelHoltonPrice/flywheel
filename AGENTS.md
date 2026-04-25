# Flywheel — agent orientation

Flywheel is a CLI for AI agent orchestrators (Cursor, Claude Code,
custom agent loops). The orchestrator drives flywheel by invoking
small, composable subcommands; flywheel manages durable workspace
state, runs blocks (Docker containers or in-process logic), and
records every input, output, and execution decision in a workspace
ledger the orchestrator can read between calls.

This file is the entry point for any AI agent working with flywheel
or a flywheel-using project. Humans are welcome too. For the
design rationale see [docs/vision.md](docs/vision.md); for
implementation decisions see [docs/architecture.md](docs/architecture.md).

## CLI surface

Four commands form the supported user-facing surface:

| Command | Purpose | Doc |
| --- | --- | --- |
| `flywheel create workspace` | Materialize a fresh workspace from a template. | [docs/cli/create-workspace.md](docs/cli/create-workspace.md) |
| `flywheel import artifact` | Register an external directory as a workspace artifact instance (wrap single files first). | [docs/cli/import-artifact.md](docs/cli/import-artifact.md) |
| `flywheel run block` | Execute one block ad hoc against a workspace. | (per-command doc not yet written) |
| `flywheel run pattern` | Execute a declarative multi-instance pattern against a workspace. | (per-command doc not yet written) |

Per-command docs land in [docs/cli/](docs/cli/) as the underlying
behavior is pinned. Until then, agents that need detail beyond
`--help` should read the `cli.py` argparse setup directly.

There are additional subcommands (`run agent`, `container`) that
predate the four-command surface and remain supported but are not
the recommended orchestration path going forward.

## Project layout flywheel expects

A flywheel-using project root contains:

* `flywheel.yaml` — declares `foundry_dir` (the directory flywheel
  manages) and optionally `project_hooks` (`module.path:ClassName`
  for the project's `ProjectHooks` implementation, consumed by
  `flywheel run pattern`).
* `<foundry_dir>/templates/workspaces/<name>.yaml` — workspace templates.
* `<foundry_dir>/templates/blocks/<name>.yaml` — block templates.
* `<foundry_dir>/templates/patterns/<name>.yaml` — pattern templates.
* `<foundry_dir>/workspaces/<name>/` — flywheel-managed workspace
  directories. Created by `flywheel create workspace`.

## Key concepts

* **Block execution** is the atomic unit of work. One block, one
  execution, one set of inputs producing outputs. Most blocks are
  Docker containers; some are in-process (`runner: lifecycle`).
  See [docs/architecture.md](docs/architecture.md).
* **All project logic runs in containers (or as host-side
  lifecycle blocks).** Flywheel itself is pure orchestration.
* **Artifacts** have two levels: *declarations* (template-level
  named types like `checkpoint` or `score`) and *instances*
  (concrete, immutable records produced by a specific block
  execution or imported by `flywheel import artifact`). Three
  storage kinds:
  * `copy` — files in the workspace.
  * `git` — references to version-controlled code, captured by
    commit at workspace-create time.
  * `incremental` — one growing instance per name per workspace,
    stored as an append-only `entries.jsonl`.

  Block bodies never call an emit-artifact API; they write files
  into a per-execution per-execution output directory and flywheel
  registers from disk after the block exits. Canonical artifact
  directories are never bind-mounted directly; flywheel always
  copies into a per-mount staging tempdir first.
* **Workspaces** are created from templates and accumulate
  artifact instances and execution records over their lifetime.
  A workspace is a *history*, not a snapshot.
* **Runs** ([`flywheel/artifact.py`](flywheel/artifact.py)
  `RunRecord`,
  [`flywheel/workspace.py`](flywheel/workspace.py) `begin_run` /
  `end_run`) are durable groupings of `BlockExecution` records
  inside one workspace. Every `PatternRunner.run()` invocation
  opens one run and stamps every execution it drives with the
  run's id. `every_n_executions` and similar trigger counters
  scope to the current run, so re-running a pattern in the same
  workspace does not pick up prior runs' executions.
  `BlockExecution.run_id` is `None` for ad hoc (non-pattern)
  executions.
* **The foundry** is the flywheel-managed directory within a
  project, holding templates and workspaces. It is a peer to the
  project source code.

## Executors and the block runtime contract

[`flywheel/executor.py`](flywheel/executor.py) defines the
`BlockExecutor` protocol that decouples block execution from any
specific runtime.

The ad hoc block-execution surface is `flywheel.execution.run_block`.
It owns prepare, invoke, artifact commit, quarantine, and ledger
recording for supported container lifecycles. `flywheel.executor`
contains protocol/result types only; concrete container execution is not
implemented there.

Logical block executions with no container body use
`runner: lifecycle` and are recorded directly by
[`flywheel/local_block.py`](flywheel/local_block.py)
`LocalBlockRecorder` rather than going through a `BlockExecutor`.
A `LocalBlockRecorder` stages each declared input into a per-mount
tempdir (snapshot for incremental inputs), allocates an execution
id and a per-execution output tempdir exposing
`ctx.output_dir(name)`, lets the block body run, registers each
declared output from disk, and writes a single `succeeded` /
`failed` execution record. There is no `running` execution
record. Post-execution checks fire after the execution record is
durable; halt directives are queued for the handoff loop to drain.

## Patterns and project hooks

Patterns ([`flywheel/pattern.py`](flywheel/pattern.py),
[`flywheel/pattern_runner.py`](flywheel/pattern_runner.py)) are
declarative multi-instance topologies. A pattern YAML lists
*instances* (each with a block, trigger, cardinality, inputs,
outputs, and free-form `overrides`) and the runner translates that
into block dispatches. Trigger kinds:

* `continuous` — fire one cohort at run start.
* `autorestart` — continuously refire while the run is alive (a
  scope-`run` halt directive ends the loop).
* `every_n_executions` — fire a cohort every N successful
  executions of a referenced block.
* `pause: [...]` — drain other instances, run a cohort, relaunch
  the paused instances.

The runner resolves per-instance `inputs` to the latest
registered instance of each named artifact (incremental inputs are
re-staged on every launch so each cycle sees newly appended
entries).

Project hooks ([`flywheel/project_hooks.py`](flywheel/project_hooks.py))
are the project-side glue `flywheel run pattern` consumes.
`ProjectHooks.init(workspace, template, project_root, args)` is
called once per run and returns a dict the runner uses to wire
the run. The current canonical key is `executor_factory`, a
`Callable[[BlockDefinition], BlockExecutor]` returning the
appropriate `BlockExecutor` for each dispatched block. Optional
keys (`per_instance_runtime_config`, project-defined defaults)
extend the surface as projects need. Hooks are discovered via
the `project_hooks` field in `flywheel.yaml` or the
`--project-hooks` CLI flag; `load_project_hooks_class()` resolves
the import path. Optional `teardown()` releases project-level
resources after the run.

(The `ProjectHooks.init` docstring in the codebase still
describes a pre-runner-battery-separation shape that returned
`AgentBlockConfig` field overrides; the runtime accepts the
new `executor_factory` key, which is what current projects
return.)

## Block templates and registry

Templates declare blocks by string reference (e.g.,
`blocks: [Eval, TrainSegment]`). Block bodies live in
`<foundry_dir>/templates/blocks/<name>.yaml` and are loaded into a
`BlockRegistry`. Inline-dict block definitions in templates are
rejected.

## Agent blocks (battery)

Agent blocks are a specific block-execution variant where an AI
agent triggers nested block executions via the host-side full-stop
handoff loop ([`flywheel/agent_handoff.py`](flywheel/agent_handoff.py)).
See [`flywheel/agent.py`](flywheel/agent.py).

* Non-blocking agent handle: `launch_agent_block()` returns an
  `AgentHandle` with `stop(reason)`, `wait()`, `is_alive()`.
  `run_agent_block()` is the blocking wrapper.
* Session artifacts: the agent runner exports the SDK session
  JSONL to the workspace on exit and can resume from a session
  artifact via `RESUME_SESSION_FILE`. Sessions are regular `copy`
  artifacts with standard provenance.
* Lifecycle tracking: `BlockExecution.stop_reason` and
  `BlockExecution.predecessor_id` track why agents were stopped
  and link resume chains. `LifecycleEvent` captures operational
  events (agent stops, group completions). See
  [docs/architecture.md](docs/architecture.md) "Lifecycle
  tracking".
* Exit reason classification: `AgentResult.exit_reason` is one of
  `completed`, `auth_failure`, `rate_limit`, `max_turns`,
  `stopped`, `crashed`. Reads `.agent_state.json` written by the
  agent runner.
* Parallel groups: `AgentGroup`
  ([`flywheel/agent_group.py`](flywheel/agent_group.py)) launches
  multiple agents with distinct workspace dirs, waits sequentially,
  and collects artifacts with optional fallbacks.
  `prepare_agent_workspace()` is a standalone function for
  workspace setup. Generic `BlockGroup`
  ([`flywheel/block_group.py`](flywheel/block_group.py)) does the
  same for any `BlockExecutor` type.

## Batteries (solve-once capabilities)

Batteries are reusable capabilities flywheel ships so projects
don't re-implement them.

* **Computer use agent (CUA)** — X11 session management,
  screenshot loop, click targeting.
* **Claude Code agent wrapper** (`batteries/claude/`) — Dockerfile
  and agent runner with pause / resume support. Nested block
  invocations are handled by a `PreToolUse` hook that intercepts
  handoff tools, exits the container, and lets the host-side
  handoff loop run the blocks before relaunching with the spliced
  session. Projects mount MCP servers at `/flywheel/mcp_servers/`.
  See [docs/architecture.md](docs/architecture.md) "Agent pause
  and resume".

## Build and test

```bash
pip install -e ".[dev]"
bash scripts/verify.sh
```

`scripts/verify.sh` runs ruff and pytest. Run it before
considering any work complete.

Before committing, also run a review pass to check for:

* Test quality (coverage gaps, edge cases, phantom tests).
* Docstring completeness (`Args` / `Returns` / `Raises` sections
  for public functions with parameters — ruff does not enforce
  this).

## Where to dig deeper

* [docs/vision.md](docs/vision.md) — what flywheel is for and
  where it is headed.
* [docs/architecture.md](docs/architecture.md) — implementation
  decisions.
* [docs/executors.md](docs/executors.md) — executor reference.
* [docs/cli/](docs/cli/) — per-command reference. Today
  covers `create workspace` and `import artifact`; the
  remaining commands land as each one's foundation work is
  pinned.
* [`tests/integration/README.md`](tests/integration/README.md) —
  integration test conventions.
* [`flywheel/cli.py`](flywheel/cli.py) — the canonical source for
  CLI argument shapes.
