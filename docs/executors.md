# The Executor Seam

Flywheel's runners — the ad-hoc `flywheel run *` commands and the
pattern runner — should not know how a block's body is performed.
They should know only that *some* implementation will perform it,
appended a `BlockExecution` record to the workspace, and produced
output artifacts.

The `BlockExecutor` protocol in `flywheel/executor.py` is that
seam. Everything above the seam (runners, CLI, pattern parsing)
deals exclusively in this protocol. Everything below the seam
(`ProcessExitExecutor`, `RequestResponseExecutor`,
`ProcessExecutor`, the agent battery) is an implementation
detail.

## The protocol

```python
class BlockExecutor(Protocol):
    def launch(
        self,
        block_name: str,
        workspace: Workspace,
        input_bindings: dict[str, str],
        *,
        execution_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        allowed_blocks: list[str] | None = None,
        state_lineage_id: str | None = None,
        run_id: str | None = None,
    ) -> ExecutionHandle: ...
```

Per-launch context falls into three buckets:

- **Bindings** — what this single execution is processing.
  `input_bindings` maps slot names to artifact instance IDs.
  `overrides` is a free-form per-launch overrides dict the
  executor consumes (CLI flag substitutions, battery-specific
  knobs). `allowed_blocks` fences what the body can touch when
  it itself drives nested executions.
- **Workspace identity** — how this execution slots into the
  durable ledger. `execution_id` lets callers pre-assign an ID;
  `state_lineage_id` controls state-chain population from a
  prior execution's captured state directory; `run_id` stamps
  the resulting record so cadence counters scope to one run.
- **Executor-specific extras** *(not on the protocol)* — see
  "Conventions" below.

The protocol returns an `ExecutionHandle` with `is_alive` /
`stop` / `wait`. `wait()` produces an `ExecutionResult` and the
durable `BlockExecution` record. Callers must call `wait()`
exactly once.

## Implementations

Three concrete executors ship with flywheel today:

- **`ProcessExitExecutor`** — one container run per execution.
  Stages inputs, starts the container, waits for exit, captures
  `/state/`, collects outputs as artifact instances, records the
  execution. The default for one-shot container blocks.
- **`RequestResponseExecutor`** — long-lived container reused
  across executions for `lifecycle: workspace_persistent`
  blocks. Starts the runtime on first call for a given
  `(block_name, workspace)` pair; subsequent launches POST a
  request to the same container.
- **`ProcessExecutor`** — local subprocess for trusted host-local
  processes (game servers, local tools).

`runner: lifecycle` blocks have no executor — they are recorded
directly by `LocalBlockRecorder`.

## The seam discipline

A grep for `agent` across `flywheel/pattern_runner.py`,
`flywheel/execution.py`, `flywheel/cli.py`, `flywheel/template.py`,
and `flywheel/pattern.py` should turn up nothing of substance.
Runners depend only on the executor protocol; they don't import
agent modules.

The agent battery is one battery among (eventually) many. A
batteries-included block is a block whose body is performed by
some pre-canned implementation that has its own non-trivial
setup story — for the agent battery, that's prompt mounting,
auth volume, MCP server wiring, allowed-tools whitelist,
isolated network, and the host-side handoff pause/resume loop.

Each battery wraps its machinery as a `BlockExecutor`
implementation. From the runner's vantage point, an
`AgentExecutor` and a `ProcessExitExecutor` are
indistinguishable — both implement `launch()` and return an
`ExecutionHandle`.

## How a project wires executors to blocks

A project's `executor_factory` (a callable supplied by
`ProjectHooks` when present) receives a `BlockDefinition` and
returns the executor that should dispatch it. Today's flywheel
default is straightforward: one-shot container blocks get a
`ProcessExitExecutor`, workspace-persistent container blocks get
a `RequestResponseExecutor`. Projects that use the agent battery
override the factory to return an `AgentExecutor` for blocks
declared as agent-battery blocks.

The `BlockExecutor` returned from the factory is the only thing
the runner uses. The factory holds the room for "this block
needs special treatment" — the runner remains generic.

## Conventions

### Container-executor extras

`ProcessExitExecutor.launch` and `RequestResponseExecutor.launch`
additionally accept three kwargs the protocol does not declare:

- `extra_env` — environment variables merged into the
  container's env on startup.
- `extra_mounts` — bind mounts appended to the substrate's
  mount list.
- `extra_docker_args` — extra `docker run` flags.

These are container-shaped concerns; a non-container executor
has no use for them. Callers that pass them implicitly assume
the executor is container-shaped. The pattern runner's
`on_tool` dispatch path does this knowingly: the executor
factory routes `on_tool` instances to container blocks by
construction.

A future battery for, say, an external-API call would not need
or accept these. A future protocol-level "runtime extras" dict
could replace them once enough non-container batteries exist to
justify the abstraction.

### Per-instance overrides via `overrides`

Free-form. The executor consumes the keys it understands;
unknown keys are ignored silently. Validation, if any, happens
at the executor level. This keeps the parser layer simple and
lets new battery-specific knobs land without protocol churn.
The cost is that a typo in a key name silently does nothing —
an acceptable tradeoff for now; revisit with light schema
validation if it bites.

### `outputs_data` and `elapsed_s` (retired)

Earlier protocol revisions carried `outputs_data` and
`elapsed_s` parameters on `launch()`. Neither was used by any
production caller. Both are gone. `ExecutionEvent.outputs_data`
remains as a dataclass field for possible future use; that's
unrelated to the launch surface.
