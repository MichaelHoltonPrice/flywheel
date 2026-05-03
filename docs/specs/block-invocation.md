# Block Invocation

Block invocation is the substrate mechanism for a committed block execution to
cause another block execution.

The route is declared on the invoking block and keyed by project-defined
termination reason. The invoking block ends first, Flywheel commits the output
slots for that termination reason, and then Flywheel runs each declared child
block through the ordinary `run_block` path. There is no live request channel
between containers.

## Declaration

A block declares invocations under `on_termination`, parallel to the existing
termination-reason output groups:

```yaml
outputs:
  eval_requested:
    - name: bot
      container_path: /output/bot
  done:
    - name: bot
      container_path: /output/bot

on_termination:
  eval_requested:
    invoke:
      - block: EvalBot
        bind:
          bot: bot
        args: ["--episodes", "${params.eval_episodes}"]
        env:
          FLYWHEEL_RESUME_PROMPT: "Continue from the evaluation result."
```

`on_termination.<reason>` must name a termination reason declared under
`outputs`. Each `invoke` entry names a child block, optional child arguments,
optional child environment values, and optional `bind` entries. A binding maps
a child input slot to either:

- a parent output slot committed for that termination reason, using the short
  form `child_input: parent_output` or the long form
  `child_input: { parent_output: "parent_output" }`; or
- a concrete artifact instance id, using
  `child_input: { artifact_id: "name@id" }`.

If a child input is omitted from `bind`, the child uses the same input
resolution rules as `flywheel run block`.

Invocation route args may contain `${params.name}` placeholders. Pattern runs
validate those references against the pattern's declared params and substitute
the resolved run values. Ad hoc `flywheel run block` can provide route-arg
values with repeated `--param KEY=VALUE` flags; ad hoc params do not affect
block environment or direct block argv.

Invocation route `env` values may also contain `${params.name}` placeholders.
The values are layered over the child block's static environment for only that
invoked execution. This is the intended way for a validation route to relaunch
the same managed agent with a retry-specific `FLYWHEEL_RESUME_PROMPT`.

Invoked child blocks may declare `state: managed`. In pattern runs, Flywheel
uses the same run/lane/block lineage key shape as top-level pattern members,
so repeated invocations of the same child block in the same lane restore the
child's previous managed state. In ad hoc block runs, Flywheel derives a
lineage key from the invoking execution, termination reason, and child block.

## Execution

After the invoking execution commits successfully, Flywheel looks up routes for
its committed `termination_reason`. Each child runs through canonical
preparation, runtime, commit, validation, state capture, and ledger recording.

Invoked children recursively dispatch their own `on_termination` routes by
default. Dispatch remains synchronous, host-side, and post-commit: a child
must finish and commit before its own routes are fired. Recursive dispatch is
bounded by a runtime depth limit; the default permits sixteen committed
executions in one invocation chain and rejects the seventeenth launch. Template load
also rejects declared invocation cycles, and the runtime keeps a same-block
cycle guard as a final failsafe.

Routes that intentionally re-enter a block already present in the current
invocation chain must declare `max_invocations_per_chain`. The value is the
maximum committed executions of the child block allowed in that chain. Such
routes are treated as explicitly bounded cycle edges during template
validation and at runtime.

All routes declared for a termination reason are attempted. A child failure is
recorded on that route's `BlockInvocation` record and does not prevent sibling
routes from running. A descendant failure is local to the descendant
`BlockExecution` and the `BlockInvocation` that launched it; it does not
retroactively rewrite parent execution status or ancestor invocation records.

## Ledger

Each fired route creates a `BlockInvocation` row in the workspace invocation
ledger. The record stores the invoking execution id, termination reason,
invoked block name, invoked execution id when one exists, concrete input
bindings, arguments, status, and error text.

The child `BlockExecution` stores `invoking_execution_id`. The invoking
execution's own status is derived only from its termination reason and output
commit result; child failure does not rewrite the invoking execution.
Recursive invocation trees are reconstructed by following
`invoking_execution_id` links and `BlockInvocation.invoked_execution_id`;
pattern member records store only direct child invocation ids.

For `flywheel run block`, the process exit status follows the invoking block's
own execution result. Invocation failures are durable ledger records, not a
retroactive failure of the invoking execution.

## Boundaries

Invocation is not pattern execution. Patterns decide what to run next from the
workspace state and own loops, cohorts, limits, and branching. Invocation is a
per-execution route from one committed outcome to another block execution.

Invocation is also not a battery-specific tool bridge. A battery may help a
block choose which termination reason to announce, but the substrate only sees
the final termination reason, committed artifacts, and declared routes.
