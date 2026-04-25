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
        args: ["--episodes", "200"]
```

`on_termination.<reason>` must name a termination reason declared under
`outputs`. Each `invoke` entry names a child block, optional child arguments,
and optional `bind` entries. A binding maps a child input slot to either:

- a parent output slot committed for that termination reason, using the short
  form `child_input: parent_output` or the long form
  `child_input: { parent_output: "parent_output" }`; or
- a concrete artifact instance id, using
  `child_input: { artifact_id: "name@id" }`.

If a child input is omitted from `bind`, the child uses the same input
resolution rules as `flywheel run block`.

Invoked child blocks may not declare `state: managed` in the first
implementation because routes do not yet declare child state lineage keys.

## Execution

After the invoking execution commits successfully, Flywheel looks up routes for
its committed `termination_reason`. Each child runs through canonical
preparation, runtime, commit, validation, state capture, and ledger recording.

For the first implementation, invoked children do not recursively dispatch
their own `on_termination` routes. Iteration, retry limits, and conditional
loops belong in the pattern resolver.

All routes declared for a termination reason are attempted. A child failure is
recorded on that route's `BlockInvocation` record and does not prevent sibling
routes from running.

## Ledger

Each fired route creates a `BlockInvocation` record in `workspace.yaml`. The
record stores the invoking execution id, termination reason, invoked block name,
invoked execution id when one exists, concrete input bindings, arguments,
status, and error text.

The child `BlockExecution` stores `invoking_execution_id`. The invoking
execution's own status is derived only from its termination reason and output
commit result; child failure does not rewrite the invoking execution.

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
