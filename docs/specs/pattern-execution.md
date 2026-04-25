# Pattern Execution

Patterns are declarative workflows for running related block
executions as one durable unit. They capture the order, grouping,
input handoff, and success rule for a multi-step workflow without
letting the workflow runner forge artifacts or write block execution
records directly.

Each pattern run opens a `RunRecord`, resolves one cohort at a time,
executes each member through `flywheel.execution.run_block`, records
cohort membership on the run, and closes the run.

Block executions produced by a pattern use the same prepare, invoke,
commit, validation, quarantine, and ledger path as ad hoc
`flywheel run block`. Pattern grouping is stored on `RunRecord`, not
on `BlockExecution`.

## Pattern Shape

Patterns live under `<foundry_dir>/templates/patterns/<name>.yaml`:

```yaml
name: train_eval
steps:
  - name: train
    cohort:
      min_successes: all
      members:
        - name: train_dueling
          block: Train
          args: [--subclass, dueling]

  - name: eval
    cohort:
      min_successes: all
      members:
        - name: eval_dueling
          block: Eval
          inputs:
            checkpoint:
              from_step: train
              member: train_dueling
              output: checkpoint
```

`steps` is an ordered, non-empty list. Step names are unique within a
pattern. Member names are unique within a step. Pattern members are
block executions.

## Cohorts

A cohort is a semantic group of peer members. Members in the same
cohort must not depend on one another's outputs. The scheduler
executes members sequentially, but that is a scheduling choice, not
part of cohort identity.

Supported success rules:

* `min_successes: all`: the cohort succeeds only if every member
  succeeds. The sequential scheduler may stop launching remaining
  members after the first failure.
* `min_successes: 1`: the cohort succeeds if at least one member
  succeeds. All declared members are launched.

Unsupported success-rule keys or values are parse errors.

## Binding

Pattern members may bind an input slot from:

* a concrete artifact instance id:

  ```yaml
  inputs:
    checkpoint:
      artifact_id: checkpoint@abc123
  ```

* a prior member output:

  ```yaml
  inputs:
    checkpoint:
      from_step: train
      member: train_dueling
      output: checkpoint
  ```

Omitted bindings are left to canonical block input resolution. Pattern
execution does not stage inputs or implement latest-artifact lookup.

## Run Records

`RunRecord` stores run-level metadata:

* run id;
* kind, conventionally `pattern:<name>`;
* status and timestamps;
* ordered step results;
* member execution ids and output bindings;
* optional run-level error.

`BlockExecution` does not carry run-specific fields. Reverse lookup
from an execution id to its run is done through run metadata.
