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

Patterns may also declare lanes and fixtures:

```yaml
name: improve_bot
lanes: [A, B, C]
fixtures:
  bot: foundry/templates/assets/bots/baseline
steps:
  - name: improve_1
    cohort:
      min_successes: all
      foreach: lanes
      block: ImproveBot
```

`steps` is an ordered, non-empty list. Step names are unique within a
pattern. Member names are unique within a step. Pattern members are
block executions. If omitted, `lanes` defaults to a single implicit
`default` lane.

## Lanes

A lane is a run-scoped artifact resolution context. Lanes allow a
single pattern run to carry multiple independent artifact pedigrees.
Members in different lanes do not see one another's outputs through
default resolution, even when the scheduler executes them sequentially.

Lane metadata is stored on `RunRecord` member and fixture records, not
on `BlockExecution` or `ArtifactInstance`. Artifact instances remain
workspace-global immutable values; the run record explains which lane
used or produced them.

The shorthand cohort form

```yaml
cohort:
  foreach: lanes
  block: ImproveBot
```

expands to one member per declared lane. The generated member name is
the lane name and the member belongs to that lane.

## Fixtures

Pattern fixtures seed each lane at run start. Each fixture maps an
artifact name to a project-relative source directory. For every
declared lane, Flywheel materializes one ordinary `copy` artifact
instance through the canonical artifact registration path, runs the
project artifact validator for that artifact name, records fixture
provenance in the artifact's `fixture_id` and `source`, and records
the lane fixture on the `RunRecord`. The fixture artifact's
`produced_by` remains empty because fixtures are not block executions;
its structured producer pointer is `fixture_id -> RunRecord.fixtures`.

Fixtures are not lazy input fallbacks. They fire before the first step
runs. After materialization, a fixture artifact is indistinguishable
from any other artifact instance to the block that consumes it.
If fixture materialization fails, the run is marked failed before any
step launches and the exception propagates to the caller.

Fixtures target `copy` artifacts only. Git artifacts are already
commit-pinned source references and are not materialized by this
fixture mechanism.

## Cohorts

A cohort is a semantic group of peer members. Members in the same
cohort must not depend on one another's outputs. The scheduler
executes members sequentially, but that is a scheduling choice, not
part of cohort identity.

Lane independence is an artifact-resolution rule, not a cohort-success
rule. A lane failure can still fail the enclosing step or run according
to the cohort's `min_successes` policy.

Supported success rules:

* `min_successes: all`: the cohort succeeds only if every member
  succeeds. The sequential scheduler may stop launching remaining
  members after the first failure.
* `min_successes: 1`: the cohort succeeds if at least one member
  succeeds. All declared members are launched.

Unsupported success-rule keys or values are parse errors.

## Input Resolution

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

For pattern members, omitted `copy` input bindings resolve to the
latest artifact for that input name in the member's lane. The lane
history is derived from:

* pattern fixtures materialized for the lane;
* prior successful members in the same lane;
* invoked child executions recorded by prior successful members in the
  same lane.

Pattern execution intentionally does not fall back to workspace-global
latest-by-name for `copy` inputs. Global latest remains an ad hoc
`flywheel run block` convenience, not a pattern default. Git inputs
continue to use the canonical git resolution path.

Lane visibility includes direct invocation children recorded by prior
successful members. Invocation children do not dispatch further routes
in v1, so there are no transitive invocation grandchildren for the
lane resolver to consider.

## Run Records

`RunRecord` stores run-level metadata:

* run id;
* kind, conventionally `pattern:<name>`;
* status and timestamps;
* declared lanes;
* fixture materializations by lane;
* ordered step results;
* member execution ids, lanes, invocation ids, and output bindings;
* optional run-level error.

`BlockExecution` does not carry run-specific fields. Reverse lookup
from an execution id to its run is done through run metadata.
