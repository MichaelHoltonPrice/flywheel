# Pattern Execution

Patterns are declarative workflows for running related block
executions as one durable unit. They capture the order, grouping,
input handoff, and success rule for a multi-step workflow without
letting the workflow runner forge artifacts or write block execution
records directly.

Each pattern run opens a `RunRecord`, materializes run fixtures,
executes the declared pattern body through
`flywheel.execution.run_block`, records executed members on the run,
and closes the run.

Block executions produced by a pattern use the same prepare, invoke,
commit, validation, quarantine, and ledger path as ad hoc
`flywheel run block`. Pattern grouping is stored on `RunRecord`, not
on `BlockExecution`.

## Pattern Shape

Patterns live under `<foundry_dir>/templates/patterns/<name>.yaml`:

```yaml
name: train_eval
params:
  model:
    type: string
    default: claude-sonnet-4-6
  eval_episodes:
    type: int
    default: 4000
do:
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

Patterns must declare a non-empty `do:` structured executable body.
For example, this pattern runs two independent lanes. Each lane runs a
local sub-pattern that repeatedly runs `ImproveBot` until the agent
exits normally or has requested five evaluations:

```yaml
name: improve_bot
params:
  model:
    type: string
    default: claude-sonnet-4-6[1m]
  max_evals:
    type: int
    default: 5
fixtures:
  bot: foundry/templates/assets/bots/baseline
do:
  - foreach:
      count: 2
    do:
      - use: improve_bot_lane
        with:
          model: ${params.model}
          max_evals: ${params.max_evals}
patterns:
  improve_bot_lane:
    params:
      model:
        type: string
      max_evals:
        type: int
    do:
      - run_until:
          block: ImproveBot
          env:
            MODEL: ${params.model}
          continue_on:
            eval_requested:
              max: ${params.max_evals}
          stop_on:
            - normal
          fail_on:
            - aborted
          after_every:
            - reason: eval_requested
              count: 20
              do:
                - cohort:
                    min_successes: 1
                    members:
                      - name: brainstorm_1
                        block: Brainstorm
                      - name: brainstorm_2
                        block: Brainstorm
```

`do` is an ordered, non-empty list of pattern body nodes. Pattern
members are block executions. If omitted, `lanes` defaults to a single
implicit `default` lane unless the root body declares a `foreach.count`,
in which case Flywheel generates stable lanes named `lane_0`, `lane_1`,
and so on. The old top-level `steps:` grammar is rejected.

## Parameters

Patterns may declare typed run parameters under `params`. Supported
types are `string`, `int`, `float`, and `bool`. A parameter may provide
a default; parameters without defaults must be supplied by the caller at
run start. The CLI supplies overrides with repeated
`--param KEY=VALUE` flags.

Resolved parameters are recorded on the `RunRecord` before the first
fixture or block execution. They are durable orchestration inputs: a
run can be inspected later to see which model, episode count, budget,
or other project-level knob was used.

Pattern member `args` and `env` values may reference parameters with
`${params.name}` placeholders:

```yaml
params:
  model:
    type: string
    default: claude-sonnet-4-6
  eval_episodes:
    type: int
    default: 4000

do:
  - name: improve
    cohort:
      foreach: lanes
      block: ImproveBot
      env:
        MODEL: ${params.model}
```

Invocation route args may also use the same placeholder syntax. Unknown
parameter names are errors; Flywheel does not silently pass unresolved
placeholders into containers.

Local sub-patterns under `patterns:` declare their own `params:`.
`use.with` supplies values for those params. There is no implicit
capture across a `use` boundary; `${params.name}` inside the local
sub-pattern refers to that sub-pattern's resolved params.

### Placeholder Grammar

The only supported placeholder form is `${params.name}`. Parameter
names use the same name rules as other Flywheel identifiers: they must
start with a letter or digit and may contain letters, digits, hyphens,
and underscores. There is no escaping syntax; strings that need a
literal `${params.name}` should avoid that exact form.

Values render as strings for container environment variables and argv.
Boolean values render as lowercase `true` or `false`; integers and
floats render with Python's normal string representation.

Flywheel validates parameter references before opening the run record.
A typo in a member env value, member arg, or routed block invocation arg
rejects the pattern run before any fixture is materialized or container
starts.

Input resolution runs before parameter substitution. This keeps
substitution failures from creating partial input artifacts or changing
which lane artifact a member consumes.

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

The structured body form can generate anonymous lanes:

```yaml
do:
  - foreach:
      count: 3
    do:
      - use: lane_pattern
```

This creates `lane_0`, `lane_1`, and `lane_2`. The scheduler executes
the body lane-major: a lane's body runs to completion before the next
lane starts. Lane-major order is a scheduling strategy, not permission
for cross-lane visibility. Each lane still sees only its own fixture,
member, and direct invocation-child history.

A root `do:` body that declares `foreach.count` lanes must contain only
root `foreach` nodes. Work that should happen per lane belongs inside
the `foreach` body. This keeps the root body from accidentally mixing
default-lane work with generated lane fixtures.

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

## Local Composition And Loops

`patterns:` declares local sub-patterns in the same file. A `use` node
runs one of those sub-patterns in the current lane:

```yaml
do:
  - use: improve_bot_lane
    with:
      model: ${params.model}
```

The sub-pattern inherits the current run and lane. It does not create a
new workspace, run record, or artifact-resolution scope.

`run_until` repeatedly runs one block in the current lane:

```yaml
run_until:
  block: ImproveBot
  continue_on:
    eval_requested:
      max: 5
  stop_on:
    - normal
```

One `run_until` loop is recorded as one `RunStepRecord` with
`kind: run_until`. Each iteration is an ordinary `RunMemberRecord`
inside that step, named `iter_1`, `iter_2`, and so on. The step records
`terminal_reason`, `stop_kind`, and `reason_counts` when the loop
stops, so operators can inspect how many iterations ran and why the
loop ended. If the block exits with a `continue_on` termination reason,
Flywheel first lets the block's `on_termination` routes dispatch. The
next iteration therefore sees both the parent block's committed outputs
and any direct invocation-child outputs from the prior iteration.

The `max` value is the budget for that continuation reason. If the
budget is reached, the loop stops after the iteration and any routed
child invocations have committed; the step records
`stop_kind: budget_exhausted`. If the block exits with a `stop_on`
reason, the loop stops successfully and records `stop_kind: stop_on`.
If the block exits with a `fail_on` reason, the block's declared
outputs are still committed, the loop records `stop_kind: fail_on`,
and the pattern run fails with that termination reason. This is useful
for project-level diagnostic exits that should preserve artifacts but
should not be treated as successful loop completion. A declared block
termination reason that appears in neither `continue_on`, `stop_on`,
nor `fail_on` is a pattern error for this loop and records
`stop_kind: unexpected_reason`. A member execution failure records
`stop_kind: failed`.

`continue_on`, `stop_on`, and `fail_on` reasons must be declared by the
block's `outputs` map, and a reason may appear in only one of those
sets. This prevents a loop from silently discarding proposed outputs
for a termination reason the block did not declare.

`run_until.after_every` declares periodic lane-local work that runs
after a counted continuation reason. Each entry has:

* `reason`: a reason listed under `continue_on`;
* `count`: a positive integer or parameter placeholder resolving to a
  positive integer;
* `do`: a non-empty structured body to run when the count is reached.

For example, a play loop can run four brainstorm agents after every
twentieth `action_requested` termination:

```yaml
run_until:
  name: play
  block: play
  continue_on:
    action_requested:
      max: ${params.max_actions}
  stop_on:
    - normal
  after_every:
    - reason: action_requested
      count: 20
      do:
        - cohort:
            min_successes: 1
            members:
              - name: brainstorm_1
                block: brainstorm
              - name: brainstorm_2
                block: brainstorm
              - name: brainstorm_3
                block: brainstorm
              - name: brainstorm_4
                block: brainstorm
```

The periodic body runs after the parent iteration commits and after
any `on_termination` child invocations for that iteration have
committed. The next loop iteration therefore sees artifacts produced
by both the parent iteration, its invocation children, and the
periodic body, all through the same lane-scoped resolver. The
periodic body executes in the current lane. Inline `cohort` nodes in a
structured body may omit `name`; if a nameless cohort has one member,
Flywheel uses that member name for the generated step name, otherwise
it uses a stable `cohort_N` name.

`run_until.resume_prompt_builder` lets a project own the prompt used
when the loop launches the same block again after the first iteration:

```yaml
run_until:
  name: play
  block: PlayAgent
  resume_prompt_builder:
    command:
      - python
      - workforce/resume_prompt.py
      - play
```

The builder command runs from the project root. Pattern params may be
referenced in command strings with `${params.name}`. Flywheel writes a
JSON context document to the command's stdin and uses stdout, stripped
of surrounding whitespace, as `FLYWHEEL_RESUME_PROMPT` for the next
block execution. Empty stdout falls back to the default generic resume
prompt. A nonzero exit, launch failure, or timeout fails the pattern.

The context is intentionally structural rather than artifact-semantic.
It includes the pattern name, run id, current lane, `run_until` name,
block name, next iteration number, declared current input slots, and
the previous edge transition when one is known. The transition includes
the previous member, its termination reason, the counted reason value,
and any `after_every` steps that just ran. When a pattern run is later
continued with `flywheel run pattern ... --resume`, Flywheel
reconstructs this transition from the existing run record so the same
callback can still distinguish, for example, a plain action edge from
an edge that just ran a periodic brainstorm body.

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
* invoked child executions, including transitive invocation descendants,
  recorded by prior successful members in the same lane.

Pattern execution intentionally does not fall back to workspace-global
latest-by-name for `copy` inputs. Global latest remains an ad hoc
`flywheel run block` convenience, not a pattern default. Git inputs
continue to use the canonical git resolution path.

Lane visibility includes all successful transitive invocation descendants
recorded by prior successful members. Failed invocations and failed child
executions are ignored, so a validator or repair child only becomes
lane-latest when its own execution commits successfully.

Managed-state blocks in patterns use a lineage key derived from the
run id, lane, and block name. The iteration member name is deliberately
excluded so repeated `run_until` iterations of the same managed block
restore and extend one lane-local state chain.

## Run Records

`RunRecord` stores run-level metadata:

* run id;
* kind, conventionally `pattern:<name>`;
* status and timestamps;
* resolved run parameters;
* declared lanes;
* fixture materializations by lane;
* ordered step results;
* step kind and loop terminal metadata for structured steps;
* member execution ids, lanes, invocation ids, and output bindings;
* optional run-level error.

`BlockExecution` does not carry run-specific fields. Reverse lookup
from an execution id to its run is done through run metadata.

## Pattern Resume

`flywheel run pattern ... --resume RUN_ID` continues an existing
logical pattern run in the same workspace. It does not create a child
run. Flywheel reopens the existing `RunRecord`, keeps the same run id,
and re-resolves declared pattern params with any supplied `--param`
overrides.

Keeping the same run id is the continuity contract:

* managed-state lineages derived from run id, lane, and block continue
  to restore the latest compatible snapshot for the logical run;
* lane-scoped artifact sequences continue appending to the same
  sequence scope;
* pattern fixtures are not rematerialized;
* completed boot/cohort work is skipped rather than launched again.

For structured `run_until` nodes, resume uses the existing
`RunStepRecord` as the loop cursor. Existing members and
`reason_counts` are preserved, the next member name continues from the
prior count (`iter_26` after `iter_25`), and continuation budgets are
interpreted as absolute totals under the newly resolved params. A run
that stopped with `budget_exhausted` therefore continues only if the
new resolved budget is greater than the already-recorded count.

Periodic `after_every` bodies use the preserved `reason_counts`, so
they fire at the next due occurrence rather than replaying old
occurrences.

Resume is deliberately same-pattern and same-lane in v1. Flywheel
rejects resuming an unknown run, a running run, a run whose kind does
not match the requested pattern, or a run whose lanes do not match the
current pattern declaration.
