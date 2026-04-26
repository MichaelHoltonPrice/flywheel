# Architecture Decisions

This document describes the implementation architecture Flywheel
currently intends to preserve. Aspirational product direction belongs
in [vision.md](vision.md). Normative details for specific subsystems
live in [docs/specs/](specs/). Work that is intentionally deferred is
listed in "Future Work" below.

## Core Boundary

Flywheel's substrate abstraction is block execution, not agent
execution.

A block execution owns:

1. input resolution and staging;
2. one runtime invocation;
3. output collection;
4. artifact validation and registration;
5. managed state restore/capture;
6. quarantine and state recovery;
7. the `BlockExecution` ledger record.

Batteries such as the bundled Claude agent are reusable project-style
code shipped with Flywheel. They may provide Dockerfiles, entrypoints,
runners, examples, and CLI conveniences, but they must enter durable
state through the same canonical block-execution path as any other
block.

## Artifacts

### Declarations and Instances

An artifact declaration is template-level: a named artifact type and
storage kind. An artifact instance is workspace-level: an immutable
record with a unique id, timestamp, storage location, and provenance.

The workspace stores instances. Templates declare what kinds of
instances may exist.

### Storage Kinds

Flywheel currently has three storage kinds:

* `copy` - an immutable directory stored under the workspace.
* `git` - a reference to a repository, commit SHA, and path.
* `incremental` - an append-oriented legacy kind with one growing
  instance per declaration.

`copy` and `git` are the current canonical kinds for new
block-execution work. `incremental` remains implemented for existing
surfaces, but the preferred future direction is tagged copy instances.

### Directory-Shaped Artifacts

Copy artifact instances are directories. Containers write output bytes
under `/output/<slot>/`, and `flywheel import artifact` also requires a
directory source. A single file must be wrapped in a directory by the
caller.

This keeps imports, block outputs, validators, and downstream mounts on
one shape.

### Validation

Projects can register artifact validators in `flywheel.yaml`.
Flywheel calls them from every artifact-finalization path. The
validator receives a staged directory containing the exact candidate
bytes that would become the artifact instance.

Flywheel does not inspect artifact contents by default. Content policy
belongs to the project validator.

When a block emits multiple output slots and one slot fails validation,
Flywheel commits slots that passed and rejects only failed slots. The
execution is still recorded as failed with `failure_phase` describing
the most downstream failure and per-slot rejection records explaining
what happened.

### Quarantine and Amendment

Rejected block outputs are copied to
`<workspace>/quarantine/<execution-id>/<slot>/` when preservation
succeeds. The failed `BlockExecution` stores a `RejectedOutput` for
each rejected slot.

Corrected bytes are registered as new artifact instances through
`flywheel fix execution` or `flywheel amend artifact`. They may carry a
`SupersedesRef` pointing backward to an accepted artifact id or a
rejected `(execution_id, slot)` pair. The predecessor is provenance,
not a resolution rule.

## State

Artifacts and state are both files, but they mean different things.

Artifacts model edge relationships between block executions: one
execution produces a named, immutable value that another execution may
bind as input.

State models node-local continuity for an execution lineage: bytes are
restored because the next execution continues the same lineage, not
because another block selected them as an input artifact.

Managed state uses a write path separate from artifacts. It has:

* state modes on block declarations and `BlockExecution`;
* lineage keys supplied by the caller or pattern runner;
* immutable `StateSnapshot` records;
* compatibility checks before restore;
* optional project-owned state validators;
* recovery preservation for rejected or failed captures.

The normative contract is in [specs/state.md](specs/state.md).

## Block Execution

The canonical implementation lives in
[../flywheel/execution.py](../flywheel/execution.py).

### Prepare

Prepare mints an execution id, resolves input bindings, stages inputs,
allocates proposal directories, creates one host tree mounted at
`/flywheel`, restores managed state when declared, and prepares output
proposal directories for the block's declared outputs.

Canonical artifact directories are never mounted directly into a block.
Inputs are copied into per-mount staging directories first.

### Run

The current supported runtime is a one-shot container. A runtime runner
receives an `ExecutionPlan`, runs the already-prepared container body,
and returns observation. It does not register artifacts, write state
snapshots, quarantine bytes, or record executions.

The default runner calls Docker. Tests can inject fake runners to prove
that prepare and commit remain canonical.

Persistent containers are deferred as a runtime variant. They must share
the same prepare and commit paths when implemented.

### Termination

The runtime reports a normalized termination reason. Substrate-reserved
reasons include `crash`, `timeout`, `interrupted`, and
`protocol_violation`. Project-defined reasons are declared by the block
and select which output slots are expected.

`BlockExecution.status` is derived from termination reason by
`flywheel.termination.derive_status`.

### Commit

Commit captures managed state for non-reserved termination reasons,
collects only the output slots declared for the observed termination
reason, validates candidates, registers accepted artifacts, quarantines
rejected outputs, and records the `BlockExecution`.

Reserved failure reasons record the execution without accepting
artifacts or advancing state.

Commit also ingests execution telemetry candidates from
`/flywheel/telemetry`. Telemetry is a separate ledger lane from
artifacts and state: valid candidates become `ExecutionTelemetry`
records tied to the execution, and malformed candidates become durable
telemetry rejection records. Telemetry ingest is non-fatal and never
changes execution status. Flywheel validates and records telemetry
candidates, and preserves rejected candidate bytes under
`telemetry_rejections/` when possible, but battery wrappers own any
stronger provenance boundary for the bytes they emit.

## Workspaces

A workspace is a directory under a project's foundry. It contains
`workspace.yaml`, artifact instances, state snapshots, run records,
proposals, quarantine, and recovery directories.

`workspace.yaml` is the durable ledger. It records workspace metadata,
artifact declarations, artifact instances, state snapshots, block
executions, execution telemetry and telemetry rejections, run records,
and lifecycle events.

Workspace mutations go through sanctioned methods such as
`register_artifact`, `register_git_artifact`, `register_state_snapshot`,
`record_execution`, `record_execution_telemetry`,
`record_rejected_telemetry`, `begin_run`, `record_run_step`, and
`end_run`. Private mutators are not public write paths.

`Workspace.save()` writes atomically via a temporary file and replace.

## Templates and Foundry Layout

A project declares its foundry directory in `flywheel.yaml`. Current
template locations are:

```text
foundry/
  templates/
    workspaces/
    blocks/
    patterns/
  workspaces/
```

Workspace templates declare artifact declarations and the block names
available in the workspace. Block declarations live in
`templates/blocks/` and define image, lifecycle, inputs, outputs,
state mode, environment, and Docker arguments. Pattern declarations
live in `templates/patterns/`.

The long-term foundry layout is still being refined, but the conceptual
split is stable: templates describe capabilities; workspaces contain
results.

## Runs and Patterns

Runs are stored as `RunRecord` objects on the workspace. A run groups
related block executions without adding run-specific fields to
`BlockExecution`.

The current pattern implementation is intentionally lean:

* A pattern is an ordered list of steps.
* Each step has one cohort.
* Each cohort has members.
* Each member is one canonical block execution.
* Cohorts support `min_successes: all` and `min_successes: 1`.
* A pattern may declare lanes, which are run-scoped artifact
  resolution contexts.
* Pattern fixtures materialize ordinary copy artifacts per lane at run
  start.
* Pattern parameters are resolved once at run start, recorded on the
  `RunRecord`, and may be substituted into member environment values,
  member args, and invocation route args.
* Execution is sequential today, even when a cohort is semantically
  parallel.

Pattern inputs can bind directly to an artifact id or to a prior
member's output in the same run. Unbound copy inputs resolve to the
latest artifact in the member's lane, not the latest workspace-global
artifact by name. The pattern runner records fixture, member, lane, and
step results on the `RunRecord`.

Ad hoc `flywheel run block` may also pass `--param KEY=VALUE`; those
values are used only for `${params.KEY}` placeholders in invocation
route args. Pattern-only member env/arg substitution remains a pattern
surface.

The resolver is separate from execution. It reads the current pattern
and run prefix, chooses the next step, and does not mutate the
workspace.

## Batteries

Batteries are solve-once capabilities bundled with Flywheel for reuse.
They are above the substrate.

The bundled Claude battery currently includes:

* `batteries/claude/Dockerfile.claude`;
* `batteries/claude/entrypoint.sh`;
* `batteries/claude/agent_runner.py`;
* `examples/hello-agent/`.

Projects invoke the battery as an ordinary block by declaring a block
that uses a project image derived from the Claude battery. Prompt bytes
belong to that derived image's agent workflow definition, not to the
workspace artifact graph. Auth and other launch details live in the
block declaration, and durable outputs remain ordinary block execution
records, artifact instances, and state snapshots.

If an agent block needs another block to run next, it announces a
project termination reason whose block declaration routes to the child.
Flywheel commits the agent block first, then invokes the child through
the same canonical block execution path.
Richer battery selection is deferred.

## Future Work

### Battery Boundaries

Continue moving Claude-specific conventions into `batteries/claude/`
and battery-provided templates. Batteries should be invoked through
ordinary `flywheel run block` execution.

Do not add agent-shaped fields to `BlockExecution` unless the field is
meaningful for ordinary block executions.

### Block Invocation

Block invocation routes from a committed execution outcome to child
block execution. Future work should add operator-facing inspection
commands, decide whether pending tool calls from batteries need their
own durable record, and rely on the pattern resolver for loops,
limits, and conditional iteration.

### Persistent Containers

Implement persistent containers as a runtime variant that shares
canonical prepare and commit. Persistent runtimes must not register
artifacts, write state snapshots, or record executions directly.

### Pattern Expressiveness

Extend patterns with loops, branching, nested patterns, richer
`min_successes` policies, and scheduler controls. Preserve the
distinction between semantic cohorts and scheduling concurrency.

### Artifact Selection and Scope

Ad hoc block execution still uses latest-by-name for unbound copy
inputs. Pattern execution uses run-scoped lane resolution instead.
Future selection may need explicit policies such as best-scoring
artifact, per-subclass defaults, or promotion between contexts.

### Incremental Replacement

Replace incremental artifacts with tagged copy instances if that still
fits real workflows. Tagged copy instances would make amendment,
validation, quarantine, and concurrency uniform across artifact kinds.

### Commit-Pinned Git Mounts

Git artifact instances record a commit SHA, but runtime mounts still
use the live working tree. Future work should mount the exact committed
contents via `git archive`, `git worktree`, or equivalent.

### Workspace Schema

Introduce schema versioning and decide whether workspaces should store
full declaration snapshots or no template schema at all. The current
partial declaration snapshot is useful but not a complete long-term
contract.
