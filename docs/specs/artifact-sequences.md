# Artifact Sequences

Artifact sequences are append-only ordered references to immutable
artifact instances. They are a ledger layer over artifacts, not a
storage kind, tag system, replacement relation, or resolver shortcut.

The primary use case is ordered history: for example an ARC game
sequence alternating `game_state` and `action` artifacts. A block can
consume the whole history as one input without relying on
workspace-global "latest by name" resolution.

## Ledger Model

A sequence entry has this logical shape:

```python
ArtifactSequenceEntry(
    sequence_name: str,
    scope: SequenceScope,
    index: int,
    artifact_id: str,
    role: str | None,
    recorded_at: datetime,
)
```

The natural key is `(sequence_name, scope, index)`. There is no
synthetic sequence-entry id. Entries are append-only; Flywheel exposes
no public mutation API for removing, reordering, or rewriting them.

`artifact_id` points at a normal immutable artifact instance. The
artifact's `produced_by` field remains the source of execution
provenance; sequence entries do not denormalize producer ids or
termination reasons.

## Scopes

Flywheel supports three concrete sequence scopes:

* `workspace`: one sequence shared across the workspace.
* `run(run_id)`: one sequence scoped to a pattern run.
* `lane(run_id, lane)`: one sequence scoped to a pattern lane.

Template declarations use scope policies rather than concrete ids:

* `workspace`
* `enclosing_run`
* `enclosing_lane`

`enclosing_run` and `enclosing_lane` require a pattern run context. An
ad-hoc `flywheel run block` of a block whose output appends to either
scope is an error before the container starts. Sequence inputs using
those scopes likewise fail during prepare if no matching run context
exists.

Lane-scoped sequence appends validate that a block-produced artifact
was produced in the same lane. During commit, the caller's explicit
run/lane context is authoritative because the pattern member record is
written only after the block returns.

## Output Declaration

An output slot may append its committed artifact instance to a sequence:

```yaml
outputs:
  normal:
    - name: game_state
      container_path: /output/game_state
      sequence:
        name: arc_game
        scope: enclosing_lane
        role: state
```

The output slot name must still be a declared artifact. The `sequence`
mapping accepts:

* `name`: required sequence name.
* `scope`: optional scope policy, default `workspace`.
* `role`: optional role label recorded on each appended entry.

Sequence append happens only after the output artifact is successfully
registered and the successful `BlockExecution` record exists. If any
expected output slot is rejected and the execution fails, no sequence
entries are appended for that execution.

## Input Declaration

An input slot may consume the whole current sequence snapshot:

```yaml
inputs:
  - name: arc_history
    container_path: /input/arc_history
    sequence:
      name: arc_game
      scope: enclosing_lane
```

The input slot name is the mount name and does not need to be an
artifact declaration. Input sequence declarations do not accept
`role`; role filtering and selector grammar are intentionally absent.
A sequence input always means "the full ordered snapshot at prepare
time."

Explicit artifact bindings still win over normal input resolution. If
no explicit binding is supplied and the slot has a sequence
declaration, Flywheel stages the sequence snapshot and records it on
`BlockExecution.input_sequence_bindings`. It does not fall through to
workspace-global latest artifact resolution. An empty sequence is a
valid binding.

## Mount Contract

Flywheel stages a sequence input as a directory containing a manifest
and one subdirectory per entry:

```text
/input/arc_history/
  manifest.json
  00000_state/
  00001_action/
```

Directory names are `{index:05d}_{role}` when `role` is set and
`{index:05d}` otherwise. Roles use Flywheel's normal name validation,
so directory names are stable without further escaping.

`manifest.json` has `manifest_version: 1` and this shape:

```json
{
  "manifest_version": 1,
  "sequence_name": "arc_game",
  "scope": {"kind": "lane", "run_id": "run_123", "lane": "lane_0"},
  "length": 2,
  "entries": [
    {
      "index": 0,
      "role": "state",
      "artifact_name": "game_state",
      "artifact_id": "game_state@abc",
      "directory": "00000_state",
      "produced_by": "exec_123"
    }
  ]
}
```

For an empty sequence, Flywheel still writes `manifest.json` with
`length: 0` and `entries: []`, and creates no entry directories.
For fixture-materialized or imported artifacts, `produced_by` is
`null`.

Flywheel copies entry artifact bytes into a per-execution staging
directory and mounts that directory at the slot's `container_path`.
It does not create one bind mount per entry.

For persistent runtimes, the same per-slot directory layout is staged
under the request's `input_root/<slot>/`; there is no per-slot bind
mount and the block declaration's `container_path` is not used by the
persistent worker. The worker discovers the input root from the
persistent-runtime request envelope.

## Execution Record

`BlockExecution.input_bindings` remains the mapping for ordinary
single-artifact input slots.

`BlockExecution.input_sequence_bindings` records sequence-shaped input
slots separately. Each binding stores the concrete scope and the exact
ordered entry refs consumed at prepare time. This snapshot is frozen:
later appends to the same sequence do not change what a prior
execution is recorded as having consumed.

## Workspace Invariants

`Workspace.record_sequence_entry` enforces:

* the referenced artifact exists;
* indices are substrate-assigned and monotonic per
  `(sequence_name, scope)`;
* the ledger is append-only;
* run and lane scopes reference existing run records;
* lane scopes reference an existing lane;
* block-produced artifacts were produced by succeeded executions;
* lane-scoped block-produced artifacts belong to the same lane.

Producer-status and producer-lane invariants apply only to artifacts
with a non-null `produced_by`. Fixture-materialized or imported
artifacts are exempt; their explicit sequence scope is the provenance
statement.

Workspace load rejects sequence entries with gaps or duplicate indices
inside one `(sequence_name, scope)` partition.

A prepare-time failure during sequence staging records the failed
`BlockExecution` without partial `input_sequence_bindings`. Sequence
input bindings are durable only when prepare completed far enough to
construct an `ExecutionPlan`.

## Deferred

The following are deliberately not part of artifact sequences:

* selector grammar such as latest, range, previous, nth, or role
  filters;
* sequence branching, forking, merging, or promotion between scopes;
* sequence-level metadata such as descriptions or closed timestamps;
* garbage collection or pruning;
* manual operator append commands;
* zero-copy sequence staging;
* best-by-score or metric-aware resolution;
* mid-execution inspection of sequences through a live substrate API.
