# Normative state model

This spec defines Flywheel's managed state model.  State is a
substrate concern, but it is not an artifact concern.

## Core Distinction

Artifacts model edge relationships between block executions: one
execution produces an immutable value that another execution may
consume through an explicit input binding or default artifact
selection rule.

State models node-local continuity across a sequence of block
executions: bytes are restored into an execution because that
execution participates in the same state lineage as a prior
execution, not because another block selected them as an artifact.

Consequences:

* State is never an input binding.
* State is never named in a pattern as an artifact dependency.
* State is never registered through `Workspace.register_artifact`.
* Artifact validators do not apply to state.
* A block that wants other blocks to consume state-like data must
  publish that data as an output artifact.

State validators may be introduced later, but they are a distinct
project hook from artifact validators.

## State Modes

Each block declares one state mode:

* `none`: the block has no substrate-visible state.
* `managed`: the substrate restores and captures state snapshots for
  each execution lineage.
* `unmanaged`: the block has stateful behavior that the substrate
  cannot capture or restore.

The canonical YAML spelling is:

```yaml
state: none
state: managed
state: unmanaged
```

For compatibility, `state: false` is equivalent to `none`, and
`state: true` is equivalent to `managed`.

`unmanaged` is a transparency declaration, not a storage mechanism.
It is intended for blocks such as persistent game servers whose
behavior depends on in-container state that Flywheel cannot serialize.
Executions of such blocks should be visibly less reproducible in the
ledger, but the substrate does not attempt to restore or capture their
state.

## Managed State Lineages

Managed state belongs to an execution lineage, not to the block
template globally.

A managed-state execution must have a `state_lineage_key`.  The key is
chosen by the caller:

* Pattern execution derives the key from pattern/run/step/member
  identity unless the pattern later grows an explicit override.
* Ad hoc `flywheel run block` requires an explicit `--state-lineage`
  when running a managed-state block.

The ad hoc command intentionally has no hidden default.  Reusing state
is a semantic decision, so operators must name the lineage they intend
to continue.

Managed state lineages are chains.  Each successful capture creates a
new snapshot whose predecessor is the previous snapshot for that
lineage, or `None` for the first snapshot.  The chain is primarily for
debugging, inspection, and recovery; the substrate restores only the
latest compatible snapshot unless a future recovery command explicitly
chooses otherwise.

## Snapshot Records

The workspace stores managed state snapshots separately from artifacts.
A snapshot record contains, at minimum:

* snapshot id
* state lineage key
* producing execution id
* predecessor snapshot id, if any
* capture timestamp
* compatibility identity
* workspace-relative byte location

State snapshots are not `ArtifactInstance`s and do not appear in the
artifact graph.

The only sanctioned write path for a managed snapshot is a future
`Workspace.register_state_snapshot(...)` API.  Private mutators may
exist underneath it, following the same privacy discipline as
artifact and execution records.

## Compatibility Identity

Before restoring a prior snapshot, Flywheel checks that it is
compatible with the execution about to run.

The M1 compatibility identity is:

* block name
* state mode
* image string
* block template content hash

If the latest snapshot for a lineage does not match the current
execution's compatibility identity, Flywheel rejects the execution
before invoke.  It does not try to migrate state, silently start over,
or recover smartly.

This rule aligns state with the broader declaration-compatibility
principle: changing templates or images must not silently reinterpret
an already-instantiated execution lineage.

## Prepare

During prepare, if the block's state mode is `managed`:

1. Validate that the caller supplied a state lineage key.
2. Resolve the latest snapshot for that lineage, if any.
3. Reject before invoke if the snapshot is incompatible.
4. Allocate an empty per-execution state mount directory.
5. If a compatible snapshot exists, copy its bytes into that mount.
6. Mount the directory into the container at `/flywheel/state`.

If no snapshot exists, the block receives an empty state mount.  Empty
state is the first-execution signal.

If snapshot restore fails after an execution id has been minted, the
execution is recorded as a prepare/stage-in failure.  The lineage is
unchanged.

Blocks with state mode `none` do not receive a state mount.  Blocks
with state mode `unmanaged` do not receive a managed state mount unless
some future runtime-specific contract says otherwise.

## Commit

After invoke, if the block's state mode is `managed`, Flywheel captures
state only for clean termination reasons:

* project-defined termination reasons capture state
* substrate-reserved runtime failures (`crash`, `timeout`,
  `interrupted`, `protocol_violation`) do not capture state

State capture happens before output artifacts are accepted.  If capture
fails:

* the execution is recorded with `failure_phase=state_capture`
* no new state snapshot is registered
* the lineage's latest snapshot remains unchanged
* output artifacts from this execution are not accepted
* Flywheel preserves the would-be state bytes in a best-effort recovery
  area

The recovery area is not artifact quarantine.  Recovery bytes are not a
snapshot, are not restored automatically, and are not part of the
artifact graph.  A future explicit operator command may promote or
discard recovery bytes.

If capture succeeds, Flywheel registers a new snapshot whose
predecessor is the previous latest snapshot for the same lineage, then
continues with output artifact commit.

## Validation

Artifact validators do not apply to state.

M1 managed state has only substrate-level checks: the substrate can
copy restored bytes into the mount and can copy captured bytes into the
workspace-owned snapshot location.

Project-owned state validators are deferred.  If introduced, they must
be separate from `ArtifactValidatorRegistry` and must run through the
canonical state-capture path.

## Patterns

Patterns own state-lineage policy for pattern-driven executions.

The default M1 derivation is lineage per pattern run, step, and member.
This prevents two unrelated members using the same block template from
silently sharing state.

Future pattern syntax may allow explicit lineage sharing across steps
or across resumed runs.  Such sharing must be recorded on `RunRecord`
or equivalent run metadata; it must not be inferred from block name
alone.

## Persistent Containers

Persistent containers are deferred by this spec.

Persistent blocks that depend on state the substrate cannot capture
should declare `state: unmanaged`.  This makes the reproducibility
limitation visible without creating a parallel artifact or state write
path.

The eventual persistent-container spec may add runtime-specific
behavior, but it must not cause persistent runtimes to write artifacts,
execution records, or managed state snapshots directly.

## Open Follow-Up

* State validator hook design.
* Operator commands for inspecting, pruning, promoting, or discarding
  state snapshots and recovery bytes.
* Retention policy for old snapshots.
* Pattern syntax for explicit state-lineage sharing.
* Compatibility behavior for deliberate state migrations.
