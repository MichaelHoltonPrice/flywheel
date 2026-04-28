# `flywheel fix execution`

## Purpose

Register a corrected `copy`-kind artifact instance for a slot
that an earlier block execution produced and the validator
rejected. The new instance carries a `supersedes` lineage
pointer at the rejected slot
(`(execution_id, slot)`) so the audit trail records *which*
failure this correction is responding to.

The original failed `BlockExecution` is never mutated. The
quarantined bytes (when present) stay where they are. The
corrected bytes become a fresh artifact instance with a fresh
id.

## When to use it

Use `fix execution` when a block execution failed with
`failure_phase=output_validate` and you have a corrected version
of the rejected slot's bytes ready to register. The classic
case: a training block produced a checkpoint that failed a
schema validator; you patched the schema in the corrected bytes
and want to land it as the new authoritative instance for that
slot, without re-running the (long, expensive) block.

Do not use `fix execution` to:

* register the *original* version of an accepted artifact —
  use [`flywheel import artifact`](import-artifact.md) instead.
* register a successor to an *accepted* (not rejected) artifact
  — use [`flywheel amend artifact`](amend-artifact.md). The
  two verbs differ only in which predecessor flavour they
  record on the new instance; both produce a fresh, validated
  `copy` instance.
* "rerun" the failed execution. `fix execution` only writes a
  new artifact instance; it does not produce a new
  `BlockExecution` record.
* fix slots that were not rejected. The slot must appear in
  the failed execution's `rejected_outputs` map; an unknown
  slot is rejected up front.

## Prerequisites

* **`flywheel.yaml` at the project root** declaring
  `foundry_dir`. Optionally `artifact_validators:` if the
  project ships validators (the corrected bytes are validated
  through the same registry an `import` would use).
* **The workspace exists** at `--workspace`.
* **The execution exists** in the workspace's ledger and its
  `rejected_outputs` map contains the named `--slot`. Both
  conditions are checked before any bytes are staged.
* **The artifact name is declared** as `kind: copy` in the
  workspace's template. Slot names are equal to artifact
  declaration names; the same kind/declaration constraints
  that apply to `import artifact` apply here.
* **The source path exists** on disk, is readable, and is a
  directory. Wrap single files in a directory first.

## Invocation

```
flywheel fix execution --workspace <path> \
  --execution <execution-id> \
  --slot <slot> \
  --from <source-path> \
  [--reason <text>]
```

Run from the project root. All flags except `--reason` are
required.

* `--workspace` — absolute or relative path to the workspace
  directory.
* `--execution` — the failed `BlockExecution.id` (e.g.
  `exec_a3f7b2e1`) whose rejected slot is being superseded.
* `--slot` — output slot name within that execution. Must
  appear in its `rejected_outputs`. The same string is used as
  the artifact declaration name for the new instance, since
  output slot names are artifact declaration names.
* `--from` — host path to the directory whose contents become
  the corrected instance.
* `--reason` — optional human-readable explanation recorded on
  the resulting `ArtifactInstance.supersedes_reason` so future
  readers (including AI agents) can see *why* the correction was
  registered without out-of-band notes. Strongly recommended.

Example:

```bash
flywheel fix execution \
  --workspace foundry/workspaces/2026-04-22-baseline \
  --execution exec_a3f7b2e1 \
  --slot checkpoint \
  --from /tmp/fixed-checkpoint \
  --reason "patched schema field 'optimizer_state' that the validator rejected"
```

## What gets created / changed

* A new directory
  `<workspace>/artifacts/<slot>@<short-uuid>/` containing a
  copy of `--from`'s contents.
* A new `ArtifactInstance` recorded in `workspace.yaml`:
  * `name` matches `--slot`.
  * `kind` is `copy`.
  * `produced_by` is `None` (corrections are not produced by a
    block execution).
  * `supersedes.rejection.execution_id` is `--execution`.
  * `supersedes.rejection.slot` is `--slot`.
  * `supersedes_reason` is `--reason` (or `None` when
    omitted).
  * `created_at` is the registration time.
* No `BlockExecution` record is created or modified. The
  original failed execution remains exactly as it was —
  `status="failed"`, `failure_phase="output_validate"`, and
  the same `rejected_outputs[slot]` entry. The lineage from
  the new instance to the old failure is the only link.
* No quarantined bytes are touched. The directory at
  `<workspace>/quarantine/<execution-id>/<slot>/` (when
  present) is left in place for inspection; deletion is left
  to a future GC story.

The workspace lock is held for the duration of the staging +
validation + ledger update.

## Artifact validation

The corrected bytes are run through the project's artifact
validator registry exactly the way [`flywheel import
artifact`](import-artifact.md) runs them — successors get no
validation discount. See [import-artifact.md § "Artifact
validation"](import-artifact.md#artifact-validation) for the
contract.

A failed validation registers nothing: the staged directory is
removed, no `ArtifactInstance` is recorded, the workspace state
is unchanged, and the original failed execution and any
existing quarantined bytes remain exactly as they were.

## Failure modes

| Condition | Exception | Recovery |
| --- | --- | --- |
| `flywheel.yaml` missing or malformed at the cwd. | `FileNotFoundError` / `ValueError`. | Run from the project root. |
| `--workspace` path does not contain a `workspace.yaml`. | `FileNotFoundError`. | Confirm the path. |
| `--execution` is not present in the workspace ledger. | `ValueError("supersedes.rejection.execution_id … does not exist …")`. | Confirm the execution id (look for `failed` records in `workspace.yaml`). |
| `--slot` is not present in that execution's `rejected_outputs`. | `ValueError("execution … has no rejected output for slot …")`. | The error message lists the rejected slots that *are* available. Pick one of those, or use [`amend artifact`](amend-artifact.md) if you meant to supersede an accepted instance. |
| `--slot` is not declared as `kind: copy` in the workspace's template. | `ValueError("Only copy artifacts can be imported …")`. | Same constraint as `import artifact`. Git artifacts are not amendable today. |
| `--from` path does not exist. | `FileNotFoundError`. | Confirm the source path. |
| `--from` is a single file. | `ValueError("Artifact source must be a directory …")`. | Wrap it in a directory and pass that. |
| The validator rejects the corrected bytes. | `flywheel.artifact_validator.ArtifactValidationError`. | Read the validator message, fix the source, retry. |

Predecessor existence is checked before any bytes are staged,
so a bad `--execution` / `--slot` fails cheaply and leaves no
partial state behind.

## Verification

Exit code zero plus the printed
`Registered '<slot>' as '<slot>@<id>' superseding rejected
slot '<slot>' of execution '<execution-id>'` line is sufficient.
To inspect:

```bash
ls <workspace>/artifacts/<slot>@<id>/
yq '.artifacts."<slot>@<id>".supersedes' \
  <workspace>/workspace.yaml
```

The `supersedes.rejected.execution` and `supersedes.rejected.slot`
fields in `workspace.yaml` are the canonical lineage record.

## Typical next steps

* Inspect the original quarantined bytes for comparison or
  archival:
  `ls <workspace>/quarantine/<execution-id>/<slot>/` (present
  when quarantine I/O succeeded; consult the
  `BlockExecution.rejected_outputs[<slot>].quarantine_path`
  field for the canonical path).
* Run a downstream block; the default input-resolution policy
  ("latest copy instance for the slot") will pick the corrected
  instance until something newer lands. Bind explicitly with
  `--bind <slot>=<artifact-id>` to pin the predecessor or the
  successor for reproducibility.

## Implementation pointers

* CLI entry: [`flywheel/cli.py`](../../flywheel/cli.py)
  `fix_execution()`.
* Substrate: [`flywheel/workspace.py`](../../flywheel/workspace.py)
  `Workspace.register_artifact()` (the `supersedes` /
  `supersedes_reason` parameters and `_check_supersedes`
  helper).
* Schema: [`flywheel/artifact.py`](../../flywheel/artifact.py)
  (`SupersedesRef`, `RejectionRef`).
* Quarantine convention:
  [`flywheel/quarantine.py`](../../flywheel/quarantine.py)
  (`quarantine_slot`).
* Tests:
  * [`tests/test_cli.py`](../../tests/test_cli.py)
    `TestMainFixExecution`.
  * [`tests/test_workspace.py`](../../tests/test_workspace.py)
    `TestRegisterArtifactSupersedes` (substrate-level
    coverage of all predecessor existence checks).
