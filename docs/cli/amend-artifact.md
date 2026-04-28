# `flywheel amend artifact`

## Purpose

Register a corrective successor to an *accepted* `copy`-kind
artifact instance. The new instance carries a `supersedes`
lineage pointer at the predecessor's artifact id, so the audit
trail records *which* prior instance this correction is
replacing.

The predecessor is never mutated. Both instances coexist in the
workspace; the default input-resolution policy ("latest copy
instance for the slot") will pick the new one until something
newer lands.

## When to use it

Use `amend artifact` when an existing accepted artifact instance
turns out to be wrong (mislabeled bytes, latent bug discovered
later, hand-edited corrections) and you want to register a
corrective successor with a recorded lineage instead of doing
an opaque `import artifact` that loses the link to the
predecessor.

Do not use `amend artifact` to:

* register a bare new instance with no lineage — use
  [`flywheel import artifact`](import-artifact.md). Use `amend`
  only when you genuinely supersede an identifiable
  predecessor.
* register a corrective successor for an output slot that was
  *rejected* by its validator (i.e. it never became an accepted
  artifact). Use [`flywheel fix execution`](fix-execution.md)
  for that case; it records the rejection-pair lineage instead
  of an artifact-id lineage.
* mutate the predecessor. `amend artifact` only writes a new
  instance; the old one stays exactly as it was, and consumers
  that explicitly bound to it (e.g. via `--bind <slot>=<id>`)
  continue to see the same bytes.
* "fork" against a different artifact name. The successor is
  registered under the predecessor's *own* declaration name;
  cross-name lineage is rejected.

## Prerequisites

* **`flywheel.yaml` at the project root** declaring
  `foundry_dir`. Optionally `artifact_validators:` if the
  project ships validators (the corrected bytes are validated
  through the same registry an `import` would use).
* **The workspace exists** at `--workspace`.
* **The predecessor artifact exists** in the workspace's
  ledger and is `kind: copy`. Both conditions are checked
  before any bytes are staged.
* **The source path exists** on disk, is readable, and is a
  directory. Wrap single files in a directory first.

## Invocation

```
flywheel amend artifact --workspace <path> \
  --artifact <artifact-id> \
  --from <source-path> \
  [--reason <text>]
```

Run from the project root. All flags except `--reason` are
required.

* `--workspace` — absolute or relative path to the workspace
  directory.
* `--artifact` — the predecessor `ArtifactInstance.id` (e.g.
  `checkpoint@a3f7b2`) being superseded. The successor is
  registered under the same declaration name; you do not
  pass the name separately.
* `--from` — host path to the directory whose contents become
  the successor instance.
* `--reason` — optional human-readable explanation recorded on
  the resulting `ArtifactInstance.supersedes_reason`.
  Strongly recommended so future readers (including AI agents) can
  see *why* the predecessor was superseded without out-of-band
  notes.

Example:

```bash
flywheel amend artifact \
  --workspace foundry/workspaces/2026-04-22-baseline \
  --artifact checkpoint@a3f7b2 \
  --from /tmp/corrected-checkpoint \
  --reason "weights from sweep run 2026-04-21#42 were truncated; this is the full upload"
```

## What gets created / changed

* A new directory
  `<workspace>/artifacts/<name>@<short-uuid>/` (where `<name>`
  is the predecessor's declaration name) containing a copy of
  `--from`'s contents.
* A new `ArtifactInstance` recorded in `workspace.yaml`:
  * `name` matches the predecessor's `name`.
  * `kind` is `copy`.
  * `produced_by` is `None` (amendments are not produced by a
    block execution).
  * `supersedes.artifact_id` is `--artifact`.
  * `supersedes_reason` is `--reason` (or `None` when
    omitted).
  * `created_at` is the registration time.
* The predecessor instance is **not** modified. Its bytes,
  metadata, and ledger entry are exactly as they were before
  the call.
* No `BlockExecution` record is created or modified. Lineage
  flows through the `ArtifactInstance.supersedes` field on the
  new instance, not through any execution metadata.

The workspace lock is held for the duration of the staging +
validation + ledger update.

## Multiple successors and forking

The schema allows multiple successors to point at the same
predecessor — `SupersedesRef` is a backward-only pointer, never
a "current child" mutation on the parent. Calling `amend
artifact` twice against the same `--artifact` produces two
sibling successors, each with the same `supersedes.artifact_id`
but different ids of their own. This is supported on purpose;
operators who want to fork-and-decide can do so without losing
either branch. The default input-resolution policy still
picks the latest by `created_at`; bind explicitly when a
specific successor is required.

## Artifact validation

The corrected bytes are run through the project's artifact
validator registry exactly the way [`flywheel import
artifact`](import-artifact.md) runs them — successors get no
validation discount. See [import-artifact.md § "Artifact
validation"](import-artifact.md#artifact-validation) for the
contract.

A failed validation registers nothing: the staged directory is
removed, no `ArtifactInstance` is recorded, the workspace state
is unchanged, and the predecessor remains the latest accepted
instance for the slot.

## Failure modes

| Condition | Exception | Recovery |
| --- | --- | --- |
| `flywheel.yaml` missing or malformed at the cwd. | `FileNotFoundError` / `ValueError`. | Run from the project root. |
| `--workspace` path does not contain a `workspace.yaml`. | `FileNotFoundError`. | Confirm the path. |
| `--artifact` is not present in the workspace ledger. | `ValueError("Artifact … does not exist in this workspace")`. | Confirm the predecessor id. List candidates with `yq '.artifacts | keys' <workspace>/workspace.yaml`. |
| `--artifact`'s declaration name is not `kind: copy`. | `ValueError("Only copy artifacts can be imported …")`. | Same constraint as `import artifact`. Git artifacts are not amendable today. |
| `--from` path does not exist. | `FileNotFoundError`. | Confirm the source path. |
| `--from` is a single file. | `ValueError("Artifact source must be a directory …")`. | Wrap it in a directory and pass that. |
| The validator rejects the corrected bytes. | `flywheel.artifact_validator.ArtifactValidationError`. | Read the validator message, fix the source, retry. |

Predecessor existence is checked before any bytes are staged,
so a bad `--artifact` fails cheaply and leaves no partial
state behind.

## Verification

Exit code zero plus the printed
`Registered '<name>' as '<name>@<id>' superseding
'<predecessor-id>'` line is sufficient. To inspect:

```bash
ls <workspace>/artifacts/<name>@<id>/
yq '.artifacts."<name>@<id>".supersedes' \
  <workspace>/workspace.yaml
```

The `supersedes.artifact` field in `workspace.yaml` is the
canonical lineage record.

## Typical next steps

* Confirm the new instance is what consumers will pick:
  `yq '.artifacts | to_entries | map(select(.value.name ==
  "<name>")) | sort_by(.value.created_at)'
  <workspace>/workspace.yaml`. The last entry is what the
  default input-resolution policy will bind.
* Run a downstream block; no new flags are needed — the
  default policy will pick the successor.
* Pin a specific instance for reproducibility:
  `--bind <slot>=<artifact-id>` on `flywheel run block`.

## Implementation pointers

* CLI entry: [`flywheel/cli.py`](../../flywheel/cli.py)
  `amend_artifact()`.
* Substrate: [`flywheel/workspace.py`](../../flywheel/workspace.py)
  `Workspace.register_artifact()` (the `supersedes` /
  `supersedes_reason` parameters and `_check_supersedes`
  helper).
* Schema: [`flywheel/artifact.py`](../../flywheel/artifact.py)
  (`SupersedesRef`).
* Tests:
  * [`tests/test_cli.py`](../../tests/test_cli.py)
    `TestMainAmendArtifact`.
  * [`tests/test_workspace.py`](../../tests/test_workspace.py)
    `TestRegisterArtifactSupersedes` (substrate-level
    coverage including same-name lineage enforcement).
