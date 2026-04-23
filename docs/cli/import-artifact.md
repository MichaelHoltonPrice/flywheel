# `flywheel import artifact`

## Purpose

Register an external directory as a fresh `copy`-kind artifact
instance inside an existing workspace. The directory's contents
are copied (never aliased) into the workspace's artifact store,
get a fresh artifact id, and bind identically from that point
on to artifacts produced by block executions.

Artifact instances are always directory-shaped, mirroring the
per-slot output directories block executions write into. To
import a single file, wrap it in a directory first; the
artifact's structure is then explicit at the call site rather
than implicit in flywheel's behavior.

## When to use it

Use `import artifact` to seed a workspace with starting state
that did not come from a block execution:

* a starting checkpoint produced outside the workspace,
* a fixture or reference dataset,
* a hand-curated config or prompt blob the workspace's blocks
  expect at a declared name.

Do not use `import artifact` to:

* register `git` or `incremental` artifacts. Only `copy`
  declarations are importable; the others are populated by
  workspace creation (`git` baselines) or by block execution
  (`incremental` appends).
* "patch" an existing instance. Artifact instances are
  immutable; an import always creates a *new* instance.
* register names not in the template's `artifacts:` list.
  Undeclared names are rejected — see
  [../architecture.md](../architecture.md) §
  "Undeclared artifacts are rejected".

## Prerequisites

* **`flywheel.yaml` at the project root** declaring
  `foundry_dir`. Optionally `artifact_validators:
  module.path:factory` if the project ships validators (see
  "Artifact validation" below); not required otherwise.
* **The workspace exists** at the path passed via
  `--workspace`. Create it first with `flywheel create
  workspace`.
* **The artifact name is declared** in the workspace's template
  *and* declared as `kind: copy`. The workspace records its
  declarations at create time; an unknown name is rejected
  even if the template was edited after the workspace was
  created.
* **The source path exists** on disk, is readable, and is a
  directory. The entire tree is copied. Single-file sources are
  rejected — wrap the file in a directory and pass that.

## Invocation

```
flywheel import artifact --workspace <path> --name <name> \
  --from <source-path> [--source <description>]
```

Run from the project root. All flags except `--source` are
required.

* `--workspace` — absolute or relative path to the workspace
  directory (the one containing `workspace.yaml`).
* `--name` — the declared artifact name to bind under
  (e.g. `checkpoint`).
* `--from` — host path to the directory whose contents become
  the new instance. Must be a directory; wrap single files
  before importing. Symlinks inside the source are resolved.
* `--source` — free-text provenance string recorded on the
  resulting `ArtifactInstance.source` field. Defaults to the
  resolved `--from` path if omitted; supply something more
  meaningful (URL, run id, ticket) when the path alone won't
  jog future readers' memory.

Example:

```bash
flywheel import artifact \
  --workspace foundry/workspaces/2026-04-22-baseline \
  --name checkpoint \
  --from /tmp/exported/checkpoint-bundle \
  --source "from sweep run 2026-04-21#42"
```

Wrapping a single file before import (the canonical pattern):

```bash
mkdir -p /tmp/cp-wrap && cp /tmp/checkpoint.pt /tmp/cp-wrap/
flywheel import artifact \
  --workspace foundry/workspaces/2026-04-22-baseline \
  --name checkpoint \
  --from /tmp/cp-wrap \
  --source "from sweep run 2026-04-21#42"
```

## What gets created / changed

* A new directory `<workspace>/artifacts/<name>@<short-uuid>/`
  containing a copy of `--from`'s contents.
* A new `ArtifactInstance` recorded in `workspace.yaml`:
  * `name` matches `--name`.
  * `kind` is `copy`.
  * `produced_by` is `None` (imports are not the result of a
    block execution; this distinguishes them in the ledger
    from block-produced instances).
  * `source` carries the `--source` string (or the resolved
    `--from` path).
  * `created_at` is the import time.
* No `BlockExecution` is recorded — `import artifact` is a
  ledger write, not a run.

The workspace lock is held for the duration of the copy +
ledger update, so concurrent commands against the same
workspace serialize against this one.

## Artifact validation

If the project's `flywheel.yaml` configures
`artifact_validators:` (a `module.path:factory` returning an
`ArtifactValidatorRegistry`) and the registry has an entry for
`--name`, the registered callable is invoked against a staged
candidate **before** any workspace mutation:

```python
validator(name, declaration, staged_path) -> None
```

`staged_path` is a flywheel-owned directory containing the
exact bytes that would become the artifact instance — never the
operator's `--from` path. This matches the validator contract
used at every other artifact-finalization site.

* Returning `None` accepts the import; the staging directory is
  atomically renamed into place and the instance is recorded.
* Raising `flywheel.artifact_validator.ArtifactValidationError`
  rejects the import. The staging directory is removed; *no*
  instance is recorded; the workspace state is unchanged. The
  CLI exits non-zero and the validator's message is printed.
* Any other exception raised by the validator is wrapped in
  `ArtifactValidationError` (with `__cause__` set) before
  propagating, so a buggy validator is reported as a
  validation failure rather than an opaque crash.

Names without a registered validator are accepted
unconditionally; flywheel never invents a default. See
[../architecture.md](../architecture.md) § "Artifact validation"
for the broader contract — in particular, how the same registry
is consulted after every block execution.

## Failure modes

| Condition | Exception | Recovery |
| --- | --- | --- |
| `flywheel.yaml` missing or malformed at the cwd. | `FileNotFoundError` / `ValueError` from the project config loader. | Run from the project root. Confirm `flywheel.yaml` exists and declares `foundry_dir`. |
| `--workspace` path does not contain a `workspace.yaml`. | `FileNotFoundError`. | Confirm the path. Create the workspace first if it doesn't exist. |
| `--name` is not declared in the workspace's artifact declarations. | `ValueError("Artifact … not declared in this workspace")`. | Add the declaration to the template, or pick a declared name. Note: editing the template does not retroactively re-declare an existing workspace; create a new workspace from the updated template. |
| `--name` is declared but not as `kind: copy`. | `ValueError("Only copy artifacts can be imported …")`. | Use the right kind for the source. Git artifacts are pulled from a clean working tree at workspace creation; incremental artifacts are appended by block executions. |
| `--from` path does not exist. | `FileNotFoundError`. | Confirm the source path. Quote paths that contain spaces. |
| `--from` path is a single file (or any non-directory). | `ValueError("Artifact source must be a directory …")`. | Wrap the file in a directory and pass that directory. |
| `artifact_validators:` factory cannot be imported or returns the wrong type. | `ImportError` / `ValueError`. | Fix `flywheel.yaml`'s import path or the factory's return type. |
| The registered validator rejects the staged candidate. | `flywheel.artifact_validator.ArtifactValidationError`. | The validator's message says why. Fix the source (or the validator), then retry. |

On any failure — staging, validation, or the final rename —
flywheel removes the staging directory under
`<workspace>/artifacts/_staging-<name>-…/` before propagating,
so a failed import leaves no partial state behind.

## Verification

Exit code zero plus the printed `Imported '<name>' as
'<name>@<id>'` line is sufficient. To inspect:

```bash
ls <workspace>/artifacts/<name>@<id>/
yq '.artifacts."<name>@<id>"' <workspace>/workspace.yaml
```

The recorded instance's `source` and `created_at` fields are
the canonical provenance — preserve the import command in your
notes if `--source` doesn't already include it.

## Typical next steps

* `flywheel run block --workspace <path> --block <name>
  --template <name>` — execute a block that consumes the
  imported artifact via its declared input slot.
* `flywheel run pattern <name> --workspace <path> --template
  <name>` — execute a pattern that consumes the imported
  artifact through the same slot resolution.

The default input-resolution policy ("latest copy instance for
the slot") will pick up the freshly imported instance until a
later block execution produces a newer one. Bind explicitly
(`--bind <slot>=<artifact-id>`) when you need to pin a
specific instance.

## Implementation pointers

* CLI entry: [`flywheel/cli.py`](../../flywheel/cli.py)
  `import_artifact()`. Argparse setup at lines 61-74.
* Workspace mutation:
  [`flywheel/workspace.py`](../../flywheel/workspace.py)
  `Workspace.register_artifact()`. Stages into
  `<workspace>/artifacts/_staging-<name>-…/`, validates, then
  atomically renames into the canonical artifact directory.
* Validator surface:
  [`flywheel/artifact_validator.py`](../../flywheel/artifact_validator.py)
  (`ArtifactValidationError`,
  `ArtifactValidatorRegistry`, `ArtifactValidator` callable
  type).
* Project config plumbing:
  [`flywheel/config.py`](../../flywheel/config.py)
  `ProjectConfig.load_artifact_validator_registry()`.
* Tests:
  * [`tests/test_workspace.py`](../../tests/test_workspace.py)
    `TestRegisterArtifact` covers happy path, validator pass,
    validator rejection (no state change), and the
    "no registered validator" passthrough.
  * [`tests/test_artifact_validator.py`](../../tests/test_artifact_validator.py)
    covers the registry surface (registration, lookup,
    exception wrapping).
  * [`tests/test_config.py`](../../tests/test_config.py)
    `TestArtifactValidators` covers `flywheel.yaml`
    parsing and factory resolution.
