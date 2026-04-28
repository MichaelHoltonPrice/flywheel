# `flywheel create workspace`

## Purpose

Materialize a fresh, empty workspace from a template. The
workspace is a durable directory under the project's foundry that
will accumulate artifact instances and execution records over its
lifetime.

## When to use it

Use `create workspace` once per logical job (a training run, an
evaluation campaign, an agent session that should be inspectable
later). Subsequent commands (`import artifact`, `run block`,
`run pattern`) operate on the workspace directory by path.

Do not use `create workspace` to "reset" an existing workspace —
flywheel raises `FileExistsError` rather than overwriting. To
start over, delete the workspace directory or pick a new name.

To run a one-off block without a persistent workspace, the
recommended path is still to create a short-lived workspace and
delete it afterwards; flywheel's machinery (artifacts, ledger,
state restore) assumes a workspace exists.

## Prerequisites

* **`flywheel.yaml` at the project root** declaring `foundry_dir`
  (the directory flywheel manages). Optional validator registry
  factories may also be declared there; they are not required for
  `create workspace`.
* **The workspace template file exists** at
  `<foundry_dir>/templates/workspaces/<template_name>.yaml`.
* **The template's referenced blocks exist** in
  `<foundry_dir>/templates/blocks/<name>.yaml` (resolved by the
  project block registry; missing blocks fail template loading, not
  workspace creation, but the failure surfaces during this command).
* **For every `git`-kind artifact declared in the template:**
  * The referenced repository (`repo:` field) is on disk and is a
    git working tree.
  * That repo has **no uncommitted changes**. Flywheel refuses to
    create a workspace from a dirty repo because the recorded
    baseline commit would not actually reflect what's on disk.
  * The `path:` referenced inside the repo exists at HEAD.

`copy`-kind artifact declarations require no pre-existing files;
instances are produced later, by block executions or by
`flywheel import artifact`.

## Invocation

```
flywheel create workspace --name <name> --template <template>
```

Both flags are required. The command runs from any directory
containing a `flywheel.yaml`; flywheel resolves the foundry from
that file. Run from the project root.

* `--name` — workspace name. Validated to letters, digits,
  hyphens, and underscores. Used as the directory name under
  `<foundry_dir>/workspaces/`.
* `--template` — template name (without `.yaml`). Must match a
  file at `<foundry_dir>/templates/workspaces/<template>.yaml`.

Example:

```bash
flywheel create workspace --name 2026-04-22-baseline --template cyberloop
```

## What gets created

A new directory at `<foundry_dir>/workspaces/<name>/` containing:

* `artifacts/` — empty subdirectory that will hold `copy`
  artifact instance dirs as they are produced.
* `workspace.yaml` — the durable workspace ledger. At creation
  time it contains:
  * `name`, `template_name`, `created_at`.
  * `artifact_declarations` — a map from declaration name to
    storage kind (`copy` or `git`).
  * `artifacts` — a map of artifact instances. At creation time
    this is empty *except* for one auto-registered `<name>@baseline`
    instance per `git` declaration, which records the resolved
    `repo` path and `commit` SHA at workspace-create time. `copy`
    declarations produce no instances at create time.
  * `executions` — empty.
  * `state_snapshots` — empty.
  * `events` — empty.
  * `runs` — empty.

The workspace is fully usable as soon as this command returns
exit code zero.

## Failure modes

| Condition | Exception | Recovery |
| --- | --- | --- |
| `flywheel.yaml` missing or malformed at the cwd. | `FileNotFoundError` / `ValueError` from project config loader. | Run from the project root. Confirm `flywheel.yaml` exists and declares `foundry_dir`. |
| Template file missing at `<foundry_dir>/templates/workspaces/<template>.yaml`. | `FileNotFoundError`. | Confirm the template name matches the file name (without `.yaml`) and lives under the configured foundry. |
| `--name` contains characters outside letters, digits, hyphens, or underscores. | `ValueError("invalid name…")`. | Pick a name that matches the allowed pattern. |
| Workspace with that name already exists. | `FileExistsError`. | Pick a different name, or delete the existing workspace directory first. |
| A `git`-kind artifact's repo has uncommitted changes. | `RuntimeError("uncommitted changes…")`. | Commit or stash the changes in that repo, then retry. |
| A `git`-kind artifact's `path:` does not exist in the repo at HEAD. | `FileNotFoundError`. | Either fix the template's `path:` field or create the missing path in the repo and commit. |
| A `git`-kind artifact declaration is missing `repo:` or `path:`. | `ValueError`. | Fix the template. |

On any failure during creation, flywheel removes the partial
workspace directory before raising
([`Workspace.create`](../../flywheel/workspace.py) `try/except`
in the body), so a failed invocation leaves no half-built
workspace behind.

## Verification

Exit code zero plus the workspace directory existing at
`<foundry_dir>/workspaces/<name>/` is sufficient. To inspect:

```bash
ls <foundry_dir>/workspaces/<name>/
cat <foundry_dir>/workspaces/<name>/workspace.yaml
```

`workspace.yaml` should list the expected `artifact_declarations`
(matching the template) and a `<name>@baseline` artifact for each
`git` declaration with the recorded `commit`.

There is no `flywheel status` command yet; reading the YAML
directly is the canonical way to inspect a workspace. The format
is stable enough to grep.

## Typical next steps

* `flywheel import artifact` — add files to the workspace as an
  immutable `copy` artifact instance (e.g., a starting checkpoint,
  a fixture). See [import-artifact.md](import-artifact.md).
* `flywheel run block` — execute one block ad hoc against the
  workspace. Per-command doc not yet written; see argparse setup
  at [`flywheel/cli.py`](../../flywheel/cli.py) line ~76.
* `flywheel run pattern` — execute a declarative pattern.
  Per-command doc not yet written; see argparse setup at
  [`flywheel/cli.py`](../../flywheel/cli.py) line ~130.

## Implementation pointers

* CLI entry: [`flywheel/cli.py`](../../flywheel/cli.py)
  `create_workspace()` (lines 275-299).
* Argparse setup: [`flywheel/cli.py`](../../flywheel/cli.py)
  lines 52-58.
* Workspace creation logic:
  [`flywheel/workspace.py`](../../flywheel/workspace.py)
  `Workspace.create()` (line 600+). Includes name validation,
  directory creation, artifact declaration registration, git
  baseline resolution with dirty-repo guard, and cleanup-on-failure.
* Project config loader (resolves `flywheel.yaml`):
  [`flywheel/cli.py`](../../flywheel/cli.py) `load_project_config()`.
* Template loader: [`flywheel/template.py`](../../flywheel/template.py)
  `Template.from_yaml()`.
* Tests:
  * [`tests/test_workspace.py`](../../tests/test_workspace.py)
    `TestWorkspaceCreate` covers directory creation, name and
    template recording, artifact declarations, git baselines,
    copy artifacts not pre-created, duplicate-name rejection,
    invalid-name rejection, dirty-repo rejection, and
    cleanup-on-failure.
  * [`tests/test_cli.py`](../../tests/test_cli.py) covers the
    CLI surface (`test_creates_workspace`,
    `test_creates_workspace_yaml`, plus negative paths).
