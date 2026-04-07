# Architecture Decisions

Design choices that shape flywheel's implementation. These serve
the vision but are distinct from it — they could change without
changing what flywheel is for.

## Artifacts

Artifacts are scoped to a workspace, immutable once recorded, and
not passed across workspace boundaries.

There are two storage kinds:

- **Copy artifacts** — files or directories stored directly in the
  workspace. Produced by block executions (checkpoints, scores,
  logs).
- **Git artifacts** — references to version-controlled code
  (repo + commit + path). Used to inject source code, configs,
  or prompts into a workspace.

From a block's perspective both behave the same: something
injectable with a name.

### Git artifacts require a clean working tree

Git artifacts are recorded as repo + commit SHA + path. Flywheel
does not support a dirty flag or content hashing to capture
uncommitted changes. If the working tree is dirty, the reference
is not reproducible, and flywheel should refuse to proceed rather
than record something it cannot recreate.

### Git artifact resolution happens twice

At **workspace creation**, flywheel records a baseline snapshot
(the current commit). This is informational — it captures what
the code looked like when the workspace was set up.

At **block execution**, flywheel re-resolves the current committed
state. This is what actually gets injected into the block and
recorded in the execution ledger. The baseline and execution-time
commits may differ if code was committed between workspace
creation and block execution. Both are worth recording.

### Artifact immutability enforcement

Once an artifact is recorded in a workspace, it cannot be
overwritten. `Workspace.record_artifact()` is the only way to
set an artifact value; it refuses to overwrite a previously
recorded artifact. This ensures the execution ledger is
trustworthy — an artifact referenced by a ledger entry will
never silently change.

## Workspaces

A workspace is a directory inside a project's workforce folder.
It is created from a template and contains all artifacts for a
unit of work.

### Directory layout

```
project_root/
├── flywheel.yaml             ← project config (harness_dir, etc.)
├── crates/                   ← project source code
└── workforce/                ← flywheel's folder
    ├── templates/            ← workspace templates
    │   └── improve_bot.yaml
    └── workspaces/
        └── my_workspace/
            ├── workspace.yaml   ← metadata + artifact registry
            └── artifacts/       ← copy artifacts stored here
```

### Templates

A template defines the capabilities of a workspace: what
artifacts exist, what blocks can run, and how containers are
built. Templates live in `workforce/templates/`.

A template declares:
- **Artifact declarations** — names and storage kinds (copy or
  git), with repo and path for git artifacts
- **Block definitions** — names, container images, and which
  declared artifacts they consume and produce

Block inputs and outputs must reference declared artifact names.
This is validated at template parse time.

### Workspace creation

`flywheel create workspace --name NAME --template TEMPLATE` runs
from the project root. It reads `flywheel.yaml` to find the
workforce directory, loads the template, resolves git artifacts
(refusing if dirty), and creates the workspace directory with
a `workspace.yaml` metadata file.
