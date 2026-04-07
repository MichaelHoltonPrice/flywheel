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
configured. Templates live in `workforce/templates/`.

A template declares:
- **Artifact declarations** — names and storage kinds (copy or
  git), with repo and path for git artifacts
- **Block definitions** — names, container images, resource
  requirements, and input/output artifact mappings with
  container paths

Block inputs and outputs must reference declared artifact names.
This is validated at template parse time.

## Block execution

### Dependency injection: build-time vs runtime

Blocks receive inputs via dependency injection. There are two
forms:

- **Runtime injection** — artifacts mounted into a running
  container as files or directories. Flywheel handles this
  directly via Docker volume mounts.
- **Build-time injection** — source code or data compiled into
  the Docker image itself (e.g., a game engine compiled into
  PyO3 bindings). Flywheel tracks the dependency via git
  artifacts for provenance, even though the injection happens
  at image build time, not container run time.

Both are real dependencies. The distinction is practical:
runtime inputs can change between runs without rebuilding the
image; build-time inputs require a rebuild. This is an
implementation detail, not a fundamental design choice — a
future block could accept source code at runtime and compile
on startup.

### Input and output slots

Each block declares its inputs and outputs as named slots,
each mapped to a container path:

```yaml
blocks:
  - name: train
    image: cyberloop-train:latest
    docker_args: ["--gpus", "all", "--shm-size", "8g"]
    inputs:
      - name: checkpoint
        container_path: /input/checkpoint
        optional: true
    outputs:
      - name: checkpoint
        container_path: /output
```

Flywheel mounts artifact directories at the declared container
paths. What files exist inside is the container's concern, not
flywheel's.

### Convention-based output recording

Flywheel creates a directory per declared output artifact and
mounts it at the block's declared container path. The container
writes files there. After the container exits, flywheel records
whatever appeared in each output directory as a CopyArtifact.

This avoids requiring containers to write a manifest or be
flywheel-aware. The template already declares the contract
(what a block produces), so a manifest would be redundant.
If we later need richer signaling (metadata, partial results,
error details), manifests can be layered on top.

### Docker configuration

Block definitions include a `docker_args` list for pass-through
Docker flags (e.g. `["--gpus", "all", "--shm-size", "8g"]`).
Flywheel does not interpret these — they are project-specific
and passed directly to `docker run` before the image name.

### Workspace creation

`flywheel create workspace --name NAME --template TEMPLATE` runs
from the project root. It reads `flywheel.yaml` to find the
workforce directory, loads the template, resolves git artifacts
(refusing if dirty), and creates the workspace directory with
a `workspace.yaml` metadata file.
