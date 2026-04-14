# Architecture Decisions

Design choices that shape flywheel's implementation. These serve
the vision but are distinct from it — they could change without
changing what flywheel is for.

Everything in the main body of this document reflects agreed-upon
design decisions. Items that have a clear design direction but are
not yet implemented are collected in the "Future work" section at
the end.

## Artifacts

### Declarations and instances

An **artifact declaration** is a template-level concept: a named
artifact type with a storage kind. "The train block produces
checkpoints."

An **artifact instance** is a concrete, immutable record: a
specific checkpoint produced by a specific block execution, with
a unique ID, a timestamp, and full provenance. Multiple instances
can exist for the same declaration within a workspace.

This distinction is fundamental. The template declares what
*types* of artifacts exist. The workspace accumulates *instances*
of those types as blocks execute.

### Storage kinds

There are two storage kinds:

- **Copy artifacts** are files or directories stored directly in
  the workspace. Block executions produce them — checkpoints,
  scores, and logs are typical examples.
- **Git artifacts** are references to version-controlled code,
  recorded as repo, commit SHA, and path. They are used to
  inject source code, configurations, or prompts into a
  workspace.

From a block's perspective both kinds behave the same way: each
is something injectable with a name.

### Artifact IDs

Each artifact instance has a unique ID in the form
`name@identifier` — for example, `checkpoint@a3f7b2`,
`engine@baseline`, `score@e9c104`. The prefix is the
artifact declaration name; the suffix is a short UUID
(or ``baseline`` for workspace-creation git artifacts).
IDs are scoped to the workspace.

### Git artifacts require a clean working tree

Git artifacts are recorded as repo + commit SHA + path. Flywheel
does not support a dirty flag or content hashing to capture
uncommitted changes. If the working tree is dirty, the reference
is not reproducible, and flywheel should refuse to proceed rather
than record something it cannot recreate.

### Git artifact resolution creates new instances

At **workspace creation**, flywheel records a baseline git
artifact instance (e.g., `engine@baseline`). This captures what
the code looked like when the workspace was set up.

At **block execution**, flywheel re-resolves the current committed
state and records a **new** git artifact instance (e.g.,
`engine@a3f7b2`). The baseline instance is never mutated. Both
coexist in the workspace and the execution record shows exactly
which instance was used.

**Current limitation:** when a git artifact is mounted into a
container, flywheel mounts the live working tree path, not the
specific recorded commit. The commit SHA is recorded for
provenance but not enforced at mount time. If the working tree
has moved to a newer commit since the artifact was recorded,
the container sees the newer code.

### Imported artifacts

Users and agents can import externally created artifacts into a
workspace via ``Workspace.register_artifact()`` or the CLI command
``flywheel import artifact``. Imported artifacts are copied into
the workspace (preserving immutability) and recorded as normal
artifact instances. They have ``produced_by=None`` (like baseline
git snapshots) but carry a ``source`` field describing their
origin. Once registered, they bind identically to block-produced
artifacts.

### Immutability

Artifact instances are immutable once created. They are never
overwritten, updated, or deleted. A new block execution produces
new instances; it does not modify existing ones. This ensures
the execution history is trustworthy — an artifact referenced
by an execution record will never silently change.

## Block executions

A **block execution** is the atomic operation in flywheel: one
block, one container, one set of input artifact instances
producing output artifact instances.

Each block execution record includes:
- A unique execution ID.
- The block name.
- Input bindings, showing which artifact instance was used for
  each input slot.
- Output bindings, showing which artifact instance was produced
  for each output slot.
- A status indicator (succeeded, failed, or interrupted).
- Runtime metadata such as the image, docker args, exit code,
  and elapsed time.

Block executions are append-only. The workspace accumulates a
history of executions that forms the complete provenance graph.

A **run** is a higher-level concept — an orchestration pattern
composed of multiple block executions, potentially in parallel.
Runs are not yet implemented; block execution is the primitive.

### Dependency injection: build-time vs runtime

Blocks receive inputs via dependency injection. There are two
forms:

- **Runtime injection** mounts artifacts into a running container
  as files or directories. Flywheel handles this directly via
  Docker volume mounts.
- **Build-time injection** compiles source code or data into the
  Docker image itself (e.g., a game engine compiled into PyO3
  bindings). Flywheel tracks the dependency via git artifacts
  for provenance, even though the injection happens at image
  build time, not container run time.

Both are real dependencies. The distinction is practical:
runtime inputs can change between executions without rebuilding
the image; build-time inputs require a rebuild. This is an
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

Flywheel mounts artifact instance directories at the declared
container paths. What files exist inside is the container's
concern, not flywheel's.

### Convention-based output recording

Flywheel creates a fresh directory per declared output and
mounts it at the block's declared container path. The container
writes files there. After the container exits, flywheel records
whatever appeared as a new artifact instance.

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

### Execution flow

1. Resolve the block definition from the template.
2. Create a new block execution ID.
3. Resolve each input slot to a concrete artifact instance.
   The caller may specify explicit bindings for any input slot.
   For unbound copy inputs, the most recent instance for that
   slot is used (a provisional "latest wins" default). For git
   inputs, re-resolve the current committed state and record
   as a new git artifact instance.
4. Allocate fresh output directories with new artifact IDs.
5. Mount input instances and output directories into the
   container.
6. Run the container.
7. On success, finalize output artifact instances, record the
   execution, and persist the workspace.
8. On failure, record the execution with failure status and
   preserve state for inspection.

### Agent block execution

An agent block is a special execution variant where an AI agent
runs inside a Docker container with the ability to trigger nested
block executions. The agent reads source code, writes artifacts,
and iteratively invokes other blocks (e.g., evaluation) via an
MCP tool.

The lifecycle:

1. Create a fresh agent workspace directory. Seed it with the
   latest artifacts from prior steps so the agent can continue
   where the previous step left off.
2. Start a block bridge service (HTTP, background thread).
3. Run the optional ``pre_launch_hook`` callback. Projects use
   this for game-specific init, artifact creation, or writing
   files to the workspace before the container starts. The
   bridge is already running, so the hook can create artifacts.
4. Launch the agent container with the workspace, source mounts,
   auth volume, and the bridge endpoint as an environment variable.
   Stderr is drained in a background thread to prevent pipe
   deadlocks.
5. Stream JSON events from the agent's stdout and log them.
6. On completion (or timeout), collect output artifacts from the
   agent workspace by matching filenames to declared output names.
7. Stop the bridge service.

A total timeout (default 4 hours) kills the agent container if
exceeded, ensuring hung agents do not block indefinitely.

### Non-blocking agent handle

``launch_agent_block()`` returns an ``AgentHandle`` immediately,
allowing the caller to control the container while it runs:

- ``handle.kill()`` — terminate the container (e.g., from a
  bridge callback when an artifact triggers a policy decision).
- ``handle.wait()`` — block until exit, stop the bridge, collect
  output artifacts, and return ``AgentResult``.  Must be called
  exactly once, even after ``kill()``.
- ``handle.is_alive()`` — check if the container is still running.

The blocking ``run_agent_block()`` is a convenience wrapper that
calls ``launch_agent_block()`` then ``handle.wait()``.

### Agent session artifacts

The agent runner exports the Claude SDK session history as
``agent_session.jsonl`` to the workspace on exit (including on
SIGTERM from ``docker stop``). If ``output_names`` includes
``agent_session``, flywheel collects this as a regular copy
artifact.

To resume a session in a new container, pass the session artifact
as an input artifact and set ``RESUME_SESSION_FILE`` in
``extra_env`` pointing to the mounted file path. The agent runner
copies the session to the SDK's expected location and passes
``resume=session_id`` to the SDK client.

This enables cross-container session resume without persistent
volumes — the session round-trips through the artifact system.

### Agent pause and resume

Long-running agents can hit API rate limits or consume excessive
budget in a single session. The agent runner
(`batteries/claude/agent_runner.py`) supports pausing and resuming
to handle both cases.

**Pause triggers:**

- **Max turns** (`MAX_TURNS` env var). The SDK stops the query
  and emits a `ResultMessage` with `subtype == "error_max_turns"`.
  The runner detects this and pauses. This is opt-in — omit
  `MAX_TURNS` to let the agent run to natural completion.
- **Rate limit rejection**. The SDK yields a `RateLimitEvent`
  with `rate_limit_info.status == "rejected"`. The runner breaks
  out of the query loop and pauses. Rate limit exceptions raised
  outside the event stream are also caught.

**Pause behavior:**

On pause, the runner:
1. Writes `.agent_state.json` to the workspace with the session
   ID, status (`"paused"`), and reason (`"max_turns"` or
   `"rate_limit"`).
2. Emits an `agent_state` JSON event to stdout.
3. Polls every 5 seconds for a `.agent_resume` file in the
   workspace.

**Resume:**

The host (or user) writes a `.agent_resume` file to the shared
workspace mount. The file content is used as the resume prompt
(an empty file defaults to "Continue from where you left off.").
The runner reads and deletes the file, then calls `query()` with
`options.resume = session_id` to continue the conversation with
full history.

On startup, the runner also checks for:
- `RESUME_SESSION` env var — resume a specific session immediately.
- A saved `.agent_state.json` with `status == "paused"` — resume
  the interrupted session automatically.

**Assumptions:** The Docker container remains running between
pause and resume. The Claude Agent SDK stores session history
as local JSONL files at
``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`` inside
the container. These files are lost when the container dies.
The ``session_id`` alone is not enough to resume — the SDK
needs the local session file.

To support cross-container resume, mount a persistent volume
at ``/home/claude/.claude/projects/`` (in addition to the
existing auth volume at ``/home/claude/.claude/``), and ensure
the working directory matches across containers. This is not
yet implemented.

Alternatively, avoid session resume entirely: capture results
as artifacts and pass them into a fresh session's prompt. This
is the approach the artifact-based architecture naturally
supports.

### Block bridge

The block bridge is a generic HTTP service that lets containers
trigger nested block executions within the same workspace. It is
not specific to evaluation -- the invoked block and what it does
are defined by the project's template, not by flywheel.

The bridge supports two modes:

**Invoke mode** (default): launches a Docker container.

1. Validates the block name against the template and an optional
   allowed-blocks list.
2. Imports the provided artifact into the workspace via
   ``register_artifact()``.
3. Looks up the block definition from the template to determine
   the image, docker args, input slots, and output slots.
4. Runs the container with proper mounts.
5. Records the output artifacts and block execution in the
   workspace with full provenance.
6. Returns the results (including any scores) to the caller.

An invocation budget (``max_invocations``) limits how many
blocks the agent can trigger per step.

**Record mode**: creates artifacts without launching a container.
Used for provenance tracking of actions that already happened
(e.g., game steps executed via a REST API). The request includes
structured output data as JSON; the bridge writes it to artifact
directories and records a ``BlockExecution`` with input/output
bindings. Record-mode blocks use the ``__record__`` sentinel as
their image in the template. Input artifact IDs are validated
for both existence and name match against the declared slot.

**Record callback**: ``BlockBridgeService`` accepts an optional
``on_record`` callback, fired after each successful record-mode
invocation with ``(block_name, outputs)``. The callback runs in
the bridge's HTTP handler thread. This enables the host to react
in real-time to artifacts created by the agent — for example,
killing the agent container via ``AgentHandle.kill()`` when a
recorded game step indicates a prediction mismatch.

### Project-provided MCP servers

Projects can provide custom MCP servers by mounting a directory
into the agent container at ``/workspace/.mcp_servers/``. The
agent runner scans this directory on startup for files matching
``*_mcp_server.py`` and registers each one by name (derived by
stripping the ``_mcp_server.py`` suffix).

An optional sidecar manifest (``*_mcp_server.json``) can list
tool names to pre-register in ``allowed_tools``. If absent, tools
are discovered via MCP handshake.

All container environment variables are passed to mounted MCP
server subprocesses. This is safe because the host controls the
container's environment, and there are no secrets leaking from
inside Docker.

## Workspaces

A workspace is a directory inside a project's foundry folder.
It is created from a template and accumulates artifact instances
and execution records over its lifetime.

### Directory layout

```
project_root/
├── flywheel.yaml             ← project config (foundry_dir)
├── crates/                   ← project source code
└── foundry/                  ← flywheel's folder
    ├── templates/            ← workspace templates
    │   └── train_eval.yaml
    └── workspaces/
        └── my_workspace/
            ├── workspace.yaml   ← metadata, artifacts, executions
            └── artifacts/       ← copy artifact instance directories
                ├── checkpoint@a3f7b2/
                ├── checkpoint@d1e8c4/
                └── score@b7a209/
```

### Templates

A template defines the capabilities of a workspace: what
artifacts can exist, what blocks can run, and how containers are
configured. Templates live in `foundry/templates/`.

A template declares:
- **Artifact declarations**, which specify names and storage
  kinds (copy or git), along with repo and path for git
  artifacts.
- **Block definitions**, which specify names, container images,
  Docker configuration, and input/output artifact mappings
  with container paths.

Block inputs and outputs must reference declared artifact names.
This is validated at template parse time.

### Workspace creation

`flywheel create workspace --name NAME --template TEMPLATE` runs
from the project root. It reads `flywheel.yaml` to find the
foundry directory, loads the template, resolves git artifacts
(refusing if dirty) as baseline instances, and creates the
workspace directory with a `workspace.yaml` metadata file.

### Workspace persistence

The workspace.yaml file contains workspace metadata (name,
template, creation time), all artifact instances keyed by ID,
and all block execution records keyed by ID. Together these
form the complete provenance record for the workspace.

## Future work

### Default binding policy

When no explicit binding is provided for a copy input slot,
the current implementation uses the most recent instance for
that slot ("latest wins"). This is a provisional default that
works for simple sequential workflows.

A more sophisticated policy might need per-subclass defaults
(the best checkpoint for dueling vs defense), or selection
based on an evaluation metric rather than recency. A formal
`current_bindings` map on the workspace could support these
patterns, but the right design is not yet clear.

### Schema versioning

The workspace.yaml format will evolve as new features are added.
A `schema_version` field should be introduced to support loading
older workspaces and migrating them forward.

### Interrupted execution handling

Block executions can be interrupted (e.g., by Ctrl+C). The
execution is recorded with "interrupted" status, and orphaned
output directories are cleaned up. Partial outputs from
interrupted executions are not preserved as artifact instances.

### Commit-pinned git mounts

Git artifact instances record a commit SHA, but the current
mount implementation uses the live working tree. A future
improvement could use ``git archive`` or ``git worktree`` to
extract the exact committed state, making explicit bindings
to historical git artifacts truly reproducible.
