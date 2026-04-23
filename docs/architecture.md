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

There are three storage kinds:

- **Copy artifacts** are directories stored directly in the
  workspace (see "Artifact instances are directory-shaped").
  Block executions produce them — checkpoints, scores, and logs
  are typical examples.  Each instance is immutable.
- **Git artifacts** are references to version-controlled code,
  recorded as repo, commit SHA, and path. They are used to
  inject source code, configurations, or prompts into a
  workspace.
- **Incremental artifacts** are append-only sequences of
  immutable JSON entries, stored as a directory containing one
  ``entries.jsonl`` file (one opaque JSON value per line).
  Unlike copy artifacts, an incremental artifact has *one*
  growing instance per name per workspace; new entries are
  appended to that instance rather than producing a new
  instance per write.  ``game_history`` is the canonical
  example: every successful ``take_action`` block appends one
  entry recording the resulting frame.  Existing entries are
  never rewritten.

From a block's perspective copy and git inputs are read-only
directories.  An incremental input is a read-only snapshot of
the file as of mount time (see "Per-mount input staging" below).
Blocks emit copy and incremental output through the per-execution
output directory; no dedicated emit-artifact API exists.

### Artifact-only data channel

The data produced by a block execution flows out of the block
exclusively via its output artifacts.  Tool calls that trigger
nested block executions return only acknowledgment — ``"OK"`` on
success or ``"ERROR: <reason>"`` on failure.  Agents that need
the resulting data read it from the corresponding artifact (in
particular the incremental artifact for sequence-shaped data),
not from the spliced ``tool_result``.  This invariant is what
keeps the artifact store the single source of truth for what a
run produced.

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
``flywheel import artifact``. The source is always a directory
(see "Artifact instances are directory-shaped"); flywheel
stages its contents into a workspace-owned directory, runs the
registered validator (if any) against the staging directory,
and on success atomically renames it into the canonical
artifact location.  Imported artifacts have ``produced_by=None``
(like baseline git snapshots) but carry a ``source`` field
describing their origin.  Once registered, they bind identically
to block-produced artifacts.

### Undeclared artifacts are rejected

Every artifact name a workspace tracks is declared in the
template under ``artifacts:``.  Both ``Workspace.register_artifact``
(``flywheel import artifact``) and post-block output collection
(``_collect_outputs`` / ``flywheel.execution.run_block``) reject
names that are not in the workspace's declarations.

The point is not that undeclared artifacts are never useful —
some projects could plausibly want a "scratch" channel — but
that admitting them would weaken every other guarantee in this
section.  Without a declaration there is no agreed kind, no
declared validator, no documented downstream consumer; the
ledger silently grows shapes nobody promised to maintain.  When
that need becomes pressing we will revisit it explicitly rather
than letting it leak in through the import path.

### Artifact instances are directory-shaped

Every artifact instance is a directory.  Block executions write
into per-slot output directories at ``/output/<slot>/`` and
``flywheel import artifact`` requires a directory source.
Importing a single file is unsupported on purpose — the caller
wraps the file in a directory first, which makes the artifact's
structure explicit at the call site rather than implicit in
flywheel's behavior.  Mirroring the block-output shape this way
is what lets every consumer (validators, mounts, downstream
blocks) see one shape regardless of how the artifact got
there.

### Artifact validation

Projects can declare validation logic for any artifact name by
registering a callable on
:class:`flywheel.artifact_validator.ArtifactValidatorRegistry`
and pointing ``flywheel.yaml``'s ``artifact_validators:`` key at
a zero-arg factory that returns one.  See
:mod:`flywheel.artifact_validator` for the validator signature
(``(name, declaration, staged_path) -> None``; raise
:class:`ArtifactValidationError` to reject) and
:doc:`cli/import-artifact` for the CLI side.

The same registry is consulted at every artifact-finalization
site so a project gets one rule for "what counts as a valid
``checkpoint``" regardless of whether the artifact arrived
through ``flywheel import artifact`` or rolled out of a block
execution.  Names with no registered validator are accepted
unconditionally; flywheel never invents a default.

The validator's contract is uniform: it is given a directory
containing the bytes that would become the artifact instance,
and decides yes or no.  How those bytes got there — a
container's per-slot output tempdir, a staged copy of an
operator's import — is flywheel's concern, not the validator's.
Validators are **stateless and candidate-only**: they see the
staged candidate and the artifact declaration, never the
workspace's prior state for the same name.  This keeps the
contract identical across imports, copy outputs, and
incremental appends, and leaves room for a future tag-based
collection model in which each contribution is naturally a
separate instance (see "Future work").

**The validator owns all content policy.**  Flywheel does not
look inside the staged directory at all — it does not check
for required files, forbid unexpected files or
sub-directories, enforce a particular layout, or apply any
kind of schema.  Whether an extra ``logs/`` sub-directory or
an unrecognized ``notes.txt`` should be allowed is a
project-and-artifact-specific question, not a flywheel one,
so the validator must answer it.  If a project wants strict
"exactly these files and nothing else" semantics, the
validator must walk the tree and reject anything unexpected;
if it wants permissive semantics, it just doesn't check.
If no validator is registered for an artifact name, the
candidate is accepted as-is — flywheel does not perform any
content checks of its own.

**Container assembles its own outputs.**  Flywheel does not
provide a project-supplied "post-run hook" that runs in the
container or on the host to transform raw output into the
artifact's final shape.  Whatever shape the block wants the
artifact to have, its container puts those bytes at
``/output/<slot>/`` before exit — using whatever scripting,
enrichment, or post-processing the entrypoint wants.  Flywheel
reads what's there after exit and does not transform it.  This
keeps the substrate's surface area small (no hook registry, no
in-template transform field) and pushes the "how do I assemble
this artifact" question to the place that actually knows the
answer: the container.

**Failure semantics for block-produced outputs.**  When a block
emits multiple output slots and one of them fails its
validator, flywheel commits the slots that *passed* and rejects
only the slot that failed.  The execution is still recorded as
``failed`` with ``failure_phase=output_validate`` and an
``error`` field that names every rejected slot, so an operator
can tell at a glance which slots survived and which need
fixing.  The rationale is that a long-running block (training
run, agent session) often produces several useful artifacts
plus one invariant we wanted to catch — discarding everything
because the invariant fired would punish the block's other
work.

The opposite policy — reject the entire execution if any slot
fails — is equally defensible for projects whose output slots
are tightly coupled (e.g., a checkpoint and its scoring
manifest must agree or neither is useful).  Commit-passing-slots
is the single global default; making this a project-level
(or per-block) choice is left to "Future work".

For ``flywheel import artifact`` the question does not arise —
there is exactly one slot per call, so a rejection always
aborts the import without committing anything.

### Immutability

Artifact instances are immutable once created. They are never
overwritten, updated, or deleted. A new block execution produces
new instances; it does not modify existing ones. This ensures
the execution history is trustworthy — an artifact referenced
by an execution record will never silently change.

### Rejected-output preservation and amendment lineage

Immutability rules out in-place fixes, so flywheel pairs
*preserve-on-reject* (don't lose the bytes) with
*amendment-as-versioning* (the operator registers a corrective
successor instance instead of mutating the predecessor).

**Quarantine of rejected outputs.**  When a block-output
validator rejects a slot (see "Artifact validation" above), the
rejected bytes are copied to
``<workspace>/quarantine/<execution-id>/<slot-name>/`` and a
``RejectedOutput(reason, quarantine_path)`` record is attached
to the failed ``BlockExecution.rejected_outputs``.  The
execution is still recorded as ``failed`` with
``failure_phase=output_validate``; quarantine is purely
operator-visible state.

Quarantine is **best-effort**: if the I/O fails (permission
error, missing source, pre-existing destination), the failure
is logged and ``quarantine_path`` is recorded as ``None``.  The
validation failure itself is the primary signal and is
recorded either way.  ``flywheel.quarantine`` owns the
``<workspace>/quarantine/<exec>/<slot>/`` convention so every
producer evolves it in one place.

Imports do not quarantine — the operator's source path is
still on disk and is the authoritative copy.

**Amendment lineage.**  An ``ArtifactInstance`` may carry an
optional ``SupersedesRef`` recording which predecessor it
supersedes:

* ``SupersedesRef(artifact_id=...)`` — supersedes an accepted
  predecessor (an existing ``ArtifactInstance`` id).
* ``SupersedesRef(rejection=RejectionRef(execution_id, slot))``
  — supersedes a quarantined slot of a failed execution.

The two flavours match the two operator workflows (``flywheel
amend artifact`` and ``flywheel fix execution`` respectively;
see [cli/amend-artifact.md](cli/amend-artifact.md) and
[cli/fix-execution.md](cli/fix-execution.md)).  The substrate
treats them uniformly — the distinction matters for CLI
ergonomics, not for storage.

The pointer is **backward-only** and the predecessor is
identified by a stable ledger handle (artifact id or
``(execution_id, slot)`` pair), never by a filesystem path.
Multiple successors may point at the same predecessor —
forking is allowed by the schema, even if the CLI doesn't
expose it yet.  The pointer is provenance / intent, not a
resolution rule: consumers still resolve by ``latest
created_at``, exactly as for plain registrations.  An optional
``supersedes_reason`` string captures *why* the successor was
registered so the audit trail is reconstructible without
out-of-band notes.

The same artifact validator that gates ``flywheel import
artifact`` runs against the corrected bytes; successors get no
validation discount.  A failed amendment registers nothing.
Predecessor existence is checked before any bytes are staged
so a bad reference fails cheaply and leaves no debris.

Today only ``copy`` artifacts can be re-registered;
``register_artifact`` rejects ``git`` and ``incremental``
predecessors.

Quarantine directories accumulate forever today; flywheel
never deletes them.  An automatic deletion policy — by age,
by linked-execution status, by operator command, or some
combination — is deferred until we have enough quarantine
volume in the wild to know which policy actually fits real
usage.  Schema fields for "preservation succeeded?" or
"preservation error" are likewise deferred until a real need
appears — the presence or absence of ``quarantine_path`` is
the only state today.

Implementation pointers:
[`flywheel/artifact.py`](../flywheel/artifact.py)
(``SupersedesRef``, ``RejectionRef``, ``RejectedOutput``);
[`flywheel/quarantine.py`](../flywheel/quarantine.py)
(``quarantine_slot``);
[`flywheel/workspace.py`](../flywheel/workspace.py)
(``Workspace.register_artifact``'s ``supersedes`` /
``supersedes_reason`` parameters and ``_check_supersedes``).

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

A **run** is a higher-level concept that likely sits between
workspace and block execution: a durable grouping of related
executions, often but not necessarily driven by a pattern.
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

### Per-execution output directories

Each block execution gets a fresh, ephemeral output directory
created via ``tempfile.mkdtemp(prefix="flywheel-exec-")``.
Inside that directory flywheel pre-creates one
``output/<output_name>/`` per declared output and either bind-
mounts it into the container at the block's declared
``container_path`` (for container blocks) or exposes it via
``ctx.output_dir(name)`` to in-process block bodies.  The block
writes files into those directories; no API is provided to
emit artifacts directly.

After the container or in-process body exits, flywheel walks
each declared output:

- For ``copy`` outputs: the directory's contents become a new
  copy artifact instance under the workspace.
- For ``incremental`` outputs: each line of
  ``output/<name>/entries.jsonl`` (and any other ``*.jsonl``
  files written there) is appended to the canonical incremental
  instance under a file lock, preserving registration order.

The ephemeral output directory is removed after registration
on success.  On failure it is retained for debugging.

This avoids requiring containers to write a manifest or be
flywheel-aware: the template's output declarations are the
contract.  It also keeps the canonical artifact store untouched
by the block body — only flywheel writes to it, and only after
observing the final on-disk state of the output directory.

### Per-mount input staging

Canonical artifact directories under
``<workspace>/artifacts/<id>/`` are never directly mounted into
a block.  For every input slot, flywheel copies the canonical
contents into a fresh per-mount staging directory created with
``tempfile.mkdtemp(prefix="flywheel-mount-")`` and mounts the
staging directory instead.  Always a full copy; never a
hardlink, never a shared inode.  The staging directory is
cleaned up after the container or in-process body exits.

This buys two things.  First, mutation isolation: a misbehaving
block that writes into a "read-only" input mount can corrupt
its private copy at worst, never the canonical instance.
Second, snapshot semantics for incremental artifacts: each
relaunch of the agent-handoff loop gets a fresh staging copy
that reflects all entries appended so far, including those
written by handoff blocks during the previous cycle.

The cost is sequential file I/O proportional to total input
size on every launch.  Today's largest inputs in cyberarc
(``game_history``) reach a few MB after a long run; we accept
the cost for the safety it buys.

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
runs inside a Docker container, reads source code, writes
artifacts, and may trigger nested block executions on the host.
The container itself runs no host-facing services; nested blocks
are produced by stopping the agent, running the block in the
host's Python process, splicing the result into the agent's
session history, and relaunching the agent against that spliced
session (see "Nested block executions from agents" below).

The single-launch lifecycle inside ``launch_agent_block``:

1. Create or reuse the agent workspace directory.  The
   workspace is **scratch only** — flywheel never seeds it
   with artifact data.  Fresh launches start with an empty
   workspace; relaunches reuse whatever the agent itself wrote
   to it during the previous cycle (notes, intermediate
   scripts).  The SDK session lives in ``/state/``, not in the
   workspace; artifact data reaches the agent exclusively
   through input mounts (see "Per-mount input staging").
2. Launch the agent container with the workspace, source mounts,
   the auth volume, and the configured environment variables
   (model, ``MAX_TURNS``, ``HANDOFF_TOOLS``, ``MCP_SERVERS``,
   etc.).  Stderr is drained in a background thread to prevent
   pipe deadlocks.
3. Stream JSON events from the agent's stdout and log them.
4. On exit (clean, timeout, cooperative stop, or handoff),
   record the agent itself as a ``BlockExecution`` under the
   caller-supplied ``block_name`` (typically the role name such
   as ``play``), register each declared output artifact from
   the block's ``/output/<name>/`` directory, and return an
   ``AgentResult``.

A total timeout (default 4 hours) kills the container if
exceeded.

For agents that need to invoke nested blocks, callers wrap
``launch_agent_block`` in :func:`flywheel.agent_handoff.run_agent_with_handoffs`.
The handoff loop owns the launch → exit → splice → relaunch
cycle; each cycle is its own ``BlockExecution`` record chained via
``predecessor_id`` so an operator inspecting the workspace sees
the cycle structure explicitly.

### Non-blocking agent handle

``launch_agent_block()`` returns an ``AgentHandle`` immediately,
allowing the caller to control the container while it runs:

- ``handle.stop(reason)`` — trigger cooperative shutdown by
  writing ``/workspace/.stop`` (the substrate sentinel the
  container runtime polls); a forced TERM/KILL follows on
  timeout.  The optional ``reason`` is recorded on the resulting
  execution record.
- ``handle.wait()`` — block until exit, join the stdout/stderr
  drain threads, collect output artifacts, and return
  ``AgentResult``.  Must be called exactly once, even after
  ``stop()``.
- ``handle.is_alive()`` — check if the container process is
  still running.

The blocking ``run_agent_block()`` is a convenience wrapper that
calls ``launch_agent_block()`` then ``handle.wait()``.  Code
that needs nested-block support uses
``run_agent_with_handoffs()`` instead.

### Agent session storage

The agent runner stores the Claude SDK session history under
``/state/`` (see ``substrate-contract.md`` for the ``state:
true`` mount contract).  Flywheel populates ``/state/`` from the
most recent prior execution's captured state on launch and
captures it back on exit, so a subsequent launch resumes from
the same session without round-tripping through the artifact
graph.  There is no separate ``agent_session`` artifact.

### Agent pause and resume

Long-running agents need to handle three independent classes of
interruption: API rate limits, exhausted turn budgets, and
nested block invocations.  All three converge on the same
mechanism — the agent's session is checkpointed to a session
JSONL artifact and the host relaunches a new container against
that artifact — but they differ in *who* drives the resume and
*what* (if anything) is appended to the session before
relaunch.

**In-container pauses (rate limit, max turns).**  The agent
runner (``batteries/claude/agent_runner.py``) detects these
two conditions from the SDK event stream:

- **Rate limit rejection.**  The SDK yields a ``RateLimitEvent``
  with ``rate_limit_info.status == "rejected"``.  The runner
  exponentially backs off (60s, 120s, 300s, 300s, 300s).
- **Max turns.**  The SDK emits a ``ResultMessage`` with
  ``subtype == "error_max_turns"`` (opt-in via the
  ``MAX_TURNS`` env var).

When backoff is exhausted (or for max turns directly), the
runner writes ``.agent_state.json`` with ``status="paused"`` and
the reason, emits an ``agent_state`` event to stdout, and polls
``.agent_resume`` in the workspace.  Writing any content (or an
empty file) to ``.agent_resume`` lets the runner consume it as
the next prompt and call ``query()`` with ``options.resume =
session_id`` to continue in the same container.  This in-place
resume only works while the container is still alive — the SDK
stores session JSONLs at
``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`` inside
the container, and those files die with the container.

**Cross-container resume.**  When the container has to die —
total-timeout, a cooperative stop, or a nested-block handoff —
the runner captures its SDK session to ``/state/`` before
exiting.  On the next launch flywheel populates ``/state/`` from
that captured snapshot and the runner resumes the same session.
See ``substrate-contract.md`` for the ``/state/`` contract.

**Handoff resume (nested blocks).**  The host-side handoff loop
(:func:`flywheel.agent_handoff.run_agent_with_handoffs`) drives
its own pause/resume cycle on top of cross-container resume.
When the agent calls a tool listed in ``HANDOFF_TOOLS``, the
runner's ``PreToolUse`` hook denies the call, captures the
intended ``tool_use`` blocks into ``pending_tool_calls.json``,
sets ``.agent_state.json`` to ``status="tool_handoff"``, and
exits cleanly so the host can run the blocks out-of-band and
splice the real ``tool_result`` into the captured session in
``/state/`` before relaunching.  See "Nested block executions
from agents" for the full mechanism.

### Recording nested block executions

Nested block executions invoked by an agent — ``predict``,
``game_step``, ``exploration_request``, ``brainstorm_request``,
and the like — are recorded in-process by
:class:`flywheel.local_block.LocalBlockRecorder`.  No host-side
HTTP service runs.  The recorder owns one workspace and exposes a
single ``begin`` context manager:

1. ``begin(block, params=..., caller=...)`` resolves the block's
   declared inputs to the latest registered instance of each
   slot, copies each into a per-mount staging tempdir (the same
   isolation rule that container blocks observe), and exposes
   the staging paths to the body.  Missing required inputs raise
   ``LocalBlockError`` *before* an execution is opened.  Incremental
   inputs are snapshotted into the staging dir at this moment,
   so the body sees a frozen view of the sequence regardless of
   appends that happen during execution.
2. The body runs.  It receives a ``LocalExecutionContext``
   carrying the allocated ``execution_id``, the resolved input
   bindings, host paths to each staged input, a per-execution
   scratch directory, and ``ctx.output_dir(name)`` returning a
   pre-created ``output/<name>/`` under the execution's
   ephemeral output tempdir.  The body writes files there;
   no ``set_output`` callback exists.
3. On clean exit the recorder walks each declared output:
   ``copy`` outputs are registered as a new instance from the
   directory contents; ``incremental`` outputs append the lines
   in ``output/<name>/entries.jsonl`` (and any other ``*.jsonl``
   files in that directory) to the canonical incremental
   instance under a file lock.  It then writes a single
   ``BlockExecution`` record with ``status="succeeded"`` and runs
   the block's configured post-check.  On body exceptions or
   output-registration failures it rolls back any partial
   artifacts and writes a single ``"failed"`` execution record carrying the
   error message.  Per-mount staging dirs and the per-execution
   output tempdir are cleaned up at the end (retained on
   failure for debugging).  There is no ``"running"`` execution record;
   nothing crosses an asynchronous boundary, so the orphaned-
   ``running`` failure mode is structurally impossible.

Post-execution callbacks (see :mod:`flywheel.post_check`) fire
synchronously after the execution record is durable.  When a callback returns
a ``HaltDirective``, the recorder appends it to an internal halt
queue.  The host-side handoff loop drains that queue between
cycles via :meth:`LocalBlockRecorder.drain_halts` and refuses to
relaunch the agent if a relevant directive is present, which is
how a project signals "stop this run, the work it produced
crossed a policy threshold."

Host-side block runners (``PredictActionRunner``,
``BrainstormRequestRunner``, ``ExplorationRequestRunner``) all
use the recorder this way: they are plain ``BlockRunner``
callables registered with the handoff loop, invoked when the
agent's ``PreToolUse`` hook intercepts the corresponding tool,
and they wrap their work in ``recorder.begin(...)`` so the
resulting execution record, artifacts, and post-check are
recorded under the same workspace lock as every other execution.
The same pattern applies outside the agent loop:
``cyberarc/tools/play_server.py`` uses a ``LocalBlockRecorder``
to record human play sessions.  ``take_action`` dispatch lives
in a different path — the pattern runner's ``on_tool`` trigger
routes it to the ``ExecuteAction`` container block rather than
a host-side runner.

All artifact-recording flows ride either
``LocalBlockRecorder`` (``runner: lifecycle`` blocks) or the
canonical ``flywheel.execution.run_block`` path for
``runner: container`` one-shot blocks.  There is no HTTP lifecycle API and no
in-container recording proxy — agents trigger recordings
exclusively through the host-side handoff loop (see "Nested
block executions from agents" below).

### Project-provided MCP servers

Projects can provide custom MCP servers by mounting a directory
into the agent container at ``/flywheel/mcp_servers/``. The
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
  kinds (``copy``, ``git``, or ``incremental``), along with
  repo and path for git artifacts.
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

**Thread safety:** ``Workspace`` uses a ``threading.Lock`` to
serialize mutations (``add_artifact``, ``add_execution``,
``add_event``, ``save``). ``save()`` writes to a temporary file
and atomically renames it to ``workspace.yaml`` via
``os.replace()``, preventing torn writes from concurrent saves
or crashes.

## Lifecycle tracking

### Block execution metadata

``BlockExecution`` includes optional fields for lifecycle tracking:

- ``stop_reason``: Why the execution ended, if it was stopped
  externally (e.g., ``"exploration_request"``,
  ``"prediction_mismatch"``, ``"timeout"``). None for normal
  completion.
- ``predecessor_id``: Execution ID that this block resumes from,
  enabling resume chains across container restarts.

### Agent execution recording

``AgentHandle.wait()`` records the agent container itself as a
``BlockExecution`` under the caller-supplied ``block_name``
(typically the role name from the pattern, e.g. ``play``). This
fills a previous gap where the workspace tracked nested block
executions (game steps, evaluations) but not the agent run that
triggered them. The ``stop_reason`` field captures why the agent
was stopped, and ``predecessor_id`` links consecutive runs of the
same agent.

### Lifecycle events

``LifecycleEvent`` is a lightweight entity for operational events
that are not block executions: agent stops, group completions,
mode transitions. Each event has:

- ``id``: Unique identifier (``"evt_hexstring"``).
- ``kind``: Event type (e.g., ``"agent_stopped"``,
  ``"group_completed"``).
- ``timestamp``: When the event occurred.
- ``execution_id``: Related execution, if any.
- ``detail``: Free-form metadata dict.

Events are stored in the workspace alongside artifacts and
executions. They are serialized to workspace.yaml under an
``events`` key (omitted when empty for backward compatibility).

## Parallel agent groups

``AgentGroup`` (in ``flywheel.agent_group``) provides a reusable
primitive for launching multiple agents in parallel and collecting
their results:

1. Each member's bind mount is auto-named under
   ``agent_workspaces/<short-uuid>/`` so two parallel launches
   in the same workspace can never clobber each other's files
   (see "Agent workspace mounts" below).
2. All agents are launched simultaneously via
   ``launch_agent_block()``.
3. Results are collected sequentially (serializes workspace writes
   to avoid races on ``workspace.yaml``).
4. After each wait, the group scans for output files in the
   directory the launcher reports back through
   ``AgentResult.agent_workspace_dir`` and registers them as
   artifacts.  A ``fallback_fn`` can generate default output for
   agents that produce nothing.
5. A ``group_completed`` lifecycle event is recorded on completion.

The ``AgentBlockConfig`` dataclass groups the 26+ parameters of
``launch_agent_block()`` into an inspectable object, making it
practical to define a base configuration and merge per-member
overrides.

### Agent workspace mounts

``prepare_agent_workspace()`` (in ``flywheel.agent``) creates the
host-side directory that gets bind-mounted into the agent
container at ``/workspace``.  The directory is **scratch only**:
flywheel never copies artifact data into it.  Whatever the agent
itself writes — notes, intermediate scripts, ``.agent_state.json``,
the SDK's session state for in-place resume — lives there for the
duration of the run.  Artifact data reaches the agent through
input mounts (per-mount staged copies of canonical instances),
never through the workspace.  The function returns an
:class:`AgentMount` that records:

- ``relative_dir``: the workspace-relative path of the mount
  (e.g., ``"agent_workspaces/abc12345"``).
- ``host_path``: the absolute host directory.
- ``container_path``: where the dir is mounted inside the
  container (always ``/workspace`` today).
- ``mode``: ``"rw"`` for the primary agent workspace.

When the caller does **not** pass ``agent_workspace_dir``, the
function mints a unique name under ``agent_workspaces/`` using a
short UUID; collisions trigger a redraw.  This is the default for
``PatternRunner``, ``BlockGroup``, and the cyberarc explorer /
brainstorm helpers — making cross-agent workspace contamination
structurally impossible during parallel runs.

When the caller *does* pass an explicit name and the target
directory already exists with content, ``prepare_agent_workspace``
raises ``FileExistsError`` rather than silently rmtree-ing.  The
old "delete and re-create" behavior was a footgun that could blow
away another agent's working state; explicit names are still
honored when the target is empty (e.g., a test pre-creates the
dir) or absent.

``BlockExecution`` records carry an ``agent_workspace_dir`` field
populated from the ``AgentMount`` so an operator inspecting an
``agent_workspaces/<id>/`` directory can find the execution that
produced it via ``workspace.yaml``.

## Patterns (declarative agent topology)

``Pattern`` (in ``flywheel.pattern``) and ``PatternRunner``
(in ``flywheel.pattern_runner``) are flywheel's orchestration
primitive for multi-agent workflows.  A pattern *declares* its
topology and timing in YAML and the runner translates that
declaration into agent launches:

```yaml
# patterns/play-brainstorm.yaml
description: One play agent + a brainstorm cohort every 20 actions.
roles:
  play:
    prompt: workforce/prompts/arc_predict_play.md
    model: claude-sonnet-4-6
    cardinality: 1
    trigger: { kind: continuous }
    inputs: [predictor, mechanics_summary]
    outputs: [game_log]
  brainstorm:
    prompt: workforce/prompts/arc_brainstorm.md
    cardinality: 6
    trigger: { kind: every_n_executions, of_block: take_action, n: 20 }
    inputs: [game_history]
    outputs: [brainstorm_result]
    extra_env: { BRAINSTORM_FOCUS: "general" }
```

### Role fields

- ``inputs`` — artifact names; the runner binds the latest
  registered instance of each before launch.  Incremental
  artifacts (e.g., ``game_history``) are bound the same way and
  re-staged per launch, so each cycle of a handoff loop sees
  every entry appended through the previous cycle.
- ``outputs`` — artifact names the role registers.  Declared on
  the role for documentation; the substrate treats block-declared
  outputs as the single source of truth.
- ``extra_env`` — per-role env vars merged on top of the
  project's ``extra_env``.  Use sparingly; most env vars belong
  in project hooks.
- ``model`` / ``max_turns`` / ``total_timeout`` /
  ``mcp_servers`` / ``allowed_tools`` — per-role overrides of
  the launcher defaults.

### Trigger vocabulary

- ``continuous`` — fire at run start; lifetime equals the run.
- ``every_n_executions`` — fire every N successful, non-synthetic
  executions of a referenced block.
- ``on_request`` (parsed, runner not yet implemented) — fire when
  an agent invokes a named tool.
- ``on_event`` (parsed, runner not yet implemented) — fire when
  a workspace event of the named kind is recorded.

The vocabulary is intentionally narrow.  Add a kind only when an
existing pattern can't be expressed without it; every kind grows
the runner's responsibility.

### Project hooks

Patterns shrink the project-side surface to the things only the
project can know — starting external resources and parsing
project-specific CLI args.  ``ProjectHooks`` (in
``flywheel.project_hooks``) declares:

- ``init(workspace, template, project_root, args) -> dict``:
  one-time setup; returns ``AgentBlockConfig`` overrides.
- ``teardown()`` (optional): release resources after the run.

Wired up in ``flywheel.yaml``:

```yaml
foundry_dir: foundry
project_hooks: myproject.project:ProjectHooks
```

### CLI

```bash
flywheel run pattern <pattern-name> \
  --workspace PATH --template TEMPLATE \
  [--project-hooks MODULE:CLASS] [--model MODEL] \
  [--max-runtime SECONDS] [-- project-specific args...]
```

Patterns are discovered as ``<project_root>/patterns/<name>.yaml``.
``run pattern`` is the only multi-agent orchestration verb.

## Nested block executions from agents

Some agent tool calls trigger nested block executions (``predict``,
``game_step``, ``brainstorm_request``); others do not (workspace
state queries, schema lookups).  For the calls that *are* block
executions, flywheel uses a "full stop" model: each nested
invocation is a real operational boundary, not a mid-session RPC.

**Why not a bridge.**  An earlier design kept the agent container
alive while a host-side HTTP bridge ran the nested block in
parallel and returned the result over an open MCP tool call.
That worked, but it leaked state across the boundary: the agent
process kept its in-memory context, the workspace could shift
underneath between calls, a crash mid-body left an orphaned
``running`` execution, and the agent and its MCP server shared fate so
a hang in one wedged the other.  We accepted some container-
restart latency to make these failure modes structurally
impossible.

**Mechanism.**  The host-side handoff loop
(:func:`flywheel.agent_handoff.run_agent_with_handoffs`) drives
each cycle as follows:

1. **Launch.**  ``launch_agent_block`` starts the agent container
   with ``HANDOFF_TOOLS`` set to the MCP tool names that should
   produce nested block executions (e.g.,
   ``mcp__arc__take_action``).  On a relaunch this also sets
   ``RESUME_SESSION_FILE`` to the spliced session JSONL produced
   by the previous cycle.
2. **Intercept.**  When the agent emits a ``tool_use`` for one of
   the listed tools, the agent runner's Claude Agent SDK
   ``PreToolUse`` hook denies it with a marker reason
   (``"handoff_to_flywheel"``).  This causes the SDK to write a
   synthetic ``deny`` ``tool_result`` into the session JSONL,
   which is exactly what the splice step needs as a target.  The
   hook also records every intercepted call (one per ``tool_use``
   in the same assistant turn) into ``pending_tool_calls.json``
   in the workspace, sets ``.agent_state.json`` to
   ``status="tool_handoff"``, and signals the runner to exit
   cleanly.
3. **Exit.**  The runner exports the SDK session JSONL to
   ``agent_session.jsonl`` and exits 0.  The container goes away;
   ``AgentHandle.wait`` records a ``BlockExecution`` for the
   agent itself with ``stop_reason="tool_handoff"``.
4. **Run blocks.**  For each pending tool call, the handoff loop
   invokes the registered ``BlockRunner`` (one per handoff tool;
   composed via ``make_tool_router`` when more than one exists).
   The runner does its work through ``LocalBlockRecorder.begin``,
   so each call produces its own ``BlockExecution`` record,
   artifacts, and post-check.  Multiple parallel tool_uses in a
   single assistant turn are handled serially: N tool_uses → N
   independent executions.
5. **Splice.**  The loop rewrites the session JSONL in place,
   replacing each synthetic ``deny`` ``tool_result`` (located by
   tool_use_id and the ``"handoff_to_flywheel"`` marker) with the
   real result returned by the corresponding ``BlockRunner``.
   This is the step that lets the agent perceive the handoff as a
   normal tool-call/result cycle.
6. **Halt check.**  The loop drains
   :meth:`LocalBlockRecorder.drain_halts`.  If any post-check
   produced a ``HaltDirective`` for a block this loop cares about,
   the loop terminates without relaunching and returns the
   collected halts on its result.
7. **Relaunch.**  Otherwise the loop calls ``launch_agent_block``
   again with ``reuse_workspace=True`` and ``RESUME_SESSION_FILE``
   pointing at the spliced JSONL.  ``predecessor_id`` chains the
   new agent execution to the previous one.  Repeat until the
   agent exits without a handoff, ``max_iterations`` is hit, or
   a halt fires.

**Why this is right.**  Each block execution is the atomic unit
it has always been advertised as: state is on disk at every
boundary, no execution can be orphaned, the agent cannot observe a
workspace mid-mutation by another execution, and host-side
runners get all the host's tooling (debuggers, real exception
traces, the workspace's own lock).  The tradeoff is that we move
brittleness from the bridge surface to the state-save surface;
the latter is local, testable by killing the agent at every
interesting point, and fails loudly rather than silently.

## Runs

A **run** is a durable grouping of :class:`BlockExecution`
records inside one workspace.  Every pattern invocation opens a
run; ad-hoc executions (direct ``executor.launch`` calls, human
play via ``tools/play_server.py``) are ungrouped —
``BlockExecution.run_id`` is ``None``.

The data model:

- :class:`flywheel.artifact.RunRecord` — one per run; fields
  ``id``, ``kind`` (e.g. ``"pattern:play-brainstorm"``),
  ``started_at``, ``finished_at``, ``status``
  (``running``/``succeeded``/``failed``/``stopped``), and an
  optional ``config_snapshot``.
- :attr:`flywheel.workspace.Workspace.runs` — ``dict[str,
  RunRecord]`` persisted to ``workspace.yaml``.
- :attr:`flywheel.artifact.BlockExecution.run_id` — stamped on
  every executor-recorded execution.  ``None`` for ad-hoc.

The :class:`flywheel.pattern_runner.PatternRunner` opens the run
at :meth:`~flywheel.pattern_runner.PatternRunner.run` start and
closes it in a ``finally`` block (status = ``succeeded`` on
clean exit, ``failed`` on exception).  The id is threaded into
every :meth:`executor.launch` and every
:class:`flywheel.agent_handoff.HandoffContext` so nested
host-side runners (predict, brainstorm, exploration) inherit it
via :meth:`flywheel.local_block.LocalBlockRecorder.begin`.

### Why runs exist

Cadence triggers (``every_n_executions``) have to scope to the
current run or they count prior runs' executions.  Without runs,
re-running a pattern in an existing workspace fires nested
cohorts immediately — the counter sees all historical
executions.  With runs, the counter filters by
``ex.run_id == self._run_id`` and starts fresh on each
invocation.

The same scoping lets one workspace host many pattern runs
without cross-contamination while sharing workspace-level
artifacts (e.g., predictor files, game history) across them.

### Workspace run policy

``Workspace.runs`` can hold any number of runs.  Today there is
no enforced policy — any caller can open as many as it wants,
though :class:`PatternRunner` only opens one per
``.run()`` call.  A future ``single``/``multi``/``single_active``
policy is discussed under "Future work".

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

### Artifact scope and transfers between contexts

If runs become first-class, artifact scope will likely need to
distinguish between:

- **workspace-scoped** artifacts, shared across all runs in the
  workspace;
- **run-scoped** artifacts, belonging to one run's lineage.

Scope should be independent of storage kind: a copy, git, or
incremental artifact may be workspace-scoped or run-scoped
depending on its declaration.

It should also be possible to move information explicitly between
contexts inside one workspace.  The likely operation is not
aliasing one artifact instance into multiple contexts, but
creating a **new** artifact instance in the target context whose
provenance points at the source instance.  Examples include:

- copying an artifact from one run to another;
- promoting a run-scoped artifact into workspace scope;
- seeding a new run from workspace-scoped artifacts or from
  artifacts produced by an earlier run.

This keeps provenance honest: movement across contexts is explicit
and recorded rather than implicit shared mutable state.

### Enforced workspace run policy

Runs exist today (see the "Runs" section above) but flywheel
does not enforce any policy on how many may open in one
workspace or whether multiple runs may be active simultaneously.

A future workspace-level policy declaration may be useful:

- ``single``: one run total;
- ``multi``: multiple runs allowed (today's implicit default);
- ``single_active``: multiple historical runs are allowed but
  only one may remain open at a time.

This would let flywheel enforce operator intent rather than
relying on convention.  A research-paper workspace might choose
``single``; a game-playing workspace might choose ``multi``.

### Schema versioning

The workspace.yaml format will evolve as new features are added.
A `schema_version` field should be introduced to support loading
older workspaces and migrating them forward.

### Interrupted execution handling

Block executions can be interrupted (e.g., by Ctrl+C). The
execution is recorded with "interrupted" status, and orphaned
output directories are cleaned up. Partial outputs from
interrupted executions are not preserved as artifact instances.

### Project-selectable artifact commit policy

The artifact-validation rules above bake in a single global
choice for what to do when one of an execution's output slots
fails its validator: commit the slots that passed and drop only
the failing one.  Some projects will want the opposite — reject
the entire execution if any slot fails — typically when the
slots are tightly coupled and a partial commit would be a worse
outcome than no commit.  A future project-level (or per-block)
policy declaration would let each project pick.  Until then,
projects that need all-or-nothing semantics can layer a
validator that *deliberately* fails every paired slot together
— clumsy but workable.

### Incremental artifacts replaced by tagged copy instances

The ``incremental`` storage kind ("one growing instance per
name; appends never produce a new instance") is on the table
for retirement.  The motivating problems are that amendment
cannot naturally mean "register a corrected new instance"
when there is only one instance, that cross-entry invariants
pretend to be a substrate feature when they are really a
project concern, and that concurrent appends require a
workspace-level lock.

The candidate replacement is a **tag** mechanism layered over
plain ``copy`` instances:

- Each block execution that would have appended to an
  incremental instead produces an ordinary ``copy`` instance
  containing just that cycle's contribution.
- A project-declared **tag** (e.g. ``game_history``) names a
  collection of copy instances and is the unit consumers
  depend on.  A block input referencing a tag receives the
  matching set as a collective input, ordered by creation
  time.
- Amendment, quarantine, validation, and concurrency become
  uniform across all artifacts.  Cross-entry invariants stop
  pretending to be a substrate feature; they are embedded in
  entry data.

The known cost is per-mount staging proportional to
collection size rather than file size; long-running
collections (thousands of game histories) will require either
manifest-style mounts, lazy resolution, or explicit
compaction (fold N instances into one merged copy).  The
exact mount layout and the rules for declaring tags are
deliberately left open here — the point is to record the
direction, not commit to a shape.

Until we take this on, amendment of incremental artifacts is
out of scope and the ``supersedes`` schema is kept
kind-agnostic so the future migration is a substrate change
rather than a schema change.

### Commit-pinned git mounts

Git artifact instances record a commit SHA, but the current
mount implementation uses the live working tree. A future
improvement could use ``git archive`` or ``git worktree`` to
extract the exact committed state, making explicit bindings
to historical git artifacts truly reproducible.
