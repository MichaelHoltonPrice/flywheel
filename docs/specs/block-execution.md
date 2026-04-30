# Normative block-execution model

This spec defines the canonical model for executing a single block in a
flywheel workspace. It is the target shape of the substrate.

Concepts named in [Deferred to follow-on specs](#deferred-to-follow-on-specs)
are real substrate concerns that will land in their own normative passes;
they are excluded here so the model stays small enough to specify cleanly.

## Scope

In scope:

* Operator-triggered block execution via `flywheel run block`.
* Inputs: `copy` artifacts (full machinery) and `git` artifacts
  (lightweight, inputs-only — see [Git artifacts](#git-artifacts)).
* Outputs: `copy` artifacts only.
* Termination reasons as a first-class concept (see
  [Termination reasons](#termination-reasons)).
* Container runtimes:
  * one-shot container runner — one container per execution
    (the preferred default).
  * persistent container runner — one long-running container hosts
    many executions (kept only as a workaround for blocks whose
    in-container state cannot yet be serialized; see
    [Runtime variants](#runtime-variants)).
* Validation, quarantine, `SupersedesRef` lineage, and artifact
  sequence inputs/outputs (see
  [artifact-sequences.md](artifact-sequences.md)).

Not covered by this spec (each addressed by its own normative spec):

* State (`state_lineage_key`, the `/flywheel/state` mount, state
  snapshot records, state restore/capture, the `state_capture` failure
  and `state_validate` failure phases); see [state.md](state.md).
* Block invocation (a committed execution outcome triggering a child
  block execution; the `invoking_execution_id` field; declaration
  syntax associating termination reasons with child blocks); see
  [block-invocation.md](block-invocation.md).
* Tagging.
* Patterns, pattern runners, pattern handoff.
* Block groups (parallel scheduling).
* Anything agent-shaped: agents, MCP, tool calls, prompts,
  Claude session splicing, project-hooks-as-agent-launchers.

## Model

A block execution proceeds in three phases. The outer two are
canonical and shared across all runtime variants; the middle is the
only phase that varies by runtime.

### 1. Prepare

`prepare(workspace, block_name, input_bindings) -> ExecutionPlan`

Steps in order. Execution identity is established first so any
downstream failure has somewhere to record itself.

1. Generate a fresh `execution_id` via
   `Workspace.generate_execution_id()`. Stamp `started_at`.
2. Allocate a per-slot proposal directory under
   `<workspace>/proposals/<execution_id>/` for each declared output
   slot (across all termination reasons — at this point the substrate
   does not yet know which will fire). See
   [Proposal-then-forge](#proposal-then-forge).
3. Resolve each declared input slot:
   * For `copy` inputs: look up the named `ArtifactInstance` (or
     apply the implicit "latest copy" default when the binding is
     not given).
   * For `git` inputs: resolve to a concrete (repo, commit, path)
     triple at execution time. See [Git artifacts](#git-artifacts).
   * For sequence inputs: resolve the declared concrete scope,
     snapshot the ordered sequence entries, stage a manifest plus one
     directory per entry, and record the snapshot on
     `BlockExecution.input_sequence_bindings`. Empty sequences are
     valid inputs.
4. Return an `ExecutionPlan` carrying everything `Invoke` needs
   (the `execution_id`, resolved input mounts, output slot
   directories, runtime config from the block declaration).

A failure at step 2 or step 3 has both an `execution_id` and a
`started_at` already in hand and is recorded via `commit_failure(...)`
(see [Failure recording for prepare-time and invoke-time errors](#failure-recording-for-prepare-time-and-invoke-time-errors)).
A failure at step 1 — workspace cannot mint an execution ID at all —
is a workspace-level error, not a block-execution error, and surfaces
to the caller without a record.

### 2. Invoke

`runtime.invoke(plan) -> InvocationResult`

The only phase that varies by runtime. Returns:

* `exit_code: int | None`
* `elapsed_s: float`
* `termination_reason: str` — read from the runtime's termination
  channel when the block exited cleanly and announced one; otherwise
  set to one of the substrate-reserved values (see
  [Termination reasons](#termination-reasons)). Always populated.
* `error: str | None` — populated for runtime-level failures (the
  block could not be invoked, the container died unexpectedly, etc.).

`Invoke` does no artifact handling, no execution-record handling, no
validation, no quarantine. Its sole job is to run the block and
report what happened.

### 3. Commit

`commit(workspace, plan, invocation_result)`

The single canonical write path for both artifacts and execution
records. Returns nothing of substantive interest to the caller; the
side effect is the registered `ArtifactInstance`s and the appended
`BlockExecution` reachable via the workspace. Concrete callers
(e.g., `flywheel run block`) may surface other invocation metadata
(such as the raw `ContainerResult`) for ergonomics; the canonical
contract is "the workspace is updated, or the call raises".

1. **Determine expected outputs from the termination reason.** Look up
   the output slot set declared for
   `invocation_result.termination_reason` in the block's declaration.
   * Substrate-reserved values describing non-clean exits (`crash`,
     `interrupted`, `timeout`, `protocol_violation`) implicitly map to
     the empty output set; declarations do not enumerate them.
   * Project-defined values map to whatever the block declared.
   * A clean-exit announcement that does not match any declared
     reason is normalized to `protocol_violation` before this step
     (see [Termination reasons](#termination-reasons)) and therefore
     also maps to the empty output set.
2. **Process each expected output slot** under the universal
   commit-passing-slots rule (see
   [Commit-passing-slots](#commit-passing-slots)). Slots are processed
   in declaration order. Each slot independently runs the pipeline:
   collect → validate → forge. Per-slot outcomes:
   * On success at the forge step: the slot is in `output_bindings`
     and its proposal directory is consumed.
   * On failure at any step: the slot is in `rejected_outputs` with a
     `phase` recording where it broke; its proposal bytes are moved to
     quarantine via `quarantine_slot(...)`. The remaining slots
     continue.

   The pipeline steps:
   * **collect**: ensure the proposal exists and is readable. I/O
     errors here record the slot as rejected at `phase="output_collect"`.
   * **validate**: run the proposal through the
     `ArtifactValidatorRegistry` against the slot's
     `ArtifactDeclaration`. Validator rejection records the slot as
     rejected at `phase="output_validate"`.
   * **forge**: call `Workspace.register_artifact(...)` with the
     proposal directory as source. `register_artifact` copies the
     bytes into the canonical immutable artifact location at
     `<workspace>/artifacts/<id>/` and records the `ArtifactInstance`.
     A failure here records the slot as rejected at
     `phase="artifact_commit"`.
3. **Build the `BlockExecution` record once** and write it via
   `Workspace.record_execution(...)` — see
   [Canonical write paths](#canonical-write-paths). The execution's
   top-level `failure_phase` is the most-downstream phase any rejected
   slot reached (`artifact_commit` > `output_validate` >
   `output_collect`); `status` follows the
   [status mapping](#status-mapping).
4. **Append artifact sequence entries** for output slots that declare
   `sequence:`. Appends happen only after the artifact instance and
   succeeded `BlockExecution` exist. If any expected output slot was
   rejected and the execution is failed, no sequence entries are
   appended for that execution.
5. **Ingest execution telemetry** from `/flywheel/telemetry`. Accepted
   telemetry and telemetry rejections are durable records tied to the
   execution. Telemetry ingest is non-fatal and never changes execution
   status.
   Recording the execution persists the workspace. Telemetry records
   are persisted separately and best-effort so telemetry write failures
   cannot turn a completed execution into a substrate failure.

### Commit-passing-slots

The substrate's universal commit policy. An execution that produces
multiple expected outputs commits the slots that pass and rejects
(quarantines) the slots that fail. Failures at any of the
`output_collect`, `output_validate`, or `artifact_commit` phases are
all per-slot — they do not abort the commit, they do not undo earlier
successes, they do not skip later slots. `output_bindings` contains
exactly the slots that successfully reached the forge;
`rejected_outputs` contains the rest, each with a `phase` recording
where it failed. There is no rollback.

A note on cost: validated proposals are copied from
`<workspace>/proposals/...` into `<workspace>/artifacts/<id>/`. For
typical block outputs this is negligible. For very large directory
trees the copy may matter; if and when that becomes a real bottleneck
the spec can introduce a same-filesystem hardlink or rename
optimization without changing the API.

### Execution telemetry

Execution telemetry is durable metadata about an execution: model
usage, cost, timing, or other measurements supplied by the runtime
wrapper or project code. It is not an artifact, not managed state, and
not a validator-controlled output.

The one-shot runtime exposes `/flywheel/telemetry` as the telemetry
candidate directory. Candidate files are flat direct children of that
directory, must have a `.json` suffix, must not exceed 256 KiB, and
must be UTF-8 JSON with this strict envelope:

```json
{
  "kind": "claude_usage",
  "source": "flywheel-claude",
  "data": {
    "total_cost_usd": 0.01
  }
}
```

`kind` is a non-empty string without surrounding whitespace. `data`
must be a JSON object. `source` is optional and must be a string when
present; an empty string is normalized to no source. Unknown envelope
keys, malformed JSON, non-object data, oversized files, nested
directories, or non-`.json` files are rejected.

Accepted candidates become `ExecutionTelemetry` records with
Flywheel-assigned `id`, `execution_id`, and `recorded_at`. Rejected
candidates become `RejectedTelemetry` records with `execution_id`,
candidate path, reason, timestamp, and a best-effort
`preserved_path` when Flywheel could copy the rejected bytes under
`telemetry_rejections/<execution_id>/<rejection_id>/`. Rejections are
warnings in the ledger; they do not fail the execution.

The substrate provides the mount and validates the envelope. A battery
that wants stronger provenance for its own telemetry must also control
who can write the candidate file. The Claude battery, for example,
keeps `/flywheel/telemetry` root-owned and derives usage telemetry
from a root-owned capture of the runner's stdout; the prompt-driven
agent cannot read or write that capture file.

### Failure recording for prepare-time and invoke-time errors

A single helper writes a `BlockExecution` for failures upstream of
step 3 of `Commit`:

```python
def commit_failure(
    workspace: Workspace,
    *,
    execution_id: str,
    block_name: str,
    started_at: datetime,
    phase: str,
    error: str,
    termination_reason: str = "crash",
) -> BlockExecution:
    ...
```

* Looks up `status` from the [status mapping](#status-mapping).
* Sets `failure_phase=phase`, `error=error`,
  `termination_reason=termination_reason`.
* Calls `Workspace.record_execution(...)`.
* For invoke-time failures that have an `ExecutionPlan`, ingests
  execution telemetry and then cleans up the proposal directory.
* For prepare-time failures that do not have an `ExecutionPlan`, cleans
  up any pre-allocated proposal directory directly.

`commit_failure` is called from `prepare()` (for proposal-allocation
and input-resolution failures, both of which happen after the
`execution_id` is minted in step 1) and from the per-runtime
`invoke()` adapters (for invoke failures). The caller passes in the
`execution_id` and `started_at` it already has. Invoke failures pass an
`ExecutionPlan` so any telemetry already written by the runtime wrapper
can still be recorded before cleanup.

## Data shapes

### `ArtifactDeclaration` (template-level)

Each declares a `name` and a `kind`. The kind→name binding is
template-wide; any block that references a declared name inherits the
kind from this declaration.

* `kind ∈ {copy, git}`.

### `OutputSlot` and `InputSlot` (block-level)

Block I/O slots reference artifacts by **name only**. They do not
carry a `kind` — kind is looked up from the template-level
`ArtifactDeclaration`.

* `OutputSlot` carries: `name`, `container_path`.
* `InputSlot` carries: `name`, `container_path`, `optional`.

At the `BlockDefinition` level, outputs are not a flat
`list[OutputSlot]`. They are grouped by termination reason — see
[Block declaration syntax](#block-declaration-syntax).

### `ArtifactInstance`

Two kinds:

* `copy` — produced by block executions; full machinery (validation,
  quarantine, supersedes lineage).
* `git` — created at execution time during input resolution; carries
  `(repo, commit, git_path)`; never produced as an output; not
  subject to validators or quarantine. See
  [Git artifacts](#git-artifacts).

`RejectedOutput` carries a `phase` field recording where the slot
failed (`output_collect`, `output_validate`, or `artifact_commit`).

### `BlockExecution`

| Field | Notes |
| --- | --- |
| `id` | Execution ID. |
| `block_name` | The block that was executed. |
| `started_at` / `finished_at` | Wall-clock bracket. |
| `status` | `succeeded \| failed \| interrupted`. Derived from `termination_reason` via the [status mapping](#status-mapping); the mapping is the single canonical rule. |
| `input_bindings` | `dict[str, str]`. Implicitly carries input parentage via each artifact's `produced_by`. |
| `output_bindings` | `dict[str, str]`. |
| `exit_code` | The container's exit code, when one is available. |
| `elapsed_s` | Wall-clock time in seconds. |
| `image` | The container image used. |
| `runner` | `"container_one_shot"` or `"container_persistent"`. |
| `failure_phase` | Set only when `status="failed"`. Values from `runtime.FAILURE_*`. |
| `error` | Human-readable error message recorded alongside `failure_phase`. |
| `rejected_outputs` | `dict[str, RejectedOutput]`. Each entry records a `reason`, the `phase` of failure (`output_collect`, `output_validate`, or `artifact_commit`), and a `quarantine_path` when quarantine I/O succeeded. |
| `termination_reason` | `str`. Always populated. Either a substrate-reserved value (`crash`, `interrupted`, `timeout`, `protocol_violation`) describing what the substrate observed, or a project-defined value the block announced via the termination channel. See [Termination reasons](#termination-reasons). |

## Block declaration syntax

Output slots are grouped by termination reason. A block must declare
at least one termination reason.

```yaml
block: SomeBlock
image: some/image:tag
inputs:
  - name: source
    container_path: /scratch/source
outputs:
  normal:
    - name: result
      container_path: /scratch/result
```

A block with multiple termination reasons:

```yaml
block: PlayAgent
image: some/image:tag
inputs:
  - name: observation
    container_path: /scratch/observation
outputs:
  normal:
    - name: turn_summary
      container_path: /scratch/turn_summary
  defer:
    - name: action
      container_path: /scratch/action
```

Termination reason names (`normal`, `defer`, etc.) are project-defined
labels. Flywheel does not interpret them.

Associations between termination reasons and invoked child blocks are
specified separately in [block-invocation.md](block-invocation.md).

## Termination reasons

A first-class concept on `BlockExecution`. Stored as a string.
Always populated.

Two namespaces share the field:

### Substrate-reserved values

Describe events the substrate itself observes. Project blocks **must
not** announce any of these values via the termination channel; the
substrate sets them based on what it saw. The reserved set is closed
and small:

* `crash` — block process exited non-zero (or the runtime otherwise
  observed a hard failure: container died unexpectedly, server-side
  exception in a persistent container, dropped connection).
* `interrupted` — block was killed by stop signal or operator
  interrupt.
* `timeout` — block exceeded a substrate-enforced deadline (when
  deadlines are introduced; the value is reserved now so the
  vocabulary is stable).
* `protocol_violation` — block exited cleanly but failed to follow
  the termination-channel protocol. Specifically, any of:
  * the block did not write to the termination channel at all;
  * the block wrote a value matching a substrate-reserved name
    (collision);
  * the block wrote a value that is not declared as one of its
    termination reasons.

### Project-defined values

Any string the block declares as a termination reason. Projects own
the semantics of these values entirely. Flywheel does not interpret
them; it only uses them to look up the expected output set in the
block declaration.

A clean-exit announcement that is neither reserved nor declared is
normalized to `protocol_violation` (the substrate substitutes the
value before recording it).

### Substrate behavior

* Read from the block's termination channel at exit time (see
  [Runtime mechanism for termination reasons](#runtime-mechanism-for-termination-reasons)).
* Normalize per the rules above (substrate-observed events override
  any block announcement; collisions and undeclared announcements
  become `protocol_violation`).
* Look up the expected output set in the block declaration. Reserved
  values implicitly map to the empty output set; declarations do not
  enumerate them.
* Store on `BlockExecution.termination_reason`.

### Status mapping

`BlockExecution.status` is derived from `termination_reason` by this
single canonical rule. `failure_phase` is set only when
`status="failed"` and is whichever phase the substrate or commit-step
recorded (see [Failure phases](#failure-phases)).

| `termination_reason` | `status` | `failure_phase` |
| --- | --- | --- |
| `crash` | `failed` | `invoke` |
| `interrupted` | `interrupted` | none |
| `timeout` | `failed` | `invoke` |
| `protocol_violation` | `failed` | `output_protocol` |
| any project-defined value | `succeeded` if every expected slot reached the forge; otherwise `failed` with the most-downstream rejected-slot phase | from commit |

## Runtime mechanism for termination reasons

Each runtime defines exactly one channel for the block to announce
its termination reason. The channel is part of the substrate
contract, not an implementation detail — block authors and runtimes
both depend on it.

### One-Shot Container Runner

One-shot containers use default-deny Docker networking. If a block does
not declare `network:`, Flywheel starts it with `--network=none`. Blocks
that need a model API, a project Docker network, or another explicit
network namespace must declare `network: <value>` in block YAML.

* **Channel**: a sidecar file at `/flywheel/termination` inside the
  container. The path is part of the runtime contract.
* **Format**: a single line of UTF-8 text, optionally trailed by a
  newline. The line content is the termination reason verbatim.
  Whitespace at start or end is stripped. Anything other than a
  single non-empty line (multiple lines, empty file, missing file,
  invalid UTF-8) is treated as "no announcement".
* **Lifecycle**: the block writes the file at any point before exit.
  The executor reads it after the container exits.
* **Substrate normalization**:
  * Non-zero container exit → `crash` (file contents ignored).
  * Zero exit, no announcement (missing file, empty file, malformed
    file) → `protocol_violation`.
  * Zero exit, announcement matches a substrate-reserved name →
    `protocol_violation`.
  * Zero exit, announcement matches a declared project reason →
    that value verbatim.
  * Zero exit, announcement matches no declared reason and no
    reserved name → `protocol_violation`.

### Persistent Container Runner

Persistent runtime details are specified in
[persistent-runtime.md](persistent-runtime.md). Its canonical
termination channel is
`/flywheel/exchange/requests/<execution_id>/termination` inside the
persistent container. The built-in HTTP response may include a
`termination_reason` fallback, but the sidecar is the primary channel.

* **Channel**: a sidecar file at
  `/flywheel/exchange/requests/<execution_id>/termination`.
* **Format**: same single-line UTF-8 format as one-shot
  `/flywheel/termination`.
* **Lifecycle**: the worker writes the file before returning from the
  request. The substrate reads it during runtime observation.
* **Substrate normalization**:
  * Failed dispatch, server exception, malformed response, or worker
    `status: failed` → `crash`.
  * Successful response, no valid announcement → `protocol_violation`.
  * Successful response, announcement matches a substrate-reserved name
    → `protocol_violation`.
  * Successful response, announcement matches a declared project reason
    → that value verbatim.
  * Successful response, announcement matches no declared reason and no
    reserved name → `protocol_violation`.

## Canonical write paths

Three sanctioned APIs. All other ways to mutate the workspace ledger
are private.

### `Workspace.register_artifact(...)` — copy artifacts only

The only sanctioned path for creating a `copy` `ArtifactInstance`.
Takes a source path (the proposal), validates (when a validator
registry and declaration are supplied), copies the bytes into the
canonical immutable artifact location at
`<workspace>/artifacts/<id>/`, handles `SupersedesRef` lineage, and
calls the private `_add_artifact`.

The executor uses this same single API to commit block-produced
outputs — see [Proposal-then-forge](#proposal-then-forge). There is
no "adopt" mode, no pre-allocation of artifact IDs, no special path
for executor-produced artifacts. `register_artifact` always owns
canonical placement.

### Proposal-then-forge

The executor mounts a per-slot **proposal directory** under
`<workspace>/proposals/<execution_id>/<slot>/` into the container.
The block writes its output bytes there. These are proposals, not
artifacts. They have no `ArtifactInstance` ID; they are not in the
ledger; they are not in the canonical artifact location.

At commit time the executor presents each proposal to the validator:

* On accept: `register_artifact` copies the proposal bytes into the
  canonical immutable artifact location (the forge) and records the
  `ArtifactInstance`.
* On reject: `quarantine_slot` moves the proposal bytes to
  quarantine.

In both cases the proposal is consumed; nothing lingers under
`proposals/` after commit. The forge entry is born only from a
validated proposal.

### `register_git_artifact(...)` — git artifacts only

The sanctioned path for creating a `git` `ArtifactInstance`.
Signature roughly:

```python
def register_git_artifact(
    workspace: Workspace,
    name: str,
    declaration: ArtifactDeclaration,
    project_root: Path,
) -> ArtifactInstance:
    ...
```

* Resolves the declared git path within the project repo to a
  concrete `(repo, commit, git_path)` triple.
* Constructs an `ArtifactInstance(kind="git", ...)`.
* Calls `_add_artifact`.
* No validators (git artifacts are inputs only — they are not produced
  by block executions, so there is nothing to validate).
* No quarantine (no rejection path).

This is intentionally lighter than `register_artifact`. Git artifacts
are project source materials; the only thing being "registered" is
the resolved snapshot reference, not produced bytes.

### `Workspace.record_execution(...)` — block executions

The sole sanctioned path for writing `BlockExecution` records.
Signature roughly:

```python
def record_execution(
    self,
    *,
    execution_id: str,
    block_name: str,
    started_at: datetime,
    termination_reason: str,
    input_bindings: dict[str, str],
    output_bindings: dict[str, str],
    runner: Literal["container_one_shot", "container_persistent"],
    image: str,
    exit_code: int | None,
    elapsed_s: float | None,
    failure_phase: str | None = None,
    error: str | None = None,
    rejected_outputs: dict[str, RejectedOutput] | None = None,
) -> BlockExecution:
    ...
```

Field defaults live here. The [status mapping](#status-mapping) is
applied here — `status` is derived from `termination_reason`, not
passed in. Status/failure_phase invariants are checked here.
Workspace persistence is triggered here (or by the caller, but in a
single conventional location).

### Privacy

`Workspace._add_artifact` and `Workspace._add_execution` are private.
They are accessible to the three sanctioned write paths above and are
otherwise off-limits. This is the canonical mechanism for preventing a
parallel pipeline from being written.

## Runtime variants

The substrate's preferred mode is **one-shot containers** — one
container per block execution. Stateless one-shot execution is
hermetic by construction. Managed one-shot state is specified
separately in [state.md](state.md).

The substrate also supports **persistent containers** — one
long-lived container hosting many executions over its lifetime — but
only as a necessity. Some blocks hold unmanaged internal state that
the substrate cannot serialize and restore between
executions, so the container has to stay up across executions to
preserve that state. Persistent containers are not a performance
optimization; they are a workaround for state the substrate cannot
manage.

Two implementations:

* **one-shot container runner** — starts a fresh container per
  execution. Block runs, container exits, executor commits.
* **persistent container runner** — starts a long-running container
  once and dispatches each execution as a request over an
  in-container control channel.

Both implement the same `Runtime` interface (the `Invoke` phase).
They share `Prepare` and `Commit` entirely. They differ only in:

* How they invoke the block (start container vs. POST to control
  channel).
* Where they read the termination sidecar
  (`/flywheel/termination` vs. the persistent exchange request dir).
* Container lifecycle (per-execution vs. across many executions).

## Failure phases

Used on `BlockExecution.failure_phase` when `status="failed"`. The
canonical enumeration lives in `flywheel/runtime.py`.

Values:

* `stage_in` — input resolution or proposal-directory allocation
  failed.
* `invoke` — runtime failed to invoke; container died unexpectedly;
  block exited non-zero (`crash`); deadline exceeded (`timeout`).
* `output_collect` — at least one output slot's proposal was
  unreadable.
* `output_validate` — at least one output slot was rejected by its
  validator.
* `artifact_commit` — at least one output slot passed validation but
  failed at the forge step.
* `output_protocol` — block exited cleanly but failed to follow the
  termination-channel protocol
  (`termination_reason="protocol_violation"`).

When multiple slots fail at different phases, the execution's
top-level `failure_phase` is the most-downstream of them
(`artifact_commit` > `output_validate` > `output_collect`). Per-slot
phase information lives on each `RejectedOutput.phase`.

## CLI entry points

* `flywheel run block` — operator-triggered single block execution.
  Wraps `prepare → invoke → commit`.
* `flywheel import artifact`.
* `flywheel fix execution`.
* `flywheel amend artifact`.

## Deferred to follow-on specs

Each item below is a real substrate concern that this spec
deliberately defers. They land in their own normative passes.

* **State.** The full state lifecycle is specified in
  [state.md](state.md). State is orthogonal to artifacts and belongs
  to execution lineages, not artifact bindings.
* **Pattern control flow.** Invocation is specified in
  [block-invocation.md](block-invocation.md). Loops, limits, failure
  handling, and lane-scoped resolution belong to
  [pattern-execution.md](pattern-execution.md).
* **Artifact sequences and future tags.** Ordered histories are
  specified in [artifact-sequences.md](artifact-sequences.md). If
  unordered tags are needed later, they should be metadata over normal
  immutable artifact instances.
* **Patterns** and the pattern runner.
* **Block groups** (parallel scheduling).
* **Project hooks**, refactored to be runtime/executor-shaped rather
  than agent-launcher-shaped.
* **Auto-lineage on bit-identical re-runs** — design decision: should
  `flywheel run block` automatically wire a `SupersedesRef` when
  re-running a block whose prior execution had a rejected slot?
* **Layering enforcement.** Introduce `flywheel/batteries/` and an
  import-direction lint that fails if anything in `flywheel/` core
  imports from it. The structural enforcement that prevents the
  agent vocabulary from re-entering core.
