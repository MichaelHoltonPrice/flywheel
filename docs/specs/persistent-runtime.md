# Persistent Runtime

Persistent runtime is the substrate variant for blocks whose runtime
state must survive across block executions. The substrate abstraction is
not HTTP: a persistent runtime takes a prepared `ExecutionPlan`, runs one
execution against a long-lived worker, and returns observed runtime
facts for the shared commit path. Flywheel ships one built-in
implementation today: a Docker container with an HTTP `/health` and
`/execute` server.

## Scope

Persistent runtimes are workspace-scoped. A workspace-persistent block
gets one container per `(workspace, block)` pair. That container may
serve executions from multiple pattern lanes and multiple pattern runs
in the same workspace. Lane isolation remains an artifact-resolution
property; a persistent worker that needs its own per-lane internal state
must key that state from request data supplied by the project.

`state: managed` is invalid with `lifecycle: workspace_persistent`.
Managed state is for restart/resume of one-shot containers. Persistent
blocks may declare `state: unmanaged` or `state: none`.

## Exchange Layout

The persistent container receives one Flywheel-owned writable mount:

```text
/flywheel/exchange/
```

This is not a scratchpad, project workspace, or artifact store. It is a
runtime exchange tree. On the host it lives at:

```text
<workspace>/runtimes/<block>/exchange/
```

Each execution gets a request directory under that exchange root:

```text
requests/<execution_id>/
  request.json
  response.json
  input/<slot>/...
  output/<slot>/...
  telemetry/...
  termination
```

The request directory is also the execution proposal root. Commit reads
outputs and telemetry directly from this directory; there is no copy-back
pipeline. Successful requests are removed after commit. Failed requests
use the same cleanup rules as one-shot executions; per-slot quarantine
still happens only for commit-time output rejections.

## Input and Output Paths

Persistent requests do not use the block declaration's `container_path`
fields. Flywheel stages every resolved input into
`input/<slot.name>/` before dispatch and expects output proposals under
`output/<slot.name>/`. The request envelope tells the worker the
container-side roots for those trees.

Inputs are copies from the workspace ledger into the request directory,
not bind mounts. A worker may read or mutate those staged files during a
request without changing the canonical artifact. The next request gets a
fresh staged copy from the resolver.

Block-level `docker_args` and static block `env` are container-start
configuration. Per-execution environment overlays are sent in the
request envelope as `env`; they are not sticky process environment for
later requests.

## Built-In HTTP Protocol

Flywheel starts the persistent Docker container detached and sets
`FLYWHEEL_CONTROL_PORT` to the port the in-container server must bind.
The exchange root is mounted at `/flywheel/exchange`.

`GET /health` must return a successful HTTP response before Flywheel
dispatches work.

`POST /execute` receives a JSON request:

```json
{
  "protocol": "1",
  "request_id": "exec_...",
  "env": {"MODEL": "sonnet"},
  "input_root": "/flywheel/exchange/requests/exec_.../input",
  "output_root": "/flywheel/exchange/requests/exec_.../output",
  "telemetry_root": "/flywheel/exchange/requests/exec_.../telemetry",
  "termination_path": "/flywheel/exchange/requests/exec_.../termination",
  "args": []
}
```

`request_id` is the same string as the eventual `BlockExecution.id`.

The worker writes output bytes under `output/<slot>/`, telemetry
candidates under `telemetry/`, and a single-line project termination
reason to `termination`. The response body must be JSON:

```json
{
  "status": "succeeded",
  "termination_reason": "normal"
}
```

`termination` is the canonical termination channel. The response
`termination_reason` is accepted only as a fallback when the termination
file is absent. If both exist and disagree, the sidecar wins. If neither
contains a valid declared reason, the request commits as
`protocol_violation`.

If the worker returns `{"status": "failed", "error": "..."}`, Flywheel
records a failed execution through the normal commit path with
termination reason `crash` and failure phase `invoke`.

## Container Identity

Persistent Docker containers use a deterministic name:

```text
flywheel-<workspace_hash>-<block>
```

They also carry labels for:

* workspace id and workspace path;
* block name;
* lifecycle `workspace_persistent`;
* scope `workspace`;
* block template compatibility hash;
* declared image and inspected image id when available;
* protocol version;
* control port.

When a container already exists, Flywheel reuses it only if the labels
match the current block declaration and image identity. Mismatch fails
loudly. Flywheel does not auto-restart stale persistent containers
because their in-memory state may be the reason they exist.

Image inspection is part of the fingerprint. If Flywheel cannot inspect
the declared image, startup fails loudly rather than labeling a
container with an unverifiable identity.

## Failure Semantics

Runtime dispatch failures after prepare are durable failed block
executions with `failure_phase: invoke`. This preserves the invariant
that every minted execution id is represented in `workspace.yaml`.

Examples:

* container cannot be started;
* health check fails;
* `/execute` connection drops;
* `/execute` returns malformed JSON;
* worker reports `status: failed`.

Commit-time failures, output validation, artifact commit, telemetry
ingest, block invocation, and pattern lane behavior are identical to
one-shot execution.

## Operator Commands

`flywheel container list --workspace <path>` lists Docker containers
with the workspace label.

`flywheel container stop --workspace <path> --template <template>
--block <block>` writes `<exchange>/.stop` as a cooperative hint, then
uses Docker stop and remove. `--force` removes the container without the
graceful stop window.

Workers should poll `/flywheel/exchange/.stop` and exit when it appears;
Flywheel still sends Docker stop after writing the sentinel. The
sentinel text is the operator-supplied stop reason.
