# Flywheel Agent Orientation

Flywheel is a CLI and Python library for reproducible block
execution and pattern-driven workflows. It gives projects durable,
inspectable workspaces where block inputs, outputs, managed state,
and run decisions are recorded as a ledger. AI agent orchestrators
(Cursor, Claude Code, custom loops) are a major intended user:
they can create workspaces, import artifacts, run blocks, and run
patterns while Flywheel owns the durable side.

`AGENTS.md` is the entry point for AI agents working on Flywheel
or a Flywheel-using project. For product direction, read
[docs/vision.md](docs/vision.md). For implementation decisions,
read [docs/architecture.md](docs/architecture.md).

## Supported CLI Surface

| Command | Purpose | Notes |
| --- | --- | --- |
| `flywheel create workspace` | Materialize a workspace from a template. | See [docs/cli/create-workspace.md](docs/cli/create-workspace.md). |
| `flywheel import artifact` | Add files to the workspace as an immutable copy artifact instance. | See [docs/cli/import-artifact.md](docs/cli/import-artifact.md). |
| `flywheel run block` | Execute one block through the canonical block-execution path. | The main substrate surface. |
| `flywheel run pattern` | Execute an ordered pattern of block-execution cohorts. | Uses `RunRecord`; no agent-specific runner. |

`flywheel container ...` remains for request-response container
management, but persistent container execution is not the current
recommended path.

When in doubt, inspect [flywheel/cli.py](flywheel/cli.py); it is
the canonical source for argument shapes.

## Project Layout

A Flywheel-using project root contains:

* `flywheel.yaml` - declares `foundry_dir` and optional validator
  registry factories.
* `<foundry_dir>/templates/workspaces/<name>.yaml` - workspace
  templates.
* `<foundry_dir>/templates/blocks/<name>.yaml` - block
  declarations loaded by the block registry.
* `<foundry_dir>/templates/patterns/<name>.yaml` - pattern
  declarations.
* `<foundry_dir>/workspaces/<name>/` - Flywheel-managed results:
  `workspace.yaml`, artifact instances, state snapshots, run
  records, proposals, quarantine, and recovery directories.

Templates describe capabilities. Workspaces contain results.

## Core Concepts

**Block execution** is the substrate primitive: one block, one
execution id, one input binding set, one runtime observation, and
one commit/finalization pass. The canonical path is in
[flywheel/execution.py](flywheel/execution.py):

1. prepare inputs, proposals, `/flywheel`, and managed state;
2. run a one-shot container body;
3. commit artifacts, managed state, quarantine/recovery, and the
   `BlockExecution` ledger record.

**Artifacts** have declarations and instances. Declarations live
in templates. Instances live in workspaces and are immutable.
Copy artifacts are directory-shaped. Git artifacts record a repo,
commit, and path. Ordered histories are represented by artifact
sequences over immutable artifact instances, not by a separate
storage kind.

**State** is not an artifact. Artifacts model edges between block
executions: one execution produces a named value that another may
bind as input. Managed state models node-local continuity for a
specific execution lineage. It is restored from and captured to
state snapshots, not through artifact binding.

**Runs** group related block executions without adding run fields
to `BlockExecution`. A `RunRecord` stores ordered step/member
results. Ad hoc `run block` invocations do not create runs.

**Patterns** are ordered declarations of semantic cohorts. The
current pattern path lives in
[flywheel/pattern_declaration.py](flywheel/pattern_declaration.py),
and [flywheel/pattern_execution.py](flywheel/pattern_execution.py).
Each member is an ordinary canonical block execution. Cohorts
support `min_successes: all` and `min_successes: 1`; scheduling is
currently sequential even when the cohort is semantically parallel.

## Batteries

Batteries are reusable project-style capabilities bundled with
Flywheel. They are not substrate abstractions.

The current bundled agent batteries are `batteries/claude/` and
`batteries/codex/`: Dockerfiles, entrypoints, and agent runners that
can be used as block images. Projects invoke them as ordinary blocks
by declaring a block that uses a derived battery image, bakes or mounts
the project prompt into the image's documented prompt path, and sets
any needed env or Docker arguments in the block declaration. Artifact
finalization, state snapshots, validation, quarantine, and
`BlockExecution` records are not battery-owned.

Important boundary:

* Core block execution must not grow Claude-, Codex-, MCP-, prompt-, or
  handoff-shaped schema fields.
* Batteries may provide images, entrypoints, runners, validators,
  examples, and CLI conveniences.
* Battery code must enter durable state through the same
  sanctioned workspace write paths as every other block.

The old handle-based agent modules have been removed. Batteries should
keep their code under `batteries/<name>/` and enter the durable
substrate through ordinary block declarations and `flywheel run block`.

## Validation

Projects can register artifact validators and state validators via
`flywheel.yaml`. Flywheel calls them from the canonical
finalization path. Validators own project content policy; the
substrate owns staging, invocation, quarantine/recovery, and
ledger consistency.

## Development

```bash
python -m venv .venv

# Windows cmd
.venv\Scripts\activate.bat

# PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -e ".[dev]"
python -m pytest -q
```

Run focused tests for the area you touch, then the full suite
before considering work complete.

## Where To Dig Deeper

* [docs/architecture.md](docs/architecture.md) - current
  implementation decisions.
* [docs/specs/block-execution.md](docs/specs/block-execution.md) -
  normative block-execution model.
* [docs/specs/state.md](docs/specs/state.md) - managed state
  contract.
* [examples/hello-agent](examples/hello-agent/) - minimal bundled
  Claude battery example.
* [examples/hello-codex](examples/hello-codex/) - minimal bundled
  Codex battery example.
