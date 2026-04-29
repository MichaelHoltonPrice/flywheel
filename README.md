# Flywheel

Orchestration framework for measurable AI improvement loops.

Flywheel runs blocks, records their inputs and outputs, and keeps a
durable workspace ledger for reproducible, pattern-driven workflows.
Projects provide domain containers and validators; Flywheel provides
the substrate for artifact tracking, managed state, patterns, and
reusable batteries.

## Use Cases

Flywheel is useful whenever a project needs repeatable block
execution, artifact provenance, and durable records of how a workflow
progressed. That includes RL training/evaluation loops, simulation
workflows, benchmark runs, human-in-the-loop review, and AI agent
orchestrators driving the CLI.

Humans and AI agents use the same surface and read the same
workspaces.

## CLI Surface

Core user-facing commands:

* `flywheel create workspace` - materialize a fresh workspace from a
  template. See [docs/cli/create-workspace.md](docs/cli/create-workspace.md).
* `flywheel import artifact` - add files to the workspace as an
  immutable artifact instance.
* `flywheel run block` - execute one block ad hoc against a workspace.
* `flywheel run pattern` - execute a declarative pattern against a
  workspace.

Per-command reference docs land in [docs/cli/](docs/cli/) as the
underlying behavior is pinned. AI agents working with Flywheel should
start at [AGENTS.md](AGENTS.md). Design rationale is in
[docs/vision.md](docs/vision.md); implementation decisions are in
[docs/architecture.md](docs/architecture.md).

## Examples

Versioned examples live under [examples/](examples/). Start with
[examples/hello-agent](examples/hello-agent/), a smoke test for the
Flywheel-provided Claude battery image, or
[examples/hello-codex](examples/hello-codex/), the parallel Codex CLI
battery example. Both invoke a battery as an ordinary block with
`flywheel run block`.

Reusable battery images live under [batteries/](batteries/). The
Claude and Codex batteries provide managed-state agent containers. The
desktop battery provides an agent-agnostic virtual desktop service with
screenshot/input APIs; projects derive from it and wire controller
blocks to it through project Docker networking.

## Setup

```bash
python -m venv .venv

# Windows cmd
.venv\Scripts\activate.bat

# PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

## Verification

```bash
bash scripts/verify.sh
```

Runs linting (ruff) and tests (pytest). Run this before considering
any work complete.

## License

Apache 2.0. Copyright Heartland AI (doing business as Hopewell AI).
See [LICENSE](LICENSE).
