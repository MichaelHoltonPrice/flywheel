# Flywheel

Orchestration framework for measurable AI improvement loops.

Flywheel wires Docker containers together into pipelines where
scores determine what happens next. It provides artifact tracking,
visibility into hybrid human+agent workflows, and reusable
capabilities (computer use agents, LLM agent wrappers) so projects
can focus on their domain logic.

## Audience

The intended primary user is an **AI agent orchestrator** (Cursor,
Claude Code, or a custom agent loop) driving flywheel as a CLI.
The orchestrator invokes small composable subcommands; flywheel
manages durable workspace state, runs blocks, and records every
input, output, and execution decision in a workspace ledger the
orchestrator can read between calls.

Humans use the same surface and read the same workspaces.

## CLI surface

Four user-facing commands:

* `flywheel create workspace` — materialize a fresh workspace from
  a template.
  ([docs/cli/create-workspace.md](docs/cli/create-workspace.md))
* `flywheel import artifact` — register an external file or
  directory as a workspace artifact instance.
* `flywheel run block` — execute one block ad hoc against a
  workspace.
* `flywheel run pattern` — execute a declarative multi-instance
  pattern against a workspace.

Per-command reference docs land in [docs/cli/](docs/cli/) as the
underlying behavior is pinned. AI agents working with flywheel
should start at [AGENTS.md](AGENTS.md). Design rationale is in
[docs/vision.md](docs/vision.md); implementation decisions are
in [docs/architecture.md](docs/architecture.md).

## Setup

```bash
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
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
