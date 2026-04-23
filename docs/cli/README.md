# Flywheel CLI reference

Per-command documentation for the `flywheel` CLI, written for AI
agent orchestrators that need to invoke flywheel as a tool. See
[../../AGENTS.md](../../AGENTS.md) for the higher-level audience
and concepts.

## Commands

* [create-workspace.md](create-workspace.md) — `flywheel create
  workspace`. Materialize a fresh workspace from a template.
* [import-artifact.md](import-artifact.md) — `flywheel import
  artifact`. Register an external directory as a `copy`-kind
  artifact instance inside an existing workspace, optionally
  subject to a project-supplied validator. (Artifact instances
  are always directory-shaped; wrap single files first.)

The remaining two user-facing commands (`run block`, `run
pattern`) gain their per-command docs as the underlying behavior
is pinned. Until then, agents should read the argparse setup at
the top of [`flywheel/cli.py`](../../flywheel/cli.py) and use
`--help`.

## Conventions used in command docs

Each per-command doc follows the same structure so an agent can
locate the section it needs without rereading prose:

* **Purpose** — one or two sentences.
* **When to use it** — and when not to (the alternative
  command).
* **Prerequisites** — project layout, workspace state, or
  external state required before invocation.
* **Invocation** — copy-pasteable CLI form with all flags.
* **What gets created / changed** — the durable side effects.
* **Failure modes** — exception types, error messages, and how
  to recover.
* **Verification** — how to confirm the command succeeded
  beyond exit code zero.
* **Typical next steps** — the commands an orchestrator usually
  invokes after this one.
* **Implementation pointers** — file:symbol references for
  agents that need to dig.
