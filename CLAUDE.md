# Flywheel — Claude Code orientation

Claude Code orchestrators (and any agent that landed here looking
for a `CLAUDE.md`) should read [AGENTS.md](AGENTS.md). It is the
canonical orientation for AI agents working with flywheel and
covers every concept this file used to.

If your task is to invoke a specific CLI command, read its
per-command reference in [docs/cli/](docs/cli/).

A note on naming: flywheel ships `batteries/claude/`, a Claude
Code agent wrapper that lets a flywheel pattern *run* a Claude
Code agent as a block. That is the inverse direction from this
file: the agent is the workload, not the orchestrator. See
[AGENTS.md](AGENTS.md) "Batteries".
