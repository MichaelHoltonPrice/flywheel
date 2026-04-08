# Flywheel

Orchestration framework for measurable AI improvement loops.

Flywheel wires Docker containers together into pipelines where
scores determine what happens next. It provides artifact tracking,
visibility into hybrid human+agent workflows, and reusable
capabilities (computer use agents, LLM agent wrappers) so projects
can focus on their domain logic.

See [docs/vision.md](docs/vision.md) for the design rationale
(includes aspirational elements) and
[docs/architecture.md](docs/architecture.md) for implementation
decisions (future work is noted separately at the end).

## Setup

```bash
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
