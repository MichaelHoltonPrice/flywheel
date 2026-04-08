# Flywheel Vision

This document describes what flywheel is for and where it is
headed. It includes aspirational elements that are not yet
implemented. For current implementation decisions, see
[architecture.md](architecture.md).

## Core idea

Flywheel is a framework for building AI systems that improve
measurably. When you can score output — via a simulator, an LLM
judge, or human review — you can break a system into blocks with
I/O contracts, track what each block produces, and improve blocks
independently.

Three things are co-equal in flywheel's design:

1. **Improvement loops.** Blocks are scored, and scores drive what
   happens next — retry, escalate, replace, or move on.
2. **Artifact tracking.** Every input and output is recorded.
   Artifacts are the connective tissue between blocks and the raw
   material for improvement.
3. **Visibility.** You cannot adapt a workflow you cannot see.
   Flywheel makes the state of every block, artifact, and decision
   inspectable by humans and agents alike.

These three reinforce each other. Visibility without artifact
tracking is anecdotal. Artifact tracking without visibility is a
data lake. Improvement loops without either are blind iteration.

## The foundry

Each project has a **foundry** — a directory managed by flywheel
that holds templates, workspaces, and their artifacts. The project
source code and the foundry are peers: the project is what's being
built; the foundry is where flywheel orchestrates the building.

Blocks consume project inputs (source code, configurations, data)
and produce foundry artifacts (checkpoints, scores, logs). Results
can also flow back into the project — blocks can modify project
source code directly, and project directories are full git
artifacts tracked by flywheel. The foundry and the project are
separate but not isolated; they influence each other through
tracked artifacts.

The foundry is also where templates live. A template declares what
a workspace can do — what artifacts exist, what blocks can run,
and how containers are configured. When you create a workspace
from a template, flywheel sets up the foundry directory structure
and begins tracking artifacts.

## Why visibility and artifact tracking matter

It is hard to predict ahead of time which problems an AI agent
team will solve autonomously and which will require human judgment.
A task that looks routine may hit an edge case that needs a human
eye; a task that looks hard may fall to a well-prompted agent on
the first try.

This unpredictability is a fundamental property of the current
state of AI, not a gap to be engineered away. It means the system
must support hybrid workflows — human and agent, in varying
proportions — and make it easy to shift between them as you learn
what works.

Artifact tracking and visibility are what make this adaptation
possible. When every block's inputs and outputs are recorded and
inspectable, a human can step into a workflow at any point,
understand what happened, and decide what should happen next. An
agent can do the same. The workflow does not need to be designed
for one mode or the other upfront.

## Blocks with I/O contracts

A block is defined by its **contract**: what it takes in, what it
produces, and how success is measured.

```
block: find_click_target
  input: {screenshot: image, action: string}
  output: {x: int, y: int, confidence: float}
```

Success criteria may be an automated metric, a human judgment, an
LLM evaluation, or something built up over time from initially
human-mediated feedback. Not every block starts with an automated
success criterion — the criterion itself can be improved as
understanding deepens.

### Implementations vs approaches

There are two ways to have multiple solutions to a problem.

**Implementations** are interchangeable behind a single contract.
A click-targeting block might have a CV template matcher, an LLM
vision agent, and a human annotator — all taking the same input
and producing the same output. Callers don't know or care which
one runs. This is the escalation chain case.

**Approaches** are structurally different blocks that address the
same goal with different contracts, inputs, and improvement
trajectories. An RL training pipeline and a bot-writing agent
both produce game-playing behavior, but they are different blocks
with different inputs (checkpoints vs source code), different
containers, and different improvement loops. They are not
interchangeable at runtime.

What connects approaches is evaluation. Both produce artifacts
that can be scored by the same evaluator, and those scores are
comparable. Flywheel's artifact tracking makes this comparison
natural — different approaches produce different artifact
provenance chains, but their evaluation artifacts land in the same
workspace and can be inspected side by side.

## Contracts and dependency injection

Contracts and dependency injection are the core structural
concepts in flywheel.

A **contract** defines what a block takes in and what it produces.
Contracts are the universal concept — they govern how blocks
relate to each other regardless of how they are wired together.

**Dependency injection** is how blocks are wired: a unit declares
the roles it needs filled, and specific block implementations are
injected at configuration time. The unit manages lifecycle and
shared context; the injected blocks provide the variable logic.

### Motivating examples

Two concepts from mathematics help explain why this design works.

**Composition** is the simplest case. When data flows naturally
between blocks — the output of one is the input of another — you
get a pipeline:

```
state → bot → action → engine.step → new_state → bot → ...
```

Each block has a clear contract, each output feeds the next input.
But composition can be an idealization. A fast evaluator might
batch episodes or skip intermediate observations. The per-step
chain describes the logic but not the implementation. It is not
always useful to decompose a system into composed elemental
operations.

**Functionals** — functions that take other functions as
arguments — capture the more general pattern. An evaluator is a
functional `E` that takes a bot (a function from game state to
action) and returns a measurement:

```
E(bot) → metrics

where:
  bot:      state → action   (injected block)
  engine:   dependency        (pinned at configuration time)
  metrics:  {mean, median, fights_won, ci}
```

The engine is a dependency that parameterizes the functional. The
bot is not a pipeline stage; it is an argument to a higher-order
function. This is more natural than composition when the system's
job is to *apply* a block within context and measure what happens.

The functional analogy is not always the right lens, however.
Consider a CUA gameplay loop with five injected callables
(read_state, decide_action, find_clicks, verify, reset). The loop
is not measuring any single one — it assembles a machine from
parts and runs it. Whether it behaves like a functional over the
bot, the CV module, or something else depends on which pieces the
caller varies and which it holds fixed.

### When dependency injection fits

Dependency injection is a good fit when the system applies blocks
within the context of a dependency, when multiple blocks operate
against the same shared context, when the relationship is better
described as "roles in a coordinated process" than "stages in a
data pipeline", or when some pieces are running systems rather
than callables.

## Escalation chains

Where multiple implementations share a contract (see above),
they can form an escalation chain — ranked cheapest first. When
the cheap implementation fails or reports low confidence, the
framework escalates to the next one. The expensive result becomes
training data for the cheap one.

```
find_click_target:
  1. CV template matching (free, fast, limited)
  2. LLM vision agent ($0.01/call, flexible)
  3. Human annotation ($0.50/call, ground truth)
```

Over time, escalation frequency decreases as cheaper
implementations improve. The primary metric is escalation rate:
how often does a block need to fall back to an expensive
implementation?

## Progressive replacement

The lifecycle of a block implementation often follows a
progression:

1. A **human** performs the task manually, and the results are
   stored.
2. An **LLM agent** is given the accumulated examples and
   automates the task, though at significant per-call cost.
3. A **code module** is trained or written using accumulated
   results. It is cheap and fast, but may not handle every case.
4. An **escalation chain** combines code for most cases with LLM
   fallback for edge cases. The code module improves as more
   edge cases are captured.

Not every block follows this progression. Some start as code. Some
never need an LLM. The framework supports any trajectory, but the
common pattern is expensive → cheap with escalation.

## Artifact provenance

Artifacts come in two forms: **declarations** and **instances**.

A declaration is a template-level concept — a named type of
artifact (e.g., "checkpoint", "score"). An instance is a
concrete, immutable record produced by a specific block
execution. A workspace accumulates instances over its lifetime;
each block execution consumes specific input instances and
produces new output instances.

This means a block can execute repeatedly within a workspace.
Each execution produces distinct outputs, and the full
provenance is explicit: which execution consumed which artifacts
and produced which artifacts. The workspace is not a snapshot —
it is a history.

Provenance is not a separate feature layered on top of the
system. It is the natural consequence of recording block
executions and artifact instances correctly. The core primitives
are BlockExecution and ArtifactInstance; provenance is the
directed acyclic graph formed by their links.

## Execution records

Every block execution is recorded. The record captures which
input artifact instances were consumed, which output artifact
instances were produced, which implementation ran, the cost
(compute, API, or human time), the latency, whether the
execution succeeded or failed and with what confidence, and
whether escalation was involved. Blocks may also emit side
artifacts — additional outputs beyond the contract such as
logs, intermediate files, or debug snapshots — that may be
useful for inspection or downstream improvement.

These records are not just for debugging. They are training data,
the evaluation corpus, and the decision basis for progressive
replacement. They are analogous to a replay buffer in
reinforcement learning, but for an entire system of blocks.

## Agent-mediated pipeline influence

When an agent step runs inside a pipeline, the agent can invoke
orchestrator-owned capabilities during its execution.

Consider an agent improving a bot. The pipeline could evaluate the
bot as a separate step after the agent finishes. But the agent
benefits from evaluating its own intermediate work: write a bot,
evaluate it, read the score, revise. The agent needs the evaluator
*during* its run, not after.

The evaluator is the same functional either way. What differs is
who initiates it and when:

- In a **pipeline eval**, the orchestrator runs eval after the
  work step, and the worker never calls eval itself.
- In an **agent-requested eval**, the agent invokes eval during
  its run. The orchestrator exposes the capability through a
  constrained tool interface, fulfills requests, and returns
  results. Intermediate evaluations can be stored as side
  artifacts.

This generalizes: an agent step can request any orchestrator-owned
capability during execution through a constrained tool interface.
The orchestrator decides which capabilities to expose. The agent
decides when to use them.

The important constraint is that this power stays bounded. The
agent can only invoke capabilities the orchestrator explicitly
provides. Without that boundary, you stop having a pipeline with
an agent in it and start having an agent that informally drives
the pipeline. The tool interface is what keeps the agent's
influence legible and governable.

## Solve-once capabilities

Some capabilities are hard to build, reusable across projects, and
belong in flywheel rather than in each project:

- **Computer use agents** provide session management, screenshot
  loops, click targeting, and verification for visual interaction
  with applications.
- **Agent wrappers** handle container launch, authentication, MCP
  tool exposure, and output collection for the Anthropic Agents
  SDK, Codex, and similar agent runtimes.

Projects provide their domain-specific containers and artifacts.
Flywheel provides the orchestration and these shared capabilities.

## Design principles

1. **Contracts first.** Define what a block does before deciding
   how it does it.
2. **Record everything.** Every execution is future training data.
3. **Cheapest first.** Always try the cheapest implementation
   before escalating.
4. **Escalate, don't fail.** Fall back to expensive implementations
   rather than returning errors.
5. **Visibility is not optional.** If a human or agent cannot
   inspect the state of a workflow, the workflow cannot adapt.
6. **Right wiring for the situation.** Compose when data flows
   naturally, and use dependency injection when the system
   coordinates roles against shared context. Do not force
   composition when it obscures the actual structure.
7. **Humans are implementations.** A human doing a task is just
   another block implementation with high cost and high accuracy.
