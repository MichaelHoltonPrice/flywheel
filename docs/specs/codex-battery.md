# Codex Battery

The Codex battery is a reusable block image for running OpenAI Codex
CLI inside Flywheel. It is a battery, not a substrate feature: Flywheel
core sees an ordinary one-shot container block that writes
`/flywheel/termination`, optional telemetry under `/flywheel/telemetry`,
managed state under `/flywheel/state`, and declared outputs under
`/output/<slot>`.

## Image Contract

The base image lives in `batteries/codex/` and provides:

* `Dockerfile.codex`, which installs `@openai/codex`;
* standard command-line tools expected by coding agents, including
  `ripgrep`;
* `entrypoint.sh`, which performs auth/session scrubbing, optional network
  isolation, managed-state staging, and termination-sidecar writing;
* `agent_runner.py`, which runs `codex exec --json` and records telemetry;
* `codex_handoff_hook.py`, a Codex `PostToolUse` hook for selected MCP
  handoff tools.

Projects normally derive their own image:

```dockerfile
FROM flywheel-codex:latest
COPY prompt/prompt.md /app/agent/prompt.md
COPY mcp_servers/ /app/agent/mcp_servers/
```

The prompt path defaults to `/app/agent/prompt.md` and can be changed
with `FLYWHEEL_AGENT_PROMPT`.

## Block Declaration

A Codex block is a normal one-shot block. Typical fields:

```yaml
name: SomeCodexAgent
image: project-codex-agent:latest
docker_args:
  - -v
  - codex-auth:/home/codex/.codex:rw
env:
  MODEL: gpt-5.5
  CODEX_AUTO_COMPACT_TOKEN_LIMIT: "200000"
  CODEX_APPROVAL_POLICY: never
  CODEX_SANDBOX: danger-full-access
  MCP_SERVER_MOUNT_DIR: /app/agent/mcp_servers
state: managed
outputs:
  normal:
    - name: result
      container_path: /output/result
```

`state: managed` is recommended when a pattern relaunches the same
agent over multiple block executions. The battery persists Codex session
files and the configured scratchpad through Flywheel managed state.

`CODEX_AUTO_COMPACT_TOKEN_LIMIT` is the preferred way to set Codex CLI's
automatic history-compaction threshold. The runner validates the value as
a positive integer and passes it to Codex as
`-c model_auto_compact_token_limit=<value>`. Put this in block YAML, or
thread it from pattern params, rather than editing the auth volume's
`config.toml` or using `CODEX_EXTRA_ARGS`. If both `CODEX_EXTRA_ARGS` and
`CODEX_AUTO_COMPACT_TOKEN_LIMIT` set the same Codex config key, the named
Flywheel setting is appended last and wins.

## Authentication

The example uses a Docker volume mounted at `/home/codex/.codex`.
Initialize that volume by running Codex's supported login flow inside
the battery image:

```bash
docker volume create codex-auth
docker run -it --rm \
  -v codex-auth:/home/codex/.codex \
  --entrypoint bash \
  flywheel-codex:latest \
  -lc "chown -R codex:codex /home/codex/.codex && su -s /bin/bash codex -c 'codex login --device-auth'"
```

For API-key auth, initialize the same volume with
`codex login --with-api-key` and pass the key on stdin, for example
`printenv OPENAI_API_KEY | codex login --with-api-key` inside the
container. `entrypoint.sh` keeps Codex credentials but removes unrelated
local session history before staging the managed-state session copy.

## Managed State

The entrypoint runs as root, stages battery-private state, locks
`/flywheel/state` to root, then runs Codex as the non-root `codex` user.
After Codex exits, the entrypoint copies these values back to managed
state:

* `codex_sessions/` - Codex CLI session files needed for `codex exec resume`;
* `session_id` - latest observed Codex thread id;
* `scratchpad/` - contents of `FLYWHEEL_SCRATCHPAD_DIR`.

The agent can read the live session files while Codex is running because
the CLI itself needs them. It cannot read the root-owned persisted
managed-state copy from previous executions.

## MCP Servers

Project MCP servers are discovered exactly like the Claude battery:
`agent_runner.py` scans `MCP_SERVER_MOUNT_DIR` for files named
`*_mcp_server.py`, and `MCP_SERVERS` selects which server names to enable.
For Codex, the runner writes `[mcp_servers.<name>]` tables into
`$CODEX_HOME/config.toml`.

Optional sidecar manifests named `*_mcp_server.json` may list tool names,
but Codex discovers actual tools through MCP at runtime.

## Tool Handoff

Codex supports `PostToolUse` hooks. The battery uses this to stop the
agent after selected MCP tools complete:

```yaml
env:
  HANDOFF_TOOL_CONFIG: |
    {
      "mcp__project__request_eval": {
        "termination_reason": "eval_requested",
        "required_paths": ["/output/bot/bot.py"],
        "result_path": "/input/score/scores.json",
        "result_label": "Evaluation result",
        "placeholder_marker": "Evaluation requested."
      }
    }
```

When the hook sees the configured tool, it verifies required paths, writes
handoff metadata, and returns `continue: false` with the configured stop
reason. The root entrypoint turns that into `/flywheel/termination`.
Flywheel then commits the parent block and invokes any declared
`on_termination` child blocks through the normal block-invocation path.

Codex's hook API does not currently provide the same durable session-JSONL
splice point used by the Claude battery. On the next relaunch, the Codex
agent resumes through `codex exec resume --last`; the project should make
the result artifact available as a normal input and explain that path in
the prompt. The battery records `result_path` and `result_label` in
managed-state handoff metadata for project prompts and future tooling.

## Telemetry

The runner writes a strict telemetry envelope to
`/flywheel/telemetry/codex_usage.json`:

```json
{
  "kind": "codex_usage",
  "source": "flywheel-codex",
  "data": {
    "thread_id": "...",
    "model": "...",
    "event_count": 12,
    "input_tokens": 100,
    "cached_input_tokens": 50,
    "output_tokens": 20
  }
}
```

Flywheel core ingests this through the generic telemetry lane. Telemetry
never decides execution status.

Raw Codex stderr is preserved as `/flywheel/telemetry/codex_stderr.txt`
when present. The runner suppresses one known non-fatal Codex
v0.125.0 rollout-recorder warning from the operator log after successful
runs, while leaving the raw stderr sidecar available for debugging.

## Boundary

Do not add Codex-specific fields to `BlockExecution`, pattern records, or
core execution. Codex auth, config, hooks, session persistence, and CLI
event parsing belong in `batteries/codex/` and derived project images.
