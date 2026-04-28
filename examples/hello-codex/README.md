# Hello Codex Example

This example is a smoke test for the Flywheel-provided Codex agent
battery. The battery is invoked as an ordinary Flywheel block whose
block declaration uses an image derived from `flywheel-codex:latest`.
It demonstrates:

* ordinary `flywheel run block` execution with a Codex battery image
* prompt delivery through `/app/agent/prompt.md`
* managed state through `/flywheel/state`
* Codex session and scratchpad persistence managed by the battery
* an explicit 200K automatic compaction threshold via
  `CODEX_AUTO_COMPACT_TOKEN_LIMIT`
* normal termination through `/flywheel/termination`

Run the commands from the Flywheel repository root. The example uses
the local Docker image tags `flywheel-codex:latest` and
`flywheel-hello-codex:latest`, plus a Docker volume named
`codex-auth`.

The `codex-auth` volume is shared across runs. Re-run the volume
bootstrap command when the Codex login expires. The bootstrap runs
Codex's supported device-auth login inside the volume; it does not copy
host credential files. You can also initialize the same volume with
`codex login --with-api-key` if API-key auth is preferable.

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -e .

docker build -t flywheel-codex:latest -f batteries\codex\Dockerfile.codex batteries\codex
docker build --no-cache -t flywheel-hello-codex:latest -f examples\hello-codex\Dockerfile.hello-codex examples\hello-codex

docker volume create codex-auth
docker run -it --rm `
  -v codex-auth:/home/codex/.codex `
  --entrypoint bash `
  flywheel-codex:latest `
  -lc "chown -R codex:codex /home/codex/.codex && su -s /bin/bash codex -c 'codex login --device-auth'"

Set-Location examples\hello-codex

python -m flywheel create workspace --name ws --template hello-codex

python -m flywheel run block --workspace foundry\workspaces\ws --template hello-codex --block HelloCodex --state-lineage hello-codex

python -m flywheel run block --workspace foundry\workspaces\ws --template hello-codex --block HelloCodex --state-lineage hello-codex

Select-String -Path foundry\workspaces\ws\workspace.yaml -Pattern "block_name","status","state_snapshot_id","state_snapshots","codex_usage"
```

Reset the example workspace:

```powershell
Remove-Item -Recurse -Force foundry\workspaces\ws
```

## macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -e .

docker build -t flywheel-codex:latest -f batteries/codex/Dockerfile.codex batteries/codex
docker build --no-cache -t flywheel-hello-codex:latest -f examples/hello-codex/Dockerfile.hello-codex examples/hello-codex

docker volume create codex-auth
docker run -it --rm \
  -v codex-auth:/home/codex/.codex \
  --entrypoint bash \
  flywheel-codex:latest \
  -lc "chown -R codex:codex /home/codex/.codex && su -s /bin/bash codex -c 'codex login --device-auth'"

cd examples/hello-codex

python -m flywheel create workspace --name ws --template hello-codex

python -m flywheel run block --workspace foundry/workspaces/ws --template hello-codex --block HelloCodex --state-lineage hello-codex

python -m flywheel run block --workspace foundry/workspaces/ws --template hello-codex --block HelloCodex --state-lineage hello-codex

grep -E "block_name|status|state_snapshot_id|state_snapshots|codex_usage" foundry/workspaces/ws/workspace.yaml
```

Reset the example workspace:

```bash
rm -rf foundry/workspaces/ws
```

The example agent image derives from `flywheel-codex:latest` and
copies `prompt/prompt.md` into `/app/agent/prompt.md`. If you edit the
block image or environment after the first run, reset the workspace
before reusing the same `--state-lineage`.
