# Hello Agent Example

This example is a smoke test for the Flywheel-provided Claude agent
battery. The battery is invoked as an ordinary Flywheel block whose
block declaration uses the Claude battery image. It demonstrates:

* ordinary `flywheel run block` execution with a battery image
* the Flywheel-provided `batteries/claude` Docker image
* prompt delivery through an example image derived from the Claude battery
* managed state through `/flywheel/state`
* agent control files under `/flywheel/control`
* normal termination through `/flywheel/termination`

Run the commands from the Flywheel repository root. The example uses
the local Docker image tags `flywheel-claude:latest` and
`flywheel-hello-agent:latest`, plus the Docker volume `claude-auth`.

The `claude-auth` volume is shared across runs. Re-run the volume
bootstrap command whenever the host Claude credentials rotate.

## Windows Cmd

```bat
python -m venv .venv
.venv\Scripts\activate.bat

python -m pip install -e .

docker build -t flywheel-claude:latest -f batteries\claude\Dockerfile.claude batteries\claude
docker build -t flywheel-hello-agent:latest -f examples\hello-agent\Dockerfile.hello-agent examples\hello-agent

docker volume create claude-auth
docker run --rm -v claude-auth:/auth -v "%USERPROFILE%\.claude:/host-claude:ro" python:3.12-slim sh -c "cp -a /host-claude/. /auth/ && chown -R 1000:1000 /auth"

cd examples\hello-agent

python -m flywheel create workspace --name ws --template hello-agent

python -m flywheel run block --workspace foundry\workspaces\ws --template hello-agent --block HelloAgent --state-lineage hello-agent

python -m flywheel run block --workspace foundry\workspaces\ws --template hello-agent --block HelloAgent --state-lineage hello-agent

findstr /C:"block_name" /C:"status" /C:"state_snapshot_id" /C:"state_snapshots" foundry\workspaces\ws\workspace.yaml
```

Reset the example workspace:

```bat
rmdir /s /q foundry\workspaces\ws
```

The example agent image derives from `flywheel-claude:latest` and
copies `prompt\prompt.md` into `/app/agent/prompt.md`. If you edit the
block image or environment after the first run, reset the workspace
before reusing the same `--state-lineage`.

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -e .

docker build -t flywheel-claude:latest -f batteries\claude\Dockerfile.claude batteries\claude
docker build -t flywheel-hello-agent:latest -f examples\hello-agent\Dockerfile.hello-agent examples\hello-agent

docker volume create claude-auth
docker run --rm -v claude-auth:/auth -v "$env:USERPROFILE\.claude:/host-claude:ro" python:3.12-slim sh -c "cp -a /host-claude/. /auth/ && chown -R 1000:1000 /auth"

Set-Location examples\hello-agent

python -m flywheel create workspace --name ws --template hello-agent

python -m flywheel run block --workspace foundry\workspaces\ws --template hello-agent --block HelloAgent --state-lineage hello-agent

python -m flywheel run block --workspace foundry\workspaces\ws --template hello-agent --block HelloAgent --state-lineage hello-agent

Select-String -Path foundry\workspaces\ws\workspace.yaml -Pattern "block_name","status","state_snapshot_id","state_snapshots"
```

Reset the example workspace:

```powershell
Remove-Item -Recurse -Force foundry\workspaces\ws
```

The example agent image derives from `flywheel-claude:latest` and
copies `prompt\prompt.md` into `/app/agent/prompt.md`. If you edit the
block image or environment after the first run, reset the workspace
before reusing the same `--state-lineage`.

## macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -e .

docker build -t flywheel-claude:latest -f batteries/claude/Dockerfile.claude batteries/claude
docker build -t flywheel-hello-agent:latest -f examples/hello-agent/Dockerfile.hello-agent examples/hello-agent

docker volume create claude-auth
docker run --rm -v claude-auth:/auth -v "$HOME/.claude:/host-claude:ro" python:3.12-slim sh -c "cp -a /host-claude/. /auth/ && chown -R 1000:1000 /auth"

cd examples/hello-agent

python -m flywheel create workspace --name ws --template hello-agent

python -m flywheel run block --workspace foundry/workspaces/ws --template hello-agent --block HelloAgent --state-lineage hello-agent

python -m flywheel run block --workspace foundry/workspaces/ws --template hello-agent --block HelloAgent --state-lineage hello-agent

grep -E "block_name|status|state_snapshot_id|state_snapshots" foundry/workspaces/ws/workspace.yaml
```

Reset the example workspace:

```bash
rm -rf foundry/workspaces/ws
```

The example agent image derives from `flywheel-claude:latest` and
copies `prompt/prompt.md` into `/app/agent/prompt.md`. If you edit the
block image or environment after the first run, reset the workspace
before reusing the same `--state-lineage`.
