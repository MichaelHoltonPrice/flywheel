# Desktop Service Battery

A desktop service is a project-owned long-lived container that exposes a
virtual desktop to ordinary controller blocks. It is not a Flywheel core
concept: Flywheel sees a normal `workspace_persistent` block, and project
controllers discover the desktop endpoint by project convention.

The reusable `flywheel-desktop` battery provides the common mechanics:

* a virtual X display with fixed resolution and DPI;
* an optional project GUI process launched from `DESKTOP_APP_COMMAND`;
* a generic HTTP surface for screenshots and input;
* a small Flywheel persistent-runtime control surface.

It intentionally has no Claude dependency and no project-specific game or
browser logic. A Claude controller, a generated Python script, or a test
harness can all drive the same desktop API.

## Ports

The battery serves two distinct contracts:

* `FLYWHEEL_CONTROL_PORT` is the persistent-runtime control port. Flywheel
  uses `/health` and `/execute` there. The built-in `/execute` is a no-op
  that writes `normal` to the request termination file. Flywheel publishes
  this port on `127.0.0.1` for the host-side persistent-runtime client; host
  processes should treat it as a Flywheel control surface, not a project API.
* `DESKTOP_API_PORT` is the project-facing computer-use API. Controller
  blocks reach it on a project Docker network, usually through a stable
  `--network-alias`.

Projects should not use the host-mapped Flywheel control port for controller
traffic. In-network controller traffic should use a URL such as:

```text
DESKTOP_URL=http://decker-desktop:8080
```

`/execute` is accepted only on `FLYWHEEL_CONTROL_PORT`, not on
`DESKTOP_API_PORT`.

## Desktop API

All coordinates are window-relative integer pixels with top-left origin. The
display size is fixed at startup through `DESKTOP_WIDTH`,
`DESKTOP_HEIGHT`, and `DESKTOP_DPI`; runtime resize is not part of the
contract.

Base endpoints:

```text
GET  /health
GET  /screenshot?format=jpeg|png&quality=N
POST /click {"x": 10, "y": 20, "button": "left", "double": false}
POST /move {"x": 10, "y": 20}
POST /type {"text": "hello"}
POST /key {"key": "Return", "modifiers": ["ctrl"]}
POST /drag {"from": [0, 0], "to": [100, 100], "duration_ms": 250}
POST /wait {"duration_ms": 500}
POST /reset
GET  /files/<relative-path>
POST /files/<relative-path>
DELETE /files/<relative-path>
```

The `/files/` endpoints are restricted to `DESKTOP_SHARED_DIR`. They are a
project-facing exchange surface for the desktop app and its controllers; they
are not a Flywheel artifact store. Controllers must copy anything durable into
their own block output directories before exit.

Action endpoints are synchronous. The controller owns timing and should call
`/wait` when it needs a render delay. The battery does not auto-wait between
actions.

Errors return JSON:

```json
{"error": "bad_request", "detail": "..."}
```

## Packaging Pattern

Projects normally derive from the base image:

```dockerfile
FROM flywheel-desktop:latest
COPY my-gui-app /app/my-gui-app
ENV DESKTOP_APP_COMMAND="/app/my-gui-app"
```

A controller block joins the same Docker network and talks to the desktop by
network alias:

```yaml
name: MyDesktop
runner: container
lifecycle: workspace_persistent
image: my-desktop:latest
network: my-cua-network
docker_args:
  - --restart=unless-stopped
  - --network-alias=my-desktop
outputs:
  normal: []

---

name: MyController
runner: container
image: my-controller:latest
network: my-cua-network
env:
  DESKTOP_URL: http://my-desktop:8080
```

The Docker network is project setup, not substrate state. Create it with
`docker network create <name>` before running the pattern.
`--network-alias` is DNS metadata for that selected network, so it stays
in `docker_args`; network membership itself belongs in `network:`.

Long-lived desktop blocks should usually pass `--restart=unless-stopped`.
That lets Docker restart the service container if the daemon restarts or the
container process exits. It does not recover a crashed GUI process inside a
still-running desktop container; controllers should inspect `/health.app` and
use `/reset` as the normal recovery path.

Controller blocks using the Claude battery should not enable
`NETWORK_ISOLATION=1` unless the battery's firewall is also extended to allow
the project Docker network. The default Claude network isolation permits
loopback, DNS, Anthropic API traffic, and explicit host ports; it does not
permit traffic to an arbitrary Docker-network alias such as
`my-desktop:8080`.

## Durability

The desktop framebuffer, GUI process, and shared directory are non-durable
service state. Durable facts must be written by controller blocks as normal
Flywheel output artifacts or artifact sequences. The desktop service must not
write workspace YAML, register artifacts, or bypass the canonical commit path.

If a project needs reproducible game progress, the controller should upload an
input save through `/files/`, reset the app, drive the desktop, download the
exported save through `/files/`, and write it to its output artifact before
the controller exits.

For visual traces, the recommended shape is one trace artifact per controller
segment, appended to a lane-scoped artifact sequence. The trace artifact can
contain many screenshot files plus action logs and error diagnostics. The
individual screenshot files do not need to be separate artifact instances.
