"""Per-instance runtime knobs forwarded to executors at launch.

Lifted out of :mod:`flywheel.pattern_runner` so both the runner
and the agent-battery wiring (:mod:`flywheel.pattern_handoff`)
can depend on these small public-ish shapes without forming an
import cycle through the runner module.

Existing call sites import these names from
:mod:`flywheel.pattern_runner` and keep working — the runner
re-exports them as a back-compat surface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from flywheel.executor import BlockExecutor
from flywheel.template import BlockDefinition


@dataclass(frozen=True)
class InstanceRuntimeConfig:
    """Runtime knobs the pattern runner applies to one instance.

    Supplied by the project layer (e.g. cyberarc's
    ``ProjectHooks``) and forwarded to the executor on launch.
    Keyed by *instance* name, not block name, because two
    instances of the same block may eventually need different
    config (e.g. different game ids, different mount directories).

    Attributes:
        extra_env: Environment variables merged into the
            container's env on startup.
        extra_mounts: Bind mounts appended to the executor's
            standard mount list.  Each entry is
            ``(host_path, container_path, mode)``.
    """

    extra_env: dict[str, str] = field(default_factory=dict)
    extra_mounts: list[tuple[str, str, str]] = field(
        default_factory=list)


# Type alias for the caller-supplied executor factory.  Given a
# block definition (the resolved target of an ``on_tool``
# instance, or of a prompt-less continuous role), the factory
# returns the executor that should dispatch calls to that block.
# Today's MVP shape is "always return the shared executor," but
# the factory signature keeps lifecycle-per-block routing
# available without an API change when the need arrives.
ExecutorFactory = Callable[[BlockDefinition], BlockExecutor]
