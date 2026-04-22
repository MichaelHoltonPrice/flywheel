"""Generic run-level defaults for the pattern runner.

Successor to the agent-flavored :class:`flywheel.agent.AgentBlockConfig`
on the runner's consumer surface.  The runner needs three things from
its caller every run: which workspace to write to, which template to
look blocks up in, and where the project lives on disk.  Anything
else any executor wants is a free-form ``defaults`` dict the executor
factory can consult when it hands out executors per block.

Keeping this struct generic — no agent fields, no battery fields — is
what lets the runner dispatch *any* block type (a one-shot container,
a workspace-persistent runtime, an agent battery, a future executor we
have not invented yet) through one path: build :class:`RunDefaults`,
hand the runner an executor factory, let the factory pick the right
executor per block.

The ``defaults`` dict is intentionally untyped.  Each executor reads
the keys it cares about and ignores the rest; collisions across
executors are a project-hooks concern, not the runner's.  Typical
contents (none required):

* ``run_id_prefix``: a short string the project wants stamped on
  every :class:`flywheel.artifact.RunRecord` the runner opens.
* ``model``: a default model identifier an agent battery would
  layer behind any per-instance override.

Step-1 of the runner / battery separation introduces this struct
without yet wiring it as the runner's ``base_config`` parameter — the
swap happens in step-2 alongside the rest of the agent surface
removal.  The dataclass exists today so callers can start producing
:class:`RunDefaults` instances incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flywheel.template import Template
from flywheel.workspace import Workspace


@dataclass(frozen=True)
class RunDefaults:
    """The runner's required per-run inputs plus a free-form defaults bag.

    Attributes:
        workspace: The workspace every block in the run reads from /
            writes to.  Becomes the durable ledger for every
            :class:`flywheel.artifact.BlockExecution` the runner
            produces.
        template: The block-resolution surface for the run.  Used by
            executor factories to look up :class:`BlockDefinition`
            objects by name and decide which executor to return for
            each block.
        project_root: Filesystem root of the project.  Used by
            executors that need to mount source directories or read
            project-relative configuration; the runner itself does
            not touch the filesystem here.
        defaults: Free-form per-run overrides any executor may
            consult.  Keys are an executor-by-executor contract;
            unknown keys are ignored.  Empty by default.
    """

    workspace: Workspace
    template: Template
    project_root: Path
    defaults: dict[str, Any] = field(default_factory=dict)
