"""Minimal project-hooks protocol consumed by ``flywheel run pattern``.

Patterns own decision logic (which roles fire, when), so the
project-side surface covers only what the pattern cannot know on
its own: starting / stopping external resources (the ARC game
server in cyberarc, a SQL fixture in a hypothetical query-eval
project), parsing project-specific CLI arguments, and supplying
the launcher overrides those arguments produce (extra env vars,
extra mounts, a pre-launch hook, etc.).

Implementors are free to add helper methods on the same class,
but only the two declared below are called by the runner.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol

from flywheel.template import Template
from flywheel.workspace import Workspace


class ProjectHooks(Protocol):
    """Project-supplied glue for ``flywheel run pattern`` runs.

    A class that satisfies this protocol is loaded from the
    ``project_hooks`` key of ``flywheel.yaml`` (or via
    ``--project-hooks`` on the CLI).  Instantiated with no
    arguments; ``init`` does the real work.
    """

    def init(
        self,
        workspace: Workspace,
        template: Template,
        project_root: Path,
        args: list[str],
    ) -> dict[str, Any]:
        """Set up project-level resources and return launcher overrides.

        Args:
            workspace: The flywheel workspace the pattern will run
                against.  Available so hooks can register initial
                artifacts or read project configuration recorded
                on the workspace.
            template: The workspace template, for hooks that need
                to inspect declared blocks / artifacts.
            project_root: The directory containing ``flywheel.yaml``.
            args: Project-specific CLI arguments — everything
                after ``--`` on the ``flywheel run pattern``
                command line.  Hooks parse this however they
                like (typically with ``argparse``).

        Returns:
            A mapping of :class:`flywheel.agent.AgentBlockConfig`
            field overrides.  The runner merges this on top of the
            CLI flags before launching agents.  Common keys:
            ``extra_env``, ``extra_mounts``, ``isolated_network``,
            ``mcp_servers``, ``allowed_tools``.

            One non-config key is also recognised:
            ``launch_fn`` — a callable with the
            :func:`flywheel.agent.launch_agent_block` signature
            that the pattern runner uses in place of the default
            launcher.  Projects that want the host-side full-stop
            handoff loop (see
            :mod:`flywheel.agent_handoff`) inject
            :func:`flywheel.agent_handoff.launch_agent_with_handoffs`
            here, partially bound with the project's
            :class:`~flywheel.agent_handoff.BlockRunner`.
            Returning ``None`` is equivalent to ``{}``.
        """
        ...

    def teardown(self) -> None:
        """Release project-level resources.

        Called once after the pattern run completes (or fails).
        Optional — the runner detects the missing method and
        skips the call rather than requiring a no-op stub.
        """
        ...


def load_project_hooks_class(import_path: str) -> type:
    """Resolve ``module.path:ClassName`` into a hooks class.

    The CLI calls this once with the value of ``project_hooks``
    from ``flywheel.yaml`` (or the ``--project-hooks`` flag) and
    instantiates the result with no arguments before invoking
    :meth:`ProjectHooks.init`.
    """
    if ":" not in import_path:
        raise ValueError(
            f"Project hooks import path must be "
            f"'module.path:ClassName', got {import_path!r}"
        )
    module_path, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
