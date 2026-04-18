"""Minimal project-hooks protocol consumed by ``flywheel run pattern``.

Patterns own decision logic (which roles fire, when), so the
project-side surface shrinks to the things only the project can
know: starting / stopping external resources (the ARC game
server in cyberarc, a SQL fixture in a hypothetical query-eval
project), parsing project-specific CLI arguments, and supplying
the launcher overrides those arguments produce (extra env vars,
extra mounts, a pre-launch hook, etc.).

This is deliberately *much* smaller than
:class:`flywheel.agent_loop.AgentLoopHooks`.  There is no
``decide``, no ``build_prompt``, no ``on_execution`` — those
concerns either move to the pattern (``decide`` becomes the
trigger vocabulary), to the role's prompt file
(``build_prompt``), or to the runner's post-execution callbacks
(``on_execution`` becomes a block-level ``post_check``).

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
            ``extra_env``, ``extra_mounts``, ``pre_launch_hook``,
            ``isolated_network``, ``mcp_servers``, ``allowed_tools``.
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

    Mirrors :func:`flywheel.agent_loop.load_hooks_class` so
    project authors only have to learn one import-path syntax.
    Kept as a separate function (rather than re-exported) so
    that retiring ``AgentLoop`` in P7 of the campaign does not
    silently break ``flywheel run pattern`` callers.
    """
    if ":" not in import_path:
        raise ValueError(
            f"Project hooks import path must be "
            f"'module.path:ClassName', got {import_path!r}"
        )
    module_path, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
