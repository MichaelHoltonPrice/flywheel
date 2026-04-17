"""Tool-to-block manifests.

A manifest declares which MCP-server tools invoke which blocks.
Each side-effecting MCP tool names exactly one block, plus the
mapping from agent-supplied tool arguments and MCP-server-tracked
state into the block's declared params.

Manifests are read by flywheel at startup and used by the
:class:`~flywheel.execution_channel.ExecutionChannel` to enforce
``(mcp_server, tool) → block`` invocations: in-container code
cannot invent new artifact effects by sending unrecognized block
names to the channel.

Schema (YAML; one file per MCP server):

.. code-block:: yaml

    mcp_server: arc

    tools:
      predict_action:
        block: predict
        args:                  # tool arg name → block param name
          action: action_id
          x: x
          y: y
        server_state:          # MCP server attr → block param name
          _last_frame: source_state
          _level: level

      take_action:
        block: take_action
        args:
          action: action_id
          x: x
          y: y

The ``args`` and ``server_state`` mappings are loaded and validated
here, but only the channel-enforcement table
(``(mcp_server, tool) → block``) is consulted by the channel.
The mappings are used by tool-side helpers (Phase 4: the unified
``@block_invocation`` decorator) to assemble block params from
agent calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from flywheel.blocks.registry import BlockRegistry


@dataclass(frozen=True)
class ToolBinding:
    """Binding of one MCP tool to one block.

    Attributes:
        tool: MCP tool name as exposed to the agent.
        block: Block name this tool invokes (must exist in the
            project's :class:`BlockRegistry`).
        args: Mapping from MCP-tool argument names to block param
            names.  E.g. ``{"action": "action_id"}`` means the
            agent's ``action`` argument flows into the block's
            ``action_id`` param.
        server_state: Mapping from MCP-server attribute names to
            block param names, for state the server tracks across
            tool calls (e.g., the most recent frame).  Read by
            tool-side helpers, ignored by the channel.
    """

    tool: str
    block: str
    args: dict[str, str] = field(default_factory=dict)
    server_state: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolBlockManifest:
    """A loaded tool-to-block manifest for one MCP server.

    Attributes:
        mcp_server: MCP server identifier (matches the value
            agents send in their ``caller.mcp_server`` field).
        tools: Tool bindings keyed by tool name.
        source: Origin file path for diagnostics.  Optional.
    """

    mcp_server: str
    tools: dict[str, ToolBinding] = field(default_factory=dict)
    source: Path | None = None

    def get(self, tool: str) -> ToolBinding | None:
        """Look up a binding by tool name; ``None`` if not declared."""
        return self.tools.get(tool)

    def __contains__(self, tool: str) -> bool:
        return tool in self.tools

    @classmethod
    def from_file(cls, path: Path) -> ToolBlockManifest:
        """Load a manifest from a YAML file.

        Args:
            path: Path to a manifest YAML file.

        Returns:
            The parsed manifest.

        Raises:
            ValueError: For any structural problem in the YAML.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            raise ValueError(f"Manifest file {path} is empty")
        if not isinstance(data, dict):
            raise ValueError(
                f"Manifest file {path} must contain a YAML mapping "
                f"at the top level, got {type(data).__name__}"
            )
        return _parse_manifest(data, source=path)


def _parse_manifest(
    data: dict, *, source: Path | None = None,
) -> ToolBlockManifest:
    """Parse a manifest dict into a :class:`ToolBlockManifest`."""
    if "mcp_server" not in data:
        raise ValueError(
            f"Manifest{(' ' + str(source)) if source else ''} is "
            f"missing required 'mcp_server' field"
        )
    mcp_server = data["mcp_server"]
    if not isinstance(mcp_server, str) or not mcp_server:
        raise ValueError(
            f"Manifest 'mcp_server' must be a non-empty string, "
            f"got {mcp_server!r}"
        )

    raw_tools = data.get("tools", {}) or {}
    if not isinstance(raw_tools, dict):
        raise ValueError(
            f"Manifest 'tools' must be a mapping, "
            f"got {type(raw_tools).__name__}"
        )

    tools: dict[str, ToolBinding] = {}
    for tool_name, raw in raw_tools.items():
        if not isinstance(raw, dict):
            raise ValueError(
                f"Manifest tool {tool_name!r}: entry must be a "
                f"mapping, got {type(raw).__name__}"
            )
        if "block" not in raw:
            raise ValueError(
                f"Manifest tool {tool_name!r}: missing required "
                f"'block' field"
            )
        block_name = raw["block"]
        if not isinstance(block_name, str) or not block_name:
            raise ValueError(
                f"Manifest tool {tool_name!r}: 'block' must be a "
                f"non-empty string, got {block_name!r}"
            )

        args = _parse_str_map(
            raw.get("args"), field="args", tool=tool_name)
        server_state = _parse_str_map(
            raw.get("server_state"),
            field="server_state",
            tool=tool_name,
        )

        tools[tool_name] = ToolBinding(
            tool=tool_name,
            block=block_name,
            args=args,
            server_state=server_state,
        )

    return ToolBlockManifest(
        mcp_server=mcp_server,
        tools=tools,
        source=source,
    )


def _parse_str_map(
    raw, *, field: str, tool: str,
) -> dict[str, str]:
    """Parse a ``{str: str}`` mapping with diagnostic context."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Manifest tool {tool!r}: {field!r} must be a mapping, "
            f"got {type(raw).__name__}"
        )
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(
                f"Manifest tool {tool!r}: {field!r} entries must "
                f"be string→string, got {k!r}={v!r}"
            )
        out[k] = v
    return out


def validate_against_registry(
    manifest: ToolBlockManifest,
    registry: BlockRegistry,
) -> None:
    """Verify every tool binding references a known block.

    Run after loading both the manifest and the project's block
    registry, before handing the manifest to the channel.  Raises
    :class:`ValueError` on the first unknown block reference so
    misconfigurations surface at startup instead of at first agent
    call.
    """
    for binding in manifest.tools.values():
        if binding.block not in registry:
            raise ValueError(
                f"Manifest binds tool {binding.tool!r} of MCP "
                f"server {manifest.mcp_server!r} to unknown block "
                f"{binding.block!r}.  Known blocks: "
                f"{registry.names()}"
            )


def load_manifests(
    files: list[Path],
    *,
    registry: BlockRegistry | None = None,
) -> list[ToolBlockManifest]:
    """Load multiple manifests; optionally validate each against a registry.

    Manifest files may live anywhere; flywheel projects conventionally
    keep them next to the MCP server module
    (e.g., ``cyberarc/mcp_servers/arc_tool_blocks.yaml``).  Callers
    decide where to look.  Two manifests for the same ``mcp_server``
    name are rejected.
    """
    manifests: list[ToolBlockManifest] = []
    seen: dict[str, Path | None] = {}
    for path in files:
        manifest = ToolBlockManifest.from_file(path)
        if manifest.mcp_server in seen:
            raise ValueError(
                f"Duplicate manifest for MCP server "
                f"{manifest.mcp_server!r}: "
                f"{seen[manifest.mcp_server]} and {path}"
            )
        seen[manifest.mcp_server] = path
        if registry is not None:
            validate_against_registry(manifest, registry)
        manifests.append(manifest)
    return manifests


def build_invocation_table(
    manifests: list[ToolBlockManifest],
) -> dict[tuple[str, str], str]:
    """Build a flat ``(mcp_server, tool) → block`` lookup table.

    This is the table the channel consults for each
    ``/execution/begin`` request.  Building it once at startup
    keeps per-request validation O(1).
    """
    table: dict[tuple[str, str], str] = {}
    for manifest in manifests:
        for binding in manifest.tools.values():
            table[(manifest.mcp_server, binding.tool)] = (
                binding.block)
    return table
