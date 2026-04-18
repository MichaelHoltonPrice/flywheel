"""First-class declarative agent patterns.

A *pattern* describes the network topology and timing of an agent
run: which roles exist, how many of each fire on a given trigger,
and what artifacts they consume / produce.  Patterns are authored
in YAML alongside templates and block definitions, and consumed
by :mod:`flywheel.pattern_runner` (added in P2 of the patterns
campaign).

This module covers parsing and validation only — there is no
runner here.  Validation happens at load time so a malformed
pattern fails the CLI immediately rather than mid-run.

Conceptual shape::

    name: play-brainstorm
    description: One play agent, six brainstormers every 20 actions.
    roles:
      play:
        prompt: workforce/prompts/arc_predict_play.md
        model: claude-sonnet-4-6
        cardinality: 1
        trigger:
          kind: continuous
        inputs: [predictor, mechanics_summary]
        outputs: [game_log]
      brainstorm:
        prompt: workforce/prompts/arc_brainstorm.md
        model: claude-sonnet-4-6
        cardinality: 6
        trigger:
          kind: every_n_executions
          of_block: take_action
          n: 20
        inputs: [game_history, mechanics_summary]
        outputs: [brainstorm_result]

Triggers are a small declarative vocabulary intentionally kept
narrow at the start of the campaign:

- ``continuous`` — fire once at run start; the role's lifetime is
  the lifetime of the run.  Used for the persistent play agent.
- ``every_n_executions`` — fire every N successful executions of
  a referenced block.  Used for periodic cohorts (brainstormers
  after every 20 ``take_action`` rows).
- ``on_request`` — fire when an agent invokes a named tool.  The
  tool string is intentionally not validated against any registry
  here; the pattern runner is the layer that knows about MCP
  surfaces.  Used for fan-out patterns like
  ``request_brainstorm`` requesting a cohort on demand.
- ``on_event`` — fire when a workspace event of the named type
  is recorded (e.g., ``surprise``).  Reserved for future use;
  parses today so patterns can declare the dependency.

Add new trigger kinds sparingly and only when an existing pattern
demands it — every kind grows the runner's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

from flywheel.validation import validate_name as _validate_name

if TYPE_CHECKING:
    from flywheel.blocks.registry import BlockRegistry


# Frozen so a Pattern can be safely shared between threads (the
# runner reads it from multiple worker contexts).  All trigger
# kinds carry their ``kind`` discriminator so callers can use
# ``match`` / ``isinstance`` either way.


@dataclass(frozen=True)
class ContinuousTrigger:
    """Fire the role exactly once at run start.

    The role's lifetime is the run's lifetime; when the role
    exits the runner treats that as a normal completion (subject
    to the pattern's termination policy, which currently lives
    in :mod:`flywheel.pattern_runner`).
    """

    kind: Literal["continuous"] = "continuous"


@dataclass(frozen=True)
class EveryNExecutionsTrigger:
    """Fire every ``n`` successful executions of ``of_block``.

    The runner consults the workspace ledger; a row counts only
    once (``status == "succeeded"`` and not synthetic).  ``n``
    must be a positive integer.  The block name must resolve
    against the project's :class:`BlockRegistry`.
    """

    of_block: str
    n: int
    kind: Literal["every_n_executions"] = "every_n_executions"


@dataclass(frozen=True)
class OnRequestTrigger:
    """Fire when an agent invokes a named tool.

    Used for fan-out patterns (e.g., ``request_brainstorm``).
    The tool name is *not* validated here — the pattern runner
    is the layer that knows which MCP tools are wired up — so
    a typo will surface the first time the trigger is consulted.
    """

    tool: str
    kind: Literal["on_request"] = "on_request"


@dataclass(frozen=True)
class OnEventTrigger:
    """Fire when a workspace event of ``event`` type is recorded.

    Reserved for future use (the surprise / mismatch events that
    today are emitted ad hoc by cyberarc hooks).  Parses today
    so patterns can declare the dependency without waiting for
    the runner to grow event support.
    """

    event: str
    kind: Literal["on_event"] = "on_event"


Trigger = (
    ContinuousTrigger
    | EveryNExecutionsTrigger
    | OnRequestTrigger
    | OnEventTrigger
)


@dataclass(frozen=True)
class Role:
    """A role in a pattern: one logical kind of agent.

    A role launches one or more identical agent instances when
    its trigger fires (``cardinality`` agents per firing).  All
    instances see the same prompt and model; per-instance
    differentiation is the runner's responsibility (e.g., a
    cohort index in the input artifacts).

    Attributes:
        name: Role identifier, unique within the pattern.
        prompt: Project-relative path to the system-prompt file
            (read as text by the runner; no templating here).
        model: Model identifier passed through to the agent
            launcher (e.g., ``"claude-sonnet-4-6"``).  None lets
            the runner / project default decide.
        cardinality: How many parallel agents to launch per
            firing.  Must be ``>= 1``.  Defaults to ``1``.
        trigger: When the role fires.  Exactly one trigger per
            role; complex schedules are expressed by adding more
            roles, not by combining triggers.
        inputs: Names of artifacts the agent reads.  Validated
            against the workspace template at runner-launch time
            (not at pattern-load time, since a single pattern
            may be reused across templates).
        outputs: Names of artifacts the agent is expected to
            register.  Same scoping as ``inputs``.
        mcp_servers: Comma-separated MCP server names enabled
            for this role's agent (passed through to the
            launcher).  None means the launcher's default.
        allowed_tools: Comma-separated tool whitelist (passed
            through unchanged).
        max_turns: Per-agent turn cap; ``None`` defers to the
            launcher default.
        total_timeout: Per-agent wall-clock cap in seconds;
            ``None`` defers to the launcher default.
    """

    name: str
    prompt: str
    trigger: Trigger
    model: str | None = None
    cardinality: int = 1
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    mcp_servers: str | None = None
    allowed_tools: str | None = None
    max_turns: int | None = None
    total_timeout: int | None = None


@dataclass(frozen=True)
class Pattern:
    """A parsed agent pattern.

    Attributes:
        name: Pattern identifier (the YAML file's stem).
        roles: The pattern's roles.  Order is the YAML order;
            the runner is free to launch them in any order, but
            tests and docs benefit from determinism.
        description: Optional human-readable description (kept
            verbatim from YAML).
    """

    name: str
    roles: list[Role]
    description: str = ""

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        block_registry: "BlockRegistry | None" = None,
    ) -> "Pattern":
        """Load and validate a pattern from a YAML file.

        Args:
            path: Path to the YAML pattern file.  The file stem
                becomes the pattern name (so the CLI can
                discover patterns by their filename, mirroring
                templates).
            block_registry: Optional block registry used to
                validate ``every_n_executions`` triggers.  When
                ``None``, block-name validation is skipped (so
                this loader can be unit-tested without a full
                project layout); supplying it is recommended in
                production.

        Returns:
            The parsed :class:`Pattern`.

        Raises:
            ValueError: For any malformed entry — missing
                required field, unknown trigger kind, role-name
                duplicate, ``cardinality`` non-positive,
                ``every_n_executions`` referencing an unknown
                block, etc.  The CLI surfaces these directly.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Pattern file {path} is empty")
        if not isinstance(data, dict):
            raise ValueError(
                f"Pattern file {path} must contain a YAML "
                f"mapping at the top level, got "
                f"{type(data).__name__}"
            )

        name = path.stem
        _validate_name(name, "Pattern")

        description = data.get("description", "")
        if not isinstance(description, str):
            raise ValueError(
                f"Pattern {name!r}: 'description' must be a "
                f"string, got {type(description).__name__}"
            )

        roles_raw = data.get("roles")
        if not isinstance(roles_raw, dict) or not roles_raw:
            raise ValueError(
                f"Pattern {name!r}: 'roles' must be a non-empty "
                f"mapping of role-name to role-spec"
            )

        seen: set[str] = set()
        roles: list[Role] = []
        for role_name, role_spec in roles_raw.items():
            if role_name in seen:
                raise ValueError(
                    f"Pattern {name!r}: duplicate role "
                    f"{role_name!r}"
                )
            seen.add(role_name)
            roles.append(
                _parse_role(
                    role_name,
                    role_spec,
                    pattern_name=name,
                    block_registry=block_registry,
                )
            )

        return cls(name=name, roles=roles, description=description)


def _parse_role(
    role_name: str,
    spec: object,
    *,
    pattern_name: str,
    block_registry: "BlockRegistry | None",
) -> Role:
    """Parse one entry of the ``roles:`` mapping into a :class:`Role`.

    Lives at module scope (rather than as a Pattern method) so
    tests can exercise it in isolation; not part of the public
    surface.
    """
    _validate_name(role_name, "Role")
    if not isinstance(spec, dict):
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"spec must be a mapping, got "
            f"{type(spec).__name__}"
        )

    prompt = spec.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"'prompt' must be a non-empty string"
        )

    cardinality = spec.get("cardinality", 1)
    if not isinstance(cardinality, int) or cardinality < 1:
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"'cardinality' must be a positive integer, got "
            f"{cardinality!r}"
        )

    trigger_raw = spec.get("trigger")
    if not isinstance(trigger_raw, dict) or "kind" not in trigger_raw:
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"'trigger' must be a mapping with a 'kind' field"
        )
    trigger = _parse_trigger(
        trigger_raw,
        role_name=role_name,
        pattern_name=pattern_name,
        block_registry=block_registry,
    )

    model = spec.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"'model' must be a string or omitted"
        )

    inputs = _parse_string_list(
        spec.get("inputs", []),
        role_name=role_name,
        pattern_name=pattern_name,
        field_name="inputs",
    )
    outputs = _parse_string_list(
        spec.get("outputs", []),
        role_name=role_name,
        pattern_name=pattern_name,
        field_name="outputs",
    )

    mcp_servers = spec.get("mcp_servers")
    if mcp_servers is not None and not isinstance(mcp_servers, str):
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"'mcp_servers' must be a comma-separated string or "
            f"omitted"
        )

    allowed_tools = spec.get("allowed_tools")
    if allowed_tools is not None and not isinstance(
            allowed_tools, str):
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"'allowed_tools' must be a comma-separated string "
            f"or omitted"
        )

    max_turns = spec.get("max_turns")
    if max_turns is not None and (
            not isinstance(max_turns, int) or max_turns < 1):
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"'max_turns' must be a positive integer or omitted"
        )

    total_timeout = spec.get("total_timeout")
    if total_timeout is not None and (
            not isinstance(total_timeout, int) or total_timeout < 1):
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"'total_timeout' must be a positive integer or "
            f"omitted"
        )

    return Role(
        name=role_name,
        prompt=prompt,
        trigger=trigger,
        model=model,
        cardinality=cardinality,
        inputs=inputs,
        outputs=outputs,
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
        max_turns=max_turns,
        total_timeout=total_timeout,
    )


def _parse_string_list(
    raw: object,
    *,
    role_name: str,
    pattern_name: str,
    field_name: str,
) -> list[str]:
    """Coerce a YAML list-of-strings, raising on any non-string entry."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"{field_name!r} must be a list of strings"
        )
    out: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            raise ValueError(
                f"Pattern {pattern_name!r} role {role_name!r}: "
                f"{field_name!r} entries must be strings, got "
                f"{type(entry).__name__}"
            )
        out.append(entry)
    return out


def _parse_trigger(
    raw: dict,
    *,
    role_name: str,
    pattern_name: str,
    block_registry: "BlockRegistry | None",
) -> Trigger:
    """Parse one ``trigger:`` mapping into a concrete trigger dataclass.

    Each ``kind`` is required to declare exactly the fields its
    dataclass expects; surplus fields are tolerated for forward
    compatibility (so old runners can ignore options new runners
    add) but missing required fields raise.
    """
    kind = raw["kind"]
    if not isinstance(kind, str):
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"trigger 'kind' must be a string, got "
            f"{type(kind).__name__}"
        )

    if kind == "continuous":
        return ContinuousTrigger()

    if kind == "every_n_executions":
        of_block = raw.get("of_block")
        n = raw.get("n")
        if not isinstance(of_block, str) or not of_block:
            raise ValueError(
                f"Pattern {pattern_name!r} role {role_name!r}: "
                f"every_n_executions trigger requires "
                f"'of_block' (string)"
            )
        if not isinstance(n, int) or n < 1:
            raise ValueError(
                f"Pattern {pattern_name!r} role {role_name!r}: "
                f"every_n_executions trigger requires 'n' "
                f"(positive integer)"
            )
        if (block_registry is not None
                and of_block not in block_registry):
            raise ValueError(
                f"Pattern {pattern_name!r} role {role_name!r}: "
                f"every_n_executions trigger references unknown "
                f"block {of_block!r}.  Known blocks: "
                f"{sorted(block_registry.names())}"
            )
        return EveryNExecutionsTrigger(of_block=of_block, n=n)

    if kind == "on_request":
        tool = raw.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError(
                f"Pattern {pattern_name!r} role {role_name!r}: "
                f"on_request trigger requires 'tool' (string)"
            )
        return OnRequestTrigger(tool=tool)

    if kind == "on_event":
        event = raw.get("event")
        if not isinstance(event, str) or not event:
            raise ValueError(
                f"Pattern {pattern_name!r} role {role_name!r}: "
                f"on_event trigger requires 'event' (string)"
            )
        return OnEventTrigger(event=event)

    raise ValueError(
        f"Pattern {pattern_name!r} role {role_name!r}: unknown "
        f"trigger kind {kind!r}.  Known kinds: continuous, "
        f"every_n_executions, on_request, on_event"
    )


def discover_patterns(directory: Path) -> dict[str, Path]:
    """Index ``*.yaml`` files in ``directory`` by their stem.

    Mirrors :meth:`flywheel.blocks.registry.BlockRegistry.from_directory`
    in spirit: returns an empty dict when the directory does not
    exist or contains no patterns, so the CLI can call this
    unconditionally.  Files starting with ``_`` are skipped to
    let authors stash drafts.

    The dict is intentionally lazy — patterns are not parsed
    here; the CLI parses on demand once the user picks one,
    which keeps ``flywheel run --help`` cheap and lets a single
    broken pattern not break the whole listing.
    """
    if not directory.exists() or not directory.is_dir():
        return {}
    out: dict[str, Path] = {}
    for path in sorted(directory.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        out[path.stem] = path
    return out
