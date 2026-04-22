"""First-class declarative agent patterns.

A *pattern* describes the network topology and timing of an agent
run: which roles exist, how many of each fire on a given trigger,
and what artifacts they consume / produce.  Patterns are authored
in YAML alongside templates and block definitions, and consumed
by :mod:`flywheel.pattern_runner`.

This module covers parsing and validation only — there is no
runner here.  Validation happens at load time so a malformed
pattern fails the CLI immediately rather than mid-run.

Conceptual shape::

    name: play-brainstorm
    description: One play agent, six brainstormers every 20 actions.
    roles:
      play:
        prompt: workforce/prompts/arc_predict_play.md
        cardinality: 1
        trigger:
          kind: continuous
        inputs: [predictor, mechanics_summary]
        outputs: [game_log]
        overrides:
          model: claude-sonnet-4-6
      brainstorm:
        prompt: workforce/prompts/arc_brainstorm.md
        cardinality: 6
        trigger:
          kind: every_n_executions
          of_block: take_action
          n: 20
        inputs: [game_history, mechanics_summary]
        outputs: [brainstorm_result]
        overrides:
          model: claude-sonnet-4-6

Per-role / per-instance executor knobs (``model``,
``mcp_servers``, ``allowed_tools``, ``max_turns``,
``total_timeout``, ...) live under the free-form
``overrides:`` mapping; the runner forwards them verbatim into
the executor's ``overrides`` dict and each executor reads what
it cares about.  The schema deliberately does not type these
fields so adding a new battery-specific knob does not require
a substrate change.

Triggers are a small declarative vocabulary intentionally kept
narrow:

- ``continuous`` — fire once at run start; the role's lifetime is
  the lifetime of the run.  Used for the persistent play agent.
- ``every_n_executions`` — fire every N successful executions of
  a referenced block.  Used for periodic cohorts (brainstormers
  after every 20 ``take_action`` executions).
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
from typing import TYPE_CHECKING, Any, Literal

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

    The runner consults the workspace ledger; an execution counts only
    once (``status == "succeeded"`` and not synthetic).  ``n``
    must be a positive integer.  The block name must resolve
    against the project's :class:`BlockRegistry`.

    Attributes:
        of_block: The block whose successful executions drive
            the trigger.
        n: Cadence.
        pause: Names of other instances to pause while the
            cohort fires.  Empty (the default) keeps the
            historical behaviour — the cohort runs in parallel
            with every other live instance.  When non-empty the
            runner stops each named instance before firing the
            cohort, waits for them to exit cleanly (so ``/state/``
            is captured), runs the cohort to completion, then
            relaunches each paused instance with
            ``predecessor_id`` set to its stopped execution so
            the session chain continues.  Useful when the cohort
            should see a quiescent workspace snapshot and the
            paused instance should resume with the cohort's
            outputs materialised.  Instance names are validated
            at pattern-parse time against the other instances in
            the same pattern.
    """

    of_block: str
    n: int
    kind: Literal["every_n_executions"] = "every_n_executions"
    pause: tuple[str, ...] = ()


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


@dataclass(frozen=True)
class OnToolTrigger:
    """Fire when a named instance invokes a named MCP tool.

    Declares handoff dispatch at pattern scope: ``instance``
    names another launchable instance in the same pattern
    (typically the continuous agent driving the run); ``tool``
    is the fully qualified MCP tool whose invocation is meant
    to trigger one execution of this instance's block.

    Load-bearing-ness caveat: the runner parses and validates
    this trigger, but runtime dispatch for matching tool calls
    still lives in whatever ``BlockRunner`` the caller's
    ``launch_fn`` supplies (today, the project-hooks tool
    router).  A follow-up slice moves that routing into
    :class:`flywheel.pattern_runner.PatternRunner` so the
    pattern's declaration becomes authoritative.  Until then,
    patterns must still arrange for the host-side router to
    cover every tool named in an ``on_tool`` trigger.
    """

    instance: str
    tool: str
    kind: Literal["on_tool"] = "on_tool"


@dataclass(frozen=True)
class AutorestartTrigger:
    """Fire at run start and re-fire when the instance exits.

    Like :class:`ContinuousTrigger` in that the role fires once
    at run start, but the runner relaunches the role whenever
    its single handle finishes — unless a ``scope="run"`` halt
    has been queued by a post-check, in which case the run ends
    normally.  Cardinality must be ``1`` for this trigger.

    Use when the role should drive the run until an
    out-of-role signal (terminal game state, post-check halt)
    tells the runner to stop — not when the agent decides to
    exit the conversation.  Typical home is a "play" role whose
    pattern relies on a state post-check (e.g.,
    ``GAME_OVER``/``WIN`` on the underlying engine block) rather
    than on the agent's own judgment to end the run.
    """

    kind: Literal["autorestart"] = "autorestart"


Trigger = (
    ContinuousTrigger
    | EveryNExecutionsTrigger
    | OnRequestTrigger
    | OnEventTrigger
    | OnToolTrigger
    | AutorestartTrigger
)


@dataclass(frozen=True)
class Role:
    """A role in a pattern: one logical kind of launchable block.

    A role launches one or more identical block instances when
    its trigger fires (``cardinality`` per firing).  Per-instance
    differentiation is the runner's responsibility (e.g., a
    cohort index in the input artifacts).

    Attributes:
        name: Role identifier, unique within the pattern.
        prompt: Project-relative path to a prompt file the
            runner reads as text and forwards via
            ``overrides["prompt"]``.  Empty string means the
            role does not carry a prompt (typical for non-agent
            blocks such as a ``workspace_persistent`` container);
            in that case the runner does not set the override
            and the executor falls back to its own defaults.
        cardinality: How many parallel launches per firing.
            Must be ``>= 1``.  Defaults to ``1``.
        trigger: When the role fires.  Exactly one trigger per
            role; complex schedules are expressed by adding more
            roles, not by combining triggers.
        inputs: Names of artifacts the role reads at launch
            time.  Validated against the workspace template at
            runner-launch time (not at pattern-load time, since
            a single pattern may be reused across templates).
        outputs: Names of artifacts the role is expected to
            register.  Same scoping as ``inputs``.
        overrides: Free-form per-launch knobs forwarded into the
            executor's ``overrides`` dict.  Keys are
            executor-specific (the agent battery honors
            ``model``, ``max_turns``, ``total_timeout``,
            ``mcp_servers``, ``allowed_tools``, ...; container
            executors honor whatever they document; unknown
            keys are silently ignored per the
            :class:`flywheel.executor.BlockExecutor` protocol).
            The runner merges this on top of the run-level
            defaults bag and below the protocol-level keys it
            sets itself (``prompt``, ``predecessor_id``).
        extra_env: Per-role environment variables passed to the
            executor as the ``extra_env`` launch kwarg per the
            substrate's container-extras convention.  Use
            sparingly; most env vars belong in project hooks.
        block_name: Name of the block this role launches.
            ``None`` means ``block_name == name`` (the legacy
            ``roles:`` grammar's implicit 1:1 mapping);
            synthesized roles produced from ``instances:``
            specs set this explicitly so the runner launches
            the right block under the instance's name.

    Roles that need step-by-step history list the canonical
    incremental artifact (e.g., ``game_history``) in their
    :attr:`inputs`; the in-process recorder appends to it in
    real time and per-mount input staging re-reads the latest
    snapshot at every relaunch.
    """

    name: str
    prompt: str
    trigger: Trigger
    cardinality: int = 1
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)
    extra_env: dict[str, str] = field(default_factory=dict)
    block_name: str | None = None


@dataclass(frozen=True)
class BlockInstance:
    """One instance of a block inside a pattern.

    Patterns declare topology as a mapping of named instances,
    each referencing a block definition by name and carrying a
    trigger describing when it fires.  Per-launch knobs the
    block's executor honors (model, MCP server selection,
    runtime caps, ...) live in the free-form :attr:`overrides`
    map; the runner merges them into the per-launch overrides
    dict and the executor reads what it cares about.

    Attributes:
        name: Instance identifier, unique within the pattern.
        block: Name of the block this instance launches.  Must
            resolve against the project's :class:`BlockRegistry`.
        trigger: When the instance fires.  For non-agent blocks
            (e.g., a ``workspace_persistent`` ``ExecuteAction``
            container), ``on_tool`` is the usual choice.
        cardinality: How many parallel launches per trigger
            firing.  Must be ``>= 1``.  Defaults to ``1``.
        prompt: Project-relative path to a prompt file; ``None``
            when the block is not prompt-driven.  Read as text
            by the runner and forwarded via
            ``overrides["prompt"]``.
        inputs: Artifact names the instance reads at launch
            time.  Resolved against the workspace on fire.
        outputs: Artifact names the instance is expected to
            register.  Informational today; the block's own
            declared outputs are the substrate source of truth.
        overrides: Free-form per-launch knobs forwarded into
            the executor's ``overrides`` dict; see :class:`Role`
            for the full contract.
        extra_env: Per-instance environment variables passed to
            the executor as the ``extra_env`` launch kwarg per
            the substrate's container-extras convention.
    """

    name: str
    block: str
    trigger: Trigger
    cardinality: int = 1
    prompt: str | None = None
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Pattern:
    """A parsed agent pattern.

    ``instances`` is the canonical topology list the pattern
    runner consumes; it is always populated.  ``roles`` is a
    backward-compat view that is populated only when the YAML
    used the legacy ``roles:`` grammar — callers who still
    iterate it keep working during the transition.  The
    ``Pattern.from_yaml`` loader synthesises an equivalent
    ``BlockInstance`` for every legacy role (with
    ``block == role.name`` per the implicit 1:1 mapping),
    so consumers that iterate ``instances`` see both grammars.

    Attributes:
        name: Pattern identifier (the YAML file's stem).
        roles: Legacy-grammar view.  Empty for ``instances:``
            patterns; populated from ``roles:`` YAML for
            backward compatibility.
        instances: Canonical topology list.  Always populated,
            regardless of which YAML grammar the pattern used.
        description: Optional human-readable description (kept
            verbatim from YAML).
    """

    name: str
    roles: list[Role]
    description: str = ""
    instances: list[BlockInstance] = field(default_factory=list)

    def iter_instances(self) -> list[BlockInstance]:
        """Return the pattern's canonical instances.

        When ``instances`` is already populated (the common case
        — ``Pattern.from_yaml`` always fills it for both
        grammars), that list is returned.  When a caller has
        constructed a :class:`Pattern` directly with only
        ``roles=[...]`` (tests, legacy code), this synthesises
        equivalent :class:`BlockInstance` objects on demand so
        the pattern runner sees one shape either way.
        """
        if self.instances:
            return self.instances
        return [_instance_from_role(r) for r in self.roles]

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
        instances_raw = data.get("instances")

        if instances_raw is not None and roles_raw is not None:
            raise ValueError(
                f"Pattern {name!r}: declare either 'roles' "
                f"(legacy) or 'instances' (current), not both"
            )
        if instances_raw is None and roles_raw is None:
            raise ValueError(
                f"Pattern {name!r}: must declare either "
                f"'instances' (current grammar) or 'roles' "
                f"(legacy grammar)"
            )

        roles: list[Role] = []
        instances: list[BlockInstance] = []

        if instances_raw is not None:
            if (not isinstance(instances_raw, dict)
                    or not instances_raw):
                raise ValueError(
                    f"Pattern {name!r}: 'instances' must be a "
                    f"non-empty mapping of instance-name to "
                    f"instance-spec"
                )
            seen: set[str] = set()
            for inst_name, inst_spec in instances_raw.items():
                if inst_name in seen:
                    raise ValueError(
                        f"Pattern {name!r}: duplicate instance "
                        f"{inst_name!r}"
                    )
                seen.add(inst_name)
                instances.append(
                    _parse_instance(
                        inst_name,
                        inst_spec,
                        pattern_name=name,
                        block_registry=block_registry,
                    )
                )
            _validate_on_tool_cross_refs(
                instances, pattern_name=name)
            _validate_pause_cross_refs(
                instances, pattern_name=name)
        else:
            if (not isinstance(roles_raw, dict)
                    or not roles_raw):
                raise ValueError(
                    f"Pattern {name!r}: 'roles' must be a "
                    f"non-empty mapping of role-name to "
                    f"role-spec"
                )
            seen = set()
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
            # Populate the canonical ``instances`` list by
            # synthesising one :class:`BlockInstance` per
            # legacy role: role name becomes the block name
            # (legacy 1:1 mapping) and the agent-specific
            # fields carry across verbatim.
            instances = [
                _instance_from_role(r) for r in roles]
            _validate_pause_cross_refs(
                instances, pattern_name=name)

        return cls(
            name=name,
            roles=roles,
            description=description,
            instances=instances,
        )


_BATTERY_OVERRIDE_KEYS = (
    "model",
    "max_turns",
    "total_timeout",
    "mcp_servers",
    "allowed_tools",
)


# Keys the runner itself owns inside the per-launch ``overrides``
# dict (``prompt`` / ``prompt_substitutions``: read from disk and
# layered in by ``PatternRunner._build_overrides``;
# ``predecessor_id``: managed by the runner for paused-instance
# session chaining) or that ride the explicit container-extras
# launch-kwarg seam (``extra_env`` / ``extra_mounts``).  Authoring
# them under a pattern's ``overrides:`` map either has no effect
# (the runner-owned keys are clobbered or unconditionally set
# per launch) or routes a value through the wrong seam
# (container extras would never reach the executor's declared
# kwarg).  Reject them at parse time so the YAML cannot quietly
# disagree with the runner about who owns the key.
_RESERVED_OVERRIDE_KEYS = (
    "prompt",
    "prompt_substitutions",
    "predecessor_id",
    "extra_env",
    "extra_mounts",
)


def _reject_legacy_battery_keys(
    spec: dict,
    *,
    pattern_name: str,
    role_name: str,
    scope: str,
) -> None:
    """Refuse top-level battery-shaped keys that now live in ``overrides:``.

    These keys used to be typed fields on :class:`Role` and
    :class:`BlockInstance`; they have moved into the free-form
    :attr:`Role.overrides` map so the schema stays
    executor-agnostic.  Catching them at parse time turns a
    silent no-op (the runner would never see them) into a
    pointed error explaining the migration.
    """
    offenders = [k for k in _BATTERY_OVERRIDE_KEYS if k in spec]
    if not offenders:
        return
    listed = ", ".join(repr(k) for k in offenders)
    raise ValueError(
        f"Pattern {pattern_name!r} {scope} {role_name!r}: "
        f"top-level key(s) {listed} are no longer supported "
        f"on a pattern {scope}.  Move them under an "
        f"``overrides:`` mapping so the runner forwards them "
        f"to the executor as opaque per-launch overrides "
        f"(executors read what they understand and ignore the "
        f"rest)."
    )


def _parse_overrides(
    spec: dict,
    *,
    pattern_name: str,
    role_name: str,
    scope: str,
) -> dict[str, Any]:
    """Parse the optional ``overrides:`` mapping.

    The mapping is forwarded verbatim to the executor (no value
    coercion here — the executor owns its key contract).  Only
    the outer shape is validated: a mapping with non-empty
    string keys.
    """
    raw = spec.get("overrides", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern {pattern_name!r} {scope} {role_name!r}: "
            f"'overrides' must be a mapping of override-name to "
            f"value, got {type(raw).__name__}"
        )
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not k:
            raise ValueError(
                f"Pattern {pattern_name!r} {scope} {role_name!r}: "
                f"'overrides' keys must be non-empty strings"
            )
        if k in _RESERVED_OVERRIDE_KEYS:
            raise ValueError(
                f"Pattern {pattern_name!r} {scope} "
                f"{role_name!r}: 'overrides.{k}' is reserved.  "
                f"'prompt' / 'prompt_substitutions' / "
                f"'predecessor_id' are set by the pattern runner "
                f"itself; 'extra_env' / 'extra_mounts' ride the "
                f"explicit container-extras launch kwargs (use the "
                f"top-level ``extra_env:`` field for env vars).  "
                f"Authoring them under ``overrides:`` would either "
                f"be silently overwritten or routed through the "
                f"wrong seam."
            )
        out[k] = v
    return out


def _parse_extra_env(
    spec: dict,
    *,
    pattern_name: str,
    role_name: str,
    scope: str,
) -> dict[str, str]:
    """Parse the optional ``extra_env:`` mapping (string→string)."""
    raw = spec.get("extra_env", {})
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern {pattern_name!r} {scope} {role_name!r}: "
            f"'extra_env' must be a mapping of name to string "
            f"value, got {type(raw).__name__}"
        )
    out: dict[str, str] = {}
    for env_name, env_value in raw.items():
        if not isinstance(env_name, str) or not env_name:
            raise ValueError(
                f"Pattern {pattern_name!r} {scope} {role_name!r}: "
                f"'extra_env' keys must be non-empty strings"
            )
        if not isinstance(env_value, str):
            raise ValueError(
                f"Pattern {pattern_name!r} {scope} {role_name!r}: "
                f"'extra_env.{env_name}' must be a string, got "
                f"{type(env_value).__name__}"
            )
        out[env_name] = env_value
    return out


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

    _reject_legacy_battery_keys(
        spec,
        pattern_name=pattern_name,
        role_name=role_name,
        scope="role",
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

    overrides = _parse_overrides(
        spec,
        pattern_name=pattern_name,
        role_name=role_name,
        scope="role",
    )
    extra_env = _parse_extra_env(
        spec,
        pattern_name=pattern_name,
        role_name=role_name,
        scope="role",
    )

    if "materialize" in spec:
        raise ValueError(
            f"Pattern {pattern_name!r} role {role_name!r}: "
            f"'materialize' is not supported.  Add the canonical "
            f"incremental artifact name (e.g., 'game_history') "
            f"to 'inputs' instead — the in-process recorder "
            f"appends to it live and the per-mount input stager "
            f"re-reads the latest snapshot every relaunch."
        )

    return Role(
        name=role_name,
        prompt=prompt,
        trigger=trigger,
        cardinality=cardinality,
        inputs=inputs,
        outputs=outputs,
        overrides=overrides,
        extra_env=extra_env,
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

    if kind == "autorestart":
        return AutorestartTrigger()

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
        pause_raw = raw.get("pause", [])
        if not isinstance(pause_raw, list):
            raise ValueError(
                f"Pattern {pattern_name!r} role {role_name!r}: "
                f"every_n_executions trigger 'pause' must be a "
                f"list of instance names, got "
                f"{type(pause_raw).__name__}"
            )
        for entry in pause_raw:
            if not isinstance(entry, str) or not entry:
                raise ValueError(
                    f"Pattern {pattern_name!r} role "
                    f"{role_name!r}: every_n_executions trigger "
                    f"'pause' entries must be non-empty strings; "
                    f"got {entry!r}"
                )
        return EveryNExecutionsTrigger(
            of_block=of_block,
            n=n,
            pause=tuple(pause_raw),
        )

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

    if kind == "on_tool":
        instance = raw.get("instance")
        tool = raw.get("tool")
        if not isinstance(instance, str) or not instance:
            raise ValueError(
                f"Pattern {pattern_name!r} role {role_name!r}: "
                f"on_tool trigger requires 'instance' (string)"
            )
        if not isinstance(tool, str) or not tool:
            raise ValueError(
                f"Pattern {pattern_name!r} role {role_name!r}: "
                f"on_tool trigger requires 'tool' (string)"
            )
        return OnToolTrigger(instance=instance, tool=tool)

    raise ValueError(
        f"Pattern {pattern_name!r} role {role_name!r}: unknown "
        f"trigger kind {kind!r}.  Known kinds: continuous, "
        f"autorestart, every_n_executions, on_request, "
        f"on_event, on_tool"
    )


def _parse_instance(
    inst_name: str,
    spec: object,
    *,
    pattern_name: str,
    block_registry: "BlockRegistry | None",
) -> BlockInstance:
    """Parse one entry of the ``instances:`` mapping.

    Instances reference a block by name and carry a trigger.
    Per-launch executor knobs (model, MCP server selection,
    runtime caps, ...) live in the free-form ``overrides:``
    map; the runner forwards them verbatim into the executor's
    ``overrides`` dict.
    """
    _validate_name(inst_name, "Instance")
    if not isinstance(spec, dict):
        raise ValueError(
            f"Pattern {pattern_name!r} instance "
            f"{inst_name!r}: spec must be a mapping, got "
            f"{type(spec).__name__}"
        )

    _reject_legacy_battery_keys(
        spec,
        pattern_name=pattern_name,
        role_name=inst_name,
        scope="instance",
    )

    block = spec.get("block")
    if not isinstance(block, str) or not block:
        raise ValueError(
            f"Pattern {pattern_name!r} instance "
            f"{inst_name!r}: 'block' must be a non-empty "
            f"string (name of a declared block)"
        )
    if (block_registry is not None
            and block not in block_registry):
        raise ValueError(
            f"Pattern {pattern_name!r} instance "
            f"{inst_name!r}: references unknown block "
            f"{block!r}.  Known blocks: "
            f"{sorted(block_registry.names())}"
        )

    trigger_raw = spec.get("trigger")
    if (not isinstance(trigger_raw, dict)
            or "kind" not in trigger_raw):
        raise ValueError(
            f"Pattern {pattern_name!r} instance "
            f"{inst_name!r}: 'trigger' must be a mapping with "
            f"a 'kind' field"
        )
    trigger = _parse_trigger(
        trigger_raw,
        role_name=inst_name,
        pattern_name=pattern_name,
        block_registry=block_registry,
    )

    cardinality = spec.get("cardinality", 1)
    if not isinstance(cardinality, int) or cardinality < 1:
        raise ValueError(
            f"Pattern {pattern_name!r} instance "
            f"{inst_name!r}: 'cardinality' must be a "
            f"positive integer, got {cardinality!r}"
        )

    prompt = spec.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError(
            f"Pattern {pattern_name!r} instance "
            f"{inst_name!r}: 'prompt' must be a string or "
            f"omitted"
        )

    inputs = _parse_string_list(
        spec.get("inputs", []),
        role_name=inst_name,
        pattern_name=pattern_name,
        field_name="inputs",
    )
    outputs = _parse_string_list(
        spec.get("outputs", []),
        role_name=inst_name,
        pattern_name=pattern_name,
        field_name="outputs",
    )

    overrides = _parse_overrides(
        spec,
        pattern_name=pattern_name,
        role_name=inst_name,
        scope="instance",
    )
    extra_env = _parse_extra_env(
        spec,
        pattern_name=pattern_name,
        role_name=inst_name,
        scope="instance",
    )

    return BlockInstance(
        name=inst_name,
        block=block,
        trigger=trigger,
        cardinality=cardinality,
        prompt=prompt,
        inputs=inputs,
        outputs=outputs,
        overrides=overrides,
        extra_env=extra_env,
    )


def _instance_from_role(role: Role) -> BlockInstance:
    """Build a :class:`BlockInstance` matching a legacy role.

    Legacy ``roles:`` grammar declares role-name == block-name
    implicitly.  The loader builds the canonical ``instances``
    list by turning each role into a one-for-one instance so
    the pattern runner can read a single shape regardless of
    which grammar the YAML used.  Agent-specific fields carry
    across verbatim.
    """
    return BlockInstance(
        name=role.name,
        block=role.name,
        trigger=role.trigger,
        cardinality=role.cardinality,
        prompt=role.prompt,
        inputs=list(role.inputs),
        outputs=list(role.outputs),
        overrides=dict(role.overrides),
        extra_env=dict(role.extra_env),
    )


def _validate_on_tool_cross_refs(
    instances: list[BlockInstance], *, pattern_name: str,
) -> None:
    """Cross-ref checks for every ``on_tool`` trigger.

    1. The referenced instance name must exist in this pattern.
    2. The referenced instance's trigger must be one the runner
       actually launches (``continuous`` or
       ``every_n_executions``) — dispatching a tool call to an
       instance that itself only fires on a tool call has no
       launchable caller and is almost certainly a typo.  Caught
       at load time so the author sees a clear error rather than
       a silent never-fires-at-runtime.
    """
    by_name: dict[str, BlockInstance] = {
        inst.name: inst for inst in instances}
    for inst in instances:
        if not isinstance(inst.trigger, OnToolTrigger):
            continue
        target_name = inst.trigger.instance
        if target_name not in by_name:
            raise ValueError(
                f"Pattern {pattern_name!r} instance "
                f"{inst.name!r}: on_tool trigger references "
                f"unknown instance {target_name!r}.  Known "
                f"instances: {sorted(by_name)}"
            )
        target = by_name[target_name]
        if not isinstance(
            target.trigger,
            (
                ContinuousTrigger,
                AutorestartTrigger,
                EveryNExecutionsTrigger,
            ),
        ):
            raise ValueError(
                f"Pattern {pattern_name!r} instance "
                f"{inst.name!r}: on_tool trigger references "
                f"{target_name!r}, whose trigger kind "
                f"{target.trigger.kind!r} cannot be a caller "
                f"(on_tool dispatch requires a launchable "
                f"source — ``continuous``, ``autorestart``, "
                f"or ``every_n_executions``)"
            )


def _validate_pause_cross_refs(
    instances: list[BlockInstance], *, pattern_name: str,
) -> None:
    """Cross-ref checks for every ``every_n_executions`` pause list.

    Each name in ``pause`` must refer to another instance in the
    same pattern.  Pausing an instance that doesn't exist is a
    silent no-op at runtime; catching it at load time turns a
    typo into a clear error.
    """
    by_name = {inst.name for inst in instances}
    for inst in instances:
        trigger = inst.trigger
        if not isinstance(trigger, EveryNExecutionsTrigger):
            continue
        for target in trigger.pause:
            if target not in by_name:
                raise ValueError(
                    f"Pattern {pattern_name!r} instance "
                    f"{inst.name!r}: trigger 'pause' references "
                    f"unknown instance {target!r}.  Known "
                    f"instances: {sorted(by_name)}"
                )
            if target == inst.name:
                raise ValueError(
                    f"Pattern {pattern_name!r} instance "
                    f"{inst.name!r}: trigger 'pause' cannot "
                    f"reference the cohort itself"
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
