"""Template definitions for flywheel workspaces.

A template declares the capabilities of a workspace: what artifacts
exist, what blocks can run, and their container images. Templates
are parsed from YAML files and validated at load time.

Block declarations live in ``workforce/blocks/<name>.yaml`` files
and are loaded into a :class:`BlockRegistry`.  The template's
``blocks:`` list contains string references to those blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

from flywheel.validation import validate_name as _validate_name

if TYPE_CHECKING:
    from flywheel.blocks.registry import BlockRegistry


@dataclass(frozen=True)
class ArtifactDeclaration:
    """A declared artifact in a template — name, storage kind, and git fields if applicable.

    Two kinds:

    * ``copy`` — directory of arbitrary files, written once per
      instance.  Each block execution that emits one creates a fresh
      instance.
    * ``git`` — reference to a path within a git repo at a specific
      commit.  Resolved at workspace creation time.
    """
    name: str
    kind: Literal["copy", "git"]
    repo: str | None = None  # git only, relative to project root
    path: str | None = None  # git only


@dataclass(frozen=True)
class InputSlot:
    """A declared input to a block.

    Attributes:
        name: Artifact declaration name this slot consumes.  At
            ``/execution/begin`` the channel resolves this to the
            latest registered instance of that name.
        container_path: Where the artifact is mounted inside a
            container block.  For ``lifecycle`` blocks it is just
            documentation; the resolver returns the host path
            either way.
        optional: If ``True``, missing instances skip the slot
            silently rather than raising.
    """

    name: str
    container_path: str
    optional: bool = False


@dataclass(frozen=True)
class OutputSlot:
    """A declared output of a block — artifact name and container mount path."""

    name: str
    container_path: str


@dataclass(frozen=True)
class BlockDefinition:
    """A declared block in a template — image, Docker flags, and I/O slots.

    Attributes:
        name: Block identifier, unique per template.
        image: Docker image.  Required for ``runner == "container"``;
            forbidden for ``runner == "lifecycle"``.
        inputs: Declared input artifact slots.
        outputs: Declared output artifact slots.
        docker_args: Extra docker run args for container blocks.
        env: Extra env vars for container blocks.
        runner: How the block is physically performed.  One of:

            - ``"container"`` (default): launched as a Docker
              container by :class:`flywheel.executor.ProcessExitExecutor`
              (one-shot) or :class:`flywheel.executor.RequestResponseExecutor`
              (persistent, when that lands).  Requires ``image``.
            - ``"lifecycle"``: has no body of its own.  The block
              is invoked exclusively through the ``/execution/begin
              + /execution/end`` lifecycle API by an MCP tool.  The
              tool→block binding lives in a tool-block manifest;
              the manifest entry is what makes the block reachable.
        runner_justification: Required free-text rationale when
            ``runner != "container"``.  Forces the author to state
            why container isolation isn't appropriate.
        post_check: Optional dotted Python path to a callable
            invoked by
            :class:`flywheel.local_block.LocalBlockRecorder` after
            the execution record is finalized.  See :mod:`flywheel.post_check`.
            ``None`` means no post-execution check is configured
            for this block.  Resolved eagerly at registry-load
            time (a typo fails startup, not the run).
        output_builder: Optional dotted Python path to a callable
            invoked on the host after the container exits and
            before flywheel's standard artifact collection reads
            the per-execution output directories.  The callback
            reads whatever the container wrote (raw, human-
            readable, agent-authored) and writes the canonical
            artifact files in place.  See
            :mod:`flywheel.output_builder`.  ``None`` means the
            standard "whatever is in the output dir becomes the
            artifact" contract applies.  Resolved eagerly at
            registry-load time.
        lifecycle: Container-lifetime model.  One of:

            - ``"one_shot"``: container lifetime equals one
              execution.
            - ``"workspace_persistent"``: container lifetime
              equals the workspace's lifetime.

            The block definition only declares intent.  How
            executions discover, start, or attach to a container
            is executor policy.  State — whether a block has
            internal state that must survive across executions —
            is a separate concern, not implied by either value.
        state: Whether the block has private memory that must
            survive across executions.  When ``True``, flywheel
            mounts a ``/state/`` directory in the container,
            populates it at start with the prior successful
            execution's captured state, and captures its
            contents on clean exit.  See
            ``cyber-root/substrate-contract.md``.  Mutually
            exclusive with ``lifecycle: workspace_persistent``
            (that lifecycle preserves state in-memory instead).
        stop_timeout_s: Seconds to wait after writing the
            cooperative stop sentinel before escalating to
            SIGTERM (and then SIGKILL after a short grace).
            Block authors whose containers poll the sentinel
            can set this to whatever cadence matches their
            cleanup window; containers that ignore the sentinel
            will always be forcibly stopped after this many
            seconds plus the TERM grace.  Default 30s.
    """

    name: str
    image: str = ""
    inputs: list[InputSlot] = field(default_factory=list)
    outputs: dict[str, list[OutputSlot]] = field(default_factory=dict)
    """Output slots grouped by termination reason.

    Keys are project-defined termination-reason labels (e.g.,
    ``"normal"``, ``"defer"``); flywheel ascribes no semantics to
    them.  Each value is the ordered list of output slots the
    block is expected to produce when it terminates with that
    reason.  Substrate-reserved reasons
    (:data:`flywheel.runtime.RESERVED_TERMINATION_REASONS`) map
    implicitly to the empty output set and are not enumerated
    here.

    The legacy YAML shape ``outputs: [<flat list>]`` parses as
    ``{DEFAULT_TERMINATION_REASON: <flat list>}`` so single-reason
    blocks need no migration.  See
    :func:`_parse_output_groups`.
    """
    docker_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    runner: Literal["container", "lifecycle"] = "container"
    runner_justification: str | None = None
    post_check: str | None = None
    output_builder: str | None = None
    lifecycle: Literal["one_shot", "workspace_persistent"] = "one_shot"
    state: bool = False
    stop_timeout_s: int = 30

    def all_output_slots(self) -> list[OutputSlot]:
        """Return every distinct output slot across all termination reasons.

        A block may declare the same slot under multiple
        termination reasons (e.g., a ``turn_summary`` slot
        produced under both ``normal`` and ``defer``); the
        returned list contains each slot name exactly once, in
        first-encounter order across termination reasons.

        Used by callers that need the union of slots — proposal
        directory pre-allocation, mount construction, template
        validation — without caring which termination reason
        each slot belongs to.
        """
        seen: set[str] = set()
        union: list[OutputSlot] = []
        for slots in self.outputs.values():
            for slot in slots:
                if slot.name in seen:
                    continue
                seen.add(slot.name)
                union.append(slot)
        return union

    def outputs_for(
        self, termination_reason: str,
    ) -> list[OutputSlot]:
        """Return the output slots expected for a termination reason.

        Empty list for substrate-reserved reasons (which always
        map to the empty output set) and for unknown
        project-defined reasons (the substrate normalizes those
        to ``protocol_violation`` before commit, but we still
        return ``[]`` defensively here).
        """
        return list(self.outputs.get(termination_reason, []))


@dataclass(frozen=True)
class Template:
    """A workspace template parsed from YAML. Validated at load time."""

    name: str
    artifacts: list[ArtifactDeclaration]
    blocks: list[BlockDefinition]

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        block_registry: BlockRegistry | None = None,
    ) -> Template:
        """Load a template from a YAML file.

        Args:
            path: Path to the YAML template file. The file stem
                becomes the template name.
            block_registry: Optional registry of pre-loaded block
                definitions.  When supplied, string entries in the
                template's ``blocks:`` list are resolved against
                this registry.  When ``None``, only inline-dict
                block entries are supported.

        Returns:
            The parsed Template.

        Raises:
            ValueError: If artifact/block names are invalid, kinds
                are unknown, names are duplicated, block I/O
                references undeclared artifacts, block outputs
                reference git artifacts, or a string block
                reference is unresolved by the registry.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        artifacts, artifact_names, artifact_kinds = _parse_artifacts(
            data.get("artifacts", []))

        block_names: set[str] = set()
        blocks: list[BlockDefinition] = []
        for entry in data.get("blocks", []):
            if not isinstance(entry, str):
                raise ValueError(
                    f"Template {path.name!r} block entry has "
                    f"unsupported type {type(entry).__name__!r}; "
                    f"templates must reference blocks by name "
                    f"(string).  Define each block in its own "
                    f"workforce/blocks/<name>.yaml file."
                )
            if block_registry is None:
                raise ValueError(
                    f"Template {path.name!r} references block "
                    f"{entry!r} by name, but no block_registry "
                    f"was provided to Template.from_yaml"
                )
            if entry not in block_registry:
                raise ValueError(
                    f"Template {path.name!r} references unknown "
                    f"block {entry!r}.  Known blocks: "
                    f"{sorted(block_registry.names())}"
                )
            block = block_registry.get(entry)

            if block.name in block_names:
                raise ValueError(
                    f"Duplicate block name {block.name!r} in "
                    f"template {path.name!r}"
                )
            block_names.add(block.name)
            _validate_block_against_artifacts(
                block, artifact_names, artifact_kinds, path.name)
            blocks.append(block)

        return cls(
            name=path.stem,
            artifacts=artifacts,
            blocks=blocks,
        )


def _parse_artifacts(
    raw: list[dict],
) -> tuple[list[ArtifactDeclaration], set[str], dict[str, str]]:
    """Parse the ``artifacts:`` list of a template.

    Returns the parsed declarations plus name→kind index data used
    by block validation.
    """
    artifact_names: set[str] = set()
    artifact_kinds: dict[str, str] = {}
    artifacts: list[ArtifactDeclaration] = []
    for entry in raw:
        kind = entry["kind"]
        name = entry["name"]
        _validate_name(name, "Artifact")
        repo = entry.get("repo")
        path_field = entry.get("path")

        if kind not in ("copy", "git"):
            raise ValueError(
                f"Artifact {name!r} has unknown kind {kind!r}")

        if name in artifact_names:
            raise ValueError(f"Duplicate artifact name {name!r}")

        if kind == "git":
            if repo is None:
                raise ValueError(
                    f"Git artifact {name!r} requires 'repo' field")
            if path_field is None:
                raise ValueError(
                    f"Git artifact {name!r} requires 'path' field")

        artifacts.append(ArtifactDeclaration(
            name=name,
            kind=kind,
            repo=repo,
            path=path_field,
        ))
        artifact_names.add(name)
        artifact_kinds[name] = kind

    return artifacts, artifact_names, artifact_kinds


# Keys accepted inside a nested input-slot mapping.  Unknown
# keys raise, so typos like ``container_pat`` or ``optionl`` are
# surfaced instead of silently defaulted.
_INPUT_SLOT_KEYS: frozenset[str] = frozenset({
    "name",
    "container_path",
    "optional",
})

# Keys accepted inside a nested output-slot mapping.
_OUTPUT_SLOT_KEYS: frozenset[str] = frozenset({
    "name",
    "container_path",
})


def _parse_input_slots(raw: list) -> list[InputSlot]:
    """Parse a list of raw input entries into InputSlot objects.

    Each entry is either a string (bare artifact name) or a
    mapping with a known set of keys; unknown keys raise so
    typos surface.
    """
    slots: list[InputSlot] = []
    for inp in raw:
        if isinstance(inp, str):
            slots.append(InputSlot(
                name=inp,
                container_path=f"/input/{inp}",
            ))
            continue

        ref = inp["name"]
        if "derive_from" in inp or "derive_kind" in inp:
            raise ValueError(
                f"Input slot {ref!r}: 'derive_from' / "
                f"'derive_kind' are not supported."
            )
        unknown = set(inp) - _INPUT_SLOT_KEYS
        if unknown:
            raise ValueError(
                f"Input slot {ref!r}: unknown key(s) "
                f"{sorted(unknown)!r}; valid keys are "
                f"{sorted(_INPUT_SLOT_KEYS)!r}"
            )

        slots.append(InputSlot(
            name=ref,
            container_path=inp.get(
                "container_path", f"/input/{ref}"),
            optional=inp.get("optional", False),
        ))
    return slots


def _parse_output_slot_list(
    raw: list, *, block_name: str, group_label: str,
) -> list[OutputSlot]:
    """Parse a flat list of raw output entries into OutputSlot objects.

    Helper for :func:`_parse_output_groups`.  Each entry is
    either a string (bare artifact name) or a mapping with a
    known set of keys.  Slot names must be unique within the
    list (a block may repeat a slot across termination reasons,
    but never within a single reason).
    """
    slots: list[OutputSlot] = []
    seen: set[str] = set()
    for out in raw:
        if isinstance(out, str):
            ref = out
            slots.append(OutputSlot(
                name=ref,
                container_path=f"/output/{ref}",
            ))
        elif isinstance(out, dict):
            ref = out["name"]
            unknown = set(out) - _OUTPUT_SLOT_KEYS
            if unknown:
                raise ValueError(
                    f"Output slot {ref!r}: unknown key(s) "
                    f"{sorted(unknown)!r}; valid keys are "
                    f"{sorted(_OUTPUT_SLOT_KEYS)!r}"
                )
            slots.append(OutputSlot(
                name=ref,
                container_path=out.get(
                    "container_path", f"/output/{ref}"),
            ))
        else:
            raise ValueError(
                f"Block {block_name!r} output entry under "
                f"{group_label!r} must be a string or mapping; "
                f"got {type(out).__name__}"
            )
        if ref in seen:
            raise ValueError(
                f"Block {block_name!r} has duplicate output "
                f"{ref!r} under termination reason "
                f"{group_label!r}")
        seen.add(ref)
    return slots


def _parse_output_groups(
    raw: object, *, block_name: str,
) -> dict[str, list[OutputSlot]]:
    """Parse a block's ``outputs:`` declaration into the grouped form.

    Two YAML shapes are accepted:

    * **Flat list** (legacy / single-termination-reason blocks)::

        outputs:
          - name: result
            container_path: /scratch/result

      Parses to ``{DEFAULT_TERMINATION_REASON: [<slots>]}`` —
      the slots all live under the single conventional
      ``"normal"`` reason.

    * **Mapping keyed by termination reason** (multi-reason
      blocks)::

        outputs:
          normal:
            - name: turn_summary
              container_path: /scratch/turn_summary
          defer:
            - name: action
              container_path: /scratch/action

      Each key is a project-defined termination-reason label.
      Substrate-reserved labels
      (:data:`flywheel.runtime.RESERVED_TERMINATION_REASONS`)
      are rejected — declarations never enumerate them; they
      implicitly map to the empty output set at commit time.

    A block with no outputs (``outputs:`` omitted, ``[]``, or
    ``{}``) parses to ``{DEFAULT_TERMINATION_REASON: []}`` — every
    block declares at least one termination reason, defaulting to
    the conventional ``"normal"`` reason with an empty output set.
    """
    from flywheel.runtime import (
        DEFAULT_TERMINATION_REASON,
        RESERVED_TERMINATION_REASONS,
    )

    if raw is None or raw == [] or raw == {}:
        return {DEFAULT_TERMINATION_REASON: []}
    if isinstance(raw, list):
        slots = _parse_output_slot_list(
            raw, block_name=block_name,
            group_label=DEFAULT_TERMINATION_REASON,
        )
        return {DEFAULT_TERMINATION_REASON: slots}
    if isinstance(raw, dict):
        groups: dict[str, list[OutputSlot]] = {}
        for reason, entries in raw.items():
            if not isinstance(reason, str):
                raise ValueError(
                    f"Block {block_name!r} outputs key "
                    f"{reason!r} must be a string termination "
                    f"reason label"
                )
            if reason in RESERVED_TERMINATION_REASONS:
                raise ValueError(
                    f"Block {block_name!r}: termination reason "
                    f"{reason!r} is reserved by the substrate "
                    f"and must not appear in declarations; "
                    f"reserved reasons map implicitly to the "
                    f"empty output set"
                )
            if not isinstance(entries, list):
                raise ValueError(
                    f"Block {block_name!r} outputs[{reason!r}] "
                    f"must be a list, got "
                    f"{type(entries).__name__}"
                )
            groups[reason] = _parse_output_slot_list(
                entries, block_name=block_name,
                group_label=reason,
            )
        return groups
    raise ValueError(
        f"Block {block_name!r}: 'outputs' must be a list "
        f"(legacy single-reason form) or a mapping keyed by "
        f"termination reason, got {type(raw).__name__}"
    )


# Top-level keys accepted in a per-block YAML (or inline block
# definition).  ``parse_block_definition`` rejects anything
# outside this set so typos — e.g., ``lifecylce`` for
# ``lifecycle`` — surface as errors instead of silently
# disappearing into the parser.
_BLOCK_YAML_KEYS: frozenset[str] = frozenset({
    "name",
    "runner",
    "image",
    "runner_justification",
    "inputs",
    "outputs",
    "docker_args",
    "env",
    "post_check",
    "output_builder",
    "lifecycle",
    "state",
    "stop_timeout_s",
})

_BLOCK_LIFECYCLES: tuple[str, ...] = ("one_shot", "workspace_persistent")


def parse_block_definition(
    entry: dict, *, source: str | None = None,
) -> BlockDefinition:
    """Parse a single block-definition mapping.

    Used by both the inline-template path (where ``entry`` is one
    item from a template's ``blocks:`` list) and by the
    :class:`BlockRegistry` (where ``entry`` is the entire YAML body
    of a per-block file).

    Args:
        entry: The block definition as a dict.
        source: Origin description for error messages
            (e.g., a file path).  Optional.

    Returns:
        A validated :class:`BlockDefinition`.  Artifact-vs-template
        validation is *not* performed here; that happens when the
        block is bound to a template via
        :meth:`Template.from_yaml`.

    Raises:
        ValueError: For any structural problem in the block YAML,
            including unknown top-level keys.
    """
    if "name" not in entry:
        raise ValueError(
            f"Block definition {('from ' + source) if source else ''} "
            f"is missing required 'name' field"
        )
    name = entry["name"]
    _validate_name(name, "Block")

    # Reject unknown top-level keys before reading any field, so a
    # typo like ``lifecylce`` produces an error naming the offending
    # key rather than being silently dropped.
    unknown = set(entry) - _BLOCK_YAML_KEYS
    if unknown:
        raise ValueError(
            f"Block {name!r}: unknown top-level key(s) "
            f"{sorted(unknown)!r}; valid keys are "
            f"{sorted(_BLOCK_YAML_KEYS)!r}"
        )

    runner = entry.get("runner", "container")
    valid_runners = ("container", "lifecycle")
    if runner not in valid_runners:
        raise ValueError(
            f"Block {name!r}: unknown runner {runner!r}; "
            f"expected one of {', '.join(valid_runners)}"
        )

    image = entry.get("image", "")
    runner_justification = entry.get("runner_justification")

    if runner == "container":
        if not image:
            raise ValueError(
                f"Block {name!r}: runner 'container' requires "
                f"a non-empty 'image' field"
            )
    elif runner == "lifecycle":
        # Lifecycle blocks have no body of their own; the channel
        # never dispatches them.  The MCP-tool manifest binding is
        # the implementation, so an image doesn't belong here.
        if image:
            raise ValueError(
                f"Block {name!r}: runner 'lifecycle' must not "
                f"declare 'image' (got {image!r})"
            )
        if not runner_justification:
            raise ValueError(
                f"Block {name!r}: runner 'lifecycle' requires "
                f"'runner_justification' (free-text rationale "
                f"for why the block has no container body)"
            )

    post_check = entry.get("post_check")
    if post_check is not None and not isinstance(post_check, str):
        raise ValueError(
            f"Block {name!r}: 'post_check' must be a dotted "
            f"Python path string (got {type(post_check).__name__})"
        )

    output_builder = entry.get("output_builder")
    if (output_builder is not None
            and not isinstance(output_builder, str)):
        raise ValueError(
            f"Block {name!r}: 'output_builder' must be a dotted "
            f"Python path string (got "
            f"{type(output_builder).__name__})"
        )
    if output_builder is not None and runner != "container":
        raise ValueError(
            f"Block {name!r}: 'output_builder' is only valid for "
            f"runner 'container' (got {runner!r}).  Lifecycle "
            f"blocks write their outputs directly via the local "
            f"recorder; no post-execution host-side rebuild step "
            f"is needed or supported."
        )

    # ``lifecycle`` is only meaningful for container blocks.
    # Reject the key entirely on non-container runners, even when
    # its value is the default ``one_shot`` — the point is to flag
    # semantically nonsensical configs, not just wrong values.
    if "lifecycle" in entry and runner != "container":
        raise ValueError(
            f"Block {name!r}: 'lifecycle' is only valid for "
            f"runner 'container' (got {runner!r})"
        )
    lifecycle = entry.get("lifecycle", "one_shot")
    if lifecycle not in _BLOCK_LIFECYCLES:
        raise ValueError(
            f"Block {name!r}: unknown lifecycle {lifecycle!r}; "
            f"expected one of {', '.join(_BLOCK_LIFECYCLES)}"
        )

    state = entry.get("state", False)
    if not isinstance(state, bool):
        raise ValueError(
            f"Block {name!r}: 'state' must be a boolean "
            f"(got {type(state).__name__})"
        )
    if state and runner != "container":
        raise ValueError(
            f"Block {name!r}: 'state: true' is only valid for "
            f"runner 'container' (got {runner!r})"
        )
    # workspace_persistent preserves state in-memory; having a
    # `/state/` mount too would address the same concern twice.
    if state and lifecycle == "workspace_persistent":
        raise ValueError(
            f"Block {name!r}: 'state: true' is mutually exclusive "
            f"with 'lifecycle: workspace_persistent' (persistent "
            f"containers preserve state in-memory, not on disk)"
        )

    # stop_timeout_s: seconds to wait for cooperative stop before
    # escalating to TERM/KILL.  Only applies to container
    # runners.  A bool check separately because isinstance(x, int)
    # is True for booleans in Python.
    stop_timeout_s = entry.get("stop_timeout_s", 30)
    if isinstance(stop_timeout_s, bool) or not isinstance(
            stop_timeout_s, int):
        raise ValueError(
            f"Block {name!r}: 'stop_timeout_s' must be a "
            f"non-negative integer (got "
            f"{type(stop_timeout_s).__name__})"
        )
    if stop_timeout_s < 0:
        raise ValueError(
            f"Block {name!r}: 'stop_timeout_s' must be "
            f"non-negative (got {stop_timeout_s})"
        )
    if "stop_timeout_s" in entry and runner != "container":
        raise ValueError(
            f"Block {name!r}: 'stop_timeout_s' is only valid "
            f"for runner 'container' (got {runner!r})"
        )

    return BlockDefinition(
        name=name,
        image=image,
        inputs=_parse_input_slots(entry.get("inputs", [])),
        outputs=_parse_output_groups(
            entry.get("outputs"), block_name=name),
        docker_args=entry.get("docker_args", []),
        env=entry.get("env", {}),
        runner=runner,
        runner_justification=runner_justification,
        post_check=post_check,
        output_builder=output_builder,
        lifecycle=lifecycle,
        state=state,
        stop_timeout_s=stop_timeout_s,
    )


def _validate_block_against_artifacts(
    block: BlockDefinition,
    artifact_names: set[str],
    artifact_kinds: dict[str, str],
    template_label: str,
) -> None:
    """Verify every block I/O slot references a declared artifact."""
    for slot in block.inputs:
        if slot.name not in artifact_names:
            raise ValueError(
                f"Block {block.name!r} input {slot.name!r} not "
                f"declared in artifacts of template "
                f"{template_label!r}"
            )
    for slot in block.all_output_slots():
        if slot.name not in artifact_names:
            raise ValueError(
                f"Block {block.name!r} output {slot.name!r} not "
                f"declared in artifacts of template "
                f"{template_label!r}"
            )
        if artifact_kinds[slot.name] == "git":
            raise ValueError(
                f"Block {block.name!r} output {slot.name!r} is a "
                f"git artifact and cannot be a block output"
            )


