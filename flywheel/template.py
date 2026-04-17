"""Template definitions for flywheel workspaces.

A template declares the capabilities of a workspace: what artifacts
exist, what blocks can run, and their container images. Templates
are parsed from YAML files and validated at load time.

Block declarations live in ``workforce/blocks/<name>.yaml`` files
and are loaded into a :class:`BlockRegistry`.  The template's
``blocks:`` list contains string references to those blocks.  The
inline-block-definition syntax was removed in Phase 5 of the
block-execution refactor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

from flywheel.validation import validate_name as _validate_name

if TYPE_CHECKING:
    from flywheel.blocks.registry import BlockRegistry


@dataclass(frozen=True)
class ArtifactDeclaration:
    """A declared artifact in a template — name, storage kind, and git fields if applicable."""
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
            container block.  For ``inprocess`` blocks it is just
            documentation; the resolver returns the host path
            either way.
        optional: If ``True``, missing instances skip the slot
            silently rather than raising.
        derive_from: When set, declares this input as a *rollup*
            of another artifact.  At begin time the channel always
            rebuilds a fresh instance of ``name`` from all
            instances of ``derive_from`` before binding, using
            ``derive_kind`` to choose the rollup function.  This
            is the freshness mechanism for things like
            ``game_history`` rolled up from ``game_step``.  When
            ``derive_from`` is set, ``optional`` is ignored — the
            rollup is always materializable as long as at least
            one source instance exists.
        derive_kind: Which rollup function to apply.  Currently
            only ``"jsonl_concat"`` is supported (delegates to
            :meth:`Workspace.materialize_sequence`).  Required
            when ``derive_from`` is set.
    """

    name: str
    container_path: str
    optional: bool = False
    derive_from: str | None = None
    derive_kind: str | None = None


@dataclass(frozen=True)
class OutputSlot:
    """A declared output of a block — artifact name and container mount path."""

    name: str
    container_path: str


@dataclass(frozen=True)
class BlockImplementation:
    """How an ``inprocess`` or ``subprocess`` block's payload is dispatched.

    Required for non-``container`` runners. Names a Python module and
    a callable inside it that the runner imports and invokes with the
    block's :class:`ExecutionContext`.

    Attributes:
        python_module: Dotted Python module path
            (e.g., ``"cyberarc.blocks.predict_block"``).
        entry: Callable name within the module (default ``"run"``).
    """

    python_module: str
    entry: str = "run"


@dataclass(frozen=True)
class BlockDefinition:
    """A declared block in a template — image, Docker flags, and I/O slots.

    Attributes:
        name: Block identifier, unique per template.
        image: Docker image.  Required for ``runner == "container"``;
            forbidden for every other runner (Phase 5 retired the
            ``__record__`` sentinel and the empty-image fallback).
        inputs: Declared input artifact slots.
        outputs: Declared output artifact slots.
        docker_args: Extra docker run args for container blocks.
        env: Extra env vars for container blocks.
        runner: How the block is physically performed.  One of:

            - ``"container"`` (default): launched as a Docker
              container by ``ContainerExecutor``.  Requires ``image``.
            - ``"inprocess"``: dispatched by the channel into a
              Python entrypoint declared in ``implementation``.
              Reserved for future phases; not currently used.
            - ``"subprocess"``: dispatched as a local subprocess.
              Reserved for future phases.
            - ``"lifecycle"``: has no body of its own.  The block
              is invoked exclusively through the ``/execution/begin
              + /execution/end`` lifecycle API by an MCP tool.  The
              tool→block binding lives in a tool-block manifest;
              the manifest entry is what makes the block reachable.
              Use this for blocks whose body is the tool function
              itself (e.g., cyberarc's ``predict``, ``game_step``,
              ``exploration_request``, ``brainstorm_request``).
        runner_justification: Required free-text rationale when
            ``runner != "container"``. Forces the author to state
            why container isolation isn't appropriate.
        implementation: Where the runner finds the payload.
            Required for ``inprocess`` and ``subprocess``.
            Forbidden for ``container`` and ``lifecycle``.
    """

    name: str
    image: str = ""
    inputs: list[InputSlot] = field(default_factory=list)
    outputs: list[OutputSlot] = field(default_factory=list)
    docker_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    runner: Literal[
        "container", "inprocess", "subprocess", "lifecycle"
    ] = "container"
    runner_justification: str | None = None
    implementation: BlockImplementation | None = None


@dataclass(frozen=True)
class ServiceDependency:
    """A declared external service dependency.

    Documents that a workspace requires an external service to
    be running.  Flywheel does not start or manage the service,
    but records the dependency for documentation, validation,
    and future automation.

    Attributes:
        name: Service name.
        url_env: Environment variable name that blocks expect
            to contain the service URL.
        description: Human-readable description.
    """

    name: str
    url_env: str
    description: str = ""


@dataclass(frozen=True)
class Template:
    """A workspace template parsed from YAML. Validated at load time."""

    name: str
    artifacts: list[ArtifactDeclaration]
    blocks: list[BlockDefinition]
    services: list[ServiceDependency] = field(default_factory=list)

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
                    f"workforce/blocks/<name>.yaml file.  Inline "
                    f"block definitions were removed in Phase 5."
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

        services: list[ServiceDependency] = []
        for entry in data.get("services", []):
            svc_name = entry["name"]
            _validate_name(svc_name, "Service")
            services.append(ServiceDependency(
                name=svc_name,
                url_env=entry["url_env"],
                description=entry.get("description", ""),
            ))

        return cls(
            name=path.stem,
            artifacts=artifacts,
            blocks=blocks,
            services=services,
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


_SUPPORTED_DERIVE_KINDS = {"jsonl_concat"}


def _parse_input_slots(raw: list) -> list[InputSlot]:
    """Parse a list of raw input entries into InputSlot objects."""
    slots: list[InputSlot] = []
    for inp in raw:
        if isinstance(inp, str):
            slots.append(InputSlot(
                name=inp,
                container_path=f"/input/{inp}",
            ))
            continue

        ref = inp["name"]
        derive_from = inp.get("derive_from")
        derive_kind = inp.get("derive_kind")
        if derive_from is not None and derive_kind is None:
            raise ValueError(
                f"Input slot {ref!r}: 'derive_from' requires "
                f"'derive_kind' to also be set"
            )
        if derive_kind is not None and derive_from is None:
            raise ValueError(
                f"Input slot {ref!r}: 'derive_kind' requires "
                f"'derive_from' to also be set"
            )
        if (derive_kind is not None
                and derive_kind not in _SUPPORTED_DERIVE_KINDS):
            raise ValueError(
                f"Input slot {ref!r}: unsupported derive_kind "
                f"{derive_kind!r}; supported: "
                f"{sorted(_SUPPORTED_DERIVE_KINDS)}"
            )

        slots.append(InputSlot(
            name=ref,
            container_path=inp.get(
                "container_path", f"/input/{ref}"),
            optional=inp.get("optional", False),
            derive_from=derive_from,
            derive_kind=derive_kind,
        ))
    return slots


def _parse_output_slots(
    raw: list, *, block_name: str,
) -> list[OutputSlot]:
    """Parse a list of raw output entries into OutputSlot objects.

    Validates that no output name is duplicated within the block.
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
        else:
            ref = out["name"]
            slots.append(OutputSlot(
                name=ref,
                container_path=out.get(
                    "container_path", f"/output/{ref}"),
            ))
        if ref in seen:
            raise ValueError(
                f"Block {block_name!r} has duplicate output {ref!r}")
        seen.add(ref)
    return slots


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
        ValueError: For any structural problem in the block YAML.
    """
    if "name" not in entry:
        raise ValueError(
            f"Block definition {('from ' + source) if source else ''} "
            f"is missing required 'name' field"
        )
    name = entry["name"]
    _validate_name(name, "Block")

    runner = entry.get("runner", "container")
    valid_runners = (
        "container", "inprocess", "subprocess", "lifecycle")
    if runner not in valid_runners:
        raise ValueError(
            f"Block {name!r}: unknown runner {runner!r}; "
            f"expected one of {', '.join(valid_runners)}"
        )

    image = entry.get("image", "")
    runner_justification = entry.get("runner_justification")
    raw_impl = entry.get("implementation")

    if runner == "container":
        if not image:
            raise ValueError(
                f"Block {name!r}: runner 'container' requires "
                f"a non-empty 'image' field"
            )
        if raw_impl is not None:
            raise ValueError(
                f"Block {name!r}: runner 'container' must not "
                f"declare 'implementation'"
            )
    elif runner == "lifecycle":
        # Lifecycle blocks have no body of their own; the channel
        # never dispatches them.  The MCP-tool manifest binding is
        # the implementation, so neither image nor implementation
        # belongs on the block definition.
        if image:
            raise ValueError(
                f"Block {name!r}: runner 'lifecycle' must not "
                f"declare 'image' (got {image!r})"
            )
        if raw_impl is not None:
            raise ValueError(
                f"Block {name!r}: runner 'lifecycle' must not "
                f"declare 'implementation' (the manifest binding "
                f"to an MCP tool is the implementation)"
            )
        if not runner_justification:
            raise ValueError(
                f"Block {name!r}: runner 'lifecycle' requires "
                f"'runner_justification' (free-text rationale "
                f"for why the block has no container body)"
            )
    else:
        if image:
            raise ValueError(
                f"Block {name!r}: runner {runner!r} must not "
                f"declare 'image' (got {image!r})"
            )
        if not runner_justification:
            raise ValueError(
                f"Block {name!r}: runner {runner!r} requires "
                f"'runner_justification' (free-text rationale)"
            )
        if raw_impl is None:
            raise ValueError(
                f"Block {name!r}: runner {runner!r} requires "
                f"'implementation' (python_module + entry)"
            )

    implementation: BlockImplementation | None = None
    if raw_impl is not None:
        if not isinstance(raw_impl, dict):
            raise ValueError(
                f"Block {name!r}: 'implementation' must be a "
                f"mapping, got {type(raw_impl).__name__}"
            )
        if "python_module" not in raw_impl:
            raise ValueError(
                f"Block {name!r}: 'implementation' is missing "
                f"required 'python_module' field"
            )
        implementation = BlockImplementation(
            python_module=raw_impl["python_module"],
            entry=raw_impl.get("entry", "run"),
        )

    return BlockDefinition(
        name=name,
        image=image,
        inputs=_parse_input_slots(entry.get("inputs", [])),
        outputs=_parse_output_slots(
            entry.get("outputs", []), block_name=name),
        docker_args=entry.get("docker_args", []),
        env=entry.get("env", {}),
        runner=runner,
        runner_justification=runner_justification,
        implementation=implementation,
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
    for slot in block.outputs:
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


def check_service_dependencies(template: Template) -> list[str]:
    """Return warnings for unset service URL environment variables.

    Checks each service dependency declared in the template and
    returns a warning string for any whose ``url_env`` is not set
    in the current environment.

    Args:
        template: The template to check.

    Returns:
        List of warning strings (empty if all services are available).
    """
    warnings: list[str] = []
    for svc in template.services:
        if not os.environ.get(svc.url_env):
            warnings.append(
                f"Service {svc.name!r} expects {svc.url_env} "
                f"but it is not set"
            )
    return warnings
