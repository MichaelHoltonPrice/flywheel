"""Template definitions for flywheel workspaces.

A template declares the capabilities of a workspace: what artifacts
exist, what blocks can run, and their container images. Templates
are parsed from YAML files and validated at load time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from flywheel.validation import validate_name as _validate_name


@dataclass(frozen=True)
class ArtifactDeclaration:
    """A declared artifact in a template — name, storage kind, and git fields if applicable."""
    name: str
    kind: Literal["copy", "git"]
    repo: str | None = None  # git only, relative to project root
    path: str | None = None  # git only


@dataclass(frozen=True)
class InputSlot:
    """A declared input to a block — artifact name, container mount path, and optionality."""

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
    """A declared block in a template — image, Docker flags, and I/O slots."""

    name: str
    image: str
    inputs: list[InputSlot]
    outputs: list[OutputSlot]
    docker_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


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
    def from_yaml(cls, path: Path) -> Template:
        """Load a template from a YAML file.

        Args:
            path: Path to the YAML template file. The file stem becomes
                the template name.

        Returns:
            The parsed Template.

        Raises:
            ValueError: If artifact/block names are invalid, kinds are unknown,
                names are duplicated, block I/O references undeclared artifacts,
                or block outputs reference git artifacts.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        artifact_names: set[str] = set()
        artifact_kinds: dict[str, str] = {}
        artifacts: list[ArtifactDeclaration] = []
        for entry in data.get("artifacts", []):
            kind = entry["kind"]
            name = entry["name"]
            _validate_name(name, "Artifact")
            repo = entry.get("repo")
            path_field = entry.get("path")

            if kind not in ("copy", "git"):
                raise ValueError(f"Artifact {name!r} has unknown kind {kind!r}")

            if name in artifact_names:
                raise ValueError(f"Duplicate artifact name {name!r}")

            if kind == "git":
                if repo is None:
                    raise ValueError(f"Git artifact {name!r} requires 'repo' field")
                if path_field is None:
                    raise ValueError(f"Git artifact {name!r} requires 'path' field")

            artifacts.append(ArtifactDeclaration(
                name=name,
                kind=kind,
                repo=repo,
                path=path_field,
            ))
            artifact_names.add(name)
            artifact_kinds[name] = kind

        block_names: set[str] = set()
        blocks: list[BlockDefinition] = []
        for entry in data.get("blocks", []):
            _validate_name(entry["name"], "Block")
            if entry["name"] in block_names:
                raise ValueError(f"Duplicate block name {entry['name']!r}")
            block_names.add(entry["name"])
            raw_inputs = entry.get("inputs", [])
            raw_outputs = entry.get("outputs", [])

            input_slots: list[InputSlot] = []
            for inp in raw_inputs:
                if isinstance(inp, str):
                    ref = inp
                    input_slots.append(InputSlot(
                        name=ref,
                        container_path=f"/input/{ref}",
                    ))
                else:
                    ref = inp["name"]
                    input_slots.append(InputSlot(
                        name=ref,
                        container_path=inp.get(
                            "container_path", f"/input/{ref}"
                        ),
                        optional=inp.get("optional", False),
                    ))
                if ref not in artifact_names:
                    raise ValueError(
                        f"Block {entry['name']!r} input {ref!r} "
                        f"not declared in artifacts"
                    )

            output_names: set[str] = set()
            output_slots: list[OutputSlot] = []
            for out in raw_outputs:
                if isinstance(out, str):
                    ref = out
                    output_slots.append(OutputSlot(
                        name=ref,
                        container_path=f"/output/{ref}",
                    ))
                else:
                    ref = out["name"]
                    output_slots.append(OutputSlot(
                        name=ref,
                        container_path=out.get(
                            "container_path", f"/output/{ref}"
                        ),
                    ))
                if ref in output_names:
                    raise ValueError(
                        f"Block {entry['name']!r} has duplicate "
                        f"output {ref!r}"
                    )
                output_names.add(ref)
                if ref not in artifact_names:
                    raise ValueError(
                        f"Block {entry['name']!r} output {ref!r} "
                        f"not declared in artifacts"
                    )
                if artifact_kinds[ref] == "git":
                    raise ValueError(
                        f"Block {entry['name']!r} output {ref!r} "
                        f"is a git artifact and cannot be a block output"
                    )

            blocks.append(BlockDefinition(
                name=entry["name"],
                image=entry["image"],
                inputs=input_slots,
                outputs=output_slots,
                docker_args=entry.get("docker_args", []),
                env=entry.get("env", {}),
            ))

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
