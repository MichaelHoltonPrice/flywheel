"""Template definitions for flywheel workspaces.

A template declares the capabilities of a workspace: what artifacts
exist, what blocks can run, and their container images. Templates
are parsed from YAML files and validated at load time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

_VALID_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def _validate_name(name: str, label: str) -> None:
    """Validate that a name is safe for use as an identifier."""
    if not name:
        raise ValueError(f"{label} name must not be empty")
    if not _VALID_NAME.match(name):
        raise ValueError(
            f"{label} name {name!r} is invalid. "
            f"Use only letters, digits, hyphens, and underscores."
        )


@dataclass(frozen=True)
class ArtifactDeclaration:
    """A declared artifact in a template — name, storage kind, and git fields if applicable."""
    name: str
    kind: Literal["copy", "git"]
    repo: str | None = None  # git only, relative to project root
    path: str | None = None  # git only


@dataclass(frozen=True)
class BlockDefinition:
    """A declared block in a template — name, container image, and artifact references."""

    name: str
    image: str
    inputs: list[str]  # artifact names
    outputs: list[str]  # artifact names


@dataclass(frozen=True)
class Template:
    """A workspace template parsed from YAML. Validated at load time."""

    name: str
    artifacts: list[ArtifactDeclaration]
    blocks: list[BlockDefinition]

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

        blocks: list[BlockDefinition] = []
        for entry in data.get("blocks", []):
            _validate_name(entry["name"], "Block")
            inputs = entry.get("inputs", [])
            outputs = entry.get("outputs", [])

            for ref in inputs:
                if ref not in artifact_names:
                    raise ValueError(
                        f"Block {entry['name']!r} input {ref!r} "
                        f"not declared in artifacts"
                    )
            for ref in outputs:
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
                inputs=inputs,
                outputs=outputs,
            ))

        return cls(name=path.stem, artifacts=artifacts, blocks=blocks)
