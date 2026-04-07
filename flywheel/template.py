from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class ArtifactDeclaration:
    name: str
    kind: Literal["copy", "git"]
    repo: str | None = None  # git only, relative to project root
    path: str | None = None  # git only


@dataclass(frozen=True)
class BlockDefinition:
    name: str
    image: str
    inputs: list[str]  # artifact names
    outputs: list[str]  # artifact names


@dataclass(frozen=True)
class Template:
    name: str
    artifacts: list[ArtifactDeclaration]
    blocks: list[BlockDefinition]

    @classmethod
    def from_yaml(cls, path: Path) -> Template:
        """Load a template from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        artifact_names: set[str] = set()
        artifact_kinds: dict[str, str] = {}
        artifacts: list[ArtifactDeclaration] = []
        for entry in data.get("artifacts", []):
            kind = entry["kind"]
            name = entry["name"]
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
