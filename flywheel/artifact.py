from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitRef:
    repo: str  # absolute path to repo root
    commit: str  # full SHA
    path: str  # relative path within repo


@dataclass(frozen=True)
class CopyArtifact:
    name: str
    path: Path  # relative to workspace artifacts/ dir


@dataclass(frozen=True)
class GitArtifact:
    name: str
    ref: GitRef


Artifact = CopyArtifact | GitArtifact
