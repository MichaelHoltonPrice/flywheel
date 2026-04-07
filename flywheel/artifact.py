"""Immutable artifact types for flywheel workspaces.

Artifacts are the unit of data exchange between blocks. Two storage
kinds exist: copy artifacts (files stored in the workspace) and git
artifacts (references to version-controlled code).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitRef:
    """A pinned reference to a path within a git repository."""

    repo: str  # absolute path to repo root
    commit: str  # full SHA
    path: str  # relative path within repo


@dataclass(frozen=True)
class CopyArtifact:
    """A file or directory copy stored in the workspace."""

    name: str
    path: Path  # relative to workspace artifacts/ dir


@dataclass(frozen=True)
class GitArtifact:
    """A reference to version-controlled code, pinned to a commit."""

    name: str
    ref: GitRef


Artifact = CopyArtifact | GitArtifact
