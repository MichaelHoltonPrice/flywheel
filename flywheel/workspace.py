"""Workspace creation, persistence, and artifact recording.

A workspace is a directory inside a project's workforce folder,
created from a template. It holds all artifacts for a unit of work
and enforces immutability once artifacts are recorded.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from flywheel.artifact import Artifact, CopyArtifact, GitArtifact, GitRef
from flywheel.template import Template

_VALID_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def _validate_name(name: str, label: str) -> None:
    """Validate that a name is safe for use as a directory or identifier."""
    if not name:
        raise ValueError(f"{label} name must not be empty")
    if not _VALID_NAME.match(name):
        raise ValueError(
            f"{label} name {name!r} is invalid. "
            f"Use only letters, digits, hyphens, and underscores."
        )


@dataclass
class Workspace:
    """A workspace instance created from a template.

    Artifacts start as None (not yet produced/resolved) and are
    filled via record_artifact(), which enforces immutability.
    """

    name: str
    path: Path
    template_name: str
    created_at: datetime
    artifacts: dict[str, Artifact | None]  # None = declared but not yet produced
    _artifact_kinds: dict[str, str]  # declared kind per artifact name

    @classmethod
    def create(cls, name: str, template: Template, workforce_dir: Path) -> Workspace:
        """Create a new workspace directory from a template.

        Creates the workspace directory, resolves git artifact baselines,
        and writes workspace.yaml metadata.

        Args:
            name: Workspace name (letters, digits, hyphens, underscores).
            template: The template defining artifacts and blocks.
            workforce_dir: Path to the workforce directory.

        Returns:
            The created Workspace instance.

        Raises:
            FileExistsError: If the workspace already exists.
            ValueError: If the name is invalid or git artifact declarations are incomplete.
            RuntimeError: If a git repo has uncommitted changes.
            FileNotFoundError: If a git artifact path does not exist in the repo.
        """
        _validate_name(name, "Workspace")
        ws_path = workforce_dir / "workspaces" / name
        if ws_path.exists():
            raise FileExistsError(f"Workspace already exists: {ws_path}")

        ws_path.mkdir(parents=True)
        (ws_path / "artifacts").mkdir()

        try:
            project_root = workforce_dir.parent
            artifacts: dict[str, Artifact | None] = {}
            artifact_kinds: dict[str, str] = {}

            for decl in template.artifacts:
                artifact_kinds[decl.name] = decl.kind
                if decl.kind == "git":
                    if decl.repo is None:
                        raise ValueError(f"Git artifact {decl.name!r} missing repo")
                    if decl.path is None:
                        raise ValueError(f"Git artifact {decl.name!r} missing path")
                    repo_path = (project_root / decl.repo).resolve()

                    # Check for dirty working tree
                    status = subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    if status.stdout.strip():
                        raise RuntimeError(
                            f"Git repo {repo_path} has uncommitted changes. "
                            f"Commit or stash before creating a workspace."
                        )

                    # Check that the declared path exists in the repo
                    artifact_path = repo_path / decl.path
                    if not artifact_path.exists():
                        raise FileNotFoundError(
                            f"Git artifact {decl.name!r} path {decl.path!r} "
                            f"does not exist in repo {repo_path}"
                        )

                    # Record baseline snapshot — what the code looked like
                    # at workspace creation. Blocks re-resolve at execution
                    # time to get the current committed state.
                    result = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    commit = result.stdout.strip()

                    artifacts[decl.name] = GitArtifact(
                        name=decl.name,
                        ref=GitRef(
                            repo=str(repo_path),
                            commit=commit,
                            path=decl.path,
                        ),
                    )
                else:
                    # copy artifact -- not yet produced
                    artifacts[decl.name] = None

            now = datetime.now(UTC)
            ws = cls(
                name=name,
                path=ws_path,
                template_name=template.name,
                created_at=now,
                artifacts=artifacts,
                _artifact_kinds=artifact_kinds,
            )
            ws.save()
            return ws
        except Exception:
            shutil.rmtree(ws_path, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path: Path) -> Workspace:
        """Load an existing workspace from its workspace.yaml.

        Args:
            path: Path to the workspace directory.

        Returns:
            The loaded Workspace instance.
        """
        yaml_path = path / "workspace.yaml"
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        artifacts: dict[str, Artifact | None] = {}
        artifact_kinds: dict[str, str] = data.get("artifact_kinds", {})
        for name, entry in data.get("artifacts", {}).items():
            if entry is None:
                artifacts[name] = None
                # Infer kind from artifact_kinds if available, default to copy
                if name not in artifact_kinds:
                    artifact_kinds[name] = "copy"
            elif entry["kind"] == "git":
                artifacts[name] = GitArtifact(
                    name=name,
                    ref=GitRef(
                        repo=entry["repo"],
                        commit=entry["commit"],
                        path=entry["path"],
                    ),
                )
                artifact_kinds.setdefault(name, "git")
            elif entry["kind"] == "copy":
                artifacts[name] = CopyArtifact(
                    name=name,
                    path=Path(entry["path"]),
                )
                artifact_kinds.setdefault(name, "copy")
            else:
                raise ValueError(f"Artifact {name!r} has unknown kind {entry['kind']!r}")

        return cls(
            name=data["name"],
            path=path,
            template_name=data["template_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            artifacts=artifacts,
            _artifact_kinds=artifact_kinds,
        )

    def record_artifact(self, name: str, artifact: Artifact) -> None:
        """Record a produced artifact.

        Args:
            name: The declared artifact slot name.
            artifact: The artifact to record.

        Raises:
            KeyError: If the name is not declared in this workspace.
            ValueError: If the slot is already recorded, or the artifact name
                does not match the slot.
            TypeError: If the artifact kind does not match the declared kind.
        """
        if name not in self.artifacts:
            raise KeyError(f"Artifact {name!r} not declared in this workspace")
        if self.artifacts[name] is not None:
            raise ValueError(f"Artifact {name!r} already recorded and is immutable")
        if artifact.name != name:
            raise ValueError(
                f"Artifact name {artifact.name!r} does not match slot {name!r}"
            )
        expected_kind = self._artifact_kinds[name]
        actual_kind = "git" if isinstance(artifact, GitArtifact) else "copy"
        if actual_kind != expected_kind:
            raise TypeError(
                f"Artifact {name!r} declared as {expected_kind!r} "
                f"but received {actual_kind!r}"
            )
        self.artifacts[name] = artifact

    def save(self) -> None:
        """Write workspace.yaml to the workspace directory."""
        serialized_artifacts: dict[str, dict[str, str] | None] = {}
        for name, artifact in self.artifacts.items():
            if artifact is None:
                serialized_artifacts[name] = None
            elif isinstance(artifact, GitArtifact):
                serialized_artifacts[name] = {
                    "kind": "git",
                    "repo": artifact.ref.repo,
                    "commit": artifact.ref.commit,
                    "path": artifact.ref.path,
                }
            elif isinstance(artifact, CopyArtifact):
                serialized_artifacts[name] = {
                    "kind": "copy",
                    "path": str(artifact.path),
                }

        data = {
            "name": self.name,
            "template_name": self.template_name,
            "created_at": self.created_at.isoformat(),
            "artifact_kinds": self._artifact_kinds,
            "artifacts": serialized_artifacts,
        }

        yaml_path = self.path / "workspace.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
