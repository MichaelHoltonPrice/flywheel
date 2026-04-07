from __future__ import annotations

from pathlib import Path

import pytest

from flywheel.artifact import Artifact, CopyArtifact, GitArtifact, GitRef


class TestGitRef:
    def test_fields_accessible(self):
        ref = GitRef(repo="/some/repo", commit="abc123", path="src/main.py")
        assert ref.repo == "/some/repo"
        assert ref.commit == "abc123"
        assert ref.path == "src/main.py"

    def test_frozen(self):
        ref = GitRef(repo="/some/repo", commit="abc123", path="src/main.py")
        with pytest.raises(AttributeError):
            ref.repo = "/other/repo"


class TestCopyArtifact:
    def test_fields(self):
        artifact = CopyArtifact(name="checkpoint", path=Path("checkpoints/latest"))
        assert artifact.name == "checkpoint"
        assert artifact.path == Path("checkpoints/latest")

    def test_path_is_path_object(self):
        artifact = CopyArtifact(name="data", path=Path("data/out"))
        assert isinstance(artifact.path, Path)

    def test_frozen(self):
        artifact = CopyArtifact(name="checkpoint", path=Path("checkpoints/latest"))
        with pytest.raises(AttributeError):
            artifact.name = "other"

    def test_frozen_path(self):
        artifact = CopyArtifact(name="checkpoint", path=Path("checkpoints/latest"))
        with pytest.raises(AttributeError):
            artifact.path = Path("other")


class TestGitArtifact:
    def test_fields(self):
        ref = GitRef(repo="/repo", commit="deadbeef", path="lib")
        artifact = GitArtifact(name="engine", ref=ref)
        assert artifact.name == "engine"
        assert artifact.ref is ref

    def test_ref_fields_through_artifact(self):
        ref = GitRef(repo="/repo", commit="deadbeef", path="lib")
        artifact = GitArtifact(name="engine", ref=ref)
        assert artifact.ref.repo == "/repo"
        assert artifact.ref.commit == "deadbeef"
        assert artifact.ref.path == "lib"

    def test_frozen(self):
        ref = GitRef(repo="/repo", commit="deadbeef", path="lib")
        artifact = GitArtifact(name="engine", ref=ref)
        with pytest.raises(AttributeError):
            artifact.name = "other"

    def test_frozen_ref(self):
        ref = GitRef(repo="/repo", commit="deadbeef", path="lib")
        artifact = GitArtifact(name="engine", ref=ref)
        with pytest.raises(AttributeError):
            artifact.ref = GitRef(repo="/x", commit="x", path="x")


class TestArtifactUnion:
    def test_copy_artifact_is_artifact(self):
        artifact = CopyArtifact(name="score", path=Path("scores/run1"))
        assert isinstance(artifact, CopyArtifact)
        # Union type checks work via isinstance for the concrete types
        assert isinstance(artifact, CopyArtifact | GitArtifact)

    def test_git_artifact_is_artifact(self):
        ref = GitRef(repo="/repo", commit="abc", path="src")
        artifact = GitArtifact(name="code", ref=ref)
        assert isinstance(artifact, GitArtifact)
        assert isinstance(artifact, CopyArtifact | GitArtifact)

    def test_non_artifact_not_in_union(self):
        assert not isinstance("not an artifact", CopyArtifact | GitArtifact)

    def test_discriminate_by_type(self):
        ref = GitRef(repo="/repo", commit="abc", path="src")
        items: list[Artifact] = [
            CopyArtifact(name="data", path=Path("data")),
            GitArtifact(name="code", ref=ref),
        ]
        copy_artifacts = [a for a in items if isinstance(a, CopyArtifact)]
        git_artifacts = [a for a in items if isinstance(a, GitArtifact)]
        assert len(copy_artifacts) == 1
        assert len(git_artifacts) == 1
        assert copy_artifacts[0].name == "data"
        assert git_artifacts[0].name == "code"
