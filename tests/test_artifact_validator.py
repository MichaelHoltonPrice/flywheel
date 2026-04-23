"""Tests for :mod:`flywheel.artifact_validator`.

Covers the registry's "no validator registered = pass" semantics,
how it surfaces validation failures, and how it wraps unexpected
exceptions.  Integration with the import / collection sites lives
in :mod:`tests.test_workspace` and :mod:`tests.test_artifact_validator_integration`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flywheel.artifact_validator import (
    ArtifactValidationError,
    ArtifactValidatorRegistry,
)
from flywheel.template import ArtifactDeclaration


@pytest.fixture
def declaration() -> ArtifactDeclaration:
    return ArtifactDeclaration(name="checkpoint", kind="copy")


class TestArtifactValidatorRegistry:
    def test_no_validator_is_noop(
        self, tmp_path: Path, declaration: ArtifactDeclaration,
    ):
        registry = ArtifactValidatorRegistry()
        # No exception, no return value.
        assert registry.validate(
            "checkpoint", declaration, tmp_path,
        ) is None

    def test_has_reflects_registration(
        self, declaration: ArtifactDeclaration,
    ):
        registry = ArtifactValidatorRegistry()
        assert not registry.has("checkpoint")

        registry.register(
            "checkpoint", lambda *_args: None,
        )
        assert registry.has("checkpoint")
        assert registry.names() == ["checkpoint"]

    def test_seed_from_mapping(
        self, tmp_path: Path, declaration: ArtifactDeclaration,
    ):
        called: list[tuple[str, Path]] = []

        def validator(name, decl, path):
            called.append((name, path))

        registry = ArtifactValidatorRegistry(
            {"checkpoint": validator},
        )
        registry.validate("checkpoint", declaration, tmp_path)
        assert called == [("checkpoint", tmp_path)]

    def test_register_replaces(
        self, tmp_path: Path, declaration: ArtifactDeclaration,
    ):
        registry = ArtifactValidatorRegistry()
        registry.register(
            "checkpoint",
            lambda *_args: (_ for _ in ()).throw(
                ArtifactValidationError("first"),
            ),
        )
        registry.register("checkpoint", lambda *_args: None)
        # Second registration wins; no error raised.
        registry.validate("checkpoint", declaration, tmp_path)

    def test_validation_error_propagates_with_context(
        self, tmp_path: Path, declaration: ArtifactDeclaration,
    ):
        def reject(name, decl, path):
            raise ArtifactValidationError("bad shape")

        registry = ArtifactValidatorRegistry({"checkpoint": reject})
        with pytest.raises(ArtifactValidationError) as exc_info:
            registry.validate(
                "checkpoint", declaration, tmp_path,
            )
        assert exc_info.value.name == "checkpoint"
        assert exc_info.value.path == tmp_path
        assert "bad shape" in str(exc_info.value)

    def test_validation_error_preserves_explicit_context(
        self, tmp_path: Path, declaration: ArtifactDeclaration,
    ):
        explicit_path = tmp_path / "explicit"

        def reject(name, decl, path):
            raise ArtifactValidationError(
                "rejected",
                name="overridden",
                path=explicit_path,
            )

        registry = ArtifactValidatorRegistry({"checkpoint": reject})
        with pytest.raises(ArtifactValidationError) as exc_info:
            registry.validate(
                "checkpoint", declaration, tmp_path,
            )
        # When the validator already populated ``name`` / ``path``
        # the registry must leave them alone.
        assert exc_info.value.name == "overridden"
        assert exc_info.value.path == explicit_path

    def test_unexpected_exception_wrapped(
        self, tmp_path: Path, declaration: ArtifactDeclaration,
    ):
        def boom(name, decl, path):
            raise RuntimeError("boom")

        registry = ArtifactValidatorRegistry({"checkpoint": boom})
        with pytest.raises(ArtifactValidationError) as exc_info:
            registry.validate(
                "checkpoint", declaration, tmp_path,
            )
        assert exc_info.value.name == "checkpoint"
        assert exc_info.value.path == tmp_path
        assert "RuntimeError" in str(exc_info.value)
        assert "boom" in str(exc_info.value)
        # The original exception is chained for debuggability.
        assert isinstance(exc_info.value.__cause__, RuntimeError)
