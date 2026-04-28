from __future__ import annotations

from pathlib import Path

import pytest

from flywheel.state_validator import (
    StateValidationError,
    StateValidatorRegistry,
)
from flywheel.template import BlockDefinition


def _block() -> BlockDefinition:
    return BlockDefinition(name="train", image="train:latest")


class TestStateValidatorRegistry:
    def test_no_validator_is_noop(self, tmp_path: Path):
        registry = StateValidatorRegistry()

        registry.validate("train", _block(), tmp_path, "lineage")

    def test_registered_validator_is_called(self, tmp_path: Path):
        seen: list[tuple[str, str, str, str]] = []

        def validator(name, block_def, path, lineage_key):
            seen.append((
                name,
                block_def.name,
                path.name,
                lineage_key,
            ))

        registry = StateValidatorRegistry({"train": validator})
        registry.validate("train", _block(), tmp_path, "lineage")

        assert seen == [("train", "train", tmp_path.name, "lineage")]

    def test_validation_error_gets_context(self, tmp_path: Path):
        original = StateValidationError("bad state")

        def reject(name, block_def, path, lineage_key):
            raise original

        registry = StateValidatorRegistry({"train": reject})

        with pytest.raises(StateValidationError) as exc_info:
            registry.validate("train", _block(), tmp_path, "lineage")

        assert str(exc_info.value) == "bad state"
        assert exc_info.value is not original
        assert exc_info.value.__cause__ is original
        assert exc_info.value.block_name == "train"
        assert exc_info.value.lineage_key == "lineage"
        assert exc_info.value.path == tmp_path
        assert original.block_name is None
        assert original.lineage_key is None
        assert original.path is None

    def test_plain_exception_is_wrapped(self, tmp_path: Path):
        def boom(name, block_def, path, lineage_key):
            raise RuntimeError("broken")

        registry = StateValidatorRegistry({"train": boom})

        with pytest.raises(StateValidationError) as exc_info:
            registry.validate("train", _block(), tmp_path, "lineage")

        assert "state validator for 'train' raised RuntimeError" in str(
            exc_info.value)
        assert exc_info.value.block_name == "train"
        assert exc_info.value.lineage_key == "lineage"
        assert exc_info.value.path == tmp_path
