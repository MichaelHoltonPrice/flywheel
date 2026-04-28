"""Project-supplied managed-state validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flywheel.template import BlockDefinition


class StateValidationError(Exception):
    """Raised when a candidate managed-state snapshot is rejected."""

    def __init__(
        self,
        message: str,
        *,
        block_name: str | None = None,
        lineage_key: str | None = None,
        path: Path | None = None,
    ) -> None:
        """Build a state-validation error."""
        super().__init__(message)
        self.block_name = block_name
        self.lineage_key = lineage_key
        self.path = path


StateValidator = Callable[[str, "BlockDefinition", Path, str], None]
"""Project-supplied callable for candidate managed state.

Signature:
``validator(block_name, block_def, staged_path, lineage_key) -> None``.

``staged_path`` is the state directory captured from
``/flywheel/state`` before Flywheel registers it as an immutable
state snapshot.  The callable must treat the directory as read-only.
Flywheel passes an isolated copy for validation, so mutations are
ignored.  Raise :class:`StateValidationError` to reject it; return
``None`` to accept it.
"""


class StateValidatorRegistry:
    """Project-owned map from block name to managed-state validator.

    Names with no registered validator are accepted unconditionally.
    State validators are intentionally separate from artifact
    validators: state is not an artifact, has no artifact declaration,
    and is not eligible as a block input binding.
    """

    def __init__(
        self,
        validators: Mapping[str, StateValidator] | None = None,
    ) -> None:
        """Seed the registry from an optional initial mapping."""
        self._validators: dict[str, StateValidator] = (
            dict(validators) if validators else {}
        )

    def register(self, block_name: str, validator: StateValidator) -> None:
        """Register a validator for a block's managed state."""
        self._validators[block_name] = validator

    def has(self, block_name: str) -> bool:
        """Return ``True`` iff a validator is registered."""
        return block_name in self._validators

    def names(self) -> list[str]:
        """Return the sorted list of block names with validators."""
        return sorted(self._validators)

    def validate(
        self,
        block_name: str,
        block_def: BlockDefinition,
        staged_path: Path,
        lineage_key: str,
    ) -> None:
        """Invoke the registered validator for ``block_name``, if any."""
        validator = self._validators.get(block_name)
        if validator is None:
            return
        try:
            validator(block_name, block_def, staged_path, lineage_key)
        except StateValidationError as exc:
            raise StateValidationError(
                str(exc),
                block_name=exc.block_name or block_name,
                lineage_key=exc.lineage_key or lineage_key,
                path=exc.path or staged_path,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise StateValidationError(
                f"state validator for {block_name!r} raised "
                f"{type(exc).__name__}: {exc}",
                block_name=block_name,
                lineage_key=lineage_key,
                path=staged_path,
            ) from exc
