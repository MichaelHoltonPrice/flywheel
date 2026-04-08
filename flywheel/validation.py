"""Shared validation utilities for flywheel."""

from __future__ import annotations

import re

_VALID_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def validate_name(name: str, label: str) -> None:
    """Validate that a name is safe for use as a directory or identifier.

    Args:
        name: The name to validate.
        label: A human-readable label for error messages (e.g., "Workspace").

    Raises:
        ValueError: If the name is empty or contains invalid characters.
    """
    if not name:
        raise ValueError(f"{label} name must not be empty")
    if not _VALID_NAME.match(name):
        raise ValueError(
            f"{label} name {name!r} is invalid. "
            f"Use only letters, digits, hyphens, and underscores."
        )
