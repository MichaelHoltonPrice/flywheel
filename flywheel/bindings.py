"""Declarative input-binding source types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactIdBinding:
    """Bind an input slot to a concrete artifact instance id."""

    artifact_id: str
