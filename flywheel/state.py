"""Managed state records and compatibility helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import yaml

if TYPE_CHECKING:
    from flywheel.template import BlockDefinition

StateMode = Literal["none", "managed", "unmanaged"]
STATE_COMPATIBILITY_HASH_FIELDS: tuple[str, ...] = (
    "inputs",
    "outputs",
    "docker_args",
    "env",
    "runner",
    "post_check",
    "output_builder",
    "lifecycle",
    "stop_timeout_s",
)
"""BlockDefinition fields included in ``block_template_hash``."""


@dataclass(frozen=True)
class StateSnapshot:
    """A captured managed-state snapshot for one execution lineage."""

    id: str
    lineage_key: str
    created_at: datetime
    produced_by: str
    predecessor_snapshot_id: str | None
    compatibility: dict[str, str]
    state_path: str


def normalize_state_mode(value: object, *, block_name: str) -> StateMode:
    """Normalize a block YAML ``state:`` value."""
    if value is None:
        return "none"
    if value is True:
        return "managed"
    if isinstance(value, str):
        if value in ("none", "managed", "unmanaged"):
            return value
        raise ValueError(
            f"Block {block_name!r}: unknown state mode {value!r}; "
            "expected 'none', 'managed', 'unmanaged', or true"
        )
    raise ValueError(
        f"Block {block_name!r}: 'state' must be one of "
        "'none', 'managed', 'unmanaged', or true "
        f"(got {type(value).__name__})"
    )


def pattern_state_lineage_key(
    run_id: str,
    lane_name: str,
    block_name: str,
) -> str:
    """Return the default managed-state lineage key for a pattern lane."""
    return f"pattern/{run_id}/lane/{lane_name}/block/{block_name}"


def state_compatibility_identity(
    block_def: BlockDefinition,
) -> dict[str, str]:
    """Return the compatibility identity for a stateful block."""
    def sequence_payload(slot) -> dict[str, str | None] | None:
        if slot.sequence is None:
            return None
        payload = {
            "name": slot.sequence.name,
            "scope": slot.sequence.scope,
        }
        if slot.sequence.role is not None:
            payload["role"] = slot.sequence.role
        return payload

    payload = {
        "inputs": sorted([
            {
                "name": slot.name,
                "container_path": slot.container_path,
                "optional": slot.optional,
                "sequence": sequence_payload(slot),
            }
            for slot in block_def.inputs
        ], key=lambda item: item["name"]),
        "outputs": {
            reason: sorted([
                {
                    "name": slot.name,
                    "container_path": slot.container_path,
                    "sequence": sequence_payload(slot),
                }
                for slot in slots
            ], key=lambda item: item["name"])
            for reason, slots in sorted(block_def.outputs.items())
        },
        "docker_args": list(block_def.docker_args),
        "env": dict(sorted(block_def.env.items())),
        "runner": block_def.runner,
        "post_check": block_def.post_check,
        "output_builder": block_def.output_builder,
        "lifecycle": block_def.lifecycle,
        "stop_timeout_s": block_def.stop_timeout_s,
    }
    dumped = yaml.safe_dump(payload, sort_keys=True)
    return {
        "block_name": block_def.name,
        "state_mode": block_def.state,
        "image": block_def.image,
        "block_template_hash": hashlib.sha256(
            dumped.encode("utf-8")
        ).hexdigest(),
    }
