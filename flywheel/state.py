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
    if value is None or value is False:
        return "none"
    if value is True:
        return "managed"
    if isinstance(value, str):
        if value in ("none", "managed", "unmanaged"):
            return value
        raise ValueError(
            f"Block {block_name!r}: unknown state mode {value!r}; "
            "expected 'none', 'managed', 'unmanaged', true, or false"
        )
    raise ValueError(
        f"Block {block_name!r}: 'state' must be one of "
        "'none', 'managed', 'unmanaged', true, or false "
        f"(got {type(value).__name__})"
    )


def pattern_state_lineage_key(
    run_id: str,
    step_name: str,
    member_name: str,
) -> str:
    """Return the default managed-state lineage key for a pattern member."""
    return f"pattern/{run_id}/step/{step_name}/member/{member_name}"


def state_compatibility_identity(
    block_def: "BlockDefinition",
) -> dict[str, str]:
    """Return the compatibility identity for a stateful block."""
    payload = {
        "name": block_def.name,
        "image": block_def.image,
        "inputs": [
            {
                "name": slot.name,
                "container_path": slot.container_path,
                "optional": slot.optional,
            }
            for slot in block_def.inputs
        ],
        "outputs": {
            reason: [
                {
                    "name": slot.name,
                    "container_path": slot.container_path,
                }
                for slot in slots
            ]
            for reason, slots in sorted(block_def.outputs.items())
        },
        "docker_args": list(block_def.docker_args),
        "env": dict(sorted(block_def.env.items())),
        "runner": block_def.runner,
        "runner_justification": block_def.runner_justification,
        "post_check": block_def.post_check,
        "output_builder": block_def.output_builder,
        "lifecycle": block_def.lifecycle,
        "state": block_def.state,
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
