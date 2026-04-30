from __future__ import annotations

from dataclasses import replace

from flywheel.state import (
    STATE_COMPATIBILITY_HASH_FIELDS,
    state_compatibility_identity,
)
from flywheel.template import BlockDefinition, InputSlot, OutputSlot


def _block(**overrides) -> BlockDefinition:
    block = BlockDefinition(
        name="train",
        image="train:latest",
        inputs=[
            InputSlot(
                name="checkpoint",
                container_path="/input/checkpoint",
                optional=True,
            ),
            InputSlot(name="engine", container_path="/input/engine"),
        ],
        outputs={
            "normal": [
                OutputSlot(
                    name="checkpoint",
                    container_path="/output/checkpoint",
                ),
                OutputSlot(name="score", container_path="/output/score"),
            ]
        },
        docker_args=["--gpus", "all"],
        env={"MODE": "train"},
        runner="container",
        runner_justification="human rationale",
        post_check="project.check:validate",
        output_builder="project.outputs:build",
        lifecycle="one_shot",
        state="managed",
        stop_timeout_s=30,
    )
    return replace(block, **overrides)


def _hash(block: BlockDefinition) -> str:
    return state_compatibility_identity(block)["block_template_hash"]


def test_state_compatibility_identity_top_level_fields():
    identity = state_compatibility_identity(_block())

    assert identity["block_name"] == "train"
    assert identity["state_mode"] == "managed"
    assert identity["image"] == "train:latest"
    assert len(identity["block_template_hash"]) == 64


def test_state_compatibility_hash_field_set_is_pinned():
    assert STATE_COMPATIBILITY_HASH_FIELDS == (
        "inputs",
        "outputs",
        "network",
        "docker_args",
        "env",
        "runner",
        "post_check",
        "output_builder",
        "lifecycle",
        "stop_timeout_s",
    )


def test_state_compatibility_ignores_top_level_identity_fields():
    original = _block()
    changed = replace(
        original,
        name="train_v2",
        image="train:v2",
        state="unmanaged",
    )

    assert _hash(changed) == _hash(original)
    assert state_compatibility_identity(changed) != (
        state_compatibility_identity(original)
    )


def test_state_compatibility_ignores_free_text_rationale():
    original = _block()
    changed = replace(
        original,
        runner_justification="updated explanation",
    )

    assert state_compatibility_identity(changed) == (
        state_compatibility_identity(original)
    )


def test_state_compatibility_hash_changes_for_operational_fields():
    original = _block()

    variants = [
        replace(
            original,
            inputs=[
                InputSlot(
                    name="checkpoint",
                    container_path="/different",
                    optional=True,
                ),
                InputSlot(name="engine", container_path="/input/engine"),
            ],
        ),
        replace(
            original,
            outputs={
                "normal": [
                    OutputSlot(
                        name="checkpoint",
                        container_path="/different",
                    ),
                    OutputSlot(name="score", container_path="/output/score"),
                ]
            },
        ),
        replace(original, network="bridge"),
        replace(original, network="cyberloop-cua"),
        replace(original, docker_args=["--shm-size", "8g"]),
        replace(original, env={"MODE": "eval"}),
        replace(original, runner="lifecycle"),
        replace(original, post_check="project.check:other"),
        replace(original, output_builder="project.outputs:other"),
        replace(original, lifecycle="workspace_persistent"),
        replace(original, stop_timeout_s=60),
    ]

    assert all(_hash(variant) != _hash(original) for variant in variants)


def test_state_compatibility_hash_changes_for_network_value():
    bridge = _block(network="bridge")
    project_network = _block(network="cyberloop-cua")

    assert _hash(bridge) != _hash(project_network)


def test_state_compatibility_hash_canonicalizes_slot_order():
    original = _block()
    reordered = replace(
        original,
        inputs=list(reversed(original.inputs)),
        outputs={"normal": list(reversed(original.outputs["normal"]))},
    )

    assert state_compatibility_identity(reordered) == (
        state_compatibility_identity(original)
    )
