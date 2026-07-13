"""Workflow graph contract and built-in template tests."""

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from video_screener.workflows import (
    WorkflowDefinition,
    load_workflow,
    write_workflow,
)


ROOT = Path(__file__).parents[1]
BUILT_IN = ROOT / "configs" / "workflows" / "vimax-screen-openmontage.json"


def _simple_workflow() -> dict:
    return {
        "workflow_id": "simple-flow",
        "name": "Simple flow",
        "version": "1",
        "nodes": [
            {
                "id": "source",
                "type": "input.text",
                "label": "Source",
                "category": "input",
                "position": {"x": 0, "y": 0},
                "inputs": [{"name": "value", "type": "text"}],
                "outputs": [{"name": "value", "type": "text"}],
            },
            {
                "id": "sink",
                "type": "output.text",
                "label": "Sink",
                "category": "output",
                "position": {"x": 200, "y": 0},
                "inputs": [{"name": "value", "type": "text"}],
                "outputs": [{"name": "value", "type": "text"}],
            },
        ],
        "edges": [
            {
                "id": "source-to-sink",
                "source_node": "source",
                "source_port": "value",
                "target_node": "sink",
                "target_port": "value",
            }
        ],
        "input_slots": [
            {
                "id": "content",
                "label": "Content",
                "type": "text",
                "target_node": "source",
                "target_port": "value",
            }
        ],
        "output_slots": [
            {
                "id": "result",
                "label": "Result",
                "type": "text",
                "source_node": "sink",
                "source_port": "value",
            }
        ],
    }


def test_builtin_three_engine_workflow_is_valid():
    workflow = load_workflow(BUILT_IN)

    assert workflow.execution_order() == [
        "project-input",
        "vimax-plan",
        "image-generation",
        "image-screen",
        "video-generation",
        "clip-screen",
        "openmontage-compose",
        "final-screen",
        "delivery-output",
    ]
    assert workflow.required_engines() == [
        "application",
        "vimax",
        "comfyui",
        "video-asset-screener",
        "openmontage",
    ]
    assert {slot.id for slot in workflow.output_slots} == {
        "final-video",
        "quality-report",
        "edit-timeline",
    }


def test_unknown_port_is_rejected():
    raw = _simple_workflow()
    raw["edges"][0]["source_port"] = "missing"

    with pytest.raises(ValidationError, match="has no output port"):
        WorkflowDefinition.model_validate(raw)


def test_incompatible_port_types_are_rejected():
    raw = _simple_workflow()
    raw["nodes"][1]["inputs"][0]["type"] = "video"

    with pytest.raises(ValidationError, match="incompatible types"):
        WorkflowDefinition.model_validate(raw)


def test_non_multiple_input_rejects_more_than_one_provider():
    raw = _simple_workflow()
    raw["input_slots"].append(
        {
            "id": "second-content",
            "label": "Second content",
            "type": "text",
            "target_node": "sink",
            "target_port": "value",
        }
    )

    with pytest.raises(ValidationError, match="accepts one provider"):
        WorkflowDefinition.model_validate(raw)


def test_required_input_must_be_connected_or_published_as_a_slot():
    raw = _simple_workflow()
    raw["edges"] = []

    with pytest.raises(ValidationError, match="required input sink.value"):
        WorkflowDefinition.model_validate(raw)


def test_cycle_is_rejected():
    raw = _simple_workflow()
    raw["nodes"][0]["inputs"][0]["required"] = False
    raw["input_slots"] = []
    raw["edges"].append(
        {
            "id": "sink-to-source",
            "source_node": "sink",
            "source_port": "value",
            "target_node": "source",
            "target_port": "value",
        }
    )

    with pytest.raises(ValidationError, match="contains a cycle"):
        WorkflowDefinition.model_validate(raw)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_workflow_round_trip(tmp_path: Path, suffix: str):
    workflow = WorkflowDefinition.model_validate(deepcopy(_simple_workflow()))
    destination = tmp_path / f"workflow{suffix}"

    write_workflow(workflow, destination)
    restored = load_workflow(destination)

    assert restored == workflow
