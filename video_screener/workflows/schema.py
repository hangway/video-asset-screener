"""Backend-neutral workflow graph and publication contracts.

The graph is the shared source of truth for CLI tools, a future node canvas,
template discovery, and execution history. Engine-specific workflow files stay
inside node configuration rather than leaking into the graph structure.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..utils.io import read_json, write_json


IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
NODE_TYPE_PATTERN = r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$"


class PortType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    JSON = "json"
    DECISION = "decision"
    ANY = "any"


class NodeCategory(str, Enum):
    INPUT = "input"
    PLANNING = "planning"
    GENERATION = "generation"
    SCREENING = "screening"
    COMPOSITION = "composition"
    OUTPUT = "output"


class WorkflowVisibility(str, Enum):
    PRIVATE = "private"
    UNLISTED = "unlisted"
    PUBLIC = "public"


class CanvasPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float


class WorkflowPort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=IDENTIFIER_PATTERN)
    type: PortType
    required: bool = True
    multiple: bool = False
    description: str = ""


class WorkflowNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=IDENTIFIER_PATTERN)
    type: str = Field(pattern=NODE_TYPE_PATTERN)
    label: str = Field(min_length=1)
    category: NodeCategory
    position: CanvasPosition
    engine: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    model_ids: tuple[str, ...] = ()
    inputs: tuple[WorkflowPort, ...] = ()
    outputs: tuple[WorkflowPort, ...] = ()
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def _label_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("node label must not be blank")
        return value

    @field_validator("model_ids")
    @classmethod
    def _model_ids_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("node model_ids must be unique")
        return value

    @model_validator(mode="after")
    def _port_names_unique(self) -> "WorkflowNode":
        for direction, ports in (("input", self.inputs), ("output", self.outputs)):
            names = [port.name for port in ports]
            if len(names) != len(set(names)):
                raise ValueError(f"node {self.id!r} has duplicate {direction} ports")
        return self


class WorkflowEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_node: str = Field(pattern=IDENTIFIER_PATTERN)
    source_port: str = Field(pattern=IDENTIFIER_PATTERN)
    target_node: str = Field(pattern=IDENTIFIER_PATTERN)
    target_port: str = Field(pattern=IDENTIFIER_PATTERN)


class WorkflowInputSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=IDENTIFIER_PATTERN)
    label: str = Field(min_length=1)
    type: PortType
    target_node: str = Field(pattern=IDENTIFIER_PATTERN)
    target_port: str = Field(pattern=IDENTIFIER_PATTERN)
    required: bool = True
    default: Any = None
    description: str = ""


class WorkflowOutputSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=IDENTIFIER_PATTERN)
    label: str = Field(min_length=1)
    type: PortType
    source_node: str = Field(pattern=IDENTIFIER_PATTERN)
    source_port: str = Field(pattern=IDENTIFIER_PATTERN)
    description: str = ""


class WorkflowPublication(BaseModel):
    """Metadata used by personal libraries, sharing, and template discovery."""

    model_config = ConfigDict(extra="forbid")

    visibility: WorkflowVisibility = WorkflowVisibility.PRIVATE
    author: str = ""
    summary: str = ""
    cover_asset: str | None = None
    use_cases: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    required_models: tuple[str, ...] = ()

    @field_validator("use_cases", "categories", "tags", "required_models")
    @classmethod
    def _publication_values_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("publication metadata values must be unique")
        return value


class WorkflowDefinition(BaseModel):
    """A validated directed acyclic graph that can be rendered or executed."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    workflow_id: str = Field(pattern=IDENTIFIER_PATTERN)
    name: str = Field(min_length=1)
    description: str = ""
    version: str = Field(min_length=1)
    nodes: tuple[WorkflowNode, ...] = Field(min_length=1)
    edges: tuple[WorkflowEdge, ...] = ()
    input_slots: tuple[WorkflowInputSlot, ...] = ()
    output_slots: tuple[WorkflowOutputSlot, ...] = ()
    publication: WorkflowPublication = Field(default_factory=WorkflowPublication)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "version")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("workflow name and version must not be blank")
        return value

    @model_validator(mode="after")
    def _graph_is_valid(self) -> "WorkflowDefinition":
        nodes = self._unique_by_id(self.nodes, "node")
        self._unique_by_id(self.edges, "edge")
        self._unique_by_id(self.input_slots, "input slot")
        self._unique_by_id(self.output_slots, "output slot")

        incoming: dict[tuple[str, str], list[str]] = {}
        edge_signatures: set[tuple[str, str, str, str]] = set()
        for edge in self.edges:
            source = self._port(nodes, edge.source_node, edge.source_port, "output")
            target = self._port(nodes, edge.target_node, edge.target_port, "input")
            self._require_compatible(source.type, target.type, f"edge {edge.id!r}")

            signature = (
                edge.source_node,
                edge.source_port,
                edge.target_node,
                edge.target_port,
            )
            if signature in edge_signatures:
                raise ValueError(f"duplicate connection on edge {edge.id!r}")
            edge_signatures.add(signature)
            incoming.setdefault((edge.target_node, edge.target_port), []).append(edge.id)

        slotted_inputs: dict[tuple[str, str], list[str]] = {}
        for slot in self.input_slots:
            target = self._port(nodes, slot.target_node, slot.target_port, "input")
            self._require_compatible(slot.type, target.type, f"input slot {slot.id!r}")
            slotted_inputs.setdefault((slot.target_node, slot.target_port), []).append(slot.id)

        for slot in self.output_slots:
            source = self._port(nodes, slot.source_node, slot.source_port, "output")
            self._require_compatible(source.type, slot.type, f"output slot {slot.id!r}")

        for node in self.nodes:
            for port in node.inputs:
                endpoint = (node.id, port.name)
                providers = incoming.get(endpoint, []) + slotted_inputs.get(endpoint, [])
                if not port.multiple and len(providers) > 1:
                    raise ValueError(
                        f"input {node.id}.{port.name} accepts one provider, got {providers}"
                    )
                if port.required and not providers:
                    raise ValueError(f"required input {node.id}.{port.name} is not connected")

        self.execution_order()
        return self

    @staticmethod
    def _unique_by_id(items: tuple[Any, ...], kind: str) -> dict[str, Any]:
        indexed: dict[str, Any] = {}
        for item in items:
            if item.id in indexed:
                raise ValueError(f"duplicate {kind} id {item.id!r}")
            indexed[item.id] = item
        return indexed

    @staticmethod
    def _port(
        nodes: dict[str, WorkflowNode],
        node_id: str,
        port_name: str,
        direction: Literal["input", "output"],
    ) -> WorkflowPort:
        node = nodes.get(node_id)
        if node is None:
            raise ValueError(f"{direction} endpoint references unknown node {node_id!r}")
        ports = node.inputs if direction == "input" else node.outputs
        for port in ports:
            if port.name == port_name:
                return port
        raise ValueError(f"node {node_id!r} has no {direction} port {port_name!r}")

    @staticmethod
    def _require_compatible(source: PortType, target: PortType, context: str) -> None:
        if source != target and PortType.ANY not in (source, target):
            raise ValueError(
                f"{context} connects incompatible types {source.value!r} and {target.value!r}"
            )

    def execution_order(self) -> list[str]:
        """Return a stable topological order, raising when the graph has a cycle."""
        node_order = {node.id: index for index, node in enumerate(self.nodes)}
        indegree = {node.id: 0 for node in self.nodes}
        adjacency: dict[str, set[str]] = {node.id: set() for node in self.nodes}
        for edge in self.edges:
            if edge.target_node not in adjacency[edge.source_node]:
                adjacency[edge.source_node].add(edge.target_node)
                indegree[edge.target_node] += 1

        ready = deque(node.id for node in self.nodes if indegree[node.id] == 0)
        ordered: list[str] = []
        while ready:
            node_id = ready.popleft()
            ordered.append(node_id)
            unlocked: list[str] = []
            for target in adjacency[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    unlocked.append(target)
            ready.extend(sorted(unlocked, key=node_order.__getitem__))

        if len(ordered) != len(self.nodes):
            cyclic = [node_id for node_id, degree in indegree.items() if degree > 0]
            raise ValueError(f"workflow graph contains a cycle involving {cyclic}")
        return ordered

    def required_engines(self) -> list[str]:
        return list(dict.fromkeys(node.engine for node in self.nodes if node.engine))

    def required_models(self) -> list[str]:
        model_ids = list(self.publication.required_models)
        for node in self.nodes:
            model_ids.extend(node.model_ids)
        return list(dict.fromkeys(model_ids))


def load_workflow(path: str | Path) -> WorkflowDefinition:
    """Load and validate a JSON or YAML workflow manifest."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        raw = read_json(source)
    elif suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("workflow manifests must use .json, .yaml, or .yml")
    if not isinstance(raw, dict):
        raise ValueError("workflow manifest root must be an object")
    return WorkflowDefinition.model_validate(raw)


def write_workflow(workflow: WorkflowDefinition, path: str | Path) -> Path:
    """Write a validated workflow manifest as JSON or YAML."""
    destination = Path(path)
    data = workflow.model_dump(mode="json")
    suffix = destination.suffix.lower()
    if suffix == ".json":
        return write_json(destination, data)
    if suffix in {".yaml", ".yml"}:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return destination
    raise ValueError("workflow manifests must use .json, .yaml, or .yml")
