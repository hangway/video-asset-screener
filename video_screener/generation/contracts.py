"""Neutral contracts for local image and video generation backends.

The screener owns these contracts so generation engines can be replaced
without changing the project manifest consumed by ViMax or OpenMontage.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Union, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..utils.io import write_json


MAX_SEED = (2**63) - 1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MediaKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class GenerationStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ModelSpec(BaseModel):
    """Exact model identity and license recorded for every generation."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)


class WorkflowSpec(BaseModel):
    """Versioned workflow identity independent of a model runtime."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ReferenceAsset(BaseModel):
    """Local reference passed to a generation workflow by semantic role."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    role: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    description: str = ""


class _GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    model: ModelSpec
    workflow: WorkflowSpec
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    seed: int = Field(
        default_factory=lambda: secrets.randbelow(MAX_SEED + 1),
        ge=0,
        le=MAX_SEED,
    )
    candidate_count: int = Field(default=1, ge=1, le=8)
    references: tuple[ReferenceAsset, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("references")
    @classmethod
    def _reference_roles_unique(
        cls, value: tuple[ReferenceAsset, ...]
    ) -> tuple[ReferenceAsset, ...]:
        roles = [reference.role for reference in value]
        if len(roles) != len(set(roles)):
            raise ValueError("reference roles must be unique within a request")
        return value


class ImageGenerationRequest(_GenerationRequest):
    kind: Literal[MediaKind.IMAGE] = MediaKind.IMAGE
    width: int = Field(default=1024, ge=64, le=16384)
    height: int = Field(default=1024, ge=64, le=16384)


class VideoGenerationRequest(_GenerationRequest):
    kind: Literal[MediaKind.VIDEO] = MediaKind.VIDEO
    width: int = Field(default=1280, ge=64, le=16384)
    height: int = Field(default=720, ge=64, le=16384)
    duration_seconds: float = Field(default=5.0, gt=0.0, le=3600.0)
    fps: float = Field(default=24.0, gt=0.0, le=240.0)
    first_frame: Path | None = None
    last_frame: Path | None = None

    @property
    def frame_count(self) -> int:
        return max(1, round(self.duration_seconds * self.fps))


GenerationRequest = Annotated[
    Union[ImageGenerationRequest, VideoGenerationRequest],
    Field(discriminator="kind"),
]


class GenerationArtifact(BaseModel):
    """One locally persisted candidate emitted by a generation backend."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    media_kind: MediaKind
    mime_type: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    candidate_index: int = Field(ge=0)
    seed: int = Field(ge=0, le=MAX_SEED)
    source_prompt_id: str = Field(min_length=1)
    source_node_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Auditable terminal result written beside generated candidates."""

    model_config = ConfigDict(extra="forbid")

    request: GenerationRequest
    status: GenerationStatus
    backend: str = Field(min_length=1)
    runtime_seconds: float = Field(ge=0.0)
    hardware: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[GenerationArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    manifest_path: Path | None = None
    completed_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _terminal_state_is_coherent(self) -> "GenerationResult":
        if self.status == GenerationStatus.SUCCEEDED:
            if not self.artifacts:
                raise ValueError("successful generation requires at least one artifact")
            if self.error:
                raise ValueError("successful generation cannot include an error")
        elif not self.error:
            raise ValueError("failed generation requires an error")

        wrong_kind = [
            artifact.path
            for artifact in self.artifacts
            if artifact.media_kind != self.request.kind
        ]
        if wrong_kind:
            raise ValueError(
                f"generation artifacts do not match request kind: {wrong_kind}"
            )
        return self


def write_generation_result(
    result: GenerationResult, path: str | Path
) -> Path:
    """Write a UTF-8 JSON manifest using JSON-native Pydantic values."""
    return write_json(path, result.model_dump(mode="json"))


@runtime_checkable
class ImageGenerationBackend(Protocol):
    name: str

    def generate_image(
        self, request: ImageGenerationRequest
    ) -> GenerationResult: ...


@runtime_checkable
class VideoGenerationBackend(Protocol):
    name: str

    def generate_video(
        self, request: VideoGenerationRequest
    ) -> GenerationResult: ...
