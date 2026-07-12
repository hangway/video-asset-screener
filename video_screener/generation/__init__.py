"""Pluggable local generation contracts and backends."""

from .contracts import (
    GenerationArtifact,
    GenerationResult,
    GenerationStatus,
    ImageGenerationBackend,
    ImageGenerationRequest,
    MediaKind,
    ModelSpec,
    ReferenceAsset,
    VideoGenerationBackend,
    VideoGenerationRequest,
    WorkflowSpec,
    write_generation_result,
)

__all__ = [
    "GenerationArtifact",
    "GenerationResult",
    "GenerationStatus",
    "ImageGenerationBackend",
    "ImageGenerationRequest",
    "MediaKind",
    "ModelSpec",
    "ReferenceAsset",
    "VideoGenerationBackend",
    "VideoGenerationRequest",
    "WorkflowSpec",
    "write_generation_result",
]
