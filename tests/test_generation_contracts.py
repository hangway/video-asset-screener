"""Neutral local-generation contract tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from video_screener.generation import (
    GenerationArtifact,
    GenerationResult,
    GenerationStatus,
    ImageGenerationRequest,
    MediaKind,
    ModelSpec,
    ReferenceAsset,
    WorkflowSpec,
    write_generation_result,
)


def _request(**overrides) -> ImageGenerationRequest:
    values = {
        "request_id": "image-job-1",
        "model": ModelSpec(
            model_id="local/image-model",
            revision="sha256:abc123",
            license="Apache-2.0",
        ),
        "workflow": WorkflowSpec(workflow_id="portrait", version="1.2"),
        "prompt": "A portrait in soft window light",
        "seed": 42,
    }
    values.update(overrides)
    return ImageGenerationRequest(**values)


def test_request_rejects_duplicate_reference_roles(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    with pytest.raises(ValidationError, match="reference roles must be unique"):
        _request(references=(
            ReferenceAsset(path=image_a, role="character"),
            ReferenceAsset(path=image_b, role="character"),
        ))


def test_request_id_cannot_escape_output_directory():
    with pytest.raises(ValidationError, match="request_id"):
        _request(request_id="../../outside")


def test_success_result_requires_an_artifact():
    with pytest.raises(ValidationError, match="requires at least one artifact"):
        GenerationResult(
            request=_request(),
            status=GenerationStatus.SUCCEEDED,
            backend="test",
            runtime_seconds=0.1,
        )


def test_generation_manifest_preserves_provenance(tmp_path):
    output = tmp_path / "candidate.png"
    output.write_bytes(b"candidate")
    request = _request()
    result = GenerationResult(
        request=request,
        status=GenerationStatus.SUCCEEDED,
        backend="test-local",
        runtime_seconds=1.25,
        hardware={"device": "test-gpu", "vram_mb": 8192},
        artifacts=[GenerationArtifact(
            path=output,
            media_kind=MediaKind.IMAGE,
            mime_type="image/png",
            sha256="0" * 64,
            size_bytes=9,
            candidate_index=0,
            seed=42,
            source_prompt_id="prompt-1",
        )],
    )

    manifest = write_generation_result(result, tmp_path / "generation.json")
    saved = json.loads(manifest.read_text(encoding="utf-8"))

    assert saved["request"]["prompt"] == request.prompt
    assert saved["request"]["model"] == {
        "model_id": "local/image-model",
        "revision": "sha256:abc123",
        "license": "Apache-2.0",
    }
    assert saved["request"]["workflow"]["version"] == "1.2"
    assert saved["request"]["seed"] == 42
    assert saved["hardware"]["device"] == "test-gpu"
    assert saved["artifacts"][0]["path"] == str(output)
