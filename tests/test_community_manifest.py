"""Community dataset manifest schema and CLI coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from video_screener.cli import app
from video_screener.community_manifest import (
    CommunityManifestRecord,
    manifest_summary,
    validate_manifest_rows,
)
from video_screener.schema import AnnotationRecord


def _record(**overrides):
    payload = {
        "schema_version": "0.1.0",
        "taxonomy_version": "0.3.1",
        "clip_id": "clip-001",
        "provider": "provider-a",
        "model": "model-a",
        "generation_mode": "text_to_video",
        "media": {
            "url": "https://example.com/clip.mp4",
            "relative_path": None,
            "sha256": "a" * 64,
            "duration_sec": 5.0,
            "width": 1280,
            "height": 720,
            "fps": 24.0,
        },
        "provenance": {
            "prompt_available": False,
            "reference_images_available": False,
            "prompt_path": None,
            "reference_manifest_path": None,
        },
        "context": {
            "asset_role": "hero",
            "shot_type": "MS",
            "content_category": "character_focus",
            "aesthetic_family": "photorealistic",
            "motion_complexity": "moderate_natural",
        },
        "annotation_path": None,
        "licenses": {
            "media_license": "CC-BY-4.0",
            "annotation_license": "CC-BY-4.0",
            "redistribution_allowed": True,
        },
        "risk_declarations": {
            "contains_real_person": False,
            "contains_logo_or_trademark": False,
            "contains_protected_character": False,
            "contains_personal_data": False,
        },
        "contribution_status": "community-submitted",
    }
    payload.update(overrides)
    return payload


def test_example_manifest_validates_without_media_download():
    path = Path(__file__).parents[1] / "examples/community_dataset/manifest.jsonl"
    records, errors = validate_manifest_rows(path)
    assert not errors
    assert len(records) == 5
    assert manifest_summary(records)["taxonomy_version"] == "0.3.1"


def test_locator_requires_exactly_one_url_or_relative_path():
    payload = _record()
    payload["media"]["relative_path"] = "clips/clip.mp4"
    with pytest.raises(ValidationError, match="exactly one"):
        CommunityManifestRecord.model_validate(payload)


@pytest.mark.parametrize("value", ["not-a-checksum", "a" * 63, "g" * 64])
def test_checksum_is_strict(value):
    payload = _record()
    payload["media"]["sha256"] = value
    with pytest.raises(ValidationError, match="sha256"):
        CommunityManifestRecord.model_validate(payload)


def test_context_tags_are_closed():
    payload = _record()
    payload["context"]["motion_complexity"] = "invented_bucket"
    with pytest.raises(ValidationError, match="taxonomy context tag"):
        CommunityManifestRecord.model_validate(payload)


def test_unknown_status_is_rejected():
    payload = _record(contribution_status="accepted")
    with pytest.raises(ValidationError, match="contribution_status"):
        CommunityManifestRecord.model_validate(payload)


def test_benchmark_requires_review_annotation_and_license():
    payload = _record(
        contribution_status="benchmark-eligible",
        annotation_path=None,
        human_verdict="PASS",
    )
    with pytest.raises(ValidationError, match="annotation_path"):
        CommunityManifestRecord.model_validate(payload)

    payload = _record(
        contribution_status="benchmark-eligible",
        annotation_path="annotations/clip-001.json",
        human_verdict="PASS",
    )
    payload["licenses"]["media_license"] = None
    with pytest.raises(ValidationError, match="media_license"):
        CommunityManifestRecord.model_validate(payload)


def test_missing_prompt_and_reference_are_explicit():
    payload = _record()
    payload["provenance"]["prompt_path"] = "private/prompt.txt"
    with pytest.raises(ValidationError, match="prompt_path"):
        CommunityManifestRecord.model_validate(payload)


def test_optional_provider_metadata_round_trips_on_legacy_annotation():
    record = AnnotationRecord(
        asset_id="legacy",
        verdict="PASS",
        provider="provider-a",
        model="model-a",
        generation_mode="text_to_video",
        prompt_available=False,
        reference_available=False,
    )
    loaded = AnnotationRecord.model_validate(record.model_dump())
    assert loaded.provider == "provider-a"
    assert loaded.prompt_available is False
    assert AnnotationRecord.model_validate({"asset_id": "old"}).provider == ""


def test_cli_reports_invalid_line_and_nonzero_exit(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(_record()) + "\n" + "{not-json}\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["validate-manifest", str(path)])
    assert result.exit_code != 0
    assert "line 2" in result.stdout


def test_cli_validates_examples(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["validate-manifest", str(path)])
    assert result.exit_code == 0
    assert "1 community manifest records valid" in result.stdout
