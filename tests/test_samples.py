"""Sanity checks on the synthesized sample clips + sidecars."""

from __future__ import annotations

import json

from tests.conftest import requires_ffmpeg
from video_screener.schema import AnnotationRecord

EXPECTED = {
    "clean_pass": "PASS",
    "watermark": "REJECT",
    "corrupted": "REJECT",
    "flicker": "FIX",
    "duplicate": "PASS",
    "lowres": "REJECT",
    "tooshort": "REJECT",
    "underexposed": "FIX",
}


@requires_ffmpeg
def test_all_samples_and_sidecars_present(synth_samples):
    for name in EXPECTED:
        assert (synth_samples / f"{name}.mp4").exists(), f"missing {name}.mp4"
        assert (synth_samples / f"{name}.json").exists(), f"missing {name}.json"


@requires_ffmpeg
def test_sidecar_expected_verdicts(synth_samples):
    for name, verdict in EXPECTED.items():
        data = json.loads((synth_samples / f"{name}.json").read_text())
        assert data["expected_verdict"] == verdict


@requires_ffmpeg
def test_sidecars_convert_to_valid_annotation_records(synth_samples):
    """Every sidecar's expected labels must form a schema-valid §7.1 record."""
    for name in EXPECTED:
        data = json.loads((synth_samples / f"{name}.json").read_text())
        rec = _sidecar_to_record(data)
        AnnotationRecord.model_validate(rec)  # raises on any taxonomy violation


@requires_ffmpeg
def test_duplicate_is_byte_identical_to_clean(synth_samples):
    a = (synth_samples / "clean_pass.mp4").read_bytes()
    b = (synth_samples / "duplicate.mp4").read_bytes()
    assert a == b, "duplicate.mp4 must be a byte-copy of clean_pass.mp4"


def _sidecar_to_record(data: dict) -> dict:
    """Build a minimal valid §7.1 dict from a sidecar's expected fields."""
    verdict = data["expected_verdict"]
    flags = data.get("expected_flags", [])
    scores = data.get("expected_scores", {}) or {}
    rec = {
        "asset_id": data["asset_id"],
        "file_path": data["file"],
        "duration_sec": 5.0,
        "verdict": verdict,
        "hard_fail_flags": flags,
        "scores": scores,
        "fix_actions": data.get("fix_actions", []),
        "reject_reason": data.get("reject_reason", ""),
        "context_tags": data.get("context_tags", {}),
    }
    if flags:
        rec["frame_evidence"] = [
            {"frame_time_sec": 0.0, "issue": data.get("note", ""), "flag_or_dimension": f}
            for f in flags
        ]
    return rec
