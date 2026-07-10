"""freezedetect/blackdetect objective signals: parsing, new synthetic clips,
prelabel evidence rules (existing delivery_failure flag only — no new flags)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from video_screener.config import PipelineConfig
from video_screener.schema import AnnotationRecord
from video_screener.stages import ingest, prelabel
from video_screener.stages.prelabel import _prelabel_asset
from video_screener.utils.ffprobe_metrics import parse_interval_frames, probe_intervals
from tests.conftest import have_ffmpeg, requires_ffmpeg


# ------------------------------ parsing ------------------------------------
def test_parse_interval_frames_pairs_and_closes_open_intervals():
    doc = {"frames": [
        {"pts_time": "0.0", "tags": {}},
        {"pts_time": "1.5", "tags": {"lavfi.black_start": "1.0"}},
        {"pts_time": "2.0", "tags": {"lavfi.black_end": "2.0"}},
        {"pts_time": "3.0", "tags": {"lavfi.freezedetect.freeze_start": "2.5"}},
        {"pts_time": "4.9", "tags": {}},
    ]}
    scan = parse_interval_frames(doc)
    assert scan.black_intervals == [{"start": 1.0, "end": 2.0}]
    # freeze ran to end of stream -> closed at the last decoded timestamp
    assert scan.freeze_intervals == [{"start": 2.5, "end": 4.9}]
    assert scan.last_pts == 4.9


def test_parse_interval_frames_empty():
    scan = parse_interval_frames({})
    assert scan.freeze_intervals == [] and scan.black_intervals == []
    assert scan.last_pts is None


# --------------------- new synthetic clips end to end ----------------------
@pytest.fixture(scope="module")
def stability_samples(tmp_path_factory):
    if not have_ffmpeg():
        pytest.skip("ffmpeg/ffprobe not available")
    from samples.make_samples import build_stability_samples

    out = tmp_path_factory.mktemp("stability_samples")
    build_stability_samples(out)
    return out


@requires_ffmpeg
def test_frozen_and_black_clips_detected(stability_samples):
    frozen = probe_intervals(stability_samples / "frozen.mp4")
    assert frozen.ok
    cov = sum(iv["end"] - iv["start"] for iv in frozen.freeze_intervals)
    assert cov >= 0.9 * frozen.last_pts        # frozen essentially throughout
    black = probe_intervals(stability_samples / "black.mp4")
    assert black.ok
    cov = sum(iv["end"] - iv["start"] for iv in black.black_intervals)
    assert cov >= 0.9 * black.last_pts


@requires_ffmpeg
def test_prelabel_rejects_frozen_and_black_via_delivery_failure(tmp_path, stability_samples):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"),
                         video_dirs=[str(stability_samples)])
    ingest.run(cfg)
    prelabel.run(cfg)
    rows = [json.loads(l) for l in
            (tmp_path / "run" / "prelabel" / "prelabels.jsonl").read_text().splitlines()]
    recs = {r["asset_id"]: r for r in rows}
    for name, kind in (("frozen", "frozen"), ("black", "black")):
        r = recs[name]
        AnnotationRecord.model_validate(r)
        assert r["verdict"] == "REJECT", f"{name}: {r['verdict']}"
        assert r["hard_fail_flags"] == ["delivery_failure"]   # EXISTING flag only
        assert kind in r["reject_reason"]
        sidecar = json.loads((stability_samples / f"{name}.json").read_text())
        assert r["verdict"] == sidecar["expected_verdict"]


@requires_ffmpeg
def test_ingest_records_intervals(tmp_path, stability_samples):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"),
                         video_dirs=[str(stability_samples)])
    ingest.run(cfg)
    index = json.loads((tmp_path / "run" / "ingest" / "index.json").read_text())
    by = {a["asset_id"]: a for a in index["assets"]}
    assert by["frozen"]["freeze_intervals"]
    assert by["black"]["black_intervals"]


# ------------------- partial intervals: evidence only ----------------------
def _asset_with_frames(tmp_path, **extra) -> dict:
    d = tmp_path / "frames"
    d.mkdir(exist_ok=True)
    rng = np.random.RandomState(1)
    frames = []
    for i in range(3):
        p = d / f"f{i}.png"
        Image.fromarray(rng.randint(90, 170, (48, 64, 3)).astype(np.uint8)).save(p)
        frames.append({"index": i, "time_sec": float(i), "path": str(p),
                       "is_scene_change": False})
    return {
        "asset_id": "partial", "file_path": str(tmp_path / "nonexistent.mp4"),
        "duration_sec": 10.0, "decode_ok": True, "sampled_frames": frames,
        **extra,
    }


def test_partial_freeze_is_evidence_not_delivery_failure(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    asset = _asset_with_frames(
        tmp_path, freeze_intervals=[{"start": 2.0, "end": 4.0}],  # 20% coverage
        black_intervals=[{"start": 8.0, "end": 9.0}],             # 10% coverage
    )
    rec = _prelabel_asset(asset, cfg)
    assert "delivery_failure" not in rec.hard_fail_flags       # no auto-REJECT
    evidence = rec.model_dump()["frame_evidence"]
    issues = [e["issue"] for e in evidence]
    assert any("frozen segment" in s for s in issues)
    assert any("black segment" in s for s in issues)
    dims = {e["flag_or_dimension"] for e in evidence}
    assert dims == {"temporal_stability"}                       # evidence only
    AnnotationRecord.model_validate(rec.model_dump())


def test_full_coverage_interval_rejects_synthetic(tmp_path):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    asset = _asset_with_frames(
        tmp_path, freeze_intervals=[{"start": 0.0, "end": 9.6}],  # 96% coverage
    )
    rec = _prelabel_asset(asset, cfg)
    assert rec.verdict == "REJECT"
    assert rec.hard_fail_flags == ["delivery_failure"]
    assert "frozen" in rec.reject_reason
