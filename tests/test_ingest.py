"""Stage 1 ingest: §5 frame sampling, perceptual-hash dedup, decode failure."""

from __future__ import annotations

from video_screener.config import PipelineConfig
from video_screener.stages import ingest
from video_screener.utils import hashing, video
from tests.conftest import requires_ffmpeg


def _cfg(workdir, samples_dir) -> PipelineConfig:
    return PipelineConfig(workdir=str(workdir), video_dirs=[str(samples_dir)])


# --------------------------- §5 sampling math -----------------------------
def test_sample_times_1fps_for_long_clip():
    cfg = PipelineConfig().ingest
    # 5s clip at 1 fps -> frames at 0,1,2,3,4 (5 frames)
    times = video.compute_sample_times(5.0, cfg)
    assert times == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_sample_times_half_second_for_short_clip():
    cfg = PipelineConfig().ingest
    # <4s clip -> every 0.5s. 2s clip -> 0,0.5,1.0,1.5 (4 frames)
    times = video.compute_sample_times(2.0, cfg)
    assert times == [0.0, 0.5, 1.0, 1.5]


def test_sample_times_below_half_second_yields_one():
    cfg = PipelineConfig().ingest
    times = video.compute_sample_times(0.29, cfg)
    assert times == [0.0]


# --------------------------- dedup math -----------------------------------
def test_signature_distance_identical_is_zero(tmp_path):
    # Two copies of the same image -> distance 0.
    from PIL import Image
    import numpy as np

    img = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype("uint8")
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    Image.fromarray(img).save(p1)
    Image.fromarray(img).save(p2)
    s1 = hashing.clip_signature([p1, p1, p1, p1])
    s2 = hashing.clip_signature([p2, p2, p2, p2])
    assert hashing.signature_distance(s1, s2) == 0.0


def test_cluster_merges_dups_only():
    # sig a==b (dist 0), c far. threshold 4 -> {a,b} one cluster, c separate.
    a = ["0000000000000000"]
    b = ["0000000000000000"]
    c = ["ffffffffffffffff"]
    clusters = hashing.cluster_near_duplicates([a, b, c], threshold=4)
    assert clusters[0] == clusters[1]
    assert clusters[2] != clusters[0]


def test_empty_signature_never_dups():
    assert hashing.signature_distance([], ["abc"]) == float("inf")
    clusters = hashing.cluster_near_duplicates([[], []], threshold=100)
    assert clusters[0] != clusters[1]  # two decode-failed clips stay separate


# --------------------------- end-to-end on samples ------------------------
@requires_ffmpeg
def test_ingest_end_to_end(tmp_path, synth_samples):
    cfg = _cfg(tmp_path / "run", synth_samples)
    summary = ingest.run(cfg)
    assert summary["n_assets"] == 8
    # exactly one byte-duplicate (duplicate.mp4 == clean_pass.mp4)
    assert summary["n_duplicates"] == 1
    # exactly one decode failure (corrupted.mp4)
    assert summary["n_decode_failures"] == 1

    import json

    idx = json.loads((tmp_path / "run" / "ingest" / "index.json").read_text())
    # A6: the index stamps the EFFECTIVE encoder (what actually runs)
    assert isinstance(idx["encoder"], str) and idx["encoder"]
    by_id = {a["asset_id"]: a for a in idx["assets"]}

    # duplicate detected and points at clean_pass
    assert by_id["duplicate"]["is_duplicate"] is True
    assert by_id["duplicate"]["duplicate_of"] == "clean_pass"
    assert by_id["clean_pass"]["is_duplicate"] is False

    # corrupted -> decode failure, zero frames
    assert by_id["corrupted"]["decode_ok"] is False
    assert by_id["corrupted"]["n_sampled"] == 0

    # long clip sampled at ~1 fps (5s -> 5 frames)
    assert by_id["clean_pass"]["n_sampled"] == 5

    # distinct clips are NOT merged with clean_pass
    assert by_id["watermark"]["cluster_id"] != by_id["clean_pass"]["cluster_id"]
    assert by_id["flicker"]["cluster_id"] != by_id["clean_pass"]["cluster_id"]


@requires_ffmpeg
def test_ingest_frames_written(tmp_path, synth_samples):
    cfg = _cfg(tmp_path / "run", synth_samples)
    ingest.run(cfg)
    frames = list((tmp_path / "run" / "ingest" / "frames" / "clean_pass").glob("*.jpg"))
    assert len(frames) == 5
