"""Pluggable metrics backends: config, shared interface contract, fallback."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from video_screener.config import PipelineConfig
from video_screener.utils import metrics as M
from video_screener.utils.metrics_backend import (
    FfprobeMetricsBackend,
    OpenCVMetricsBackend,
    TechnicalAssessment,
    build_metrics_backend,
    segment_metric_summary,
)
from tests.conftest import requires_ffmpeg

SERIES_KEYS = {"time_sec", "brightness", "temporal"}


@pytest.fixture
def frame_dir(tmp_path):
    """A few readable frames with mid-grey content."""
    d = tmp_path / "frames"
    d.mkdir()
    rng = np.random.RandomState(0)
    paths = []
    for i in range(4):
        arr = np.clip(rng.randint(100, 160, (48, 64, 3)), 0, 255).astype(np.uint8)
        p = d / f"f{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    return paths


# ------------------------------- config ------------------------------------
def test_config_backend_default_and_validation():
    # default flipped to ffprobe after the item-4 verdict-parity gate passed
    assert PipelineConfig().metrics_backend == "ffprobe"
    assert PipelineConfig(metrics_backend="opencv").metrics_backend == "opencv"
    with pytest.raises(Exception):
        PipelineConfig(metrics_backend="imagemagick")


def test_build_backend_resolves_by_name():
    assert isinstance(build_metrics_backend(PipelineConfig()), FfprobeMetricsBackend)
    cfg = PipelineConfig(metrics_backend="opencv")
    assert isinstance(build_metrics_backend(cfg), OpenCVMetricsBackend)


# --------------------------- interface contract ----------------------------
def _check_assessment(ta: TechnicalAssessment):
    for s in (ta.sharpness, ta.exposure, ta.temporal):
        assert isinstance(s, int) and 0 <= s <= 4
    assert isinstance(ta.fix_signals, list)
    assert all(isinstance(s, str) for s in ta.fix_signals)
    assert isinstance(ta.evidence, dict) and ta.evidence.get("backend")


def _check_series(series: list[dict]):
    assert isinstance(series, list) and series
    for fr in series:
        assert SERIES_KEYS <= set(fr)
        assert 0.0 <= fr["brightness"] <= 255.0
        assert fr["temporal"] is None or fr["temporal"] >= 0.0


def test_opencv_backend_contract(frame_dir):
    cfg = PipelineConfig()
    be = OpenCVMetricsBackend()
    ta = be.assess(None, frame_dir, cfg.prelabel)
    _check_assessment(ta)
    _check_series(be.frame_series(None, frame_dir))


@requires_ffmpeg
def test_ffprobe_backend_contract(samples_dir, frame_dir):
    cfg = PipelineConfig()
    be = FfprobeMetricsBackend()
    ta = be.assess(str(samples_dir / "clean_pass.mp4"), frame_dir, cfg.prelabel)
    _check_assessment(ta)
    assert ta.evidence["backend"] == "ffprobe"            # no fallback needed
    series = be.frame_series(str(samples_dir / "clean_pass.mp4"), frame_dir)
    _check_series(series)
    assert all(fr["time_sec"] is not None for fr in series)  # native timestamps


def test_opencv_backend_matches_legacy_functions(frame_dir):
    """The opencv backend is the original measurement path: scores must be
    bit-identical to calling utils.metrics directly (default behaviour)."""
    cfg = PipelineConfig()
    ta = OpenCVMetricsBackend().assess(None, frame_dir, cfg.prelabel)
    cm = M.compute_clip_metrics(frame_dir)
    assert ta.sharpness == M.sharpness_score(cm.sharpness_var, cfg.prelabel)
    exp_s, exp_fix = M.exposure_score(cm.brightness, cm.clip_fraction, cfg.prelabel)
    temp_s, temp_fix = M.temporal_score(cm.brightness_std, cfg.prelabel)
    assert (ta.exposure, ta.temporal) == (exp_s, temp_s)
    assert ta.fix_signals == exp_fix + temp_fix


@requires_ffmpeg
def test_ffprobe_backend_falls_back_on_unreadable_video(samples_dir, frame_dir):
    """A clip ffprobe can't decode must not crash prelabel: the backend falls
    back to the opencv statistics over the sampled frames."""
    cfg = PipelineConfig()
    be = FfprobeMetricsBackend()
    ta = be.assess(str(samples_dir / "corrupted.mp4"), frame_dir, cfg.prelabel)
    _check_assessment(ta)
    assert "fallback" in ta.evidence["backend"]
    ocv = OpenCVMetricsBackend().assess(None, frame_dir, cfg.prelabel)
    assert (ta.sharpness, ta.exposure, ta.temporal) == (
        ocv.sharpness, ocv.exposure, ocv.temporal
    )


# ------------------------- segment summary (evidence) ----------------------
def test_segment_metric_summary_math():
    series = (
        [{"time_sec": 0.0, "brightness": 10.0, "temporal": None}]
        + [{"time_sec": float(t), "brightness": 100.0, "temporal": 1.0}
           for t in range(1, 9)]
        + [{"time_sec": 9.0, "brightness": 100.0, "temporal": 30.0}]
    )
    out = segment_metric_summary(series, duration=9.0, window_sec=1.0)
    assert out["head"]["n_frames"] == 1 and out["tail"]["n_frames"] == 1
    assert out["head"]["brightness_mean"] == 10.0
    assert out["head"]["temporal_mean"] is None          # no temporal at frame 0
    assert out["tail"]["temporal_mean"] == 30.0
    assert out["body"]["brightness_mean"] == 100.0


def test_segment_metric_summary_degenerate():
    assert segment_metric_summary([], duration=5.0) is None
    assert segment_metric_summary(
        [{"time_sec": None, "brightness": 1.0, "temporal": None}], duration=5.0
    ) is None
    # everything inside the head window -> no body baseline
    assert segment_metric_summary(
        [{"time_sec": 0.1, "brightness": 1.0, "temporal": None}], duration=0.5
    ) is None
