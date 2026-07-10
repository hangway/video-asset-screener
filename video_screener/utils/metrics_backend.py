"""Pluggable per-clip technical-metrics backends (ONE interface).

Two backends satisfy the same contract, selected by ``cfg.metrics_backend``:

- ``"opencv"`` (default): the original hand-rolled statistics over the
  ingest-sampled frames (``utils.metrics``) — variance-of-Laplacian
  sharpness, mean-luma exposure, brightness-std flicker.
- ``"ffprobe"``: ffmpeg signalstats/blurdetect per-frame signals over ALL
  decoded frames (``utils.ffprobe_metrics``, QCTools lineage) — blurdetect
  blurriness, YAVG exposure, YDIF flicker. Falls back to the opencv path if
  the probe fails (e.g. unreadable stream) so prelabel behaviour degrades
  gracefully rather than erroring.

Contract (both backends):

- ``assess(video_path, frame_paths, pcfg) -> TechnicalAssessment``
  0-4 scores for the three technical dimensions + ordered fix signals +
  raw backend-scale aggregates in ``evidence``. Metric scales differ per
  backend, so thresholds are backend-scoped in ``PrelabelConfig`` — scores
  and fix signals are the normalized output.
- ``frame_series(video_path, frame_paths) -> list[dict]``
  per-frame records ``{"time_sec": float|None, "brightness": float,
  "temporal": float|None}`` (backends may add extra keys). Used as evidence
  by the consistency edge-stability check; routing there stays
  embedding-based.

This upgrades the MEASUREMENT layer only: taxonomy v0.3.1 scoring semantics
(0-4 dims, gates, flags) are unchanged, and the embedding encoder is not
involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import metrics as M
from .ffprobe_metrics import probe_signal_stats


@dataclass
class TechnicalAssessment:
    """Normalized output of a metrics backend for one clip."""

    sharpness: int
    exposure: int
    temporal: int
    fix_signals: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


class OpenCVMetricsBackend:
    """Hand-rolled frame statistics (the original measurement path)."""

    name = "opencv"

    def assess(self, video_path, frame_paths: list[str],
               pcfg) -> TechnicalAssessment:
        cm = M.compute_clip_metrics(frame_paths)
        sharp = M.sharpness_score(cm.sharpness_var, pcfg)
        exp_s, exp_fix = M.exposure_score(cm.brightness, cm.clip_fraction, pcfg)
        temp_s, temp_fix = M.temporal_score(cm.brightness_std, pcfg)
        return TechnicalAssessment(
            sharpness=sharp, exposure=exp_s, temporal=temp_s,
            fix_signals=exp_fix + temp_fix,
            evidence={
                "backend": self.name,
                "sharpness_var": cm.sharpness_var,
                "brightness": cm.brightness,
                "clip_fraction": cm.clip_fraction,
                "brightness_std": cm.brightness_std,
                "n_frames": cm.n_frames,
            },
        )

    def frame_series(self, video_path, frame_paths: list[str]) -> list[dict]:
        out: list[dict] = []
        prev: float | None = None
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            b = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
            out.append({
                "time_sec": None,          # sampled frames carry no timestamps here
                "brightness": b,
                "temporal": abs(b - prev) if prev is not None else None,
            })
            prev = b
        return out


class FfprobeMetricsBackend:
    """ffmpeg signalstats/blurdetect signals (QCTools lineage).

    Aggregation: brightness = mean YAVG (same 0-255 scale as opencv);
    flicker = mean YDIF (mean per-frame absolute luma change — NOT the same
    scale as opencv's brightness-std); blurriness = mean blurdetect value
    (higher = blurrier — inverted sense vs variance-of-Laplacian). Thresholds
    for these scales live in backend-scoped prelabel config.
    """

    name = "ffprobe"

    def __init__(self):
        self._fallback = OpenCVMetricsBackend()

    def assess(self, video_path, frame_paths: list[str],
               pcfg) -> TechnicalAssessment:
        stats = probe_signal_stats(video_path) if video_path else None
        if stats is None or not stats.ok:
            ta = self._fallback.assess(video_path, frame_paths, pcfg)
            ta.evidence["backend"] = f"{self.name}->opencv-fallback"
            return ta
        fcfg = pcfg.ffprobe
        vals = {k: [f[k] for f in stats.frames if f.get(k) is not None]
                for k in ("YAVG", "YDIF", "YLOW", "YHIGH", "blur")}
        mean = {k: (float(np.mean(v)) if v else 0.0) for k, v in vals.items()}
        # exposure: YAVG shares the 0-255 luma scale with the opencv path; a
        # frame counts as clipped only when it is MOSTLY crushed (10th
        # percentile pinned white / 90th percentile pinned black) — matching
        # the "critical detail loss" semantics of the opencv per-pixel
        # clip_fraction, so bright-but-detailed flicker frames don't count
        n = max(len(stats.frames), 1)
        clip_frac = sum(
            1 for f in stats.frames
            if (f.get("YLOW") is not None and f["YLOW"] >= fcfg.clip_white_ylow_min)
            or (f.get("YHIGH") is not None and f["YHIGH"] <= fcfg.clip_black_yhigh_max)
        ) / n
        exp_s, exp_fix = M.exposure_score(mean["YAVG"], clip_frac, pcfg)
        # sharpness: blurdetect is higher-is-blurrier -> descending bins
        sharp = int(sum(1 for t in fcfg.blur_bins if mean["blur"] <= t))
        # temporal flicker: mean YDIF (mean abs per-pixel luma change / frame)
        if mean["YDIF"] >= fcfg.ydif_high:
            temp_s, temp_fix = 1, []
        elif mean["YDIF"] >= fcfg.ydif_low:
            temp_s, temp_fix = 2, ["deflicker"]
        else:
            temp_s, temp_fix = 3, []
        return TechnicalAssessment(
            sharpness=sharp, exposure=exp_s, temporal=temp_s,
            fix_signals=exp_fix + temp_fix,
            evidence={
                "backend": self.name,
                "yavg_mean": mean["YAVG"],
                "ydif_mean": mean["YDIF"],
                "blur_mean": mean["blur"],
                "clip_fraction": clip_frac,
                "n_frames": len(stats.frames),
            },
        )

    def frame_series(self, video_path, frame_paths: list[str]) -> list[dict]:
        stats = probe_signal_stats(video_path) if video_path else None
        if stats is None or not stats.ok:
            return self._fallback.frame_series(video_path, frame_paths)
        return [{
            "time_sec": f["time_sec"],
            "brightness": f["YAVG"] if f["YAVG"] is not None else 0.0,
            "temporal": f["YDIF"],
            "blur": f["blur"],
            "TOUT": f["TOUT"],
        } for f in stats.frames]


_BACKENDS = {
    "opencv": OpenCVMetricsBackend,
    "ffprobe": FfprobeMetricsBackend,
}


def build_metrics_backend(cfg):
    """Resolve the metrics backend from ``cfg.metrics_backend``."""
    return _BACKENDS[cfg.metrics_backend]()


def segment_metric_summary(series: list[dict], duration: float | None,
                           window_sec: float = 1.0) -> dict | None:
    """Head/tail/body mean brightness + temporal from a frame_series.

    Evidence-only companion to the consistency edge-stability check (routing
    there stays embedding-based). Returns None when the series carries no
    usable timestamps or the body would be empty.
    """
    timed = [f for f in series if f.get("time_sec") is not None]
    if not timed:
        return None
    t = np.array([f["time_sec"] for f in timed])
    end = float(duration) if duration else float(t.max())
    seg_of = np.where(t < window_sec, "head",
                      np.where(t > end - window_sec, "tail", "body"))
    if not (seg_of == "body").any():
        return None
    out: dict = {}
    for seg in ("head", "tail", "body"):
        rows = [f for f, s in zip(timed, seg_of) if s == seg]
        bright = [f["brightness"] for f in rows if f.get("brightness") is not None]
        temp = [f["temporal"] for f in rows if f.get("temporal") is not None]
        out[seg] = {
            "n_frames": len(rows),
            "brightness_mean": round(float(np.mean(bright)), 3) if bright else None,
            "temporal_mean": round(float(np.mean(temp)), 4) if temp else None,
        }
    return out
