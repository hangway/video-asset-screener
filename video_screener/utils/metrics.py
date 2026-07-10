"""Frame-level technical metrics + score mapping for prelabel heuristics.

All metrics are computed from decoded frames (no network, deterministic).
Scores are mapped onto the taxonomy 0-4 scale using thresholds from
``PrelabelConfig`` (calibrated on the synthetic samples).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ClipMetrics:
    sharpness_var: float = 0.0        # mean variance-of-Laplacian
    brightness: float = 0.0          # mean luma (0-255)
    clip_fraction: float = 0.0       # fraction of pure-black/white pixels
    brightness_std: float = 0.0      # std of per-frame mean luma (flicker)
    n_frames: int = 0


def compute_clip_metrics(frame_paths: list[str]) -> ClipMetrics:
    """Aggregate frame metrics across a clip's sampled frames."""
    sharps: list[float] = []
    brights: list[float] = []
    clips: list[float] = []
    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharps.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        brights.append(float(g.mean()))
        clips.append(float((g < 8).mean() + (g > 247).mean()))
    if not sharps:
        return ClipMetrics()
    return ClipMetrics(
        sharpness_var=float(np.mean(sharps)),
        brightness=float(np.mean(brights)),
        clip_fraction=float(np.mean(clips)),
        brightness_std=float(np.std(brights)) if len(brights) > 1 else 0.0,
        n_frames=len(sharps),
    )


def sharpness_score(var: float, cfg) -> int:
    """Map variance-of-Laplacian to a 0-4 sharpness score."""
    return int(sum(1 for t in cfg.sharpness_bins if var >= t))


def exposure_score(brightness: float, clip_fraction: float, cfg) -> tuple[int, list[str]]:
    """Map brightness + clipping to a 0-4 exposure score + fix signals."""
    if clip_fraction >= cfg.exposure_clip_fraction:
        return 0, []  # severe clipping, critical detail loss -> unusable
    if brightness < cfg.exposure_severe_dark or brightness > cfg.exposure_severe_bright:
        return 1, ["color_grade"]
    if brightness < cfg.exposure_dark:
        return 2, ["color_grade", "exposure_correction"]  # underexposed, recoverable
    if brightness > cfg.exposure_bright:
        return 2, ["color_grade"]                          # overexposed, recoverable
    return 3, []  # well-exposed (4 reserved for verified full histogram)


def temporal_score(brightness_std: float, cfg) -> tuple[int, list[str]]:
    """Map per-frame brightness oscillation (flicker) to a 0-4 temporal score
    + fix signals. Returns (score, fix_signals)."""
    if brightness_std >= cfg.flicker_high:
        return 1, []                     # severe flicker -> hard to fix (REJECT path)
    if brightness_std >= cfg.flicker_low:
        return 2, ["deflicker"]          # borderline flicker -> FIX
    return 3, []                          # stable (4 reserved for verified rock-solid)
