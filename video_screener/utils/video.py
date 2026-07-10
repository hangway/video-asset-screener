"""Video probing + frame sampling per taxonomy.md §5.

Frame sampling rule (§5):
  - Uniform 1 fps + all scene-change / cut frames.
  - For very short clips (<4s): sample every 0.5s (or all frames).
  - Longer clips: 1 fps is usually sufficient.

Decode-failure detection: a clip that ffprobe cannot parse, or from which zero
frames can be decoded, is a delivery failure (§2.8) — surfaced to the caller so
prelabel can raise ``delivery_failure``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Silence libav/ffmpeg decode-error spam from OpenCV (e.g. when probing a
# deliberately corrupted clip). Must be set before cv2 opens any stream.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np

try:
    cv2.setLogLevel(0)  # 0 = SILENT
except Exception:  # pragma: no cover
    pass

# PySceneDetect logs an INFO line per clip; quiet it.
logging.getLogger("pyscenedetect").setLevel(logging.ERROR)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif"}


@dataclass
class VideoMeta:
    path: str
    duration_sec: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    n_frames_total: int | None = None
    probe_ok: bool = False
    error: str = ""


@dataclass
class FrameInfo:
    index: int
    time_sec: float
    path: str
    is_scene_change: bool = False


@dataclass
class SampleResult:
    frames: list[FrameInfo] = field(default_factory=list)
    decode_ok: bool = False
    n_decoded: int = 0
    error: str = ""


def probe(path: str | Path) -> VideoMeta:
    """ffprobe a video into a ``VideoMeta``. Never raises."""
    path = str(path)
    meta = VideoMeta(path=path)
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:  # pragma: no cover - env dependent
        meta.error = f"ffprobe exec failed: {e}"
        return meta
    if res.returncode != 0:
        meta.error = res.stderr.strip()[-400:] or "ffprobe failed"
        return meta
    try:
        info = json.loads(res.stdout)
    except json.JSONDecodeError:
        meta.error = "ffprobe produced no parseable output"
        return meta
    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        meta.error = "no video stream"
        return meta
    v = vstreams[0]
    meta.width = _to_int(v.get("width"))
    meta.height = _to_int(v.get("height"))
    meta.fps = _parse_rate(v.get("avg_frame_rate") or v.get("r_frame_rate"))
    dur = info.get("format", {}).get("duration") or v.get("duration")
    meta.duration_sec = _to_float(dur)
    nb = v.get("nb_frames")
    meta.n_frames_total = _to_int(nb)
    meta.probe_ok = True
    return meta


def compute_sample_times(duration_sec: float, cfg) -> list[float]:
    """Uniform sample timestamps per §5 (excludes scene-change frames, which
    are merged in separately)."""
    if duration_sec is None or duration_sec <= 0:
        return [0.0]
    if duration_sec < cfg.short_clip_threshold_sec:
        interval = cfg.short_clip_interval_sec
    else:
        interval = 1.0 / max(cfg.fps, 1e-6)
    times: list[float] = []
    t = 0.0
    while t < duration_sec - 1e-6:
        times.append(round(t, 3))
        t += interval
    if not times:
        times = [0.0]
    return times


def detect_scene_times(path: str | Path, cfg) -> list[float]:
    """Best-effort scene-change timestamps via PySceneDetect. Returns [] on any
    failure so ingest degrades gracefully."""
    if not cfg.scene_detect:
        return []
    try:
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(str(path))
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=cfg.scene_threshold))
        sm.detect_scenes(video, show_progress=False)
        scenes = sm.get_scene_list()
        # Use each scene's start time (skip the first, which is t=0).
        return [s[0].get_seconds() for s in scenes[1:]]
    except Exception:
        return []


def extract_frames(path: str | Path, cfg, out_dir: str | Path,
                   duration_sec: float | None) -> SampleResult:
    """Decode + save sampled frames. Combines uniform §5 sampling with
    best-effort scene-change frames, capped at ``cfg.max_frames``."""
    path = str(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = SampleResult()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        result.error = "cv2 could not open video"
        return result

    vfps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if duration_sec is None or duration_sec <= 0:
        duration_sec = (total / vfps) if vfps > 0 else 0.0

    uniform = set(compute_sample_times(duration_sec, cfg))
    scene = set(round(t, 3) for t in detect_scene_times(path, cfg))
    target_times = sorted(uniform | scene)
    scene_lookup = {round(t, 3) for t in scene}

    # Cap the number of target times (keep uniform coverage).
    if len(target_times) > cfg.max_frames:
        idxs = np.linspace(0, len(target_times) - 1, cfg.max_frames).astype(int)
        target_times = [target_times[i] for i in sorted(set(idxs))]

    # Sequentially read, selecting the frame whose timestamp first passes each
    # target time. Robust across codecs where seeking is unreliable.
    targets = list(target_times)
    ti = 0
    frame_idx = 0
    saved = 0
    fmt = cfg.frame_format
    while ti < len(targets):
        ok, frame = cap.read()
        if not ok:
            break
        cur_t = (frame_idx / vfps) if vfps > 0 else float(frame_idx)
        # advance past any targets we've already reached
        while ti < len(targets) and cur_t + 1e-6 >= targets[ti]:
            t = targets[ti]
            fp = out_dir / f"frame_{saved:04d}.{fmt}"
            try:
                cv2.imwrite(str(fp), frame)
            except Exception as e:  # pragma: no cover
                result.error = f"imwrite failed: {e}"
                cap.release()
                return result
            result.frames.append(
                FrameInfo(index=saved, time_sec=round(t, 3), path=str(fp),
                          is_scene_change=round(t, 3) in scene_lookup)
            )
            saved += 1
            ti += 1
        frame_idx += 1

    cap.release()

    # If we exhausted the video before hitting all targets but did decode some
    # frames, that's fine. Zero decoded frames => decode failure.
    result.n_decoded = saved
    result.decode_ok = saved > 0
    if saved == 0 and not result.error:
        result.error = "zero frames decoded (corrupt/unplayable stream)"
    return result


# --------------------------- small helpers --------------------------------
def _to_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_rate(r: str | None) -> float | None:
    if not r or r == "0/0":
        return None
    if "/" in r:
        num, den = r.split("/")
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else None
        except ValueError:
            return None
    return _to_float(r)


def find_videos(dirs: list[str]) -> list[Path]:
    """Recursively find video files under the given directories, sorted."""
    found: list[Path] = []
    for d in dirs:
        base = Path(d)
        if base.is_file() and base.suffix.lower() in VIDEO_EXTS:
            found.append(base)
            continue
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                found.append(p)
    # stable, de-duplicated
    seen: set[str] = set()
    out: list[Path] = []
    for p in sorted(found):
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out
