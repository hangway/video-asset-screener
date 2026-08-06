"""ffprobe/lavfi per-frame signal metrics (QCTools lineage).

One ffprobe invocation per clip runs the decoded frames through ffmpeg's
``signalstats`` (with the optional TOUT temporal-outlier stat enabled) and
``blurdetect`` filters and returns their per-frame measurements:

- ``YAVG``  : average luma (0-255) — brightness
- ``YDIF``  : mean absolute luma difference vs the previous frame — flicker /
              temporal instability (0 for a static frame pair)
- ``YLOW``  : luma 10th percentile — shadows
- ``YHIGH`` : luma 90th percentile — highlights
- ``TOUT``  : fraction of temporal-outlier pixels (dropout/noise candidates)
- ``blur``  : blurdetect edge-width blurriness (higher = blurrier)

These are the standardized signals popularized by the QCTools archive-QC
project; they replace hand-rolled OpenCV statistics without touching the
taxonomy — this is a measurement backend, not a policy change.

Decode failures never raise: a corrupted clip yields ``ok=False`` with the
ffprobe error preserved, so callers keep their own delivery-failure handling.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# canonical per-frame record keys, in emission order
SIGNAL_KEYS = ("YAVG", "YDIF", "YLOW", "YHIGH", "TOUT", "blur")

_TAG_MAP = {
    "lavfi.signalstats.YAVG": "YAVG",
    "lavfi.signalstats.YDIF": "YDIF",
    "lavfi.signalstats.YLOW": "YLOW",
    "lavfi.signalstats.YHIGH": "YHIGH",
    "lavfi.signalstats.TOUT": "TOUT",
    "lavfi.blur": "blur",
}

_FILTERGRAPH = "signalstats=stat=tout,blurdetect"


@dataclass
class SignalStats:
    """Per-frame signal metrics for one clip. ``frames`` entries are dicts
    with ``time_sec`` + the SIGNAL_KEYS (missing measurements are None)."""

    frames: list[dict] = field(default_factory=list)
    ok: bool = False
    error: str = ""


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _escape_lavfi_path(path: str) -> str:
    """Quote a filename for the ``movie=`` lavfi source.

    ffmpeg applies TWO escaping levels: the filtergraph parser strips the
    outer single quotes (quoted text is literal at that level), then the
    movie filter's own option parser splits on ``:`` and unescapes ``\\``.
    The path therefore needs option-level escaping *inside* the quotes:
    ``\\`` -> ``\\\\`` and ``:`` -> ``\\:``. Without the colon escape a
    Windows drive letter terminates the filename option at ``C`` and every
    ffprobe-backed metric fails on Windows (audit A5, PR #10)."""
    return ("'"
            + path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            + "'")


def parse_signal_frames(doc: dict) -> list[dict]:
    """Parse ffprobe ``-of json`` output into per-frame records."""
    out: list[dict] = []
    for fr in doc.get("frames", []) or []:
        tags = fr.get("tags") or {}
        rec: dict = {"time_sec": _to_float(fr.get("pts_time"))}
        for tag, key in _TAG_MAP.items():
            rec[key] = _to_float(tags.get(tag))
        out.append(rec)
    return out


@dataclass
class IntervalScan:
    """freezedetect/blackdetect intervals for one clip (seconds).

    Intervals are ``{"start": s, "end": e}``; a detection still open at end
    of stream is closed at the last decoded timestamp."""

    freeze_intervals: list[dict] = field(default_factory=list)
    black_intervals: list[dict] = field(default_factory=list)
    last_pts: float | None = None
    ok: bool = False
    error: str = ""


_FREEZE_START = "lavfi.freezedetect.freeze_start"
_FREEZE_END = "lavfi.freezedetect.freeze_end"
_BLACK_START = "lavfi.black_start"
_BLACK_END = "lavfi.black_end"


def parse_interval_frames(doc: dict) -> IntervalScan:
    """Pair up start/end interval tags from ffprobe JSON; close open
    intervals at the last seen timestamp."""
    scan = IntervalScan()
    open_freeze: float | None = None
    open_black: float | None = None
    for fr in doc.get("frames", []) or []:
        pts = _to_float(fr.get("pts_time"))
        if pts is not None:
            scan.last_pts = pts
        tags = fr.get("tags") or {}
        fs, fe = _to_float(tags.get(_FREEZE_START)), _to_float(tags.get(_FREEZE_END))
        bs, be = _to_float(tags.get(_BLACK_START)), _to_float(tags.get(_BLACK_END))
        if fs is not None:
            open_freeze = fs
        if fe is not None and open_freeze is not None:
            scan.freeze_intervals.append({"start": open_freeze, "end": fe})
            open_freeze = None
        if bs is not None:
            open_black = bs
        if be is not None and open_black is not None:
            scan.black_intervals.append({"start": open_black, "end": be})
            open_black = None
    end = scan.last_pts
    if open_freeze is not None and end is not None and end > open_freeze:
        scan.freeze_intervals.append({"start": open_freeze, "end": end})
    if open_black is not None and end is not None and end > open_black:
        scan.black_intervals.append({"start": open_black, "end": end})
    return scan


def probe_intervals(path: str | Path, freeze_noise_db: float = -60.0,
                    freeze_min_sec: float = 1.0, black_min_sec: float = 0.5,
                    black_pic_th: float = 0.98,
                    timeout: int = 300) -> IntervalScan:
    """Detect frozen-video and black intervals in ONE ffprobe call
    (``freezedetect`` + ``blackdetect``). Never raises; decode failures
    return ``ok=False`` with the diagnostic preserved."""
    graph = (
        f"movie={_escape_lavfi_path(str(path))},"
        f"freezedetect=n={freeze_noise_db}dB:d={freeze_min_sec},"
        f"blackdetect=d={black_min_sec}:pic_th={black_pic_th}"
    )
    tags = ",".join([_FREEZE_START, _FREEZE_END, _BLACK_START, _BLACK_END])
    cmd = [
        "ffprobe", "-v", "error", "-f", "lavfi", "-i", graph,
        "-show_entries", f"frame=pts_time:frame_tags={tags}", "-of", "json",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:  # pragma: no cover - env dependent
        return IntervalScan(error=f"ffprobe exec failed: {e}")
    if res.returncode != 0:
        return IntervalScan(error=res.stderr.strip()[-400:] or "ffprobe failed")
    try:
        doc = json.loads(res.stdout)
    except json.JSONDecodeError:
        return IntervalScan(error="ffprobe produced no parseable output")
    scan = parse_interval_frames(doc)
    scan.ok = True
    return scan


def probe_signal_stats(path: str | Path, timeout: int = 300) -> SignalStats:
    """Run signalstats+blurdetect over ``path`` in ONE ffprobe call.

    Never raises: exec errors, non-zero exits (corrupt/unreadable streams),
    unparseable output, and zero decoded frames all return ``ok=False`` with
    a diagnostic ``error``.
    """
    src = f"movie={_escape_lavfi_path(str(path))},{_FILTERGRAPH}"
    entries = "frame=pts_time:frame_tags=" + ",".join(_TAG_MAP)
    cmd = [
        "ffprobe", "-v", "error", "-f", "lavfi", "-i", src,
        "-show_entries", entries, "-of", "json",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:  # pragma: no cover - env dependent
        return SignalStats(error=f"ffprobe exec failed: {e}")
    if res.returncode != 0:
        return SignalStats(error=res.stderr.strip()[-400:] or "ffprobe failed")
    try:
        doc = json.loads(res.stdout)
    except json.JSONDecodeError:
        return SignalStats(error="ffprobe produced no parseable output")
    frames = parse_signal_frames(doc)
    if not frames:
        return SignalStats(error="no frames decoded")
    return SignalStats(frames=frames, ok=True)
