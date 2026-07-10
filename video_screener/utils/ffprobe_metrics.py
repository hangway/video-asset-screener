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
    """Quote a filename for the ``movie=`` lavfi source: single-quote the
    whole path and backslash-escape embedded backslashes and quotes."""
    return "'" + path.replace("\\", "\\\\").replace("'", "\\'") + "'"


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
