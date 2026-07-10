"""Synthesize tiny ffmpeg-generated test clips with expected-verdict sidecars.

Each clip exercises a distinct usability condition from taxonomy.md. Sidecars
(``<name>.json``) record the *expected* human verdict/flags/scores and act as
ground-truth labels for the samples-based end-to-end run and tests.

Run directly (``python -m samples.make_samples``) or import ``build_samples``.

Distinctness matters: every clip uses a different lavfi base pattern so the
perceptual-hash dedup only fires on the intentional byte-duplicate.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

SIZE = "320x240"
FPS = 24
DUR = 5.0  # default clip duration (seconds) -> "short" bucket (3-8s)

# lavfi base sources chosen to be visually distinct from one another.
_BASES = {
    "clean_pass": "mandelbrot=size={size}:rate={fps}",
    "watermark": "gradients=size={size}:rate={fps}:x0=0:y0=0:x1={w}:y1={h}",
    "corrupted": "life=size={size}:rate={fps}:mold=10:ratio=0.1",
    "flicker": "testsrc2=size={size}:rate={fps}",
    "lowres": "mandelbrot=size={size}:rate={fps}:start_x=0.3:start_y=-0.4",
    "tooshort": "rgbtestsrc=size={size}:rate={fps}",
    "underexposed": "mandelbrot=size={size}:rate={fps}:start_x=-0.5:start_y=0.6",
}

_FALLBACK = "testsrc2=size={size}:rate={fps}"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _base_filter(name: str) -> str:
    w, h = SIZE.split("x")
    return _BASES.get(name, _FALLBACK).format(size=SIZE, fps=FPS, w=w, h=h)


def _encode(src_filter: str, extra_vf: str, out: Path, dur: float) -> None:
    """Encode a lavfi source (+ optional filter chain) to an mp4."""
    vf = src_filter
    if extra_vf:
        vf = f"{src_filter},{extra_vf}"
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", vf,
        "-t", f"{dur}", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-movflags", "+faststart", str(out),
    ]
    res = _run(cmd)
    if res.returncode != 0 or not out.exists():
        # Fall back to a guaranteed-available source, preserving the extra_vf.
        fb = _FALLBACK.format(size=SIZE, fps=FPS)
        vf = f"{fb},{extra_vf}" if extra_vf else fb
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", vf,
            "-t", f"{dur}", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-movflags", "+faststart", str(out),
        ]
        res = _run(cmd)
        if res.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {out.name}:\n{res.stderr[-800:]}"
            )


def _sidecar(out_dir: Path, name: str, data: dict) -> None:
    (out_dir / f"{name}.json").write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------
# Individual clip builders
# --------------------------------------------------------------------------
def _ctx(**kw) -> dict:
    base = dict(
        shot_type=["WS"], asset_role=["supporting"], source=["generated"],
        content_category=["abstract_motion"], aesthetic_family=["abstract_experimental"],
        motion_complexity="moderate_natural", duration_bucket="short",
        text_presence="none",
    )
    base.update(kw)
    return base


def build_samples(out_dir: str | Path) -> list[Path]:
    """Build all sample clips + sidecars into ``out_dir``. Idempotent-ish:
    always regenerates to keep bytes deterministic across ffmpeg versions."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    built: list[Path] = []

    # 1) clean PASS -------------------------------------------------------
    clean = out / "clean_pass.mp4"
    _encode(_base_filter("clean_pass"), "", clean, DUR)
    _sidecar(out, "clean_pass", {
        "asset_id": "clean_pass",
        "file": "clean_pass.mp4",
        "expected_verdict": "PASS",
        "expected_flags": [],
        "expected_scores": {
            "sharpness_focus": 4, "exposure_dynamic_range": 3,
            "composition_framing": 3, "prompt_fidelity_coherence": 3,
            "temporal_stability": 4, "motion_quality": 3,
        },
        "fix_actions": [], "reject_reason": "",
        "context_tags": _ctx(),
        "note": "sharp, well-exposed, temporally stable synthetic clip",
        "near_duplicate_of": None,
    })
    built.append(clean)

    # 2) watermark overlay -> REJECT (watermark_contamination) -----------
    wm = out / "watermark.mp4"
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fontfile = font if Path(font).exists() else ""
    fontopt = f"fontfile={fontfile}:" if fontfile else ""
    wm_vf = (
        f"drawtext={fontopt}text='(c) SCREENERTEST WATERMARK':fontcolor=white@0.85:"
        "fontsize=22:x=10:y=H-40:box=1:boxcolor=black@0.4,"
        "drawbox=x=iw-90:y=10:w=80:h=30:color=red@0.7:t=fill"
    )
    _encode(_base_filter("watermark"), wm_vf, wm, DUR)
    _sidecar(out, "watermark", {
        "asset_id": "watermark",
        "file": "watermark.mp4",
        "expected_verdict": "REJECT",
        "expected_flags": ["watermark_contamination"],
        "expected_scores": {
            "sharpness_focus": 3, "exposure_dynamic_range": 3,
            "composition_framing": 3, "prompt_fidelity_coherence": 2,
            "temporal_stability": 3, "motion_quality": 3,
        },
        "fix_actions": [],
        "reject_reason": "burned-in watermark / logo overlay (watermark_contamination)",
        "context_tags": _ctx(text_presence="prominent", content_category=["text_overlay"]),
        "note": "burned-in watermark text + logo box in corner",
        "near_duplicate_of": None,
    })
    built.append(wm)

    # 3) truncated/corrupted file -> REJECT (delivery_failure) -----------
    corrupt_full = out / "_corrupt_full.mp4"
    _encode(_base_filter("corrupted"), "", corrupt_full, DUR)
    corrupt = out / "corrupted.mp4"
    raw = corrupt_full.read_bytes()
    # Truncate hard to break the moov/mdat structure -> unprobeable/unplayable.
    corrupt.write_bytes(raw[: max(2048, len(raw) // 40)])
    corrupt_full.unlink(missing_ok=True)
    _sidecar(out, "corrupted", {
        "asset_id": "corrupted",
        "file": "corrupted.mp4",
        "expected_verdict": "REJECT",
        "expected_flags": ["delivery_failure"],
        "expected_scores": {},
        "fix_actions": [],
        "reject_reason": "truncated/corrupted stream, cannot be decoded (delivery_failure)",
        "context_tags": _ctx(),
        "note": "valid mp4 truncated to a tiny prefix -> decode failure",
        "near_duplicate_of": None,
    })
    built.append(corrupt)

    # 4) flicker injection -> FIX (deflicker) ----------------------------
    fl = out / "flicker.mp4"
    # hue brightness oscillation at ~0.35 Hz -> visible global-brightness
    # pumping even when sampled at 1 fps (§5). This isolates flicker from motion.
    fl_vf = "hue=b=4*sin(2*PI*t*0.35)"
    _encode(_base_filter("flicker"), fl_vf, fl, DUR)
    _sidecar(out, "flicker", {
        "asset_id": "flicker",
        "file": "flicker.mp4",
        "expected_verdict": "FIX",
        "expected_flags": [],
        "expected_scores": {
            "sharpness_focus": 3, "exposure_dynamic_range": 2,
            "composition_framing": 3, "prompt_fidelity_coherence": 3,
            "temporal_stability": 2, "motion_quality": 3,
        },
        "fix_actions": ["deflicker"],
        "reject_reason": "",
        "context_tags": _ctx(motion_complexity="moderate_natural"),
        "note": "periodic brightness flicker -> temporal_stability borderline, deflickerable",
        "near_duplicate_of": None,
    })
    built.append(fl)

    # 5) byte-duplicate of clean -> dedup target -------------------------
    dup = out / "duplicate.mp4"
    shutil.copyfile(clean, dup)
    _sidecar(out, "duplicate", {
        "asset_id": "duplicate",
        "file": "duplicate.mp4",
        "expected_verdict": "PASS",
        "expected_flags": [],
        "expected_scores": {
            "sharpness_focus": 4, "exposure_dynamic_range": 3,
            "composition_framing": 3, "prompt_fidelity_coherence": 3,
            "temporal_stability": 4, "motion_quality": 3,
        },
        "fix_actions": [], "reject_reason": "",
        "context_tags": _ctx(),
        "note": "exact byte-duplicate of clean_pass.mp4 (ingest dedup must catch it)",
        "near_duplicate_of": "clean_pass",
    })
    built.append(dup)

    # 6) very-low-res -> REJECT (sharpness severe) -----------------------
    lr = out / "lowres.mp4"
    # Downscale hard then bilinear-upscale -> genuine softening/blur (details
    # unreadable), not blocky nearest-neighbor edges.
    lr_vf = f"scale=40:30,scale={SIZE.replace('x', ':')},boxblur=2:1"
    _encode(_base_filter("lowres"), lr_vf, lr, DUR)
    _sidecar(out, "lowres", {
        "asset_id": "lowres",
        "file": "lowres.mp4",
        "expected_verdict": "REJECT",
        "expected_flags": [],
        "expected_scores": {
            "sharpness_focus": 0, "exposure_dynamic_range": 2,
            "composition_framing": 2, "prompt_fidelity_coherence": 2,
            "temporal_stability": 2, "motion_quality": 2,
        },
        "fix_actions": [],
        "reject_reason": "severe resolution loss, main subject unreadable (sharpness_focus=0)",
        "context_tags": _ctx(),
        "note": "downscaled to 48x36 then upscaled -> unrecoverable blur",
        "near_duplicate_of": None,
    })
    built.append(lr)

    # 7) below-min-duration (<0.5s) -> REJECT ----------------------------
    ts = out / "tooshort.mp4"
    _encode(_base_filter("tooshort"), "", ts, 0.3)
    _sidecar(out, "tooshort", {
        "asset_id": "tooshort",
        "file": "tooshort.mp4",
        "expected_verdict": "REJECT",
        # NOTE: taxonomy has no dedicated "too_short" flag; sub-minimum duration
        # is mapped to delivery_failure as a documented placeholder (see notes.md).
        "expected_flags": ["delivery_failure"],
        "expected_scores": {},
        "fix_actions": [],
        "reject_reason": "duration 0.3s is below the usable minimum (delivery_failure placeholder)",
        "context_tags": _ctx(duration_bucket="very_short"),
        "note": "0.3s clip, below the <0.5s / min-usable-duration floor (§5)",
        "near_duplicate_of": None,
    })
    built.append(ts)

    # 8) borderline underexposure -> FIX (color grade) -------------------
    ue = out / "underexposed.mp4"
    ue_vf = "eq=brightness=-0.32:contrast=0.72"
    _encode(_base_filter("underexposed"), ue_vf, ue, DUR)
    _sidecar(out, "underexposed", {
        "asset_id": "underexposed",
        "file": "underexposed.mp4",
        "expected_verdict": "FIX",
        "expected_flags": [],
        "expected_scores": {
            "sharpness_focus": 3, "exposure_dynamic_range": 2,
            "composition_framing": 3, "prompt_fidelity_coherence": 3,
            "temporal_stability": 3, "motion_quality": 3,
        },
        "fix_actions": ["color_grade", "exposure_correction"],
        "reject_reason": "",
        "context_tags": _ctx(aesthetic_family=["stylized_cinematic"]),
        "note": "mild underexposure + flat contrast -> recoverable via grading",
        "near_duplicate_of": None,
    })
    built.append(ue)

    # Manifest of expected outcomes for convenience.
    manifest = {
        "taxonomy_version": "0.3.1",
        "clips": [p.name for p in built],
        "count": len(built),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return built


def build_stability_samples(out_dir: str | Path) -> list[Path]:
    """Extra clips exercising the objective freezedetect/blackdetect signals
    (frozen video, all-black video). Kept OUT of the core 8-clip set so the
    long-standing sample counts stay stable; tests build these on demand."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    built: list[Path] = []

    # a) fully frozen video: one detailed frame held for the whole clip ------
    frozen = out / "frozen.mp4"
    still = out / "_frozen_frame.png"
    res = _run(["ffmpeg", "-y", "-f", "lavfi",
                "-i", _base_filter("clean_pass"), "-frames:v", "1", str(still)])
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {still.name}:\n{res.stderr[-800:]}")
    res = _run(["ffmpeg", "-y", "-loop", "1", "-i", str(still),
                "-t", f"{DUR}", "-r", f"{FPS}", "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-movflags", "+faststart", str(frozen)])
    still.unlink(missing_ok=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {frozen.name}:\n{res.stderr[-800:]}")
    _sidecar(out, "frozen", {
        "asset_id": "frozen",
        "file": "frozen.mp4",
        "expected_verdict": "REJECT",
        "expected_flags": ["delivery_failure"],
        "expected_scores": {},
        "fix_actions": [],
        "reject_reason": "video frozen for its whole duration, no usable "
                         "motion content (delivery_failure)",
        "context_tags": _ctx(motion_complexity="static_or_minimal"),
        "note": "single frame held for the full clip -> freezedetect fires",
        "near_duplicate_of": None,
    })
    built.append(frozen)

    # b) all-black video ------------------------------------------------------
    black = out / "black.mp4"
    _encode(f"color=c=black:size={SIZE}:rate={FPS}", "", black, DUR)
    _sidecar(out, "black", {
        "asset_id": "black",
        "file": "black.mp4",
        "expected_verdict": "REJECT",
        "expected_flags": ["delivery_failure"],
        "expected_scores": {},
        "fix_actions": [],
        "reject_reason": "video black for its whole duration, no usable "
                         "content (delivery_failure)",
        "context_tags": _ctx(motion_complexity="static_or_minimal"),
        "note": "solid black frames for the full clip -> blackdetect fires",
        "near_duplicate_of": None,
    })
    built.append(black)
    return built


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent)
    paths = build_samples(target)
    print(f"built {len(paths)} clips into {target}")
    for p in paths:
        print(" -", p.name, f"({p.stat().st_size} bytes)")
