"""ViMax working-dir adapter (source-verified layout).

Layout citations live in docs/audits/vimax-layout.md (HKUDS/ViMax commit
0253853). Two shapes are discovered:

- script2video: ``{working_dir}/shots/{idx}/video.mp4``
- idea2video:   ``{working_dir}/scene_{n}/shots/{idx}/video.mp4``

Each shot directory may carry ``shot_description.json`` (a ViMax
``ShotDescription`` dump); its ``visual_desc``/``motion_desc``/``audio_desc``
fields are the shot's prompt text. Shot idx + prompt are carried OUTSIDE the
taxonomy records (asset_id encoding + a ``vimax_manifest.json`` sidecar +
the HTML report) — §7.1/§7.2 schemas are closed and stay untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VimaxShot:
    shot_idx: int
    scene: str | None            # "scene_0" for idea2video nesting, else None
    video_path: str
    prompt: str = ""             # visual_desc (display prompt)
    motion_desc: str = ""
    audio_desc: str = ""
    variation_type: str = ""
    description: dict = field(default_factory=dict)  # raw sidecar (subset)

    @property
    def asset_id(self) -> str:
        base = f"shot_{self.shot_idx:03d}"
        return f"{self.scene}_{base}" if self.scene else base


def _read_description(shot_dir: Path) -> dict:
    p = shot_dir / "shot_description.json"
    if not p.exists():
        return {}
    try:
        doc = json.loads(p.read_text())
        return doc if isinstance(doc, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _shots_under(shots_dir: Path, scene: str | None) -> list[VimaxShot]:
    out: list[VimaxShot] = []
    if not shots_dir.is_dir():
        return out
    for entry in shots_dir.iterdir():
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        video = entry / "video.mp4"
        if not video.is_file():
            continue
        desc = _read_description(entry)
        out.append(VimaxShot(
            shot_idx=int(entry.name),
            scene=scene,
            video_path=str(video),
            prompt=str(desc.get("visual_desc") or ""),
            motion_desc=str(desc.get("motion_desc") or ""),
            audio_desc=str(desc.get("audio_desc") or ""),
            variation_type=str(desc.get("variation_type") or ""),
            description={k: desc[k] for k in
                         ("idx", "cam_idx", "is_last", "ff_desc", "lf_desc")
                         if k in desc},
        ))
    out.sort(key=lambda s: s.shot_idx)
    return out


def discover_shots(working_dir: str | Path) -> list[VimaxShot]:
    """Discover every shot clip in a ViMax working_dir (both layouts).

    Returns shots ordered by (scene, shot_idx). Raises FileNotFoundError for
    a missing working_dir; an existing dir with no shots returns []."""
    root = Path(working_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"vimax working_dir not found: {root}")
    shots = _shots_under(root / "shots", scene=None)
    scene_dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name.startswith("scene_")),
        key=lambda d: d.name,
    )
    for sd in scene_dirs:
        shots.extend(_shots_under(sd / "shots", scene=sd.name))
    return shots


def manifest_entry(shot: VimaxShot, asset_id: str) -> dict:
    """vimax_manifest.json row: the free-form context that must not enter
    the closed §7.1/§7.2 records."""
    return {
        "asset_id": asset_id,
        "shot_idx": shot.shot_idx,
        "scene": shot.scene,
        "video_path": shot.video_path,
        "prompt": shot.prompt,
        "motion_desc": shot.motion_desc,
        "audio_desc": shot.audio_desc,
        "variation_type": shot.variation_type,
        "description": shot.description,
    }
