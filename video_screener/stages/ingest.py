"""Stage 1 — ingest.

Scan video directories, probe each clip, sample frames per taxonomy §5, compute
a perceptual-hash clip signature, cluster near-duplicates, and write an asset
index. Decode failures are recorded (not dropped) so prelabel can raise
``delivery_failure``.

Artifacts (under ``<workdir>/ingest/``):
  - ``index.json``            asset index (metadata + sampled-frame paths + dedup)
  - ``frames/<asset_id>/*``   sampled frames
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import PipelineConfig
from ..taxonomy_schema import TAXONOMY_VERSION, duration_bucket
from ..utils import hashing, video
from ..utils.ffprobe_metrics import probe_intervals
from ..utils.io import write_json


def _asset_id_for(path: Path, taken: set[str]) -> str:
    stem = path.stem
    aid = stem
    i = 1
    while aid in taken:
        aid = f"{stem}_{i}"
        i += 1
    taken.add(aid)
    return aid


def run(cfg: PipelineConfig) -> dict[str, Any]:
    ingest_dir = cfg.stage_dir("ingest")
    frames_root = ingest_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    paths = video.find_videos(cfg.video_dirs)
    assets: list[dict[str, Any]] = []
    taken: set[str] = set()

    for path in paths:
        aid = _asset_id_for(path, taken)
        meta = video.probe(path)
        out_dir = frames_root / aid
        sample = video.extract_frames(
            path, cfg.ingest, out_dir, meta.duration_sec
        )
        frame_paths = [f.path for f in sample.frames]
        signature = hashing.clip_signature(frame_paths)

        # objective interval evidence (freezedetect/blackdetect); best-effort:
        # a failed scan leaves the lists empty, prelabel rules simply see none
        freeze_iv: list[dict] = []
        black_iv: list[dict] = []
        if cfg.ingest.interval_scan and sample.decode_ok:
            scan = probe_intervals(
                path,
                freeze_noise_db=cfg.ingest.freeze_noise_db,
                freeze_min_sec=cfg.ingest.freeze_min_sec,
                black_min_sec=cfg.ingest.black_min_sec,
                black_pic_th=cfg.ingest.black_pic_th,
            )
            if scan.ok:
                freeze_iv = [{k: round(v, 3) for k, v in iv.items()}
                             for iv in scan.freeze_intervals]
                black_iv = [{k: round(v, 3) for k, v in iv.items()}
                            for iv in scan.black_intervals]

        dur = meta.duration_sec
        assets.append({
            "asset_id": aid,
            "file_path": str(path),
            "duration_sec": dur,
            "duration_bucket": duration_bucket(dur) if dur else "",
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "n_frames_total": meta.n_frames_total,
            "probe_ok": meta.probe_ok,
            "decode_ok": sample.decode_ok,
            "error": meta.error or sample.error,
            "n_sampled": len(sample.frames),
            "sampled_frames": [
                {"index": f.index, "time_sec": f.time_sec, "path": f.path,
                 "is_scene_change": f.is_scene_change}
                for f in sample.frames
            ],
            "signature": signature,
            "freeze_intervals": freeze_iv,
            "black_intervals": black_iv,
        })

    # Near-duplicate clustering at clip level.
    sigs = [a["signature"] for a in assets]
    clusters = hashing.cluster_near_duplicates(
        sigs, cfg.ingest.dedup_hamming_threshold
    )
    # First asset (by stable order) in each cluster is the representative.
    seen_cluster: dict[int, str] = {}
    n_dupes = 0
    for a, c in zip(assets, clusters):
        a["cluster_id"] = c
        if c in seen_cluster:
            a["is_duplicate"] = True
            a["duplicate_of"] = seen_cluster[c]
            n_dupes += 1
        else:
            a["is_duplicate"] = False
            a["duplicate_of"] = None
            seen_cluster[c] = a["asset_id"]

    index = {
        "taxonomy_version": TAXONOMY_VERSION,
        "stage": "ingest",
        "video_dirs": cfg.video_dirs,
        "n_assets": len(assets),
        "n_unique": len(seen_cluster),
        "n_duplicates": n_dupes,
        "n_decode_failures": sum(1 for a in assets if not a["decode_ok"]),
        "assets": assets,
    }
    write_json(ingest_dir / "index.json", index)

    return {
        "stage": "ingest",
        "n_assets": len(assets),
        "n_unique": len(seen_cluster),
        "n_duplicates": n_dupes,
        "n_decode_failures": index["n_decode_failures"],
        "index_path": str(ingest_dir / "index.json"),
    }
