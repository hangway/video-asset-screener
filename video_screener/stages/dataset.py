"""Stage 4 — dataset.

Join final annotations (stage 3) with the ingest index (stage 1), build
leakage-free train/val/test splits (no near-dup or same-source across splits),
emit a class-balance report per verdict and per flag, and export the records in
the training format consumed by stage 5.

Artifacts (under ``<workdir>/dataset/``):
  - ``splits.json``         asset_id -> split + group structure
  - ``{train,val,test}.jsonl``  training records (annotation + frame paths)
  - ``balance_report.json`` per-split verdict/flag/score-histogram counts
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..config import PipelineConfig
from ..data.splits import Asset, leakage_pairs, make_splits
from ..taxonomy_schema import DIMENSIONS, HARD_FAIL_FLAGS, TAXONOMY_VERSION, VERDICTS
from ..utils.io import read_json, read_jsonl, write_json, write_jsonl


def _load_joined(cfg: PipelineConfig) -> list[dict]:
    ann_path = cfg.stage_dir("annotate") / "annotations.jsonl"
    idx_path = cfg.stage_dir("ingest") / "index.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"{ann_path} missing; run annotate first")
    if not idx_path.exists():
        raise FileNotFoundError(f"{idx_path} missing; run ingest first")
    annotations = read_jsonl(ann_path)
    index = {a["asset_id"]: a for a in read_json(idx_path)["assets"]}
    joined: list[dict] = []
    for rec in annotations:
        aid = rec["asset_id"]
        idx = index.get(aid, {})
        frames = [f["path"] for f in idx.get("sampled_frames", [])]
        # skip decode-failed clips with no frames from the *trainable* set, but
        # keep them recorded (they carry the delivery_failure label with no
        # frames -> not usable as model input).
        joined.append({
            **rec,
            "cluster_id": idx.get("cluster_id", -1),
            "frames": frames,
            "trainable": len(frames) > 0,
            # explicit source id if annotator provided one, else empty
            "source_id": (rec.get("context_tags", {}) or {}).get("source_id", ""),
        })
    return joined


def _balance_report(records: list[dict], assignment: dict[str, str]) -> dict:
    report: dict[str, Any] = {"per_split": {}, "overall": {}}
    splits = ["train", "val", "test"]
    for split in splits + ["overall"]:
        subset = (
            records if split == "overall"
            else [r for r in records if assignment.get(r["asset_id"]) == split]
        )
        verdict_counts = Counter(r["verdict"] for r in subset)
        flag_counts: Counter = Counter()
        for r in subset:
            flag_counts.update(r.get("hard_fail_flags", []))
        dim_hist = {d: Counter() for d in DIMENSIONS}
        for r in subset:
            for d in DIMENSIONS:
                v = (r.get("scores") or {}).get(d)
                if v is not None:
                    dim_hist[d][int(v)] += 1
        entry = {
            "n": len(subset),
            "n_trainable": sum(1 for r in subset if r.get("trainable")),
            "verdicts": {v: verdict_counts.get(v, 0) for v in VERDICTS},
            "flags": {f: flag_counts.get(f, 0) for f in HARD_FAIL_FLAGS},
            "dim_score_hist": {d: {str(k): dim_hist[d][k] for k in sorted(dim_hist[d])}
                               for d in DIMENSIONS},
        }
        if split == "overall":
            report["overall"] = entry
        else:
            report["per_split"][split] = entry
    return report


def run(cfg: PipelineConfig) -> dict[str, Any]:
    records = _load_joined(cfg)
    assets = [
        Asset(asset_id=r["asset_id"], cluster_id=r.get("cluster_id", -1),
              source_id=r.get("source_id", ""), verdict=r["verdict"])
        for r in records
    ]
    dcfg = cfg.dataset
    result = make_splits(
        assets, dcfg.train_frac, dcfg.val_frac, dcfg.test_frac, dcfg.seed
    )
    leaks = leakage_pairs(assets, result.assignment)
    if leaks:  # invariant; never expected
        raise RuntimeError(f"split leakage detected: {leaks}")

    ds_dir = cfg.stage_dir("dataset")
    # write per-split jsonl (training format = annotation record + frames)
    per_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in records:
        per_split[result.assignment[r["asset_id"]]].append(r)
    for split, rows in per_split.items():
        write_jsonl(ds_dir / f"{split}.jsonl", rows)

    splits_doc = {
        "taxonomy_version": TAXONOMY_VERSION,
        "seed": dcfg.seed,
        "fractions": {"train": dcfg.train_frac, "val": dcfg.val_frac,
                      "test": dcfg.test_frac},
        "assignment": result.assignment,
        "groups": result.groups,
        "group_split": result.group_split,
        "n_groups": len(result.groups),
        "leakage_pairs": leaks,
    }
    write_json(ds_dir / "splits.json", splits_doc)

    report = _balance_report(records, result.assignment)
    write_json(ds_dir / "balance_report.json", report)

    return {
        "stage": "dataset",
        "n_assets": len(records),
        "n_groups": len(result.groups),
        "n_trainable": sum(1 for r in records if r.get("trainable")),
        "split_sizes": {s: len(rows) for s, rows in per_split.items()},
        "leakage_pairs": len(leaks),
        "splits_path": str(ds_dir / "splits.json"),
        "balance_report_path": str(ds_dir / "balance_report.json"),
    }
