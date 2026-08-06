"""Stage 6 — evaluate.

Run the trained model over a held-out split (default: test), then report:
verdict confusion matrix + accuracy, per-dimension ordinal metrics, per-flag
precision/recall, metrics stratified by aesthetic_family + motion_complexity, a
worst-failure gallery, and a verdict-vs-flags/dims consistency check (§1) with
its rate.

Artifacts: ``<workdir>/evaluate/eval_report.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..config import PipelineConfig
from ..eval_metrics import (
    consistency_rate,
    dimension_metrics,
    flag_metrics,
    stratified_verdict_accuracy,
    verdict_confusion,
)
from ..models.encoder import build_encoder, validate_checkpoint_encoder
from ..models.model import MultiTaskScreener
from ..taxonomy_schema import DIMENSIONS, HARD_FAIL_FLAGS, TAXONOMY_VERSION, VERDICTS
from ..utils.io import read_jsonl, write_json
from .train import load_model


def _predict(model: MultiTaskScreener, encoder, rec: dict, max_frames: int) -> dict:
    feats = encoder.encode_paths(rec["frames"])
    if feats.shape[0] == 0:
        return {}
    if feats.shape[0] > max_frames:
        import numpy as np
        idx = np.linspace(0, feats.shape[0] - 1, max_frames).astype(int)
        feats = feats[idx]
    x = torch.from_numpy(feats).unsqueeze(0)
    mask = torch.ones(1, x.shape[1])
    with torch.no_grad():
        out = model(x, mask)
    verdict = VERDICTS[int(out["verdict_logits"].argmax(dim=-1))]
    probs = torch.softmax(out["verdict_logits"], dim=-1)[0]
    levels = MultiTaskScreener.predict_levels(out["dim_thresh_probs"])
    pred_scores = {d: int(levels[d][0]) for d in DIMENSIONS}
    flag_probs = out["flag_probs"][0]
    pred_flags = [HARD_FAIL_FLAGS[i] for i in range(len(HARD_FAIL_FLAGS))
                  if float(flag_probs[i]) > 0.5]
    return {
        "verdict": verdict,
        "verdict_conf": float(probs.max()),
        "scores": pred_scores,
        "flags": pred_flags,
        "flag_probs": {HARD_FAIL_FLAGS[i]: float(flag_probs[i])
                       for i in range(len(HARD_FAIL_FLAGS))},
    }


def _error_score(true_v: str, pred_v: str, true_s: dict, pred_s: dict,
                 true_f: list[str], pred_f: list[str]) -> float:
    score = 2.0 if true_v != pred_v else 0.0
    for d in DIMENSIONS:
        t, p = true_s.get(d), pred_s.get(d)
        if t is not None and p is not None:
            score += abs(int(t) - int(p)) * 0.25
    score += len(set(true_f) ^ set(pred_f)) * 1.0
    return score


def run(cfg: PipelineConfig, split: str = "test") -> dict[str, Any]:
    ckpt_path = cfg.stage_dir("train") / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} missing; run the train stage first")
    model, ckpt = load_model(ckpt_path)
    encoder = build_encoder(cfg)
    validate_checkpoint_encoder(ckpt, encoder)

    recs = read_jsonl(cfg.stage_dir("dataset") / f"{split}.jsonl")
    evaluable = [r for r in recs if r.get("frames")]

    y_true: list[str] = []
    y_pred: list[str] = []
    true_by_dim: dict[str, list] = {d: [] for d in DIMENSIONS}
    pred_by_dim: dict[str, list] = {d: [] for d in DIMENSIONS}
    true_flags: list[list[str]] = []
    pred_flags: list[list[str]] = []
    pred_scores_list: list[dict] = []
    strata_aes: list[str] = []
    strata_motion: list[str] = []
    per_clip: list[dict] = []

    for rec in evaluable:
        pred = _predict(model, encoder, rec, cfg.model.max_frames)
        if not pred:
            continue
        tv = rec["verdict"]
        ts = rec.get("scores") or {}
        tf = rec.get("hard_fail_flags", [])
        y_true.append(tv)
        y_pred.append(pred["verdict"])
        for d in DIMENSIONS:
            true_by_dim[d].append(ts.get(d))
            pred_by_dim[d].append(pred["scores"].get(d))
        true_flags.append(tf)
        pred_flags.append(pred["flags"])
        pred_scores_list.append({d: float(pred["scores"][d]) for d in DIMENSIONS})
        ctx = rec.get("context_tags", {}) or {}
        strata_aes.append((ctx.get("aesthetic_family") or ["(none)"])[0]
                          if ctx.get("aesthetic_family") else "(none)")
        strata_motion.append(ctx.get("motion_complexity") or "(none)")
        per_clip.append({
            "asset_id": rec["asset_id"],
            "true_verdict": tv, "pred_verdict": pred["verdict"],
            "true_flags": tf, "pred_flags": pred["flags"],
            "true_scores": {d: ts.get(d) for d in DIMENSIONS},
            "pred_scores": pred["scores"],
            "verdict_conf": pred["verdict_conf"],
            "error_score": _error_score(tv, pred["verdict"], ts, pred["scores"],
                                        tf, pred["flags"]),
        })

    worst = sorted(per_clip, key=lambda c: c["error_score"], reverse=True)
    worst_gallery = [c for c in worst if c["error_score"] > 0][: cfg.evaluate.worst_gallery_size]

    report = {
        "taxonomy_version": TAXONOMY_VERSION,
        "split": split,
        "n_evaluated": len(y_true),
        "n_skipped_no_frames": len(recs) - len(evaluable),
        "encoder": ckpt.get("encoder"),
        "verdict": verdict_confusion(y_true, y_pred),
        "dimensions": dimension_metrics(true_by_dim, pred_by_dim),
        "flags": flag_metrics(true_flags, pred_flags),
        "stratified": {
            "aesthetic_family": stratified_verdict_accuracy(strata_aes, y_true, y_pred),
            "motion_complexity": stratified_verdict_accuracy(strata_motion, y_true, y_pred),
        },
        "consistency": consistency_rate(y_pred, pred_scores_list, pred_flags),
        "worst_failures": worst_gallery,
        "per_clip": per_clip,
    }
    out = cfg.stage_dir("evaluate") / "eval_report.json"
    write_json(out, report)
    return {
        "stage": "evaluate",
        "split": split,
        "n_evaluated": len(y_true),
        "verdict_accuracy": round(report["verdict"]["accuracy"], 3),
        "verdict_macro_f1": round(report["verdict"]["macro_f1"], 3),
        "flag_micro_f1": report["flags"]["micro"]["f1"],
        "consistency_rate_inconsistent": round(report["consistency"]["rate"], 3),
        "n_worst_failures": len(worst_gallery),
        "report_path": str(out),
    }
