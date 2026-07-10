"""Stage 7 — screen (batch inference).

Point at a folder of clips and get taxonomy §7.2 routing: PASS / FIX / REJECT
with primary reasons, suggested fix_actions, a confidence, and needs_human_review
— plus a self-contained shareable HTML report.

Routing is taxonomy-consistent by construction: the objective delivery_failure
rule (decode failure / sub-minimum duration) is applied *before* the model, then
the model's predicted flags + dimension scores are turned into a verdict via the
shared §1/§5/§6 ``derive_verdict``. Every emitted record is validated against
the §7.2 schema before it is written.

Confidence (unified definition): the probability the *deciding source* assigns
to the emitted verdict, on one 0-1 scale across all three routing paths:

- objective delivery rule -> 1.0 (the rule is deterministic, not a model score);
- flag-forced REJECT      -> the strongest triggered flag's sigmoid probability
                             (the flags are what forced the verdict);
- verdict-head routing    -> the head's softmax probability of the *emitted*
                             verdict (also when the §1 PASS gate downgrades the
                             head's argmax — then the reported value is the
                             head's probability for the downgraded verdict and
                             ``needs_human_review`` is set).

Confidence remains **uncalibrated**: it is not fit to a held-out validation set
(the bundled 8-clip toy set is too small) — documented placeholder per §7.2;
calibrate before production use.

Reference consistency (optional, ``consistency.reference_dir``): clip frames
are scored against per-subject reference images in the same embedding space
(worst-frame best-match cosine, §5). A clip below
``consistency.min_reference_similarity`` raises the EXISTING
``reference_inconsistency`` hard-fail flag and routes through the normal
flag-forced REJECT path; its confidence contribution is the similarity
deficit ``clamp(1 - score, 0, 1)`` standing in for the flag sigmoid (the
consistency check, not the flag head, is the deciding source). Per-clip
scores and worst per-frame offenders are written to
``consistency_report.json`` next to the other screen outputs.

The same report carries within-clip drift (max consecutive-frame cosine
distance) as an AUXILIARY signal: a clip above
``consistency.max_frame_drift`` is a morphing candidate and gets
``needs_human_review`` — drift never changes the verdict or raises a flag on
its own (a hard cut is a legitimate reason for a big jump).

Head/tail edge stability: per-frame similarity and drift are analyzed
separately for the head window, tail window (``consistency.edge_window_sec``)
and the clip body. An edge that is a statistical outlier vs the body (beyond
``consistency.edge_outlier_sigma`` body standard deviations) yields a
``trim_head``/``trim_tail`` fix_action with a suggested trim duration and
downgrades a PASS to FIX (§1: trims are minor allowed fixes) — edge
instability never causes a REJECT.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

import torch

from ..aggregate import derive_verdict
from ..config import PipelineConfig
from ..consistency import (
    ReferenceIndex,
    edge_stability,
    frame_drift,
    index_references,
    score_clip,
)
from ..models.encoder import build_encoder
from ..models.model import MultiTaskScreener
from ..schema import InferenceRecord
from ..taxonomy_schema import (
    ABSOLUTE_MIN_DURATION_SEC,
    DIMENSIONS,
    HARD_FAIL_FLAGS,
    TAXONOMY_VERSION,
    VERDICTS,
)
from ..utils import video
from ..utils.io import write_json, write_jsonl
from .train import load_model

_FIX_FOR_DIM = {
    "sharpness_focus": "sharpen",
    "exposure_dynamic_range": "color_grade",
    "composition_framing": "reframe_crop",
    "temporal_stability": "deflicker",
    "motion_quality": "retime_motion",
}


def _fix_signals(scores: dict[str, int], gate: dict[str, int]) -> list[str]:
    sig: list[str] = []
    # borderline (at or just above gate) technical issues suggest a fix
    if scores.get("temporal_stability", 4) <= 2:
        sig.append("deflicker")
    if scores.get("exposure_dynamic_range", 4) <= 2:
        sig.append("color_grade")
    if scores.get("sharpness_focus", 4) <= 2:
        sig.append("sharpen")
    seen: set[str] = set()
    return [s for s in sig if not (s in seen or seen.add(s))]


def _consistency_entry(aid: str, feats, frame_times: list[Optional[float]],
                       ref_index: ReferenceIndex, cfg: PipelineConfig,
                       duration: Optional[float] = None) -> Optional[dict]:
    """Score one clip's frame embeddings against the reference index and
    build its consistency_report.json entry."""
    cons = score_clip(feats, ref_index)
    if cons is None:
        return None
    k = max(1, cfg.consistency.report_worst_k)
    order = sorted(range(len(cons.per_frame)), key=lambda i: cons.per_frame[i])
    worst = [{
        "frame_index": int(i),
        "time_sec": frame_times[i] if i < len(frame_times) else None,
        "similarity": round(cons.per_frame[i], 4),
    } for i in order[:k]]
    # within-clip drift (auxiliary): entry i = distance frames i -> i+1
    drift = frame_drift(feats)
    max_drift = float(drift.max()) if drift.size else 0.0
    max_at = int(drift.argmax()) if drift.size else None
    # head/tail edge stability vs the clip body (None when not analyzable)
    edge = edge_stability(
        cons.per_frame, [float(v) for v in drift], frame_times, duration,
        cfg.consistency.edge_window_sec, cfg.consistency.edge_outlier_sigma,
    )
    return {
        "asset_id": aid,
        "subject": cons.subject,
        "score": round(cons.score, 4),
        "per_subject": {s: round(v, 4) for s, v in cons.per_subject.items()},
        "per_frame": [round(v, 4) for v in cons.per_frame],
        "worst_frames": worst,
        "below_threshold": cons.score < cfg.consistency.min_reference_similarity,
        "drift": [round(float(v), 4) for v in drift],
        "max_drift": round(max_drift, 4),
        "max_drift_between": (
            {"frame_index": max_at,
             "time_sec": frame_times[max_at] if max_at < len(frame_times) else None}
            if max_at is not None else None
        ),
        "drift_exceeds_threshold": max_drift > cfg.consistency.max_frame_drift,
        "edge_stability": edge,
    }


def _screen_one(model, encoder, meta: video.VideoMeta, sample: video.SampleResult,
                cfg: PipelineConfig, aid: str,
                ref_index: Optional[ReferenceIndex] = None,
                ) -> tuple[InferenceRecord, Optional[dict]]:
    gate = cfg.resolved_gate_min()
    dur = meta.duration_sec

    # (1) objective delivery_failure -> REJECT, no model needed
    decode_failed = not sample.decode_ok or len(sample.frames) == 0
    too_short = dur is not None and dur < ABSOLUTE_MIN_DURATION_SEC
    if decode_failed or too_short:
        why = "corrupt/unplayable stream" if decode_failed else \
              f"duration {dur:.2f}s below usable minimum"
        # confidence 1.0: the objective rule decides deterministically
        return InferenceRecord(
            asset_id=aid, verdict="REJECT", confidence=1.0,
            hard_fail_flags=["delivery_failure"], scores={},
            fix_actions=[], primary_reasons=["delivery_failure"],
            needs_human_review=False,
        ), None

    # (2) model inference
    import numpy as np
    feats = encoder.encode_paths([f.path for f in sample.frames])
    frame_times: list[Optional[float]] = (
        [f.time_sec for f in sample.frames]
        if feats.shape[0] == len(sample.frames)   # encoder may skip bad frames
        else [None] * feats.shape[0]
    )
    if feats.shape[0] > cfg.model.max_frames:
        idx = np.linspace(0, feats.shape[0] - 1, cfg.model.max_frames).astype(int)
        feats = feats[idx]
        frame_times = [frame_times[i] for i in idx]
    x = torch.from_numpy(feats).unsqueeze(0)
    mask = torch.ones(1, x.shape[1])
    with torch.no_grad():
        out = model(x, mask)
    head_probs = torch.softmax(out["verdict_logits"], dim=-1)[0]
    head_verdict = VERDICTS[int(head_probs.argmax())]
    levels = MultiTaskScreener.predict_levels(out["dim_thresh_probs"])
    pred_scores = {d: int(levels[d][0]) for d in DIMENSIONS}
    flag_probs = out["flag_probs"][0]
    pred_flags = [HARD_FAIL_FLAGS[i] for i in range(len(HARD_FAIL_FLAGS))
                  if float(flag_probs[i]) > 0.5]

    # (2b) reference consistency: below-threshold clips raise the existing
    # reference_inconsistency flag and flow through the normal flag path.
    cons_entry = None
    cons_strength = None
    if ref_index is not None:
        cons_entry = _consistency_entry(aid, feats, frame_times, ref_index, cfg,
                                        duration=dur)
        if cons_entry is not None and cons_entry["below_threshold"]:
            if "reference_inconsistency" not in pred_flags:
                pred_flags.append("reference_inconsistency")
            # similarity deficit stands in for the flag sigmoid (deciding
            # source is the consistency check, not the flag head)
            cons_strength = max(0.0, min(1.0, 1.0 - cons_entry["score"]))
    # morphing candidate (auxiliary): review only, never verdict-changing
    drift_review = bool(cons_entry and cons_entry["drift_exceeds_threshold"])

    # (3) predicted hard-fail flag -> REJECT (hard-fail semantics, §2)
    fix_sig = _fix_signals(pred_scores, gate)
    if pred_flags:
        # confidence = strongest triggered flag's strength: the flag sigmoid
        # for model-triggered flags, the similarity deficit for a
        # consistency-triggered reference_inconsistency
        strengths = [float(flag_probs[HARD_FAIL_FLAGS.index(f)]) for f in pred_flags
                     if float(flag_probs[HARD_FAIL_FLAGS.index(f)]) > 0.5]
        if cons_strength is not None:
            strengths.append(cons_strength)
        conf = max(strengths)
        return InferenceRecord(
            asset_id=aid, verdict="REJECT", confidence=max(0.0, min(1.0, conf)),
            hard_fail_flags=pred_flags, scores={d: float(pred_scores[d]) for d in DIMENSIONS},
            fix_actions=[], primary_reasons=pred_flags[:3],
            needs_human_review=(
                conf < cfg.screen.review_confidence_threshold or drift_review
            ),
        ), cons_entry

    # (4) head-primary verdict (the head is grounded in dims+flags and is more
    # robust than re-deriving from brittle per-dim level predictions), with the
    # §1 PASS gate enforced: a PASS must clear every dimension gate.
    verdict = head_verdict
    needs_review = False
    zero_dims = [d for d in DIMENSIONS if pred_scores[d] == 0]
    subgate = [d for d in DIMENSIONS if pred_scores[d] < gate.get(d, 2)]
    hard_sub = [d for d in subgate if d in
                {"temporal_stability", "motion_quality", "prompt_fidelity_coherence"}]
    if verdict == "PASS" and subgate:
        # don't pass through a flawed clip (§1); hard-to-fix dim -> REJECT else FIX
        verdict = "REJECT" if (zero_dims and hard_sub) else "FIX"
        needs_review = True

    # (4b) unstable head/tail -> FIX with a trim suggestion (§1: trims are
    # minor allowed fixes). Edge instability only downgrades a PASS; it is
    # never a rejection cause, and a FIX/REJECT from other rules stands.
    trim_suggestions = (
        ((cons_entry or {}).get("edge_stability") or {}).get("trim_suggestions") or []
    )
    if trim_suggestions and verdict == "PASS":
        verdict = "FIX"

    # confidence = head softmax of the *emitted* verdict (which may be the
    # PASS-gate downgrade rather than the head argmax)
    final_idx = VERDICTS.index(verdict)
    conf = float(head_probs[final_idx])
    needs_review = needs_review or drift_review or (
        conf < cfg.screen.review_confidence_threshold and verdict in ("FIX", "REJECT")
    )

    # reasons + fix_actions
    if verdict == "PASS":
        reasons: list[str] = []
        fix_actions: list[str] = []
    else:
        reasons = [f"{d}={pred_scores[d]}" for d in subgate][:3]
        fix_actions = list(fix_sig)
        for d in subgate:
            if d in _FIX_FOR_DIM and _FIX_FOR_DIM[d] not in fix_actions:
                fix_actions.append(_FIX_FOR_DIM[d])
        if verdict == "FIX":
            for t in trim_suggestions:   # trim_head / trim_tail
                if t["action"] not in fix_actions:
                    fix_actions.append(t["action"])
                reasons.append(
                    f"edge_instability_{t['action'].removeprefix('trim_')}"
                )
        if verdict == "FIX" and not fix_actions:
            fix_actions = ["post_production"]
        if verdict == "REJECT":
            fix_actions = []
        if not reasons:
            reasons = [f"verdict_head={verdict}"]

    return InferenceRecord(
        asset_id=aid,
        verdict=verdict,
        confidence=max(0.0, min(1.0, conf)),
        hard_fail_flags=[],
        scores={d: float(pred_scores[d]) for d in DIMENSIONS},
        fix_actions=fix_actions,
        primary_reasons=reasons[:3],
        needs_human_review=bool(needs_review),
    ), cons_entry


def run(cfg: PipelineConfig, out: Optional[str] = None,
        video_dir: Optional[list[str]] = None) -> dict[str, Any]:
    ckpt_path = cfg.stage_dir("train") / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} missing; run the train stage first")
    model, ckpt = load_model(ckpt_path)
    encoder = build_encoder(cfg)

    out_dir = Path(out) if out else cfg.stage_dir("screen")
    frames_root = out_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    # optional reference index for consistency checks (same encoder, cached)
    ref_index: Optional[ReferenceIndex] = None
    if cfg.consistency.reference_dir:
        ref_index = index_references(
            cfg.consistency.reference_dir, encoder, cfg.root / "reference_cache"
        )

    dirs = video_dir or cfg.video_dirs
    paths = video.find_videos(dirs)
    records: list[dict] = []
    taken: set[str] = set()
    report_rows: list[dict] = []
    cons_entries: list[dict] = []

    for path in paths:
        aid = path.stem
        n = 1
        while aid in taken:
            aid = f"{path.stem}_{n}"; n += 1
        taken.add(aid)
        meta = video.probe(path)
        sample = video.extract_frames(path, cfg.ingest, frames_root / aid, meta.duration_sec)
        rec, cons = _screen_one(model, encoder, meta, sample, cfg, aid, ref_index)
        if cons is not None:
            cons_entries.append(cons)
        InferenceRecord.model_validate(rec.model_dump())  # contract check
        records.append(rec.model_dump())
        thumb = _thumb_data_uri(sample.frames[len(sample.frames) // 2].path
                                if sample.frames else None)
        report_rows.append({**rec.model_dump(), "file_path": str(path),
                            "duration_sec": meta.duration_sec, "thumb": thumb})

    write_jsonl(out_dir / "screen_results.jsonl", records)

    routing = {"PASS": [], "FIX": [], "REJECT": []}
    for r in records:
        routing[r["verdict"]].append(r["asset_id"])
    routing_doc = {
        "taxonomy_version": TAXONOMY_VERSION,
        "n": len(records),
        "counts": {v: len(routing[v]) for v in VERDICTS},
        "n_needs_review": sum(1 for r in records if r["needs_human_review"]),
        "queues": routing,
    }
    write_json(out_dir / "routing.json", routing_doc)

    summary = {
        "stage": "screen",
        "n_screened": len(records),
        "counts": routing_doc["counts"],
        "n_needs_review": routing_doc["n_needs_review"],
        "results_path": str(out_dir / "screen_results.jsonl"),
        "report_path": str(out_dir / "screen_report.html"),
    }
    cons_doc = None
    if ref_index is not None:
        cons_doc = _build_consistency_doc(cons_entries, ref_index, cfg)
        write_json(out_dir / "consistency_report.json", cons_doc)
        summary["consistency_report_path"] = str(out_dir / "consistency_report.json")
        summary["n_reference_inconsistent"] = cons_doc["n_below_threshold"]

    html = _build_report(report_rows, routing_doc, cons_doc, ref_index)
    (out_dir / "screen_report.html").write_text(html)

    return summary


def _build_consistency_doc(entries: list[dict], index: ReferenceIndex,
                           cfg: PipelineConfig) -> dict:
    """Assemble consistency_report.json: run metadata + per-clip entries."""
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "reference_dir": cfg.consistency.reference_dir,
        "encoder": index.encoder_name,
        "threshold": cfg.consistency.min_reference_similarity,
        "subjects": {s: len(r.paths) for s, r in index.subjects.items()},
        "n_clips": len(entries),
        "n_below_threshold": sum(1 for e in entries if e["below_threshold"]),
        "n_drift_flagged": sum(1 for e in entries if e.get("drift_exceeds_threshold")),
        "drift_threshold": cfg.consistency.max_frame_drift,
        "clips": entries,
    }


def _thumb_data_uri(frame_path: Optional[str], max_w: int = 240) -> str:
    if not frame_path or not Path(frame_path).exists():
        return ""
    try:
        from PIL import Image
        import io

        im = Image.open(frame_path).convert("RGB")
        w, h = im.size
        im = im.resize((max_w, int(h * max_w / w)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


_BADGE = {"PASS": "#1a7f37", "FIX": "#9a6700", "REJECT": "#cf222e"}


def _consistency_sections(cons_doc: dict,
                          ref_index: Optional[ReferenceIndex]) -> str:
    """Reference gallery + per-subject consistency ranking (report sections)."""
    # gallery: embedded reference thumbnails per subject
    galleries = []
    if ref_index is not None:
        for name, refs in ref_index.subjects.items():
            thumbs = "".join(
                f'<img src="{u}" title="{Path(p).name}">'
                for p in refs.paths if (u := _thumb_data_uri(p, max_w=96))
            )
            galleries.append(
                f'<div class="subj"><span class="sname">{name}</span> '
                f'({len(refs.paths)} refs) {thumbs}</div>'
            )
    # ranking: per subject, clips ordered best -> worst consistency score
    by_subject: dict[str, list[dict]] = {}
    for e in cons_doc.get("clips", []):
        by_subject.setdefault(e["subject"], []).append(e)
    tables = []
    def _markers(e: dict) -> str:
        m = "⚑ inconsistent" if e["below_threshold"] else ""
        if e.get("drift_exceeds_threshold"):
            m += " ⚠ drift"
        for t in (e.get("edge_stability") or {}).get("trim_suggestions") or []:
            m += f' ✂ {t["action"]} {t["suggested_trim_sec"]}s'
        return m

    for name in sorted(by_subject):
        ranked = sorted(by_subject[name], key=lambda e: e["score"], reverse=True)
        rows_html = "".join(
            f'<tr class="{"below" if e["below_threshold"] else ""}">'
            f'<td>{i + 1}</td><td>{e["asset_id"]}</td>'
            f'<td>{e["score"]:.3f}</td><td>{e.get("max_drift", 0):.3f}</td>'
            f'<td>{_markers(e)}</td></tr>'
            for i, e in enumerate(ranked)
        )
        tables.append(
            f'<h3>{name}</h3><table class="rank">'
            f'<tr><th>#</th><th>asset</th><th>consistency</th>'
            f'<th>max drift</th><th></th></tr>{rows_html}</table>'
        )
    return f"""<section class="consistency">
<h2>Reference consistency</h2>
<div class="sum">threshold {cons_doc['threshold']} ·
{cons_doc['n_below_threshold']} below threshold ·
{cons_doc['n_drift_flagged']} drift-flagged</div>
<div class="gallery">{''.join(galleries)}</div>
{''.join(tables)}</section>"""


def _build_report(rows: list[dict], routing: dict,
                  cons_doc: Optional[dict] = None,
                  ref_index: Optional[ReferenceIndex] = None) -> str:
    counts = routing["counts"]
    cons_html = _consistency_sections(cons_doc, ref_index) if cons_doc else ""
    cards = []
    for r in rows:
        color = _BADGE.get(r["verdict"], "#555")
        flags = ", ".join(r["hard_fail_flags"]) or "—"
        fixes = ", ".join(r["fix_actions"]) or "—"
        reasons = ", ".join(r["primary_reasons"]) or "—"
        scores = " ".join(f"{d.split('_')[0]}:{int(v)}" for d, v in (r.get("scores") or {}).items())
        review = " ⚠ needs review" if r["needs_human_review"] else ""
        img = f'<img src="{r["thumb"]}">' if r.get("thumb") else '<div class="noimg">no frame</div>'
        cards.append(f"""<div class="card">
  {img}
  <div class="meta">
    <div class="hd"><span class="badge" style="background:{color}">{r['verdict']}</span>
      <span class="aid">{r['asset_id']}</span>
      <span class="conf">conf {r['confidence']:.2f}{review}</span></div>
    <div class="row"><b>flags</b>: {flags}</div>
    <div class="row"><b>fix</b>: {fixes}</div>
    <div class="row"><b>reasons</b>: {reasons}</div>
    <div class="row scores">{scores}</div>
  </div></div>""")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Video Asset Screening Report</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0d1117;color:#e6edf3}}
header{{padding:20px 28px;border-bottom:1px solid #30363d}}
h1{{margin:0 0 6px;font-size:20px}} .sum{{color:#8b949e;font-size:14px}}
.counts span{{display:inline-block;margin-right:14px;font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;padding:20px 28px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden;display:flex;flex-direction:column}}
.card img{{width:100%;display:block;background:#000}} .noimg{{padding:40px;text-align:center;color:#8b949e;background:#000}}
.meta{{padding:12px 14px}} .hd{{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}}
.badge{{color:#fff;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:700}}
.aid{{font-weight:600}} .conf{{color:#8b949e;font-size:12px;margin-left:auto}}
.row{{font-size:13px;color:#c9d1d9;margin:3px 0}} .row b{{color:#8b949e}}
.scores{{color:#8b949e;font-family:ui-monospace,monospace;font-size:12px;margin-top:6px}}
.consistency{{padding:20px 28px;border-bottom:1px solid #30363d}}
.consistency h2{{margin:0 0 6px;font-size:17px}} .consistency h3{{margin:14px 0 6px;font-size:14px;color:#c9d1d9}}
.gallery .subj{{margin:8px 0;font-size:13px;color:#c9d1d9}}
.gallery .sname{{font-weight:600}}
.gallery img{{height:48px;border-radius:4px;margin:0 3px;vertical-align:middle;border:1px solid #30363d}}
table.rank{{border-collapse:collapse;font-size:13px}}
table.rank th,table.rank td{{padding:4px 12px;text-align:left;border-bottom:1px solid #21262d}}
table.rank th{{color:#8b949e;font-weight:600}}
table.rank tr.below td{{color:#f85149}}
</style></head><body>
<header><h1>Video Asset Usability Screening</h1>
<div class="sum">taxonomy v{TAXONOMY_VERSION} · {routing['n']} clips screened ·
{routing['n_needs_review']} need human review</div>
<div class="counts" style="margin-top:8px">
<span style="color:{_BADGE['PASS']}">PASS {counts['PASS']}</span>
<span style="color:{_BADGE['FIX']}">FIX {counts['FIX']}</span>
<span style="color:{_BADGE['REJECT']}">REJECT {counts['REJECT']}</span></div>
</header>{cons_html}<div class="grid">{''.join(cards)}</div></body></html>"""
