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
from html import escape
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..aggregate import derive_verdict
from ..config import PipelineConfig
from ..consistency import (
    ReferenceIndex,
    edge_stability,
    frame_drift,
    index_reference_paths,
    index_references,
    merge_reference_indexes,
    score_clip,
    score_keyframes,
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
from ..utils.metrics_backend import build_metrics_backend, segment_metric_summary
from ..vimax import (
    ViMaxProject,
    load_vimax_project,
    score_vimax_boundaries,
)
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


def _consistency_entry(
    aid: str,
    feats: np.ndarray,
    frame_times: list[Optional[float]],
    ref_index: Optional[ReferenceIndex],
    cfg: PipelineConfig,
    duration: Optional[float] = None,
    tech_series: Optional[list[dict]] = None,
    expected_subjects: tuple[str, ...] = (),
    keyframe_embeddings: Optional[dict[str, np.ndarray]] = None,
    vimax_metadata: Optional[dict[str, Any]] = None,
) -> Optional[dict]:
    """Build one reference, keyframe, and temporal consistency report entry."""
    # A ViMax environment-only shot intentionally has no assigned characters;
    # do not compare it against an arbitrary portrait from the global index.
    should_score_subjects = (
        ref_index is not None
        and (vimax_metadata is None or bool(expected_subjects))
    )
    cons = (
        score_clip(feats, ref_index, expected_subjects or None)
        if should_score_subjects else None
    )
    keyframes = score_keyframes(
        feats,
        (keyframe_embeddings or {}).get("first"),
        (keyframe_embeddings or {}).get("last"),
    )
    if (cons is None and keyframes is None and not expected_subjects
            and vimax_metadata is None):
        return None

    expected = list(expected_subjects)
    available_subjects = (
        {name for name, refs in ref_index.subjects.items() if refs.embeddings.size}
        if ref_index is not None else set()
    )
    missing_expected = [name for name in expected if name not in available_subjects]
    missing_keyframes: list[str] = []
    if vimax_metadata is not None:
        supplied = keyframe_embeddings or {}
        if not vimax_metadata.get("first_frame_path"):
            missing_keyframes.append("first")
        elif "first" not in supplied:
            missing_keyframes.append("first_unreadable")
        if vimax_metadata.get("last_frame_path") and "last" not in supplied:
            missing_keyframes.append("last_unreadable")

    best_subject = cons.best_subject if cons is not None else None
    mismatch_delta = 0.0
    subject_mismatch = False
    if cons is not None and expected and best_subject not in expected:
        expected_scores = [
            cons.per_subject[name] for name in expected if name in cons.per_subject
        ]
        if expected_scores:
            mismatch_delta = cons.per_subject[best_subject] - max(expected_scores)
            subject_mismatch = (
                mismatch_delta > cfg.consistency.subject_mismatch_margin
            )

    reference_score = cons.score if cons is not None else None
    keyframe_score = keyframes.score if keyframes is not None else None
    available_scores = [
        score for score in (reference_score, keyframe_score) if score is not None
    ]
    overall_score = min(available_scores) if available_scores else None
    reference_below = bool(
        reference_score is not None
        and reference_score < cfg.consistency.min_reference_similarity
    )
    keyframe_below = bool(
        keyframe_score is not None
        and keyframe_score < cfg.consistency.min_keyframe_similarity
    )

    per_frame = cons.per_frame if cons is not None else []
    k = max(1, cfg.consistency.report_worst_k)
    order = sorted(range(len(per_frame)), key=lambda i: per_frame[i])
    worst = [{
        "frame_index": int(i),
        "time_sec": frame_times[i] if i < len(frame_times) else None,
        "similarity": round(per_frame[i], 4),
    } for i in order[:k]]

    drift = frame_drift(feats)
    max_drift = float(drift.max()) if drift.size else 0.0
    max_at = int(drift.argmax()) if drift.size else None
    edge = None
    if cons is not None:
        edge = edge_stability(
            per_frame, [float(v) for v in drift], frame_times, duration,
            cfg.consistency.edge_window_sec, cfg.consistency.edge_outlier_sigma,
        )
    if edge is not None and tech_series:
        edge["technical"] = segment_metric_summary(
            tech_series, duration, cfg.consistency.edge_window_sec
        )

    subject_label = (
        cons.subject if cons is not None
        else " + ".join(expected_subjects) or "vimax_keyframes"
    )
    return {
        "asset_id": aid,
        "subject": subject_label,
        "expected_subjects": expected,
        "missing_expected_subjects": missing_expected,
        "missing_keyframes": missing_keyframes,
        "best_subject": best_subject,
        "subject_mismatch": subject_mismatch,
        "subject_mismatch_delta": round(mismatch_delta, 4),
        "reference_score": (
            round(reference_score, 4) if reference_score is not None else None
        ),
        "keyframe_score": (
            round(keyframe_score, 4) if keyframe_score is not None else None
        ),
        "score": round(overall_score, 4) if overall_score is not None else None,
        "per_subject": (
            {name: round(value, 4) for name, value in cons.per_subject.items()}
            if cons is not None else {}
        ),
        "per_frame": [round(value, 4) for value in per_frame],
        "worst_frames": worst,
        "reference_below_threshold": reference_below,
        "keyframe_below_threshold": keyframe_below,
        "below_threshold": bool(
            reference_below or keyframe_below or subject_mismatch
        ),
        "keyframes": ({
            "head_similarity": (
                round(keyframes.head_similarity, 4)
                if keyframes and keyframes.head_similarity is not None else None
            ),
            "tail_similarity": (
                round(keyframes.tail_similarity, 4)
                if keyframes and keyframes.tail_similarity is not None else None
            ),
            "threshold": cfg.consistency.min_keyframe_similarity,
        } if keyframes is not None else None),
        "drift": [round(float(value), 4) for value in drift],
        "max_drift": round(max_drift, 4),
        "max_drift_between": (
            {"frame_index": max_at,
             "time_sec": frame_times[max_at] if max_at < len(frame_times) else None}
            if max_at is not None else None
        ),
        "drift_exceeds_threshold": max_drift > cfg.consistency.max_frame_drift,
        "edge_stability": edge,
        "vimax": vimax_metadata,
    }


def _screen_one(model, encoder, meta: video.VideoMeta, sample: video.SampleResult,
                cfg: PipelineConfig, aid: str,
                ref_index: Optional[ReferenceIndex] = None,
                metrics_backend=None,
                expected_subjects: tuple[str, ...] = (),
                keyframe_embeddings: Optional[dict[str, np.ndarray]] = None,
                vimax_metadata: Optional[dict[str, Any]] = None,
                endpoint_sink: Optional[
                    dict[str, dict[str, np.ndarray]]
                ] = None,
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
    if endpoint_sink is not None and feats.shape[0]:
        endpoint_sink[aid] = {
            "head": feats[0].copy(),
            "tail": feats[-1].copy(),
        }
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
    if (ref_index is not None or expected_subjects or keyframe_embeddings
            or vimax_metadata is not None):
        tech_series = (
            metrics_backend.frame_series(meta.path, [f.path for f in sample.frames])
            if metrics_backend is not None else None
        )
        cons_entry = _consistency_entry(
            aid, feats, frame_times, ref_index, cfg,
            duration=dur,
            tech_series=tech_series,
            expected_subjects=expected_subjects,
            keyframe_embeddings=keyframe_embeddings,
            vimax_metadata=vimax_metadata,
        )
        if cons_entry is not None and cons_entry["below_threshold"]:
            if "reference_inconsistency" not in pred_flags:
                pred_flags.append("reference_inconsistency")
            # similarity deficit stands in for the flag sigmoid (deciding
            # source is the consistency check, not the flag head)
            if cons_entry["score"] is not None:
                cons_strength = max(
                    0.0, min(1.0, 1.0 - cons_entry["score"])
                )
    # morphing candidate (auxiliary): review only, never verdict-changing
    drift_review = bool(cons_entry and cons_entry["drift_exceeds_threshold"])
    target_review = bool(
        cons_entry and (
            cons_entry.get("missing_expected_subjects")
            or cons_entry.get("missing_keyframes")
        )
    )

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
                conf < cfg.screen.review_confidence_threshold
                or drift_review or target_review
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
    needs_review = needs_review or drift_review or target_review or (
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


def _index_vimax_keyframes(
    project: ViMaxProject,
    encoder,
    cache_dir: Path,
) -> dict[str, dict[str, np.ndarray]]:
    paths: dict[str, list[Path]] = {}
    targets: dict[str, tuple[str, str]] = {}
    for shot in project.shots:
        for role, path in (
            ("first", shot.first_frame_path),
            ("last", shot.last_frame_path),
        ):
            if path is None:
                continue
            key = f"{shot.asset_id}__{role}"
            paths[key] = [path]
            targets[key] = (shot.asset_id, role)
    if not paths:
        return {}

    index = index_reference_paths(paths, encoder, cache_dir)
    encoded: dict[str, dict[str, np.ndarray]] = {}
    for key, refs in index.subjects.items():
        if refs.embeddings.shape[0]:
            asset_id, role = targets[key]
            encoded.setdefault(asset_id, {})[role] = refs.embeddings[0]
    return encoded


def _build_reference_index(
    cfg: PipelineConfig,
    encoder,
    vimax_project: Optional[ViMaxProject],
) -> Optional[ReferenceIndex]:
    indexes: list[ReferenceIndex] = []
    cache_dir = cfg.root / "reference_cache"
    if cfg.consistency.reference_dir:
        indexes.append(index_references(
            cfg.consistency.reference_dir, encoder, cache_dir
        ))
    if vimax_project is not None and vimax_project.reference_paths:
        indexes.append(index_reference_paths(
            vimax_project.reference_paths, encoder, cache_dir
        ))
    if indexes:
        return merge_reference_indexes(*indexes)
    if vimax_project is not None:
        return ReferenceIndex(encoder_name=encoder.name)
    return None


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

    vimax_project = (
        load_vimax_project(cfg.consistency.vimax_workdir)
        if cfg.consistency.vimax_workdir else None
    )
    ref_index = _build_reference_index(cfg, encoder, vimax_project)
    vimax_keyframes = (
        _index_vimax_keyframes(
            vimax_project, encoder, cfg.root / "reference_cache"
        )
        if vimax_project is not None else {}
    )
    mbackend = build_metrics_backend(cfg)

    if vimax_project is not None:
        shots = vimax_project.rendered_shots
        if not shots:
            raise ValueError(
                f"no rendered ViMax shots/*/video.mp4 found under {vimax_project.root}"
            )
        paths = [shot.video_path for shot in shots]
    else:
        dirs = video_dir or cfg.video_dirs
        paths = video.find_videos(dirs)

    records: list[dict] = []
    taken: set[str] = set()
    report_rows: list[dict] = []
    cons_entries: list[dict] = []
    endpoint_embeddings: dict[str, dict[str, np.ndarray]] = {}

    for path in paths:
        vimax_shot = (
            vimax_project.shot_for_video(path) if vimax_project is not None
            else None
        )
        aid = vimax_shot.asset_id if vimax_shot else path.stem
        n = 1
        base_aid = aid
        while aid in taken:
            aid = f"{base_aid}_{n}"
            n += 1
        taken.add(aid)
        meta = video.probe(path)
        sample = video.extract_frames(
            path, cfg.ingest, frames_root / aid, meta.duration_sec,
            include_tail=vimax_shot is not None,
        )
        keyframes = vimax_keyframes.get(aid) if vimax_shot else None
        rec, cons = _screen_one(
            model, encoder, meta, sample, cfg, aid, ref_index,
            metrics_backend=mbackend,
            expected_subjects=(vimax_shot.expected_subjects if vimax_shot else ()),
            keyframe_embeddings=keyframes,
            vimax_metadata=(vimax_shot.report_metadata() if vimax_shot else None),
            endpoint_sink=endpoint_embeddings,
        )
        if cons is not None:
            cons_entries.append(cons)
        records.append(rec.model_dump())
        thumb = _thumb_data_uri(sample.frames[len(sample.frames) // 2].path
                                if sample.frames else None)
        report_rows.append({**rec.model_dump(), "file_path": str(path),
                            "duration_sec": meta.duration_sec, "thumb": thumb})

    boundaries: list[dict[str, Any]] = []
    if vimax_project is not None:
        boundaries = score_vimax_boundaries(
            vimax_project.rendered_shots,
            endpoint_embeddings,
            cfg.consistency.min_same_camera_boundary_similarity,
        )
        flagged_assets = {
            asset_id
            for boundary in boundaries if boundary["below_threshold"]
            for asset_id in (
                boundary["from_asset_id"], boundary["to_asset_id"]
            )
        }
        for record in records:
            if record["asset_id"] in flagged_assets:
                record["needs_human_review"] = True
                reason = "same_camera_boundary_drift"
                if reason not in record["primary_reasons"]:
                    record["primary_reasons"] = (
                        record["primary_reasons"] + [reason]
                    )[:3]

    record_by_asset = {record["asset_id"]: record for record in records}
    for row in report_rows:
        row.update(record_by_asset[row["asset_id"]])
    for record in records:
        InferenceRecord.model_validate(record)
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
    if ref_index is not None or vimax_project is not None:
        report_index = ref_index or ReferenceIndex(encoder_name=encoder.name)
        cons_doc = _build_consistency_doc(
            cons_entries, report_index, cfg,
            vimax_project=vimax_project,
            boundaries=boundaries,
        )
        write_json(out_dir / "consistency_report.json", cons_doc)
        summary["consistency_report_path"] = str(out_dir / "consistency_report.json")
        summary["n_reference_inconsistent"] = cons_doc["n_below_threshold"]
    if vimax_project is not None:
        summary["vimax_workdir"] = str(vimax_project.root)
        summary["n_vimax_missing_videos"] = len(vimax_project.missing_video_shots)
        summary["n_boundary_flagged"] = sum(
            1 for boundary in boundaries if boundary["below_threshold"]
        )

    html = _build_report(report_rows, routing_doc, cons_doc, ref_index)
    (out_dir / "screen_report.html").write_text(html, encoding="utf-8")

    return summary


def _build_consistency_doc(
    entries: list[dict],
    index: ReferenceIndex,
    cfg: PipelineConfig,
    vimax_project: Optional[ViMaxProject] = None,
    boundaries: Optional[list[dict[str, Any]]] = None,
) -> dict:
    """Assemble consistency_report.json: run metadata + per-clip entries."""
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "reference_dir": cfg.consistency.reference_dir,
        "vimax_workdir": (
            str(vimax_project.root) if vimax_project is not None else None
        ),
        "encoder": index.encoder_name,
        "threshold": cfg.consistency.min_reference_similarity,
        "keyframe_threshold": cfg.consistency.min_keyframe_similarity,
        "subjects": {s: len(r.paths) for s, r in index.subjects.items()},
        "n_clips": len(entries),
        "n_below_threshold": sum(1 for e in entries if e["below_threshold"]),
        "n_drift_flagged": sum(1 for e in entries if e.get("drift_exceeds_threshold")),
        "drift_threshold": cfg.consistency.max_frame_drift,
        "boundary_threshold": cfg.consistency.min_same_camera_boundary_similarity,
        "n_boundary_flagged": sum(
            1 for boundary in (boundaries or []) if boundary["below_threshold"]
        ),
        "boundaries": boundaries or [],
        "vimax_warnings": vimax_project.warnings if vimax_project else [],
        "vimax_missing_video_assets": (
            [shot.asset_id for shot in vimax_project.missing_video_shots]
            if vimax_project else []
        ),
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


def _score_text(value: Any) -> str:
    return f"{float(value):.3f}" if value is not None else "-"


def _vimax_timeline(cons_doc: dict) -> str:
    clips = [entry for entry in cons_doc.get("clips", []) if entry.get("vimax")]
    if not clips:
        return ""

    by_sequence: dict[str, list[dict]] = {}
    for entry in clips:
        by_sequence.setdefault(entry["vimax"]["sequence_id"], []).append(entry)

    shot_tables = []
    for sequence_id, entries in sorted(by_sequence.items()):
        ordered = sorted(entries, key=lambda entry: entry["vimax"]["shot_idx"])
        rows = []
        for entry in ordered:
            meta = entry["vimax"]
            expected = ", ".join(meta.get("expected_subjects") or []) or "-"
            markers = []
            if entry.get("subject_mismatch"):
                markers.append("assigned subject mismatch")
            if entry.get("missing_expected_subjects"):
                markers.append("missing portrait reference")
            if entry.get("missing_keyframes"):
                markers.append("missing keyframe reference")
            if entry.get("keyframe_below_threshold"):
                markers.append("keyframe drift")
            if entry.get("drift_exceeds_threshold"):
                markers.append("within-shot drift")
            rows.append(
                f'<tr class="{"below" if entry["below_threshold"] else ""}">'
                f'<td>{meta["shot_idx"]}</td><td>{meta.get("camera_idx", "-")}</td>'
                f'<td>{escape(expected)}</td><td>{_score_text(entry.get("reference_score"))}</td>'
                f'<td>{_score_text(entry.get("keyframe_score"))}</td>'
                f'<td>{_score_text(entry.get("max_drift"))}</td>'
                f'<td>{escape(", ".join(markers) or "-")}</td></tr>'
            )
        shot_tables.append(
            f'<h3>ViMax sequence: {escape(sequence_id)}</h3><table class="rank">'
            '<tr><th>shot</th><th>camera</th><th>expected characters</th>'
            '<th>portrait</th><th>keyframe</th><th>drift</th><th>review</th></tr>'
            f'{"".join(rows)}</table>'
        )

    boundary_rows = []
    for boundary in cons_doc.get("boundaries", []):
        cameras = (
            f'{boundary["from_camera_idx"]} (same)'
            if boundary["same_camera"]
            else f'{boundary["from_camera_idx"]} -> {boundary["to_camera_idx"]}'
        )
        boundary_rows.append(
            f'<tr class="{"below" if boundary["below_threshold"] else ""}">'
            f'<td>{escape(boundary["sequence_id"])}</td>'
            f'<td>{boundary["from_shot_idx"]} -> {boundary["to_shot_idx"]}</td>'
            f'<td>{cameras}</td><td>{boundary["similarity"]:.3f}</td>'
            f'<td>{"needs review" if boundary["below_threshold"] else "-"}</td></tr>'
        )
    boundaries = ""
    if boundary_rows:
        boundaries = (
            '<h3>Shot boundaries</h3><table class="rank">'
            '<tr><th>sequence</th><th>shots</th><th>camera</th>'
            '<th>similarity</th><th>review</th></tr>'
            f'{"".join(boundary_rows)}</table>'
        )
    return f"""<div class="vimax">
<h2>ViMax continuity</h2>
<div class="sum">keyframe threshold {cons_doc['keyframe_threshold']} ·
same-camera boundary threshold {cons_doc['boundary_threshold']} ·
{cons_doc['n_boundary_flagged']} boundaries flagged</div>
{''.join(shot_tables)}{boundaries}</div>"""


def _consistency_sections(cons_doc: dict,
                          ref_index: Optional[ReferenceIndex]) -> str:
    """Reference gallery + per-subject consistency ranking (report sections)."""
    # gallery: embedded reference thumbnails per subject
    galleries = []
    if ref_index is not None:
        for name, refs in ref_index.subjects.items():
            thumbs = "".join(
                f'<img src="{u}" title="{escape(Path(p).name)}">'
                for p in refs.paths if (u := _thumb_data_uri(p, max_w=96))
            )
            galleries.append(
                f'<div class="subj"><span class="sname">{escape(name)}</span> '
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
        if e.get("subject_mismatch"):
            m += " ⚠ subject mismatch"
        if e.get("keyframe_below_threshold"):
            m += " ⚠ keyframe"
        if e.get("missing_expected_subjects"):
            m += " ⚠ missing refs"
        if e.get("missing_keyframes"):
            m += " ⚠ missing keyframe"
        for t in (e.get("edge_stability") or {}).get("trim_suggestions") or []:
            m += f' ✂ {t["action"]} {t["suggested_trim_sec"]}s'
        return m

    for name in sorted(by_subject):
        ranked = sorted(
            by_subject[name],
            key=lambda e: e["score"] if e["score"] is not None else -2.0,
            reverse=True,
        )
        rows_html = "".join(
            f'<tr class="{"below" if e["below_threshold"] else ""}">'
            f'<td>{i + 1}</td><td>{escape(e["asset_id"])}</td>'
            f'<td>{_score_text(e["score"])}</td>'
            f'<td>{_score_text(e.get("max_drift", 0))}</td>'
            f'<td>{_markers(e)}</td></tr>'
            for i, e in enumerate(ranked)
        )
        tables.append(
            f'<h3>{escape(name)}</h3><table class="rank">'
            f'<tr><th>#</th><th>asset</th><th>consistency</th>'
            f'<th>max drift</th><th></th></tr>{rows_html}</table>'
        )
    return f"""<section class="consistency">
<h2>Reference consistency</h2>
<div class="sum">threshold {cons_doc['threshold']} ·
{cons_doc['n_below_threshold']} below threshold ·
{cons_doc['n_drift_flagged']} drift-flagged</div>
<div class="gallery">{''.join(galleries)}</div>
{_vimax_timeline(cons_doc)}
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
