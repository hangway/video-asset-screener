"""Stage 2 — prelabel.

Auto pre-annotation from the ingest index. Computes technical dimensions
(sharpness, exposure) and a temporal-stability heuristic (brightness-flicker)
from frame metrics, sets the objective ``delivery_failure`` flag (decode failure
or sub-minimum duration), and emits taxonomy §7.1 annotation records with a
heuristic verdict as a *starting point* for human review.

Scope (per goal-prompt): sharpness/exposure/temporal-stability + optional
MLLM prompt-fidelity. Subjective flags (watermark, AI artifact, legal, etc.)
and composition/motion nuance are deferred to the human annotate stage — every
prelabel record has ``needs_human_review = True``.

Artifacts: ``<workdir>/prelabel/prelabels.jsonl`` (one §7.1 record per asset).
"""

from __future__ import annotations

from typing import Any, Optional

from ..aggregate import derive_verdict
from ..config import PipelineConfig
from ..schema import AnnotationRecord
from ..taxonomy_schema import (
    ABSOLUTE_MIN_DURATION_SEC,
    DIMENSIONS,
    TAXONOMY_VERSION,
    duration_bucket,
)
from ..utils.io import read_json, write_jsonl
from ..utils.metrics_backend import build_metrics_backend


def _prelabel_asset(asset: dict, cfg: PipelineConfig,
                    backend=None) -> AnnotationRecord:
    pcfg = cfg.prelabel
    gate = cfg.resolved_gate_min()
    aid = asset["asset_id"]
    dur = asset.get("duration_sec")
    frame_paths = [f["path"] for f in asset.get("sampled_frames", [])]

    flags: list[str] = []
    frame_evidence: list[dict[str, Any]] = []
    fix_signals: list[str] = []
    reject_reason = ""

    # -- objective delivery_failure detection (decode / duration) ----------
    decode_failed = not asset.get("decode_ok", False)
    too_short = dur is not None and dur < ABSOLUTE_MIN_DURATION_SEC
    if decode_failed or too_short:
        flags.append("delivery_failure")
        why = ("corrupt/unplayable stream" if decode_failed
               else f"duration {dur:.2f}s below usable minimum")
        reject_reason = f"{why} (delivery_failure)"
        frame_evidence.append({
            "frame_time_sec": 0.0, "issue": why,
            "flag_or_dimension": "delivery_failure",
        })

    # -- objective interval evidence (ingest freezedetect/blackdetect) -----
    freeze_iv = asset.get("freeze_intervals") or []
    black_iv = asset.get("black_intervals") or []

    def _coverage(intervals: list[dict]) -> float:
        if not dur:
            return 0.0
        return sum(max(0.0, iv["end"] - iv["start"]) for iv in intervals) / dur

    fcov, bcov = _coverage(freeze_iv), _coverage(black_iv)
    if "delivery_failure" not in flags and \
            max(fcov, bcov) >= pcfg.still_coverage_reject_frac:
        # no usable content: the clip is frozen/black essentially throughout
        # -> EXISTING delivery_failure semantics (§2.8), no new flag.
        # An all-black clip is also static, so prefer the more specific
        # "black" description when black coverage itself clears the bar.
        kind, cov, iv0 = (("black", bcov, black_iv[0])
                          if bcov >= pcfg.still_coverage_reject_frac
                          else ("frozen", fcov, freeze_iv[0]))
        flags.append("delivery_failure")
        why = f"video {kind} for ~{cov:.0%} of its duration"
        reject_reason = f"{why} (delivery_failure)"
        frame_evidence.append({
            "frame_time_sec": iv0["start"], "issue": why,
            "flag_or_dimension": "delivery_failure",
        })
    else:
        # partial intervals: temporal evidence for the human annotate stage
        for kind, ivs in (("frozen", freeze_iv), ("black", black_iv)):
            for iv in ivs:
                frame_evidence.append({
                    "frame_time_sec": iv["start"],
                    "issue": f"{kind} segment {iv['start']:.2f}s-{iv['end']:.2f}s",
                    "flag_or_dimension": "temporal_stability",
                })

    # -- technical dimension scores (via the pluggable metrics backend) ----
    scores: dict[str, Optional[int]] = {d: None for d in DIMENSIONS}
    if frame_paths:
        backend = backend or build_metrics_backend(cfg)
        ta = backend.assess(asset.get("file_path"), frame_paths, pcfg)
        scores["sharpness_focus"] = ta.sharpness
        scores["exposure_dynamic_range"] = ta.exposure
        scores["temporal_stability"] = ta.temporal
        fix_signals += ta.fix_signals
        # Priors for dims not assessable from pixels alone.
        scores["composition_framing"] = pcfg.composition_prior
        scores["motion_quality"] = pcfg.motion_prior
        # prompt_fidelity_coherence: N/A offline (no prompt/reference) unless MLLM.
        scores["prompt_fidelity_coherence"] = None

    # -- derive heuristic verdict (§1/§5/§6) ------------------------------
    if flags:
        verdict_info = {
            "verdict": "REJECT", "fix_actions": [],
            "reject_reason": reject_reason,
            "primary_reasons": flags[:3],
        }
    else:
        # de-dup fix signals preserving order
        seen: set[str] = set()
        fix_signals = [s for s in fix_signals if not (s in seen or seen.add(s))]
        verdict_info = derive_verdict(
            scores, flags, gate, fix_signals=fix_signals, duration_sec=dur
        )

    ctx = {
        "shot_type": [], "asset_role": [], "source": ["generated"],
        "content_category": [], "aesthetic_family": [],
        "motion_complexity": "", "text_presence": "",
        "duration_bucket": duration_bucket(dur) if dur else "",
    }

    rec = AnnotationRecord(
        asset_id=aid,
        file_path=asset.get("file_path", ""),
        duration_sec=dur,
        verdict=verdict_info["verdict"],
        hard_fail_flags=flags,
        scores={k: v for k, v in scores.items() if v is not None},
        context_tags=ctx,
        fix_actions=verdict_info.get("fix_actions", []),
        reject_reason=verdict_info.get("reject_reason", ""),
        frame_evidence=frame_evidence,
        needs_human_review=True,      # prelabels always need human confirmation
        reviewer="prelabel-auto",
    )
    return rec


def run(cfg: PipelineConfig) -> dict[str, Any]:
    index_path = cfg.stage_dir("ingest") / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"ingest index not found at {index_path}; run `pipeline run ingest` first"
        )
    index = read_json(index_path)
    backend = build_metrics_backend(cfg)
    records: list[dict] = []
    verdict_counts: dict[str, int] = {"PASS": 0, "FIX": 0, "REJECT": 0}
    for asset in index["assets"]:
        rec = _prelabel_asset(asset, cfg, backend=backend)
        records.append(rec.model_dump())
        verdict_counts[rec.verdict] += 1

    out = cfg.stage_dir("prelabel") / "prelabels.jsonl"
    write_jsonl(out, records)
    return {
        "stage": "prelabel",
        "n_records": len(records),
        "verdicts": verdict_counts,
        "metrics_backend": backend.name,
        "prelabels_path": str(out),
        "taxonomy_version": TAXONOMY_VERSION,
    }
