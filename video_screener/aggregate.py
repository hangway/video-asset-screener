"""Deterministic clip-level aggregation + verdict derivation (taxonomy §1/§5/§6).

This is the single source of truth for turning per-frame evidence into a
clip-level verdict, shared by prelabel (heuristic labels), screen (model output
routing), and evaluate (consistency checks). Keeping it in one place guarantees
the aggregation rules are applied identically everywhere.

§5 aggregation:
  1. Any frame triggers a hard-fail flag -> clip REJECT (flags max-pooled).
  2. temporal_stability & motion_quality -> WORST frame (min score).
  3. All other dims -> MEAN across frames.

§6 priority when deriving a verdict from scores/flags:
  1. Any hard-fail flag -> REJECT.
  2. Prompt fidelity below gate -> REJECT (not a minor fix).
  3. Temporal/motion below gate (severe) -> REJECT (hard to fix w/o regen).
  4. Technical dims (sharpness/exposure/composition) below gate, or a detected
     fixable issue -> FIX. Otherwise PASS.
"""

from __future__ import annotations

from statistics import mean
from typing import Optional

from .taxonomy_schema import (
    DIMENSIONS,
    MEAN_FRAME_DIMENSIONS,
    WORST_FRAME_DIMENSIONS,
)

# Dims whose sub-gate failure forces REJECT rather than FIX (§6).
_HARD_TO_FIX_DIMS = set(WORST_FRAME_DIMENSIONS) | {"prompt_fidelity_coherence"}
# Dims that are post-production fixable (§1 FIX examples, §6).
_FIXABLE_DIMS = {"sharpness_focus", "exposure_dynamic_range", "composition_framing"}


def pool_dimension(dim: str, per_frame_scores: list[Optional[float]]) -> Optional[float]:
    """Pool one dimension's per-frame scores to clip level per §5."""
    vals = [s for s in per_frame_scores if s is not None]
    if not vals:
        return None
    if dim in WORST_FRAME_DIMENSIONS:
        return float(min(vals))          # worst frame (most conservative)
    return float(mean(vals))             # mean-pool


def pool_frame_dimension_scores(
    per_frame: list[dict[str, Optional[float]]]
) -> dict[str, Optional[float]]:
    """Aggregate a list of per-frame score dicts to clip-level scores (§5)."""
    out: dict[str, Optional[float]] = {}
    for dim in DIMENSIONS:
        out[dim] = pool_dimension(dim, [fr.get(dim) for fr in per_frame])
    return out


def pool_frame_flags(
    per_frame_flag_probs: list[dict[str, float]], threshold: float = 0.5
) -> dict[str, float]:
    """Max-pool per-frame flag probabilities to clip level (§5 rule 1: any
    frame triggering a flag triggers the clip)."""
    clip: dict[str, float] = {}
    for fr in per_frame_flag_probs:
        for flag, p in fr.items():
            clip[flag] = max(clip.get(flag, 0.0), p)
    return clip


def derive_verdict(
    scores: dict[str, Optional[float]],
    flags: list[str],
    gate_min: dict[str, int],
    *,
    fix_signals: Optional[list[str]] = None,
    duration_sec: Optional[float] = None,
) -> dict:
    """Derive (verdict, fix_actions, reject_reason, primary_reasons) from
    clip-level scores + flags following §1/§5/§6.

    ``fix_signals`` are detected fixable issues (e.g. ["deflicker"]) that
    justify FIX even when all gates are met.
    """
    fix_signals = list(fix_signals or [])
    reasons: list[str] = []

    # (1) Any hard-fail flag -> REJECT.
    if flags:
        return {
            "verdict": "REJECT",
            "fix_actions": [],
            "reject_reason": "; ".join(flags),
            "primary_reasons": flags[:3],
            "needs_human_review": False,
        }

    # Severe: any scored dim == 0 (unusable per §3 anchors).
    zero_dims = [d for d in DIMENSIONS if scores.get(d) == 0]
    if zero_dims:
        return {
            "verdict": "REJECT",
            "fix_actions": [],
            "reject_reason": f"{zero_dims[0]}=0 (unusable)",
            "primary_reasons": [f"{d}=0" for d in zero_dims[:3]],
            "needs_human_review": False,
        }

    # Gate violations (score strictly below gate min), split by fixability.
    gate_viol = [
        d for d in DIMENSIONS
        if scores.get(d) is not None and scores[d] < gate_min.get(d, 2)
    ]
    hard_viol = [d for d in gate_viol if d in _HARD_TO_FIX_DIMS]
    tech_viol = [d for d in gate_viol if d in _FIXABLE_DIMS]

    if hard_viol:
        return {
            "verdict": "REJECT",
            "fix_actions": [],
            "reject_reason": "; ".join(
                f"{d}={scores[d]:.1f}<gate{gate_min.get(d,2)}" for d in hard_viol
            ),
            "primary_reasons": [d for d in hard_viol[:3]],
            "needs_human_review": False,
        }

    if tech_viol or fix_signals:
        actions = list(fix_signals)
        for d in tech_viol:
            actions.append(_FIX_ACTION_FOR.get(d, "post_production"))
        # de-dup preserving order
        seen: set[str] = set()
        actions = [a for a in actions if not (a in seen or seen.add(a))]
        reasons = [f"{d}={scores[d]:.1f}" for d in tech_viol] + fix_signals
        return {
            "verdict": "FIX",
            "fix_actions": actions or ["post_production"],
            "reject_reason": "",
            "primary_reasons": reasons[:3],
            "needs_human_review": False,
        }

    return {
        "verdict": "PASS",
        "fix_actions": [],
        "reject_reason": "",
        "primary_reasons": [],
        "needs_human_review": False,
    }


_FIX_ACTION_FOR = {
    "sharpness_focus": "sharpen",
    "exposure_dynamic_range": "color_grade",
    "composition_framing": "reframe_crop",
}


def consistency_violations(
    verdict: str, scores: dict[str, Optional[float]], flags: list[str]
) -> list[str]:
    """Return taxonomy-coherence violations for a (verdict, scores, flags)
    triple. Used by evaluate's verdict-vs-flags/dims consistency check.

    Examples of violations:
      - PASS co-occurring with any hard-fail flag (§1).
      - PASS co-occurring with any dimension score of 0 (§1 gate rule).
      - A hard-fail flag present but verdict != REJECT (§2).
    """
    v: list[str] = []
    if verdict == "PASS" and flags:
        v.append("PASS_with_flag")
    if verdict == "PASS":
        zero = [d for d in DIMENSIONS if scores.get(d) == 0]
        if zero:
            v.append(f"PASS_with_zero_dim:{zero[0]}")
        below = [
            d for d in DIMENSIONS
            if scores.get(d) is not None and scores[d] < 2
        ]
        if below:
            v.append(f"PASS_with_subgate_dim:{below[0]}")
    if flags and verdict != "REJECT":
        v.append("flag_without_reject")
    return v
