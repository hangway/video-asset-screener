"""Frozen, machine-readable projection of taxonomy.md (v0.3.1).

taxonomy.md is prose + JSON and is the *human* source of truth. This module
freezes the parts that code must treat as a **closed vocabulary**: the 9
canonical hard-fail flag IDs (§2), the 6 scored dimension names (§3), the 3
verdict labels (§1), per-dimension gate minimums (§3), and the context-tag
allowed values (§4).

Nothing here may be invented or edited without a taxonomy version bump. A test
(tests/test_taxonomy_freeze.py) asserts these constants still match the tables
in taxonomy.md §2/§3 so drift is caught mechanically.
"""

from __future__ import annotations

from types import MappingProxyType

# The taxonomy version this schema is frozen against. Pinned everywhere the
# pipeline writes artifacts so downstream consumers can detect drift.
TAXONOMY_VERSION = "0.3.1"

# ---------------------------------------------------------------------------
# §1 Verdict (top-level label)
# ---------------------------------------------------------------------------
VERDICTS: tuple[str, ...] = ("PASS", "FIX", "REJECT")

# ---------------------------------------------------------------------------
# §2 Hard-fail flags — closed vocabulary. Any one triggers REJECT.
# Order matches taxonomy.md §2 sections 2.1 .. 2.9 and is STABLE: the flag
# head index in the model corresponds to this order.
# ---------------------------------------------------------------------------
HARD_FAIL_FLAGS: tuple[str, ...] = (
    "watermark_contamination",   # 2.1
    "severe_ai_artifact",        # 2.2
    "brief_non_compliance",      # 2.3
    "reference_inconsistency",   # 2.4
    "legal_copyright_risk",      # 2.5
    "privacy_data_risk",         # 2.6
    "safety_policy_risk",        # 2.7
    "delivery_failure",          # 2.8
    "dataset_contamination",     # 2.9
)

# Section number (as printed in taxonomy.md §2) per flag, for audit/reporting.
HARD_FAIL_FLAG_SECTIONS: MappingProxyType = MappingProxyType({
    "watermark_contamination": "2.1",
    "severe_ai_artifact": "2.2",
    "brief_non_compliance": "2.3",
    "reference_inconsistency": "2.4",
    "legal_copyright_risk": "2.5",
    "privacy_data_risk": "2.6",
    "safety_policy_risk": "2.7",
    "delivery_failure": "2.8",
    "dataset_contamination": "2.9",
})

# ---------------------------------------------------------------------------
# §3 Scored dimensions (0-4). Keys synced to §7.1 `scores` object (snake_case).
# Order is STABLE: dimension head/regressor index corresponds to this order.
# ---------------------------------------------------------------------------
DIMENSIONS: tuple[str, ...] = (
    "sharpness_focus",
    "exposure_dynamic_range",
    "composition_framing",
    "prompt_fidelity_coherence",
    "temporal_stability",
    "motion_quality",
)

# §3 "Gate min" per dimension. Baseline gate; may be raised for hero assets
# (context-dependent, applied in prelabel/screen, never mutated here).
GATE_MIN: MappingProxyType = MappingProxyType({
    "sharpness_focus": 2,
    "exposure_dynamic_range": 2,
    "composition_framing": 2,
    "prompt_fidelity_coherence": 2,
    "temporal_stability": 2,
    "motion_quality": 2,
})

# Score scale is 0..4 inclusive (integers), per §3.
SCORE_MIN = 0
SCORE_MAX = 4

# ---------------------------------------------------------------------------
# §5 Clip-level aggregation. temporal_stability & motion_quality use the
# WORST frame (most conservative). For a 0-4 scale where 0 is worst, "worst"
# means the MINIMUM per-frame score. All other dimensions use the MEAN.
# ---------------------------------------------------------------------------
WORST_FRAME_DIMENSIONS: tuple[str, ...] = (
    "temporal_stability",
    "motion_quality",
)
MEAN_FRAME_DIMENSIONS: tuple[str, ...] = tuple(
    d for d in DIMENSIONS if d not in WORST_FRAME_DIMENSIONS
)

# ---------------------------------------------------------------------------
# §4 Context tags (multi-label / single value). Allowed values are the
# universal buckets listed in §4. §4 permits free-form values during
# exploration, so these are used for validation warnings, not hard rejects.
# ---------------------------------------------------------------------------
CONTEXT_TAG_LIST_FIELDS: tuple[str, ...] = (
    "shot_type",
    "asset_role",
    "source",
    "content_category",
    "aesthetic_family",
)
CONTEXT_TAG_SCALAR_FIELDS: tuple[str, ...] = (
    "motion_complexity",
    "duration_bucket",
    "text_presence",
)

CONTEXT_TAG_VALUES: MappingProxyType = MappingProxyType({
    "shot_type": ("WS", "MS", "CU", "ECU", "insert", "OTS", "POV", "tracking"),
    "asset_role": (
        "hero", "supporting", "background", "plate", "reference",
        "establishing", "detail", "transition",
    ),
    "source": ("generated", "stock", "real-shot", "hybrid"),
    "content_category": (
        "character_focus", "full_body_action", "environment_establishing",
        "object_detail", "abstract_motion", "text_overlay",
        "multi_subject_interaction", "infographic", "talking_head",
    ),
    "aesthetic_family": (
        "photorealistic", "stylized_cinematic", "anime_2d", "3d_render",
        "illustration", "painterly", "cyberpunk_noir", "fantasy",
        "documentary", "abstract_experimental",
    ),
    "motion_complexity": (
        "static_or_minimal", "moderate_natural",
        "high_complexity_physics", "chaotic_or_stylized",
    ),
    "duration_bucket": ("very_short", "short", "medium", "long"),
    "text_presence": ("none", "minor", "prominent", "heavy"),
})

# §4 duration_bucket thresholds (seconds). very_short <3, short 3-8,
# medium 8-20, long >20.
DURATION_BUCKET_BOUNDS: tuple[tuple[str, float, float], ...] = (
    ("very_short", 0.0, 3.0),
    ("short", 3.0, 8.0),
    ("medium", 8.0, 20.0),
    ("long", 20.0, float("inf")),
)

# ---------------------------------------------------------------------------
# §5 Minimum usable duration (general recommendation). Clips shorter than the
# absolute floor are effectively unusable (delivery/spec failure). We use the
# insert/transition floor (1s) as the hard technical floor and 3s as the
# general social/short-form recommendation.
# ---------------------------------------------------------------------------
MIN_USABLE_DURATION_SEC = 3.0          # general recommendation (§5)
ABSOLUTE_MIN_DURATION_SEC = 1.0        # insert/transition floor (§5)


def index_of_flag(flag_id: str) -> int:
    """Return the stable head index for a canonical flag ID."""
    return HARD_FAIL_FLAGS.index(flag_id)


def index_of_dimension(dim: str) -> int:
    """Return the stable regressor index for a canonical dimension name."""
    return DIMENSIONS.index(dim)


def is_valid_flag(flag_id: str) -> bool:
    return flag_id in HARD_FAIL_FLAGS


def is_valid_dimension(dim: str) -> bool:
    return dim in DIMENSIONS


def duration_bucket(duration_sec: float) -> str:
    """Map a duration to its §4 duration_bucket."""
    for name, lo, hi in DURATION_BUCKET_BOUNDS:
        if lo <= duration_sec < hi:
            return name
    return "long"


NUM_FLAGS = len(HARD_FAIL_FLAGS)
NUM_DIMENSIONS = len(DIMENSIONS)
NUM_VERDICTS = len(VERDICTS)
