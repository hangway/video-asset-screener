"""Pydantic models for the taxonomy output schemas (taxonomy.md §7).

- ``AnnotationRecord``  -> §7.1 annotation schema (human labeling + training data)
- ``InferenceRecord``   -> §7.2 minimal inference output schema (screen stage)

Every stage emits and validates against these. The closed vocabulary from
``taxonomy_schema`` is enforced here: invalid flag IDs or dimension keys are
rejected at construction time, not silently accepted.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .taxonomy_schema import (
    DIMENSIONS,
    HARD_FAIL_FLAGS,
    SCORE_MAX,
    SCORE_MIN,
    VERDICTS,
)


# ---------------------------------------------------------------------------
# §7.1 sub-objects
# ---------------------------------------------------------------------------
class Scores(BaseModel):
    """§3 dimension scores. Integers 0-4 or null (null = N/A for that dim)."""

    model_config = ConfigDict(extra="forbid")

    sharpness_focus: Optional[int] = None
    exposure_dynamic_range: Optional[int] = None
    composition_framing: Optional[int] = None
    prompt_fidelity_coherence: Optional[int] = None
    temporal_stability: Optional[int] = None
    motion_quality: Optional[int] = None

    @field_validator("*")
    @classmethod
    def _in_range(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("scores must be integers 0-4 or null")
        if not (SCORE_MIN <= v <= SCORE_MAX):
            raise ValueError(f"score {v} out of range [{SCORE_MIN},{SCORE_MAX}]")
        return v

    def as_dict(self) -> dict[str, Optional[int]]:
        return {d: getattr(self, d) for d in DIMENSIONS}


class ContextTags(BaseModel):
    """§4 context tags. List fields are multi-label; scalar fields single value.

    §4 explicitly allows free-form values during exploration, so unknown values
    are accepted here (validation *warnings* are surfaced by config/prelabel,
    not hard errors) — but the field *shape* is enforced.
    """

    model_config = ConfigDict(extra="forbid")

    shot_type: list[str] = Field(default_factory=list)
    asset_role: list[str] = Field(default_factory=list)
    source: list[str] = Field(default_factory=list)
    content_category: list[str] = Field(default_factory=list)
    aesthetic_family: list[str] = Field(default_factory=list)
    motion_complexity: str = ""
    duration_bucket: str = ""
    text_presence: str = ""


class FrameEvidence(BaseModel):
    """§7.1 frame_evidence entry. Required for every triggered flag (§5)."""

    model_config = ConfigDict(extra="forbid")

    frame_time_sec: Optional[float] = None
    issue: str = ""
    flag_or_dimension: str = ""


# ---------------------------------------------------------------------------
# §7.1 Annotation record
# ---------------------------------------------------------------------------
class AnnotationRecord(BaseModel):
    """Full §7.1 annotation record. This is the training/eval data contract."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = ""
    file_path: str = ""
    duration_sec: Optional[float] = None
    verdict: str = ""
    hard_fail_flags: list[str] = Field(default_factory=list)
    scores: Scores = Field(default_factory=Scores)
    context_tags: ContextTags = Field(default_factory=ContextTags)
    fix_actions: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    reject_reason: str = ""
    frame_evidence: list[FrameEvidence] = Field(default_factory=list)
    reviewer: str = ""
    review_date: str = ""

    @field_validator("verdict")
    @classmethod
    def _verdict_allowed(cls, v: str) -> str:
        if v not in VERDICTS:
            raise ValueError(
                f"verdict {v!r} not in closed set {VERDICTS}"
            )
        return v

    @field_validator("hard_fail_flags")
    @classmethod
    def _flags_closed_vocab(cls, v: list[str]) -> list[str]:
        bad = [f for f in v if f not in HARD_FAIL_FLAGS]
        if bad:
            raise ValueError(
                f"hard_fail_flags {bad} not in canonical closed set "
                f"{HARD_FAIL_FLAGS}"
            )
        # de-dupe while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for f in v:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    @model_validator(mode="after")
    def _taxonomy_rules(self) -> "AnnotationRecord":
        """Enforce §7.1 cross-field rules and §1 verdict/flag coherence."""
        if self.verdict == "REJECT":
            # §7.1: reject_reason non-empty AND (>=1 flag OR >=1 score of 0)
            if not self.reject_reason.strip():
                raise ValueError("REJECT requires a non-empty reject_reason")
            has_flag = len(self.hard_fail_flags) > 0
            has_zero = any(
                getattr(self.scores, d) == 0 for d in DIMENSIONS
            )
            if not (has_flag or has_zero):
                raise ValueError(
                    "REJECT requires >=1 hard-fail flag OR >=1 dimension score of 0"
                )
        if self.verdict == "FIX":
            # §7.1: FIX requires non-empty fix_actions
            if not self.fix_actions:
                raise ValueError("FIX requires non-empty fix_actions")
        if self.verdict == "PASS":
            # §1: PASS => zero hard-fail flags
            if self.hard_fail_flags:
                raise ValueError(
                    "PASS cannot co-occur with any hard-fail flag "
                    "(§1: PASS requires zero hard-fail flags)"
                )
        # §1/§2: any hard-fail flag => must be REJECT
        if self.hard_fail_flags and self.verdict != "REJECT":
            raise ValueError(
                "a hard-fail flag present requires verdict REJECT (§2)"
            )
        # §5 auditability: each triggered flag needs frame_evidence
        if self.hard_fail_flags:
            evidenced = {
                fe.flag_or_dimension for fe in self.frame_evidence
            }
            missing = [f for f in self.hard_fail_flags if f not in evidenced]
            if missing:
                raise ValueError(
                    f"frame_evidence required for triggered flag(s) {missing} (§5)"
                )
        return self


# ---------------------------------------------------------------------------
# §7.2 Minimal inference output record
# ---------------------------------------------------------------------------
class InferenceRecord(BaseModel):
    """§7.2 minimal inference output. Emitted by the screen stage."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = ""
    verdict: str = ""
    confidence: float = 0.0
    hard_fail_flags: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    fix_actions: list[str] = Field(default_factory=list)
    primary_reasons: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    # Provenance, not taxonomy vocabulary (audit A6): the effective frame
    # encoder that produced this record, so a CLIP->deterministic downgrade
    # is visible in the artifact. Empty string = pre-A6 record.
    encoder: str = ""

    @field_validator("verdict")
    @classmethod
    def _verdict_allowed(cls, v: str) -> str:
        if v not in VERDICTS:
            raise ValueError(f"verdict {v!r} not in closed set {VERDICTS}")
        return v

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        return v

    @field_validator("hard_fail_flags")
    @classmethod
    def _flags_closed_vocab(cls, v: list[str]) -> list[str]:
        bad = [f for f in v if f not in HARD_FAIL_FLAGS]
        if bad:
            raise ValueError(
                f"hard_fail_flags {bad} not in canonical closed set"
            )
        return v

    @field_validator("scores")
    @classmethod
    def _scores_keys_and_range(cls, v: dict[str, float]) -> dict[str, float]:
        # §7.2: scores object may be sparse but keys must be canonical dims.
        bad = [k for k in v if k not in DIMENSIONS]
        if bad:
            raise ValueError(f"score keys {bad} not in canonical dimensions")
        for k, val in v.items():
            if val is not None and not (SCORE_MIN <= val <= SCORE_MAX):
                raise ValueError(f"score {k}={val} out of range")
        return v

    @field_validator("primary_reasons")
    @classmethod
    def _reasons_count(cls, v: list[str]) -> list[str]:
        # §7.2: 1-3 short strings. Allow empty only for clean PASS convenience,
        # but cap at 3 to honor the contract.
        if len(v) > 3:
            raise ValueError("primary_reasons must be 1-3 strings")
        return v

    @model_validator(mode="after")
    def _coherence(self) -> "InferenceRecord":
        # §1: PASS => zero hard-fail flags; any flag => REJECT.
        if self.verdict == "PASS" and self.hard_fail_flags:
            raise ValueError("PASS cannot co-occur with a hard-fail flag")
        if self.hard_fail_flags and self.verdict != "REJECT":
            raise ValueError("hard-fail flag present requires verdict REJECT")
        return self
