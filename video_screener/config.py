"""Pipeline configuration: YAML on disk, validated by pydantic on load.

Invalid flag IDs, dimension names, or verdicts referenced anywhere in a config
are rejected at load time (closed-vocabulary discipline). One ``PipelineConfig``
object is threaded through every stage; each stage reads its own sub-config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .taxonomy_schema import (
    DIMENSIONS,
    GATE_MIN,
    HARD_FAIL_FLAGS,
    TAXONOMY_VERSION,
)


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fps: float = 1.0                       # §5 uniform 1 fps
    short_clip_threshold_sec: float = 4.0  # §5 "<4s"
    short_clip_interval_sec: float = 0.5   # §5 sample every 0.5s for <4s clips
    scene_detect: bool = True              # §5 add scene-change frames
    scene_threshold: float = 27.0          # scenedetect ContentDetector default
    max_frames: int = 64                   # cap frames per clip (memory guard)
    phash_size: int = 16                   # perceptual hash size
    dedup_hamming_threshold: int = 4       # <= this Hamming distance => near-dup
    frame_format: str = "jpg"
    # Objective interval scan (ffprobe freezedetect + blackdetect): records
    # frozen-video and black intervals per asset as evidence for prelabel.
    interval_scan: bool = True
    freeze_noise_db: float = -60.0         # freezedetect noise tolerance
    freeze_min_sec: float = 1.0            # minimum freeze duration to report
    black_min_sec: float = 0.5             # minimum black interval to report
    black_pic_th: float = 0.98             # fraction of black pixels per frame


class FfprobePrelabelThresholds(BaseModel):
    """Backend-scoped thresholds for the ffprobe signalstats/blurdetect
    metrics (their scales differ from the opencv statistics — YDIF is a mean
    per-pixel luma change, blurdetect is higher-is-blurrier — so opencv
    numbers must never be reused here)."""

    model_config = ConfigDict(extra="forbid")

    # Calibrated by MEASUREMENT on samples/ (see notes.md "signal-layer
    # upgrade"): clean/duplicate blur=7.63, lowres=15.60, flicker=4.73,
    # underexposed=5.98, watermark=3.58; YDIF clean=3.24, flicker=7.36,
    # others <2; no sample is mostly-crushed (clip fraction 0.0).
    # blurdetect blurriness bins, DESCENDING: score = #(blur_mean <= bin).
    blur_bins: list[float] = Field(default_factory=lambda: [15.0, 12.0, 10.0, 8.0])
    # mean-YDIF flicker thresholds (mean abs per-pixel luma change / frame)
    ydif_low: float = 5.0     # above -> borderline flicker (FIX)
    ydif_high: float = 12.0   # above -> severe flicker
    # clipping proxy: a frame counts as clipped when it is mostly crushed —
    # 10th percentile pinned white (YLOW >= this) or 90th percentile pinned
    # black (YHIGH <= this). The clip's clipped-frame fraction is compared
    # against the shared exposure_clip_fraction.
    clip_white_ylow_min: float = 247.0
    clip_black_yhigh_max: float = 8.0


class PrelabelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Ascending variance-of-Laplacian bins -> sharpness score = #bins exceeded
    # (calibrated on the synthetic samples). 5 scores (0-4) from 4 thresholds.
    sharpness_bins: list[float] = Field(default_factory=lambda: [60.0, 150.0, 400.0, 1200.0])
    # Mean-brightness bounds (0-255) for a "well-exposed" frame.
    exposure_dark: float = 55.0            # below -> underexposed (recoverable)
    exposure_bright: float = 215.0         # above -> overexposed (recoverable)
    exposure_severe_dark: float = 25.0     # below -> exposure score 1
    exposure_severe_bright: float = 240.0  # above -> exposure score 1
    # Fraction of clipped (pure black/white) pixels that counts as severe.
    exposure_clip_fraction: float = 0.35
    # Temporal flicker: std of per-frame mean brightness (global pumping).
    flicker_low: float = 18.0              # above -> borderline flicker (FIX)
    flicker_high: float = 48.0             # above -> severe flicker
    # Neutral priors for dims that cannot be assessed from pixels alone.
    composition_prior: int = 3
    motion_prior: int = 3
    enable_mllm: bool = False              # optional MLLM prelabel (off offline)
    # Frozen-video / black intervals covering at least this fraction of the
    # clip mean there is no usable content -> the EXISTING delivery_failure
    # flag (§2.8: asset cannot be properly ingested downstream). Shorter
    # intervals become frame_evidence only.
    still_coverage_reject_frac: float = 0.9
    # Backend-scoped thresholds for metrics_backend="ffprobe" (different
    # measurement scales; the opencv thresholds above must not be reused).
    ffprobe: FfprobePrelabelThresholds = Field(
        default_factory=FfprobePrelabelThresholds
    )


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    seed: int = 1234
    # Group key(s) to keep together across splits (no leakage). Near-dups
    # (same phash bucket) and same-source always co-locate.
    group_by: list[str] = Field(default_factory=lambda: ["phash_bucket", "source_group"])

    @field_validator("train_frac", "val_frac", "test_frac")
    @classmethod
    def _frac_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("split fractions must be in [0,1]")
        return v


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder: str = "auto"          # auto | deterministic | clip:<name>
    feature_dim: int = 512         # per-frame feature dimensionality
    d_model: int = 256             # temporal transformer width
    n_layers: int = 3              # 2-4 per design
    n_heads: int = 4
    dropout: float = 0.1
    max_frames: int = 64


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = 20
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 1234
    # Loss weights across the three task groups.
    w_verdict: float = 1.0
    w_dims: float = 1.0
    w_flags: float = 1.0
    # Up-weight the positive class for hard-fail flags: missing a hard-fail
    # (false negative) costs more than a false alarm (design rule).
    flag_pos_weight: float = 4.0
    device: str = "cpu"
    num_workers: int = 0
    early_stop_patience: int = 6


class EvaluateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stratify_by: list[str] = Field(
        default_factory=lambda: ["aesthetic_family", "motion_complexity"]
    )
    worst_gallery_size: int = 12


class ConsistencyConfig(BaseModel):
    """Reference-consistency settings (wired to the existing
    ``reference_inconsistency`` flag; taxonomy v0.3.1 unchanged)."""

    model_config = ConfigDict(extra="forbid")

    # Directory of reference images: one subdirectory per subject (loose
    # images fall under subject "default"). None disables consistency checks.
    reference_dir: Optional[str] = None
    # Clip score = worst-frame best-match cosine vs the assigned subject's
    # references (§5). Below this the existing reference_inconsistency
    # hard-fail flag is raised at screen time.
    min_reference_similarity: float = 0.5
    # How many worst per-frame offenders to list per clip in the report.
    report_worst_k: int = 3
    # Within-clip drift: max consecutive-frame cosine distance above this
    # marks a morphing candidate -> needs_human_review (never auto-REJECT).
    max_frame_drift: float = 0.35
    # Head/tail edge stability: each edge window's per-frame similarity +
    # drift is compared against the clip body; a statistical outlier edge
    # (beyond edge_outlier_sigma body standard deviations) suggests
    # trim_head/trim_tail and routes FIX per §1 (trims are minor), never
    # REJECT.
    edge_window_sec: float = 1.0
    edge_outlier_sigma: float = 3.0


class ScreenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Confidence below this on a FIX/REJECT boundary sets needs_human_review.
    review_confidence_threshold: float = 0.55
    # Gate overrides for stratified gates (§4). Maps context value -> per-dim
    # gate min override. Validated against canonical dimension names.
    gate_overrides: dict[str, dict[str, int]] = Field(default_factory=dict)

    @field_validator("gate_overrides")
    @classmethod
    def _gate_keys(cls, v: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        for ctx, overrides in v.items():
            bad = [d for d in overrides if d not in DIMENSIONS]
            if bad:
                raise ValueError(
                    f"gate_overrides[{ctx!r}] references non-canonical "
                    f"dimensions {bad}"
                )
        return v


class DescribeConfig(BaseModel):
    """Optional semantic enrichment through the external ``video-analyzer`` CLI.

    This stage is deliberately separate from screening: its LLM description and
    transcript are informational artifacts and never change a PASS/FIX/REJECT
    decision.  The executable is an optional integration rather than a package
    dependency so the offline screening pipeline remains installable as-is.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    executable: str = "video-analyzer"
    client: Literal["ollama", "openai_api"] = "ollama"
    model: Optional[str] = None
    ollama_url: Optional[str] = None
    api_url: Optional[str] = None
    # Secret values are read from the environment and are never written into
    # stage artifacts.  The external CLI currently accepts the key as an
    # argument, so callers should also avoid sharing process listings.
    api_key_env: Optional[str] = None
    prompt: str = ""
    duration_sec: Optional[float] = None
    max_frames: Optional[int] = None
    whisper_model: Optional[str] = None
    language: Optional[str] = None
    device: Optional[str] = None
    temperature: Optional[float] = None
    keep_frames: bool = False
    timeout_sec: int = 900
    # Optional cost-control filter. When set, describe reads the existing
    # screen results and enriches only matching verdicts (for example PASS).
    screen_verdicts: Optional[list[Literal["PASS", "FIX", "REJECT"]]] = None

    @field_validator("duration_sec")
    @classmethod
    def _duration_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("describe.duration_sec must be > 0")
        return v

    @field_validator("max_frames")
    @classmethod
    def _frames_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("describe.max_frames must be > 0")
        return v

    @field_validator("temperature")
    @classmethod
    def _temperature_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("describe.temperature must be in [0, 1]")
        return v

    @field_validator("timeout_sec")
    @classmethod
    def _timeout_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("describe.timeout_sec must be > 0")
        return v


class PipelineConfig(BaseModel):
    """Root config validated on load. ``taxonomy_version`` is pinned."""

    model_config = ConfigDict(extra="forbid")

    taxonomy_version: str = TAXONOMY_VERSION
    workdir: str = "runs/default"
    video_dirs: list[str] = Field(default_factory=lambda: ["samples"])
    # Technical-metrics measurement backend: "ffprobe" (signalstats/
    # blurdetect, QCTools lineage — default since the sample verdict-parity
    # gate passed) or "opencv" (the original hand-rolled statistics, kept as
    # the fallback; the ffprobe backend also degrades to it per clip when a
    # probe fails or the binary is missing).
    metrics_backend: str = "ffprobe"
    # Optional explicit gate minimums; defaults to taxonomy GATE_MIN. Keys
    # must be canonical dimension names.
    gate_min: dict[str, int] = Field(default_factory=lambda: dict(GATE_MIN))

    ingest: IngestConfig = Field(default_factory=IngestConfig)
    prelabel: PrelabelConfig = Field(default_factory=PrelabelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    screen: ScreenConfig = Field(default_factory=ScreenConfig)
    describe: DescribeConfig = Field(default_factory=DescribeConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)

    @field_validator("metrics_backend")
    @classmethod
    def _backend_allowed(cls, v: str) -> str:
        if v not in ("opencv", "ffprobe"):
            raise ValueError(
                f"metrics_backend {v!r} not in ('opencv', 'ffprobe')"
            )
        return v

    @field_validator("taxonomy_version")
    @classmethod
    def _pin_version(cls, v: str) -> str:
        if v != TAXONOMY_VERSION:
            raise ValueError(
                f"config pins taxonomy_version {v!r} but code is frozen "
                f"against {TAXONOMY_VERSION!r}. Bump code + re-freeze first."
            )
        return v

    @field_validator("gate_min")
    @classmethod
    def _gate_keys(cls, v: dict[str, int]) -> dict[str, int]:
        bad = [d for d in v if d not in DIMENSIONS]
        if bad:
            raise ValueError(f"gate_min references non-canonical dimensions {bad}")
        return v

    # -- paths derived from workdir ----------------------------------------
    @property
    def root(self) -> Path:
        return Path(self.workdir)

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage

    def resolved_gate_min(self) -> dict[str, int]:
        merged = dict(GATE_MIN)
        merged.update(self.gate_min)
        return merged


def _reject_unknown_flag_ids(raw: dict[str, Any]) -> None:
    """Scan a raw config dict for any string that *looks* like a flag ID but
    is not in the canonical set. Catches typos like ``watermark_contam`` in
    fields we don't model explicitly (e.g. free-form override maps).
    """

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, val in obj.items():
                # keys that are meant to be flag IDs
                if k in {"flags", "hard_fail_flags", "flag_pos_weights"}:
                    vals = val if isinstance(val, (list, tuple)) else (
                        list(val.keys()) if isinstance(val, dict) else [val]
                    )
                    bad = [f for f in vals if f not in HARD_FAIL_FLAGS]
                    if bad:
                        raise ValueError(
                            f"config field {k!r} contains non-canonical "
                            f"flag IDs {bad}"
                        )
                walk(val)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)

    walk(raw)


def load_config(path: Optional[str | Path] = None, **overrides: Any) -> PipelineConfig:
    """Load + validate a YAML config. Returns defaults if ``path`` is None."""
    raw: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config {p} must be a YAML mapping")
    raw.update(overrides)
    _reject_unknown_flag_ids(raw)
    return PipelineConfig.model_validate(raw)
