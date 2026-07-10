"""Stage 2 prelabel: technical-dim heuristics, delivery_failure, schema."""

from __future__ import annotations

import json

from video_screener.aggregate import derive_verdict
from video_screener.config import PipelineConfig
from video_screener.schema import AnnotationRecord
from video_screener.stages import ingest, prelabel
from tests.conftest import requires_ffmpeg


def _run(tmp_path, samples_dir, metrics_backend: str = "opencv"):
    cfg = PipelineConfig(workdir=str(tmp_path / "run"), video_dirs=[str(samples_dir)],
                         metrics_backend=metrics_backend)
    ingest.run(cfg)
    prelabel.run(cfg)
    rows = [
        json.loads(l)
        for l in (tmp_path / "run" / "prelabel" / "prelabels.jsonl").read_text().splitlines()
    ]
    return {r["asset_id"]: r for r in rows}


@requires_ffmpeg
def test_all_prelabels_validate(tmp_path, synth_samples):
    recs = _run(tmp_path, synth_samples)
    for r in recs.values():
        AnnotationRecord.model_validate(r)  # raises on taxonomy violation


@requires_ffmpeg
def test_delivery_failure_on_corrupt_and_short(tmp_path, synth_samples):
    recs = _run(tmp_path, synth_samples)
    assert "delivery_failure" in recs["corrupted"]["hard_fail_flags"]
    assert recs["corrupted"]["verdict"] == "REJECT"
    assert "delivery_failure" in recs["tooshort"]["hard_fail_flags"]
    assert recs["tooshort"]["verdict"] == "REJECT"
    # objective flags carry frame_evidence (§5)
    ev = {e["flag_or_dimension"] for e in recs["corrupted"]["frame_evidence"]}
    assert "delivery_failure" in ev


@requires_ffmpeg
def test_lowres_sharpness_zero_rejects(tmp_path, synth_samples):
    recs = _run(tmp_path, synth_samples)
    assert recs["lowres"]["scores"]["sharpness_focus"] == 0
    assert recs["lowres"]["verdict"] == "REJECT"


@requires_ffmpeg
def test_flicker_is_fix_with_deflicker(tmp_path, synth_samples):
    recs = _run(tmp_path, synth_samples)
    assert recs["flicker"]["verdict"] == "FIX"
    assert "deflicker" in recs["flicker"]["fix_actions"]
    assert recs["flicker"]["scores"]["temporal_stability"] <= 2


@requires_ffmpeg
def test_underexposed_is_fix_with_grade(tmp_path, synth_samples):
    recs = _run(tmp_path, synth_samples)
    assert recs["underexposed"]["verdict"] == "FIX"
    assert recs["underexposed"]["scores"]["exposure_dynamic_range"] <= 2
    assert any("grade" in a or "exposure" in a for a in recs["underexposed"]["fix_actions"])


@requires_ffmpeg
def test_clean_and_duplicate_pass(tmp_path, synth_samples):
    recs = _run(tmp_path, synth_samples)
    assert recs["clean_pass"]["verdict"] == "PASS"
    assert recs["duplicate"]["verdict"] == "PASS"
    assert recs["clean_pass"]["scores"]["sharpness_focus"] == 4


@requires_ffmpeg
def test_all_prelabels_flagged_for_human_review(tmp_path, synth_samples):
    recs = _run(tmp_path, synth_samples)
    assert all(r["needs_human_review"] for r in recs.values())


@requires_ffmpeg
def test_prelabel_matches_sidecar_on_objective_cases(tmp_path, synth_samples):
    """Prelabel should match sidecar verdicts on all objective/technical cases.
    watermark is a documented exception (subjective flag deferred to human)."""
    recs = _run(tmp_path, synth_samples)
    for name in ["clean_pass", "duplicate", "corrupted", "flicker", "lowres",
                 "tooshort", "underexposed"]:
        sidecar = json.loads((synth_samples / f"{name}.json").read_text())
        assert recs[name]["verdict"] == sidecar["expected_verdict"], (
            f"{name}: prelabel {recs[name]['verdict']} != "
            f"sidecar {sidecar['expected_verdict']}"
        )
    # watermark: prelabel cannot detect the subjective watermark flag; it is
    # deferred to the human annotate stage (documented limitation).
    assert recs["watermark"]["verdict"] == "PASS"
    assert recs["watermark"]["hard_fail_flags"] == []


@requires_ffmpeg
def test_prelabel_ffprobe_backend_matches_sidecars_at_least_as_well(tmp_path, synth_samples):
    """Item-3 acceptance: with metrics_backend=ffprobe and the MEASURED
    thresholds, prelabel verdicts match sidecar ground truth on the same 7/8
    clips as the opencv backend (watermark stays the known subjective miss)."""
    recs = _run(tmp_path, synth_samples, metrics_backend="ffprobe")
    for name in ["clean_pass", "duplicate", "corrupted", "flicker", "lowres",
                 "tooshort", "underexposed"]:
        sidecar = json.loads((synth_samples / f"{name}.json").read_text())
        assert recs[name]["verdict"] == sidecar["expected_verdict"], (
            f"{name}: ffprobe prelabel {recs[name]['verdict']} != "
            f"sidecar {sidecar['expected_verdict']}"
        )
    assert recs["watermark"]["verdict"] == "PASS"          # known miss, both backends
    # the calibrated signals drive the same routing causes as opencv
    assert recs["lowres"]["scores"]["sharpness_focus"] == 0
    assert recs["flicker"]["scores"]["temporal_stability"] <= 2
    assert "deflicker" in recs["flicker"]["fix_actions"]
    assert recs["underexposed"]["scores"]["exposure_dynamic_range"] <= 2
    for r in recs.values():
        AnnotationRecord.model_validate(r)


# --------------------------- unit: derive_verdict --------------------------
def test_derive_verdict_flag_forces_reject():
    out = derive_verdict({}, ["watermark_contamination"], {}, )
    assert out["verdict"] == "REJECT"
    assert "watermark_contamination" in out["reject_reason"]


def test_derive_verdict_zero_dim_rejects():
    scores = {"sharpness_focus": 0.0, "temporal_stability": 3.0}
    out = derive_verdict(scores, [], {"sharpness_focus": 2, "temporal_stability": 2})
    assert out["verdict"] == "REJECT"


def test_derive_verdict_temporal_subgate_rejects():
    # temporal below gate is hard-to-fix -> REJECT (§6)
    scores = {d: 3.0 for d in ["sharpness_focus", "exposure_dynamic_range",
                               "composition_framing", "prompt_fidelity_coherence",
                               "motion_quality"]}
    scores["temporal_stability"] = 1.0
    gate = {d: 2 for d in scores}
    out = derive_verdict(scores, [], gate)
    assert out["verdict"] == "REJECT"


def test_derive_verdict_tech_subgate_fixes():
    # sharpness below gate (but not 0) -> FIX (fixable)
    scores = {d: 3.0 for d in ["exposure_dynamic_range", "composition_framing",
                               "prompt_fidelity_coherence", "temporal_stability",
                               "motion_quality"]}
    scores["sharpness_focus"] = 1.0
    gate = {d: 2 for d in scores}
    out = derive_verdict(scores, [], gate)
    assert out["verdict"] == "FIX"
    assert out["fix_actions"]


def test_derive_verdict_fix_signal_only():
    scores = {d: 3.0 for d in ["sharpness_focus", "exposure_dynamic_range",
                               "composition_framing", "prompt_fidelity_coherence",
                               "temporal_stability", "motion_quality"]}
    gate = {d: 2 for d in scores}
    out = derive_verdict(scores, [], gate, fix_signals=["deflicker"])
    assert out["verdict"] == "FIX"
    assert "deflicker" in out["fix_actions"]


def test_derive_verdict_clean_pass():
    scores = {d: 3.0 for d in ["sharpness_focus", "exposure_dynamic_range",
                               "composition_framing", "prompt_fidelity_coherence",
                               "temporal_stability", "motion_quality"]}
    gate = {d: 2 for d in scores}
    out = derive_verdict(scores, [], gate)
    assert out["verdict"] == "PASS"
