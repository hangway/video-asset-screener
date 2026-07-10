"""Schema validation of annotation (§7.1) and inference (§7.2) records."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_screener.config import load_config
from video_screener.schema import AnnotationRecord, InferenceRecord


# --------------------------- §7.1 annotation ------------------------------
def _pass_record(**kw):
    base = dict(
        asset_id="a1",
        file_path="samples/a1.mp4",
        duration_sec=5.0,
        verdict="PASS",
        hard_fail_flags=[],
        scores=dict(
            sharpness_focus=3,
            exposure_dynamic_range=3,
            composition_framing=3,
            prompt_fidelity_coherence=3,
            temporal_stability=3,
            motion_quality=3,
        ),
    )
    base.update(kw)
    return base


def test_valid_pass_record():
    rec = AnnotationRecord.model_validate(_pass_record())
    assert rec.verdict == "PASS"
    assert rec.scores.sharpness_focus == 3


def test_invalid_flag_id_rejected():
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate(
            _pass_record(verdict="REJECT", hard_fail_flags=["not_a_real_flag"])
        )


def test_pass_cannot_have_flag():
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate(
            _pass_record(hard_fail_flags=["watermark_contamination"])
        )


def test_reject_requires_reason_and_evidence():
    # flag present but no reason -> fail
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate(
            _pass_record(
                verdict="REJECT",
                hard_fail_flags=["watermark_contamination"],
                reject_reason="",
            )
        )
    # flag present, reason present, but no frame_evidence -> fail (§5)
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate(
            _pass_record(
                verdict="REJECT",
                hard_fail_flags=["watermark_contamination"],
                reject_reason="visible watermark",
                frame_evidence=[],
            )
        )
    # complete REJECT is valid
    rec = AnnotationRecord.model_validate(
        _pass_record(
            verdict="REJECT",
            hard_fail_flags=["watermark_contamination"],
            reject_reason="visible watermark",
            frame_evidence=[
                dict(
                    frame_time_sec=1.0,
                    issue="logo bottom-right",
                    flag_or_dimension="watermark_contamination",
                )
            ],
        )
    )
    assert rec.verdict == "REJECT"


def test_reject_via_zero_score_no_flag():
    rec = AnnotationRecord.model_validate(
        _pass_record(
            verdict="REJECT",
            scores=dict(
                sharpness_focus=0,
                exposure_dynamic_range=2,
                composition_framing=2,
                prompt_fidelity_coherence=2,
                temporal_stability=2,
                motion_quality=2,
            ),
            reject_reason="severe blur, sharpness=0",
        )
    )
    assert rec.verdict == "REJECT"


def test_fix_requires_fix_actions():
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate(_pass_record(verdict="FIX", fix_actions=[]))
    rec = AnnotationRecord.model_validate(
        _pass_record(verdict="FIX", fix_actions=["color_grade"])
    )
    assert rec.fix_actions == ["color_grade"]


def test_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate(
            _pass_record(scores=dict(sharpness_focus=5))
        )


def test_null_score_is_na():
    rec = AnnotationRecord.model_validate(
        _pass_record(scores=dict(temporal_stability=None, motion_quality=None))
    )
    assert rec.scores.temporal_stability is None


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate(_pass_record(bogus="x"))


# --------------------------- §7.2 inference -------------------------------
def test_inference_valid():
    rec = InferenceRecord.model_validate(
        dict(
            asset_id="a1",
            verdict="FIX",
            confidence=0.8,
            scores={"sharpness_focus": 2.5},
            fix_actions=["deflicker"],
            primary_reasons=["temporal_stability=2"],
        )
    )
    assert rec.confidence == 0.8


def test_inference_confidence_bounds():
    with pytest.raises(ValidationError):
        InferenceRecord.model_validate(dict(verdict="PASS", confidence=1.5))


def test_inference_bad_score_key():
    with pytest.raises(ValidationError):
        InferenceRecord.model_validate(
            dict(verdict="PASS", confidence=0.5, scores={"not_a_dim": 3})
        )


def test_inference_pass_with_flag_rejected():
    with pytest.raises(ValidationError):
        InferenceRecord.model_validate(
            dict(
                verdict="PASS",
                confidence=0.9,
                hard_fail_flags=["watermark_contamination"],
            )
        )


def test_inference_too_many_reasons():
    with pytest.raises(ValidationError):
        InferenceRecord.model_validate(
            dict(
                verdict="REJECT",
                confidence=0.9,
                hard_fail_flags=["delivery_failure"],
                primary_reasons=["a", "b", "c", "d"],
            )
        )


# --------------------------- config validation ----------------------------
def test_config_defaults_load():
    cfg = load_config()
    assert cfg.taxonomy_version == "0.3.1"
    assert cfg.model.n_layers in (2, 3, 4)


def test_config_rejects_bad_flag_ids(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "train:\n  flag_pos_weights:\n    not_a_flag: 3.0\n"
    )
    with pytest.raises(ValueError):
        load_config(bad)


def test_config_rejects_bad_gate_dim(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("gate_min:\n  not_a_dim: 2\n")
    with pytest.raises((ValueError, Exception)):
        load_config(bad)
