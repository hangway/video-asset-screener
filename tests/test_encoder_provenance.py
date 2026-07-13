"""Encoder resolution + provenance (audit A6).

An EXPLICIT ``clip:<name>`` spec that fails to load must raise — the silent
fallback to DeterministicEncoder stripped reference-consistency screening of
meaning (that encoder cannot judge identity) with no trace in any artifact.
``auto`` keeps the fallback as part of its contract, and the effective name
is stamped into stage artifacts so a downgrade is always visible.
"""

from __future__ import annotations

import pytest

import video_screener.models.encoder as enc
from video_screener.config import PipelineConfig


def _failing_clip_encoder(*args, **kwargs):
    raise ImportError("open_clip unavailable (simulated)")


@pytest.fixture(autouse=True)
def _clear_name_cache():
    enc._NAME_CACHE.clear()
    yield
    enc._NAME_CACHE.clear()


def test_explicit_clip_spec_failure_raises(monkeypatch):
    monkeypatch.setattr(enc, "ClipEncoder", _failing_clip_encoder)
    cfg = PipelineConfig()
    cfg.model.encoder = "clip:ViT-B-32"
    with pytest.raises(RuntimeError, match=r"clip:ViT-B-32.*failed to load"):
        enc.build_encoder(cfg)


def test_explicit_clip_spec_error_names_cause(monkeypatch):
    monkeypatch.setattr(enc, "ClipEncoder", _failing_clip_encoder)
    cfg = PipelineConfig()
    cfg.model.encoder = "clip:whatever"
    with pytest.raises(RuntimeError) as ei:
        enc.build_encoder(cfg)
    assert "ImportError" in str(ei.value)
    assert isinstance(ei.value.__cause__, ImportError)


def test_auto_still_falls_back_to_deterministic(monkeypatch):
    monkeypatch.setattr(enc, "ClipEncoder", _failing_clip_encoder)
    cfg = PipelineConfig()
    cfg.model.encoder = "auto"
    built = enc.build_encoder(cfg)
    assert isinstance(built, enc.DeterministicEncoder)
    assert built.name == "deterministic"


def test_effective_encoder_name_reports_auto_downgrade(monkeypatch):
    monkeypatch.setattr(enc, "ClipEncoder", _failing_clip_encoder)
    cfg = PipelineConfig()
    cfg.model.encoder = "auto"
    # the stamp records what actually runs, not the configured spec
    assert enc.effective_encoder_name(cfg) == "deterministic"


def test_effective_encoder_name_is_memoized(monkeypatch):
    calls = []

    def counting_build(cfg):
        calls.append(1)
        return enc.DeterministicEncoder(feature_dim=cfg.model.feature_dim)

    monkeypatch.setattr(enc, "build_encoder", counting_build)
    cfg = PipelineConfig()
    cfg.model.encoder = "auto"
    assert enc.effective_encoder_name(cfg) == "deterministic"
    assert enc.effective_encoder_name(cfg) == "deterministic"
    assert len(calls) == 1


def test_inference_record_accepts_encoder_field():
    from video_screener.schema import InferenceRecord

    rec = InferenceRecord(asset_id="a", verdict="PASS", encoder="deterministic")
    assert InferenceRecord.model_validate(rec.model_dump()).encoder == "deterministic"
    # pre-A6 records (no encoder key) still validate
    old = rec.model_dump()
    old.pop("encoder")
    assert InferenceRecord.model_validate(old).encoder == ""
