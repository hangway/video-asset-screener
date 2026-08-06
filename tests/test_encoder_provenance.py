"""Encoder resolution + provenance (audit A6).

An EXPLICIT ``clip:<name>`` spec that fails to load must raise — the silent
fallback to DeterministicEncoder stripped reference-consistency screening of
meaning (that encoder cannot judge identity) with no trace in any artifact.
``auto`` keeps the fallback as part of its contract, and the effective name
is stamped into stage artifacts so a downgrade is always visible.
"""

from __future__ import annotations

from types import SimpleNamespace

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


def test_unknown_encoder_spec_does_not_fall_through_to_auto(monkeypatch):
    def should_not_try_auto(*args, **kwargs):
        pytest.fail("an unknown spec must be rejected before auto resolution")

    monkeypatch.setattr(enc, "ClipEncoder", should_not_try_auto)
    cfg = PipelineConfig()
    cfg.model.encoder = "clpi:ViT-B-32"
    with pytest.raises(
        ValueError,
        match=r"clpi:ViT-B-32.*auto.*deterministic.*clip:<name>",
    ):
        enc.build_encoder(cfg)


def test_checkpoint_encoder_validation_accepts_match_and_legacy():
    current = SimpleNamespace(name="deterministic")
    enc.validate_checkpoint_encoder({"encoder": "deterministic"}, current)
    enc.validate_checkpoint_encoder({}, current)


def test_checkpoint_encoder_validation_rejects_mismatch():
    current = SimpleNamespace(name="deterministic")
    with pytest.raises(
        RuntimeError,
        match=r"checkpoint encoder 'clip:ViT-B-32'.*effective encoder "
              r"'deterministic'.*different feature space",
    ):
        enc.validate_checkpoint_encoder({"encoder": "clip:ViT-B-32"}, current)


@pytest.mark.parametrize("stage_name", ["evaluate", "screen"])
def test_inference_stage_checks_checkpoint_encoder(
    monkeypatch, tmp_path, stage_name
):
    if stage_name == "evaluate":
        from video_screener.stages import evaluate as stage
    else:
        from video_screener.stages import screen as stage

    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    ckpt_path = cfg.stage_dir("train") / "model.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.touch()
    monkeypatch.setattr(
        stage,
        "load_model",
        lambda path: (object(), {"encoder": "clip:ViT-B-32"}),
    )
    monkeypatch.setattr(
        stage,
        "build_encoder",
        lambda config: SimpleNamespace(name="deterministic"),
    )

    with pytest.raises(RuntimeError, match="checkpoint encoder"):
        stage.run(cfg)


def test_train_resume_checks_checkpoint_encoder(monkeypatch, tmp_path):
    from video_screener.stages import train as stage
    from video_screener.utils.io import write_jsonl

    cfg = PipelineConfig(workdir=str(tmp_path / "run"))
    record = {
        "asset_id": "a",
        "frames": ["missing.jpg"],
        "verdict": "PASS",
        "scores": {},
        "hard_fail_flags": [],
    }
    dataset_dir = cfg.stage_dir("dataset")
    dataset_dir.mkdir(parents=True)
    write_jsonl(dataset_dir / "train.jsonl", [record])
    write_jsonl(dataset_dir / "val.jsonl", [])
    ckpt_path = cfg.stage_dir("train") / "model.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.touch()

    monkeypatch.setattr(
        stage,
        "build_encoder",
        lambda config: SimpleNamespace(name="deterministic", feature_dim=8),
    )
    monkeypatch.setattr(
        stage.torch,
        "load",
        lambda *args, **kwargs: {"encoder": "clip:ViT-B-32"},
    )

    with pytest.raises(RuntimeError, match="checkpoint encoder"):
        stage.run(cfg, resume=True)


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
