"""Stage 5 train: encoder determinism, overfit sanity, resume, checkpoint."""

from __future__ import annotations

import numpy as np

from video_screener.config import PipelineConfig
from video_screener.models.encoder import DeterministicEncoder, build_encoder
from video_screener.stages import annotate, dataset, ingest, prelabel, train
from tests.conftest import requires_ffmpeg


def _prep(tmp_path, samples_dir, **train_over) -> PipelineConfig:
    cfg = PipelineConfig(workdir=str(tmp_path / "run"), video_dirs=[str(samples_dir)])
    for k, v in train_over.items():
        setattr(cfg.train, k, v)
    ingest.run(cfg)
    prelabel.run(cfg)
    annotate.run(cfg, auto=True)
    dataset.run(cfg)
    return cfg


# ------------------------------ encoder -----------------------------------
def test_deterministic_encoder_is_deterministic(tmp_path):
    enc = DeterministicEncoder(feature_dim=128)
    img = (np.random.RandomState(0).rand(48, 64, 3) * 255).astype("uint8")
    import cv2
    p = tmp_path / "f.png"
    cv2.imwrite(str(p), img)
    a = enc.encode_paths([str(p)])
    b = enc.encode_paths([str(p)])
    assert a.shape == (1, 128)
    assert np.allclose(a, b)


def test_encoder_feature_dim_matches_config():
    cfg = PipelineConfig()
    cfg.model.encoder = "deterministic"
    enc = build_encoder(cfg)
    assert enc.feature_dim == cfg.model.feature_dim
    assert enc.name == "deterministic"


def test_encoder_empty_on_no_frames():
    enc = DeterministicEncoder(feature_dim=64)
    out = enc.encode_paths([])
    assert out.shape == (0, 64)


# ------------------------------ training ----------------------------------
@requires_ffmpeg
def test_training_reduces_loss(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples, epochs=25, seed=0)
    summary = train.run(cfg)
    import json
    log = json.loads((tmp_path / "run" / "train" / "train_log.json").read_text())
    first = log["history"][0]["train"]["loss"]
    last = log["history"][-1]["train"]["loss"]
    assert last < first, f"loss did not decrease: {first} -> {last}"
    assert last < 0.75 * first          # meaningful learning on the tiny set
    assert summary["encoder"] == "deterministic"
    assert summary["feature_dim"] == 512


@requires_ffmpeg
def test_checkpoint_loads_and_infers(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples, epochs=8)
    train.run(cfg)
    import torch
    model, ckpt = train.load_model(tmp_path / "run" / "train" / "model.pt")
    assert ckpt["taxonomy_version"] == "0.3.1"
    assert ckpt["flags"] and len(ckpt["flags"]) == 9
    out = model(torch.randn(1, 5, ckpt["feature_dim"]), torch.ones(1, 5))
    assert out["verdict_logits"].shape == (1, 3)


@requires_ffmpeg
def test_resume_continues_training(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples, epochs=4, early_stop_patience=99)
    s1 = train.run(cfg, resume=False)
    assert s1["epochs_run"] == 4
    # resume with more epochs -> history continues
    cfg.train.epochs = 9
    s2 = train.run(cfg, resume=True)
    import json
    log = json.loads((tmp_path / "run" / "train" / "train_log.json").read_text())
    epochs_seen = [h["epoch"] for h in log["history"]]
    assert max(epochs_seen) == 8            # continued to epoch index 8 (9 total)
    assert len(epochs_seen) == 9


@requires_ffmpeg
def test_features_cached(tmp_path, synth_samples):
    cfg = _prep(tmp_path, synth_samples, epochs=2)
    train.run(cfg)
    feat_files = list((tmp_path / "run" / "train" / "features").glob("*.npy"))
    assert len(feat_files) >= 4     # one per trainable clip (train+val)
