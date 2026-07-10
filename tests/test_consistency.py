"""Reference-consistency tests: config, reference indexing, embedding cache."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from video_screener.config import PipelineConfig, load_config
from video_screener.consistency import (
    DEFAULT_SUBJECT,
    ReferenceIndex,
    SubjectReferences,
    frame_reference_similarity,
    index_references,
    score_clip,
)
from video_screener.models.encoder import DeterministicEncoder


class CountingEncoder(DeterministicEncoder):
    """DeterministicEncoder that counts encode_paths calls (cache assertions)."""

    def __init__(self, feature_dim: int = 64):
        super().__init__(feature_dim=feature_dim)
        self.calls = 0

    def encode_paths(self, frame_paths):
        self.calls += 1
        return super().encode_paths(frame_paths)


def _write_img(path, seed: int, size: int = 24) -> None:
    rng = np.random.RandomState(seed)
    Image.fromarray(rng.randint(0, 255, (size, size, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def ref_dir(tmp_path):
    d = tmp_path / "refs"
    (d / "subject_a").mkdir(parents=True)
    (d / "subject_b").mkdir()
    _write_img(d / "subject_a" / "a1.png", seed=1)
    _write_img(d / "subject_a" / "a2.png", seed=2)
    _write_img(d / "subject_b" / "b1.png", seed=3)
    _write_img(d / "loose.png", seed=4)          # -> subject "default"
    (d / "subject_a" / "notes.txt").write_text("not an image")  # ignored
    (d / "empty_subject").mkdir()                 # no images -> not indexed
    return d


# ------------------------------- config -----------------------------------
def test_config_reference_dir_default_none():
    cfg = PipelineConfig()
    assert cfg.consistency.reference_dir is None


def test_config_reference_dir_from_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("consistency:\n  reference_dir: refs/\n")
    cfg = load_config(p)
    assert cfg.consistency.reference_dir == "refs/"


def test_config_consistency_rejects_unknown_fields(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("consistency:\n  reference_direction: oops\n")
    with pytest.raises(Exception):
        load_config(p)


# ------------------------------ indexing -----------------------------------
def test_index_references_per_subject(ref_dir, tmp_path):
    enc = CountingEncoder(feature_dim=64)
    idx = index_references(ref_dir, enc, tmp_path / "cache")
    assert isinstance(idx, ReferenceIndex)
    assert set(idx.subjects) == {"subject_a", "subject_b", DEFAULT_SUBJECT}
    assert idx.subjects["subject_a"].embeddings.shape == (2, 64)
    assert idx.subjects["subject_b"].embeddings.shape == (1, 64)
    assert idx.subjects[DEFAULT_SUBJECT].embeddings.shape == (1, 64)
    assert idx.n_images == 4 and bool(idx)
    assert idx.encoder_name == enc.name
    # non-image and empty-subject entries are excluded
    assert all(p.endswith(".png") for s in idx.subjects.values() for p in s.paths)


def test_index_references_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        index_references(tmp_path / "nope", CountingEncoder(), tmp_path / "cache")


# ------------------------------- cache -------------------------------------
def test_index_cache_hit_skips_encoder(ref_dir, tmp_path):
    cache = tmp_path / "cache"
    enc = CountingEncoder(feature_dim=64)
    first = index_references(ref_dir, enc, cache)
    assert enc.calls == 3  # one encode per subject

    enc2 = CountingEncoder(feature_dim=64)
    second = index_references(ref_dir, enc2, cache)
    assert enc2.calls == 0  # fully served from cache
    for s in first.subjects:
        assert np.allclose(first.subjects[s].embeddings, second.subjects[s].embeddings)


def test_index_cache_invalidated_by_content_change(ref_dir, tmp_path):
    cache = tmp_path / "cache"
    enc = CountingEncoder(feature_dim=64)
    index_references(ref_dir, enc, cache)

    _write_img(ref_dir / "subject_a" / "a1.png", seed=99)  # change one subject
    enc2 = CountingEncoder(feature_dim=64)
    idx = index_references(ref_dir, enc2, cache)
    assert enc2.calls == 1  # only subject_a re-encoded
    assert idx.subjects["subject_a"].embeddings.shape == (2, 64)


def test_index_cache_keyed_by_encoder_name(ref_dir, tmp_path):
    cache = tmp_path / "cache"
    enc = CountingEncoder(feature_dim=64)
    index_references(ref_dir, enc, cache)
    other = CountingEncoder(feature_dim=64)
    other.name = "other-encoder"
    index_references(ref_dir, other, cache)
    assert other.calls == 3  # different encoder -> no cross-encoder cache hits


# --------------------- clip-vs-reference scoring math -----------------------
def _index_of(**subjects) -> ReferenceIndex:
    return ReferenceIndex(encoder_name="synthetic", subjects={
        name: SubjectReferences(name, [f"{name}{i}.png" for i in range(len(emb))],
                                np.asarray(emb, dtype=np.float32))
        for name, emb in subjects.items()
    })


def test_frame_similarity_is_best_match_per_frame():
    refs = np.eye(3, 4, dtype=np.float32)          # three reference views
    frames = np.array([[1, 0, 0, 0],
                       [0, 2, 0, 0],                # scale must not matter
                       [1, 1, 0, 0]], dtype=np.float32)
    pf = frame_reference_similarity(frames, refs)
    assert np.allclose(pf, [1.0, 1.0, 1 / np.sqrt(2)], atol=1e-6)


def test_score_clip_worst_frame_aggregation_and_subject_assignment():
    idx = _index_of(a=[[1, 0, 0, 0]], b=[[0, 1, 0, 0]])
    c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)     # 60 degrees off subject a
    frames = np.array([[1, 0, 0, 0], [c, s, 0, 0]], dtype=np.float32)
    cons = score_clip(frames, idx)
    # vs a: per-frame [1.0, 0.5] -> min 0.5 ; vs b: [0.0, 0.866] -> min 0.0
    assert cons.subject == "a"
    assert abs(cons.score - 0.5) < 1e-6             # §5 worst frame, not mean
    assert np.allclose(cons.per_frame, [1.0, 0.5], atol=1e-6)
    assert abs(cons.per_subject["b"] - 0.0) < 1e-6


def test_score_clip_empty_inputs_return_none():
    idx = _index_of(a=[[1, 0, 0, 0]])
    assert score_clip(np.zeros((0, 4), dtype=np.float32), idx) is None
    assert score_clip(np.ones((2, 4), dtype=np.float32),
                      ReferenceIndex(encoder_name="x")) is None
