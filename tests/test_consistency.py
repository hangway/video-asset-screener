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
    assert cfg.consistency.vimax_workdir is None


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


def test_config_vimax_workdir_and_thresholds(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "consistency:\n"
        "  vimax_workdir: .working_dir/session/script2video\n"
        "  min_keyframe_similarity: 0.6\n"
        "  min_same_camera_boundary_similarity: 0.4\n"
    )
    cfg = load_config(p)
    assert cfg.consistency.vimax_workdir.endswith("script2video")
    assert cfg.consistency.min_keyframe_similarity == pytest.approx(0.6)
    assert cfg.consistency.min_same_camera_boundary_similarity == pytest.approx(0.4)


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


def test_score_clip_expected_subjects_constrain_assignment():
    idx = _index_of(
        expected=[[0.6, 0.8, 0.0, 0.0]],
        intruder=[[0.0, 1.0, 0.0, 0.0]],
    )
    frames = np.array([[0.0, 1.0, 0.0, 0.0]] * 2, dtype=np.float32)

    unconstrained = score_clip(frames, idx)
    constrained = score_clip(frames, idx, expected_subjects=("expected",))

    assert unconstrained.subject == "intruder"
    assert constrained.subject == "expected"
    assert constrained.best_subject == "intruder"
    assert abs(constrained.score - 0.8) < 1e-6


def test_score_clip_pools_multiple_expected_subjects_per_frame():
    idx = _index_of(alice=[[1, 0, 0, 0]], bob=[[0, 1, 0, 0]])
    frames = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    cons = score_clip(frames, idx, expected_subjects=("alice", "bob"))

    assert cons.subject == "alice + bob"
    assert cons.expected_subjects == ("alice", "bob")
    assert np.allclose(cons.per_frame, [1.0, 1.0])
    assert cons.score == pytest.approx(1.0)


def test_keyframe_similarity_compares_actual_endpoints():
    from video_screener.consistency import score_keyframes

    frames = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    result = score_keyframes(
        frames,
        first_frame_embedding=np.array([1, 0, 0], dtype=np.float32),
        last_frame_embedding=np.array([0, 1, 0], dtype=np.float32),
    )
    assert result.head_similarity == pytest.approx(1.0)
    assert result.tail_similarity == pytest.approx(0.0)
    assert result.score == pytest.approx(0.0)


# ----------------------------- within-clip drift ----------------------------
def test_frame_drift_consecutive_cosine_distance():
    from video_screener.consistency import frame_drift

    e0, e1 = [1, 0, 0, 0], [0, 1, 0, 0]
    frames = np.array([e0, e0, e1, e1], dtype=np.float32)
    d = frame_drift(frames)
    # steady, orthogonal jump (distance 1), steady
    assert np.allclose(d, [0.0, 1.0, 0.0], atol=1e-6)
    assert d.argmax() == 1  # the jump is between frames 1 and 2


def test_frame_drift_scale_invariant_and_degenerate():
    from video_screener.consistency import frame_drift

    scaled = np.array([[1, 0, 0, 0], [5, 0, 0, 0]], dtype=np.float32)
    assert np.allclose(frame_drift(scaled), [0.0], atol=1e-6)
    assert frame_drift(np.ones((1, 4), dtype=np.float32)).size == 0
    assert frame_drift(np.zeros((0, 4), dtype=np.float32)).size == 0


# --------------------------- edge stability math ----------------------------
def test_edge_outlier_head_by_similarity_tail_by_drift():
    from video_screener.consistency import edge_stability

    # 10 frames at 1 fps, duration 9s -> head = frame 0, tail = frame 9
    times = [float(i) for i in range(10)]
    sim = [0.2] + [0.9] * 9                # head similarity collapses
    drift = [0.01] * 8 + [0.5]             # tail pair (8 -> 9) jumps
    es = edge_stability(sim, drift, times, duration=9.0,
                        window_sec=1.0, sigma=3.0)
    assert es["head"]["outlier"] is True   # sim outlier (low vs body)
    assert es["tail"]["outlier"] is True   # drift outlier (high vs body)
    actions = {t["action"]: t["suggested_trim_sec"] for t in es["trim_suggestions"]}
    # cut up to the first body frame / from the last body frame
    assert actions == {"trim_head": 1.0, "trim_tail": 1.0}
    assert es["body"]["n_frames"] == 8


def test_edge_stability_stable_clip_has_no_outliers():
    from video_screener.consistency import edge_stability

    times = [float(i) for i in range(10)]
    es = edge_stability([0.9] * 10, [0.01] * 9, times, duration=9.0)
    assert es["head"]["outlier"] is False and es["tail"]["outlier"] is False
    assert es["trim_suggestions"] == []


def test_edge_stability_within_sigma_is_not_an_outlier():
    from video_screener.consistency import edge_stability

    # body sim noisy (std ~0.05); head dip of one std must NOT trigger at 3σ
    body = [0.85, 0.95, 0.85, 0.95, 0.85, 0.95, 0.85, 0.95]
    times = [float(i) for i in range(10)]
    es = edge_stability([0.85] + body + [0.9], [0.01] * 9, times, duration=9.0,
                        sigma=3.0)
    assert es["head"]["outlier"] is False


def test_tail_window_anchored_to_last_sample_not_duration():
    from video_screener.consistency import edge_stability

    # Real 8 s clip at 1 fps: compute_sample_times() emits t < duration, so
    # the last sample is 7.0 — a duration-anchored tail (t > 8-1) was empty
    # for EVERY integer-length clip (dogfood: tail.n_frames == 0 on all
    # three 8 s Veo clips). The tail must anchor at the last sample instead.
    times = [float(i) for i in range(8)]  # 0..7, exactly what sampling produces
    es = edge_stability([0.9] * 8, [0.01] * 7, times, duration=8.0,
                        window_sec=1.0, sigma=3.0)
    assert es["head"]["n_frames"] == 1   # frame 0.0
    assert es["tail"]["n_frames"] == 1   # frame 7.0 — was 0 before the fix
    assert es["body"]["n_frames"] == 6   # frames 1.0..6.0


def test_tail_outlier_fires_with_realistic_sample_times():
    from video_screener.consistency import edge_stability

    # Dogfood shape: 8 s Veo clip, uniform 1 fps + scene frames at 0.5/4.5.
    times = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0]
    sim = [0.9] * 9 + [0.2]                # tail similarity collapses
    drift = [0.01] * 8 + [0.5]             # tail pair (6.0 -> 7.0) jumps
    es = edge_stability(sim, drift, times, duration=8.0,
                        window_sec=1.0, sigma=3.0)
    assert es["head"]["n_frames"] == 2 and es["tail"]["n_frames"] == 1
    assert es["tail"]["outlier"] is True
    actions = {t["action"]: t["suggested_trim_sec"] for t in es["trim_suggestions"]}
    # cut from the last body frame (6.0) to the clip end (duration 8.0)
    assert actions == {"trim_tail": 2.0}


def test_edge_stability_degenerate_inputs_return_none():
    from video_screener.consistency import edge_stability

    # unknown frame times -> not analyzable
    assert edge_stability([0.9, 0.9], [0.0], [0.0, None], duration=2.0) is None
    # clip shorter than ~2 windows -> no body baseline
    assert edge_stability([0.9, 0.9], [0.0], [0.0, 1.4], duration=1.5) is None
    assert edge_stability([], [], [], duration=1.0) is None
