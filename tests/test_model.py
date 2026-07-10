"""Stage 5 unit tests: model shapes, CORAL rank-consistency, §5 pooling, losses."""

from __future__ import annotations

import numpy as np
import torch

from video_screener.models.losses import coral_loss, flag_bce, masked_dim_loss
from video_screener.models.model import (
    MultiTaskScreener,
    _masked_max,
    _masked_mean,
    _masked_min,
)
from video_screener.taxonomy_schema import DIMENSIONS, NUM_FLAGS


def test_forward_shapes_and_ranges():
    torch.manual_seed(0)
    m = MultiTaskScreener(feature_dim=64, d_model=32, n_layers=2, n_heads=4)
    feats = torch.randn(3, 5, 64)
    mask = torch.ones(3, 5)
    out = m(feats, mask)
    assert out["verdict_logits"].shape == (3, 3)
    assert out["flag_probs"].shape == (3, NUM_FLAGS)
    assert (out["flag_probs"] >= 0).all() and (out["flag_probs"] <= 1).all()
    assert set(out["dim_thresh_probs"]) == set(DIMENSIONS)
    for d in DIMENSIONS:
        assert out["dim_thresh_probs"][d].shape == (3, 4)
        assert out["dim_scores"][d].shape == (3,)
        assert (out["dim_scores"][d] >= 0).all() and (out["dim_scores"][d] <= 4).all()


def test_coral_rank_consistency():
    """CORAL thresholds are monotonically decreasing, so clip cumulative probs
    decrease across k and the predicted level (count of probs>0.5) is
    rank-consistent."""
    torch.manual_seed(1)
    m = MultiTaskScreener(feature_dim=32, d_model=16, n_layers=2, n_heads=2)
    feats = torch.randn(4, 6, 32)
    out = m(feats, torch.ones(4, 6))
    for d in DIMENSIONS:
        probs = out["dim_thresh_probs"][d]              # [B,4]
        # non-increasing across thresholds
        diffs = probs[:, 1:] - probs[:, :-1]
        assert (diffs <= 1e-5).all(), f"{d} thresholds not monotone"
    levels = MultiTaskScreener.predict_levels(out["dim_thresh_probs"])
    for d in DIMENSIONS:
        assert (levels[d] >= 0).all() and (levels[d] <= 4).all()


def test_pooling_worst_mean_max():
    # per-frame values [B=1, T=3, C=1]; valid all True
    valid = torch.tensor([[True, True, True]])
    x = torch.tensor([[[0.2], [0.9], [0.5]]])
    assert torch.isclose(_masked_min(x, valid)[0, 0], torch.tensor(0.2))
    assert torch.isclose(_masked_max(x, valid)[0, 0], torch.tensor(0.9))
    assert torch.isclose(_masked_mean(x, valid)[0, 0], torch.tensor(0.5333), atol=1e-3)


def test_pooling_respects_mask():
    valid = torch.tensor([[True, True, False]])  # 3rd frame padded
    x = torch.tensor([[[0.2], [0.9], [0.01]]])   # padded value ignored
    assert torch.isclose(_masked_min(x, valid)[0, 0], torch.tensor(0.2))  # not 0.01
    assert torch.isclose(_masked_mean(x, valid)[0, 0], torch.tensor(0.55))


def test_worst_frame_dims_use_min():
    """temporal_stability/motion_quality use worst-frame (min) pooling: a single
    bad frame drives the clip score down."""
    torch.manual_seed(2)
    m = MultiTaskScreener(feature_dim=16, d_model=16, n_layers=1, n_heads=2)
    # one frame with very different features
    feats = torch.zeros(1, 4, 16)
    feats[0, 2] = 5.0
    out = m(feats, torch.ones(1, 4))
    # temporal_stability clip prob equals the elementwise min across frames
    m.eval()
    with torch.no_grad():
        frames = m.temporal(feats, torch.zeros(1, 4, dtype=torch.bool))
        pf = torch.sigmoid(m.dim_heads["temporal_stability"](frames))  # [1,4,4]
        expected = pf.min(dim=1).values
        got = m(feats, torch.ones(1, 4))["dim_thresh_probs"]["temporal_stability"]
    assert torch.allclose(got, expected, atol=1e-5)


# ------------------------------ losses ------------------------------------
def test_coral_loss_masks_na():
    probs = torch.tensor([[0.9, 0.8, 0.3, 0.1], [0.5, 0.5, 0.5, 0.5]])
    # second row masked (target -1) -> loss only from first row
    masked = coral_loss(probs, torch.tensor([2, -1]))
    only_first = coral_loss(probs[:1], torch.tensor([2]))
    assert torch.isclose(masked, only_first, atol=1e-6)


def test_coral_loss_all_masked_is_zero_grad():
    probs = torch.rand(3, 4, requires_grad=True)
    loss = coral_loss(probs, torch.tensor([-1, -1, -1]))
    assert float(loss) == 0.0
    loss.backward()  # should not error


def test_masked_dim_loss_ignores_none_dims():
    probs = {d: torch.rand(2, 4) for d in DIMENSIONS}
    tgt = {d: torch.tensor([-1, -1]) for d in DIMENSIONS}  # all N/A
    assert float(masked_dim_loss(probs, tgt)) == 0.0


def test_flag_bce_pos_weight_penalizes_misses_more():
    # a missed positive (pred 0.1 when target 1) with pos_weight>1 costs more
    # than a false alarm (pred 0.9 when target 0) of equal probability gap.
    miss = flag_bce(torch.tensor([[0.1]]), torch.tensor([[1.0]]), pos_weight=4.0)
    false_alarm = flag_bce(torch.tensor([[0.9]]), torch.tensor([[0.0]]), pos_weight=4.0)
    assert float(miss) > float(false_alarm)


def test_verdict_grounded_in_dims_flags():
    """The verdict head input includes dim scores + flag probs (grounding)."""
    m = MultiTaskScreener(feature_dim=16, d_model=16, n_layers=1, n_heads=2)
    # in_features of first verdict layer = d_model + n_dims + n_flags
    first = m.verdict_head.fc[0]
    assert first.in_features == 16 + len(DIMENSIONS) + NUM_FLAGS
