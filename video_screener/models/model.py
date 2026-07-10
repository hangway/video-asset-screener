"""Assembled multi-task screener: transformer + heads + §5 clip pooling.

forward() returns clip-level outputs ready for both loss and inference:
  - verdict_logits [B,3]
  - dim_thresh_probs {dim: [B,4]}  (clip-level CORAL cumulative probs)
  - dim_scores      {dim: [B]}     (expected 0-4 score = sum of thresh probs)
  - flag_probs      [B,9]          (max-pooled per §5 rule 1)

Clip pooling (§5): temporal_stability & motion_quality use worst-frame (min of
per-frame cumulative probs); other dims use mean; flags use max (any frame
triggering triggers the clip).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..taxonomy_schema import (
    DIMENSIONS,
    NUM_DIMENSIONS,
    NUM_FLAGS,
    WORST_FRAME_DIMENSIONS,
)
from .heads import CoralDimHead, FlagHead, TemporalTransformer, VerdictHead


def _masked_mean(x: torch.Tensor, valid: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Mean over valid frames. x [B,T,...], valid [B,T]."""
    v = valid.unsqueeze(-1).float()
    s = (x * v).sum(dim=dim)
    n = v.sum(dim=dim).clamp(min=1.0)
    return s / n


def _masked_min(x: torch.Tensor, valid: torch.Tensor, dim: int = 1) -> torch.Tensor:
    big = x.masked_fill(~valid.unsqueeze(-1), float("inf"))
    return big.min(dim=dim).values


def _masked_max(x: torch.Tensor, valid: torch.Tensor, dim: int = 1) -> torch.Tensor:
    small = x.masked_fill(~valid.unsqueeze(-1), float("-inf"))
    return small.max(dim=dim).values


class MultiTaskScreener(nn.Module):
    def __init__(self, feature_dim: int, d_model: int = 256, n_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.temporal = TemporalTransformer(
            feature_dim, d_model, n_layers, n_heads, dropout
        )
        self.verdict_head = VerdictHead(d_model, NUM_DIMENSIONS, NUM_FLAGS)
        self.dim_heads = nn.ModuleDict({d: CoralDimHead(d_model) for d in DIMENSIONS})
        self.flag_head = FlagHead(d_model, NUM_FLAGS)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> dict:
        """features [B,T,feature_dim], mask [B,T] (1 = valid frame)."""
        valid = mask.bool()
        # Guard: a fully-padded row would break attention (softmax over all
        # -inf -> NaN) and make _masked_min/_masked_max emit +/-inf into the
        # verdict head. Treat frame 0 as valid so every row has >=1 valid
        # frame; the row's output is meaningless but stays finite.
        all_pad = ~valid.any(dim=1)
        if all_pad.any():
            valid = valid.clone()
            valid[all_pad, 0] = True
        pad_mask = ~valid                                    # True = PAD
        frames = self.temporal(features, key_padding_mask=pad_mask)  # [B,T,d]
        clip_embed = _masked_mean(frames, valid)             # [B,d]

        dim_thresh_probs: dict[str, torch.Tensor] = {}
        dim_scores: dict[str, torch.Tensor] = {}
        for d, head in self.dim_heads.items():
            per_frame_logits = head(frames)                  # [B,T,4]
            per_frame_probs = torch.sigmoid(per_frame_logits)
            if d in WORST_FRAME_DIMENSIONS:
                clip_probs = _masked_min(per_frame_probs, valid)   # worst frame
            else:
                clip_probs = _masked_mean(per_frame_probs, valid)  # mean frame
            dim_thresh_probs[d] = clip_probs                 # [B,4]
            dim_scores[d] = clip_probs.sum(dim=-1)           # expected 0-4

        flag_logits = self.flag_head(frames)                 # [B,T,9]
        flag_probs = _masked_max(torch.sigmoid(flag_logits), valid)  # [B,9]

        # verdict is grounded in the dim scores + flag probs (§1)
        dim_score_vec = torch.stack([dim_scores[d] for d in DIMENSIONS], dim=-1)  # [B,6]
        verdict_logits = self.verdict_head(clip_embed, dim_score_vec, flag_probs)  # [B,3]

        return {
            "verdict_logits": verdict_logits,
            "dim_thresh_probs": dim_thresh_probs,
            "dim_scores": dim_scores,
            "flag_probs": flag_probs,
        }

    @staticmethod
    def predict_levels(dim_thresh_probs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Rank-consistent ordinal prediction: level = #(P(y>k) > 0.5)."""
        return {d: (p > 0.5).sum(dim=-1) for d, p in dim_thresh_probs.items()}
