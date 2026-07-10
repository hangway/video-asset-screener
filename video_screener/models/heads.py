"""Temporal transformer + task heads.

Architecture: frozen per-frame features -> input projection -> lightweight
temporal transformer (2-4 layers) -> per-frame contextual embeddings, consumed
by three head groups:

- ``VerdictHead``    : 3-way softmax on the mean-pooled clip embedding.
- ``CoralDimHead``   : per-frame CORAL ordinal head (rank-consistent thresholds)
                        for one dimension; K=5 levels -> 4 thresholds.
- ``FlagHead``       : per-frame independent sigmoid logits for the 9 flags.

Clip-level pooling (§5) is applied in ``model.MultiTaskScreener``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..taxonomy_schema import NUM_FLAGS, SCORE_MAX


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    def __init__(self, feature_dim: int, d_model: int = 256, n_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """x [B,T,feature_dim], key_padding_mask [B,T] (True = PAD)."""
        h = self.input_proj(x)
        h = self.pos(h)
        return self.encoder(h, src_key_padding_mask=key_padding_mask)


class VerdictHead(nn.Module):
    """Verdict head grounded in the clip embedding PLUS the (well-learned) dim
    scores and flag probabilities. This mirrors taxonomy §1 — the verdict is a
    function of the dimension gates and hard-fail flags — and avoids the head
    collapsing to the class marginal on small data."""

    def __init__(self, d_model: int, n_dims: int, n_flags: int, n_verdicts: int = 3):
        super().__init__()
        in_dim = d_model + n_dims + n_flags
        self.fc = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, n_verdicts)
        )

    def forward(self, clip_embed: torch.Tensor, dim_scores: torch.Tensor,
                flag_probs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([clip_embed, dim_scores, flag_probs], dim=-1)
        return self.fc(x)


class CoralDimHead(nn.Module):
    """Rank-consistent CORAL head for one ordinal dimension.

    A single shared slope + monotonically DECREASING thresholds guarantees
    P(y>0) >= P(y>1) >= ... so the predicted level (count of probs>0.5) is
    always rank-consistent."""

    def __init__(self, d_model: int, n_levels: int = SCORE_MAX + 1):
        super().__init__()
        self.slope = nn.Linear(d_model, 1, bias=False)
        self.b0 = nn.Parameter(torch.zeros(1))
        # non-negative gaps -> strictly decreasing thresholds
        self.gaps = nn.Parameter(torch.zeros(n_levels - 2))
        self.n_thresh = n_levels - 1

    def thresholds(self) -> torch.Tensor:
        # bias_0 = b0 ; bias_k = b0 - cumsum(softplus(gaps))
        decs = torch.cumsum(torch.nn.functional.softplus(self.gaps), dim=0)
        return torch.cat([self.b0, self.b0 - decs])  # [n_thresh]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [B,T,d_model] -> per-frame threshold logits [B,T,n_thresh]."""
        s = self.slope(frames)                       # [B,T,1]
        return s + self.thresholds().view(1, 1, -1)  # broadcast


class FlagHead(nn.Module):
    def __init__(self, d_model: int, n_flags: int = NUM_FLAGS):
        super().__init__()
        self.fc = nn.Linear(d_model, n_flags)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [B,T,d_model] -> per-frame flag logits [B,T,9]."""
        return self.fc(frames)
