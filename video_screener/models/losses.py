"""Multi-task losses.

- verdict: 3-way cross-entropy (softmax).
- 6 dimensions: CORAL ordinal loss (rank-consistent), N/A (null) dims MASKED
  out of the loss.
- 9 hard-fail flags: independent BCE with the positive class UP-WEIGHTED
  (missing a hard-fail costs more than a false alarm).

Dimension + flag losses operate on clip-level *probabilities* (already pooled
across frames per taxonomy §5), so BCE-on-probabilities is used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-6


def verdict_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """3-way softmax cross-entropy. logits [B,3], target [B] in {0,1,2}."""
    return F.cross_entropy(logits, target)


def coral_loss(
    clip_thresh_probs: torch.Tensor,   # [B, K-1] cumulative P(y>k), in (0,1)
    target_level: torch.Tensor,        # [B] long in 0..K-1, or -1 = N/A (masked)
) -> torch.Tensor:
    """CORAL ordinal loss over clip-level threshold probabilities.

    For a K-level ordinal (K=5, thresholds K-1=4), the binary target for
    threshold k is 1 iff level > k. N/A labels (target_level < 0) are masked."""
    b, km1 = clip_thresh_probs.shape
    valid = target_level >= 0
    if valid.sum() == 0:
        return clip_thresh_probs.sum() * 0.0  # keep graph, zero contribution
    # Numerical stability: BCE must stay on POOLED PROBABILITIES, not logits.
    # §5 mean-frame pooling averages per-frame sigmoids, and sigmoid does not
    # commute with the mean, so no clip-level logit exists whose sigmoid equals
    # the pooled prob (min/max pooling would commute, but only 2 of 6 dims use
    # min). The clamp bounds the loss at -ln(_EPS) ~= 13.8 and, because clamp
    # passes zero gradient outside its range, saturated pooled probs stop
    # contributing gradient instead of exploding. See notes.md
    # "CORAL-on-logits assessment" for the full deferral rationale.
    probs = clip_thresh_probs[valid].clamp(_EPS, 1 - _EPS)
    lvl = target_level[valid].unsqueeze(1)             # [n,1]
    ks = torch.arange(km1, device=probs.device).unsqueeze(0)  # [1,K-1]
    targets = (lvl > ks).float()                       # [n,K-1]
    loss = F.binary_cross_entropy(probs, targets, reduction="mean")
    return loss


def masked_dim_loss(
    dim_thresh_probs: dict[str, torch.Tensor],
    dim_targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Sum CORAL loss across all dimensions (each masking its own N/A rows)."""
    total = None
    for dim, probs in dim_thresh_probs.items():
        l = coral_loss(probs, dim_targets[dim])
        total = l if total is None else total + l
    n = max(len(dim_thresh_probs), 1)
    return (total / n) if total is not None else torch.tensor(0.0)


def flag_bce(
    clip_flag_probs: torch.Tensor,     # [B,9] in (0,1)
    targets: torch.Tensor,             # [B,9] in {0,1}
    pos_weight: float = 4.0,
) -> torch.Tensor:
    """Independent BCE per flag with the positive class up-weighted."""
    p = clip_flag_probs.clamp(_EPS, 1 - _EPS)
    loss = -(pos_weight * targets * torch.log(p) + (1 - targets) * torch.log(1 - p))
    return loss.mean()


def combined_loss(
    outputs: dict,
    labels: dict,
    w_verdict: float = 1.0,
    w_dims: float = 1.0,
    w_flags: float = 1.0,
    flag_pos_weight: float = 4.0,
) -> tuple[torch.Tensor, dict]:
    lv = verdict_ce(outputs["verdict_logits"], labels["verdict"])
    ld = masked_dim_loss(outputs["dim_thresh_probs"], labels["dim_levels"])
    lf = flag_bce(outputs["flag_probs"], labels["flags"], flag_pos_weight)
    total = w_verdict * lv + w_dims * ld + w_flags * lf
    return total, {
        "loss": float(total.detach()),
        "verdict": float(lv.detach()),
        "dims": float(ld.detach()),
        "flags": float(lf.detach()),
    }
