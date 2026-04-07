from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    """
    Focal loss for imbalanced binary quality labels (down-weights easy examples).
    """
    gamma = float(gamma)
    if gamma <= 0:
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    with torch.no_grad():
        prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    focal = (1.0 - p_t.clamp(min=1e-6, max=1.0 - 1e-6)) ** gamma * bce
    if alpha is not None:
        a = float(alpha)
        w = a * targets + (1.0 - a) * (1.0 - targets)
        focal = focal * w
    return focal.mean()


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.15,
    min_target_gap: float = 0.05,
) -> torch.Tensor:
    """
    Pairwise ranking for quality/adherence ordering.
    If target[i] > target[j] by at least min_target_gap, enforce pred[i] > pred[j] + margin.
    """
    b = int(pred.shape[0])
    if b < 2:
        return pred.new_zeros(())

    pred_i = pred.unsqueeze(1)
    pred_j = pred.unsqueeze(0)
    target_i = target.unsqueeze(1)
    target_j = target.unsqueeze(0)
    gap = target_i - target_j
    mask = gap > float(min_target_gap)
    if not mask.any():
        return pred.new_zeros(())

    dif = pred_i - pred_j
    losses = F.relu(float(margin) - dif)
    return losses[mask].mean()

