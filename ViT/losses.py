from __future__ import annotations

import torch
import torch.nn.functional as F


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

    # Hinge: max(0, margin - (pred_i - pred_j))
    dif = pred_i - pred_j
    losses = F.relu(float(margin) - dif)
    return losses[mask].mean()

