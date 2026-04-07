"""Unit tests for diffusion preference DPO surrogate loss."""

from __future__ import annotations

import torch
from utils.training.diffusion_dpo_loss import dpo_preference_loss


def test_dpo_logit_clip_changes_loss_when_logits_would_be_large() -> None:
    win_p = torch.tensor(10.0)
    lose_p = torch.tensor(0.0)
    win_r = torch.tensor(5.0)
    lose_r = torch.tensor(5.0)
    unclamped = dpo_preference_loss(win_p, lose_p, win_r, lose_r, beta=500.0, logit_clip=None)
    clamped = dpo_preference_loss(win_p, lose_p, win_r, lose_r, beta=500.0, logit_clip=1.0)
    assert torch.isfinite(unclamped)
    assert torch.isfinite(clamped)
    assert abs(float(unclamped.detach()) - float(clamped.detach())) > 1e-6


def test_dpo_loss_grad_with_clip() -> None:
    win_p = torch.tensor(2.0, requires_grad=True)
    lose_p = torch.tensor(1.0, requires_grad=True)
    win_r = torch.tensor(1.5).detach()
    lose_r = torch.tensor(1.2).detach()
    loss = dpo_preference_loss(win_p, lose_p, win_r, lose_r, beta=100.0, logit_clip=5.0)
    loss.backward()
    assert win_p.grad is not None and torch.isfinite(win_p.grad)
