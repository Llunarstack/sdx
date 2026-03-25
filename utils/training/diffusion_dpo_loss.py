"""
Surrogate **DPO-style** loss for diffusion preference optimization (reference only).

Full Diffusion-DPO (e.g. Wallace et al.) runs two forward passes per pair (win/lose) with
shared noise and timestep. This module exposes the **scalar** Bradley–Terry / DPO term so you
can plug in ``-MSE`` or similar as a proxy log-likelihood **after** you compute per-image losses.

Stage-2 trainer: ``scripts/tools/training/train_diffusion_dpo.py`` (pairwise images + T5 prompts).
``PreferencePair`` rows: ``utils/training/preference_jsonl.py``; on-disk tensors:
``utils/training/preference_image_dataset.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_preference_loss(
    implicit_logp_win: torch.Tensor,
    implicit_logp_lose: torch.Tensor,
    implicit_logp_ref_win: torch.Tensor,
    implicit_logp_ref_lose: torch.Tensor,
    *,
    beta: float = 5000.0,
) -> torch.Tensor:
    """
    Standard DPO loss on implicit log-probabilities (higher = more preferred).

    Typically ``implicit_logp_* = -mse`` or ``-0.5 * mse / sigma^2`` from a denoising step.
    Reference terms are usually ``.detach()`` so only the policy (student) is trained.
    """
    b = float(beta)
    pi = b * ((implicit_logp_win - implicit_logp_lose) - (implicit_logp_ref_win - implicit_logp_ref_lose))
    return -F.logsigmoid(pi).mean()


def neg_mse_as_logp(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scalar surrogate: negative spatial MSE (mean over batch)."""
    return -F.mse_loss(pred, target)


__all__ = ["dpo_preference_loss", "neg_mse_as_logp"]
