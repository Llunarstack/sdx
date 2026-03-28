from __future__ import annotations

import torch


def attention_token_coverage(attn_probs: torch.Tensor) -> torch.Tensor:
    """
    Attention token coverage score using max patch activation per token.
    Input: (B, H, N, L); Output: (B, L)
    """
    if attn_probs.dim() != 4:
        raise ValueError("attn_probs must be shaped (B, H, N, L)")
    a = attn_probs.mean(dim=1)  # (B,N,L)
    return a.max(dim=1).values  # (B,L)


def weighted_patch_alignment_score(
    attn_probs: torch.Tensor,
    token_weights: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Weighted global alignment score in [0,1]-ish range for monitoring/adaptive control.
    """
    cov = attention_token_coverage(attn_probs)  # (B,L)
    tw = token_weights.to(device=cov.device, dtype=cov.dtype)
    if tw.shape != cov.shape:
        raise ValueError(f"token_weights shape {tuple(tw.shape)} must match coverage shape {tuple(cov.shape)}")
    num = (cov * tw).sum(dim=1)
    den = tw.sum(dim=1).clamp(min=eps)
    return (num / den).clamp(min=0.0)


def coverage_shortfall_loss(
    attn_probs: torch.Tensor,
    *,
    target: float = 0.03,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Penalize token coverage below a target.
    """
    cov = attention_token_coverage(attn_probs)
    per = torch.relu(float(target) - cov)
    if token_weights is not None:
        tw = token_weights.to(device=per.device, dtype=per.dtype)
        if tw.shape != per.shape:
            raise ValueError("token_weights shape must match (B, L)")
        den = tw.sum(dim=1).clamp(min=1e-8)
        return ((per * tw).sum(dim=1) / den).mean()
    return per.mean()

