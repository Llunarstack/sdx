from __future__ import annotations

import torch


def attention_entropy(attn_probs: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """
    Entropy over token dimension from attention probabilities.
    Input: (B, H, N, L) attention probabilities.
    Output: (B,) mean entropy across heads and patches.
    """
    if attn_probs.dim() != 4:
        raise ValueError("attn_probs must be shaped (B, H, N, L)")
    p = attn_probs.to(dtype=torch.float32).clamp(min=eps)
    ent = -(p * p.log()).sum(dim=-1)  # (B,H,N)
    return ent.mean(dim=(1, 2))


def adaptive_cfg_from_attention(
    base_cfg: float,
    attn_probs: torch.Tensor,
    *,
    min_ratio: float = 0.75,
    max_ratio: float = 1.2,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Map attention entropy to per-sample CFG:
    - high entropy (uncertain attention): increase CFG
    - low entropy (focused attention): decrease CFG
    """
    ent = attention_entropy(attn_probs)
    z = (ent - ent.mean()) / (ent.std().clamp(min=1e-6))
    z = z / max(float(temperature), 1e-6)
    gate = torch.sigmoid(z)  # 0..1
    ratio = float(min_ratio) + (float(max_ratio) - float(min_ratio)) * gate
    return ratio * float(base_cfg)


def fuse_condition_scales(
    *,
    base_control_scale: float,
    base_adapter_scale: float,
    progress: float,
    frontload_control: bool = True,
) -> tuple[float, float]:
    """
    Produce (control_scale, adapter_scale) at a given normalized step progress.
    """
    p = max(0.0, min(1.0, float(progress)))
    if frontload_control:
        c_mul = 1.2 - 0.35 * p
    else:
        c_mul = 0.9 + 0.2 * p
    a_mul = 0.9 + 0.25 * p
    return float(base_control_scale) * c_mul, float(base_adapter_scale) * a_mul

