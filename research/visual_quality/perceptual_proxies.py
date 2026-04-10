"""
No-reference proxies on RGB tensors (``[0, 1]``, ``NCHW``).

Use for **A/B ranking** of generations or ablations — not as a ground-truth “real vs fake” test.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def parse_rgb01_bchw(x: Tensor) -> Tensor:
    """Validate ``(B, 3, H, W)`` in roughly ``[0, 1]`` and return float tensor."""
    if x.dim() != 4 or int(x.shape[1]) != 3:
        raise ValueError("expected RGB tensor (B, 3, H, W)")
    return x.detach().float().clamp(0.0, 1.0)


def _luminance(rgb: Tensor) -> Tensor:
    """Rec. 601 luma, ``(B,1,H,W)``."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def laplacian_sharpness(rgb01: Tensor) -> Tensor:
    """
    Variance of a discrete Laplacian on luma — higher usually means more fine detail / less blur.

    Returns ``(B,)`` float32 on the same device as ``rgb01``.
    """
    x = parse_rgb01_bchw(rgb01)
    y = _luminance(x)
    kernel = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]],
        dtype=y.dtype,
        device=y.device,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(y, kernel, padding=1)
    return lap.view(lap.shape[0], -1).var(dim=1)


def colorfulness_std(rgb01: Tensor) -> Tensor:
    """
    Per-image std of RGB channels (mean over spatial dims). Useful to flag greywash or neon blow-out.

    Returns ``(B,)``.
    """
    x = parse_rgb01_bchw(rgb01)
    # per-channel spatial std, then mean across channels
    b = x.shape[0]
    flat = x.view(b, 3, -1)
    std = flat.std(dim=2)
    return std.mean(dim=1)


def exposure_naturalness(rgb01: Tensor) -> Tensor:
    """
    Score in ``(0, 1]``: 1 when mean luma is near mid-grey; falls off when image is very dark/bright.

    Returns ``(B,)``.
    """
    x = parse_rgb01_bchw(rgb01)
    y = _luminance(x)
    mu = y.view(y.shape[0], -1).mean(dim=1).clamp(0.0, 1.0)
    # Triangle peak at 0.5
    d = (mu - 0.5).abs() * 2.0
    return (1.0 - d).clamp(0.0, 1.0)


def _normalize01(t: Tensor, eps: float = 1e-8) -> Tensor:
    lo = t.min()
    hi = t.max()
    span = hi - lo
    if float(span.item()) < eps:
        return torch.ones_like(t)
    return (t - lo) / span


def combined_quality_proxy(
    rgb01: Tensor,
    *,
    w_sharp: float = 1.0,
    w_color: float = 0.35,
    w_exp: float = 0.5,
) -> Tensor:
    """
    Single scalar per batch row for **relative** ranking (higher = fewer obvious issues on these axes).

    For ``B > 1``, sharpness and colorfulness are min–max normalized **across the batch** so you
    can pick the best of several candidates in one forward. For ``B == 1``, use a fixed-scale
    blend (``log1p(laplacian)``) so ``best_of_n`` across separate tensors still compares sensibly.
    """
    s = laplacian_sharpness(rgb01)
    c = colorfulness_std(rgb01)
    e = exposure_naturalness(rgb01)
    if int(s.shape[0]) <= 1:
        return w_sharp * torch.log1p(s.clamp(min=0.0)) + w_color * c + w_exp * e
    s_n = _normalize01(s)
    c_n = _normalize01(c)
    return w_sharp * s_n + w_color * c_n + w_exp * e
