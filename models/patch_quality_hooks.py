"""Mid-network patch residuals: naturalness + anatomy (sample-time kwargs)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _move_eval(mod: nn.Module, device=None, dtype=None) -> nn.Module:
    kwargs: dict = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    if kwargs:
        mod = mod.to(**kwargs)
    mod.eval()
    return mod


def ensure_patch_quality_modules(
    module: nn.Module,
    hidden_size: int,
    *,
    num_heads: int = 8,
    device=None,
    dtype=None,
    with_anatomy: bool = True,
) -> None:
    """Attach naturalness + anatomy modules once (eval, on ``device``/``dtype``)."""
    from .anti_ai_naturalness import AntiAINaturalnessController

    hidden = int(hidden_size)

    ctrl = getattr(module, "_naturalness_ctrl", None)
    if ctrl is None or int(getattr(ctrl, "hidden_size", -1) or -1) != hidden:
        ctrl = AntiAINaturalnessController(hidden)
        module._naturalness_ctrl = ctrl
    module._naturalness_ctrl = _move_eval(ctrl, device, dtype)

    if not with_anatomy:
        return

    from .anatomy_attention import AnatomyAwareAttention

    heads = int(num_heads or 8)
    learned = getattr(module, "_anatomy_attn", None)
    if learned is None or int(getattr(learned, "hidden_size", -1) or -1) != hidden:
        learned = AnatomyAwareAttention(hidden, num_heads=heads)
        module._anatomy_attn = learned
    module._anatomy_attn = _move_eval(learned, device, dtype)


def apply_patch_quality_hooks(
    module: nn.Module,
    x_out: torch.Tensor,
    *,
    h_lat: int,
    w_lat: int,
    kwargs: dict,
) -> torch.Tensor:
    """Optional mid-network naturalness + anatomy residuals (sample-time kwargs)."""
    nat_s = float(kwargs.get("naturalness_strength", 0.0) or 0.0)
    anat_s = float(kwargs.get("anatomy_attention_strength", 0.0) or 0.0)
    if nat_s <= 0.0 and anat_s <= 0.0:
        return x_out

    patch = int(getattr(getattr(module, "x_embedder", None), "patch_size", [2])[0] or 2)
    h_p = max(1, int(h_lat) // patch)
    w_p = max(1, int(w_lat) // patch)
    n = int(x_out.shape[1])
    side = int(round(n**0.5))
    if side * side == n:
        h_p = w_p = side

    learned_anatomy = bool(kwargs.get("anatomy_learned", False))
    if nat_s > 0.0 or learned_anatomy:
        ensure_patch_quality_modules(
            module,
            int(x_out.shape[-1]),
            num_heads=int(getattr(module, "num_heads", 8) or 8),
            device=x_out.device,
            dtype=x_out.dtype,
            with_anatomy=learned_anatomy,
        )

    out = x_out
    if nat_s > 0.0:
        from .anti_ai_naturalness import detect_medium

        ctrl = module._naturalness_ctrl
        medium = kwargs.get("naturalness_medium")
        if medium is None:
            medium = detect_medium(str(kwargs.get("naturalness_prompt", "") or ""))
        out = ctrl(out, medium, h_p, w_p, strength=nat_s)

    if anat_s > 0.0:
        from .anatomy_attention import apply_anatomy_spatial_prior

        out = apply_anatomy_spatial_prior(out, h_p, w_p, strength=anat_s)
        if learned_anatomy:
            learned = module._anatomy_attn
            out = learned(out, anatomy_mask=kwargs.get("anatomy_mask"))
    return out


__all__ = ["apply_patch_quality_hooks", "ensure_patch_quality_modules"]
