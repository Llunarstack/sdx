"""
DiT text-model profiling: parameter counts, weight size estimates, variant listing.

Used by ``scripts/tools/dit_variant_compare.py`` and tests. Does not load checkpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch.nn as nn
from models import DiT_models_text
from models.enhanced_dit import EnhancedDiT_models


def latent_side_from_image_size(image_size: int) -> int:
    """Spatial latent grid side (SDX uses VAE 8× downsample)."""
    return max(1, int(image_size) // 8)


def default_dit_profile_kwargs(
    *,
    image_size: int = 256,
    text_dim: int = 4096,
    learn_sigma: bool = True,
) -> Dict[str, Any]:
    """
    Minimal kwargs aligned with ``get_dit_build_kwargs`` defaults (no REPA, no MoE, no extras).
    Pass to ``DiT_models_text[name](**kwargs)`` or Enhanced constructors.
    """
    latent = latent_side_from_image_size(image_size)
    return {
        "input_size": latent,
        "text_dim": int(text_dim),
        "learn_sigma": bool(learn_sigma),
        "class_dropout_prob": 0.1,
        "num_ar_blocks": 0,
        "use_xformers": True,
        "style_embed_dim": 0,
        "control_cond_dim": 0,
        "creativity_embed_dim": 0,
        "size_embed_dim": 0,
        "patch_se": False,
        "patch_se_reduction": 8,
        "repa_out_dim": 0,
        "repa_projector_hidden_dim": 0,
        "ssm_every_n": 0,
        "ssm_kernel_size": 7,
        "num_register_tokens": 0,
        "use_rope": False,
        "rope_base": 10000.0,
        "kv_merge_factor": 1,
        "token_routing_enabled": False,
        "token_routing_strength": 1.0,
        "moe_num_experts": 0,
        "moe_top_k": 2,
    }


def dit_parameter_report(model: nn.Module) -> Dict[str, Any]:
    """Totals and bf16/fp32 on-disk size estimates for full state dict tensors."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "size_fp32_gib": total * 4 / (1024**3),
        "size_bf16_gib": total * 2 / (1024**3),
    }


def instantiate_dit_text(
    model_name: str,
    *,
    image_size: int = 256,
    text_dim: int = 4096,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Build a registered DiT by name (text DiT, Predecessor, Supreme, or EnhancedDiT)."""
    if model_name not in DiT_models_text:
        raise KeyError(f"Unknown model_name {model_name!r}. Options: {sorted(DiT_models_text)}")
    if model_name in EnhancedDiT_models:
        return instantiate_enhanced_dit(model_name, image_size=image_size, extra_kwargs=extra_kwargs)

    kw = default_dit_profile_kwargs(image_size=image_size, text_dim=text_dim)
    if extra_kwargs:
        kw.update(extra_kwargs)
    return DiT_models_text[model_name](**kw)


def default_enhanced_dit_profile_kwargs(*, image_size: int = 256) -> Dict[str, Any]:
    """Kwargs for ``EnhancedDiT`` factories (class-label conditioning, not T5 cross-attn)."""
    return {
        "input_size": latent_side_from_image_size(image_size),
        "learn_sigma": True,
        "class_dropout_prob": 0.1,
        "enable_spatial_control": True,
        "enable_anatomy_awareness": True,
        "enable_text_rendering": True,
        "enable_consistency": True,
    }


def instantiate_enhanced_dit(
    model_name: str,
    *,
    image_size: int = 256,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    if model_name not in EnhancedDiT_models:
        raise KeyError(f"Unknown EnhancedDiT {model_name!r}. Options: {sorted(EnhancedDiT_models)}")
    kw = default_enhanced_dit_profile_kwargs(image_size=image_size)
    if extra_kwargs:
        kw.update(extra_kwargs)
    return EnhancedDiT_models[model_name](**kw)


def list_dit_text_variant_names() -> List[str]:
    """Text-conditioned DiT + Predecessor + Supreme (excludes EnhancedDiT class-label models)."""
    return sorted(k for k in DiT_models_text.keys() if k not in EnhancedDiT_models)


def list_enhanced_dit_variant_names() -> List[str]:
    return sorted(EnhancedDiT_models.keys())


def list_all_dit_registry_names() -> List[str]:
    """Every name in ``DiT_models_text`` (includes EnhancedDiT-*)."""
    return sorted(DiT_models_text.keys())
