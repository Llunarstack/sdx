"""
Part-aware / grounding utilities for training (object–text alignment, captions, foveation).

These map common research directions (segmentation-assisted training, hierarchical captions,
cross-attention vs region alignment, multi-scale crops) into **small, testable building blocks**.
Full SAM/VLM pipelines are out of scope here; wire your own preprocessors to produce masks and
JSONL fields that this module consumes.

Typical JSONL extensions::

    "grounding_mask": "relative/or/absolute/path/to/mask.png"   # grayscale or RGB; white = foreground
    "caption_global": "A man sits on a chair."
    "caption_local": "Chair: dark oak; Man: blue denim jacket."

See ``foreground_attention_alignment_loss`` and ``merge_hierarchical_captions``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hierarchical / part-level captions (training-time dropout)
# ---------------------------------------------------------------------------


def merge_hierarchical_captions(
    base_caption: str,
    *,
    caption_global: Optional[str] = None,
    caption_local: Optional[str] = None,
    entity_captions: Optional[Dict[str, str]] = None,
    separator: str = " | ",
) -> str:
    """
    Flatten optional global/local/entity captions into one T5 string.
    Empty parts are skipped.
    """
    parts: List[str] = []
    if caption_global and caption_global.strip():
        parts.append(caption_global.strip())
    if base_caption and base_caption.strip():
        parts.append(base_caption.strip())
    if caption_local and caption_local.strip():
        parts.append(caption_local.strip())
    if entity_captions:
        for k, v in entity_captions.items():
            vv = (v or "").strip()
            if vv:
                parts.append(f"{k}: {vv}")
    return separator.join(parts) if parts else (base_caption or "")


def hierarchical_caption_dropout(
    merged: str,
    *,
    p_drop_global: float = 0.0,
    p_drop_local: float = 0.0,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Randomly drop segments separated by `` | `` (as produced by ``merge_hierarchical_captions``)
    to force the model to work with partial supervision.
    """
    if p_drop_global <= 0 and p_drop_local <= 0:
        return merged
    r = rng or random
    segs = [s.strip() for s in merged.split("|") if s.strip()]
    if len(segs) <= 1:
        return merged
    # Heuristic: first segment = global, last = local if 3+ parts; middle = base.
    keep: List[str] = []
    for i, s in enumerate(segs):
        if i == 0 and p_drop_global > 0 and r.random() < p_drop_global:
            continue
        if i == len(segs) - 1 and len(segs) >= 2 and p_drop_local > 0 and r.random() < p_drop_local:
            continue
        keep.append(s)
    return " | ".join(keep) if keep else merged


def merge_hierarchical_jsonl_record(sample: Dict[str, Any], separator: str = " | ") -> str:
    """Read optional keys from a manifest row and return a merged caption string."""
    base = (sample.get("caption") or "").strip()
    cg = sample.get("caption_global")
    cl = sample.get("caption_local")
    ent = sample.get("entity_captions")
    if isinstance(ent, dict):
        ec = {str(k): str(v) for k, v in ent.items()}
    else:
        ec = None
    return merge_hierarchical_captions(
        base,
        caption_global=cg if isinstance(cg, str) else None,
        caption_local=cl if isinstance(cl, str) else None,
        entity_captions=ec,
        separator=separator,
    )


# ---------------------------------------------------------------------------
# Masks on patch grid (align with DiT patch tokens)
# ---------------------------------------------------------------------------


def image_mask_to_patch_weights(
    mask_b1hw: torch.Tensor,
    *,
    num_patches_h: int,
    num_patches_w: int,
) -> torch.Tensor:
    """
    Downsample a binary/soft mask ``(B, 1, H, W)`` to ``(B, N)`` where
    ``N = num_patches_h * num_patches_w`` using average pooling.

    Uses the native C++ kernel (``sdx_mask_ops``) when built — avoids
    PyTorch autograd overhead for this pure data-prep operation.
    Falls back to ``F.adaptive_avg_pool2d`` otherwise.

    Args:
        mask_b1hw: Float mask tensor of shape ``(B, 1, H, W)``.
        num_patches_h: Number of patch rows.
        num_patches_w: Number of patch columns.

    Returns:
        Patch weight tensor of shape ``(B, num_patches_h * num_patches_w)``.
    """
    if mask_b1hw.dim() != 4 or mask_b1hw.shape[1] != 1:
        raise ValueError("mask_b1hw must be (B, 1, H, W)")

    patch_rows, patch_cols = int(num_patches_h), int(num_patches_w)
    _, _, height, width = mask_b1hw.shape

    # Try native C++ kernel (no autograd, no Python loop).
    if height % patch_rows == 0 and width % patch_cols == 0:
        try:
            from sdx_native.mask_ops_native import maybe_mask_to_patch_weights_native

            mask_np = mask_b1hw.detach().float().cpu().numpy()
            result = maybe_mask_to_patch_weights_native(mask_np, patch_rows, patch_cols)
            if result is not None:
                return torch.from_numpy(result).to(
                    device=mask_b1hw.device, dtype=mask_b1hw.dtype
                )
        except Exception:
            pass

    # PyTorch fallback.
    mask_float = mask_b1hw.to(dtype=torch.float32)
    pooled_mask = F.adaptive_avg_pool2d(mask_float, (patch_rows, patch_cols))
    return pooled_mask.flatten(1)


def foreground_attention_alignment_loss(
    attn_weights: torch.Tensor,
    patch_foreground_b_n: torch.Tensor,
    *,
    token_start: int = 0,
    token_end: int = 0,
    sample_valid: Optional[torch.Tensor] = None,
    min_fg_patch_mass: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Encourage cross-attention mass on patches inside ``patch_foreground_b_n`` (B, N).

    ``attn_weights``: (B, num_heads, N, L) as returned by ``DiT_Text`` block with ``return_attn=True``.
    Pools heads; optionally restricts text positions to ``[token_start, token_end)`` (end 0 = all).

    Loss: per-sample 1 - cosine similarity between normalized patch attention and normalized mask.

    ``sample_valid``: optional (B,) bool — mean only over True rows (e.g. mixed batches with padded masks).
    ``min_fg_patch_mass``: if > 0, rows with ``patch_foreground_b_n.sum(dim=1) < min_fg_patch_mass`` are excluded
    (empty masks would otherwise produce a degenerate target).
    """
    if attn_weights.dim() != 4:
        raise ValueError("attn_weights must be (B, heads, N, L)")
    batch_size, _, patch_count, token_count = attn_weights.shape
    if patch_foreground_b_n.shape[0] != batch_size or patch_foreground_b_n.shape[1] != patch_count:
        raise ValueError(f"mask patch count {patch_foreground_b_n.shape} != attn N={patch_count}")

    token_end_idx = token_count if token_end <= 0 else min(token_end, token_count)
    token_start_idx = max(0, min(token_start, token_end_idx))
    attention_patch_mean = attn_weights[:, :, :, token_start_idx:token_end_idx].mean(dim=(1, 3))  # (B, N)
    attention_patch_mean = attention_patch_mean.to(dtype=torch.float32)
    patch_foreground = patch_foreground_b_n.to(device=attention_patch_mean.device, dtype=attention_patch_mean.dtype).clamp(
        0.0, 1.0
    )

    row_ok = torch.ones(batch_size, device=attention_patch_mean.device, dtype=torch.bool)
    if sample_valid is not None:
        sample_valid_mask = sample_valid.to(device=attention_patch_mean.device)
        if sample_valid_mask.shape != (batch_size,):
            raise ValueError(f"sample_valid must be (B,), got {sample_valid_mask.shape}")
        row_ok = row_ok & sample_valid_mask
    if min_fg_patch_mass and float(min_fg_patch_mass) > 0:
        row_ok = row_ok & (patch_foreground.sum(dim=1) >= float(min_fg_patch_mass))

    attention_norm = attention_patch_mean / (attention_patch_mean.sum(dim=1, keepdim=True) + eps)
    foreground_norm = patch_foreground / (patch_foreground.sum(dim=1, keepdim=True) + eps)
    cosine_similarity = (attention_norm * foreground_norm).sum(dim=1)
    per_sample_loss = 1.0 - cosine_similarity
    if row_ok.any():
        return per_sample_loss[row_ok].mean()
    # No valid rows: zero loss but keep grad path through attention (e.g. DDP / compile).
    return attn_weights.mean() * 0.0


def token_coverage_from_cross_attention(attn_weights: torch.Tensor, *, token_start: int = 0, token_end: int = 0) -> torch.Tensor:
    """
    Return per-sample per-token coverage scores from cross-attention maps.

    ``attn_weights`` shape: (B, heads, N_patches, L_tokens).
    Coverage uses ``max`` over patches after averaging heads, inspired by token "excitation" style methods.
    """
    if attn_weights.dim() != 4:
        raise ValueError("attn_weights must be (B, heads, N, L)")
    _, _, _, token_count = attn_weights.shape
    token_end_idx = token_count if token_end <= 0 else min(token_end, token_count)
    token_start_idx = max(0, min(token_start, token_end_idx))
    attention_mean = attn_weights[:, :, :, token_start_idx:token_end_idx].mean(dim=1)  # (B, N, Lt)
    token_coverage = attention_mean.max(dim=1).values  # (B, Lt)
    return token_coverage


def token_coverage_loss(
    attn_weights: torch.Tensor,
    *,
    token_start: int = 0,
    token_end: int = 0,
    target_coverage: float = 0.025,
    sample_valid: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalize neglected prompt tokens by pushing max patch attention per token above ``target_coverage``.

    Loss = mean(relu(target_coverage - coverage_token)), optionally weighted by token_weights.
    """
    token_coverage = token_coverage_from_cross_attention(attn_weights, token_start=token_start, token_end=token_end)  # (B, Lt)
    per_token_loss = torch.relu(float(target_coverage) - token_coverage)
    if token_weights is not None:
        token_weight_values = token_weights.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
        token_end_idx = token_weight_values.shape[1] if token_end <= 0 else min(token_end, token_weight_values.shape[1])
        token_start_idx = max(0, min(token_start, token_end_idx))
        token_weight_values = token_weight_values[:, token_start_idx:token_end_idx]
        if token_weight_values.shape == per_token_loss.shape:
            per_token_loss = per_token_loss * token_weight_values
    per_sample_loss = per_token_loss.mean(dim=1)
    if sample_valid is not None:
        sample_valid_mask = sample_valid.to(device=per_sample_loss.device)
        if sample_valid_mask.shape == per_sample_loss.shape and sample_valid_mask.any():
            return per_sample_loss[sample_valid_mask].mean()
        if sample_valid_mask.shape == per_sample_loss.shape:
            return attn_weights.mean() * 0.0
    return per_sample_loss.mean()


# ---------------------------------------------------------------------------
# Foveated / multi-scale crop box (data augmentation)
# ---------------------------------------------------------------------------


def foveated_random_crop_box(
    height: int,
    width: int,
    *,
    crop_frac: float,
    rng: Optional[random.Random] = None,
) -> Tuple[int, int, int, int]:
    """
    Return (y0, x0, y1, x1) for a random crop covering ``crop_frac`` of the shorter side
    (then clamped to image bounds). Use to bias training toward random subregions (foveated views).
    """
    r = rng or random
    frac = max(0.1, min(1.0, float(crop_frac)))
    ch = max(1, int(round(height * frac)))
    cw = max(1, int(round(width * frac)))
    ch = min(ch, height)
    cw = min(cw, width)
    y0 = r.randint(0, max(0, height - ch))
    x0 = r.randint(0, max(0, width - cw))
    return y0, x0, y0 + ch, x0 + cw


@dataclass
class PartAwareCaptionConfig:
    """Optional toggles for dataset / training (mirrors TrainConfig subset)."""

    use_hierarchical_merge: bool = False
    hierarchical_separator: str = " | "
    hierarchical_drop_global_p: float = 0.0
    hierarchical_drop_local_p: float = 0.0


def apply_part_aware_caption_pipeline(
    raw_caption: str,
    sample: Dict[str, Any],
    cfg: PartAwareCaptionConfig,
    rng: Optional[random.Random] = None,
) -> str:
    """Merge + optional dropout for one manifest row."""
    if not cfg.use_hierarchical_merge:
        return raw_caption
    merged = merge_hierarchical_jsonl_record(sample, separator=cfg.hierarchical_separator)
    if not merged.strip():
        merged = raw_caption
    return hierarchical_caption_dropout(
        merged,
        p_drop_global=cfg.hierarchical_drop_global_p,
        p_drop_local=cfg.hierarchical_drop_local_p,
        rng=rng,
    )


def dit_patch_grid_hw(num_patches: int) -> Tuple[int, int]:
    """Square grid side for DiT patch tokens (``num_patches = h * w``)."""
    n = int(num_patches)
    s = int(round(n**0.5))
    if s * s != n:
        raise ValueError(f"num_patches={n} is not a perfect square (non-square latent grids unsupported here)")
    return s, s


def compute_dit_attn_grounding_loss(
    *,
    train_model: torch.nn.Module,
    diffusion,
    latents_bchw: torch.Tensor,
    t: torch.Tensor,
    model_kwargs: Dict[str, Any],
    grounding_mask_b1hw: torch.Tensor,
    training_noise: torch.Tensor,
    noise_offset: float = 0.0,
    token_start: int = 0,
    token_end: int = 0,
    sample_valid: Optional[torch.Tensor] = None,
    min_fg_patch_mass: float = 0.0,
) -> torch.Tensor:
    """
    Auxiliary loss: align **first DiT block** cross-attention (``return_attn=True`` in ``DiT_Text``)
    with a foreground mask downsampled to the patch grid.

    **Limitations:** Only ``DiT_Text`` (and clones with the same API) is supported; attention is taken
    from block 0 only. Incompatible with grad checkpointing during this forward — checkpointing is
    temporarily disabled on the model while ``return_attn`` runs.
    """
    if not hasattr(train_model, "num_patches"):
        raise TypeError("train_model must expose num_patches (e.g. DiT_Text)")
    attn_w = capture_dit_block0_cross_attn(
        train_model=train_model,
        diffusion=diffusion,
        latents_bchw=latents_bchw,
        t=t,
        model_kwargs=model_kwargs,
        training_noise=training_noise,
        noise_offset=noise_offset,
    )
    return grounding_loss_from_attn(
        attn_w=attn_w,
        train_model=train_model,
        grounding_mask_b1hw=grounding_mask_b1hw,
        token_start=token_start,
        token_end=token_end,
        sample_valid=sample_valid,
        min_fg_patch_mass=min_fg_patch_mass,
    )


def capture_dit_block0_cross_attn(
    *,
    train_model: torch.nn.Module,
    diffusion,
    latents_bchw: torch.Tensor,
    t: torch.Tensor,
    model_kwargs: Dict[str, Any],
    training_noise: torch.Tensor,
    noise_offset: float = 0.0,
) -> torch.Tensor:
    """Run one forward pass at x_t and return block-0 cross-attention (B, heads, N, L).

    Note: grad checkpointing is temporarily disabled for this pass because
    ``return_attn=True`` requires the full forward graph to be materialised.
    The returned attention tensor is detached — gradients flow through the
    main training loss, not through this auxiliary capture.
    """
    mk = dict(model_kwargs)
    mk.pop("return_attn", None)
    was_ckpt = bool(getattr(train_model, "_grad_checkpointing", False))
    if was_ckpt:
        train_model._grad_checkpointing = False
    try:
        with torch.no_grad():
            x_t = diffusion.q_sample(latents_bchw, t, noise=training_noise, noise_offset=noise_offset)
        out = train_model(x_t, t, return_attn=True, **mk)
    finally:
        if was_ckpt:
            train_model._grad_checkpointing = True
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError("Expected model(x,t,return_attn=True) -> (tensor, attn_weights)")
    _, attn_w = out
    return attn_w


def grounding_loss_from_attn(
    *,
    attn_w: torch.Tensor,
    train_model: torch.nn.Module,
    grounding_mask_b1hw: torch.Tensor,
    token_start: int = 0,
    token_end: int = 0,
    sample_valid: Optional[torch.Tensor] = None,
    min_fg_patch_mass: float = 0.0,
) -> torch.Tensor:
    """Compute grounding loss from pre-captured attention."""
    npatch = int(getattr(train_model, "num_patches", 0))
    if attn_w.shape[2] > npatch:
        attn_w = attn_w[:, :, :npatch, :]
    ph, pw = dit_patch_grid_hw(npatch)
    patch_fg = image_mask_to_patch_weights(grounding_mask_b1hw, num_patches_h=ph, num_patches_w=pw)
    return foreground_attention_alignment_loss(
        attn_w,
        patch_fg,
        token_start=token_start,
        token_end=token_end,
        sample_valid=sample_valid,
        min_fg_patch_mass=min_fg_patch_mass,
    )


def token_coverage_loss_from_attn(
    *,
    attn_w: torch.Tensor,
    token_start: int = 0,
    token_end: int = 0,
    target_coverage: float = 0.025,
    sample_valid: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute token coverage loss from pre-captured block-0 attention."""
    return token_coverage_loss(
        attn_w,
        token_start=token_start,
        token_end=token_end,
        target_coverage=target_coverage,
        sample_valid=sample_valid,
        token_weights=token_weights,
    )
