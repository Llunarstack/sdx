"""Per-step training schedules and objectives: block-AR and timestep curricula, OT noise pairing, part-aware captions and grounding losses."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

_VALID_ORDERS = ("raster", "zorder", "snake", "spiral")


def normalize_ar_blocks(v: int) -> int:
    """Snap arbitrary int to supported AR regime {0,2,4}."""
    try:
        blocks = int(v)
    except Exception:
        return 0
    if blocks <= 0:
        return 0
    if blocks <= 2:
        return 2
    return 4


def parse_ar_order_mix(spec: str | None) -> list[str]:
    """
    Parse comma-separated AR orders (e.g. ``"raster,zorder,snake"``).

    Invalid entries are ignored; duplicates are removed while preserving order.
    """
    if not spec or not str(spec).strip():
        return []
    out: list[str] = []
    seen = set()
    for raw_order in str(spec).split(","):
        order = raw_order.strip().lower()
        if not order or order not in _VALID_ORDERS or order in seen:
            continue
        seen.add(order)
        out.append(order)
    return out


def choose_ar_order_for_step(step: int, *, base_order: str, mix_orders: Sequence[str]) -> str:
    """Deterministic step-indexed AR traversal order."""
    if not mix_orders:
        fallback_order = str(base_order or "raster").strip().lower()
        return fallback_order if fallback_order in _VALID_ORDERS else "raster"
    order_index = max(0, int(step)) % len(mix_orders)
    return str(mix_orders[order_index])


def num_ar_blocks_for_step(
    step: int,
    *,
    warmup_steps: int,
    target_blocks: int,
    start_blocks: int = 0,
) -> int:
    """
    Piecewise constant curriculum: ``start_blocks`` until ``warmup_steps``, then ``target_blocks``.

    Typical: ``start_blocks=0``, ``target_blocks=2``, ``warmup_steps=10_000``.
    """
    if step < int(warmup_steps):
        return int(start_blocks)
    return int(target_blocks)


def num_ar_blocks_linear_ramp(
    step: int,
    *,
    ramp_start: int,
    ramp_end: int,
    start_blocks: int = 0,
    end_blocks: int = 2,
) -> int:
    """Linearly interpolate between start_blocks and end_blocks over [ramp_start, ramp_end]."""
    ramp_start_step, ramp_end_step = int(ramp_start), int(ramp_end)
    if ramp_end_step <= ramp_start_step:
        return int(end_blocks)
    if step <= ramp_start_step:
        return int(start_blocks)
    if step >= ramp_end_step:
        return int(end_blocks)
    ramp_fraction = (float(step) - float(ramp_start_step)) / float(ramp_end_step - ramp_start_step)
    # Snap to valid discrete values {0,2,4} when close
    interpolated_blocks = (1.0 - ramp_fraction) * float(start_blocks) + ramp_fraction * float(end_blocks)
    if interpolated_blocks < 1.0:
        return 0
    if interpolated_blocks < 3.0:
        return 2
    return 4


def resolve_ar_for_step(
    step: int,
    *,
    base_blocks: int,
    base_order: str,
    curriculum_mode: str = "none",
    warmup_steps: int = 0,
    ramp_start: int = 0,
    ramp_end: int = 0,
    curriculum_start_blocks: int = -1,
    curriculum_target_blocks: int = -1,
    order_mix: str | None = None,
) -> tuple[int, str]:
    """
    Resolve runtime ``(num_ar_blocks, ar_block_order)`` for a training step.

    curriculum_mode:
      - ``none``: fixed blocks (from base)
      - ``step``: piecewise constant at warmup boundary
      - ``linear``: linear ramp over [ramp_start, ramp_end]
    """
    mode = str(curriculum_mode or "none").strip().lower()
    base_blocks_normalized = normalize_ar_blocks(base_blocks)
    start_blocks_normalized = normalize_ar_blocks(curriculum_start_blocks if int(curriculum_start_blocks) >= 0 else 0)
    target_blocks_normalized = normalize_ar_blocks(
        curriculum_target_blocks if int(curriculum_target_blocks) >= 0 else base_blocks_normalized
    )
    step_index = max(0, int(step))
    if mode == "step":
        resolved_blocks = normalize_ar_blocks(
            num_ar_blocks_for_step(
                step_index,
                warmup_steps=max(0, int(warmup_steps)),
                target_blocks=target_blocks_normalized,
                start_blocks=start_blocks_normalized,
            )
        )
    elif mode == "linear":
        ramp_start_step = max(0, int(ramp_start))
        ramp_end_step = (
            max(ramp_start_step + 1, int(ramp_end))
            if int(ramp_end) > 0
            else max(ramp_start_step + 1, ramp_start_step + max(1, int(warmup_steps)))
        )
        resolved_blocks = normalize_ar_blocks(
            num_ar_blocks_linear_ramp(
                step_index,
                ramp_start=ramp_start_step,
                ramp_end=ramp_end_step,
                start_blocks=start_blocks_normalized,
                end_blocks=target_blocks_normalized,
            )
        )
    else:
        resolved_blocks = base_blocks_normalized

    if resolved_blocks <= 0:
        return 0, "raster"
    order_mix_list = parse_ar_order_mix(order_mix)
    order = choose_ar_order_for_step(step_index, base_order=base_order, mix_orders=order_mix_list)
    return resolved_blocks, order


_CACHE: dict[str, list[TimestepCurriculumPhase]] = {}


@dataclass(slots=True, frozen=True)
class TimestepCurriculumPhase:
    start_step: int
    mode: str
    logit_mean: float | None = None
    logit_std: float | None = None


def parse_timestep_curriculum_schedule(schedule: str) -> list[TimestepCurriculumPhase]:
    """Parse a non-empty curriculum string; raises ``ValueError`` on bad input."""
    text = str(schedule or "").strip()
    if not text:
        return []
    if text in _CACHE:
        return list(_CACHE[text])

    phases: list[TimestepCurriculumPhase] = []
    for raw_seg in text.split("|"):
        seg = raw_seg.strip()
        if not seg:
            continue
        parts = [p.strip() for p in seg.split(":")]
        if len(parts) < 2:
            raise ValueError(f"curriculum segment must be start:mode[:mean:std], got {seg!r}")
        start_step = int(parts[0])
        mode = parts[1].strip().lower().replace("-", "_")
        lm: float | None = None
        ls: float | None = None
        if len(parts) >= 4:
            lm = float(parts[2])
            ls = float(parts[3])
        elif len(parts) == 3:
            raise ValueError(f"curriculum segment {seg!r}: provide both mean and std after mode, or neither")
        phases.append(TimestepCurriculumPhase(start_step=start_step, mode=mode, logit_mean=lm, logit_std=ls))

    if not phases:
        return []
    phases.sort(key=lambda p: p.start_step)
    _CACHE[text] = phases
    return list(phases)


def resolve_timestep_kwargs_for_step(cfg: Any, global_step: int) -> dict[str, Any]:
    """
    Resolve kwargs for ``diffusion.timestep_sampling.sample_training_timesteps``.

    When ``timestep_curriculum_schedule`` is set, picks the active phase by ``global_step``;
    otherwise uses ``timestep_sample_mode`` / logit fields on ``cfg``.
    """
    schedule = str(getattr(cfg, "timestep_curriculum_schedule", "") or "").strip()
    base_mode = str(getattr(cfg, "timestep_sample_mode", "uniform"))
    base_mean = float(getattr(cfg, "timestep_logit_mean", 0.0))
    base_std = float(getattr(cfg, "timestep_logit_std", 1.0))

    if not schedule:
        return {"mode": base_mode, "logit_mean": base_mean, "logit_std": base_std}

    phases = parse_timestep_curriculum_schedule(schedule)
    if not phases:
        return {"mode": base_mode, "logit_mean": base_mean, "logit_std": base_std}

    step = int(global_step)
    active = phases[0]
    for p in phases:
        if p.start_step <= step:
            active = p
        else:
            break

    mean = active.logit_mean if active.logit_mean is not None else base_mean
    std = active.logit_std if active.logit_std is not None else base_std
    return {"mode": active.mode, "logit_mean": float(mean), "logit_std": float(std)}


PairMode = Literal["soft", "hungarian"]


def sinkhorn_log_plan(cost: torch.Tensor, *, reg: float, n_iters: int) -> torch.Tensor:
    """Log of the entropic OT plan with uniform marginals (rows and columns each sum to ``1/B``).

    Log-domain updates (Peyre & Cuturi 2019, Sec. 4.4) so ``cost / reg`` in the thousands does not underflow.
    """
    b = cost.shape[0]
    log_k = -cost.double() / max(float(reg), 1e-12)
    log_marginal = torch.full((b,), -math.log(b), dtype=torch.float64, device=cost.device)
    f = torch.zeros_like(log_marginal)
    g = torch.zeros_like(log_marginal)
    for _ in range(max(1, int(n_iters))):
        f = log_marginal - torch.logsumexp(log_k + g[None, :], dim=1)
        g = log_marginal - torch.logsumexp(log_k + f[:, None], dim=0)
    return log_k + f[:, None] + g[None, :]


def sample_permutation_from_plan(log_plan: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Column index per row, each column used once: rows in random order draw from their plan row over free columns."""
    b = log_plan.shape[0]
    free = torch.ones(b, dtype=torch.bool, device=log_plan.device)
    perm = torch.empty(b, dtype=torch.long, device=log_plan.device)
    for i in torch.randperm(b, generator=generator).tolist():
        logits = log_plan[i].masked_fill(~free, float("-inf"))
        j = int(torch.multinomial(torch.softmax(logits, dim=0).float().cpu(), 1, generator=generator))
        perm[i] = j
        free[j] = False
    return perm


def pair_noise_to_latents(
    latents: torch.Tensor,
    noise: torch.Tensor,
    *,
    reg: float = 0.05,
    n_iters: int = 40,
    mode: PairMode = "soft",
) -> torch.Tensor:
    """Return ``noise`` with its batch rows permuted so row ``i`` is paired with ``latents[i]`` by (entropic) OT."""
    if latents.shape != noise.shape:
        raise ValueError("latents and noise must have the same shape")
    b = latents.shape[0]
    if b < 2 or reg <= 0.0:
        return noise
    cost = torch.cdist(latents.reshape(b, -1).float(), noise.reshape(b, -1).float()) ** 2
    if mode == "soft":
        # ``reg`` is in units of the cost spread (std), so its meaning does not depend on latent size.
        cost = (cost - cost.mean()) / cost.std().clamp(min=1e-12)
        perm = sample_permutation_from_plan(sinkhorn_log_plan(cost, reg=reg, n_iters=n_iters))
        return noise[perm]
    if mode == "hungarian":
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        perm = torch.empty(b, dtype=torch.long, device=noise.device)
        perm[torch.as_tensor(rows, device=noise.device)] = torch.as_tensor(cols, device=noise.device)
        return noise[perm]
    raise ValueError(f"Unknown mode {mode!r}")


# Hierarchical / part-level captions (training-time dropout)


def merge_hierarchical_captions(
    base_caption: str,
    *,
    caption_global: str | None = None,
    caption_local: str | None = None,
    entity_captions: dict[str, str] | None = None,
    separator: str = " | ",
) -> str:
    """
    Flatten optional global/local/entity captions into one T5 string.
    Empty parts are skipped.
    """
    parts: list[str] = []
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
    rng: random.Random | None = None,
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
    keep: list[str] = []
    for i, s in enumerate(segs):
        if i == 0 and p_drop_global > 0 and r.random() < p_drop_global:
            continue
        if i == len(segs) - 1 and len(segs) >= 2 and p_drop_local > 0 and r.random() < p_drop_local:
            continue
        keep.append(s)
    return " | ".join(keep) if keep else merged


def merge_hierarchical_jsonl_record(sample: dict[str, Any], separator: str = " | ") -> str:
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


# Masks on patch grid (align with DiT patch tokens)


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
            from sdx_native.ffi import maybe_mask_to_patch_weights_native

            mask_np = mask_b1hw.detach().float().cpu().numpy()
            result = maybe_mask_to_patch_weights_native(mask_np, patch_rows, patch_cols)
            if result is not None:
                return torch.from_numpy(result).to(device=mask_b1hw.device, dtype=mask_b1hw.dtype)
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
    sample_valid: torch.Tensor | None = None,
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
    patch_foreground = patch_foreground_b_n.to(
        device=attention_patch_mean.device, dtype=attention_patch_mean.dtype
    ).clamp(0.0, 1.0)

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


def token_coverage_from_cross_attention(
    attn_weights: torch.Tensor, *, token_start: int = 0, token_end: int = 0
) -> torch.Tensor:
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
    sample_valid: torch.Tensor | None = None,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Penalize neglected prompt tokens by pushing max patch attention per token above ``target_coverage``.

    Loss = mean(relu(target_coverage - coverage_token)), optionally weighted by token_weights.
    """
    token_coverage = token_coverage_from_cross_attention(
        attn_weights, token_start=token_start, token_end=token_end
    )  # (B, Lt)
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


# Foveated / multi-scale crop box (data augmentation)


def foveated_random_crop_box(
    height: int,
    width: int,
    *,
    crop_frac: float,
    rng: random.Random | None = None,
) -> tuple[int, int, int, int]:
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


@dataclass(slots=True)
class PartAwareCaptionConfig:
    """Optional toggles for dataset / training (mirrors TrainConfig subset)."""

    use_hierarchical_merge: bool = False
    hierarchical_separator: str = " | "
    hierarchical_drop_global_p: float = 0.0
    hierarchical_drop_local_p: float = 0.0


def apply_part_aware_caption_pipeline(
    raw_caption: str,
    sample: dict[str, Any],
    cfg: PartAwareCaptionConfig,
    rng: random.Random | None = None,
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


def dit_patch_grid_hw(num_patches: int) -> tuple[int, int]:
    """Square grid side for DiT patch tokens (``num_patches = h * w``)."""
    n = int(num_patches)
    s = int(round(n**0.5))
    if s * s != n:
        raise ValueError(f"num_patches={n} is not a perfect square (non-square latent grids unsupported here)")
    return s, s


def capture_dit_block0_cross_attn(
    *,
    train_model: torch.nn.Module,
    diffusion,
    latents_bchw: torch.Tensor,
    t: torch.Tensor,
    model_kwargs: dict[str, Any],
    training_noise: torch.Tensor,
) -> torch.Tensor:
    """Run one forward pass at ``x_t = q_sample(latents, t, training_noise)`` and return block-0 cross-attention
    ``(B, heads, N, L)``. ``training_noise`` must be the exact noise of the main loss (noise offset included).

    Grad checkpointing is disabled for this pass because ``return_attn=True`` needs the full graph; gradients of
    the auxiliary loss flow through the returned attention.
    """
    mk = dict(model_kwargs)
    mk.pop("return_attn", None)
    was_ckpt = bool(getattr(train_model, "_grad_checkpointing", False))
    if was_ckpt:
        train_model._grad_checkpointing = False
    try:
        with torch.no_grad():
            x_t = diffusion.q_sample(latents_bchw, t, noise=training_noise)
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
    sample_valid: torch.Tensor | None = None,
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
    sample_valid: torch.Tensor | None = None,
    token_weights: torch.Tensor | None = None,
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
