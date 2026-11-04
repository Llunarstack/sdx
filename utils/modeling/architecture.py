"""DiT architecture helpers: variant profiling, block-AR conditioning for the ViT scorer, training-free block caches and the 2026 architecture theme map."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def latent_side_from_image_size(image_size: int) -> int:
    """Spatial latent grid side (SDX uses VAE 8× downsample)."""
    return max(1, int(image_size) // 8)


def default_dit_profile_kwargs(
    *,
    image_size: int = 256,
    text_dim: int = 4096,
    learn_sigma: bool = True,
) -> dict[str, Any]:
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


def dit_parameter_report(model: nn.Module) -> dict[str, Any]:
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
    extra_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Build a registered DiT by name (text DiT, Predecessor, Supreme, or EnhancedDiT)."""
    from models import DiT_models_text
    from models.dit_variants import EnhancedDiT_models

    if model_name not in DiT_models_text:
        raise KeyError(f"Unknown model_name {model_name!r}. Options: {sorted(DiT_models_text)}")
    if model_name in EnhancedDiT_models:
        return instantiate_enhanced_dit(model_name, image_size=image_size, extra_kwargs=extra_kwargs)

    kw = default_dit_profile_kwargs(image_size=image_size, text_dim=text_dim)
    if extra_kwargs:
        kw.update(extra_kwargs)
    return DiT_models_text[model_name](**kw)


def default_enhanced_dit_profile_kwargs(*, image_size: int = 256) -> dict[str, Any]:
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
    extra_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    from models.dit_variants import EnhancedDiT_models

    if model_name not in EnhancedDiT_models:
        raise KeyError(f"Unknown EnhancedDiT {model_name!r}. Options: {sorted(EnhancedDiT_models)}")
    kw = default_enhanced_dit_profile_kwargs(image_size=image_size)
    if extra_kwargs:
        kw.update(extra_kwargs)
    return EnhancedDiT_models[model_name](**kw)


def list_all_dit_registry_names() -> list[str]:
    """Every name in ``DiT_models_text`` (includes EnhancedDiT-*)."""
    from models import DiT_models_text

    return sorted(DiT_models_text.keys())


# One-hot layout: [full (0), ar_2x2 (2), ar_4x4 (4), unknown]
AR_COND_DIM: int = 4

_VALID_AR: frozenset[int] = frozenset({0, 2, 4})

# Human-readable names for logs / manifests / UIs
AR_REGIME_NAMES: dict[int, str] = {
    0: "full_bidirectional",
    2: "block_ar_2x2",
    4: "block_ar_4x4",
    -1: "unknown",
}

# Order for flat JSONL key scan (first hit wins)
_AR_FLAT_KEYS: tuple[str, ...] = (
    "num_ar_blocks",
    "dit_num_ar_blocks",
    "ar_blocks",
    "generator_num_ar_blocks",
)

_NESTED_KEYS: tuple[str, ...] = ("dit_config", "train_config", "checkpoint_config", "model_config")


def normalize_num_ar_blocks(value: Any) -> int:
    """
    Return ``0``, ``2``, ``4``, or ``-1`` (unknown / invalid).
    Accepts int-like strings (e.g. JSON ``"2"``).
    """
    if value is None:
        return -1
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return -1
    try:
        v = int(value)
    except (TypeError, ValueError):
        return -1
    if v in _VALID_AR:
        return v
    return -1


def ar_regime_label(num_ar_blocks: int) -> str:
    """Stable slug: ``full_bidirectional``, ``block_ar_2x2``, ``block_ar_4x4``, ``unknown``."""
    return AR_REGIME_NAMES.get(int(num_ar_blocks), AR_REGIME_NAMES[-1])


def parse_num_ar_blocks_from_row(row: Mapping[str, Any], *, max_nested_depth: int = 2) -> int:
    """
    Parse AR regime from a manifest or metadata dict.

    Checks flat keys first, then optional nested config blobs (depth-limited).
    """
    return _parse_num_ar_blocks_from_row_depth(row, depth=0, max_depth=max_nested_depth)


def _parse_num_ar_blocks_from_row_depth(row: Mapping[str, Any], *, depth: int, max_depth: int) -> int:
    for key in _AR_FLAT_KEYS:
        if key in row:
            return normalize_num_ar_blocks(row.get(key))
    if depth >= max_depth:
        return -1
    for nk in _NESTED_KEYS:
        sub = row.get(nk)
        if isinstance(sub, Mapping):
            v = _parse_num_ar_blocks_from_row_depth(sub, depth=depth + 1, max_depth=max_depth)
            if v != -1:
                return v
    return -1


def _extract_num_ar_from_config_obj(cfg: Any) -> int:
    if cfg is None:
        return -1
    if isinstance(cfg, dict):
        for k in _AR_FLAT_KEYS:
            if k in cfg:
                return normalize_num_ar_blocks(cfg[k])
        return -1
    for k in _AR_FLAT_KEYS:
        if hasattr(cfg, k):
            return normalize_num_ar_blocks(getattr(cfg, k, None))
    return -1


def num_ar_blocks_from_checkpoint_dict(ckpt: Mapping[str, Any]) -> int:
    """
    Read ``num_ar_blocks`` from a loaded SDX-style checkpoint dict (``torch.load``).

    Looks at ``ckpt["config"]`` (dict or dataclass-like). Returns ``-1`` if missing.
    """
    return _extract_num_ar_from_config_obj(ckpt.get("config"))


def read_num_ar_blocks_from_checkpoint(path: str | Path) -> int:
    """
    Load checkpoint from disk (CPU) and return ``num_ar_blocks`` for tagging manifests / ViT defaults.

    Does not instantiate DiT - safe for large checkpoints.
    """
    p = Path(path)
    if not p.is_file():
        return -1
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return -1
    if not isinstance(ckpt, dict):
        return -1
    v = num_ar_blocks_from_checkpoint_dict(ckpt)
    if v != -1:
        return v
    # Some exports store flat
    return normalize_num_ar_blocks(ckpt.get("num_ar_blocks"))


def tag_manifest_row_ar(
    row: Mapping[str, Any],
    num_ar_blocks: int,
    *,
    overwrite: bool = False,
    primary_key: str = "num_ar_blocks",
) -> dict[str, Any]:
    """
    Copy *row* and set ``num_ar_blocks`` (and mirror ``dit_num_ar_blocks``) for ViT / digest tools.

    If *overwrite* is False, leaves existing valid AR fields unchanged.
    Unknown *num_ar_blocks* (not in 0,2,4) -> return unchanged copy.
    """
    out = dict(row)
    if num_ar_blocks not in _VALID_AR:
        return out
    current = parse_num_ar_blocks_from_row(out)
    if not overwrite and current != -1:
        return out
    v = int(num_ar_blocks)
    out[primary_key] = v
    out["dit_num_ar_blocks"] = v
    out["ar_regime"] = ar_regime_label(v)
    return out


def ar_conditioning_vector(num_ar_blocks: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Single-row vector ``(AR_COND_DIM,)``: one-hot for DiT AR regime + unknown bucket.
    """
    v = torch.zeros(AR_COND_DIM, device=device, dtype=dtype)
    if num_ar_blocks == 0:
        v[0] = 1.0
    elif num_ar_blocks == 2:
        v[1] = 1.0
    elif num_ar_blocks == 4:
        v[2] = 1.0
    else:
        v[3] = 1.0
    return v


def batch_ar_conditioning(
    values: Sequence[int],
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """``(B, AR_COND_DIM)`` from normalized ``num_ar_blocks`` (use ``-1`` for unknown)."""
    rows = [ar_conditioning_vector(int(v), device=device, dtype=dtype) for v in values]
    return torch.stack(rows, dim=0)


def default_unknown_ar_batch(batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
    """All-unknown regime (index 3)."""
    u = torch.zeros(batch_size, AR_COND_DIM, device=device, dtype=dtype)
    u[:, 3] = 1.0
    return u


def ar_bridge_summary_lines() -> str:
    """Help text for CLIs (DiT <-> ViT)."""
    return (
        "DiT num_ar_blocks: 0=full bidirectional (default Meta DiT-style), 2=2x2 block AR, 4=4x4 block AR. "
        "ViT uses a 4-D one-hot (full / ar2 / ar4 / unknown) fused with caption stats. "
        "Tag JSONL with num_ar_blocks or pass --default-num-ar-blocks on ViT infer."
    )


# ==================================================================================================
# block_cache: **Block-wise DiT cache** (BWCache / TeaCache-inspired, training-free).
#
# Caches per-block **residuals** across denoising steps. When the step fingerprint
# (timestep + latent stats) changes little, reuses cached block deltas instead of
# re-running transformer blocks — typically 1.3–2× fewer block FLOPs at modest thresholds.
#
# See ``docs/RESEARCH.md``.
# ==================================================================================================


@dataclass(slots=True)
class BlockCacheConfig:
    rel_l1_threshold: float = 0.18
    """Accumulated relative L1 on step fingerprint before full recompute."""
    max_reuse_streak: int = 3
    recompute_every: int = 4
    """Force full block pass every N denoising steps (anti drift)."""
    enabled: bool = True
    cfg_split_fingerprint: bool = False
    """Fingerprint only the cond half of a 2×B CFG batch (Cache-DiT / DBCache)."""


@dataclass(slots=True)
class BlockCacheStats:
    total_forwards: int = 0
    block_skips: int = 0
    full_recomputes: int = 0


class BlockDiTCache:
    """
    Per-sample-loop block cache. Attach to DiT via ``forward(..., block_cache=self)``.

    TeaCache-style: accumulate relative L1 distance between consecutive step
    fingerprints; when below threshold, apply cached per-block residuals.
    """

    def __init__(self, config: BlockCacheConfig | None = None) -> None:
        self.config = config or BlockCacheConfig()
        self.stats = BlockCacheStats()
        self._fingerprints: list[torch.Tensor] = []
        self._residuals: dict[int, torch.Tensor] = {}
        self._reuse_streak = 0
        self._step_index = 0
        self._accumulated_dist = 0.0
        self._force_full = True

    def reset(self) -> None:
        self.stats = BlockCacheStats()
        self._fingerprints.clear()
        self._residuals.clear()
        self._reuse_streak = 0
        self._step_index = 0
        self._accumulated_dist = 0.0
        self._force_full = True

    def begin_forward(self, fingerprint: torch.Tensor) -> None:
        """Call once per DiT forward with a (B, D) or (D,) step fingerprint."""
        cfg = self.config
        self.stats.total_forwards += 1
        self._step_index += 1
        fp = fingerprint.detach().float().reshape(-1)
        if not cfg.enabled:
            self._force_full = True
            return
        if self._step_index % max(1, int(cfg.recompute_every)) == 0:
            self._force_full = True
            self._accumulated_dist = 0.0
            self._reuse_streak = 0
            self.stats.full_recomputes += 1
            self._fingerprints = [fp]
            return
        if not self._fingerprints:
            self._force_full = True
            self._fingerprints = [fp]
            return
        prev = self._fingerprints[-1]
        dist = (fp - prev).abs().mean() / (prev.abs().mean() + 1e-6)
        self._accumulated_dist += float(dist.item())
        self._fingerprints.append(fp)
        if (
            self._accumulated_dist < float(cfg.rel_l1_threshold)
            and self._reuse_streak < int(cfg.max_reuse_streak)
            and self._residuals
        ):
            self._force_full = False
            self._reuse_streak += 1
        else:
            self._force_full = True
            self._accumulated_dist = 0.0
            self._reuse_streak = 0
            self.stats.full_recomputes += 1

    def should_skip_block(self, block_index: int) -> bool:
        return not self._force_full and block_index in self._residuals

    def apply_residual(self, x: torch.Tensor, block_index: int) -> torch.Tensor:
        res = self._residuals[block_index]
        self.stats.block_skips += 1
        if res.shape == x.shape:
            return x + res
        return x + res.to(dtype=x.dtype, device=x.device)

    def note_block(self, block_index: int, x_before: torch.Tensor, x_after: torch.Tensor) -> None:
        self._residuals[block_index] = (x_after - x_before).detach()

    @staticmethod
    def fingerprint_from_tensors(
        t_emb: torch.Tensor,
        x: torch.Tensor,
        *,
        cfg_split: bool = False,
    ) -> torch.Tensor:
        """Build step fingerprint from timestep embedding + latent token mean."""
        te = t_emb
        xs = x
        if bool(cfg_split) and te.shape[0] >= 2 and te.shape[0] % 2 == 0:
            half = te.shape[0] // 2
            te = te[:half]
            xs = xs[:half]
        te = te.detach().float().reshape(te.shape[0], -1).mean(dim=0)
        xm = xs.detach().float().reshape(xs.shape[0], -1).mean(dim=0)
        return torch.cat([te, xm], dim=0)


# ==================================================================================================
# feature_cache: **Feature-cache policy** for DiT sampling (SpeCa / TeaCache-inspired).
#
# When consecutive latent updates are small, reuse the previous model epsilon prediction
# for one step — optional speed win; disable when quality-critical.
# ==================================================================================================


@dataclass(slots=True)
class FeatureCacheState:
    """Per-sequence cache for sample loop."""

    last_x: torch.Tensor | None = None
    last_pred: torch.Tensor | None = None
    reuse_count: int = 0
    skip_count: int = 0


@dataclass(slots=True)
class FeatureCacheConfig:
    delta_threshold: float = 0.012
    max_reuse_streak: int = 2
    enabled: bool = True


class FeatureCachePolicy:
    """Decide whether to reuse previous DiT output at step ``i``."""

    def __init__(self, config: FeatureCacheConfig | None = None) -> None:
        self.config = config or FeatureCacheConfig()
        self.state = FeatureCacheState()

    def reset(self) -> None:
        self.state = FeatureCacheState()

    def maybe_reuse(
        self,
        x: torch.Tensor,
        fresh_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        If ``x`` changed little since last step, return cached ``last_pred`` instead of
        ``fresh_pred`` (caller still runs one forward — this is for optional external hooks).

        For inline use in sample_loop: call ``should_skip_forward`` before forward instead.
        """
        cfg = self.config
        st = self.state
        if not cfg.enabled or st.last_x is None or st.last_pred is None:
            st.last_x = x.detach()
            st.last_pred = fresh_pred.detach()
            return fresh_pred
        delta = (x - st.last_x).abs().mean()
        if (
            float(delta.item()) < cfg.delta_threshold
            and st.reuse_count < cfg.max_reuse_streak
            and st.last_pred.shape == fresh_pred.shape
        ):
            st.reuse_count += 1
            st.skip_count += 1
            return st.last_pred
        st.reuse_count = 0
        st.last_x = x.detach()
        st.last_pred = fresh_pred.detach()
        return fresh_pred

    def should_skip_forward(self, x: torch.Tensor) -> bool:
        """True if caller may reuse ``last_pred`` without running DiT."""
        cfg = self.config
        st = self.state
        if not cfg.enabled or st.last_x is None or st.last_pred is None:
            return False
        delta = (x - st.last_x).abs().mean()
        return float(delta.item()) < cfg.delta_threshold and st.reuse_count < cfg.max_reuse_streak

    def cached_prediction(self) -> torch.Tensor:
        """Return last cached prediction (must exist)."""
        if self.state.last_pred is None:
            raise RuntimeError("feature cache miss")
        self.state.reuse_count += 1
        self.state.skip_count += 1
        return self.state.last_pred

    def note_fresh(self, x: torch.Tensor, pred: torch.Tensor) -> None:
        self.state.reuse_count = 0
        self.state.last_x = x.detach()
        self.state.last_pred = pred.detach()


# ==================================================================================================
# taylor_cache: **TaylorSeer** block feature forecasting (ICCV 2025).
#
# Upgrades block cache from static reuse to Taylor-series prediction of per-block
# residuals using finite-difference derivatives across anchor timesteps.
# ==================================================================================================


@dataclass(slots=True)
class TaylorCacheConfig(BlockCacheConfig):
    use_taylor: bool = True
    cache_interval: int = 4
    """Steps between full block recomputes (N in TaylorSeer paper)."""
    max_order: int = 1
    """Taylor expansion order (0 = plain reuse, 1 = linear forecast)."""


def taylor_forecast_tensor(
    history: list[torch.Tensor],
    steps_since_anchor: int,
    *,
    interval: int = 4,
    max_order: int = 1,
) -> torch.Tensor:
    """
    Predict block residual at ``steps_since_anchor`` steps after last anchor.

    Uses finite differences on the last ``max_order+1`` anchor residuals::
        F_pred = F_0 + (dF / dt) * k + ...
    """
    if not history:
        raise ValueError("empty history")
    k = float(max(1, int(steps_since_anchor)))
    n = float(max(1, int(interval)))
    f0 = history[-1]
    if int(max_order) <= 0 or len(history) < 2:
        return f0
    f1 = history[-2]
    d1 = (f0 - f1) / n
    pred = f0 + d1 * k
    if int(max_order) >= 2 and len(history) >= 3:
        f2 = history[-3]
        d2 = (f0 - 2.0 * f1 + f2) / (n * n)
        pred = pred + 0.5 * d2 * (k * k)
    return pred


class TaylorBlockCache(BlockDiTCache):
    """Block cache with TaylorSeer forecast instead of zero-order reuse."""

    def __init__(self, config: TaylorCacheConfig | None = None) -> None:
        self.taylor_config = config or TaylorCacheConfig()
        super().__init__(self.taylor_config)
        self._anchor_residuals: dict[int, list[torch.Tensor]] = {}
        self._steps_since_anchor = 0

    def reset(self) -> None:
        super().reset()
        self._anchor_residuals.clear()
        self._steps_since_anchor = 0

    def begin_forward(self, fingerprint: torch.Tensor) -> None:
        was_full = self._force_full
        super().begin_forward(fingerprint)
        if self._force_full:
            self._steps_since_anchor = 0
        elif not was_full:
            self._steps_since_anchor += 1

    def note_block(self, block_index: int, x_before: torch.Tensor, x_after: torch.Tensor) -> None:
        super().note_block(block_index, x_before, x_after)
        if self._force_full:
            res = (x_after - x_before).detach()
            hist = self._anchor_residuals.setdefault(block_index, [])
            hist.append(res)
            keep = int(self.taylor_config.max_order) + 1
            if len(hist) > keep:
                del hist[:-keep]

    def apply_residual(self, x: torch.Tensor, block_index: int) -> torch.Tensor:
        cfg = self.taylor_config
        if cfg.use_taylor and block_index in self._anchor_residuals and self._steps_since_anchor > 0:
            hist = self._anchor_residuals[block_index]
            if hist:
                pred = taylor_forecast_tensor(
                    hist,
                    self._steps_since_anchor,
                    interval=int(cfg.cache_interval),
                    max_order=int(cfg.max_order),
                )
                self.stats.block_skips += 1
                if pred.shape == x.shape:
                    return x + pred
                return x + pred.to(dtype=x.dtype, device=x.device)
        return super().apply_residual(x, block_index)


# ==================================================================================================
# dynamic_dit: **DyDiT-style** dynamic compute helpers (ICLR 2025 / DyDiT++).
#
# Training-free inference hooks:
# - **Timestep-wise dynamic width (TDW)**: scale model output by denoise progress.
# - **Spatial token importance (SDT-lite)**: heuristic mask for "easy" latent regions.
#
# Does not modify checkpoint weights — scales activations during sampling.
# ==================================================================================================


def timestep_dynamic_width(
    progress: float,
    *,
    early: float = 0.88,
    late: float = 1.0,
    power: float = 1.0,
) -> float:
    """
    Width multiplier in ``[early, late]`` vs denoise progress (0=noisy → 1=clean).

    Early timesteps run slightly narrower (less over-processing).
    """
    p = float(max(0.0, min(1.0, progress))) ** float(power)
    return float(early) + (float(late) - float(early)) * p


def spatial_token_importance(latent: torch.Tensor) -> torch.Tensor:
    """
    Per-token importance in ``[0,1]`` from local gradient magnitude (SDT-lite).

    ``latent``: (B, C, H, W) → returns (B, 1, H, W) importance map.
    """
    if latent.ndim != 4:
        return torch.ones(latent.shape[0], 1, 1, 1, device=latent.device, dtype=latent.dtype)
    x = latent.float()
    gx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    gy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
    gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))
    imp = (gx + gy) * 0.5
    imp = imp / (imp.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return imp.to(dtype=latent.dtype)


def apply_dynamic_width(
    out: torch.Tensor,
    progress: float,
    *,
    early: float = 0.88,
    late: float = 1.0,
) -> torch.Tensor:
    """Scale model prediction by TDW multiplier."""
    s = timestep_dynamic_width(progress, early=early, late=late)
    return out * s


def apply_sdt_latent_blend(
    latent: torch.Tensor,
    model_delta: torch.Tensor,
    *,
    keep_threshold: float = 0.35,
) -> torch.Tensor:
    """
    Blend model update toward zero on low-importance spatial regions.

    ``model_delta`` is the velocity/noise increment for this step.
    """
    imp = spatial_token_importance(latent)
    gate = (imp >= float(keep_threshold)).to(dtype=model_delta.dtype)
    return model_delta * gate


# ==================================================================================================
# cfg_rejection: **CFG-Rejection** — early denoise path filtering via accumulated guidance score.
#
# When generating multiple candidates, abort trajectories whose early CFG score
# differences suggest low final quality (arXiv:2505.23343 scaffold).
#
# Training-free; useful with ``--num > 1`` best-of-N.
# ==================================================================================================


@dataclass
class CFGRejectionTracker:
    """Track per-trajectory CFG score gaps during early denoise steps."""

    tau_steps: int = 4
    scores: list[float] = field(default_factory=list)

    def note(self, out_cond: torch.Tensor, out_uncond: torch.Tensor) -> float:
        """Instantaneous guidance gap (L2 norm of cond-uncond)."""
        delta = (out_cond - out_uncond).float()
        g = float(delta.pow(2).mean().sqrt().item())
        self.scores.append(g)
        return g

    def accumulated_early_score(self) -> float:
        if not self.scores:
            return 0.0
        use = self.scores[: max(1, int(self.tau_steps))]
        return float(sum(use))

    def should_reject(self, threshold: float, *, best_accum: float | None = None) -> bool:
        acc = self.accumulated_early_score()
        if best_accum is None:
            return acc > float(threshold)
        return acc > float(best_accum) * 1.35 + float(threshold)


def pick_best_candidate_index(
    accum_scores: list[float],
    *,
    reject_threshold: float = 0.0,
) -> int:
    """Lower accumulated early CFG gap often correlates with better paths (heuristic)."""
    if not accum_scores:
        return 0
    ranked = sorted(range(len(accum_scores)), key=lambda i: accum_scores[i])
    for idx in ranked:
        if accum_scores[idx] <= float(reject_threshold) or reject_threshold <= 0:
            return idx
    return ranked[0]


class ParityStatus(str, Enum):
    """Rough alignment with an external research/product theme."""

    IMPLEMENTED = "implemented"  # Feature exists and is usable in train/sample
    PARTIAL = "partial"  # Related knobs or subset (e.g. AR blocks, not full AR tokenizer LM)
    RESEARCH = "research"  # Documented only; no dedicated trainer
    NOT_IN_REPO = "not_in_repo"  # Not present


@dataclass(frozen=True, slots=True)
class ThemeMapping:
    theme_id: str
    label: str
    status: ParityStatus
    summary: str
    repo_paths: tuple[str, ...]
    cli_flags: tuple[str, ...] = ()


# Canonical ordered list (single source of truth for docs/tests).
THEMES: tuple[ThemeMapping, ...] = (
    ThemeMapping(
        "flow_matching",
        "Flow matching / rectified flow",
        ParityStatus.PARTIAL,
        "Train: --flow-matching-training + diffusion/flow_matching.py. Sample: sample_loop(..., flow_matching_sample=True) "
        "or sample.py --flow-matching-sample / auto when checkpoint has flow_matching_training.",
        (
            "diffusion/flow_matching.py",
            "diffusion/gaussian_diffusion.py",
            "sample.py",
            "train.py",
            "docs/RESEARCH.md",
        ),
        ("--flow-matching-training", "--flow-matching-sample", "--flow-solver", "--prediction-type v"),
    ),
    ThemeMapping(
        "diffusion_bridges",
        "Diffusion bridges (A→B distributions)",
        ParityStatus.PARTIAL,
        "Linear latent interpolation (diffusion/latent_bridge.py) + optional VP shuffle-pair auxiliary loss in train "
        "(diffusion/bridge_training.py, --bridge-aux-weight); not a full Schrödinger-bridge trainer.",
        (
            "diffusion/latent_bridge.py",
            "diffusion/bridge_training.py",
            "train.py",
            "sample.py",
            "docs/RESEARCH.md",
        ),
        ("--bridge-aux-weight", "--bridge-aux-lambda"),
    ),
    ThemeMapping(
        "hybrid_ar_diffusion",
        "Hybrid AR planner + diffusion decoder",
        ParityStatus.PARTIAL,
        "Block-causal DiT (num_ar_blocks) + AR/ViT bridge — not a separate billion-parameter AR image tokenizer.",
        ("models/attention.py", "models/dit_text.py", "utils/modeling/architecture.py", "docs/MODEL_STACK.md"),
        ("--num-ar-blocks",),
    ),
    ThemeMapping(
        "vision_mamba_ssm",
        "Mamba / SSM vision backbones",
        ParityStatus.NOT_IN_REPO,
        "DiT uses transformer attention; no Mamba backbone in this repo.",
        (),
        (),
    ),
    ThemeMapping(
        "dmd_distillation",
        "DMD / one-step / consistency distillation",
        ParityStatus.PARTIAL,
        "Same-arch teacher→student MSE on shared (x_t,t) in scripts/tools/trainers.py (kd-distill); "
        "no DMD / one-step student or ADD.",
        ("scripts/tools/trainers.py", "docs/RESEARCH.md"),
        (),
    ),
    ThemeMapping(
        "rae_semantic_latent",
        "RAE / semantic-first latents",
        ParityStatus.IMPLEMENTED,
        "RAE autoencoder + RAELatentBridge; REPA aligns to frozen vision encoders.",
        ("models/rae_latent_bridge.py", "train.py", "sample.py"),
        ("--autoencoder-type rae", "--repa-weight"),
    ),
    ThemeMapping(
        "repa_alignment",
        "REPA (representation alignment)",
        ParityStatus.IMPLEMENTED,
        "Auxiliary alignment to DINOv2 (or configured encoder) during training.",
        ("train.py", "docs/MODEL_STACK.md"),
        ("--repa-weight", "--repa-encoder-model"),
    ),
    ThemeMapping(
        "physical_grounding",
        "Physics-informed / physical consistency layers",
        ParityStatus.NOT_IN_REPO,
        "No built-in physics simulator; optional verifier scores (e.g. exposure) in test_time_pick.",
        ("utils/quality/test_time_pick.py", "docs/RESEARCH.md"),
        ("--pick-best combo_exposure",),
    ),
    ThemeMapping(
        "rag_prompt_grounding",
        "RAG-style facts in prompt",
        ParityStatus.IMPLEMENTED,
        "merge_facts_into_prompt and JSONL loaders — no retrieval implementation.",
        ("utils/prompt/rag.py",),
        (),
    ),
    ThemeMapping(
        "orchestration_pipeline",
        "Multi-stage designer→verifier pipeline",
        ParityStatus.IMPLEMENTED,
        "orchestrate_pipeline.py, pick-best, orchestration roles.",
        ("scripts/tools/generate.py", "utils/agentic/agent.py"),
        ("--num", "--pick-best"),
    ),
    ThemeMapping(
        "resolution_buckets",
        "Multi-resolution / aspect buckets (training)",
        ParityStatus.IMPLEMENTED,
        "train.py --resolution-buckets; single-GPU; no val-split with buckets.",
        ("data/t2i_dataset.py", "train.py"),
        ("--resolution-buckets",),
    ),
    # --- Workflow / industry commentary (docs/RESEARCH.md) ---
    ThemeMapping(
        "workflow_structural_coherence",
        "Multi-view / structural coherence (industry stacks)",
        ParityStatus.NOT_IN_REPO,
        "No built-in 3D/pose engine; use data, control paths, character sheets, resolution tooling.",
        ("docs/RESEARCH.md", "utils/consistency.py"),
        (),
    ),
    ThemeMapping(
        "llada_discrete_diffusion_text",
        "Discrete diffusion on text / unified LM+image (LLaDA-class ideas)",
        ParityStatus.NOT_IN_REPO,
        "SDX: T5 (AR text encoding) + DiT diffusion — not diffusion over discrete text tokens.",
        ("docs/RESEARCH.md", "docs/RESEARCH.md"),
        (),
    ),
    ThemeMapping(
        "test_time_inference_scaling",
        "Test-time compute / critique-style inference (industry narratives)",
        ParityStatus.PARTIAL,
        "Refinement pass, pick-best, orchestrate_pipeline — explicit scoring, not proprietary latent self-critique.",
        ("sample.py", "utils/quality/test_time_pick.py", "scripts/tools/generate.py"),
        ("--num", "--pick-best"),
    ),
    ThemeMapping(
        "live_grounding_web",
        "Live web grounding during generation",
        ParityStatus.NOT_IN_REPO,
        "rag_prompt merges user-supplied facts; no built-in web retrieval in core train/sample.",
        ("utils/prompt/rag.py", "docs/RESEARCH.md"),
        (),
    ),
    ThemeMapping(
        "diffusion_leverage_roadmap",
        "Prioritized diffusion model upgrades (compound levers)",
        ParityStatus.RESEARCH,
        "Narrative doc: data/latent/objective/conditioning/inference/alignment sequencing; maps ideas to train.py, dit_text, diffusion/, RAE/REPA.",
        ("docs/RESEARCH.md", "docs/RESEARCH.md"),
        (),
    ),
    ThemeMapping(
        "auxiliary_structure_supervision",
        "Auxiliary depth/edge/segmentation heads on denoiser",
        ParityStatus.NOT_IN_REPO,
        "Structure losses for layout/comics not wired; would extend DiT outputs or intermediate features.",
        ("models/dit_text.py", "models/enhanced_dit.py", "docs/RESEARCH.md"),
        (),
    ),
    ThemeMapping(
        "reference_image_adapter_conditioning",
        "Reference-image / IP-Adapter-style conditioning for identity lock",
        ParityStatus.PARTIAL,
        "CLIP vision -> ReferenceTokenProjector -> extra cross-attn tokens in DiT_Text; optional trained --reference-adapter-pt. Also --init-image, --post-reference-image, character sheet.",
        (
            "models/dit_text.py",
            "models/reference_token_projection.py",
            "sample.py",
            "utils/generation/sample_core.py",
        ),
        (
            "--reference-image",
            "--reference-strength",
            "--reference-tokens",
            "--reference-adapter-pt",
            "--init-image",
            "--post-reference-image",
            "--character-sheet",
        ),
    ),
    ThemeMapping(
        "self_attention_guidance_blur",
        "Blur-based self-attention guidance (extra forward on blurred latent)",
        ParityStatus.IMPLEMENTED,
        "sample_loop: pred += sag_scale * (pred - pred(blur(x))); ~2× forwards when sag_scale and sag_blur_sigma > 0.",
        ("diffusion/gaussian_diffusion.py", "diffusion/sampling_utils.py", "sample.py"),
        ("--sag-blur-sigma", "--sag-scale"),
    ),
    # --- Next-gen “super-model” pillars (design doc: docs/RESEARCH.md) ---
    ThemeMapping(
        "nextgen_semantic_geometric_dual",
        "Next-gen: semantic–geometric dual backbone (global planner + DiT)",
        ParityStatus.PARTIAL,
        "No separate Mamba architect network; dual-stage layout, block AR, --ssm-every-n mixer, and rich conditioning approximate layout-first→detail.",
        (
            "docs/RESEARCH.md",
            "sample.py",
            "utils/generation/guidance.py",
            "models/dit_text.py",
        ),
        ("--dual-stage-layout", "--num-ar-blocks", "--ssm-every-n"),
    ),
    ThemeMapping(
        "nextgen_vlm_in_loop_critic",
        "Next-gen: in-loop VLM / discriminative correction",
        ParityStatus.PARTIAL,
        "CLIP guard, optional mid-loop CLIP CFG monitor, volatile CFG; not a full VLM gradient critic or localized rewind.",
        (
            "docs/RESEARCH.md",
            "utils/generation/guidance.py",
            "diffusion/gaussian_diffusion.py",
            "sample.py",
        ),
        ("--clip-guard-threshold", "--clip-monitor-every", "--volatile-cfg-boost", "--pick-best"),
    ),
    ThemeMapping(
        "nextgen_fourier_operator_diffusion",
        "Next-gen: Fourier / neural-operator style diffusion",
        ParityStatus.PARTIAL,
        "Spectral SFP loss + inference spectral-coherence latent blend; not full NOD denoising on arbitrary grids.",
        (
            "docs/RESEARCH.md",
            "diffusion/spectral_sfp.py",
            "utils/generation/guidance.py",
            "docs/RESEARCH.md",
        ),
        ("--spectral-sfp-loss", "--spectral-coherence-latent"),
    ),
    ThemeMapping(
        "nextgen_dpo_image_alignment",
        "Next-gen: direct preference optimization on images / latents",
        ParityStatus.PARTIAL,
        "Stage-2 script train_diffusion_dpo.py + DPO loss + preference JSONL/image dataset; not merged into train.py main loop.",
        (
            "docs/RESEARCH.md",
            "scripts/tools/trainers.py",
            "utils/training/preference.py",
            "utils/training/preference.py",
            "utils/training/preference.py",
            "diffusion/gaussian_diffusion.py",
        ),
        (),
    ),
    # --- Fast-gen math blueprint (docs/RESEARCH.md) ---
    ThemeMapping(
        "consistency_flow_matching_velocity",
        "Consistency / velocity self-consistency on flow fields (Consistency-FM class)",
        ParityStatus.PARTIAL,
        "Prototype rectified-flow-style velocity MSE in train (--flow-matching-training, diffusion/flow_matching.py); "
        "not Consistency-FM trajectory matching and not VP-compatible sampling without a flow sampler.",
        (
            "docs/RESEARCH.md",
            "diffusion/flow_matching.py",
            "train.py",
            "diffusion/gaussian_diffusion.py",
            "docs/RESEARCH.md",
        ),
        ("--flow-matching-training", "--prediction-type v", "--steps"),
    ),
    ThemeMapping(
        "dual_solver_time_warping",
        "Dual-regime solvers: ε/v/x₀ mixing + log vs linear time (learned τ)",
        ParityStatus.PARTIAL,
        "Single prediction type per checkpoint; non-uniform training timesteps (logit_normal, high_noise) — not dynamic dual integrator.",
        (
            "docs/RESEARCH.md",
            "diffusion/timestep_sampling.py",
            "train.py",
            "sample.py",
        ),
        ("--prediction-type", "--timestep-sample-mode"),
    ),
    ThemeMapping(
        "add_adversarial_distillation",
        "Adversarial Diffusion Distillation (ADD-class teacher→student)",
        ParityStatus.NOT_IN_REPO,
        "No adversarial distillation or dual-head discriminator student training.",
        ("docs/RESEARCH.md", "docs/RESEARCH.md"),
        (),
    ),
    ThemeMapping(
        "rectified_flow_ot_coupling",
        "Rectified flow + OT pairing (Sinkhorn / Hungarian noise–data coupling)",
        ParityStatus.PARTIAL,
        "Optional mini-batch OT noise coupling (--ot-noise-pair-reg) and optional flow-matching training path "
        "(--flow-matching-training); not a full continuous-time rectified-flow ODE trainer + sampler.",
        (
            "docs/RESEARCH.md",
            "utils/training/objectives.py",
            "diffusion/flow_matching.py",
            "train.py",
            "docs/RESEARCH.md",
        ),
        ("--ot-noise-pair-reg", "--ot-noise-pair-mode", "--flow-matching-training"),
    ),
    # --- Prompt-accuracy blueprint (docs/RESEARCH.md) ---
    ThemeMapping(
        "geometric_latent_split_blueprint",
        "Geometric–latent split (GLS): structural blueprint before texture",
        ParityStatus.PARTIAL,
        "Dual-stage layout and domain/layout priors in hooks; no dedicated depth–normal–edge transformer locked as immutable constraint.",
        (
            "docs/RESEARCH.md",
            "sample.py",
            "utils/generation/guidance.py",
        ),
        ("--dual-stage-layout", "--domain-prior-latent"),
    ),
    ThemeMapping(
        "discriminative_denoise_vlm_loop",
        "Discriminative denoising: VLM / critic in the sampling loop",
        ParityStatus.PARTIAL,
        "CLIP guard refine and volatile CFG; no frozen VLM every-k-steps with localized latent rewind / gradient reroll.",
        (
            "docs/RESEARCH.md",
            "utils/generation/guidance.py",
            "sample.py",
        ),
        (
            "--clip-guard-threshold",
            "--clip-monitor-every",
            "--volatile-cfg-boost",
            "--pick-best",
        ),
    ),
    ThemeMapping(
        "frequency_domain_global_coherence",
        "Frequency-domain / FNO-style global coherence (prompt-scale narrative)",
        ParityStatus.PARTIAL,
        "Spectral SFP training loss + inference ``--spectral-coherence-latent`` (FFT lowfreq blend); not full FNO denoising forward.",
        (
            "docs/RESEARCH.md",
            "diffusion/spectral_sfp.py",
            "utils/generation/guidance.py",
            "docs/RESEARCH.md",
        ),
        ("--spectral-sfp-loss", "--spectral-coherence-latent"),
    ),
)


def theme_by_id(theme_id: str) -> ThemeMapping | None:
    """Return theme by ``theme_id`` or None."""
    tid = (theme_id or "").strip().lower()
    for t in THEMES:
        if t.theme_id == tid:
            return t
    return None


def themes_as_dict() -> list[dict[str, object]]:
    """JSON-friendly list for tooling."""
    out: list[dict[str, object]] = []
    for t in THEMES:
        out.append(
            {
                "theme_id": t.theme_id,
                "label": t.label,
                "status": t.status.value,
                "summary": t.summary,
                "repo_paths": list(t.repo_paths),
                "cli_flags": list(t.cli_flags),
            }
        )
    return out


def summary_table_md() -> str:
    """Minimal Markdown table for pasting into docs or logs."""
    lines = [
        "| Theme | Status | SDX pointers |",
        "| :--- | :--- | :--- |",
    ]
    for t in THEMES:
        ptr = ", ".join(t.repo_paths[:2]) if t.repo_paths else "—"
        lines.append(f"| **{t.label}** | `{t.status.value}` | {ptr} |")
    return "\n".join(lines)
