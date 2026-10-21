"""**Block-wise DiT cache** (BWCache / TeaCache-inspired, training-free).

Caches per-block **residuals** across denoising steps. When the step fingerprint
(timestep + latent stats) changes little, reuses cached block deltas instead of
re-running transformer blocks — typically 1.3–2× fewer block FLOPs at modest thresholds.

See ``docs/research/SUPERIOR_RESEARCH_2026.md``.

Also holds what used to be: block_cache, feature_cache, taylor_cache, dynamic_dit, cfg_rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# ==================================================================================================
# block_cache: **Block-wise DiT cache** (BWCache / TeaCache-inspired, training-free).
#
# Caches per-block **residuals** across denoising steps. When the step fingerprint
# (timestep + latent stats) changes little, reuses cached block deltas instead of
# re-running transformer blocks — typically 1.3–2× fewer block FLOPs at modest thresholds.
#
# See ``docs/research/SUPERIOR_RESEARCH_2026.md``.
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


__all__ = [
    "BlockCacheConfig",
    "BlockCacheStats",
    "BlockDiTCache",
    "FeatureCacheConfig",
    "FeatureCachePolicy",
    "FeatureCacheState",
    "TaylorBlockCache",
    "TaylorCacheConfig",
    "taylor_forecast_tensor",
    "apply_dynamic_width",
    "apply_sdt_latent_blend",
    "spatial_token_importance",
    "timestep_dynamic_width",
    "CFGRejectionTracker",
    "pick_best_candidate_index",
]
