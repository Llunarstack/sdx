"""Preference learning: Diffusion-DPO data and losses, Flow-GRPO / BranchGRPO / DenseGRPO / TP-GRPO, LADD distillation and the user feedback bus."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.safety.minor_guard import reasons

# ==================================================================================================
# preference_jsonl: Pairwise preference rows for Diffusion-DPO (``python -m scripts.tools trainers dpo``).
#
# Each line is a JSON object. Supported keys (aliases in parentheses):
#
# - ``win_image_path`` (``winner``, ``preferred``): path to the preferred image
# - ``lose_image_path`` (``loser``, ``rejected``): path to the non-preferred image
# - ``caption`` (``prompt``, ``text``): shared conditioning text
# ==================================================================================================


@dataclass(frozen=True, slots=True)
class PreferencePair:
    win_path: str
    lose_path: str
    prompt: str
    raw: dict[str, Any]


def _pick_str(d: dict[str, Any], keys: tuple) -> str | None:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def parse_preference_row(obj: dict[str, Any]) -> PreferencePair | None:
    win = _pick_str(obj, ("win_image_path", "winner", "preferred", "win"))
    lose = _pick_str(obj, ("lose_image_path", "loser", "rejected", "lose"))
    prompt = _pick_str(obj, ("caption", "prompt", "text")) or ""
    if not win or not lose:
        return None
    return PreferencePair(win_path=win, lose_path=lose, prompt=prompt, raw=dict(obj))


def iter_preference_jsonl(path: str | Path) -> Iterator[PreferencePair]:
    """Yield ``PreferencePair`` for each valid row; skip blank lines and invalid rows."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            pair = parse_preference_row(obj)
            if pair is not None:
                yield pair


# ==================================================================================================
# dpo_advanced: Advanced **Diffusion-DPO** helpers inspired by 2025–2026 alignment research.
#
# - Timestep-aware weighting (upweight high-noise steps; compensates signal attenuation)
# - Safeguarded DPO margin (SDPO-style; caps destructive loser pushes)
# - EMA reference model updates (dynamic ref vs frozen ref)
#
# References: Rethinking DPO in Diffusion (AAAI 2026 oral); Diffusion-SDPO (safeguarded updates).
# ==================================================================================================


def timestep_dpo_weight(
    t: torch.Tensor,
    num_timesteps: int,
    *,
    mode: str = "high_noise",
    power: float = 0.5,
) -> torch.Tensor:
    """
    Per-sample weight for DPO loss at timestep ``t``.

    ``high_noise``: larger ``t`` (more noise) gets higher weight — matches diffusion-DPO
    timestep imbalance findings (low-noise steps dominate unless reweighted).
    """
    T = max(1, int(num_timesteps) - 1)
    tn = t.detach().float().clamp(0, T) / float(T)
    if mode in ("uniform", "none", ""):
        return torch.ones_like(tn)
    if mode == "low_noise":
        w = (1.0 - tn) ** float(power)
    else:
        w = tn ** float(power)
    return w.clamp(min=0.05) + 0.95 * w / (w.mean() + 1e-8)


def safeguard_dpo_margins(
    implicit_logp_win: torch.Tensor,
    implicit_logp_lose: torch.Tensor,
    *,
    strength: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SDPO-style surrogate: shrink loser advantage when winner is already strongly preferred.

    Operates on scalar implicit log-probs (typically ``-loss``). Does not require separate
    backward through winner/loser branches.
    """
    if strength <= 0.0:
        return implicit_logp_win, implicit_logp_lose
    margin = implicit_logp_win - implicit_logp_lose
    # When margin is positive (winner better), slightly pull loser up toward winner
    adjust = torch.clamp(margin, min=0.0) * (1.0 - float(strength))
    lose_adj = implicit_logp_lose + adjust
    return implicit_logp_win, lose_adj


def safeguarded_dpo_preference_loss(
    implicit_logp_win: torch.Tensor,
    implicit_logp_lose: torch.Tensor,
    implicit_logp_ref_win: torch.Tensor,
    implicit_logp_ref_lose: torch.Tensor,
    *,
    beta: float = 5000.0,
    logit_clip: float | None = None,
    safeguard_strength: float = 0.85,
    timestep_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """DPO loss with optional SDPO margin safeguard + per-timestep weights."""
    lw, ll = safeguard_dpo_margins(implicit_logp_win, implicit_logp_lose, strength=float(safeguard_strength))
    b = float(beta)
    pi = b * ((lw - ll) - (implicit_logp_ref_win - implicit_logp_ref_lose))
    if logit_clip is not None and float(logit_clip) > 0.0:
        c = float(logit_clip)
        pi = pi.clamp(-c, c)
    loss = -F.logsigmoid(pi)
    if timestep_weights is not None:
        w = timestep_weights.detach().float()
        if w.shape == loss.shape:
            return (loss * w).sum() / (w.sum() + 1e-8)
    return loss.mean()


@torch.no_grad()
def ema_update_reference(
    ref_model: torch.nn.Module,
    policy_model: torch.nn.Module,
    *,
    alpha: float = 0.01,
) -> None:
    """Soft-update reference toward policy (``ref = (1-a)*ref + a*policy``)."""
    a = float(max(0.0, min(1.0, alpha)))
    ref_sd = ref_model.state_dict()
    pol_sd = policy_model.state_dict()
    for k in ref_sd:
        if k not in pol_sd:
            continue
        rp = ref_sd[k]
        pp = pol_sd[k]
        if torch.is_floating_point(rp):
            ref_sd[k].mul_(1.0 - a).add_(pp.detach(), alpha=a)
    ref_model.load_state_dict(ref_sd)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False


# ==================================================================================================
# preference_image_dataset: Load win/lose image pairs from preference JSONL for Diffusion-DPO-style training.
# ==================================================================================================


def _resolve_path(p: str, root: Path | None) -> Path:
    path = Path(p)
    if path.is_file():
        return path
    if root is not None:
        cand = root / path
        if cand.is_file():
            return cand
    return path


def _load_rgb_tensor(path: Path, image_size: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    im = im.resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = torch.from_numpy(np.array(im).astype("float32") / 255.0)
    arr = arr.permute(2, 0, 1)
    arr = (arr - 0.5) / 0.5
    return arr


class PreferenceImageDataset(torch.utils.data.Dataset):
    """
    Rows from ``iter_preference_jsonl`` (minus prompts involving minors); each sample is win/lose tensors
    ``(3,H,W)`` in [-1, 1].
    """

    def __init__(self, jsonl_path: str | Path, *, image_size: int, image_root: str | Path | None = None):
        self.rows: list[PreferencePair] = [
            r for r in iter_preference_jsonl(jsonl_path) if not reasons(r.prompt, training=True)
        ]
        if not self.rows:
            raise ValueError(f"No valid preference rows in {jsonl_path}")
        self.image_size = int(image_size)
        self.root = Path(image_root) if image_root else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        wp = _resolve_path(r.win_path, self.root)
        lp = _resolve_path(r.lose_path, self.root)
        if not wp.is_file():
            raise FileNotFoundError(f"Win image not found: {wp}")
        if not lp.is_file():
            raise FileNotFoundError(f"Lose image not found: {lp}")
        return {
            "win": _load_rgb_tensor(wp, self.image_size),
            "lose": _load_rgb_tensor(lp, self.image_size),
            "prompt": r.prompt if r.prompt.strip() else " ",
        }


def collate_preference_batch(batch: list[dict]) -> dict:
    return {
        "win": torch.stack([b["win"] for b in batch], dim=0),
        "lose": torch.stack([b["lose"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
    }


@dataclass(slots=True)
class FlowGRPOConfig:
    num_samples: int = 4
    train_steps_per_prompt: int = 1
    sde_noise_scale: float = 0.35
    """Injected noise for SDE-style exploration during rollouts."""
    kl_coef: float = 0.02
    """MSE penalty vs reference (anti reward hacking)."""
    advantage_clip: float = 3.0
    reward_temperature: float = 1.0
    denoise_steps: int = 8
    """Reduced steps during rollout (Flow-GRPO denoising reduction)."""


def flow_sde_step(
    x: torch.Tensor,
    v_pred: torch.Tensor,
    s_cur: float,
    s_next: float,
    *,
    noise_level: float = 0.7,
) -> torch.Tensor:
    """
    One step of Flow-GRPO's marginal-preserving reverse SDE (Liu et al. 2025, Eq. 9) for ``x_s = (1-s) x0 + s eps``:

    ``x' = x - [v + sigma^2/(2s) (x + (1-s) v)] ds + sigma sqrt(ds) z``, ``sigma = noise_level * sqrt(s / (1-s))``,
    ``ds = s_cur - s_next > 0``. The drift adds ``sigma^2/2`` times the score ``-(x + (1-s) v)/s`` so every ``x_s`` keeps
    the probability-flow ODE's marginal while the injected noise gives GRPO distinct rollouts to compare.
    """
    ds = float(s_cur) - float(s_next)
    # sigma is unbounded at s = 1; like the reference code, use the next grid point in the denominator there.
    denom = 1.0 - (float(s_next) if float(s_cur) >= 1.0 else float(s_cur))
    sigma = float(noise_level) * (float(s_cur) / max(denom, 1e-6)) ** 0.5
    drift = v_pred + (sigma**2 / (2.0 * max(float(s_cur), 1e-6))) * (x + (1.0 - float(s_cur)) * v_pred)
    return x - drift * ds + sigma * ds**0.5 * torch.randn_like(x)


def group_relative_advantages(rewards: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """GRPO-style advantages: (r - mean) / (std + eps) within group."""
    r = rewards.detach().float()
    if r.numel() <= 1:
        return torch.zeros_like(r)
    mu = r.mean()
    sd = r.std(unbiased=False)
    return ((r - mu) / (sd + eps)).clamp(-3.0, 3.0)


def grpo_weighted_loss(
    per_sample_loss: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip: float = 3.0,
    temperature: float = 1.0,
    extra_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reward-weighted regression over one rollout group (Peters & Schaal 2007; Lee et al. 2023 for text-to-image):
    ``sum_i w_i L_i / sum_i w_i`` with ``w_i = exp(clip(A_i) / temperature)``, so the denoising loss is fitted hardest
    on the highest-reward rollouts. ``extra_weights`` (e.g. per-timestep factors) multiply ``w``.

    The weights are normalised over the group, so pass all of its samples at once: a single sample's weight cancels.
    """
    if per_sample_loss.numel() < 2:
        raise ValueError("grpo_weighted_loss needs the whole group (>= 2 samples); one sample's weight cancels out")
    adv = advantages.detach().float().view(-1).clamp(-float(clip), float(clip))
    w = torch.exp(adv / max(float(temperature), 1e-6))
    if extra_weights is not None:
        w = w * extra_weights.detach().float().view(-1)
    w = w.to(per_sample_loss.dtype)
    return (per_sample_loss.view(-1) * w).sum() / (w.sum() + 1e-8)


def reference_kl_penalty(
    policy_pred: torch.Tensor,
    ref_pred: torch.Tensor,
    *,
    coef: float = 0.02,
) -> torch.Tensor:
    """Simple MSE KL surrogate between policy and frozen reference."""
    if coef <= 0.0:
        return policy_pred.new_zeros(())
    return float(coef) * F.mse_loss(policy_pred, ref_pred.detach())


def decode_latent_to_rgb_uint8(
    latent: torch.Tensor,
    vae: torch.nn.Module,
    *,
    latent_scale: float = 0.18215,
    ae_type: str = "kl",
    rae_bridge: Any = None,
) -> np.ndarray:
    """Decode single latent (1,C,H,W) to RGB uint8 numpy."""
    z = latent / float(latent_scale) if str(ae_type) == "kl" else latent
    if str(ae_type) == "rae" and rae_bridge is not None:
        z = rae_bridge.dit_to_rae(z)
    decoded = vae.decode(z)
    img01 = ((decoded.sample if hasattr(decoded, "sample") else decoded) * 0.5 + 0.5).clamp(0, 1)
    return (img01[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def rollout_flow_sample(
    model: torch.nn.Module,
    diffusion: Any,
    shape: tuple[int, ...],
    *,
    model_kwargs_cond: dict[str, Any],
    model_kwargs_uncond: dict[str, Any] | None,
    cfg_scale: float,
    steps: int,
    device: str,
    sde_noise_scale: float = 0.35,
    flow: bool = True,
    flow_shift: float = 1.0,
) -> torch.Tensor:
    """
    Exploratory rollout for GRPO. Flow models integrate :func:`flow_sde_step` on the (SD3-shifted) time grid with CFG;
    VP models use ``sample_loop`` with DDIM ``eta = sde_noise_scale`` (eta = 1 is ancestral DDPM).
    """
    if not flow:
        return diffusion.sample_loop(
            model,
            shape,
            model_kwargs_cond=model_kwargs_cond,
            model_kwargs_uncond=model_kwargs_uncond,
            cfg_scale=float(cfg_scale),
            num_inference_steps=max(2, int(steps)),
            eta=float(sde_noise_scale),
            timestep_schedule="euler",
            device=device,
            dtype=torch.float32,
        )
    from diffusion.flow_matching import flow_time_to_index
    from diffusion.schedules import time_shift

    num_t = int(diffusion.num_timesteps)
    grid = time_shift(torch.linspace(1.0, 0.0, max(2, int(steps)) + 1), float(flow_shift)).tolist()
    x = torch.randn(shape, device=device, dtype=torch.float32)
    guided = model_kwargs_uncond is not None and float(cfg_scale) != 1.0
    for s_cur, s_next in zip(grid[:-1], grid[1:], strict=True):
        t = flow_time_to_index(torch.full((shape[0],), s_cur, device=device), num_t)
        v = model(x, t, **model_kwargs_cond)[:, : shape[1]].float()
        if guided:
            v_u = model(x, t, **model_kwargs_uncond)[:, : shape[1]].float()
            v = v_u + float(cfg_scale) * (v - v_u)
        x = flow_sde_step(x, v, s_cur, s_next, noise_level=sde_noise_scale) if s_next > 0 else x - s_cur * v
    return x


@dataclass(slots=True)
class SRPOConfig:
    """SRPO realism controls (Tencent Hunyuan 2025, "Directly Aligning the Full Diffusion Trajectory")."""

    control_positive: str = "natural skin texture, realistic photograph, true color"
    control_negative: str = "oily glossy skin, waxy plastic render, ai generated look"
    late_timestep_cutoff: float = 0.75
    """Trajectory fraction (0=noise, 1=clean) past which updates are masked: late steps over-polish skin."""
    reward_temperature: float = 1.0


def semantic_relative_reward(
    score_fn: Callable[[Any, str], Any], images: Any, prompt: str, *, config: SRPOConfig
) -> Any:
    """``score(prompt + positive) - score(prompt + negative)``: the reward model's global aesthetic bias
    (the shared "AI look") cancels and only the control axis remains."""
    pos = f"{prompt}, {config.control_positive}" if prompt else config.control_positive
    neg = f"{prompt}, {config.control_negative}" if prompt else config.control_negative
    return (score_fn(images, pos) - score_fn(images, neg)) / max(config.reward_temperature, 1e-6)


def timestep_reward_mask(trajectory_fractions: torch.Tensor, *, cutoff: float = 0.75) -> torch.Tensor:
    """1.0 for early/mid trajectory positions (0=noise, 1=clean), 0.0 past ``cutoff`` (Direct-Align guard)."""
    return (trajectory_fractions.float() <= float(cutoff)).float()


# ==================================================================================================
# branch_grpo: **BranchGRPO** scaffold — tree-structured rollouts for diffusion GRPO (2025).
#
# At scheduled split steps, trajectories branch into multiple children while reusing shared
# prefix computation. Depth-wise reward fusion aggregates leaf scores back to branches.
#
# Inspired by BranchGRPO (structured branching in diffusion models).
# ==================================================================================================


@dataclass(slots=True)
class BranchGRPOConfig:
    branch_factor: int = 2
    """Children per split node."""
    split_step_fractions: tuple[float, ...] = (0.35, 0.65)
    """Fractions along denoise trajectory where branching occurs."""
    fuse: str = "mean"
    """Leaf reward fusion: mean | max | min."""


def split_steps_from_fractions(total_steps: int, fractions: Sequence[float]) -> list[int]:
    """Map fractional positions to integer step indices in ``[0, total_steps)``."""
    n = max(1, int(total_steps))
    out: list[int] = []
    for f in fractions:
        idx = int(round(float(f) * (n - 1)))
        idx = max(0, min(n - 1, idx))
        if not out or out[-1] != idx:
            out.append(idx)
    return out


def enumerate_branch_paths(
    total_steps: int,
    *,
    branch_factor: int = 2,
    split_fractions: Sequence[float] = (0.35, 0.65),
) -> list[tuple[int, ...]]:
    """
    Return leaf paths as tuples of branch choices (0..branch_factor-1) per split.

    Does not run the model — planning helper for trainers.
    """
    splits = split_steps_from_fractions(total_steps, split_fractions)
    if not splits:
        return [(0,)]
    k = max(1, int(branch_factor))
    paths: list[tuple[int, ...]] = [(0,)]
    for _ in splits:
        paths = [p + (b,) for p in paths for b in range(k)]
    return paths


def branch_relative_advantages(
    path_rewards: torch.Tensor,
    path_ids: Sequence[str],
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    GRPO-style advantages over branch leaves sharing the same prompt.

    ``path_rewards``: (P,), one reward per leaf path.
    """
    r = path_rewards.detach().float().view(-1)
    if r.numel() <= 1:
        return torch.zeros_like(r)
    mu = r.mean()
    sd = r.std(unbiased=False)
    return ((r - mu) / (sd + eps)).clamp(-3.0, 3.0)


def branch_rollout_flow_samples(
    rollout_fn,
    *,
    num_paths: int,
    steps: int,
    branch_factor: int = 2,
    split_fractions: Sequence[float] = (0.35, 0.65),
    base_seed: int = 0,
) -> list[torch.Tensor]:
    """
    Run multiple rollout paths with distinct seeds (BranchGRPO scaffold).

    Full prefix-sharing would require hooking inside ``sample_loop``; this helper
    reuses the planning API and runs independent short rollouts per leaf path.
    """
    paths = enumerate_branch_paths(
        int(steps),
        branch_factor=int(branch_factor),
        split_fractions=split_fractions,
    )
    n = max(1, int(num_paths))
    paths = paths[:n] if len(paths) >= n else paths + [paths[-1]] * (n - len(paths))
    latents: list[torch.Tensor] = []
    for i, _path in enumerate(paths):
        if base_seed >= 0:
            torch.manual_seed(int(base_seed) + i)
        latents.append(rollout_fn())
    return latents


# ==================================================================================================
# dense_grpo: **DenseGRPO** helpers — step-wise dense rewards for flow-matching alignment (2026).
#
# Estimates per-denoise-step reward gain via short ODE decode of intermediate latents,
# plus reward-aware SDE noise scaling (calibrated exploration by timestep).
# ==================================================================================================


@dataclass(slots=True)
class DenseGRPOConfig:
    ode_refine_steps: int = 4
    """Short deterministic ODE steps to estimate x0 from x_t."""
    reward_aware_sde: bool = True
    base_sde_scale: float = 0.35


def flow_fraction_from_t_index(t_idx: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """Map discrete VP index to flow fraction in [0, 1] (noise → data)."""
    T = max(1, int(num_timesteps) - 1)
    return t_idx.detach().float().clamp(0, T) / float(T)


def estimate_x0_from_xt_flow(
    x_t: torch.Tensor,
    v_pred: torch.Tensor,
    t_frac: torch.Tensor,
) -> torch.Tensor:
    """One-step x0 estimate from flow velocity: ``x0 ≈ x_t - t * v``."""
    tv = t_frac.view(-1, 1, 1, 1).to(device=x_t.device, dtype=x_t.dtype)
    return x_t - tv * v_pred


def short_ode_refine_x0(
    model: torch.nn.Module,
    diffusion: object,
    x_t: torch.Tensor,
    t_start: torch.Tensor,
    *,
    model_kwargs: dict,
    steps: int = 4,
    flow: bool = True,
) -> torch.Tensor:
    """
    Deterministic short rollout from ``x_t`` toward x0 (DenseGRPO ODE decode) with ``sample_loop``'s ``x_init``
    resume; ``flow`` selects the flow-matching or VP trajectory to match the checkpoint.
    """
    if not hasattr(diffusion, "sample_loop"):
        v = model(x_t, t_start, **model_kwargs)
        if v.shape[1] > x_t.shape[1]:
            v = v[:, : x_t.shape[1]]
        tf = flow_fraction_from_t_index(t_start, int(getattr(diffusion, "num_timesteps", 1000)))
        return estimate_x0_from_xt_flow(x_t, v, tf)
    t0 = int(t_start.reshape(-1)[0].detach().cpu().item())
    return diffusion.sample_loop(
        model,
        x_t.shape,
        model_kwargs_cond=model_kwargs,
        model_kwargs_uncond=None,
        cfg_scale=1.0,
        num_inference_steps=max(1, int(steps)),
        device=str(x_t.device),
        dtype=x_t.dtype,
        x_init=x_t,
        start_timestep=t0,
        flow_matching_sample=bool(flow),
        flow_solver="euler",
    )


def step_advantages_from_gains(gains: Sequence[float], *, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative advantages on dense step gains (one timestep, G samples)."""
    return group_relative_advantages(torch.tensor(list(gains), dtype=torch.float32), eps=eps)


def score_latent_reward(
    x0_hat: torch.Tensor,
    prompt: str,
    *,
    vae: torch.nn.Module,
    score_fn: Callable[[np.ndarray, str], float],
    latent_scale: float = 0.18215,
    ae_type: str = "kl",
    rae_bridge: object = None,
) -> float:
    """Decode latent estimate and run reward model."""
    rgb = decode_latent_to_rgb_uint8(x0_hat, vae, latent_scale=latent_scale, ae_type=ae_type, rae_bridge=rae_bridge)
    return float(score_fn(rgb, prompt))


# ==================================================================================================
# flash_grpo: **Flash-GRPO** helpers — efficient single-step / iso-temporal GRPO (2025–2026).
#
# - **Iso-temporal grouping**: all rollouts in a GRPO group share the same denoise timestep
#   for SDE exploration so advantages reflect policy quality, not timestep difficulty.
# - **Temporal gradient rectification**: normalize policy gradients by SDE discretization scale λ(t).
#
# Inspired by Flash-GRPO (video alignment); applicable to image DiT flow training scaffolds.
# ==================================================================================================


def sample_iso_temporal_index(
    num_timesteps: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one timestep index in ``[0, num_timesteps)`` for a whole GRPO group."""
    T = max(1, int(num_timesteps))
    return torch.randint(0, T, (1,), device=device, generator=generator, dtype=torch.long)


def sde_discretization_lambda(t_frac: float | torch.Tensor, *, floor: float = 1e-4) -> float | torch.Tensor:
    """
    Time-dependent SDE scaling λ(t) ∝ sqrt(t(1-t)) on flow fraction.

    Used to rectify gradient magnitude across timesteps (Flash-GRPO).
    """
    if isinstance(t_frac, torch.Tensor):
        u = t_frac.detach().float().clamp(0.0, 1.0)
        lam = torch.sqrt(u * (1.0 - u) + float(floor))
        return lam
    u = float(max(0.0, min(1.0, float(t_frac))))
    return float(max((u * (1.0 - u)) ** 0.5, floor**0.5))


def rectify_policy_gradient(
    grad_or_loss_scale: torch.Tensor,
    t_frac: float | torch.Tensor,
    *,
    floor: float = 1e-4,
) -> torch.Tensor:
    """Divide by λ(t) so mid-trajectory steps do not dominate updates."""
    lam = sde_discretization_lambda(t_frac, floor=floor)
    if isinstance(lam, torch.Tensor):
        return grad_or_loss_scale / lam.clamp_min(float(floor))
    return grad_or_loss_scale / max(float(lam), float(floor))


def iso_temporal_group_advantages(
    rewards: torch.Tensor,
    timestep_indices: torch.Tensor,
    *,
    num_timesteps: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    GRPO advantages normalized **within (timestep bucket, prompt group)**.

    ``rewards``: (N,), ``timestep_indices``: (N,) discrete step indices.
    """
    r = rewards.detach().float().view(-1)
    t = timestep_indices.detach().long().view(-1)
    if r.numel() <= 1:
        return torch.zeros_like(r)
    adv = torch.zeros_like(r)
    for ti in t.unique().tolist():
        mask = t == int(ti)
        if mask.sum() <= 1:
            continue
        g = r[mask]
        mu = g.mean()
        sd = g.std(unbiased=False)
        adv[mask] = (g - mu) / (sd + eps)
    if adv.abs().sum() == 0:
        mu = r.mean()
        sd = r.std(unbiased=False)
        adv = (r - mu) / (sd + eps)
    return adv.clamp(-3.0, 3.0)


# ==================================================================================================
# grpo_guard: **GRPO-Guard** — mitigate implicit over-optimization in flow GRPO (2025).
#
# RatioNorm on importance ratios + per-timestep gradient reweighting.
#
# Wang et al., arXiv:2510.22319.
# ==================================================================================================


@dataclass(slots=True)
class GRPOGuardConfig:
    clip_low: float = 0.8
    clip_high: float = 1.25
    ratio_norm: bool = True
    grad_reweight: bool = True


def ratio_norm_advantages(
    advantages: torch.Tensor,
    *,
    clip_low: float = 0.8,
    clip_high: float = 1.25,
) -> torch.Tensor:
    """
    Stabilize GRPO advantages by normalizing magnitude and clipping extreme ratios.
    """
    adv = advantages.detach().float()
    if adv.numel() <= 1:
        return adv
    mu = adv.mean()
    sd = adv.std(unbiased=False) + 1e-6
    normed = (adv - mu) / sd
    imp = torch.exp(normed.clamp(-2.0, 2.0))
    imp = imp / (imp.mean() + 1e-6)
    return imp.clamp(float(clip_low), float(clip_high))


def grpo_guard_weighted_loss(
    per_sample_loss: torch.Tensor,
    advantages: torch.Tensor,
    t_fracs: torch.Tensor,
    *,
    config: GRPOGuardConfig | None = None,
) -> torch.Tensor:
    """Reward-weighted loss over one rollout group with GRPO-Guard's corrections: RatioNorm-normalised weights
    (increasing in the advantage, clipped to ``[clip_low, clip_high]``) and ``1 / lambda(t)`` per-timestep gradient
    reweighting. Weights are normalised over the group, so pass all of its samples at once."""
    cfg = config or GRPOGuardConfig()
    adv = advantages.detach().float().view(-1)
    if adv.numel() < 2:
        raise ValueError("grpo_guard_weighted_loss needs the whole group (>= 2 samples)")
    if cfg.ratio_norm:
        w = ratio_norm_advantages(adv, clip_low=cfg.clip_low, clip_high=cfg.clip_high)
    else:
        w = torch.exp(adv.clamp(-3.0, 3.0))
    if cfg.grad_reweight:
        w = w / sde_discretization_lambda(t_fracs.detach().float().view(-1).expand_as(w)).clamp_min(1e-4)
    w = w.to(per_sample_loss.dtype)
    return (per_sample_loss.view(-1) * w).sum() / (w.sum() + 1e-8)


# ==================================================================================================
# turning_point_grpo: **TurningPoint-GRPO (TP-GRPO)** — step incremental rewards + turning-point long-term credit.
#
# Alleviates sparse terminal rewards in flow-based GRPO by:
# 1. Incremental step rewards ``ΔR_t = R_t - R_{t-1}``
# 2. Turning points where ``sign(ΔR_t) != sign(ΔR_{t+1})`` get aggregated long-term bonus
#
# Tong et al., arXiv:2602.06422 (2026).
# ==================================================================================================


@dataclass(slots=True)
class TurningPointGRPOConfig:
    long_term_weight: float = 0.5
    """Weight on aggregated reward assigned at turning-point steps."""
    use_incremental_only: bool = False
    """If True, skip turning-point bonus (ablation)."""


def incremental_rewards_from_trajectory(step_rewards: Sequence[float]) -> list[float]:
    """``ΔR_t`` from absolute per-step rewards ``R_t``."""
    prev = 0.0
    out: list[float] = []
    for r in step_rewards:
        rv = float(r)
        out.append(rv - prev)
        prev = rv
    return out


def detect_turning_point_indices(incremental: Sequence[float]) -> list[int]:
    """
    Turning point at step ``t`` when ``sign(ΔR_t) != sign(ΔR_{t+1})`` (non-zero signs).
    """
    inc = [float(x) for x in incremental]
    if len(inc) < 2:
        return []
    tps: list[int] = []
    for t in range(len(inc) - 1):
        a, b = inc[t], inc[t + 1]
        if a == 0.0 or b == 0.0:
            continue
        if (a > 0) != (b > 0):
            tps.append(t)
    return tps


def long_term_aggregate_reward(
    incremental: Sequence[float],
    terminal_reward: float,
) -> float:
    """Simple long-term signal: terminal minus mean incremental."""
    inc = list(incremental)
    if not inc:
        return float(terminal_reward)
    return float(terminal_reward) - sum(inc) / len(inc)


def tp_grpo_step_weights(
    step_rewards: Sequence[float],
    terminal_reward: float,
    *,
    config: TurningPointGRPOConfig | None = None,
) -> list[float]:
    """
    Per-step training weights combining incremental + turning-point long-term bonus.

    Returns one scalar weight per denoise step (same length as ``step_rewards``).
    """
    cfg = config or TurningPointGRPOConfig()
    inc = incremental_rewards_from_trajectory(step_rewards)
    if not inc:
        return [float(terminal_reward)]

    weights = list(inc)
    if not cfg.use_incremental_only:
        lt = long_term_aggregate_reward(inc, terminal_reward)
        for t in detect_turning_point_indices(inc):
            weights[t] = weights[t] + float(cfg.long_term_weight) * lt
    return weights


@dataclass(slots=True)
class LADDConfig:
    """Weights for composite LADD-style objective."""

    mse_teacher: float = 1.0
    """Match student output to teacher output at same (x_t, t)."""

    adversarial: float = 0.1
    """Non-saturating GAN-style term on discriminator logits (student as generator)."""

    r1_gamma: float = 0.0
    """Optional R1 gradient penalty on real latents (0 = off)."""


class LatentPatchDiscriminator(nn.Module):
    """
    Lightweight conv discriminator on latent tensors ``(B, C, H, W)``.
    Outputs a **scalar logit** per batch (mean pools spatially).
    """

    def __init__(self, in_channels: int, base: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def softplus_loss_disc(real_logit: torch.Tensor, fake_logit: torch.Tensor) -> torch.Tensor:
    """Hinge-free logistic loss: softplus(-real) + softplus(fake)."""
    return F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()


def softplus_loss_gen(fake_logit: torch.Tensor) -> torch.Tensor:
    """Generator wants discriminator to think fake is real."""
    return F.softplus(-fake_logit).mean()


def teacher_student_mse(
    teacher: nn.Module, student: nn.Module, x_t: torch.Tensor, t: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Per-element MSE between teacher and student predictions (same kwargs for both).
    Typical kwargs: ``y`` (class), ``text_embeds``, etc., matching your model API.
    """
    with torch.no_grad():
        t_out = teacher(x_t, t, **kwargs)
    s_out = student(x_t, t, **kwargs)
    if t_out.shape != s_out.shape:
        raise ValueError(f"teacher {t_out.shape} vs student {s_out.shape}")
    return (s_out - t_out).pow(2).mean()


def ladd_discriminator_step(
    D: LatentPatchDiscriminator,
    opt_d: torch.optim.Optimizer,
    latents_real: torch.Tensor,
    latents_fake: torch.Tensor,
    *,
    cfg: LADDConfig,
) -> dict[str, float]:
    """
    One discriminator update. Pass **detached** fake latents (e.g. student-predicted clean latent or x0_hat).
    """
    latents_fake = latents_fake.detach()
    opt_d.zero_grad(set_to_none=True)
    real_l = D(latents_real)
    fake_l = D(latents_fake)
    loss = softplus_loss_disc(real_l, fake_l)
    if cfg.r1_gamma > 0.0:
        latents_real.requires_grad_(True)
        real_l2 = D(latents_real)
        grad_real = torch.autograd.grad(
            outputs=real_l2.sum(),
            inputs=latents_real,
            create_graph=True,
            only_inputs=True,
        )[0]
        r1 = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        loss = loss + 0.5 * cfg.r1_gamma * r1
    loss.backward()
    opt_d.step()
    return {"d_loss": float(loss.detach().cpu())}


def ladd_generator_step(
    D: LatentPatchDiscriminator,
    student: nn.Module,
    opt_g: torch.optim.Optimizer,
    x_t: torch.Tensor,
    t: torch.Tensor,
    teacher: nn.Module,
    *,
    cfg: LADDConfig,
    latent_for_d: torch.Tensor | None = None,
    **model_kw,
) -> dict[str, float]:
    """
    One student (generator) step: MSE to teacher + optional adversarial signal on ``latent_for_d``.

    If ``latent_for_d`` is None, adversarial term is skipped (distillation-only).
    ``latent_for_d`` should be a tensor D can score (e.g. predicted x0 or denoised latent), **not detached**.
    """
    opt_g.zero_grad(set_to_none=True)
    loss = cfg.mse_teacher * teacher_student_mse(teacher, student, x_t, t, **model_kw)
    stats: dict[str, float] = {"g_mse": float((loss / max(cfg.mse_teacher, 1e-8)).detach().cpu())}
    if cfg.adversarial > 0.0 and latent_for_d is not None:
        logit_fake = D(latent_for_d)
        adv = softplus_loss_gen(logit_fake)
        loss = loss + cfg.adversarial * adv
        stats["g_adv"] = float(adv.detach().cpu())
    loss.backward()
    opt_g.step()
    stats["g_total"] = float(loss.detach().cpu())
    return stats


EventKind = Literal["like", "dislike", "pair", "pick", "generate"]


def default_feedback_log() -> Path:
    return Path("outputs") / "feedback" / "feedback.jsonl"


@dataclass
class FeedbackEvent:
    event: EventKind
    prompt: str = ""
    image_path: str = ""
    win_image_path: str = ""
    lose_image_path: str = ""
    score: float | None = None
    ckpt: str = ""
    seed: int | None = None
    run_id: str = ""
    media_type: str = "image"
    source: str = "user"
    timestamp: float = field(default_factory=lambda: time.time())
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", {}) or {}
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in d or d[k] in (None, "", []):
                    d[k] = v
        # Drop empty optionals for compact JSONL
        return {k: v for k, v in d.items() if v is not None and v != "" and v != {}}


def append_feedback(
    event: FeedbackEvent | dict[str, Any],
    *,
    log_path: str | Path | None = None,
) -> Path:
    """Append one feedback event. Returns the log path."""
    path = Path(log_path) if log_path else default_feedback_log()
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(event, FeedbackEvent):
        row = event.to_dict()
    else:
        row = dict(event)
        row.setdefault("timestamp", time.time())
        row.setdefault("source", "user")
        row.setdefault("run_id", row.get("run_id") or uuid.uuid4().hex[:12])
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def iter_feedback(log_path: str | Path | None = None) -> Iterable[dict[str, Any]]:
    path = Path(log_path) if log_path else default_feedback_log()
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def record_like(
    image_path: str,
    *,
    prompt: str = "",
    score: float = 1.0,
    ckpt: str = "",
    seed: int | None = None,
    run_id: str = "",
    media_type: str = "image",
    log_path: str | Path | None = None,
    notes: str = "",
) -> Path:
    return append_feedback(
        FeedbackEvent(
            event="like",
            prompt=prompt,
            image_path=str(image_path),
            score=float(score),
            ckpt=ckpt,
            seed=seed,
            run_id=run_id or uuid.uuid4().hex[:12],
            media_type=media_type,
            notes=notes,
        ),
        log_path=log_path,
    )


def record_dislike(
    image_path: str,
    *,
    prompt: str = "",
    ckpt: str = "",
    seed: int | None = None,
    run_id: str = "",
    media_type: str = "image",
    log_path: str | Path | None = None,
    notes: str = "",
) -> Path:
    return append_feedback(
        FeedbackEvent(
            event="dislike",
            prompt=prompt,
            image_path=str(image_path),
            score=0.0,
            ckpt=ckpt,
            seed=seed,
            run_id=run_id or uuid.uuid4().hex[:12],
            media_type=media_type,
            notes=notes,
        ),
        log_path=log_path,
    )


def record_pair(
    win_image_path: str,
    lose_image_path: str,
    *,
    prompt: str = "",
    ckpt: str = "",
    seed: int | None = None,
    run_id: str = "",
    media_type: str = "image",
    log_path: str | Path | None = None,
    notes: str = "",
) -> Path:
    return append_feedback(
        FeedbackEvent(
            event="pair",
            prompt=prompt,
            win_image_path=str(win_image_path),
            lose_image_path=str(lose_image_path),
            image_path=str(win_image_path),
            ckpt=ckpt,
            seed=seed,
            run_id=run_id or uuid.uuid4().hex[:12],
            media_type=media_type,
            notes=notes,
        ),
        log_path=log_path,
    )


def record_pick(
    win_image_path: str,
    lose_paths: list[str],
    *,
    prompt: str = "",
    ckpt: str = "",
    seed: int | None = None,
    run_id: str = "",
    media_type: str = "image",
    log_path: str | Path | None = None,
) -> list[Path]:
    """User picked one winner among candidates → one pair per loser."""
    out: list[Path] = []
    rid = run_id or uuid.uuid4().hex[:12]
    for lose in lose_paths:
        if not lose or str(lose) == str(win_image_path):
            continue
        out.append(
            record_pair(
                win_image_path,
                lose,
                prompt=prompt,
                ckpt=ckpt,
                seed=seed,
                run_id=rid,
                media_type=media_type,
                log_path=log_path,
                notes="pick",
            )
        )
    # Also record a like on the winner for taste updates.
    record_like(
        win_image_path,
        prompt=prompt,
        ckpt=ckpt,
        seed=seed,
        run_id=rid,
        media_type=media_type,
        log_path=log_path,
        notes="pick_winner",
    )
    return out


def record_generation(
    image_path: str,
    *,
    prompt: str = "",
    ckpt: str = "",
    seed: int | None = None,
    run_id: str = "",
    media_type: str = "image",
    candidates: list[str] | None = None,
    log_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Provenance row after sample.py / video keyframe save (not a preference)."""
    return append_feedback(
        FeedbackEvent(
            event="generate",
            prompt=prompt,
            image_path=str(image_path),
            ckpt=ckpt,
            seed=seed,
            run_id=run_id or uuid.uuid4().hex[:12],
            media_type=media_type,
            extra={
                **(extra or {}),
                **({"candidates": list(candidates)} if candidates else {}),
            },
        ),
        log_path=log_path,
    )


def pairs_from_feedback(
    log_path: str | Path | None = None,
    *,
    same_prompt_only: bool = True,
    include_synthetic: bool = True,
    max_pairs_per_prompt: int = 8,
) -> list[dict[str, Any]]:
    """
    Convert feedback events into DPO rows ``{win_image_path, lose_image_path, caption, source}``.

    - Explicit ``pair`` / ``pick`` events → direct pairs (margin 0).
    - Optional synthetic: each like vs dislike with matching prompt.
    """
    likes: list[dict[str, Any]] = []
    dislikes: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    per_prompt: dict[str, int] = {}

    def _budget(prompt: str) -> bool:
        key = prompt or ""
        n = per_prompt.get(key, 0)
        if n >= max_pairs_per_prompt:
            return False
        per_prompt[key] = n + 1
        return True

    for row in iter_feedback(log_path):
        ev = str(row.get("event") or "").lower()
        prompt = str(row.get("prompt") or row.get("caption") or "").strip()
        if ev in ("pair", "pick"):
            win = str(row.get("win_image_path") or row.get("image_path") or "").strip()
            lose = str(row.get("lose_image_path") or "").strip()
            if win and lose and win != lose and _budget(prompt):
                pairs.append(
                    {
                        "win_image_path": win,
                        "lose_image_path": lose,
                        "caption": prompt,
                        "source": "user",
                        "weight": 1.5,
                        "margin": 0.0,
                    }
                )
            continue
        if ev == "like":
            likes.append(row)
        elif ev == "dislike":
            dislikes.append(row)

    if include_synthetic:
        for win in likes:
            wp = str(win.get("prompt") or "").strip()
            wimg = str(win.get("image_path") or "").strip()
            if not wimg:
                continue
            for lose in dislikes:
                lp = str(lose.get("prompt") or "").strip()
                limg = str(lose.get("image_path") or "").strip()
                if not limg or limg == wimg:
                    continue
                if same_prompt_only and wp and lp and wp != lp:
                    continue
                caption = wp or lp
                if not _budget(caption):
                    continue
                pairs.append(
                    {
                        "win_image_path": wimg,
                        "lose_image_path": limg,
                        "caption": caption,
                        "source": "user_synthetic",
                        "weight": 1.2,
                        "margin": 0.0,
                    }
                )
    return pairs


def export_dpo_jsonl(
    out_path: str | Path,
    *,
    log_path: str | Path | None = None,
    same_prompt_only: bool = True,
    include_synthetic: bool = True,
) -> int:
    pairs = pairs_from_feedback(
        log_path,
        same_prompt_only=same_prompt_only,
        include_synthetic=include_synthetic,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(pairs)


def update_user_taste_from_feedback(
    *,
    log_path: str | Path | None = None,
    taste_path: str | Path | None = None,
    max_notes: int = 64,
) -> Path | None:
    """Push recent like prompts into ``UserTaste.likes_notes`` / hates from dislikes."""
    try:
        from utils.generation.planners import UserTaste, default_taste_path, load_user_taste, save_user_taste
    except Exception:
        return None
    tp = Path(taste_path) if taste_path else default_taste_path()
    taste = load_user_taste(tp) if tp.is_file() else UserTaste()
    likes_notes = list(taste.likes_notes or [])
    hates = list(taste.hates or [])
    for row in iter_feedback(log_path):
        ev = str(row.get("event") or "").lower()
        prompt = str(row.get("prompt") or "").strip()
        notes = str(row.get("notes") or "").strip()
        if ev == "like" and prompt and prompt not in likes_notes:
            likes_notes.append(prompt[:240])
        if ev == "dislike":
            frag = notes or prompt
            if frag and frag not in hates:
                hates.append(frag[:160])
    taste.likes_notes = likes_notes[-max_notes:]
    taste.hates = hates[-max_notes:]
    taste.updated_at = time.time()
    save_user_taste(taste, tp)
    return tp
