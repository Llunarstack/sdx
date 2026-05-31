"""
Advanced **Diffusion-DPO** helpers inspired by 2025–2026 alignment research.

- Timestep-aware weighting (upweight high-noise steps; compensates signal attenuation)
- Safeguarded DPO margin (SDPO-style; caps destructive loser pushes)
- EMA reference model updates (dynamic ref vs frozen ref)

References: Rethinking DPO in Diffusion (AAAI 2026 oral); Diffusion-SDPO (safeguarded updates).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


__all__ = [
    "ema_update_reference",
    "safeguard_dpo_margins",
    "safeguarded_dpo_preference_loss",
    "timestep_dpo_weight",
]
