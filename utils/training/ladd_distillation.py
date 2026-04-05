"""
LADD-style **latent adversarial diffusion distillation** scaffolding (Flux Schnell family).

This is **not** a drop-in trainer: it provides **loss pieces** you can call from a custom loop
where you have a **teacher** and **student** denoiser (same latent shape) and optionally a
**discriminator** on latents or predicted x0.

References: Latent Adversarial Diffusion Distillation (LADD); adversarial + distillation on
denoising predictions. Hook your DiT / UNet ``forward(x, t, **kw)`` here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
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


def teacher_student_mse(teacher: nn.Module, student: nn.Module, x_t: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
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
) -> Dict[str, float]:
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
    latent_for_d: Optional[torch.Tensor] = None,
    **model_kw,
) -> Dict[str, float]:
    """
    One student (generator) step: MSE to teacher + optional adversarial signal on ``latent_for_d``.

    If ``latent_for_d`` is None, adversarial term is skipped (distillation-only).
    ``latent_for_d`` should be a tensor D can score (e.g. predicted x0 or denoised latent), **not detached**.
    """
    opt_g.zero_grad(set_to_none=True)
    loss = cfg.mse_teacher * teacher_student_mse(teacher, student, x_t, t, **model_kw)
    stats: Dict[str, float] = {"g_mse": float((loss / max(cfg.mse_teacher, 1e-8)).detach().cpu())}
    if cfg.adversarial > 0.0 and latent_for_d is not None:
        logit_fake = D(latent_for_d)
        adv = softplus_loss_gen(logit_fake)
        loss = loss + cfg.adversarial * adv
        stats["g_adv"] = float(adv.detach().cpu())
    loss.backward()
    opt_g.step()
    stats["g_total"] = float(loss.detach().cpu())
    return stats


def estimate_x0_from_eps(x_t: torch.Tensor, eps_hat: torch.Tensor, alphabar_t: torch.Tensor) -> torch.Tensor:
    """
    VP-DDMP-style x0 estimate: ``x0 = (x_t - sqrt(1-ab) * eps) / sqrt(ab)``.
    ``alphabar_t`` shape ``(B,1,1,1)`` or broadcastable.
    """
    ab = alphabar_t
    return (x_t - torch.sqrt(1.0 - ab) * eps_hat) / torch.sqrt(ab.clamp(min=1e-8))
