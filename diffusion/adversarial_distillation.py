"""
Adversarial Diffusion Distillation (ADD) for SDX.

Distills a large teacher DiT into a fast student that generates in 1-4 steps.
Combines regression-to-teacher (prevents blurring) with adversarial training
(prevents mode collapse and keeps high-frequency sharpness).

Architecture:
- Teacher: frozen full-step DiT (your trained checkpoint)
- Student: same architecture, trained to match teacher in 1-4 steps
- Discriminator: multi-scale patch discriminator that operates on both
  decoded images AND intermediate DiT features (feature-level discrimination
  is much more powerful than pixel-level alone)

Why ADD beats pure consistency distillation:
- Pure regression blurs high frequencies (MSE averages modes)
- Pure GAN collapses to a few modes
- ADD combines both: regression keeps semantics, GAN keeps sharpness
- Feature-level discrimination catches structural errors that pixel-level misses

Training procedure:
1. Sample x0 from teacher (full denoising, frozen)
2. Student generates x0_student in K steps (K=1,2,4)
3. Discriminator loss: real=teacher output, fake=student output
4. Student loss: regression(student, teacher) + adversarial(student)
5. Alternate D and G updates

This is the technique behind SDXL-Turbo, FLUX-Schnell, and similar fast models.

Usage:
    teacher = load_model(teacher_ckpt)
    student = load_model(student_ckpt)  # same arch, different weights
    distiller = ADDDistiller(teacher, student, diffusion, device)
    distiller.train(dataloader, steps=50000)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-scale patch discriminator
# ---------------------------------------------------------------------------


class PatchDiscriminatorBlock(nn.Module):
    """Single scale of the multi-scale discriminator."""

    def __init__(self, in_channels: int, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, hidden, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = hidden
        for i in range(1, n_layers):
            ch_next = min(ch * 2, 512)
            stride = 2 if i < n_layers - 1 else 1
            layers += [
                nn.Conv2d(ch, ch_next, 4, stride=stride, padding=1, bias=False),
                nn.GroupNorm(min(32, ch_next), ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch_next
        layers.append(nn.Conv2d(ch, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScalePatchDiscriminator(nn.Module):
    """
    Multi-scale patch discriminator operating at 3 spatial scales.

    Operates on decoded RGB images. Each scale catches different artifacts:
    - Scale 1 (full res): fine texture, sharpness, grain
    - Scale 2 (1/2 res): mid-frequency structure, faces, hands
    - Scale 3 (1/4 res): global composition, color, layout

    Also accepts optional feature maps from the DiT for feature-level discrimination.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 64,
        n_scales: int = 3,
        n_layers: int = 3,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_scales = n_scales

        self.discriminators = nn.ModuleList(
            [PatchDiscriminatorBlock(in_channels, hidden, n_layers) for _ in range(n_scales)]
        )

        # Optional feature discriminator (operates on DiT intermediate features)
        self.feature_disc = None
        if feature_dim is not None:
            self.feature_disc = nn.Sequential(
                nn.Linear(feature_dim, hidden * 4),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden * 4, hidden * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden * 2, 1),
            )

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: RGB image (B, 3, H, W) in [-1, 1]
            features: Optional DiT features (B, N, D) for feature discrimination

        Returns:
            (predictions, intermediates) — predictions[i] is the i-th scale output
        """
        preds = []
        intermediates = []

        x_curr = x
        for disc in self.discriminators:
            pred = disc(x_curr)
            preds.append(pred)
            intermediates.append(x_curr)
            x_curr = self.downsample(x_curr)

        if features is not None and self.feature_disc is not None:
            # Pool features to a single vector and discriminate
            feat_pooled = features.mean(dim=1)  # (B, D)
            feat_pred = self.feature_disc(feat_pooled)
            preds.append(feat_pred)

        return preds, intermediates


# ---------------------------------------------------------------------------
# ADD loss functions
# ---------------------------------------------------------------------------


def hinge_d_loss(real_preds: List[torch.Tensor], fake_preds: List[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for discriminator: max(0, 1-real) + max(0, 1+fake)."""
    d_loss = torch.tensor(0.0, device=real_preds[0].device)
    for real, fake in zip(real_preds, fake_preds):
        d_loss = d_loss + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return d_loss / len(real_preds)


def hinge_g_loss(fake_preds: List[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for generator: -mean(fake)."""
    g_loss = torch.tensor(0.0, device=fake_preds[0].device)
    for fake in fake_preds:
        g_loss = g_loss - fake.mean()
    return g_loss / len(fake_preds)


def feature_matching_loss(
    real_intermediates: List[torch.Tensor],
    fake_intermediates: List[torch.Tensor],
) -> torch.Tensor:
    """
    Feature matching loss: L1 between discriminator intermediate features.
    Stabilizes GAN training by matching feature statistics, not just final predictions.
    """
    fm_loss = torch.tensor(0.0, device=real_intermediates[0].device)
    for real, fake in zip(real_intermediates, fake_intermediates):
        fm_loss = fm_loss + F.l1_loss(fake, real.detach())
    return fm_loss / len(real_intermediates)


def regression_loss(
    student_x0: torch.Tensor,
    teacher_x0: torch.Tensor,
    *,
    loss_type: str = "l2",
) -> torch.Tensor:
    """
    Regression loss between student and teacher outputs.

    l2: MSE (standard, can blur)
    l1: MAE (sharper edges, less sensitive to outliers)
    huber: Smooth L1 (compromise)
    lpips_proxy: Frequency-weighted MSE (approximates perceptual loss without LPIPS)
    """
    if loss_type == "l1":
        return F.l1_loss(student_x0, teacher_x0)
    elif loss_type == "huber":
        return F.smooth_l1_loss(student_x0, teacher_x0)
    elif loss_type == "lpips_proxy":
        # Frequency-weighted MSE: emphasize high frequencies (edges, texture)
        err = student_x0.float() - teacher_x0.float()
        fft = torch.fft.rfft2(err, dim=(-2, -1), norm="ortho")
        power = fft.real.pow(2) + fft.imag.pow(2)
        # Weight: higher weight for high frequencies
        h, w = err.shape[-2], err.shape[-1]
        fy = torch.fft.fftfreq(h, device=err.device).abs().view(h, 1)
        fx = torch.fft.rfftfreq(w, device=err.device).abs().view(1, -1)
        freq_weight = 1.0 + 3.0 * (fy + fx).clamp(0, 1)
        return (power * freq_weight).mean()
    else:  # l2
        return F.mse_loss(student_x0, teacher_x0)


# ---------------------------------------------------------------------------
# Student sampler: K-step generation
# ---------------------------------------------------------------------------


class StudentSampler:
    """
    K-step sampler for the student model.

    Uses a fixed timestep schedule optimized for few-step generation:
    - K=1: single step from t=T to t=0 (most aggressive)
    - K=2: two steps, split at t=T/2
    - K=4: four steps with cosine spacing (best quality/speed tradeoff)
    """

    def __init__(self, num_timesteps: int = 1000, k_steps: int = 4):
        self.T = int(num_timesteps)
        self.k = int(k_steps)

    def get_timesteps(self) -> List[int]:
        """Get the K timesteps for student generation."""
        if self.k == 1:
            return [self.T - 1]
        elif self.k == 2:
            return [self.T - 1, self.T // 2]
        else:
            # Cosine spacing: more steps at low noise (fine detail)
            import math

            steps = []
            for i in range(self.k):
                t_frac = 1.0 - i / (self.k - 1)
                t_cos = 0.5 * (1.0 + math.cos(math.pi * (1.0 - t_frac)))
                t_idx = int(t_cos * (self.T - 1))
                steps.append(max(0, min(self.T - 1, t_idx)))
            return steps

    @torch.no_grad()
    def sample_teacher(
        self,
        teacher: nn.Module,
        diffusion: Any,
        shape: Tuple[int, ...],
        model_kwargs: Dict[str, Any],
        device: torch.device,
        cfg_scale: float = 7.5,
        model_kwargs_uncond: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Generate a full-quality sample from the teacher (frozen)."""
        teacher.eval()
        x0 = diffusion.sample_loop(
            teacher,
            shape,
            model_kwargs_cond=model_kwargs,
            model_kwargs_uncond=model_kwargs_uncond,
            cfg_scale=cfg_scale,
            cfg_rescale=0.0,
            num_inference_steps=50,
            eta=0.0,
            device=device,
            dtype=torch.float32,
        )
        return x0

    def sample_student(
        self,
        student: nn.Module,
        diffusion: Any,
        shape: Tuple[int, ...],
        model_kwargs: Dict[str, Any],
        device: torch.device,
        cfg_scale: float = 7.5,
        model_kwargs_uncond: Optional[Dict[str, Any]] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate a K-step sample from the student."""
        student.eval()
        x0 = diffusion.sample_loop(
            student,
            shape,
            model_kwargs_cond=model_kwargs,
            model_kwargs_uncond=model_kwargs_uncond,
            cfg_scale=cfg_scale,
            cfg_rescale=0.0,
            num_inference_steps=self.k,
            eta=0.0,
            device=device,
            dtype=torch.float32,
        )
        return x0


# ---------------------------------------------------------------------------
# ADD Distiller: main training class
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ADDConfig:
    """Configuration for ADD distillation."""

    k_steps: int = 4  # Student inference steps (1, 2, or 4)
    teacher_cfg_scale: float = 7.5  # CFG for teacher generation
    student_cfg_scale: float = 7.5  # CFG for student generation
    regression_weight: float = 1.0  # Weight for regression loss
    adversarial_weight: float = 0.1  # Weight for adversarial loss
    feature_matching_weight: float = 10.0  # Weight for feature matching
    regression_loss_type: str = "lpips_proxy"  # "l1", "l2", "huber", "lpips_proxy"
    disc_lr: float = 1e-4  # Discriminator learning rate
    student_lr: float = 5e-5  # Student learning rate
    disc_hidden: int = 64  # Discriminator hidden channels
    disc_n_scales: int = 3  # Number of discriminator scales
    disc_n_layers: int = 3  # Layers per discriminator scale
    grad_clip: float = 1.0  # Gradient clipping
    disc_update_every: int = 1  # Update discriminator every N student steps
    warmup_steps: int = 1000  # Steps before adversarial loss kicks in
    latent_scale: float = 0.18215  # VAE latent scale
    log_every: int = 100
    save_every: int = 2000
    save_dir: str = "./add_checkpoints"


class ADDDistiller:
    """
    Adversarial Diffusion Distillation trainer.

    Distills a teacher DiT into a K-step student using combined
    regression + adversarial + feature matching losses.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        diffusion: Any,
        vae: nn.Module,
        device: torch.device,
        cfg: Optional[ADDConfig] = None,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.diffusion = diffusion
        self.vae = vae.to(device)
        self.device = device
        self.cfg = cfg or ADDConfig()

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        # Discriminator
        self.discriminator = MultiScalePatchDiscriminator(
            in_channels=3,
            hidden=self.cfg.disc_hidden,
            n_scales=self.cfg.disc_n_scales,
            n_layers=self.cfg.disc_n_layers,
        ).to(device)

        # Optimizers
        self.student_opt = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.cfg.student_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        self.disc_opt = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.cfg.disc_lr,
            betas=(0.9, 0.999),
        )

        self.sampler = StudentSampler(
            num_timesteps=getattr(diffusion, "num_timesteps", 1000),
            k_steps=self.cfg.k_steps,
        )

        self._step = 0

    def _decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to RGB image in [-1, 1]."""
        with torch.no_grad():
            z = latent / self.cfg.latent_scale
            img = self.vae.decode(z).sample
            return img.clamp(-1, 1)

    def train_step(
        self,
        model_kwargs_cond: Dict[str, Any],
        model_kwargs_uncond: Optional[Dict[str, Any]],
        shape: Tuple[int, ...],
    ) -> Dict[str, float]:
        """
        Single ADD training step.

        Returns dict of loss components.
        """
        # ---- Step 1: Generate teacher sample (frozen, no grad) ----
        with torch.no_grad():
            teacher_x0 = self.sampler.sample_teacher(
                self.teacher,
                self.diffusion,
                shape,
                model_kwargs_cond,
                self.device,
                cfg_scale=self.cfg.teacher_cfg_scale,
                model_kwargs_uncond=model_kwargs_uncond,
            )
            teacher_rgb = self._decode_latent(teacher_x0)

        # ---- Step 2: Generate student sample ----
        self.student.train()
        student_x0 = self.sampler.sample_student(
            self.student,
            self.diffusion,
            shape,
            model_kwargs_cond,
            self.device,
            cfg_scale=self.cfg.student_cfg_scale,
            model_kwargs_uncond=model_kwargs_uncond,
        )
        student_rgb = self._decode_latent(student_x0)

        # ---- Step 3: Update discriminator ----
        disc_loss_val = 0.0
        if self._step % self.cfg.disc_update_every == 0:
            self.disc_opt.zero_grad()

            real_preds, real_ints = self.discriminator(teacher_rgb.detach())
            fake_preds, _ = self.discriminator(student_rgb.detach())

            disc_loss = hinge_d_loss(real_preds, fake_preds)
            disc_loss.backward()
            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.grad_clip)
            self.disc_opt.step()
            disc_loss_val = float(disc_loss.item())

        # ---- Step 4: Update student ----
        self.student_opt.zero_grad()

        # Regression loss
        reg_loss = regression_loss(
            student_x0,
            teacher_x0.detach(),
            loss_type=self.cfg.regression_loss_type,
        )

        # Adversarial + feature matching (after warmup)
        adv_loss_val = 0.0
        fm_loss_val = 0.0
        total_loss = self.cfg.regression_weight * reg_loss

        if self._step >= self.cfg.warmup_steps:
            fake_preds_g, fake_ints_g = self.discriminator(student_rgb)
            real_preds_g, real_ints_g = self.discriminator(teacher_rgb.detach())

            adv_loss = hinge_g_loss(fake_preds_g)
            fm_loss = feature_matching_loss(real_ints_g, fake_ints_g)

            total_loss = (
                self.cfg.regression_weight * reg_loss
                + self.cfg.adversarial_weight * adv_loss
                + self.cfg.feature_matching_weight * fm_loss
            )
            adv_loss_val = float(adv_loss.item())
            fm_loss_val = float(fm_loss.item())

        total_loss.backward()
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.grad_clip)
        self.student_opt.step()

        self._step += 1

        return {
            "step": self._step,
            "regression_loss": float(reg_loss.item()),
            "adversarial_loss": adv_loss_val,
            "feature_matching_loss": fm_loss_val,
            "discriminator_loss": disc_loss_val,
            "total_loss": float(total_loss.item()),
        }

    def train(
        self,
        dataloader: Any,
        steps: int = 50000,
        encode_text_fn: Optional[Callable] = None,
    ) -> List[Dict[str, float]]:
        """
        Full ADD training loop.

        Args:
            dataloader: DataLoader yielding batches with 'captions' key
            steps: Total training steps
            encode_text_fn: Function(captions) → (cond_emb, uncond_emb)

        Returns:
            List of loss dicts per step
        """
        save_path = Path(self.cfg.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        history = []

        loader_iter = iter(dataloader)

        for step in range(steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                batch = next(loader_iter)

            captions = batch.get("captions", [""] * 4)
            B = len(captions)

            # Get text embeddings
            if encode_text_fn is not None:
                cond_emb, uncond_emb = encode_text_fn(captions)
                model_kwargs_cond = {"encoder_hidden_states": cond_emb}
                model_kwargs_uncond = {"encoder_hidden_states": uncond_emb}
            else:
                # Placeholder
                model_kwargs_cond = {}
                model_kwargs_uncond = None

            # Determine shape from batch
            pv = batch.get("pixel_values")
            if pv is not None:
                _, C, H, W = pv.shape
                latent_h, latent_w = H // 8, W // 8
            else:
                latent_h = latent_w = 32  # default 256px
            shape = (B, 4, latent_h, latent_w)

            losses = self.train_step(model_kwargs_cond, model_kwargs_uncond, shape)
            history.append(losses)

            if step % self.cfg.log_every == 0:
                _log.info(
                    f"ADD step {step}/{steps}: "
                    f"reg={losses['regression_loss']:.4f} "
                    f"adv={losses['adversarial_loss']:.4f} "
                    f"fm={losses['feature_matching_loss']:.4f} "
                    f"disc={losses['discriminator_loss']:.4f}"
                )

            if step % self.cfg.save_every == 0 and step > 0:
                ckpt = {
                    "student": self.student.state_dict(),
                    "discriminator": self.discriminator.state_dict(),
                    "student_opt": self.student_opt.state_dict(),
                    "disc_opt": self.disc_opt.state_dict(),
                    "step": step,
                    "config": self.cfg,
                }
                torch.save(ckpt, save_path / f"add_step_{step:06d}.pt")
                _log.info(f"Saved ADD checkpoint at step {step}")

        return history

    def export_student(self, output_path: str) -> None:
        """Export the distilled student model as a standard SDX checkpoint."""
        torch.save(
            {
                "model": self.student.state_dict(),
                "add_distilled": True,
                "k_steps": self.cfg.k_steps,
                "distillation_steps": self._step,
            },
            output_path,
        )
        _log.info(f"Exported distilled student to {output_path}")


__all__ = [
    "ADDDistiller",
    "ADDConfig",
    "MultiScalePatchDiscriminator",
    "StudentSampler",
    "regression_loss",
    "hinge_d_loss",
    "hinge_g_loss",
    "feature_matching_loss",
]
