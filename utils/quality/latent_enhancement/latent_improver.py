"""Latent space improvements for higher quality image generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentNormalization(nn.Module):
    """Improved latent normalization for better training stability."""

    def __init__(self, latent_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(latent_dim))
        self.bias = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize latent with learnable scale/bias."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps

        x_norm = (x - mean) / std
        x_norm = x_norm * self.scale + self.bias

        return x_norm


class LatentSharpening(nn.Module):
    """Sharpen latent features for crisper details."""

    def __init__(self, latent_dim: int, sharpness: float = 0.5):
        super().__init__()
        self.sharpness = sharpness
        self.filter = nn.Parameter(
            torch.tensor([[-0.25, -0.5, -0.25], [-0.5, 3.0, -0.5], [-0.25, -0.5, -0.25]]).unsqueeze(0).unsqueeze(0)
            / 3.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sharpening to latent maps."""
        if x.dim() == 4:
            sharpened = F.conv2d(x, self.filter.expand(x.shape[1], -1, -1, -1), padding=1, groups=x.shape[1])

            return x + self.sharpness * (sharpened - x)

        return x


class LatentChannelAttention(nn.Module):
    """Channel-wise attention for latent features."""

    def __init__(self, latent_dim: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim // reduction, latent_dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention."""
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class LatentDiffusionRegularization(nn.Module):
    """Regularize latent space to improve sample quality."""

    def __init__(self):
        pass

    def compute_regularity_loss(self, latents: torch.Tensor) -> torch.Tensor:
        """Compute latent space regularity loss.

        Encourages smooth transitions in latent space.
        """
        grad = torch.autograd.grad(
            outputs=latents.sum(), inputs=latents, create_graph=True, retain_graph=True
        )[0]

        regularity = (grad ** 2).mean()
        return regularity

    def compute_isotropy_loss(self, latents: torch.Tensor) -> torch.Tensor:
        """Encourage isotropic (uniform) latent space."""
        latents_norm = F.normalize(latents, dim=-1)

        cov = torch.mm(latents_norm.t(), latents_norm) / latents.shape[0]

        target = torch.eye(latents.shape[-1], device=latents.device) / latents.shape[-1]

        isotropy_loss = F.mse_loss(cov, target)
        return isotropy_loss


class AdaptiveLatentScaling:
    """Adaptively scale latents based on content."""

    def __init__(self, num_scales: int = 4):
        self.num_scales = num_scales
        self.scale_predictors = nn.ModuleList(
            [nn.Linear(4, 1) for _ in range(num_scales)]
        )

    def predict_scales(self, latents: torch.Tensor) -> torch.Tensor:
        """Predict optimal scales per sample."""
        stats = torch.cat(
            [
                latents.mean(dim=(2, 3)),
                latents.std(dim=(2, 3)),
                latents.min(dim=2)[0].min(dim=2)[0],
                latents.max(dim=2)[0].max(dim=2)[0],
            ],
            dim=-1,
        )

        scales = [predictor(stats) for predictor in self.scale_predictors]
        return torch.cat(scales, dim=-1)

    def apply_adaptive_scaling(self, latents: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Apply adaptive scaling to latents."""
        scales = scales.view(scales.shape[0], -1, 1, 1)
        return latents * scales


class LatentPerturbation:
    """Controlled perturbation for better diversity."""

    def __init__(self, strength: float = 0.1):
        self.strength = strength

    def perturb(self, latents: torch.Tensor, perturbation_type: str = "gaussian") -> torch.Tensor:
        """Apply controlled perturbation."""
        if perturbation_type == "gaussian":
            noise = torch.randn_like(latents) * self.strength
        elif perturbation_type == "uniform":
            noise = (torch.rand_like(latents) - 0.5) * 2 * self.strength
        else:
            noise = 0

        return latents + noise


class LatentContrastiveSharpening:
    """Use contrastive learning to sharpen latent features."""

    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature

    def compute_sharpening_loss(self, z: torch.Tensor, z_aug: torch.Tensor) -> torch.Tensor:
        """Sharpen by pulling same-sample augmentations together."""
        z = F.normalize(z, dim=-1)
        z_aug = F.normalize(z_aug, dim=-1)

        similarity = torch.mm(z, z_aug.t()) / self.temperature

        labels = torch.arange(z.shape[0], device=z.device)
        loss = F.cross_entropy(similarity, labels)

        return loss


class LatentMixing:
    """Mix latents for style/content disentanglement."""

    @staticmethod
    def mix_latents(z1: torch.Tensor, z2: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """Mix two latent samples."""
        return alpha * z1 + (1 - alpha) * z2

    @staticmethod
    def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float = 0.5) -> torch.Tensor:
        """Spherical linear interpolation for smoother transitions."""
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)

        dot = (z1_norm * z2_norm).sum(dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)

        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        w1 = torch.sin((1.0 - t) * theta) / sin_theta
        w2 = torch.sin(t * theta) / sin_theta

        result = w1 * z1 + w2 * z2
        return result

    @staticmethod
    def style_content_mix(z_style: torch.Tensor, z_content: torch.Tensor, style_weight: float = 0.5) -> torch.Tensor:
        """Mix style and content latents."""
        z_style_stats = torch.cat([z_style.mean(dim=-1), z_style.std(dim=-1)], dim=-1)
        z_content_norm = F.normalize(z_content, dim=-1)

        mixed = z_content_norm * (1.0 - style_weight) + F.normalize(z_style, dim=-1) * style_weight
        return mixed
