"""
Auxiliary structure supervision for DiT training.

Adds depth, edge, and surface normal prediction heads to the DiT denoiser.
These heads are trained with auxiliary losses that force the model to learn
geometric structure explicitly — not just as a side effect of pixel prediction.

Why this matters:
- Standard diffusion learns geometry implicitly from pixel statistics
- Explicit structure supervision forces the model to "understand" depth,
  edges, and normals as first-class concepts
- This dramatically improves: spatial reasoning, object placement, perspective,
  "X in front of Y" relationships, and lighting consistency
- The heads are DISCARDED at inference — they only improve the backbone's
  internal representations during training

Architecture:
- Lightweight prediction heads attached to intermediate DiT features
- Depth head: predicts relative depth map (0=near, 1=far)
- Edge head: predicts edge magnitude map
- Normal head: predicts surface normal map (3-channel, normalized)
- Segmentation head: predicts coarse semantic regions

Supervision sources (choose based on your data):
1. Marigold depth (pretrained/Marigold-Depth-v1-1) — pseudo-labels from RGB
2. Marigold normals (pretrained/Marigold-Normals-v1-1) — pseudo-labels
3. Sobel edges — computed directly from images, no model needed
4. SAM2 segments (pretrained/SAM2-Hiera-Large) — coarse segmentation
5. GroundingDINO (pretrained/GroundingDINO-Base) — object bounding boxes

Usage in train.py:
    aux_supervisor = AuxiliaryStructureSupervisor(
        hidden_size=cfg.hidden_size,
        use_depth=True,
        use_edges=True,
        use_normals=False,  # requires Marigold normals
    )
    # In training loop:
    aux_loss = aux_supervisor.compute_loss(
        dit_features=intermediate_features,
        images=pixel_values,
        t=timesteps,
    )
    total_loss = denoise_loss + cfg.aux_structure_weight * aux_loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)
_PRETRAINED = Path(__file__).resolve().parents[2] / "pretrained"


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------


class DepthHead(nn.Module):
    """
    Lightweight depth prediction head.
    Input: (B, N, D) DiT features → Output: (B, 1, H, W) depth map in [0, 1]
    """

    def __init__(self, hidden_size: int, patch_size: int = 2, latent_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.n_patches = (latent_size // patch_size) ** 2
        self.h_patches = latent_size // patch_size

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, patch_size * patch_size),  # one depth value per pixel
            nn.Sigmoid(),  # depth in [0, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, D) — N = n_patches, D = hidden_size
        Returns:
            depth: (B, 1, H, W) where H=W=latent_size
        """
        B, N, D = features.shape
        h = self.h_patches
        p = self.patch_size

        # Predict per-patch depth values
        depth_patches = self.head(features)  # (B, N, p*p)
        depth_patches = depth_patches.view(B, h, h, p, p)
        # Rearrange to (B, 1, H, W)
        depth = depth_patches.permute(0, 1, 3, 2, 4).contiguous()
        depth = depth.view(B, 1, h * p, h * p)
        return depth


class EdgeHead(nn.Module):
    """
    Edge magnitude prediction head.
    Input: (B, N, D) DiT features → Output: (B, 1, H, W) edge map in [0, 1]
    """

    def __init__(self, hidden_size: int, patch_size: int = 2, latent_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.h_patches = latent_size // patch_size

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, patch_size * patch_size),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N, D = features.shape
        h = self.h_patches
        p = self.patch_size
        edge_patches = self.head(features)  # (B, N, p*p)
        edge_patches = edge_patches.view(B, h, h, p, p)
        edge = edge_patches.permute(0, 1, 3, 2, 4).contiguous()
        return edge.view(B, 1, h * p, h * p)


class NormalHead(nn.Module):
    """
    Surface normal prediction head.
    Input: (B, N, D) DiT features → Output: (B, 3, H, W) unit normals
    """

    def __init__(self, hidden_size: int, patch_size: int = 2, latent_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.h_patches = latent_size // patch_size

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 3 * patch_size * patch_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N, D = features.shape
        h = self.h_patches
        p = self.patch_size
        normal_patches = self.head(features)  # (B, N, 3*p*p)
        normal_patches = normal_patches.view(B, h, h, 3, p, p)
        normal = normal_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        normal = normal.view(B, 3, h * p, h * p)
        # Normalize to unit vectors
        normal = F.normalize(normal, dim=1, eps=1e-8)
        return normal


class SegmentationHead(nn.Module):
    """
    Coarse semantic segmentation head (foreground/background/sky/ground).
    Input: (B, N, D) DiT features → Output: (B, n_classes, H, W)
    """

    def __init__(
        self,
        hidden_size: int,
        n_classes: int = 4,
        patch_size: int = 2,
        latent_size: int = 32,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.h_patches = latent_size // patch_size

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, n_classes * patch_size * patch_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, N, D = features.shape
        h = self.h_patches
        p = self.patch_size
        C = self.n_classes
        seg_patches = self.head(features)  # (B, N, C*p*p)
        seg_patches = seg_patches.view(B, h, h, C, p, p)
        seg = seg_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return seg.view(B, C, h * p, h * p)


# ---------------------------------------------------------------------------
# Pseudo-label generators (no training needed)
# ---------------------------------------------------------------------------


def compute_sobel_edges(images_rgb: torch.Tensor) -> torch.Tensor:
    """
    Compute Sobel edge magnitude from RGB images.
    No external model needed — pure PyTorch.

    Args:
        images_rgb: (B, 3, H, W) in [-1, 1] or [0, 1]
    Returns:
        edges: (B, 1, H, W) in [0, 1]
    """
    # Convert to grayscale
    gray = 0.299 * images_rgb[:, 0:1] + 0.587 * images_rgb[:, 1:2] + 0.114 * images_rgb[:, 2:3]

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=images_rgb.dtype, device=images_rgb.device).view(
        1, 1, 3, 3
    )
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=images_rgb.dtype, device=images_rgb.device).view(
        1, 1, 3, 3
    )

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)

    # Normalize to [0, 1]
    magnitude = magnitude / (magnitude.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return magnitude.clamp(0, 1)


def compute_depth_pseudo_labels(
    images_rgb: torch.Tensor,
    device: torch.device,
    target_size: Tuple[int, int] = (64, 64),
) -> Optional[torch.Tensor]:
    """
    Generate depth pseudo-labels using Marigold (pretrained/Marigold-Depth-v1-1).
    Falls back to a simple vertical gradient prior if Marigold is unavailable.

    Args:
        images_rgb: (B, 3, H, W) in [-1, 1]
        device: Target device
        target_size: Output size (H, W)

    Returns:
        depth: (B, 1, H, W) in [0, 1] or None if unavailable
    """
    marigold_path = _PRETRAINED / "Marigold-Depth-v1-1"

    if marigold_path.exists():
        try:
            from diffusers import MarigoldDepthPipeline
            from PIL import Image

            pipe = MarigoldDepthPipeline.from_pretrained(
                str(marigold_path),
                torch_dtype=torch.float16,
            ).to(device)
            pipe.set_progress_bar_config(disable=True)

            B = images_rgb.shape[0]
            depths = []
            for i in range(B):
                img_np = ((images_rgb[i].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
                pil = Image.fromarray(img_np)
                with torch.no_grad():
                    result = pipe(pil, num_inference_steps=4, ensemble_size=1)
                depth_np = result.prediction[0]
                depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
                depth_t = F.interpolate(depth_t, size=target_size, mode="bilinear", align_corners=False)
                depths.append(depth_t)

            del pipe
            return torch.cat(depths, dim=0).to(device)
        except Exception as e:
            _log.debug(f"Marigold depth failed: {e}")

    # Fallback: vertical gradient prior (objects tend to be higher = farther)
    B, _, H, W = images_rgb.shape
    th, tw = target_size
    y = torch.linspace(0, 1, th, device=device)
    depth_prior = y.view(1, 1, th, 1).expand(B, 1, th, tw)
    return depth_prior


def compute_normal_pseudo_labels(
    images_rgb: torch.Tensor,
    device: torch.device,
    target_size: Tuple[int, int] = (64, 64),
) -> Optional[torch.Tensor]:
    """
    Generate surface normal pseudo-labels using Marigold normals.
    Falls back to None if unavailable.
    """
    normals_path = _PRETRAINED / "Marigold-Normals-v1-1"
    if not normals_path.exists():
        return None

    try:
        from diffusers import MarigoldNormalsPipeline
        from PIL import Image

        pipe = MarigoldNormalsPipeline.from_pretrained(
            str(normals_path),
            torch_dtype=torch.float16,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        B = images_rgb.shape[0]
        normals = []
        for i in range(B):
            img_np = ((images_rgb[i].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            pil = Image.fromarray(img_np)
            with torch.no_grad():
                result = pipe(pil, num_inference_steps=4, ensemble_size=1)
            normal_np = result.prediction[0]  # (H, W, 3) in [-1, 1]
            normal_t = torch.from_numpy(normal_np).permute(2, 0, 1).unsqueeze(0)
            normal_t = F.interpolate(normal_t, size=target_size, mode="bilinear", align_corners=False)
            normal_t = F.normalize(normal_t, dim=1)
            normals.append(normal_t)

        del pipe
        return torch.cat(normals, dim=0).to(device)
    except Exception as e:
        _log.debug(f"Marigold normals failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Timestep-aware loss weighting
# ---------------------------------------------------------------------------


def structure_loss_weight(t: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """
    Weight structure losses by timestep.

    Structure supervision is most useful at HIGH noise (large t) where the model
    is establishing coarse layout. At low noise (small t), the model is refining
    details and structure is already determined.

    Returns per-sample weights in [0, 1].
    """
    t_norm = t.float() / max(num_timesteps - 1, 1)
    # Sigmoid ramp: high weight for t > 0.3, low weight for t < 0.1
    weight = torch.sigmoid((t_norm - 0.2) * 10.0)
    return weight.view(-1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Main supervisor class
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AuxStructureConfig:
    """Configuration for auxiliary structure supervision."""

    use_depth: bool = True
    use_edges: bool = True
    use_normals: bool = False  # Requires Marigold normals
    use_segmentation: bool = False  # Requires SAM2 or similar
    depth_weight: float = 0.5
    edge_weight: float = 0.3
    normal_weight: float = 0.2
    seg_weight: float = 0.1
    patch_size: int = 2
    latent_size: int = 32  # latent spatial size (image_size // 8)
    n_seg_classes: int = 4
    use_pseudo_labels: bool = True  # Generate labels from pretrained models
    pseudo_label_every: int = 10  # Regenerate pseudo-labels every N steps
    timestep_weighting: bool = True  # Weight loss by timestep


class AuxiliaryStructureSupervisor:
    """
    Auxiliary structure supervision for DiT training.

    Attaches lightweight prediction heads to DiT intermediate features
    and trains them with depth/edge/normal supervision.

    The heads are ONLY used during training — they're discarded at inference.
    Their purpose is to force the backbone to learn geometric representations.

    Usage:
        supervisor = AuxiliaryStructureSupervisor(hidden_size=1152)
        # In training loop:
        aux_loss, aux_metrics = supervisor.compute_loss(
            features=dit_intermediate_features,  # (B, N, D)
            images=pixel_values,                 # (B, 3, H, W)
            t=timesteps,                         # (B,)
        )
        total_loss = denoise_loss + 0.1 * aux_loss
    """

    def __init__(
        self,
        hidden_size: int,
        cfg: Optional[AuxStructureConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.hidden_size = hidden_size
        self.cfg = cfg or AuxStructureConfig()
        self.device = device or torch.device("cpu")

        p = self.cfg.patch_size
        ls = self.cfg.latent_size

        # Build heads
        self.heads = nn.ModuleDict()
        if self.cfg.use_depth:
            self.heads["depth"] = DepthHead(hidden_size, p, ls)
        if self.cfg.use_edges:
            self.heads["edges"] = EdgeHead(hidden_size, p, ls)
        if self.cfg.use_normals:
            self.heads["normals"] = NormalHead(hidden_size, p, ls)
        if self.cfg.use_segmentation:
            self.heads["seg"] = SegmentationHead(hidden_size, self.cfg.n_seg_classes, p, ls)

        # Move heads to device
        for head in self.heads.values():
            head.to(self.device)

        self._pseudo_label_cache: Dict[str, torch.Tensor] = {}
        self._step = 0

    def parameters(self):
        """Return head parameters for optimizer."""
        for head in self.heads.values():
            yield from head.parameters()

    def _get_pseudo_labels(
        self,
        images: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Get or compute pseudo-labels for the current batch."""
        labels = {}

        # Edges: always compute (no external model needed)
        if self.cfg.use_edges:
            edges = compute_sobel_edges(images)
            edges = F.interpolate(edges, size=target_size, mode="bilinear", align_corners=False)
            labels["edges"] = edges

        # Depth: use Marigold or fallback
        if self.cfg.use_depth and self.cfg.use_pseudo_labels:
            if self._step % self.cfg.pseudo_label_every == 0:
                depth = compute_depth_pseudo_labels(images, self.device, target_size)
                if depth is not None:
                    self._pseudo_label_cache["depth"] = depth
            if "depth" in self._pseudo_label_cache:
                labels["depth"] = self._pseudo_label_cache["depth"]

        # Normals: use Marigold normals
        if self.cfg.use_normals and self.cfg.use_pseudo_labels:
            if self._step % self.cfg.pseudo_label_every == 0:
                normals = compute_normal_pseudo_labels(images, self.device, target_size)
                if normals is not None:
                    self._pseudo_label_cache["normals"] = normals
            if "normals" in self._pseudo_label_cache:
                labels["normals"] = self._pseudo_label_cache["normals"]

        return labels

    def compute_loss(
        self,
        features: torch.Tensor,
        images: torch.Tensor,
        t: torch.Tensor,
        num_timesteps: int = 1000,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute auxiliary structure supervision loss.

        Args:
            features: DiT intermediate features (B, N, D)
            images: Input images (B, 3, H, W) in [-1, 1]
            t: Timestep indices (B,)
            num_timesteps: Total diffusion timesteps

        Returns:
            (total_aux_loss, metrics_dict)
        """
        B, N, D = features.shape
        target_h = target_w = self.cfg.latent_size
        target_size = (target_h, target_w)

        # Get pseudo-labels
        labels = self._get_pseudo_labels(images, target_size)

        # Timestep weight
        if self.cfg.timestep_weighting:
            t_weight = structure_loss_weight(t, num_timesteps)
        else:
            t_weight = torch.ones(B, 1, 1, 1, device=features.device)

        total_loss = torch.tensor(0.0, device=features.device)
        metrics = {}

        # Depth loss
        if "depth" in self.heads and "depth" in labels:
            pred_depth = self.heads["depth"](features)
            target_depth = labels["depth"].to(features.device)
            if pred_depth.shape != target_depth.shape:
                target_depth = F.interpolate(
                    target_depth, size=pred_depth.shape[-2:], mode="bilinear", align_corners=False
                )
            depth_loss = (F.l1_loss(pred_depth, target_depth, reduction="none") * t_weight).mean()
            total_loss = total_loss + self.cfg.depth_weight * depth_loss
            metrics["depth_loss"] = float(depth_loss.item())

        # Edge loss
        if "edges" in self.heads and "edges" in labels:
            pred_edges = self.heads["edges"](features)
            target_edges = labels["edges"].to(features.device)
            if pred_edges.shape != target_edges.shape:
                target_edges = F.interpolate(
                    target_edges, size=pred_edges.shape[-2:], mode="bilinear", align_corners=False
                )
            edge_loss = (F.binary_cross_entropy(pred_edges, target_edges, reduction="none") * t_weight).mean()
            total_loss = total_loss + self.cfg.edge_weight * edge_loss
            metrics["edge_loss"] = float(edge_loss.item())

        # Normal loss
        if "normals" in self.heads and "normals" in labels:
            pred_normals = self.heads["normals"](features)
            target_normals = labels["normals"].to(features.device)
            if pred_normals.shape != target_normals.shape:
                target_normals = F.interpolate(
                    target_normals, size=pred_normals.shape[-2:], mode="bilinear", align_corners=False
                )
            # Angular loss: 1 - cosine similarity
            cos_sim = (pred_normals * target_normals).sum(dim=1, keepdim=True)
            normal_loss = ((1.0 - cos_sim) * t_weight).mean()
            total_loss = total_loss + self.cfg.normal_weight * normal_loss
            metrics["normal_loss"] = float(normal_loss.item())

        metrics["total_aux_loss"] = float(total_loss.item())
        self._step += 1

        return total_loss, metrics

    def state_dict(self) -> Dict[str, Any]:
        """Save head weights."""
        return {k: v.state_dict() for k, v in self.heads.items()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load head weights."""
        for k, v in state.items():
            if k in self.heads:
                self.heads[k].load_state_dict(v)


__all__ = [
    "AuxiliaryStructureSupervisor",
    "AuxStructureConfig",
    "DepthHead",
    "EdgeHead",
    "NormalHead",
    "SegmentationHead",
    "compute_sobel_edges",
    "compute_depth_pseudo_labels",
    "structure_loss_weight",
]
