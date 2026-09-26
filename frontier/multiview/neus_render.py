"""
NeuS-style volumetric rendering + photometric multi-view training.

Why this brick
--------------
``fit_sdf`` teaches the field from an *analytic* SDF — great for a proof, useless
for real data, because real supervision is **images**, not distance functions.
This module closes that gap using the NeuS recipe (Wang et al. 2021), which is the
method behind modern image-to-3D pipelines (e.g. Stable-Fast-3D predicts a
tri-plane and supervises it exactly this way):

  1. March samples along each camera ray.
  2. Convert the SDF at each sample into an opacity with an *unbiased,
     occlusion-aware* logistic "S-density" (a sigmoid with a learnable sharpness
     ``inv_s`` that rises during training to crisp-up the surface).
  3. Alpha-composite the per-sample colors to get a rendered pixel.
  4. Minimize the difference between rendered and ground-truth pixels
     (photometric loss), plus a silhouette/mask loss and the eikonal regularizer.

Because every ray reads the one shared tri-plane field, the views stay mutually
consistent for free — the whole point of SDX-Spatial. Colors here are the PBR
albedo (Lambertian); view-dependent radiance is a later refinement.

References: NeuS (arxiv 2106.10689); Stable-Fast-3D (tri-plane + differentiable
marching tetrahedra + PBR).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .training import eikonal_loss, sample_points, sdf_and_gradient
from .triplane import SpatialLatentField

PointFn = Callable[[Tensor], Tensor]  # (R, S, 3) -> (R, S, C)


def look_at_rays(
    cam_pos: Tensor,
    target: Tensor,
    *,
    fov_deg: float = 45.0,
    height: int = 32,
    width: int = 32,
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> tuple[Tensor, Tensor]:
    """Build pinhole camera rays looking from ``cam_pos`` at ``target``.

    Returns ``(rays_o, rays_d)`` each ``(height*width, 3)``; ``rays_d`` is unit.
    """
    device = cam_pos.device
    forward = F.normalize(target - cam_pos, dim=-1)
    up_t = torch.tensor(up, dtype=cam_pos.dtype, device=device)
    right = F.normalize(torch.cross(forward, up_t, dim=-1), dim=-1)
    true_up = torch.cross(right, forward, dim=-1)

    half = torch.tan(torch.deg2rad(torch.tensor(fov_deg, device=device)) * 0.5)
    aspect = width / height
    ys, xs = torch.meshgrid(
        torch.linspace(1, -1, height, device=device) * half,
        torch.linspace(-1, 1, width, device=device) * half * aspect,
        indexing="ij",
    )
    dirs = forward + xs.reshape(-1, 1) * right + ys.reshape(-1, 1) * true_up
    dirs = F.normalize(dirs, dim=-1)
    origins = cam_pos.expand_as(dirs).contiguous()
    return origins, dirs


class SingleVariance(nn.Module):
    """NeuS learnable surface sharpness: ``inv_s = exp(10 * variance)``.

    Small at the start (fuzzy surface, smooth gradients) and grows during training
    to sharpen the zero level set.
    """

    def __init__(self, init: float = 0.3) -> None:
        super().__init__()
        self.variance = nn.Parameter(torch.tensor(init))

    def forward(self) -> Tensor:
        return torch.exp(self.variance * 10.0).clamp(1e-6, 1e6)


def volume_render_sdf(
    sdf_fn: PointFn,
    color_fn: PointFn,
    rays_o: Tensor,
    rays_d: Tensor,
    inv_s: Tensor,
    *,
    near: float = 0.1,
    far: float = 4.0,
    n_samples: int = 64,
    jitter: bool = False,
) -> dict[str, Tensor]:
    """Render rays through an SDF with NeuS alpha compositing.

    Args:
        sdf_fn: ``(R, S, 3) -> (R, S)`` or ``(R, S, 1)`` signed distances.
        color_fn: ``(R, S, 3) -> (R, S, 3)`` radiance (albedo) per sample.
        rays_o, rays_d: ``(R, 3)`` ray origins and unit directions.
        inv_s: scalar surface sharpness (from :class:`SingleVariance`).

    Returns dict with ``rgb`` (R,3), ``depth`` (R,), ``opacity`` (R,),
    ``weights`` (R, S-1), ``points`` (R, S, 3), ``sdf`` (R, S).
    """
    r = rays_o.shape[0]
    t = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)
    t = near + (far - near) * t
    t = t.expand(r, n_samples).clone()
    if jitter:
        mid = 0.5 * (t[:, 1:] + t[:, :-1])
        upper = torch.cat([mid, t[:, -1:]], -1)
        lower = torch.cat([t[:, :1], mid], -1)
        t = lower + (upper - lower) * torch.rand_like(t)

    pts = rays_o[:, None, :] + t[:, :, None] * rays_d[:, None, :]  # (R, S, 3)
    sdf = sdf_fn(pts).reshape(r, n_samples)
    colors = color_fn(pts)  # (R, S, 3)

    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_cdf = torch.sigmoid(prev_sdf * inv_s)
    next_cdf = torch.sigmoid(next_sdf * inv_s)
    # NeuS discrete opacity: fraction of the logistic CDF crossed over the segment.
    alpha = ((prev_cdf - next_cdf) / (prev_cdf + 1e-5)).clamp(0.0, 1.0)  # (R, S-1)

    trans = torch.cumprod(torch.cat([torch.ones(r, 1, device=alpha.device), 1.0 - alpha + 1e-7], dim=1), dim=1)[:, :-1]
    weights = alpha * trans  # (R, S-1)

    t_mid = 0.5 * (t[:, :-1] + t[:, 1:])
    seg_color = 0.5 * (colors[:, :-1] + colors[:, 1:])
    rgb = torch.sum(weights[:, :, None] * seg_color, dim=1)
    depth = torch.sum(weights * t_mid, dim=1)
    opacity = torch.sum(weights, dim=1)
    return {
        "rgb": rgb,
        "depth": depth,
        "opacity": opacity,
        "weights": weights,
        "points": pts,
        "sdf": sdf,
    }


class NeusRenderer(nn.Module):
    """Volume-render a :class:`SpatialLatentField` from arbitrary cameras.

    Pass a :class:`~frontier.multiview.radiance_sh.SHRadianceDecoder` to make the
    color **view-dependent**: each sample's radiance is then a function of both its
    features and the ray direction, so the field can learn specular highlights and
    reflections (metal, glass, wet surfaces) that plain albedo cannot express. With
    no decoder it falls back to the view-independent PBR albedo.
    """

    def __init__(self, n_samples: int = 64, sh_decoder: nn.Module | None = None) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.deviation = SingleVariance()
        self.sh_decoder = sh_decoder

    def render(
        self,
        field: SpatialLatentField,
        planes: tuple[Tensor, Tensor, Tensor],
        rays_o: Tensor,
        rays_d: Tensor,
        *,
        near: float = 0.1,
        far: float = 4.0,
        jitter: bool = False,
    ) -> dict[str, Tensor]:
        single = tuple(p[:1] for p in planes)

        def sdf_fn(p: Tensor) -> Tensor:
            flat = p.reshape(1, -1, 3)
            return field.decode_sdf(single, flat).reshape(p.shape[:-1])

        if self.sh_decoder is None:

            def color_fn(p: Tensor) -> Tensor:
                flat = p.reshape(1, -1, 3)
                return field.decode_pbr(single, flat).albedo.reshape(*p.shape[:-1], 3)
        else:

            def color_fn(p: Tensor) -> Tensor:
                flat = p.reshape(1, -1, 3)
                feat = field.query(single, flat).reshape(*p.shape[:-1], -1)  # (R, S, F)
                view = rays_d[:, None, :].expand(p.shape[0], p.shape[1], 3)  # (R, S, 3)
                return self.sh_decoder(feat, view)

        return volume_render_sdf(
            sdf_fn,
            color_fn,
            rays_o,
            rays_d,
            self.deviation(),
            near=near,
            far=far,
            n_samples=self.n_samples,
            jitter=jitter,
        )


def photometric_loss(rendered_rgb: Tensor, gt_rgb: Tensor) -> Tensor:
    """L1 between rendered and ground-truth pixels (robust to outliers)."""
    return (rendered_rgb - gt_rgb).abs().mean()


def mask_loss(opacity: Tensor, gt_mask: Tensor) -> Tensor:
    """Binary cross-entropy pushing accumulated opacity toward the silhouette."""
    return F.binary_cross_entropy(opacity.clamp(1e-4, 1 - 1e-4), gt_mask)


class View(nn.Module):
    """A single supervised camera: rays + ground-truth pixels (and optional mask)."""

    def __init__(self, rays_o: Tensor, rays_d: Tensor, rgb: Tensor, mask: Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("rays_o", rays_o)
        self.register_buffer("rays_d", rays_d)
        self.register_buffer("rgb", rgb)
        self.register_buffer("mask", mask if mask is not None else torch.zeros(0))


def fit_from_views(
    field: SpatialLatentField,
    renderer: NeusRenderer,
    views: list[View],
    *,
    cond: Tensor | None = None,
    steps: int = 200,
    lr: float = 1e-3,
    rays_per_step: int = 512,
    eikonal_weight: float = 0.1,
    mask_weight: float = 0.5,
    near: float = 0.1,
    far: float = 4.0,
) -> list[float]:
    """Train the field to reproduce a set of posed images (NeuS objective).

    Each step renders a random batch of rays from a random view and minimizes
    photometric + mask + eikonal losses. Returns the photometric-loss history.
    """
    device = views[0].rgb.device
    field.to(device).train()
    renderer.to(device).train()
    if cond is None:
        cond = torch.zeros(1, field.cfg.cond_dim, device=device)
    params = list(field.parameters()) + list(renderer.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    history: list[float] = []

    for _ in range(steps):
        view = views[int(torch.randint(len(views), (1,)))]
        n_rays = view.rays_o.shape[0]
        sel = torch.randint(n_rays, (min(rays_per_step, n_rays),), device=device)

        planes = field.encode(cond)
        out = renderer.render(field, planes, view.rays_o[sel], view.rays_d[sel], near=near, far=far, jitter=True)
        loss = photometric_loss(out["rgb"], view.rgb[sel])
        if view.mask.numel():
            loss = loss + mask_weight * mask_loss(out["opacity"], view.mask[sel])

        pts = sample_points(1, rays_per_step, device=device)
        _, grad = sdf_and_gradient(field, planes, pts)
        loss = loss + eikonal_weight * eikonal_loss(grad)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(photometric_loss(out["rgb"], view.rgb[sel]).detach()))

    field.eval()
    renderer.eval()
    return history
