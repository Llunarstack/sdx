"""
Tri-plane spatial latent field — the 3D core of SDX ("SDX-Spatial").

Why this exists
---------------
A 2D diffusion model learns a *projection* of an object: it knows what a dragon
looks like from one camera, but has no memory that the front view and the side
view are the same thing. That is why AI 3D breaks — each view is generated
independently and they disagree.

The fix is to stop generating pixels and instead generate a **shared 3D field**.
Every camera angle is then just a *query* into that one field, so multi-view
consistency is guaranteed by construction rather than hoped for.

We represent that field with a **tri-plane**: three orthogonal 2D feature grids
(XY, YZ, XZ). A point in space is looked up by projecting it onto each plane,
bilinearly sampling, and summing the three features. This is far cheaper than a
dense 3D voxel grid (O(R^2) memory instead of O(R^3)) while still resolving fine
detail — the representation used by EG3D / TensoRF-style 3D generators.

The field feeds three lightweight decoder heads, exactly the split the SDX-3D
blueprint calls for:

    conditioning (from SDX DiT)
              |
        [TriPlaneGenerator]
              |
        tri-plane latents  (XY, YZ, XZ)
              |
        query(points) -> per-point feature
              |
   ---------------------------------------------
   |                  |                        |
 [SplatDecoder]   [SDFDecoder]           [PBRDecoder]
 Gaussian splats   signed distance        albedo / roughness /
 (real-time view)  (marching-cubes mesh)  metallic / normal

Scope
-----
This module is the *representation + decoders* — the piece everything else in
the 3D roadmap (kinematic rigging, weathering automata, genetic crossover,
haptics) attaches to. Training objectives, mesh extraction (marching cubes),
and a production splat rasterizer are deliberately out of scope here; the
``render_view`` sphere-tracer included is a compact, differentiable reference
renderer for validating multi-view consistency, not a real-time path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TriPlaneConfig:
    """Shape and capacity of the spatial latent field."""

    cond_dim: int = 256  # dim of the conditioning vector from SDX's DiT
    plane_channels: int = 32  # feature channels per plane
    grid_res: int = 32  # resolution of each plane (must be a power of two >= 8)
    base_channels: int = 128  # hidden width of the tri-plane generator
    feature_dim: int = 64  # per-point feature dim after aggregation
    decoder_hidden: int = 128  # hidden width of the decoder-head MLPs

    def __post_init__(self) -> None:
        if self.grid_res < 8 or (self.grid_res & (self.grid_res - 1)) != 0:
            raise ValueError(f"grid_res must be a power of two >= 8, got {self.grid_res}")


# ---------------------------------------------------------------------------
# Decoder output containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GaussianSplats:
    """A cloud of 3D Gaussians — the real-time / interactive representation."""

    positions: Tensor  # (B, N, 3) world position = query point + learned offset
    scales: Tensor  # (B, N, 3) per-axis stddev, strictly positive
    rotations: Tensor  # (B, N, 4) unit quaternion (w, x, y, z)
    opacities: Tensor  # (B, N, 1) in [0, 1]
    colors: Tensor  # (B, N, 3) in [0, 1]


@dataclass(slots=True)
class PBRSample:
    """Physically based surface properties at queried points."""

    albedo: Tensor  # (B, N, 3) in [0, 1]
    roughness: Tensor  # (B, N, 1) in [0, 1]
    metallic: Tensor  # (B, N, 1) in [0, 1]
    normal: Tensor  # (B, N, 3) unit vector


# ---------------------------------------------------------------------------
# Tri-plane generator: conditioning vector -> three feature planes
# ---------------------------------------------------------------------------


class TriPlaneGenerator(nn.Module):
    """Map a conditioning vector to the three orthogonal feature planes.

    Starts from a learned 4x4 seed and upsamples with transpose convolutions
    to ``grid_res`` — a small StyleGAN-style synthesis net. The final 1x1 conv
    produces ``3 * plane_channels`` maps which are split into the XY/YZ/XZ
    planes.
    """

    def __init__(self, cfg: TriPlaneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n_up = int(round(math.log2(cfg.grid_res // 4)))  # 4 -> grid_res

        self.seed = nn.Linear(cfg.cond_dim, cfg.base_channels * 4 * 4)
        ups: list[nn.Module] = []
        for _ in range(n_up):
            ups += [
                nn.ConvTranspose2d(cfg.base_channels, cfg.base_channels, 4, stride=2, padding=1),
                nn.GroupNorm(8, cfg.base_channels),
                nn.SiLU(),
            ]
        self.up = nn.Sequential(*ups)
        self.to_planes = nn.Conv2d(cfg.base_channels, 3 * cfg.plane_channels, 1)

    def forward(self, cond: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b = cond.shape[0]
        h = self.seed(cond).view(b, self.cfg.base_channels, 4, 4)
        h = self.up(h)
        planes = self.to_planes(h)  # (B, 3C, R, R)
        c = self.cfg.plane_channels
        return planes[:, :c], planes[:, c : 2 * c], planes[:, 2 * c :]


# ---------------------------------------------------------------------------
# Decoder heads
# ---------------------------------------------------------------------------


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class SplatDecoder(nn.Module):
    """Per-point feature -> a 3D Gaussian (offset, scale, rotation, opacity, color)."""

    def __init__(self, cfg: TriPlaneConfig, offset_scale: float = 0.1) -> None:
        super().__init__()
        self.offset_scale = offset_scale
        # 3 offset + 3 log-scale + 4 quat + 1 opacity + 3 color = 14
        self.net = _mlp(cfg.feature_dim, cfg.decoder_hidden, 14)

    def forward(self, feat: Tensor, points: Tensor) -> GaussianSplats:
        raw = self.net(feat)
        offset, log_scale, quat, opacity, color = torch.split(raw, [3, 3, 4, 1, 3], dim=-1)
        return GaussianSplats(
            positions=points + torch.tanh(offset) * self.offset_scale,
            scales=F.softplus(log_scale) + 1e-4,
            rotations=F.normalize(quat, dim=-1),
            opacities=torch.sigmoid(opacity),
            colors=torch.sigmoid(color),
        )


class SDFDecoder(nn.Module):
    """Per-point feature -> signed distance (negative inside, positive outside)."""

    def __init__(self, cfg: TriPlaneConfig) -> None:
        super().__init__()
        self.net = _mlp(cfg.feature_dim, cfg.decoder_hidden, 1)

    def forward(self, feat: Tensor) -> Tensor:
        return self.net(feat)  # (B, N, 1)


class PBRDecoder(nn.Module):
    """Per-point feature -> physically based surface properties."""

    def __init__(self, cfg: TriPlaneConfig) -> None:
        super().__init__()
        self.net = _mlp(cfg.feature_dim, cfg.decoder_hidden, 3 + 1 + 1 + 3)

    def forward(self, feat: Tensor) -> PBRSample:
        raw = self.net(feat)
        albedo, roughness, metallic, normal = torch.split(raw, [3, 1, 1, 3], dim=-1)
        return PBRSample(
            albedo=torch.sigmoid(albedo),
            roughness=torch.sigmoid(roughness),
            metallic=torch.sigmoid(metallic),
            normal=F.normalize(normal, dim=-1),
        )


# ---------------------------------------------------------------------------
# Top-level spatial latent field
# ---------------------------------------------------------------------------


class SpatialLatentField(nn.Module):
    """The shared 3D field: conditioning -> tri-plane -> queryable surface.

    Multi-view consistency is structural: :meth:`query` maps an *object-space*
    point to features independent of any camera, so two different viewpoints
    that hit the same point read identical geometry, color, and material. There
    is no separate "make the views agree" step to get wrong.
    """

    def __init__(self, cfg: TriPlaneConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TriPlaneConfig()
        self.generator = TriPlaneGenerator(self.cfg)
        self.to_feature = nn.Sequential(
            nn.Linear(self.cfg.plane_channels, self.cfg.feature_dim),
            nn.SiLU(),
        )
        self.splat_decoder = SplatDecoder(self.cfg)
        self.sdf_decoder = SDFDecoder(self.cfg)
        self.pbr_decoder = PBRDecoder(self.cfg)

    # -- representation --------------------------------------------------

    def encode(self, cond: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Conditioning vector (B, cond_dim) -> (XY, YZ, XZ) planes."""
        return self.generator(cond)

    def query(self, planes: tuple[Tensor, Tensor, Tensor], points: Tensor) -> Tensor:
        """Sample the field at 3D ``points`` (B, N, 3), coords in [-1, 1].

        Projects each point onto the three planes, bilinearly samples, and sums
        the three features, then lifts to ``feature_dim``. Out-of-range points
        clamp to the plane border rather than wrapping.
        """
        xy, yz, xz = planes
        px, py, pz = points[..., 0], points[..., 1], points[..., 2]
        f = (
            self._sample_plane(xy, px, py) + self._sample_plane(yz, py, pz) + self._sample_plane(xz, px, pz)
        )  # (B, N, plane_channels)
        return self.to_feature(f)

    @staticmethod
    def _sample_plane(plane: Tensor, a: Tensor, b: Tensor) -> Tensor:
        """Bilinearly sample ``plane`` (B, C, H, W) at coords (a->x, b->y) in [-1, 1].

        Implemented by hand rather than with ``F.grid_sample`` because the eikonal
        loss needs the *second* derivative w.r.t. the query point, and
        ``grid_sample``'s double-backward is not implemented in PyTorch. Plain
        arithmetic + gather is twice-differentiable. Matches ``align_corners=True``
        with border padding (out-of-range coords clamp to the edge).
        """
        b_sz, c, h, w = plane.shape
        n = a.shape[-1]
        x = ((a + 1) * 0.5 * (w - 1)).clamp(0, w - 1)
        y = ((b + 1) * 0.5 * (h - 1)).clamp(0, h - 1)
        x0 = torch.floor(x).clamp(0, w - 1)
        y0 = torch.floor(y).clamp(0, h - 1)
        x1 = (x0 + 1).clamp(0, w - 1)
        y1 = (y0 + 1).clamp(0, h - 1)
        wx = (x - x0).unsqueeze(-1)  # (B, N, 1) — carries the gradient
        wy = (y - y0).unsqueeze(-1)
        x0i, x1i = x0.long(), x1.long()
        y0i, y1i = y0.long(), y1.long()
        flat = plane.reshape(b_sz, c, h * w)

        def corner(yi: Tensor, xi: Tensor) -> Tensor:
            idx = (yi * w + xi).unsqueeze(1).expand(b_sz, c, n)
            return torch.gather(flat, 2, idx).transpose(1, 2)  # (B, N, C)

        v00, v01 = corner(y0i, x0i), corner(y0i, x1i)
        v10, v11 = corner(y1i, x0i), corner(y1i, x1i)
        vx0 = v00 * (1 - wx) + v01 * wx
        vx1 = v10 * (1 - wx) + v11 * wx
        return vx0 * (1 - wy) + vx1 * wy

    # -- decoders (convenience wrappers) --------------------------------

    def decode_splats(self, planes: tuple[Tensor, Tensor, Tensor], points: Tensor) -> GaussianSplats:
        return self.splat_decoder(self.query(planes, points), points)

    def decode_sdf(self, planes: tuple[Tensor, Tensor, Tensor], points: Tensor) -> Tensor:
        return self.sdf_decoder(self.query(planes, points))

    def decode_pbr(self, planes: tuple[Tensor, Tensor, Tensor], points: Tensor) -> PBRSample:
        return self.pbr_decoder(self.query(planes, points))

    # -- reference renderer ---------------------------------------------

    def render_view(
        self,
        planes: tuple[Tensor, Tensor, Tensor],
        cam_origin: Tensor,
        ray_dirs: Tensor,
        *,
        steps: int = 24,
        surface_eps: float = 1e-2,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sphere-trace the SDF to render one camera view of the shared field.

        This is the "perfect angle" feature made concrete: pass any camera and
        it renders that view from the *same* field. It exists to validate
        consistency and to be differentiable, not to be fast.

        Args:
            cam_origin: (B, 3) camera position in field space.
            ray_dirs:   (B, P, 3) unit ray directions, one per pixel.

        Returns:
            ``(rgb, depth, hit)`` with shapes (B, P, 3), (B, P), (B, P).
        """
        o = cam_origin.unsqueeze(1)  # (B, 1, 3)
        t = torch.zeros(ray_dirs.shape[:-1], device=ray_dirs.device, dtype=ray_dirs.dtype)
        sdf = torch.zeros_like(t)
        for _ in range(steps):
            pos = o + t.unsqueeze(-1) * ray_dirs
            sdf = self.decode_sdf(planes, pos).squeeze(-1)
            # Clamp the marching step for stability with an untrained field.
            t = t + sdf.clamp(-0.05, 0.25)
        pos = o + t.unsqueeze(-1) * ray_dirs
        pbr = self.decode_pbr(planes, pos)
        hit = (sdf.abs() < surface_eps).to(ray_dirs.dtype)
        rgb = pbr.albedo * hit.unsqueeze(-1)
        return rgb, t, hit

    # -- mesh export -----------------------------------------------------

    @torch.no_grad()
    def to_mesh(
        self,
        planes: tuple[Tensor, Tensor, Tensor],
        *,
        resolution: int = 48,
        bounds: tuple[float, float] = (-1.0, 1.0),
        with_color: bool = False,
        batch_points: int = 65536,
    ):
        """Extract a watertight mesh (and optional vertex colors) for one item.

        Evaluates this field's SDF on a grid and runs marching tetrahedra. Only
        the first item of ``planes`` is meshed — meshing is a per-object export
        step. Returns ``(vertices, faces)`` or ``(vertices, faces, colors)``.
        """
        import numpy as np

        from .mesh_export import extract_mesh

        single = tuple(p[:1] for p in planes)
        device = single[0].device

        def sdf_fn(points: np.ndarray) -> np.ndarray:
            out = np.empty(points.shape[0], dtype=np.float32)
            for start in range(0, points.shape[0], batch_points):
                chunk = points[start : start + batch_points]
                pt = torch.from_numpy(chunk).to(device=device, dtype=single[0].dtype).unsqueeze(0)
                out[start : start + batch_points] = self.decode_sdf(single, pt).squeeze(0).squeeze(-1).cpu().numpy()
            return out

        verts, faces = extract_mesh(sdf_fn, resolution=resolution, bounds=bounds)
        if not with_color:
            return verts, faces
        if len(verts) == 0:
            return verts, faces, np.zeros((0, 3), np.float32)
        pt = torch.from_numpy(verts).to(device=device, dtype=single[0].dtype).unsqueeze(0)
        colors = self.decode_pbr(single, pt).albedo.squeeze(0).cpu().numpy()
        return verts, faces, colors


def build_spatial_field(**overrides: object) -> SpatialLatentField:
    """Construct a :class:`SpatialLatentField` from keyword config overrides."""
    return SpatialLatentField(TriPlaneConfig(**overrides))  # type: ignore[arg-type]
