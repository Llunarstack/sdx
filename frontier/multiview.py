"""SDX-Spatial: the 3D core. A tri-plane latent field that decodes to one consistent object from any camera.

``triplane`` is the field itself; ``training`` fits it to an SDF, ``neus_render`` renders and fits it from views,
``mesh_export`` / ``validate`` turn it into a checked mesh, ``radiance_sh`` adds view-dependent colour,
``rigging`` skeletons and skins the mesh, ``asset_graph`` splits it into semantic parts, ``crossbreed`` mixes
conditionings, ``bridge`` lifts 2D DiT features into the field, and ``views`` renders turntables / contact sheets.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from dataclasses import field as dataclass_field

import numpy as np
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


SdfTarget = Callable[[Tensor], Tensor]  # points (B, N, 3) -> sdf (B, N, 1)


def sample_points(batch: int, n: int, *, device: torch.device | str = "cpu") -> Tensor:
    """Uniform points in the ``[-1, 1]^3`` field cube, shape ``(batch, n, 3)``."""
    return torch.empty(batch, n, 3, device=device).uniform_(-1.0, 1.0)


def sdf_and_gradient(
    field: SpatialLatentField, planes: tuple[Tensor, Tensor, Tensor], points: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(sdf, d sdf / d points)``. The gradient is what eikonal acts on.

    ``create_graph=True`` keeps the gradient differentiable so the eikonal penalty
    itself trains the network (a second-order term — this is why the tri-plane uses
    a hand-written bilinear sampler that supports double backward).
    """
    points = points.detach().requires_grad_(True)
    sdf = field.decode_sdf(planes, points)
    grad = torch.autograd.grad(sdf, points, grad_outputs=torch.ones_like(sdf), create_graph=True)[0]
    return sdf, grad


def eikonal_loss(grad: Tensor) -> Tensor:
    """Penalize deviation of the SDF gradient norm from 1."""
    return ((grad.norm(dim=-1) - 1.0) ** 2).mean()


def sdf_regression_loss(pred: Tensor, target: Tensor) -> Tensor:
    """MSE toward a target SDF, with extra weight near the surface.

    Errors close to the zero level set matter far more than errors deep inside or
    far outside, so weight samples by how near the target places them.
    """
    weight = torch.exp(-4.0 * target.abs())
    return (weight * (pred - target) ** 2).mean()


def surface_normal(field: SpatialLatentField, planes: tuple[Tensor, Tensor, Tensor], points: Tensor) -> Tensor:
    """Unit surface normal = normalized SDF gradient (for shading / normal loss)."""
    _, grad = sdf_and_gradient(field, planes, points)
    return F.normalize(grad, dim=-1)


@dataclass(slots=True)
class FitResult:
    losses: list[float] = dataclass_field(default_factory=list)
    eikonal: list[float] = dataclass_field(default_factory=list)
    cond: Tensor | None = None


def fit_sdf(
    field: SpatialLatentField,
    target_sdf: SdfTarget,
    *,
    cond: Tensor | None = None,
    steps: int = 300,
    lr: float = 1e-3,
    n_points: int = 2048,
    eikonal_weight: float = 0.1,
    device: torch.device | str = "cpu",
) -> FitResult:
    """Overfit ``field`` to a single target SDF shape.

    ``cond`` is the fixed conditioning vector the shape is stored under (defaults
    to zeros); training adjusts the field weights so ``encode(cond)`` reproduces
    the target. Returns the loss history and the ``cond`` used (so the caller can
    ``encode`` + ``to_mesh`` afterwards).
    """
    field.to(device).train()
    if cond is None:
        cond = torch.zeros(1, field.cfg.cond_dim, device=device)
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    result = FitResult(cond=cond)

    for _ in range(steps):
        planes = field.encode(cond)
        points = sample_points(cond.shape[0], n_points, device=device)
        target = target_sdf(points)

        pred, grad = sdf_and_gradient(field, planes, points)
        reg = sdf_regression_loss(pred, target)
        eik = eikonal_loss(grad)
        loss = reg + eikonal_weight * eik

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        result.losses.append(float(reg.detach()))
        result.eikonal.append(float(eik.detach()))

    field.eval()
    return result


def sphere_sdf(radius: float) -> SdfTarget:
    """Convenience analytic target: a centered sphere of the given radius."""

    def fn(points: Tensor) -> Tensor:
        return points.norm(dim=-1, keepdim=True) - radius

    return fn


# A cube's 8 corners, in the local numbering used throughout this module.
#   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
#   4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
_CORNER_OFFSETS = np.array(
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    dtype=np.int64,
)

# Split a cube into 6 tetrahedra all sharing the main diagonal 0-6. Every cube
# uses this identical pattern, which is what guarantees a watertight result.
_CUBE_TETS = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)

# For a 4-corner tetrahedron, code bit m is set when corner m is *inside*
# (sdf < isolevel). Each entry lists triangles; a triangle is 3 edges; an edge
# (a, b) is interpolated between local tet corners a and b. Winding is fixed
# afterwards from the field gradient, so orientation here is unimportant.
_TET_TRI_TABLE: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    1: (((0, 1), (0, 2), (0, 3)),),
    2: (((1, 0), (1, 2), (1, 3)),),
    4: (((2, 0), (2, 1), (2, 3)),),
    8: (((3, 0), (3, 1), (3, 2)),),
    7: (((3, 0), (3, 1), (3, 2)),),
    11: (((2, 0), (2, 1), (2, 3)),),
    13: (((1, 0), (1, 2), (1, 3)),),
    14: (((0, 1), (0, 2), (0, 3)),),
    3: (((0, 2), (0, 3), (1, 3)), ((0, 2), (1, 3), (1, 2))),
    5: (((0, 1), (0, 3), (2, 3)), ((0, 1), (2, 3), (2, 1))),
    6: (((1, 0), (1, 3), (2, 3)), ((1, 0), (2, 3), (2, 0))),
    9: (((0, 1), (0, 2), (3, 2)), ((0, 1), (3, 2), (3, 1))),
    10: (((1, 0), (1, 2), (3, 2)), ((1, 0), (3, 2), (3, 0))),
    12: (((2, 0), (2, 1), (3, 1)), ((2, 0), (3, 1), (3, 0))),
}

SdfFn = Callable[[np.ndarray], np.ndarray]


def extract_mesh(
    sdf_fn: SdfFn,
    *,
    resolution: int = 48,
    bounds: tuple[float, float] = (-1.0, 1.0),
    isolevel: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a watertight triangle mesh from a signed distance function.

    Args:
        sdf_fn: maps an ``(N, 3)`` array of points to an ``(N,)`` array of signed
            distances (negative inside the surface).
        resolution: number of cells per axis; the grid is ``resolution+1`` samples.
        bounds: ``(min, max)`` extent of the sampling cube on every axis.
        isolevel: the level set to extract (0 for a standard SDF).

    Returns:
        ``(vertices, faces)`` as float32 ``(V, 3)`` and int64 ``(F, 3)`` arrays.
        Vertices are merged so the mesh is a closed manifold.
    """
    axis = np.linspace(bounds[0], bounds[1], resolution + 1).astype(np.float64)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    pts = np.stack((gx, gy, gz), axis=-1).reshape(-1, 3).astype(np.float32)
    grid = np.asarray(sdf_fn(pts), dtype=np.float64).reshape(len(axis), len(axis), len(axis))

    verts, faces = _marching_tetrahedra(grid, axis, isolevel)
    return _weld_vertices(verts, faces)


def _marching_tetrahedra(grid: np.ndarray, axis: np.ndarray, iso: float) -> tuple[np.ndarray, np.ndarray]:
    res = len(axis) - 1
    ii, jj = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    x_of = axis  # coordinate lookup along any axis

    vert_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    vcount = 0

    # Process one z-slab at a time so peak memory stays O(res^2) even at high res.
    for k in range(res):
        cv = [grid[o[0] + ii, o[1] + jj, k + o[2]] for o in _CORNER_OFFSETS]  # 8 x (res,res)
        cc = [
            np.stack((x_of[o[0] + ii], x_of[o[1] + jj], np.full_like(ii, x_of[k + o[2]], dtype=float)), axis=-1)
            for o in _CORNER_OFFSETS
        ]  # 8 x (res,res,3)

        for tet in _CUBE_TETS:
            val = [cv[c] for c in tet]
            pos = [cc[c] for c in tet]
            code = sum(((val[m] < iso).astype(np.int64) << m) for m in range(4))

            for code_val, tris in _TET_TRI_TABLE.items():
                mask = code == code_val
                if not mask.any():
                    continue
                inside = [m for m in range(4) if (code_val >> m) & 1]
                inside_centroid = np.mean(np.stack([pos[m][mask] for m in inside]), axis=0)

                for tri in tris:
                    v = [_edge_point(val[a], val[b], pos[a], pos[b], mask, iso) for a, b in tri]
                    v0, v1, v2 = _orient(v[0], v[1], v[2], inside_centroid)
                    m = v0.shape[0]
                    vert_blocks += [v0, v1, v2]
                    idx = np.arange(m)
                    face_blocks.append(np.stack((vcount + idx, vcount + m + idx, vcount + 2 * m + idx), axis=1))
                    vcount += 3 * m

    if not vert_blocks:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    return (
        np.concatenate(vert_blocks, axis=0).astype(np.float32),
        np.concatenate(face_blocks, axis=0).astype(np.int64),
    )


def _edge_point(
    va: np.ndarray, vb: np.ndarray, pa: np.ndarray, pb: np.ndarray, mask: np.ndarray, iso: float
) -> np.ndarray:
    """Linear zero-crossing between two corners, on the masked cells."""
    a, b = va[mask], vb[mask]
    pa_m, pb_m = pa[mask], pb[mask]
    denom = b - a
    denom[denom == 0.0] = 1e-12
    t = np.clip((iso - a) / denom, 0.0, 1.0)
    return pa_m + t[:, None] * (pb_m - pa_m)


def _orient(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, inside_centroid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip winding so each face normal points away from the solid interior."""
    normal = np.cross(v1 - v0, v2 - v0)
    outward = (v0 + v1 + v2) / 3.0 - inside_centroid
    flip = np.sum(normal * outward, axis=1) < 0.0
    v1o = np.where(flip[:, None], v2, v1)
    v2o = np.where(flip[:, None], v1, v2)
    return v0, v1o, v2o


def _weld_vertices(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices so shared edges are shared, giving a manifold."""
    if len(verts) == 0:
        return verts, faces
    key = np.round(verts.astype(np.float64), 6)
    _, first_idx, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    welded = verts[first_idx]
    remapped = inverse[faces]
    # Drop any degenerate faces that collapsed during welding.
    good = (remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2]) & (remapped[:, 0] != remapped[:, 2])
    return welded.astype(np.float32), remapped[good].astype(np.int64)


def write_obj(
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    vertex_colors: np.ndarray | None = None,
) -> None:
    """Write a Wavefront ``.obj``. Optional per-vertex RGB (in [0,1]) is written
    as extended ``v x y z r g b`` lines, which Blender and MeshLab read."""
    lines: list[str] = []
    if vertex_colors is None:
        for x, y, z in vertices:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    else:
        for (x, y, z), (r, g, b) in zip(vertices, np.clip(vertex_colors, 0.0, 1.0), strict=True):
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}")
    for a, b, c in faces + 1:  # OBJ is 1-indexed
        lines.append(f"f {a} {b} {c}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


@dataclass(slots=True)
class ValidationReport:
    watertight: bool
    euler_characteristic: int
    n_vertices: int
    n_faces: int
    surface_area: float
    volume: float
    center_of_mass: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    stable: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.watertight and not self.warnings


def _edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            key = (int(min(x, y)), int(max(x, y)))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _tri_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _volume_and_com(v: np.ndarray, f: np.ndarray) -> tuple[float, np.ndarray]:
    """Signed volume and volume centroid via the divergence theorem (tets to origin)."""
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    signed_vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
    total = float(signed_vol.sum())
    if abs(total) < 1e-12:
        return 0.0, np.zeros(3)
    tet_centroid = (v0 + v1 + v2) / 4.0  # 4th corner is the origin
    com = (tet_centroid * signed_vol[:, None]).sum(axis=0) / total
    return abs(total), com


def _is_stable(v: np.ndarray, com: np.ndarray, up_axis: int) -> bool:
    """Will it stand? Check the COM projects inside the base contact footprint."""
    axes = [i for i in range(3) if i != up_axis]
    ground = v[:, up_axis].min()
    thickness = 0.02 * (v[:, up_axis].max() - ground + 1e-9)
    base = v[v[:, up_axis] <= ground + thickness][:, axes]
    if len(base) < 3:
        return False
    lo, hi = base.min(axis=0), base.max(axis=0)
    p = com[axes]
    # Conservative footprint test: COM within the base's axis-aligned span.
    return bool(np.all(p >= lo - 1e-6) and np.all(p <= hi + 1e-6))


def validate_mesh(vertices: np.ndarray, faces: np.ndarray, *, up_axis: int = 1) -> ValidationReport:
    """Run structural / physical checks on a triangle mesh."""
    warnings: list[str] = []
    n_v, n_f = len(vertices), len(faces)
    if n_v == 0 or n_f == 0:
        return ValidationReport(
            False, 0, 0, 0, 0.0, 0.0, (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), False, ["empty mesh"]
        )

    counts = _edge_counts(faces)
    boundary = [e for e, n in counts.items() if n == 1]
    nonmanifold = [e for e, n in counts.items() if n > 2]
    watertight = not boundary and not nonmanifold
    if boundary:
        warnings.append(f"{len(boundary)} boundary edges (holes) — not watertight")
    if nonmanifold:
        warnings.append(f"{len(nonmanifold)} non-manifold edges")

    euler = n_v - len(counts) + n_f
    area = float(_tri_areas(vertices, faces).sum())
    volume, com = _volume_and_com(vertices, faces)
    if volume < 1e-8:
        warnings.append("near-zero volume — degenerate or open surface")

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    dims = bbox_max - bbox_min
    if float(dims.min()) < 1e-4:
        warnings.append("paper-thin in one axis — no printable thickness")

    stable = _is_stable(vertices, com, up_axis)
    if not stable:
        warnings.append("center of mass outside base — will tip over")

    return ValidationReport(
        watertight=watertight,
        euler_characteristic=int(euler),
        n_vertices=n_v,
        n_faces=n_f,
        surface_area=area,
        volume=volume,
        center_of_mass=tuple(float(x) for x in com),
        bbox_min=tuple(float(x) for x in bbox_min),
        bbox_max=tuple(float(x) for x in bbox_max),
        dimensions=tuple(float(x) for x in dims),
        stable=stable,
        warnings=warnings,
    )


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

    Pass a :class:`SHRadianceDecoder` to make the
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


# Standard real spherical-harmonic normalization constants (degrees 0..3).
_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def sh_num_coeffs(degree: int) -> int:
    """Number of SH coefficients for a given max degree: ``(degree + 1)^2``."""
    return (degree + 1) ** 2


def eval_sh_basis(dirs: Tensor, degree: int) -> Tensor:
    """Evaluate the real SH basis at unit directions ``dirs`` (..., 3).

    Returns ``(..., (degree+1)^2)``. Degree 0 is the constant term; higher orders
    add directional variation.
    """
    if not 0 <= degree <= 3:
        raise ValueError(f"degree must be 0..3, got {degree}")
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    out = [torch.full_like(x, _C0)]
    if degree >= 1:
        out += [-_C1 * y, _C1 * z, -_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        out += [
            _C2[0] * x * y,
            _C2[1] * y * z,
            _C2[2] * (2.0 * zz - xx - yy),
            _C2[3] * x * z,
            _C2[4] * (xx - yy),
        ]
    if degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        out += [
            _C3[0] * y * (3.0 * xx - yy),
            _C3[1] * x * y * z,
            _C3[2] * y * (4.0 * zz - xx - yy),
            _C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            _C3[4] * x * (4.0 * zz - xx - yy),
            _C3[5] * z * (xx - yy),
            _C3[6] * x * (xx - 3.0 * yy),
        ]
    return torch.stack(out, dim=-1)


class SHRadianceDecoder(nn.Module):
    """Per-point feature -> SH coefficients -> view-dependent RGB.

    At degree 0 this reduces to a constant color (equivalent to albedo); raising
    the degree lets the same point show highlights and reflections that move with
    the camera.
    """

    def __init__(self, feature_dim: int, degree: int = 3, hidden: int = 128) -> None:
        super().__init__()
        self.degree = degree
        self.n_coeffs = sh_num_coeffs(degree)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * self.n_coeffs),
        )

    def coefficients(self, feat: Tensor) -> Tensor:
        """SH coefficients per point, shape ``(..., 3, n_coeffs)``."""
        raw = self.net(feat)
        return raw.reshape(*raw.shape[:-1], 3, self.n_coeffs)

    def forward(self, feat: Tensor, view_dirs: Tensor) -> Tensor:
        """Radiance (RGB in [0, 1]) at each point seen from ``view_dirs``.

        ``feat`` is ``(..., F)`` and ``view_dirs`` is ``(..., 3)`` (unit vectors);
        they broadcast on the leading dims.
        """
        coeffs = self.coefficients(feat)  # (..., 3, K)
        basis = eval_sh_basis(view_dirs, self.degree)  # (..., K)
        rgb = (coeffs * basis.unsqueeze(-2)).sum(dim=-1)  # (..., 3)
        # 3DGS convention: SH encodes color around a 0.5 grey, then clamp to [0,1].
        return torch.clamp(rgb + 0.5, 0.0, 1.0)


# Camera *directions* (the camera sits at dir * radius, looking at the origin).
CANONICAL_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, 0.0, -1.0),
    "back": (0.0, 0.0, 1.0),
    "right": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "top": (0.0, 1.0, 0.0),
    "bottom": (0.0, -1.0, 0.0),
    "three_quarter": (0.7, 0.5, -0.7),
}


def viewpoint_rays(
    name: str,
    *,
    radius: float = 2.5,
    fov_deg: float = 45.0,
    height: int = 64,
    width: int = 64,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Camera rays for a named canonical viewpoint."""
    if name not in CANONICAL_DIRECTIONS:
        raise KeyError(f"unknown viewpoint {name!r}; options: {sorted(CANONICAL_DIRECTIONS)}")
    d = torch.tensor(CANONICAL_DIRECTIONS[name], dtype=torch.float32, device=device)
    cam_pos = torch.nn.functional.normalize(d, dim=0) * radius
    # For top/bottom the world-up is parallel to the view dir, so pick a different up.
    up = (0.0, 0.0, 1.0) if name in ("top", "bottom") else (0.0, 1.0, 0.0)
    return look_at_rays(cam_pos, torch.zeros(3, device=device), fov_deg=fov_deg, height=height, width=width, up=up)


def _render_image(
    field: SpatialLatentField,
    planes: tuple[Tensor, Tensor, Tensor],
    renderer: NeusRenderer,
    rays_o: Tensor,
    rays_d: Tensor,
    height: int,
    width: int,
    *,
    near: float,
    far: float,
    background: float,
) -> np.ndarray:
    """Render rays and composite the RGB over a flat background using opacity."""
    with torch.no_grad():
        out = renderer.render(field, planes, rays_o, rays_d, near=near, far=far)
    rgb = out["rgb"].reshape(height, width, 3)
    alpha = out["opacity"].reshape(height, width, 1).clamp(0, 1)
    composited = rgb * alpha + background * (1.0 - alpha)
    return composited.clamp(0, 1).cpu().numpy()


def perfect_angles(
    field: SpatialLatentField,
    planes: tuple[Tensor, Tensor, Tensor],
    renderer: NeusRenderer,
    *,
    names: list[str] | None = None,
    radius: float = 2.5,
    fov_deg: float = 45.0,
    resolution: int = 64,
    near: float = 0.5,
    far: float = 5.0,
    background: float = 1.0,
) -> dict[str, np.ndarray]:
    """Render a dict of ``name -> (H, W, 3)`` images, all of the same object."""
    names = names or ["front", "right", "back", "left", "top", "three_quarter"]
    device = planes[0].device
    out: dict[str, np.ndarray] = {}
    for name in names:
        o, d = viewpoint_rays(name, radius=radius, fov_deg=fov_deg, height=resolution, width=resolution, device=device)
        out[name] = _render_image(
            field, planes, renderer, o, d, resolution, resolution, near=near, far=far, background=background
        )
    return out


def turntable(
    field: SpatialLatentField,
    planes: tuple[Tensor, Tensor, Tensor],
    renderer: NeusRenderer,
    *,
    n_frames: int = 8,
    elevation_deg: float = 20.0,
    radius: float = 2.5,
    fov_deg: float = 45.0,
    resolution: int = 64,
    near: float = 0.5,
    far: float = 5.0,
    background: float = 1.0,
) -> list[np.ndarray]:
    """Render an orbit of frames around the object (azimuth sweep)."""
    device = planes[0].device
    el = math.radians(elevation_deg)
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        az = 2 * math.pi * i / n_frames
        cam = torch.tensor(
            [radius * math.cos(el) * math.sin(az), radius * math.sin(el), radius * math.cos(el) * math.cos(az)],
            dtype=torch.float32,
            device=device,
        )
        o, d = look_at_rays(cam, torch.zeros(3, device=device), fov_deg=fov_deg, height=resolution, width=resolution)
        frames.append(
            _render_image(
                field, planes, renderer, o, d, resolution, resolution, near=near, far=far, background=background
            )
        )
    return frames


def contact_sheet(images: list[np.ndarray], *, cols: int = 3, pad: int = 4, background: float = 1.0) -> np.ndarray:
    """Tile images into a single ``(H, W, 3)`` grid with padding."""
    if not images:
        raise ValueError("no images to tile")
    h, w, _ = images[0].shape
    rows = math.ceil(len(images) / cols)
    sheet = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), background, dtype=np.float32)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        sheet[y : y + h, x : x + w] = img
    return sheet


def save_image(path: str, image: np.ndarray) -> None:
    """Write a float ``[0, 1]`` image (H, W, 3) to a PNG."""
    from PIL import Image

    arr = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


@dataclass(slots=True)
class BridgeConfig:
    text_dim: int = 1024  # dim of the pooled text embedding from SDX
    image_dim: int = 1024  # dim of each image patch token (0 disables the image path)
    cond_dim: int = 256  # must match TriPlaneConfig.cond_dim
    num_heads: int = 8  # heads for image-token attention pooling
    hidden: int = 512


class AttentionPool(nn.Module):
    """Pool a variable-length token sequence into one vector via a learned query."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        b = tokens.shape[0]
        q = self.query.expand(b, -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
        return self.norm(pooled.squeeze(1))  # (B, dim)


class Lift2Dto3D(nn.Module):
    """Adapter: 2D SDX conditioning -> tri-plane conditioning vector."""

    def __init__(self, cfg: BridgeConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BridgeConfig()
        self.text_proj = nn.Linear(self.cfg.text_dim, self.cfg.hidden)
        self.use_image = self.cfg.image_dim > 0
        if self.use_image:
            self.image_pool = AttentionPool(self.cfg.image_dim, self.cfg.num_heads)
            self.image_proj = nn.Linear(self.cfg.image_dim, self.cfg.hidden)
        self.mix = nn.Sequential(
            nn.LayerNorm(self.cfg.hidden),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden, self.cfg.cond_dim),
        )

    def forward(
        self,
        text_embed: Tensor,
        image_tokens: Tensor | None = None,
        image_token_mask: Tensor | None = None,
    ) -> Tensor:
        """Return a conditioning vector ``(B, cond_dim)``.

        Args:
            text_embed: ``(B, text_dim)`` pooled text embedding.
            image_tokens: optional ``(B, L, image_dim)`` patch tokens for image->3D.
            image_token_mask: optional ``(B, L)`` bool mask, ``True`` where a token
                is padding and should be ignored by the pool.
        """
        h = self.text_proj(text_embed)
        if image_tokens is not None:
            if not self.use_image:
                raise ValueError("bridge built with image_dim=0 but image_tokens were passed")
            pooled = self.image_pool(image_tokens, key_padding_mask=image_token_mask)
            h = h + self.image_proj(pooled)
        return self.mix(h)


def lerp(a: Tensor, b: Tensor, t: float) -> Tensor:
    """Straight-line interpolation between two conditioning vectors."""
    return (1.0 - t) * a + t * b


def slerp(a: Tensor, b: Tensor, t: float, *, eps: float = 1e-7) -> Tensor:
    """Spherical interpolation — travels along the hypersphere, preserving norm.

    Better than ``lerp`` for latent codes: it keeps the interpolant on the shell
    the two parents live on, so the midpoint is a plausible object rather than a
    washed-out average that decodes to mush.

    Near-parallel inputs (|sin ω| tiny) fall back to lerp to avoid NaN from the
    division by sin(ω). Unit vectors are interpolated then rescaled to the
    lerped parent norms.
    """
    a_flat, b_flat = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    a_norm = a_flat.norm(dim=-1, keepdim=True)
    b_norm = b_flat.norm(dim=-1, keepdim=True)
    na = a_flat / (a_norm + eps)
    nb = b_flat / (b_norm + eps)
    dot = (na * nb).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    # Fall back to lerp when vectors are nearly collinear (ω ≈ 0 or π).
    use_lerp = sin_omega.abs() < 1e-4
    wa = torch.sin((1.0 - t) * omega) / sin_omega.clamp(min=eps)
    wb = torch.sin(t * omega) / sin_omega.clamp(min=eps)
    unit = wa * na + wb * nb
    unit_lerp = (1.0 - t) * na + t * nb
    unit = torch.where(use_lerp, unit_lerp, unit)
    out_norm = (1.0 - t) * a_norm + t * b_norm
    out = unit * out_norm
    return out.reshape_as(a)


def blend_conditioning(cond_a: Tensor, cond_b: Tensor, t: float = 0.5) -> Tensor:
    """Breed two objects at the latent level (semantic fusion). ``t`` biases toward B."""
    return slerp(cond_a, cond_b, t)


def spatial_graft(
    sdf_a: SdfFn,
    sdf_b: SdfFn,
    *,
    axis: int = 1,
    split: float = 0.0,
    smooth: float = 0.08,
) -> SdfFn:
    """Chimera: object A below ``split`` on ``axis``, object B above, blended.

    The seam uses a logistic weight so the two bodies fuse continuously instead of
    showing a hard cut, which keeps the grafted surface watertight and meshable.
    """

    def grafted(points: np.ndarray) -> np.ndarray:
        coord = points[:, axis]
        w = 1.0 / (1.0 + np.exp(-(coord - split) / smooth))  # 0 below seam -> 1 above
        return (1.0 - w) * sdf_a(points) + w * sdf_b(points)

    return grafted


def mutate_conditioning(cond: Tensor, *, strength: float = 0.1, seed: int | None = None) -> Tensor:
    """Random mutation of a genotype — explore nearby variants of one object."""
    gen = None
    if seed is not None:
        gen = torch.Generator(device=cond.device).manual_seed(seed)
    noise = torch.randn(cond.shape, generator=gen, device=cond.device, dtype=cond.dtype)
    return cond + strength * noise


@dataclass(slots=True)
class AABB:
    """Axis-aligned box marking the region a part owns (its edit scope)."""

    lo: np.ndarray
    hi: np.ndarray

    def contains(self, points: np.ndarray) -> np.ndarray:
        return np.all((points >= self.lo) & (points <= self.hi), axis=-1)

    @staticmethod
    def around(center: tuple[float, float, float], half: float) -> AABB:
        c = np.asarray(center, dtype=np.float64)
        return AABB(c - half, c + half)


@dataclass(slots=True)
class SemanticPart:
    name: str
    sdf_fn: SdfFn
    region: AABB


class AssetGraph:
    """A named collection of parts composed by CSG union."""

    def __init__(self) -> None:
        self._parts: dict[str, SemanticPart] = {}

    @property
    def parts(self) -> dict[str, SemanticPart]:
        return dict(self._parts)

    def add(self, part: SemanticPart) -> AssetGraph:
        self._parts[part.name] = part
        return self

    def edit_part(self, name: str, sdf_fn: SdfFn, region: AABB | None = None) -> AssetGraph:
        """Replace one part's geometry. Every other part is untouched."""
        if name not in self._parts:
            raise KeyError(f"no part named {name!r}")
        old = self._parts[name]
        self._parts[name] = SemanticPart(name, sdf_fn, region or old.region)
        return self

    def remove_part(self, name: str) -> AssetGraph:
        self._parts.pop(name, None)
        return self

    def composed_sdf(self, points: np.ndarray) -> np.ndarray:
        """Union of all parts: the closest surface wins (min signed distance).

        Outside a part's AABB the part contributes a large positive SDF so it
        cannot pull the union surface outside its edit region.
        """
        if not self._parts:
            return np.full(len(points), 1e3)
        out = np.full(len(points), np.inf)
        for part in self._parts.values():
            sdf = np.asarray(part.sdf_fn(points), dtype=np.float64)
            outside = ~part.region.contains(points)
            sdf = np.where(outside, 1e3, sdf)
            out = np.minimum(out, sdf)
        return out

    def to_mesh(
        self, *, resolution: int = 48, bounds: tuple[float, float] = (-1.0, 1.0)
    ) -> tuple[np.ndarray, np.ndarray]:
        return extract_mesh(self.composed_sdf, resolution=resolution, bounds=bounds)


def part_from_field(
    name: str,
    field,  # SpatialLatentField
    planes,  # tuple[Tensor, Tensor, Tensor]
    region: AABB,
    *,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    batch: int = 65536,
) -> SemanticPart:
    """Wrap a tri-plane field as a part, placed at ``offset`` and sized by ``scale``."""
    import torch

    single = tuple(p[:1] for p in planes)
    device = single[0].device
    off = np.asarray(offset, dtype=np.float32)

    def sdf_fn(points: np.ndarray) -> np.ndarray:
        local = (points - off) / scale
        out = np.empty(len(points), dtype=np.float32)
        for s in range(0, len(points), batch):
            chunk = torch.from_numpy(local[s : s + batch].astype(np.float32)).to(device)
            val = field.decode_sdf(single, chunk.unsqueeze(0)).squeeze(0).squeeze(-1)
            out[s : s + batch] = (val * scale).detach().cpu().numpy()
        return out

    return SemanticPart(name, sdf_fn, region)


@dataclass(slots=True)
class Skeleton:
    """Joints and their tree structure.

    ``joints`` is ``(J, 3)``; ``parents[j]`` is the index of joint ``j``'s parent,
    or ``-1`` for a root. Parents always have a smaller index than their children
    (a valid topological order), which keeps forward kinematics a single pass.
    """

    joints: np.ndarray
    parents: np.ndarray

    def __post_init__(self) -> None:
        self.joints = np.asarray(self.joints, dtype=np.float64).reshape(-1, 3)
        self.parents = np.asarray(self.parents, dtype=np.int64).reshape(-1)
        if len(self.joints) != len(self.parents):
            raise ValueError("joints and parents must have equal length")

    @property
    def n_joints(self) -> int:
        return len(self.joints)


def auto_skeleton_from_points(points: np.ndarray, *, n_joints: int = 5) -> Skeleton:
    """Heuristic skeleton: a chain of joints along the shape's principal axis.

    Stand-in for a learned UniRig-style predictor. Good enough to rig elongated
    objects (limbs, tools, creatures) and to exercise the downstream skinning /
    LBS machinery end-to-end.
    """
    if n_joints < 2:
        raise ValueError("need at least 2 joints")
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    # Principal axis = eigenvector of the covariance with the largest eigenvalue.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    t = centered @ axis
    span = np.linspace(t.min(), t.max(), n_joints)
    joints = center[None, :] + span[:, None] * axis[None, :]
    parents = np.array([-1] + list(range(n_joints - 1)), dtype=np.int64)
    return Skeleton(joints, parents)


def tokenize_skeleton(skel: Skeleton) -> list[tuple[int, float, float, float]]:
    """Serialize a skeleton to a token sequence (UniRig-style tree tokenization).

    Each token is ``(parent_index, x, y, z)`` in depth-first order. This is the
    representation an autoregressive model would emit; :func:`detokenize_skeleton`
    is the exact inverse.
    """
    tokens: list[tuple[int, float, float, float]] = []
    for j in range(skel.n_joints):
        x, y, z = skel.joints[j]
        tokens.append((int(skel.parents[j]), float(x), float(y), float(z)))
    return tokens


def detokenize_skeleton(tokens: list[tuple[int, float, float, float]]) -> Skeleton:
    """Inverse of :func:`tokenize_skeleton`."""
    parents = np.array([t[0] for t in tokens], dtype=np.int64)
    joints = np.array([t[1:] for t in tokens], dtype=np.float64)
    return Skeleton(joints, parents)


def _bone_segment(skel: Skeleton, j: int) -> tuple[np.ndarray, np.ndarray]:
    """The bone ending at joint ``j`` (from its parent). Roots are a zero-length point."""
    p = skel.parents[j]
    start = skel.joints[j] if p < 0 else skel.joints[p]
    return start, skel.joints[j]


def skinning_weights(vertices: np.ndarray, skel: Skeleton, *, temperature: float = 0.1) -> np.ndarray:
    """Per-vertex influence of each joint, ``(V, J)`` rows summing to 1.

    A vertex is bound to joints by proximity to the *bone* ending at each joint
    (point-to-segment distance), softmax-weighted. Closer bone => more influence.
    """
    v = np.asarray(vertices, dtype=np.float64)
    dists = np.empty((len(v), skel.n_joints), dtype=np.float64)
    for j in range(skel.n_joints):
        a, b = _bone_segment(skel, j)
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-12:
            dists[:, j] = np.linalg.norm(v - a, axis=1)
        else:
            t = np.clip(((v - a) @ ab) / denom, 0.0, 1.0)
            proj = a[None, :] + t[:, None] * ab[None, :]
            dists[:, j] = np.linalg.norm(v - proj, axis=1)
    logits = -dists / temperature
    logits -= logits.max(axis=1, keepdims=True)  # stabilize
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def forward_kinematics(skel: Skeleton, local_rotations: np.ndarray) -> np.ndarray:
    """Global joint transforms from per-joint local rotations.

    ``local_rotations`` is ``(J, 3, 3)``. Returns ``(J, 4, 4)`` world transforms.
    With identity rotations the joints stay at their rest positions.
    """
    r = np.asarray(local_rotations, dtype=np.float64)
    globals_ = np.zeros((skel.n_joints, 4, 4))
    for j in range(skel.n_joints):
        p = skel.parents[j]
        rest_offset = skel.joints[j] - (skel.joints[p] if p >= 0 else np.zeros(3))
        local = np.eye(4)
        local[:3, :3] = r[j]
        local[:3, 3] = rest_offset
        globals_[j] = local if p < 0 else globals_[p] @ local
    return globals_


def skinning_matrices(skel: Skeleton, local_rotations: np.ndarray) -> np.ndarray:
    """Per-joint deformation matrices mapping rest vertices to the posed skeleton."""
    posed = forward_kinematics(skel, local_rotations)
    rest = forward_kinematics(skel, np.broadcast_to(np.eye(3), (skel.n_joints, 3, 3)))
    return posed @ np.linalg.inv(rest)


def linear_blend_skinning(vertices: np.ndarray, weights: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Deform vertices by a weighted blend of per-joint transforms (classic LBS).

    ``vertices`` ``(V, 3)``, ``weights`` ``(V, J)``, ``transforms`` ``(J, 4, 4)``.
    Identity transforms leave the mesh unchanged.
    """
    v = np.asarray(vertices, dtype=np.float64)
    homog = np.concatenate([v, np.ones((len(v), 1))], axis=1)  # (V, 4)
    # (V, J, 4) = each joint applied to every vertex, then weighted-summed.
    per_joint = np.einsum("jab,vb->vja", transforms, homog)
    blended = np.einsum("vj,vja->va", weights, per_joint)
    return blended[:, :3]


def rig_mesh(vertices: np.ndarray, *, n_joints: int = 5, temperature: float = 0.1):
    """Convenience: auto-skeleton + skinning weights for a mesh in one call."""
    skel = auto_skeleton_from_points(vertices, n_joints=n_joints)
    weights = skinning_weights(vertices, skel, temperature=temperature)
    return skel, weights
