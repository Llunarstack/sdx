"""
View-dependent radiance via spherical harmonics (SH).

The PBR/albedo color the field predicts is *view-independent* — a point looks the
same from every angle. Real materials don't: metal, glass, wet surfaces, and
polished wood change color with the viewing direction (specular highlights,
reflections, Fresnel). 3D Gaussian Splatting solves this with per-point spherical
harmonics — degree-3 SH, i.e. 16 coefficients per RGB channel — evaluated against
the view direction at render time. This module gives SDX the same capability.

SH is the right basis because it's a compact, smooth, rotation-friendly way to
encode "how does the color of this point vary over the sphere of directions."
Degree 0 is a constant (matches plain albedo); each higher degree adds finer
directional variation (sharper highlights).

Reference: 3D Gaussian Splatting (Kerbl et al. 2023) uses degree-3 SH for
appearance; the real-SH basis constants below are the standard ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

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
