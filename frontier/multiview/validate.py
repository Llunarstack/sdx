"""
Differentiable Simulation Validator (DSV) — the "reality checking" pass.

From the original SDX-3D notes: before an asset ships, a critic should ask "does
this object physically exist?" — is it watertight, what does it weigh, where is
its center of mass, will it stand up or tip over. This runs those checks on an
extracted mesh so generation can flag or fix problems instead of exporting broken
geometry (holes that ruin 3D prints, top-heavy models that fall over).

Pure geometry on numpy arrays — no simulator dependency. "Differentiable" is
aspirational for now (these are analytic quantities you *can* differentiate); the
current use is validation and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
