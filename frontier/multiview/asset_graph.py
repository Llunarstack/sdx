"""
Semantic Asset Graph (SAG) — non-destructive, part-wise 3D editing.

The original SDX-3D pain point: "make the hilt a mantis claw" and the whole sword
regenerates, losing the blade you liked. The fix is to stop treating an object as
one monolithic field. An asset is a graph of named *parts*, each its own SDF
occupying a region of space. The object is their union (CSG min of signed
distances). Editing or swapping one part leaves every other part's geometry
*bit-for-bit identical*, because they are independent fields — the blade simply
isn't recomputed when you touch the hilt.

Parts can be analytic SDFs or wrapped tri-plane fields (:func:`part_from_field`),
so the same graph composes hand-authored primitives and generated content.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .mesh_export import extract_mesh

NumpySdf = Callable[[np.ndarray], np.ndarray]  # (N, 3) -> (N,)


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
    sdf_fn: NumpySdf
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

    def edit_part(self, name: str, sdf_fn: NumpySdf, region: AABB | None = None) -> AssetGraph:
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
