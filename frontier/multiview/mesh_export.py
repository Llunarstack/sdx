"""
Mesh extraction: turn the SDF field into a real triangle mesh (.obj).

Why marching *tetrahedra* and not marching cubes
------------------------------------------------
Marching cubes needs a 256-entry triangle table that is painful to transcribe
correctly and easy to get subtly wrong. Marching tetrahedra splits every grid
cube into 6 tetrahedra, each of which has only 16 sign cases with a tiny,
hand-verifiable table. Using the *same* cube→tet decomposition everywhere makes
the tessellation crack-free (watertight), which is exactly what you need for a
mesh you'll import into Blender / a game engine or send to a 3D printer.

It emits a few more triangles than marching cubes, but for a generative asset
that gets remeshed/decimated downstream that is a fine trade for correctness and
zero extra dependencies (numpy only — so it runs locally and on RunPod).

This is the ``SDFDecoder`` payoff from :mod:`frontier.multiview.triplane`: the
network predicts a signed distance field, and this turns that field into
geometry.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

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
