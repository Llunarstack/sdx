"""Tests for SDF -> mesh extraction (marching tetrahedra).

Correctness is checked against an analytic sphere: the extracted surface must be
a closed manifold (Euler characteristic 2 for a genus-0 surface) whose vertices
lie on the sphere.
"""

from __future__ import annotations

import numpy as np
from frontier.multiview import extract_mesh, write_obj


def _sphere_sdf(radius: float):
    def fn(points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points, axis=1) - radius

    return fn


def test_sphere_mesh_is_nonempty():
    verts, faces = extract_mesh(_sphere_sdf(0.6), resolution=24)
    assert len(verts) > 0
    assert len(faces) > 0
    assert faces.max() < len(verts)  # faces index valid vertices


def test_sphere_vertices_lie_on_surface():
    radius = 0.6
    verts, _ = extract_mesh(_sphere_sdf(radius), resolution=32)
    r = np.linalg.norm(verts, axis=1)
    cell = 2.0 / 32
    # Linear interpolation puts vertices within ~one cell of the true surface.
    assert np.abs(r - radius).max() < cell


def test_sphere_is_closed_manifold():
    """Euler characteristic V - E + F == 2 for a watertight genus-0 sphere."""
    verts, faces = extract_mesh(_sphere_sdf(0.6), resolution=24)
    v = len(verts)
    f = len(faces)
    edges = set()
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            edges.add((min(x, y), max(x, y)))
    e = len(edges)
    assert v - e + f == 2, f"not closed: V={v} E={e} F={f}, chi={v - e + f}"


def test_every_edge_shared_by_two_faces():
    """A watertight mesh has no boundary edges — each is used exactly twice."""
    _, faces = extract_mesh(_sphere_sdf(0.6), resolution=24)
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for x, y in ((a, b), (b, c), (c, a)):
            key = (min(x, y), max(x, y))
            counts[key] = counts.get(key, 0) + 1
    assert all(n == 2 for n in counts.values())


def test_empty_field_yields_empty_mesh():
    # SDF positive everywhere -> no surface.
    verts, faces = extract_mesh(lambda p: np.ones(len(p)), resolution=8)
    assert len(verts) == 0
    assert len(faces) == 0


def test_write_obj_roundtrip(tmp_path):
    verts, faces = extract_mesh(_sphere_sdf(0.6), resolution=16)
    out = tmp_path / "sphere.obj"
    colors = np.abs(verts)  # any [0,1]-ish colors
    write_obj(str(out), verts, faces, vertex_colors=colors)
    text = out.read_text(encoding="utf-8")
    assert text.count("\nv ") + text.startswith("v ") >= len(verts) - 1
    assert "f " in text
    # face indices are 1-based and within range
    for line in text.splitlines():
        if line.startswith("f "):
            idx = [int(t) for t in line.split()[1:]]
            assert all(1 <= i <= len(verts) for i in idx)
