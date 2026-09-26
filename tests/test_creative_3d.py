"""Tests for the creative SDX-3D features drawn from the original design notes:
perfect angles, the semantic asset graph, the reality-check validator, and
genetic cross-breeding.
"""

from __future__ import annotations

import numpy as np
import torch
from frontier.multiview import (
    AABB,
    AssetGraph,
    NeusRenderer,
    SemanticPart,
    SpatialLatentField,
    TriPlaneConfig,
    blend_conditioning,
    contact_sheet,
    mutate_conditioning,
    perfect_angles,
    slerp,
    spatial_graft,
    validate_mesh,
    viewpoint_rays,
)
from frontier.multiview.mesh_export import extract_mesh


def _sphere_sdf(radius: float, center=(0.0, 0.0, 0.0)):
    c = np.asarray(center)

    def fn(p: np.ndarray) -> np.ndarray:
        return np.linalg.norm(p - c, axis=1) - radius

    return fn


# --------------------------------------------------------------------------- #
# Perfect Angles
# --------------------------------------------------------------------------- #


def test_viewpoint_camera_positions():
    o, d = viewpoint_rays("front", radius=2.5, height=8, width=8)
    # Camera origin sits on the -Z axis at the requested radius, all rays share it.
    assert torch.allclose(o[0], torch.tensor([0.0, 0.0, -2.5]), atol=1e-5)
    assert torch.allclose(d.norm(dim=-1), torch.ones(64), atol=1e-5)


def test_top_view_up_vector_is_nondegenerate():
    # up is parallel to view dir for top/bottom; rays must still be finite & unit.
    _, d = viewpoint_rays("top", height=6, width=6)
    assert torch.isfinite(d).all()
    assert torch.allclose(d.norm(dim=-1), torch.ones(36), atol=1e-5)


def test_perfect_angles_returns_all_views_same_object():
    torch.manual_seed(0)
    field = SpatialLatentField(TriPlaneConfig(cond_dim=16, plane_channels=8, grid_res=16, feature_dim=16))
    renderer = NeusRenderer(n_samples=24)
    planes = field.encode(torch.zeros(1, 16))
    imgs = perfect_angles(field, planes, renderer, names=["front", "right", "top"], resolution=12)
    assert set(imgs) == {"front", "right", "top"}
    for im in imgs.values():
        assert im.shape == (12, 12, 3)
        assert (im >= 0).all() and (im <= 1).all()


def test_contact_sheet_tiles_images():
    imgs = [np.full((10, 10, 3), v, dtype=np.float32) for v in (0.2, 0.4, 0.6, 0.8)]
    sheet = contact_sheet(imgs, cols=2, pad=2)
    # 2x2 grid of 10px tiles + 3 pads of 2px each way.
    assert sheet.shape == (2 * 10 + 3 * 2, 2 * 10 + 3 * 2, 3)


# --------------------------------------------------------------------------- #
# Semantic Asset Graph — non-destructive editing
# --------------------------------------------------------------------------- #


def test_editing_one_part_leaves_others_bit_identical():
    graph = (
        AssetGraph()
        .add(SemanticPart("a", _sphere_sdf(0.25, (-0.5, 0, 0)), AABB.around((-0.5, 0, 0), 0.35)))
        .add(SemanticPart("b", _sphere_sdf(0.25, (0.5, 0, 0)), AABB.around((0.5, 0, 0), 0.35)))
    )
    v0, _ = graph.to_mesh(resolution=40)
    b_region = v0[v0[:, 0] > 0.1]
    b_before = np.array(sorted(map(tuple, np.round(b_region, 6))))

    # Edit only part "a" (grow it). Part "b" must not move at all.
    graph.edit_part("a", _sphere_sdf(0.32, (-0.5, 0, 0)))
    v1, _ = graph.to_mesh(resolution=40)
    b_after = np.array(sorted(map(tuple, np.round(v1[v1[:, 0] > 0.1], 6))))

    assert b_before.shape == b_after.shape
    assert np.array_equal(b_before, b_after)


def test_graph_union_contains_both_parts():
    graph = (
        AssetGraph()
        .add(SemanticPart("a", _sphere_sdf(0.25, (-0.5, 0, 0)), AABB.around((-0.5, 0, 0), 0.35)))
        .add(SemanticPart("b", _sphere_sdf(0.25, (0.5, 0, 0)), AABB.around((0.5, 0, 0), 0.35)))
    )
    v, _ = graph.to_mesh(resolution=40)
    assert (v[:, 0] < -0.1).any() and (v[:, 0] > 0.1).any()  # both lobes present


# --------------------------------------------------------------------------- #
# Reality-check validator
# --------------------------------------------------------------------------- #


def test_validator_on_sphere_matches_analytic_quantities():
    radius = 0.6
    v, f = extract_mesh(_sphere_sdf(radius), resolution=48)
    report = validate_mesh(v, f)
    assert report.watertight
    assert report.euler_characteristic == 2
    # Volume 4/3 pi r^3, area 4 pi r^2 — within a few % for this resolution.
    assert abs(report.volume - (4 / 3) * np.pi * radius**3) < 0.03
    assert abs(report.surface_area - 4 * np.pi * radius**2) < 0.1
    assert max(abs(c) for c in report.center_of_mass) < 0.02  # centered
    assert all(abs(d - 2 * radius) < 0.05 for d in report.dimensions)


def test_validator_flags_open_mesh():
    # A single triangle: open, degenerate volume -> warnings, not watertight.
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    f = np.array([[0, 1, 2]], dtype=np.int64)
    report = validate_mesh(v, f)
    assert not report.watertight
    assert report.warnings


# --------------------------------------------------------------------------- #
# Genetic cross-breeding
# --------------------------------------------------------------------------- #


def test_slerp_hits_endpoints():
    a = torch.randn(2, 16)
    b = torch.randn(2, 16)
    assert torch.allclose(slerp(a, b, 0.0), a, atol=1e-4)
    assert torch.allclose(slerp(a, b, 1.0), b, atol=1e-4)


def test_blend_conditioning_is_between_parents():
    a = torch.randn(1, 32)
    b = torch.randn(1, 32)
    mid = blend_conditioning(a, b, 0.5)
    assert mid.shape == a.shape
    assert not torch.allclose(mid, a) and not torch.allclose(mid, b)


def test_spatial_graft_takes_lower_from_a_upper_from_b():
    a = _sphere_sdf(0.5)  # bigger body (A) for the lower half
    b = _sphere_sdf(0.3)  # smaller body (B) for the upper half
    chimera = spatial_graft(a, b, axis=1, split=0.0, smooth=0.05)

    below = np.array([[0.0, -0.4, 0.0]])
    above = np.array([[0.0, 0.4, 0.0]])
    assert np.isclose(chimera(below), a(below), atol=1e-3)  # A dominates below seam
    assert np.isclose(chimera(above), b(above), atol=1e-3)  # B dominates above seam

    # And it still meshes to something watertight.
    v, f = extract_mesh(chimera, resolution=40)
    assert len(v) > 0
    assert validate_mesh(v, f).watertight


def test_mutation_perturbs_but_stays_close():
    cond = torch.zeros(1, 16)
    m = mutate_conditioning(cond, strength=0.1, seed=1)
    assert m.shape == cond.shape
    assert 0.0 < m.abs().mean() < 1.0
