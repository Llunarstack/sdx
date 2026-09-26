"""Tests for view-dependent SH radiance and the auto-rigging machinery."""

from __future__ import annotations

import numpy as np
import torch
from frontier.multiview import (
    SHRadianceDecoder,
    auto_skeleton_from_points,
    detokenize_skeleton,
    eval_sh_basis,
    forward_kinematics,
    linear_blend_skinning,
    rig_mesh,
    sh_num_coeffs,
    skinning_matrices,
    skinning_weights,
    tokenize_skeleton,
)
from frontier.multiview.rigging import Skeleton

# --------------------------------------------------------------------------- #
# Spherical-harmonic view-dependent radiance
# --------------------------------------------------------------------------- #


def test_sh_coeff_counts():
    assert [sh_num_coeffs(d) for d in range(4)] == [1, 4, 9, 16]


def test_sh_degree0_is_constant_over_directions():
    d1 = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
    d2 = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
    b1 = eval_sh_basis(d1, 0)
    b2 = eval_sh_basis(d2, 0)
    assert b1.shape == (10, 1)
    assert torch.allclose(b1, b2)  # DC term ignores direction
    assert torch.allclose(b1, torch.full_like(b1, 0.28209479177387814))


def test_sh_basis_shape_and_direction_dependence():
    dirs = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    basis = eval_sh_basis(dirs, 3)
    assert basis.shape == (5, 16)
    # Different directions produce different higher-order responses.
    a = eval_sh_basis(torch.tensor([[1.0, 0.0, 0.0]]), 3)
    b = eval_sh_basis(torch.tensor([[0.0, 1.0, 0.0]]), 3)
    assert not torch.allclose(a, b)


def test_sh_radiance_degree0_is_view_independent():
    torch.manual_seed(0)
    dec = SHRadianceDecoder(feature_dim=16, degree=0)
    feat = torch.randn(4, 16)
    d1 = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    d2 = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    c1 = dec(feat, d1)
    c2 = dec(feat, d2)
    assert c1.shape == (4, 3)
    assert torch.allclose(c1, c2)  # no directional terms -> flat color


def test_sh_radiance_degree3_varies_with_view_and_is_bounded():
    torch.manual_seed(1)
    dec = SHRadianceDecoder(feature_dim=16, degree=3)
    feat = torch.randn(4, 16)
    d1 = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    d2 = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    c1, c2 = dec(feat, d1), dec(feat, d2)
    assert not torch.allclose(c1, c2)  # reflections move with the camera
    assert torch.all((c1 >= 0) & (c1 <= 1))


def test_sh_radiance_gradients_flow():
    dec = SHRadianceDecoder(feature_dim=8, degree=2)
    feat = torch.randn(3, 8, requires_grad=True)
    dirs = torch.nn.functional.normalize(torch.randn(3, 3), dim=-1)
    dec(feat, dirs).mean().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()


# --------------------------------------------------------------------------- #
# Auto-rigging: skeleton, tokenization, skinning, LBS, FK
# --------------------------------------------------------------------------- #


def _bar_points(n=400):
    """A long thin bar along +x — an obvious principal axis for rigging."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, n)
    y = rng.uniform(-0.05, 0.05, n)
    z = rng.uniform(-0.05, 0.05, n)
    return np.stack([x, y, z], axis=1)


def test_auto_skeleton_lies_on_principal_axis_and_chains():
    skel = auto_skeleton_from_points(_bar_points(), n_joints=5)
    assert skel.n_joints == 5
    # Joints should be spread along x and near-zero in y, z.
    assert np.ptp(skel.joints[:, 0]) > 1.0
    assert np.abs(skel.joints[:, 1:]).max() < 0.2
    # Parent chain: -1, 0, 1, 2, 3
    assert list(skel.parents) == [-1, 0, 1, 2, 3]


def test_skeleton_tokenization_roundtrip():
    skel = auto_skeleton_from_points(_bar_points(), n_joints=6)
    tokens = tokenize_skeleton(skel)
    assert len(tokens) == 6
    back = detokenize_skeleton(tokens)
    assert np.allclose(back.joints, skel.joints)
    assert np.array_equal(back.parents, skel.parents)


def test_skinning_weights_are_a_partition():
    pts = _bar_points()
    skel = auto_skeleton_from_points(pts, n_joints=5)
    w = skinning_weights(pts, skel)
    assert w.shape == (len(pts), 5)
    assert np.all(w >= 0)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_skinning_binds_vertex_to_nearest_joint():
    skel = auto_skeleton_from_points(_bar_points(), n_joints=5)
    # A vertex sitting right at the last joint should be dominated by it.
    v = skel.joints[-1][None, :]
    w = skinning_weights(v, skel, temperature=0.05)
    assert int(np.argmax(w[0])) == skel.n_joints - 1


def test_forward_kinematics_rest_pose_returns_joints():
    skel = auto_skeleton_from_points(_bar_points(), n_joints=5)
    rot = np.broadcast_to(np.eye(3), (skel.n_joints, 3, 3))
    g = forward_kinematics(skel, rot)
    assert np.allclose(g[:, :3, 3], skel.joints, atol=1e-9)


def test_lbs_identity_is_a_noop():
    pts = _bar_points()
    skel, w = rig_mesh(pts, n_joints=5)
    transforms = np.broadcast_to(np.eye(4), (skel.n_joints, 4, 4))
    out = linear_blend_skinning(pts, w, transforms)
    assert np.allclose(out, pts, atol=1e-9)


def test_lbs_bends_the_bar_when_a_joint_rotates():
    pts = _bar_points()
    skel, w = rig_mesh(pts, n_joints=5)
    # Rotate the last joint 90° about z. (The skeleton's axis sign is arbitrary,
    # so identify vertices by their binding weight, not by a hard-coded x-side.)
    rots = np.broadcast_to(np.eye(3), (skel.n_joints, 3, 3)).copy()
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    rots[-1] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    mats = skinning_matrices(skel, rots)
    posed = linear_blend_skinning(pts, w, mats)

    shift = np.linalg.norm(posed - pts, axis=1)
    bound = w[:, -1] > 0.5  # vertices controlled by the rotated joint
    free = w[:, -1] < 0.01  # vertices essentially unbound to it
    assert bound.any() and free.any()
    assert shift[bound].mean() > 0.05  # they actually swung
    assert shift[free].mean() < 0.2 * shift[bound].mean()  # the rest stayed ~put


def test_skeleton_validates_length_mismatch():
    import pytest

    with pytest.raises(ValueError):
        Skeleton(joints=np.zeros((3, 3)), parents=np.array([-1, 0]))
