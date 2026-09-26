"""
Kinematic co-synthesis — make a generated mesh animation-ready (auto-rigging).

A raw mesh is a dead shell: to animate it you need a *skeleton* (joints in a tree),
*skinning weights* (which joints move which vertices), and *linear blend skinning*
(deform the mesh from joint poses). Modern learned riggers — UniRig (SIGGRAPH
2025), MagicArticulate, Anymate — predict the skeleton with transformer "skeleton
tree tokenization" trained on rigged-model datasets.

This module implements the full production machinery that consumes such a skeleton
— tree tokenization, distance-based skinning, forward kinematics, and linear blend
skinning — plus a *heuristic* skeleton predictor (principal-axis chain) so the
whole pipeline runs end-to-end today. The heuristic stands in for the learned
UniRig-style predictor; swap it out when a trained model is available. Everything
is numpy to match the mesh side of the stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
