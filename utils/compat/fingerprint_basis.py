"""
Source base-model **fingerprint** extraction and removal for the adapter bridge.

When a foreign LoRA/LoHa/LoKr is bridged onto sdx (see ``adapter_bridge``), its
reconstructed ΔW carries two entangled things:

  1. the **concept** you want (a character, an outfit, a style), and
  2. the **source base-model fingerprint** — output directions that recur across
     *concept-unrelated* adapters of the same base (SD1.5 softness, Pony /
     Illustrious anime tell, Flux waxy skin) plus generic LoRA overfit.

The concept lives in adapter-specific directions; the fingerprint is the part
that is *common across many adapters of the same source*. So we estimate the
fingerprint as the dominant shared subspace of the ΔW **output space** over a
corpus of same-source adapters, then subtract that subspace from any new bridged
ΔW — keeping the concept, dropping the "this looks like {base model}" plague.

Key detail: architecture family (``asset_sniffer``) cannot separate Pony vs
Illustrious vs base SDXL — they share a UNet. The fingerprint is *aesthetic*, so
it must be measured empirically per corpus, which is exactly what this does.

Math
----
Each same-source adapter contributes a projected ΔW (in the sdx target layer's
output space). Its dominant left-singular vectors are the output directions it
writes. Normalising each adapter to unit energy (so no single adapter dominates)
and pooling those weighted directions across the corpus, the directions that
*recur* survive an SVD while per-adapter concept directions average down. The
top-k pooled left vectors form an orthonormal fingerprint basis ``F`` (out×k).

Removal is an orthogonal projection of ΔW's rows out of ``F``::

    ΔW_clean = ΔW - strength · F (Fᵀ ΔW)

``strength`` in [0, 1] dials from "keep everything" to "fully remove the shared
directions". Over-subtraction erodes the concept too (an anime LoRA's charm is
partly its base look), so strength is meant to be tuned by the critic loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from utils.compat.adapter_bridge import collect_foreign_deltas, map_sdx_targets
from utils.compat.lycoris_math import project_delta

__all__ = [
    "FingerprintBasis",
    "build_fingerprint_basis",
    "fingerprint_directions",
    "project_out",
    "depth_bucket",
]


def depth_bucket(depth_fraction: float, num_buckets: int) -> int:
    """Quantise a 0..1 depth fraction into ``num_buckets`` buckets."""
    b = int(float(depth_fraction) * num_buckets)
    return max(0, min(num_buckets - 1, b))


@dataclass(slots=True)
class FingerprintBasis:
    """Per-(role, depth-bucket) orthonormal fingerprint directions for a source."""

    family: str = "unknown"
    num_buckets: int = 6
    n_adapters: int = 0
    # (role, bucket) -> F of shape (out_features, k), columns orthonormal.
    directions: dict[tuple[str, int], torch.Tensor] = field(default_factory=dict)

    def get(self, role: str, depth_fraction: float) -> torch.Tensor | None:
        return self.directions.get((role, depth_bucket(depth_fraction, self.num_buckets)))

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "family": self.family,
            "num_buckets": self.num_buckets,
            "n_adapters": self.n_adapters,
            "directions": {f"{role}@{bucket}": F for (role, bucket), F in self.directions.items()},
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path) -> FingerprintBasis:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        directions: dict[tuple[str, int], torch.Tensor] = {}
        for key, F in payload.get("directions", {}).items():
            role, _, bucket = key.rpartition("@")
            directions[(role, int(bucket))] = F
        return cls(
            family=payload.get("family", "unknown"),
            num_buckets=int(payload.get("num_buckets", 6)),
            n_adapters=int(payload.get("n_adapters", 0)),
            directions=directions,
        )


def fingerprint_directions(deltas: list[torch.Tensor], k: int) -> torch.Tensor | None:
    """
    Extract the top-``k`` shared output directions from a list of ΔW matrices.

    Each ``delta`` is (out, in). Every adapter is normalised to unit Frobenius
    norm so a single strong adapter cannot dominate; its top left-singular
    vectors (scaled by singular value) are pooled, then an SVD of the pool
    returns the recurring directions. Returns an orthonormal ``(out, k)`` tensor,
    or ``None`` if there is nothing usable.
    """
    if not deltas:
        return None
    out_dim = deltas[0].shape[0]
    cols: list[torch.Tensor] = []
    for d in deltas:
        d = d.float()
        if d.ndim != 2 or d.shape[0] != out_dim:
            continue
        fro = torch.linalg.norm(d)
        if not torch.isfinite(fro) or fro <= 1e-12:
            continue
        d = d / fro
        # Left singular vectors = output directions this adapter writes into.
        u, s, _ = torch.linalg.svd(d, full_matrices=False)
        r = min(k, u.shape[1])
        cols.append(u[:, :r] * s[:r].clamp(min=0).sqrt().unsqueeze(0))
    if not cols:
        return None
    pool = torch.cat(cols, dim=1)  # (out, sum_r)
    u, s, _ = torch.linalg.svd(pool, full_matrices=False)
    kk = max(1, min(int(k), u.shape[1]))
    F = u[:, :kk]
    # Guard: drop near-zero-energy columns (no recurring structure there).
    keep = s[:kk] > (s[0] * 1e-3 if s.numel() else 0.0)
    if keep.any():
        F = F[:, keep]
    return F.contiguous()


def project_out(
    delta: torch.Tensor,
    basis: FingerprintBasis | None,
    role: str,
    depth_fraction: float,
    strength: float,
) -> torch.Tensor:
    """
    Remove the source fingerprint from a bridged ΔW (out, in).

    ``ΔW_clean = ΔW - strength · F (Fᵀ ΔW)``. No-op when strength <= 0 or no
    basis exists for this (role, depth). Only the fingerprint rows matching the
    delta's out dimension are used, so mismatched buckets are safely skipped.
    """
    if basis is None or strength <= 0.0:
        return delta
    F = basis.get(role, depth_fraction)
    if F is None or F.shape[0] != delta.shape[0]:
        return delta
    F = F.to(dtype=delta.dtype, device=delta.device)
    coeff = F.transpose(0, 1) @ delta  # (k, in)
    return delta - float(min(1.0, max(0.0, strength))) * (F @ coeff)


def build_fingerprint_basis(
    model: torch.nn.Module,
    adapter_paths: list[str | Path],
    *,
    family: str = "unknown",
    rank: int = 32,
    num_buckets: int = 6,
    k: int = 6,
) -> FingerprintBasis:
    """
    Estimate a source fingerprint from a corpus of same-source adapters.

    Each adapter's foreign deltas are projected into the matched sdx target
    layer's output space (mirroring ``bridge_apply``), grouped by (role,
    depth-bucket), and reduced to shared directions via
    :func:`fingerprint_directions`. Pass adapters that are *concept-diverse* but
    share a base model — the more varied the concepts, the cleaner the
    fingerprint (concept directions cancel, base-model tells survive).
    """
    targets = map_sdx_targets(model)
    grouped: dict[tuple[str, int], list[torch.Tensor]] = {}
    n_used = 0
    for path in adapter_paths:
        try:
            deltas = collect_foreign_deltas(path)
        except Exception:
            continue
        used_any = False
        for fd in deltas:
            candidates = targets.get(fd.role) or []
            if not candidates:
                continue
            _, _, module = min(candidates, key=lambda c: abs(c[1] - fd.depth_fraction))
            base = getattr(module, "linear", module)
            out_f, in_f = int(base.out_features), int(base.in_features)
            # Match bridge_apply's fused-qkv slicing so basis dims line up.
            proj_out = out_f
            if fd.role in ("q", "k", "v") and out_f == 3 * in_f:
                proj_out = out_f // 3
            try:
                delta = project_delta(fd.delta, (proj_out, in_f), rank=rank)
            except Exception:
                continue
            grouped.setdefault((fd.role, depth_bucket(fd.depth_fraction, num_buckets)), []).append(delta)
            used_any = True
        n_used += 1 if used_any else 0

    directions: dict[tuple[str, int], torch.Tensor] = {}
    for key, deltas in grouped.items():
        F = fingerprint_directions(deltas, k=k)
        if F is not None and F.numel() > 0:
            directions[key] = F
    return FingerprintBasis(family=family, num_buckets=num_buckets, n_adapters=n_used, directions=directions)
