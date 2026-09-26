"""
Tests for source base-model fingerprint extraction/removal (utils/compat/fingerprint_basis.py)
and its wiring into the adapter bridge + anti-AI routing.

The math tests plant a known shared "fingerprint" output direction across many
synthetic adapters and verify it is recovered and can be subtracted while a
per-adapter "concept" direction survives.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from models.anti_ai_naturalness import SOURCE_FAMILY_BIAS, MediumProfile, StyleFamilyRouter
from utils.compat.fingerprint_basis import (
    FingerprintBasis,
    build_fingerprint_basis,
    depth_bucket,
    fingerprint_directions,
    project_out,
)


def test_fingerprint_directions_recovers_shared_direction():
    torch.manual_seed(0)
    out_dim, in_dim, n = 64, 32, 24
    f = torch.randn(out_dim)
    f = f / f.norm()  # planted shared output direction
    deltas = []
    for _ in range(n):
        a = torch.randn(in_dim)
        concept = 0.6 * torch.randn(out_dim, in_dim)  # per-adapter, random output dirs
        delta = torch.outer(f, a) * 3.0 + concept  # strong shared f + concept
        deltas.append(delta)
    F = fingerprint_directions(deltas, k=4)
    assert F is not None and F.shape[0] == out_dim
    # Top recovered direction should align with the planted fingerprint.
    align = float(torch.abs(F[:, 0] @ f))
    assert align > 0.9, f"alignment too low: {align}"


def test_project_out_removes_fingerprint_keeps_concept():
    torch.manual_seed(1)
    out_dim, in_dim = 48, 24
    f = torch.randn(out_dim)
    f = f / f.norm()
    basis = FingerprintBasis(family="test", num_buckets=6, directions={("mlp_in", 0): f.unsqueeze(1)})
    a = torch.randn(in_dim)
    concept = torch.randn(out_dim, in_dim)
    # Remove any concept component along f so we can check it's preserved.
    concept = concept - torch.outer(f, f @ concept)
    delta = torch.outer(f, a) * 2.0 + concept
    cleaned = project_out(delta, basis, "mlp_in", depth_fraction=0.0, strength=1.0)
    # Fingerprint component along f should be ~gone.
    resid_fp = float(torch.linalg.norm(f @ cleaned))
    assert resid_fp < 1e-4, resid_fp
    # Concept (orthogonal to f) should be preserved.
    assert torch.allclose(cleaned, concept, atol=1e-4)
    # strength=0 is a no-op.
    assert torch.equal(project_out(delta, basis, "mlp_in", 0.0, 0.0), delta)
    # Unknown role / mismatched dim is a safe no-op.
    assert torch.equal(project_out(delta, basis, "qkv", 0.0, 1.0), delta)


def test_basis_save_load_roundtrip():
    b = FingerprintBasis(
        family="sdxl",
        num_buckets=4,
        n_adapters=10,
        directions={("qkv", 1): torch.randn(12, 3), ("mlp_in", 2): torch.randn(20, 5)},
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fp.pt"
        b.save(p)
        loaded = FingerprintBasis.load(p)
    assert loaded.family == "sdxl" and loaded.num_buckets == 4 and loaded.n_adapters == 10
    assert set(loaded.directions) == {("qkv", 1), ("mlp_in", 2)}
    assert torch.allclose(loaded.directions[("mlp_in", 2)], b.directions[("mlp_in", 2)])


def test_depth_bucket():
    assert depth_bucket(0.0, 6) == 0
    assert depth_bucket(0.99, 6) == 5
    assert depth_bucket(1.0, 6) == 5  # clamped
    assert depth_bucket(0.5, 6) == 3


def test_source_family_routing_bias():
    router = StyleFamilyRouter()
    base = router.weights_for(MediumProfile(family="2d"))
    flux = router.weights_for(MediumProfile(family="2d"), source_family="flux")
    # Flux bias boosts texture and color.
    assert flux["texture"] > base["texture"]
    assert flux["color"] > base["color"]
    assert abs(flux["texture"] - base["texture"] * SOURCE_FAMILY_BIAS["flux"]["texture"]) < 1e-6
    # Unknown source is a no-op.
    assert router.weights_for(MediumProfile(family="2d"), source_family="nope") == base


# --- Bridge integration --------------------------------------------------------


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)


class _Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = _Attn(d)
        self.mlp = _Mlp(d)


class _Model(nn.Module):
    def __init__(self, d=16, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(d) for _ in range(depth)])


def _synthetic_fc1_adapter(d: int, seed: int, shared_f: torch.Tensor) -> dict:
    """A kohya-style LoRA on mlp.fc1 that writes into a shared output direction."""
    g = torch.Generator().manual_seed(seed)
    out_f, in_f, rank = 4 * d, d, 4
    up = torch.randn(out_f, rank, generator=g) * 0.2
    up[:, 0] = shared_f * 1.5  # planted fingerprint column, shared across adapters
    down = torch.randn(rank, in_f, generator=g) * 0.2
    return {
        "lora_unet_blocks_0_mlp_fc1.lora_down.weight": down,
        "lora_unet_blocks_0_mlp_fc1.lora_up.weight": up,
        "lora_unet_blocks_0_mlp_fc1.alpha": torch.tensor(float(rank)),
    }


def test_bridge_build_and_apply_with_fingerprint():
    from utils.compat.adapter_bridge import bridge_apply

    d = 16
    shared_f = torch.randn(4 * d)
    shared_f = shared_f / shared_f.norm()
    corpus = [_synthetic_fc1_adapter(d, s, shared_f) for s in range(12)]

    basis = build_fingerprint_basis(_Model(d), corpus, family="sdxl", rank=4, num_buckets=6, k=4)
    assert basis.n_adapters == 12
    assert ("mlp_in", 0) in basis.directions
    F = basis.directions[("mlp_in", 0)]
    assert F.shape[0] == 4 * d
    assert float(torch.abs(F[:, 0] @ shared_f)) > 0.85  # recovered the planted tell

    # Bridge a fresh adapter with and without fingerprint stripping; both must run.
    adapter = _synthetic_fc1_adapter(d, 999, shared_f)
    m0 = _Model(d)
    rep0 = bridge_apply(m0, adapter, scale=1.0, rank=4)
    assert rep0.mapped > 0
    m1 = _Model(d)
    rep1 = bridge_apply(m1, adapter, scale=1.0, rank=4, fingerprint=basis, strip_source_fingerprint=0.9)
    assert rep1.mapped == rep0.mapped  # stripping doesn't change coverage

    # The stripped model's fc1 adapter should differ from the un-stripped one.
    def _fc1_delta(model):
        # Compare the adapter's effect through the wrapped layer's forward.
        mod = model.blocks[0].mlp.fc1
        base = getattr(mod, "linear", mod)
        return mod(torch.eye(base.in_features))

    with torch.no_grad():
        out0 = _fc1_delta(m0)
        out1 = _fc1_delta(m1)
    assert not torch.allclose(out0, out1, atol=1e-4), "fingerprint stripping had no effect"


if __name__ == "__main__":
    test_fingerprint_directions_recovers_shared_direction()
    print("[ok] recover shared direction")
    test_project_out_removes_fingerprint_keeps_concept()
    print("[ok] project out")
    test_basis_save_load_roundtrip()
    print("[ok] save/load")
    test_depth_bucket()
    print("[ok] depth bucket")
    test_source_family_routing_bias()
    print("[ok] source routing bias")
    test_bridge_build_and_apply_with_fingerprint()
    print("[ok] bridge integration")
    print("\nFingerprint stripping verified.")
