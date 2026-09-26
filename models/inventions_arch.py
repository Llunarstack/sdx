"""Architecture moonshots 6,8,9,10,17,91–99 — trainable scaffolds.

These modules are real PyTorch code (forward works) so they can be trained later.
They are not claiming pretrained weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RelationTransformerHead",
    "NegationTokenBank",
    "CountClassToken",
    "AttributeFirewallMask",
    "ObjectCentricDiTStub",
    "ConstraintProjection",
    "GlyphNativeTokenizer",
    "PhysicsContactPotential",
    "MemoryAugmentedKV",
    "LayoutDiffusionPrior",
    "DreamCriticStub",
    "CausalEditGraph",
    "concept_algebra_loss",
]


class RelationTransformerHead(nn.Module):
    """#6 Auxiliary pairwise spatial relation logits."""

    def __init__(self, dim: int = 256, n_relations: int = 8, n_heads: int = 4):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True),
            num_layers=2,
        )
        self.proj = nn.Linear(dim, n_relations)

    def forward(self, slot_tokens: torch.Tensor) -> torch.Tensor:
        # slot_tokens: B, N, D → B, N, N, R via pairwise diff
        h = self.enc(slot_tokens)
        b, n, d = h.shape
        hi = h.unsqueeze(2).expand(b, n, n, d)
        hj = h.unsqueeze(1).expand(b, n, n, d)
        return self.proj(hi - hj)


class NegationTokenBank(nn.Module):
    """#8 Learned ¬concept embeddings."""

    def __init__(self, vocab: int = 4096, dim: int = 768):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)

    def forward(self, concept_ids: torch.Tensor) -> torch.Tensor:
        return -self.emb(concept_ids)  # negation as sign flip + learned bank


class CountClassToken(nn.Module):
    """#9 Discrete count 0..8 class token."""

    def __init__(self, dim: int = 768, max_count: int = 8):
        super().__init__()
        self.emb = nn.Embedding(max_count + 1, dim)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        return self.emb(counts.clamp(0, self.emb.num_embeddings - 1))


class AttributeFirewallMask:
    """#10 Build cross-attn masks so attr tokens only attend owner noun span."""

    @staticmethod
    def build(seq_len: int, owner_spans: list[tuple[int, int]], attr_indices: list[int]) -> torch.Tensor:
        # mask: True = allow attend
        m = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for ai in attr_indices:
            m[ai, :] = False
            for s, e in owner_spans:
                m[ai, s:e] = True
        return m


class ObjectCentricDiTStub(nn.Module):
    """#91 Slots as first-class tokens (minimal stub)."""

    def __init__(self, dim: int = 256, n_slots: int = 8, depth: int = 4):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(n_slots, dim) * 0.02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(dim, 4, batch_first=True) for _ in range(depth)])
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, T, D patches; prepend slots
        b = x.shape[0]
        slots = self.slots.unsqueeze(0).expand(b, -1, -1)
        h = torch.cat([slots, x], dim=1)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


class ConstraintProjection(nn.Module):
    """#92 Project denoise update to reduce constraint energy (scaffold)."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, dx: torch.Tensor, constraint_grad: torch.Tensor) -> torch.Tensor:
        # dx ← dx - λ ∇E
        return dx - self.scale * constraint_grad


class GlyphNativeTokenizer(nn.Module):
    """#93 Bytes↔patch bridge stub."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.byte_emb = nn.Embedding(256, dim)
        self.to_patch = nn.Linear(dim, dim)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        return self.to_patch(self.byte_emb(byte_ids.clamp(0, 255)))


def physics_contact_potential(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    """#94 Soft collision energy between disks (B,N,2) centers."""
    # pairwise distances
    d = torch.cdist(centers, centers)
    min_d = radii.unsqueeze(-1) + radii.unsqueeze(-2)
    overlap = F.relu(min_d - d)
    overlap = overlap.triu(diagonal=1)
    return overlap.pow(2).sum(dim=(-1, -2))


class PhysicsContactPotential(nn.Module):
    def forward(self, centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        return physics_contact_potential(centers, radii)


class MemoryAugmentedKV(nn.Module):
    """#98 Retrieve board patch memories into KV."""

    def __init__(self, dim: int = 256, mem_size: int = 64):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(mem_size, dim) * 0.02)
        self.q = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # return K,V from memory attended by x
        q = self.q(x)
        attn = torch.softmax(q @ self.mem.t() / (q.shape[-1] ** 0.5), dim=-1)
        v = attn @ self.mem
        return self.mem.unsqueeze(0).expand(x.shape[0], -1, -1), v


class LayoutDiffusionPrior(nn.Module):
    """#17 Two-stage: denoise box coords before pixels (tiny MLP stub)."""

    def __init__(self, n_boxes: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_boxes * 4 + 1, 128), nn.SiLU(), nn.Linear(128, n_boxes * 4))

    def forward(self, boxes_flat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([boxes_flat, t], dim=-1))


class DreamCriticStub(nn.Module):
    """#97 Critic that scores failure modes (27 logits)."""

    def __init__(self, dim: int = 256, n_modes: int = 27):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, n_modes))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


@dataclass
class CausalEditNode:
    node_id: str
    depends_on: list[str]
    region: str
    prompt: str


class CausalEditGraph:
    """#96 Edit graph: regenerate only dependents."""

    def __init__(self) -> None:
        self.nodes: dict[str, CausalEditNode] = {}

    def add(self, node: CausalEditNode) -> None:
        self.nodes[node.node_id] = node

    def dirty_closure(self, changed: str) -> list[str]:
        out = {changed}
        changed_flag = True
        while changed_flag:
            changed_flag = False
            for nid, n in self.nodes.items():
                if nid in out:
                    continue
                if any(d in out for d in n.depends_on):
                    out.add(nid)
                    changed_flag = True
        return list(out)


def concept_algebra_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    emb_ab: torch.Tensor,
) -> torch.Tensor:
    """#7 Encourage emb(A)+emb(B) ≈ emb(A∧B) for held-out recombinations."""
    return F.mse_loss(emb_a + emb_b, emb_ab)
