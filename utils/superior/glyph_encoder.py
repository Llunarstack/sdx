"""
Deterministic **byte-level** text sidecar encoder (ByT5-class experiments without HF ByT5).

Maps UTF-8 bytes to learned-hash features — not SOTA glyph OCR, but a real tensor path for
``GlyphToCondProjector`` in ``utils/generation/inference_research_hooks.py``.
"""

from __future__ import annotations

import hashlib
from typing import List

import torch


class ByteHashGlyphEncoder:
    """
    Encode strings to ``(B, L, D)`` via hashed byte n-grams.

    Same string → same tensor (reproducible). Suitable for residual add to T5 states.
    """

    def __init__(self, embed_dim: int = 64, max_bytes: int = 256) -> None:
        self.embed_dim = int(embed_dim)
        self.max_bytes = int(max_bytes)

    def _hash_vec(self, token: bytes) -> torch.Tensor:
        h = hashlib.sha256(token).digest()
        # Expand digest to embed_dim floats in [-1, 1]
        reps = (self.embed_dim + len(h) - 1) // len(h)
        buf = (h * reps)[: self.embed_dim]
        arr = torch.tensor([b / 127.5 - 1.0 for b in buf], dtype=torch.float32)
        return arr

    def encode_utf8(self, texts: List[str], device: torch.device) -> torch.Tensor:
        b = max(1, len(texts))
        out = torch.zeros((b, self.max_bytes, self.embed_dim), device=device, dtype=torch.float32)
        for bi, text in enumerate(texts[:b]):
            raw = (text or "").encode("utf-8", errors="ignore")[: self.max_bytes]
            for i, byte in enumerate(raw):
                out[bi, i] = self._hash_vec(bytes([byte]))
            # Bigram mix on adjacent bytes
            for i in range(len(raw) - 1):
                out[bi, i] = out[bi, i] + 0.25 * self._hash_vec(raw[i : i + 2]).to(device)
        return out


__all__ = ["ByteHashGlyphEncoder"]
