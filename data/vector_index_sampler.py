"""
Vector-indexed dataset sampling: **similarity-based** batch construction for training.

- **In-memory** (NumPy): cosine / L2 top-k, hard-negative mining — no extra deps.
- **Qdrant** (optional): ``pip install qdrant-client`` + running Qdrant; store embeddings and
  query by vector for "focus on hard / similar" sampling strategies.

This does **not** compute CLIP/ViT embeddings; pass precomputed matrices or ingest via your own job.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

JsonlRow = Dict[str, Any]


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


@dataclass
class InMemoryVectorIndex:
    """
    Embeddings shape ``(N, D)``, parallel list of manifest rows or path strings.
    """

    embeddings: np.ndarray
    rows: List[JsonlRow]
    normalized: bool = False

    def __post_init__(self) -> None:
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be (N, D)")
        if len(self.rows) != self.embeddings.shape[0]:
            raise ValueError("rows length must match N")
        if self.normalized:
            self._z = self.embeddings
        else:
            self._z = _l2_normalize_rows(self.embeddings)

    def topk_cosine(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) for k largest cosine similarities."""
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        q = _l2_normalize_rows(q)
        sims = (self._z @ q.T).reshape(-1)
        k = min(k, sims.shape[0])
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]

    def sample_batch_biased(
        self,
        rng: random.Random,
        *,
        batch_size: int,
        similarity_focus: float = 0.5,
        k_neighbors: int = 32,
    ) -> List[JsonlRow]:
        """
        Mix uniform random rows with neighbors of a random anchor (focus on "dense" regions of embedding space).
        ``similarity_focus`` in [0,1]: fraction of batch drawn from neighbor pool.
        """
        n = len(self.rows)
        if n == 0:
            return []
        anchor = rng.randrange(n)
        neigh_idx, _ = self.topk_cosine(self.embeddings[anchor], max(k_neighbors, batch_size))
        pool = list(neigh_idx)
        out: List[JsonlRow] = []
        n_sim = int(round(batch_size * similarity_focus))
        n_uni = batch_size - n_sim
        for _ in range(n_sim):
            out.append(self.rows[rng.choice(pool)])
        for _ in range(n_uni):
            out.append(self.rows[rng.randrange(n)])
        rng.shuffle(out)
        return out


def load_jsonl_embeddings_npz(
    jsonl_path: Path,
    npz_path: Path,
) -> InMemoryVectorIndex:
    """
    Load ``.npz`` with ``embeddings`` (N,D) in **the same row order** as ``jsonl_path`` (non-empty lines).
    """
    z = np.load(npz_path)
    emb = np.asarray(z["embeddings"], dtype=np.float32)
    rows: List[JsonlRow] = []
    with jsonl_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if emb.shape[0] != len(rows):
        raise ValueError(f"npz N={emb.shape[0]} vs jsonl rows={len(rows)}")
    return InMemoryVectorIndex(embeddings=emb, rows=rows, normalized=False)


class QdrantVectorSampler:
    """
    Thin wrapper when ``qdrant-client`` is installed and a collection holds ``embedding`` + payload
    with manifest fields.
    """

    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:6333",
        collection: str = "sdx_training",
        vector_name: Optional[str] = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("pip install qdrant-client") from e
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._vector_name = vector_name

    def search(self, query: np.ndarray, k: int) -> List[Dict[str, Any]]:
        qv = query.astype(np.float32).reshape(-1).tolist()
        kwargs: Dict[str, Any] = {"collection_name": self._collection, "query_vector": qv, "limit": k}
        if self._vector_name:
            kwargs["using"] = self._vector_name
        hits = self._client.search(**kwargs)
        out = []
        for h in hits:
            pl = dict(h.payload or {})
            pl["_score"] = float(h.score)
            pl["_id"] = str(h.id)
            out.append(pl)
        return out

    def scroll_all_ids(self, batch: int = 256) -> List[Union[str, int]]:
        ids: List[Union[str, int]] = []
        offset = None
        while True:
            points, offset = self._client.scroll(self._collection, limit=batch, offset=offset, with_payload=False)
            if not points:
                break
            ids.extend(p.id for p in points)
            if offset is None:
                break
        return ids
