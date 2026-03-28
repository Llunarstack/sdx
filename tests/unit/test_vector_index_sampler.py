from __future__ import annotations

import random

import numpy as np

from data.vector_index_sampler import InMemoryVectorIndex


def test_in_memory_topk_and_batch():
    emb = np.eye(5, dtype=np.float32)
    rows = [{"i": i} for i in range(5)]
    idx = InMemoryVectorIndex(embeddings=emb, rows=rows, normalized=True)
    q = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ii, sc = idx.topk_cosine(q, 2)
    assert 0 in ii
    rng = random.Random(0)
    batch = idx.sample_batch_biased(rng, batch_size=4, similarity_focus=0.75, k_neighbors=3)
    assert len(batch) == 4
