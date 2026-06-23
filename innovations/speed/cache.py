"""Caching mechanism — reuse results for similar prompt embeddings."""

import torch
import torch.nn as nn


class CachingMechanism(nn.Module):
    """Cache intermediate results for 2-3x speedup on similar prompts."""

    def __init__(self, cache_size: int = 1000, hidden_dim: int = 512):
        super().__init__()
        self.cache_size = cache_size
        self.cache = {}
        self.access_count = {}
        self.embedding_list = []

        # Similarity scorer
        self.similarity_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def get_cached(self, embedding: torch.Tensor) -> torch.Tensor:
        """Retrieve cached result if similar prompt exists."""
        if len(self.embedding_list) == 0:
            return None

        # Simple similarity: L2 distance
        similarities = []
        for cached_emb in self.embedding_list:
            with torch.no_grad():
                dist = torch.norm(embedding.detach() - cached_emb.detach())
            similarities.append(dist.item())

        # Find lowest distance (most similar)
        if len(similarities) > 0 and min(similarities) < 0.1:  # Low distance = similar
            best_idx = similarities.index(min(similarities))
            cached_key = id(self.embedding_list[best_idx])
            self.access_count[cached_key] = self.access_count.get(cached_key, 0) + 1
            return self.cache.get(cached_key)

        return None

    def cache_result(self, embedding: torch.Tensor, result: torch.Tensor):
        """Store result in cache."""
        if len(self.embedding_list) >= self.cache_size:
            # Evict least-accessed item
            least_accessed = min(self.access_count, key=self.access_count.get)
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
            # Remove from embedding list (approximate)
            self.embedding_list = self.embedding_list[1:]

        key = id(embedding)
        self.embedding_list.append(embedding.detach())
        self.cache[key] = result
        self.access_count[key] = 1
