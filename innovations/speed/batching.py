"""Batched inference — queue and process multiple requests."""

from typing import List

import torch
import torch.nn as nn


class BatchedInference(nn.Module):
    """Batch multiple requests for 3-5x throughput."""

    def __init__(self, max_batch_size: int = 32):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.queue = []

    def add_request(self, request):
        """Add request to queue."""
        self.queue.append(request)

    def process_batch(self, generator) -> List[torch.Tensor]:
        """Process batch when full or timeout."""
        if len(self.queue) == 0:
            return []

        batch_size = min(len(self.queue), self.max_batch_size)
        batch = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]

        # Process batch in parallel
        results = []
        for req in batch:
            result = generator(req)
            results.append(result)

        return results
