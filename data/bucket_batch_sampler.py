"""Batch sampler that keeps one resolution bucket per batch (aspect-ratio bucketing).

``Text2ImageDataset`` with ``resolution_buckets`` assigns every sample a fixed bucket by aspect ratio
(``_bucket_assign``); each epoch reshuffles within buckets and the batch order.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from .t2i_dataset import Text2ImageDataset


class ResolutionBucketBatchSampler(Sampler[list[int]]):
    """Yields index batches that share a bucket id, so ``collate_t2i`` can stack them."""

    def __init__(
        self,
        dataset: Text2ImageDataset,
        batch_size: int,
        *,
        drop_last: bool = True,
        shuffle_batches: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        if not getattr(dataset, "resolution_buckets", None):
            raise ValueError("ResolutionBucketBatchSampler requires dataset.resolution_buckets")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle_batches = shuffle_batches
        self.generator = generator
        groups: defaultdict[int, list[int]] = defaultdict(list)
        for i, bucket in enumerate(dataset._bucket_assign):
            groups[bucket].append(i)
        self._groups = dict(groups)

    def __len__(self) -> int:
        bs = self.batch_size
        return sum(len(idxs) // bs if self.drop_last else -(-len(idxs) // bs) for idxs in self._groups.values())

    def _shuffled(self, items: list) -> list:
        perm = torch.randperm(len(items), generator=self.generator).tolist()
        return [items[j] for j in perm]

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        for idxs in self._groups.values():
            idxs = self._shuffled(idxs)
            for j in range(0, len(idxs), self.batch_size):
                chunk = idxs[j : j + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk)
        if self.shuffle_batches:
            batches = self._shuffled(batches)
        yield from batches
