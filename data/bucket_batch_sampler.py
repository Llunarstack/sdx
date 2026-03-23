"""Batch sampler that keeps **one resolution bucket per batch** (IMPROVEMENTS §1.1).

``Text2ImageDataset`` must have ``resolution_buckets`` set and ``_bucket_assign`` populated
via ``set_epoch`` before each training epoch.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Iterator, List, Optional

import torch
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from .t2i_dataset import Text2ImageDataset


class ResolutionBucketBatchSampler(Sampler[List[int]]):
    """
    Yields batches of indices that share the same bucket id so ``collate_t2i`` stacks tensors.
    """

    def __init__(
        self,
        dataset: "Text2ImageDataset",
        batch_size: int,
        *,
        drop_last: bool = True,
        shuffle_batches: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if not getattr(dataset, "resolution_buckets", None):
            raise ValueError("ResolutionBucketBatchSampler requires dataset.resolution_buckets")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle_batches = shuffle_batches
        self.generator = generator

    def __len__(self) -> int:
        groups: defaultdict[int, int] = defaultdict(int)
        for i in range(len(self.dataset)):
            groups[self.dataset._bucket_assign[i]] += 1
        total = 0
        for c in groups.values():
            if self.drop_last:
                total += c // self.batch_size
            else:
                total += (c + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self) -> Iterator[List[int]]:
        rng = self.generator
        groups: defaultdict[int, List[int]] = defaultdict(list)
        for i in range(len(self.dataset)):
            b = self.dataset._bucket_assign[i]
            groups[b].append(i)
        batches: List[List[int]] = []
        for idxs in groups.values():
            if rng is not None:
                perm = torch.randperm(len(idxs), generator=rng).tolist()
                idxs = [idxs[j] for j in perm]
            else:
                import random

                random.shuffle(idxs)
            for j in range(0, len(idxs), self.batch_size):
                chunk = idxs[j : j + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk)
        if self.shuffle_batches:
            if rng is not None:
                perm = torch.randperm(len(batches), generator=rng).tolist()
                batches = [batches[k] for k in perm]
            else:
                import random

                random.shuffle(batches)
        yield from batches
