"""Batch sampler that keeps **one resolution bucket per batch** (IMPROVEMENTS §1.1).

``Text2ImageDataset`` must have ``resolution_buckets`` set and ``_bucket_assign`` populated
via ``set_epoch`` before each training epoch.
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
    """
    Yields batches of indices that share the same bucket id so ``collate_t2i`` stacks tensors.
    """

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
        self._cached_epoch: int | None = None
        self._cached_groups: dict[int, list[int]] | None = None

    def _bucket_groups(self) -> dict[int, list[int]]:
        epoch = int(getattr(self.dataset, "_epoch", 0))
        if self._cached_groups is not None and self._cached_epoch == epoch:
            return self._cached_groups
        groups: defaultdict[int, list[int]] = defaultdict(list)
        for i in range(len(self.dataset)):
            groups[self.dataset._bucket_assign[i]].append(i)
        self._cached_groups = dict(groups)
        self._cached_epoch = epoch
        return self._cached_groups

    def __len__(self) -> int:
        groups = self._bucket_groups()
        total = 0
        for idxs in groups.values():
            c = len(idxs)
            if self.drop_last:
                total += c // self.batch_size
            else:
                total += (c + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self) -> Iterator[list[int]]:
        rng = self.generator
        groups = self._bucket_groups()
        batches: list[list[int]] = []
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
