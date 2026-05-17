"""Dataset loaders for text-to-image training.

Torch-backed exports (``Text2ImageDataset``, etc.) are **lazy** so ``import data.caption_utils``
does not load PyTorch. This avoids heavy native init during tests and lightweight scripts.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any, List

_HAS_TORCH = find_spec("torch") is not None

__all__ = ["Text2ImageDataset", "collate_t2i", "ResolutionBucketBatchSampler"]


def __getattr__(name: str) -> Any:
    if name == "Text2ImageDataset" or name == "collate_t2i":
        if not _HAS_TORCH:
            raise RuntimeError(f"sdx.data.{name} requires PyTorch; install torch or import caption utilities only.")
        mod = import_module(".t2i_dataset", __package__)
        return getattr(mod, name)
    if name == "ResolutionBucketBatchSampler":
        if not _HAS_TORCH:
            raise RuntimeError(
                "sdx.data.ResolutionBucketBatchSampler requires PyTorch; install torch or avoid this import."
            )
        mod = import_module(".bucket_batch_sampler", __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
