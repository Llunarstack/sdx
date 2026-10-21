"""Dataset loaders for text-to-image training.

The torch-backed exports are imported lazily, so ``import data.caption_utils`` (and other torch-free helpers)
never loads PyTorch.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any

_HAS_TORCH = find_spec("torch") is not None

_LAZY_EXPORTS = {
    "Text2ImageDataset": ".t2i_dataset",
    "collate_t2i": ".t2i_dataset",
    "ResolutionBucketBatchSampler": ".bucket_batch_sampler",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if not _HAS_TORCH:
        raise RuntimeError(f"{__name__}.{name} requires PyTorch; install torch or import the torch-free helpers only.")
    return getattr(import_module(module, __package__), name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
