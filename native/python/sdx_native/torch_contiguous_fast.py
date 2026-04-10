"""Torch ↔ numpy contiguity helpers (zero-copy when possible)."""

from __future__ import annotations

from typing import Any


def ensure_contiguous_torch(t: Any) -> Any:
    """Return tensor contiguous (clone only if needed)."""
    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError("expected Tensor")
    return t.contiguous()


def torch_to_numpy_contiguous(t: Any) -> Any:
    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError("expected Tensor")
    if not t.is_contiguous():
        t = t.contiguous()
    return t.detach().cpu().numpy()


def numpy_to_torch_float32(arr: Any, *, device: Any = None) -> Any:
    import numpy as np
    import torch

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if arr.dtype == np.float64:
        t = t.float()
    elif arr.dtype in (np.float32, np.float16):
        t = t.float()
    if device is not None:
        t = t.to(device)
    return t
