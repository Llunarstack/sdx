"""Deterministic seeds for Python, NumPy, and PyTorch (CPU + CUDA if available)."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic_cudnn: bool = False) -> None:
    """
    Set seeds for ``random``, ``numpy``, ``torch``.

    *deterministic_cudnn*: if True, sets ``torch.backends.cudnn.deterministic=True``
    and ``benchmark=False`` (slower, more reproducible on GPU).
    """
    s = int(seed)
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    try:
        import numpy as np

        np.random.seed(s)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def worker_seed_fn(base_seed: int, rank: int = 0):
    """
    Return a ``worker_init_fn`` for ``DataLoader`` compatible with PyTorch:

        DataLoader(..., worker_init_fn=toolkit.training.worker_seed_fn(42, rank=0))
    """

    def _fn(worker_id: int) -> None:
        try:
            import numpy as np

            np.random.seed(base_seed + rank * 1000 + worker_id)
        except ImportError:
            pass
        try:
            import torch

            torch.manual_seed(base_seed + rank * 1000 + worker_id)
        except ImportError:
            pass

    return _fn


__all__ = ["seed_everything", "worker_seed_fn"]
