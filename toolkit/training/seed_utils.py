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


__all__ = ["seed_everything"]
