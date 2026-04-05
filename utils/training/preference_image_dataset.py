"""Load win/lose image pairs from preference JSONL for Diffusion-DPO-style training."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from utils.training.preference_jsonl import PreferencePair, iter_preference_jsonl

__all__ = ["PreferenceImageDataset", "collate_preference_batch"]


def _resolve_path(p: str, root: Optional[Path]) -> Path:
    path = Path(p)
    if path.is_file():
        return path
    if root is not None:
        cand = root / path
        if cand.is_file():
            return cand
    return path


def _load_rgb_tensor(path: Path, image_size: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    im = im.resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = torch.from_numpy(np.array(im).astype("float32") / 255.0)
    arr = arr.permute(2, 0, 1)
    arr = (arr - 0.5) / 0.5
    return arr


class PreferenceImageDataset(torch.utils.data.Dataset):
    """
    Rows from ``iter_preference_jsonl``; each sample is win/lose tensors ``(3,H,W)`` in [-1, 1].
    """

    def __init__(self, jsonl_path: str | Path, *, image_size: int, image_root: str | Path | None = None):
        self.rows: List[PreferencePair] = list(iter_preference_jsonl(jsonl_path))
        if not self.rows:
            raise ValueError(f"No valid preference rows in {jsonl_path}")
        self.image_size = int(image_size)
        self.root = Path(image_root) if image_root else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        wp = _resolve_path(r.win_path, self.root)
        lp = _resolve_path(r.lose_path, self.root)
        if not wp.is_file():
            raise FileNotFoundError(f"Win image not found: {wp}")
        if not lp.is_file():
            raise FileNotFoundError(f"Lose image not found: {lp}")
        return {
            "win": _load_rgb_tensor(wp, self.image_size),
            "lose": _load_rgb_tensor(lp, self.image_size),
            "prompt": r.prompt if r.prompt.strip() else " ",
        }


def collate_preference_batch(batch: List[dict]) -> dict:
    return {
        "win": torch.stack([b["win"] for b in batch], dim=0),
        "lose": torch.stack([b["lose"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
    }
