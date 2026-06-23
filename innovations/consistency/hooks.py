"""
Bridge consistency engine to sampling seeds and character sheets.

Production paths:
  - ``sample.py --seed``, ``--deterministic``
  - Character sheets: ``sample.py --character-sheet``, ``pipelines/book_comic/``
  - Multi-character: ``models/multi_character.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .engine import ConsistencyEngine


@dataclass(slots=True)
class ConsistencyBundle:
    """Knobs to pass into a generation run for reproducibility."""

    seed: int
    character_id: Optional[str] = None
    style_name: Optional[str] = None
    variation: float = 0.0
    seed_embedding: Optional[torch.Tensor] = None


def build_consistency_bundle(
    *,
    seed: int,
    character_id: Optional[str] = None,
    style_name: Optional[str] = None,
    variation: float = 0.0,
    engine: Optional[ConsistencyEngine] = None,
) -> ConsistencyBundle:
    """Encode seed + optional character/style into a consistency bundle."""
    eng = engine or ConsistencyEngine()
    emb = eng.seeding.encode_seed(int(seed))
    if character_id:
        eng.character.encode_character(emb, character_id)
    if style_name:
        eng.style.capture_style(emb, style_name)
    return ConsistencyBundle(
        seed=int(seed),
        character_id=character_id,
        style_name=style_name,
        variation=float(variation),
        seed_embedding=emb,
    )


def torch_seed_from_bundle(bundle: ConsistencyBundle) -> int:
    """Return the integer seed for ``torch.manual_seed`` / ``sample.py --seed``."""
    return int(bundle.seed)


__all__ = ["ConsistencyBundle", "build_consistency_bundle", "torch_seed_from_bundle"]
