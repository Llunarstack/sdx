"""
Pairwise preference rows for **future** diffusion preference optimization (DPO-class).

This module only defines a **schema** and **streaming loader** — it does **not** hook into
``train.py`` yet. Use it to validate JSONL exports before wiring a second-stage trainer.

Each line is a JSON object. Supported keys (aliases in parentheses):

- ``win_image_path`` (``winner``, ``preferred``): path to the preferred image
- ``lose_image_path`` (``loser``, ``rejected``): path to the non-preferred image
- ``caption`` (``prompt``, ``text``): shared conditioning text
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


@dataclass(frozen=True)
class PreferencePair:
    win_path: str
    lose_path: str
    prompt: str
    raw: Dict[str, Any]


def _pick_str(d: Dict[str, Any], keys: tuple) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def parse_preference_row(obj: Dict[str, Any]) -> Optional[PreferencePair]:
    win = _pick_str(obj, ("win_image_path", "winner", "preferred", "win"))
    lose = _pick_str(obj, ("lose_image_path", "loser", "rejected", "lose"))
    prompt = _pick_str(obj, ("caption", "prompt", "text")) or ""
    if not win or not lose:
        return None
    return PreferencePair(win_path=win, lose_path=lose, prompt=prompt, raw=dict(obj))


def iter_preference_jsonl(path: str | Path) -> Iterator[PreferencePair]:
    """Yield ``PreferencePair`` for each valid row; skip blank lines and invalid rows."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            pair = parse_preference_row(obj)
            if pair is not None:
                yield pair


__all__ = ["PreferencePair", "iter_preference_jsonl", "parse_preference_row"]
