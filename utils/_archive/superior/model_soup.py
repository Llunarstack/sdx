"""
**Model soup**: average multiple DiT checkpoints for robust inference weights.

Same architecture + compatible keys only; useful after multi-seed training or DPO + base blend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import torch


def _load_state(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt.get("ema") or ckpt.get("model")
    if not isinstance(sd, dict):
        raise ValueError(f"No model/ema state in {path}")
    return {k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu() for k, v in sd.items()}


def average_state_dicts(
    states: Sequence[Dict[str, torch.Tensor]],
    *,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Element-wise weighted average of compatible state dicts."""
    if not states:
        raise ValueError("states must be non-empty")
    keys = set(states[0].keys())
    for s in states[1:]:
        keys &= set(s.keys())
    if not keys:
        raise ValueError("No shared keys across checkpoints")
    w = list(weights) if weights is not None else [1.0 / len(states)] * len(states)
    if len(w) != len(states):
        raise ValueError("weights length must match states")
    ws = sum(w) or 1.0
    w = [x / ws for x in w]
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for i, st in enumerate(states):
            t = st[k]
            if not t.is_floating_point():
                out[k] = t.clone()
                acc = None
                break
            part = t.float() * w[i]
            acc = part if acc is None else acc + part
        if acc is not None:
            out[k] = acc.to(dtype=states[0][k].dtype)
    return out


def soup_checkpoints(
    paths: Iterable[Union[str, Path]],
    *,
    weights: Optional[Sequence[float]] = None,
    template_ckpt: Optional[Union[str, Path]] = None,
) -> Dict[str, torch.Tensor]:
    """Load and average multiple ``.pt`` checkpoints."""
    plist = [Path(p) for p in paths]
    states = [_load_state(p) for p in plist]
    return average_state_dicts(states, weights=weights)


def save_soup_checkpoint(
    averaged: Dict[str, torch.Tensor],
    *,
    template_path: Union[str, Path],
    out_path: Union[str, Path],
    tag: str = "model_soup",
) -> Path:
    """Write soup weights using metadata from ``template_path``."""
    tpl = torch.load(str(template_path), map_location="cpu", weights_only=False)
    tpl["model"] = averaged
    tpl["ema"] = averaged
    meta = tpl.get("meta") if isinstance(tpl.get("meta"), dict) else {}
    meta["model_soup"] = tag
    tpl["meta"] = meta
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tpl, op)
    return op


__all__ = ["average_state_dicts", "save_soup_checkpoint", "soup_checkpoints"]
