"""Pairwise preference rows for Diffusion-DPO (``scripts/tools/training/train_diffusion_dpo.py``).

Each line is a JSON object. Supported keys (aliases in parentheses):

- ``win_image_path`` (``winner``, ``preferred``): path to the preferred image
- ``lose_image_path`` (``loser``, ``rejected``): path to the non-preferred image
- ``caption`` (``prompt``, ``text``): shared conditioning text

Also holds what used to be: preference_jsonl, dpo_advanced, preference_image_dataset.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.safety.minor_guard import reasons

# ==================================================================================================
# preference_jsonl: Pairwise preference rows for Diffusion-DPO (``scripts/tools/training/train_diffusion_dpo.py``).
#
# Each line is a JSON object. Supported keys (aliases in parentheses):
#
# - ``win_image_path`` (``winner``, ``preferred``): path to the preferred image
# - ``lose_image_path`` (``loser``, ``rejected``): path to the non-preferred image
# - ``caption`` (``prompt``, ``text``): shared conditioning text
# ==================================================================================================


@dataclass(frozen=True, slots=True)
class PreferencePair:
    win_path: str
    lose_path: str
    prompt: str
    raw: dict[str, Any]


def _pick_str(d: dict[str, Any], keys: tuple) -> str | None:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def parse_preference_row(obj: dict[str, Any]) -> PreferencePair | None:
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


# ==================================================================================================
# dpo_advanced: Advanced **Diffusion-DPO** helpers inspired by 2025–2026 alignment research.
#
# - Timestep-aware weighting (upweight high-noise steps; compensates signal attenuation)
# - Safeguarded DPO margin (SDPO-style; caps destructive loser pushes)
# - EMA reference model updates (dynamic ref vs frozen ref)
#
# References: Rethinking DPO in Diffusion (AAAI 2026 oral); Diffusion-SDPO (safeguarded updates).
# ==================================================================================================


def timestep_dpo_weight(
    t: torch.Tensor,
    num_timesteps: int,
    *,
    mode: str = "high_noise",
    power: float = 0.5,
) -> torch.Tensor:
    """
    Per-sample weight for DPO loss at timestep ``t``.

    ``high_noise``: larger ``t`` (more noise) gets higher weight — matches diffusion-DPO
    timestep imbalance findings (low-noise steps dominate unless reweighted).
    """
    T = max(1, int(num_timesteps) - 1)
    tn = t.detach().float().clamp(0, T) / float(T)
    if mode in ("uniform", "none", ""):
        return torch.ones_like(tn)
    if mode == "low_noise":
        w = (1.0 - tn) ** float(power)
    else:
        w = tn ** float(power)
    return w.clamp(min=0.05) + 0.95 * w / (w.mean() + 1e-8)


def safeguard_dpo_margins(
    implicit_logp_win: torch.Tensor,
    implicit_logp_lose: torch.Tensor,
    *,
    strength: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SDPO-style surrogate: shrink loser advantage when winner is already strongly preferred.

    Operates on scalar implicit log-probs (typically ``-loss``). Does not require separate
    backward through winner/loser branches.
    """
    if strength <= 0.0:
        return implicit_logp_win, implicit_logp_lose
    margin = implicit_logp_win - implicit_logp_lose
    # When margin is positive (winner better), slightly pull loser up toward winner
    adjust = torch.clamp(margin, min=0.0) * (1.0 - float(strength))
    lose_adj = implicit_logp_lose + adjust
    return implicit_logp_win, lose_adj


def safeguarded_dpo_preference_loss(
    implicit_logp_win: torch.Tensor,
    implicit_logp_lose: torch.Tensor,
    implicit_logp_ref_win: torch.Tensor,
    implicit_logp_ref_lose: torch.Tensor,
    *,
    beta: float = 5000.0,
    logit_clip: float | None = None,
    safeguard_strength: float = 0.85,
    timestep_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """DPO loss with optional SDPO margin safeguard + per-timestep weights."""
    lw, ll = safeguard_dpo_margins(implicit_logp_win, implicit_logp_lose, strength=float(safeguard_strength))
    b = float(beta)
    pi = b * ((lw - ll) - (implicit_logp_ref_win - implicit_logp_ref_lose))
    if logit_clip is not None and float(logit_clip) > 0.0:
        c = float(logit_clip)
        pi = pi.clamp(-c, c)
    loss = -F.logsigmoid(pi)
    if timestep_weights is not None:
        w = timestep_weights.detach().float()
        if w.shape == loss.shape:
            return (loss * w).sum() / (w.sum() + 1e-8)
    return loss.mean()


@torch.no_grad()
def ema_update_reference(
    ref_model: torch.nn.Module,
    policy_model: torch.nn.Module,
    *,
    alpha: float = 0.01,
) -> None:
    """Soft-update reference toward policy (``ref = (1-a)*ref + a*policy``)."""
    a = float(max(0.0, min(1.0, alpha)))
    ref_sd = ref_model.state_dict()
    pol_sd = policy_model.state_dict()
    for k in ref_sd:
        if k not in pol_sd:
            continue
        rp = ref_sd[k]
        pp = pol_sd[k]
        if torch.is_floating_point(rp):
            ref_sd[k].mul_(1.0 - a).add_(pp.detach(), alpha=a)
    ref_model.load_state_dict(ref_sd)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False


# ==================================================================================================
# preference_image_dataset: Load win/lose image pairs from preference JSONL for Diffusion-DPO-style training.
# ==================================================================================================


def _resolve_path(p: str, root: Path | None) -> Path:
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
    Rows from ``iter_preference_jsonl`` (minus prompts involving minors); each sample is win/lose tensors
    ``(3,H,W)`` in [-1, 1].
    """

    def __init__(self, jsonl_path: str | Path, *, image_size: int, image_root: str | Path | None = None):
        self.rows: list[PreferencePair] = [
            r for r in iter_preference_jsonl(jsonl_path) if not reasons(r.prompt, training=True)
        ]
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


def collate_preference_batch(batch: list[dict]) -> dict:
    return {
        "win": torch.stack([b["win"] for b in batch], dim=0),
        "lose": torch.stack([b["lose"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
    }


__all__ = [
    "PreferencePair",
    "iter_preference_jsonl",
    "parse_preference_row",
    "ema_update_reference",
    "safeguard_dpo_margins",
    "safeguarded_dpo_preference_loss",
    "timestep_dpo_weight",
    "PreferenceImageDataset",
    "collate_preference_batch",
]
