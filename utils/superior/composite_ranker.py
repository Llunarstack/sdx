"""
Multi-metric **candidate ranking** beyond a single ``--pick-best`` mode.

Combines existing ``test_time_pick`` scores with cheap heuristics (sharpness, exposure, palette).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from utils.quality import test_time_pick as ttp


@dataclass(slots=True)
class RankWeights:
    pick_metric: str = "combo"
    sharpness: float = 0.12
    exposure: float = 0.08
    palette: float = 0.05
    hpsv2: float = 0.0
    hf_reward: float = 0.0


class CompositeRanker:
    """Score RGB uint8 HWC images and return best index."""

    def __init__(self, weights: Optional[RankWeights] = None) -> None:
        self.weights = weights or RankWeights()

    def _sharpness(self, img: np.ndarray) -> float:
        gray = img.mean(axis=2) if img.ndim == 3 else img.astype(np.float32)
        gx = np.abs(np.diff(gray, axis=1)).mean()
        gy = np.abs(np.diff(gray, axis=0)).mean()
        return float(gx + gy)

    def _exposure_score(self, img: np.ndarray) -> float:
        lum = img.mean(axis=2) / 255.0
        mean = float(lum.mean())
        return float(np.exp(-((mean - 0.45) ** 2) / 0.08))

    def _palette_score(self, img: np.ndarray) -> float:
        if img.ndim != 3 or img.shape[2] < 3:
            return 0.5
        std = img.reshape(-1, 3).astype(np.float32).std(axis=0).mean() / 128.0
        return float(np.clip(std, 0.0, 1.0))

    def score_images(
        self,
        images: Sequence[np.ndarray],
        *,
        prompt: str = "",
        device: str = "cuda",
        vit_ckpt: str = "",
    ) -> List[float]:
        """Per-image composite score (higher is better)."""
        n = len(images)
        if n == 0:
            return []
        w = self.weights
        imgs = list(images)
        _idx, base = ttp.pick_best_indices(
            imgs,
            prompt,
            w.pick_metric,
            device,
            vit_ckpt_path=vit_ckpt or "",
        )
        base_n = ttp._norm01(base) if base else [0.5] * n
        sharp = ttp._norm01([self._sharpness(im) for im in imgs])
        expo = ttp._norm01([self._exposure_score(im) for im in imgs])
        pal = ttp._norm01([self._palette_score(im) for im in imgs])
        hps: List[float] = [0.5] * n
        hf_r: List[float] = [0.5] * n
        if w.hpsv2 > 0 and prompt.strip():
            try:
                from utils.modeling.hf_loaders import score_hpsv2

                raw = [score_hpsv2(im, prompt, device=device) for im in imgs]
                hps = ttp._norm01([float(x if x is not None else 0.5) for x in raw])
            except Exception:
                pass
        if w.hf_reward > 0 and prompt.strip():
            try:
                from utils.modeling.hf_reward import HFRewardPanel

                panel = HFRewardPanel(device=device)
                raw = [panel.score(im, prompt=prompt) for im in imgs]
                hf_r = ttp._norm01(raw)
            except Exception:
                pass
        out: List[float] = []
        for i in range(n):
            s = float(base_n[i])
            s += w.sharpness * sharp[i]
            s += w.exposure * expo[i]
            s += w.palette * pal[i]
            s += w.hpsv2 * hps[i]
            s += w.hf_reward * hf_r[i]
            out.append(s)
        mx, mn = max(out), min(out)
        if mx > mn:
            out = [(x - mn) / (mx - mn) for x in out]
        return out

    def pick_best_index(
        self,
        images: Sequence[np.ndarray],
        *,
        prompt: str = "",
        device: str = "cuda",
        vit_ckpt: str = "",
    ) -> Tuple[int, List[float]]:
        scores = self.score_images(images, prompt=prompt, device=device, vit_ckpt=vit_ckpt)
        return int(np.argmax(scores)), scores


__all__ = ["CompositeRanker", "RankWeights"]
