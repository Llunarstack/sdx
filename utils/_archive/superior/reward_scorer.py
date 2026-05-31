"""
Unified **reward scoring** for ranking and preference mining (ViT + heuristics + optional CLIP).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from utils.quality import test_time_pick as ttp


@dataclass(slots=True)
class RewardWeights:
    composite: float = 0.35
    vit: float = 0.30
    sharpness: float = 0.10
    exposure: float = 0.08
    clip: float = 0.07
    hf_reward: float = 0.10


class UnifiedRewardScorer:
    """Score RGB uint8 images with optional ViT and CLIP."""

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        *,
        vit_ckpt: str = "",
        clip_model_id: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
    ) -> None:
        self.weights = weights or RewardWeights()
        self.vit_ckpt = vit_ckpt
        self.clip_model_id = clip_model_id
        self.device = device

    def score(
        self,
        rgb_uint8: np.ndarray,
        *,
        prompt: str = "",
        composite_hint: Optional[float] = None,
    ) -> float:
        w = self.weights
        parts: list[tuple[float, float]] = []

        if composite_hint is not None:
            parts.append((w.composite, float(composite_hint)))
        else:
            parts.append((w.composite, ttp.score_aesthetic_proxy(rgb_uint8)))

        if self.vit_ckpt and prompt.strip():
            import tempfile

            from PIL import Image
            from utils.superior.vit_mining import score_image_vit

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                Image.fromarray(rgb_uint8).save(tmp.name)
                _q, vit_r = score_image_vit(tmp.name, prompt, vit_ckpt=self.vit_ckpt, device=self.device)
            Path(tmp.name).unlink(missing_ok=True)
            parts.append((w.vit, vit_r))

        sharp = float(np.clip(ttp.score_edge_sharpness(rgb_uint8) / 400.0, 0, 1))
        parts.append((w.sharpness, sharp))
        parts.append((w.exposure, ttp.score_exposure_balance(rgb_uint8)))

        if w.clip > 0 and prompt.strip():
            clip_s = ttp.score_clip_similarity([rgb_uint8], prompt, device=self.device, model_id=self.clip_model_id)
            if clip_s:
                x = float(clip_s[0])
                parts.append((w.clip, 1.0 / (1.0 + np.exp(-x / 10.0))))

        if w.hf_reward > 0:
            try:
                from utils.modeling.hf_reward import HFRewardPanel

                panel = HFRewardPanel(device=self.device)
                parts.append((w.hf_reward, panel.score(rgb_uint8, prompt=prompt)))
            except Exception:
                pass

        total_w = sum(p[0] for p in parts) or 1.0
        return float(sum(wt * val for wt, val in parts) / total_w)


__all__ = ["RewardWeights", "UnifiedRewardScorer"]
