"""
**Online reward** scaffold for flow-GRPO / RL-style fine-tuning (research hook).

Combines benchmark composite proxies with optional ViT scores — used to rank samples
during multi-candidate generation or future online training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from utils.quality import test_time_pick as ttp


@dataclass(slots=True)
class OnlineRewardConfig:
    clip_weight: float = 0.25
    aesthetic_weight: float = 0.20
    sharpness_weight: float = 0.15
    vit_weight: float = 0.40
    vit_ckpt: str = ""
    device: str = "cuda"


class OnlineRewardModel:
    """Score RGB uint8 images for online RL / best-of-N (higher is better)."""

    def __init__(self, config: Optional[OnlineRewardConfig] = None) -> None:
        self.config = config or OnlineRewardConfig()

    def score_one(self, rgb_uint8: np.ndarray, *, prompt: str = "") -> float:
        c = self.config
        parts: List[tuple[float, float]] = []
        if prompt.strip():
            clip = ttp.score_clip_similarity(
                [rgb_uint8], prompt, device=c.device, model_id="openai/clip-vit-base-patch32"
            )
            if clip:
                x = float(clip[0])
                parts.append((c.clip_weight, 1.0 / (1.0 + np.exp(-x / 10.0))))
        parts.append((c.aesthetic_weight, ttp.score_aesthetic_proxy(rgb_uint8)))
        parts.append((c.sharpness_weight, float(np.clip(ttp.score_edge_sharpness(rgb_uint8) / 400.0, 0, 1))))
        if prompt.strip():
            try:
                from utils.modeling.hf_reward import HFRewardPanel

                panel = HFRewardPanel(device=c.device)
                bd = panel.score_breakdown(rgb_uint8, prompt=prompt)
                if bd.parts:
                    parts.append((0.25, bd.composite))
            except Exception:
                pass
        if c.vit_weight > 0 and c.vit_ckpt and prompt.strip():
            try:
                import tempfile

                from PIL import Image

                from utils.superior.vit_mining import score_image_vit

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    Image.fromarray(rgb_uint8).save(tmp.name)
                    _q, vit_r = score_image_vit(tmp.name, prompt, vit_ckpt=c.vit_ckpt, device=c.device)
                parts.append((c.vit_weight, vit_r))
            except Exception:
                pass
        tw = sum(p[0] for p in parts) or 1.0
        return float(sum(w * v for w, v in parts) / tw)

    def rank_indices(self, images: Sequence[np.ndarray], *, prompt: str = "") -> List[int]:
        scores = [self.score_one(im, prompt=prompt) for im in images]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return order


__all__ = ["OnlineRewardConfig", "OnlineRewardModel"]
