"""
Unified Hugging Face **reward panel** for ranking, eval, and DPO mining.

Combines available HF scorers (HPSv2, PickScore, CLIP-H/14, OneAlign, CAFE)
with graceful fallback when weights are missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass(slots=True)
class HFRewardWeights:
    hpsv2: float = 0.16
    pickscore: float = 0.16
    clip_h14: float = 0.12
    onealign: float = 0.12
    cafe_aesthetic: float = 0.12
    perceptclip: float = 0.12
    image_reward: float = 0.10
    musiq: float = 0.08
    siglip2: float = 0.07
    clip_iqa: float = 0.07


@dataclass(slots=True)
class HFRewardBreakdown:
    composite: float
    parts: Dict[str, float] = field(default_factory=dict)


class HFRewardPanel:
    """Score RGB uint8 images with multiple HF reward backends."""

    def __init__(
        self,
        weights: Optional[HFRewardWeights] = None,
        *,
        device: str = "cuda",
    ) -> None:
        self.weights = weights or HFRewardWeights()
        self.device = device

    def score_breakdown(self, rgb_uint8: np.ndarray, *, prompt: str = "") -> HFRewardBreakdown:
        from utils.modeling import hf_loaders as hl

        parts: Dict[str, float] = {}
        w = self.weights

        if w.hpsv2 > 0 and prompt.strip():
            v = hl.score_hpsv2(rgb_uint8, prompt, device=self.device)
            if v is not None:
                parts["hpsv2"] = float(v)

        if w.pickscore > 0 and prompt.strip():
            v = hl.score_pickscore(rgb_uint8, prompt, device=self.device)
            if v is not None:
                parts["pickscore"] = float(v)

        if w.clip_h14 > 0 and prompt.strip():
            v = hl.score_clip_h14(rgb_uint8, prompt, device=self.device)
            if v is not None:
                parts["clip_h14"] = float(v)

        if w.onealign > 0:
            v = hl.score_onealign(rgb_uint8, prompt=prompt, device=self.device)
            if v is not None:
                parts["onealign"] = float(v)

        if w.cafe_aesthetic > 0:
            v = hl.score_cafe_aesthetic(rgb_uint8, device=self.device)
            if v is not None:
                parts["cafe_aesthetic"] = float(v)

        if w.perceptclip > 0:
            v = hl.score_perceptclip(rgb_uint8, prompt=prompt, device=self.device)
            if v is not None:
                parts["perceptclip"] = float(v)

        if w.image_reward > 0 and prompt.strip():
            v = hl.score_image_reward(rgb_uint8, prompt, device=self.device)
            if v is not None:
                parts["image_reward"] = float(v)

        if w.musiq > 0:
            v = hl.score_musiq(rgb_uint8, device=self.device)
            if v is not None:
                parts["musiq"] = float(v)

        if w.siglip2 > 0 and prompt.strip():
            v = hl.score_siglip2(rgb_uint8, prompt, device=self.device)
            if v is not None:
                parts["siglip2"] = float(v)

        if w.clip_iqa > 0:
            v = hl.score_clip_iqa(rgb_uint8, device=self.device)
            if v is not None:
                parts["clip_iqa"] = float(v)

        if not parts:
            return HFRewardBreakdown(composite=0.5, parts={})

        weight_map = {
            "hpsv2": w.hpsv2,
            "pickscore": w.pickscore,
            "clip_h14": w.clip_h14,
            "onealign": w.onealign,
            "cafe_aesthetic": w.cafe_aesthetic,
            "perceptclip": w.perceptclip,
            "image_reward": w.image_reward,
            "musiq": w.musiq,
            "siglip2": w.siglip2,
            "clip_iqa": w.clip_iqa,
        }
        tw = sum(weight_map[k] for k in parts) or 1.0
        composite = float(sum(weight_map[k] * parts[k] for k in parts) / tw)
        return HFRewardBreakdown(composite=composite, parts=parts)

    def score(self, rgb_uint8: np.ndarray, *, prompt: str = "") -> float:
        return self.score_breakdown(rgb_uint8, prompt=prompt).composite

    def rank_indices(
        self,
        images: List[np.ndarray],
        *,
        prompt: str = "",
    ) -> List[int]:
        scores = [self.score(im, prompt=prompt) for im in images]
        return sorted(range(len(scores)), key=lambda i: -scores[i])


__all__ = ["HFRewardBreakdown", "HFRewardPanel", "HFRewardWeights"]
