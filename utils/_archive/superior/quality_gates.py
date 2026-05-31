"""
Pre/post generation **quality gates** (fast heuristics + optional CLIP).

Use before accepting a candidate or after decode in multi-candidate flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from utils.quality import test_time_pick as ttp


@dataclass(slots=True)
class GateThresholds:
    min_sharpness: float = 0.15
    min_exposure: float = 0.25
    min_clip: float = 0.0
    min_aesthetic: float = 0.0
    min_hf_reward: float = 0.0
    max_nsfw: float = 1.0
    max_watermark: float = 1.0
    clip_model_id: str = "openai/clip-vit-base-patch32"


@dataclass(slots=True)
class GateResult:
    passed: bool
    scores: dict = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)


class QualityGateRunner:
    """Run configurable gates on uint8 RGB images."""

    def __init__(self, thresholds: Optional[GateThresholds] = None) -> None:
        self.thresholds = thresholds or GateThresholds()

    def evaluate(
        self,
        rgb_uint8: np.ndarray,
        *,
        prompt: str = "",
        device: str = "cpu",
    ) -> GateResult:
        th = self.thresholds
        scores: dict = {}
        failures: List[str] = []

        sharp = ttp.score_edge_sharpness(rgb_uint8)
        scores["sharpness"] = sharp
        if sharp < th.min_sharpness:
            failures.append("sharpness")

        expo = ttp.score_exposure_balance(rgb_uint8)
        scores["exposure"] = expo
        if expo < th.min_exposure:
            failures.append("exposure")

        aes = ttp.score_aesthetic_proxy(rgb_uint8)
        scores["aesthetic"] = aes
        if th.min_aesthetic > 0 and aes < th.min_aesthetic:
            failures.append("aesthetic")

        if th.min_clip > 0 and prompt.strip():
            clip_scores = ttp.score_clip_similarity([rgb_uint8], prompt, device=device, model_id=th.clip_model_id)
            scores["clip"] = float(clip_scores[0]) if clip_scores else 0.0
            if scores["clip"] < th.min_clip:
                failures.append("clip")

        if th.min_hf_reward > 0:
            try:
                from utils.modeling.hf_reward import HFRewardPanel

                hf_s = HFRewardPanel(device=device).score(rgb_uint8, prompt=prompt)
                scores["hf_reward"] = hf_s
                if hf_s < th.min_hf_reward:
                    failures.append("hf_reward")
            except Exception:
                pass

        if th.max_nsfw < 1.0:
            try:
                from utils.modeling.hf_loaders import score_nsfw_probability

                nsfw_p = score_nsfw_probability(rgb_uint8, device=device)
                if nsfw_p is not None:
                    scores["nsfw"] = float(nsfw_p)
                    if nsfw_p > th.max_nsfw:
                        failures.append("nsfw")
            except Exception:
                pass

        if th.max_watermark < 1.0:
            try:
                from utils.modeling.hf_loaders import score_watermark_probability

                wm_p = score_watermark_probability(rgb_uint8, device=device)
                if wm_p is not None:
                    scores["watermark"] = float(wm_p)
                    if wm_p > th.max_watermark:
                        failures.append("watermark")
            except Exception:
                pass

        return GateResult(passed=len(failures) == 0, scores=scores, failures=failures)

    def filter_passing(
        self,
        images: Sequence[np.ndarray],
        *,
        prompt: str = "",
        device: str = "cpu",
    ) -> List[int]:
        """Indices of images that pass all gates."""
        return [i for i, im in enumerate(images) if self.evaluate(im, prompt=prompt, device=device).passed]


__all__ = ["GateResult", "GateThresholds", "QualityGateRunner"]
