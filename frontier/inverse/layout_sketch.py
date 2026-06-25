"""
Inverse layout: image → box-layout JSON for edit loops.

Full VLM path is planned; this module provides a deterministic scaffold +
color-blob heuristic for tests and Comfy node wiring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class InferredRegion:
    name: str
    box: Tuple[float, float, float, float]  # normalized xyxy
    prompt: str = ""
    confidence: float = 0.5


@dataclass
class LayoutSketch:
    global_prompt: str = ""
    regions: List[InferredRegion] = field(default_factory=list)
    source_size: Tuple[int, int] = (0, 0)

    def to_box_layout(self) -> Dict[str, Any]:
        return {
            "global_prompt": self.global_prompt,
            "regions": [{"name": r.name, "box": list(r.box), "prompt": r.prompt or r.name} for r in self.regions],
        }

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_box_layout(), indent=2), encoding="utf-8")


class LayoutSketchInferer:
    """
    Heuristic region finder via coarse color clustering on downsampled RGB.

    Replace ``infer_from_tensor`` with VLM bbox output when available.
    """

    def __init__(self, grid: int = 4, min_area_frac: float = 0.08) -> None:
        self.grid = max(2, int(grid))
        self.min_area_frac = float(min_area_frac)

    def infer_from_tensor(
        self,
        image: torch.Tensor,
        *,
        global_prompt: str = "",
        max_regions: int = 4,
    ) -> LayoutSketch:
        """
        ``image``: (C,H,W) or (1,C,H,W) in [0,1] or [0,255].
        """
        x = image.detach().float()
        if x.dim() == 4:
            x = x[0]
        if x.max() > 1.5:
            x = x / 255.0
        c, h, w = x.shape
        sketch = LayoutSketch(global_prompt=global_prompt, source_size=(w, h))

        small = F.interpolate(x.unsqueeze(0), size=(self.grid, self.grid), mode="area").squeeze(0)
        labels = self._quantize(small)
        regions = self._components(labels, h, w, max_regions=max_regions)
        sketch.regions = regions
        return sketch

    def _quantize(self, small: torch.Tensor) -> torch.Tensor:
        # (C,g,g) -> integer label per cell via 8-bin per channel hash
        q = (small.clamp(0, 1) * 7).long()
        return (q[0] * 64 + q[1] * 8 + q[2]).view(-1)

    def _components(
        self,
        labels: torch.Tensor,
        h: int,
        w: int,
        *,
        max_regions: int,
    ) -> List[InferredRegion]:
        g = self.grid
        counts: Dict[int, int] = {}
        for lab in labels.tolist():
            counts[lab] = counts.get(lab, 0) + 1
        total = g * g
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        regions: List[InferredRegion] = []
        idx = 0
        for lab, cnt in ranked:
            if cnt / total < self.min_area_frac:
                continue
            cells = (labels == lab).nonzero(as_tuple=False).view(-1).tolist()
            ys = [c // g for c in cells]
            xs = [c % g for c in cells]
            x0, x1 = min(xs) / g, (max(xs) + 1) / g
            y0, y1 = min(ys) / g, (max(ys) + 1) / g
            regions.append(
                InferredRegion(
                    name=f"region_{idx}",
                    box=(x0, y0, x1, y1),
                    confidence=min(1.0, cnt / total + 0.2),
                )
            )
            idx += 1
            if idx >= max_regions:
                break
        return regions


__all__ = ["InferredRegion", "LayoutSketch", "LayoutSketchInferer"]
