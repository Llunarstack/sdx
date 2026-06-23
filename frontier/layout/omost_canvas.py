"""
Omost-style virtual Canvas → SDX box-layout JSON.

Lets an LLM (or script) call ``set_global_description`` / ``add_local_description``
and export ``examples/box_layout*.json`` for ``sample.py --box-layout``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Omost uses 9 anchor boxes; we expose the same grid for LLM compatibility.
OMOST_GRID: Dict[str, tuple[float, float, float, float]] = {
    "top_left": (0.0, 0.0, 0.33, 0.33),
    "top": (0.33, 0.0, 0.66, 0.33),
    "top_right": (0.66, 0.0, 1.0, 0.33),
    "left": (0.0, 0.33, 0.33, 0.66),
    "center": (0.33, 0.33, 0.66, 0.66),
    "right": (0.66, 0.33, 1.0, 0.66),
    "bottom_left": (0.0, 0.66, 0.33, 1.0),
    "bottom": (0.33, 0.66, 0.66, 1.0),
    "bottom_right": (0.66, 0.66, 1.0, 1.0),
}


@dataclass
class LocalDescription:
    prompt: str
    box: tuple[float, float, float, float]
    name: str
    negative: str = ""
    priority: int = 5
    reference: str = ""


@dataclass
class OmostCanvas:
    """Virtual canvas agent (Omost-compatible API surface)."""

    global_description: str = ""
    global_negative: str = ""
    local_regions: List[LocalDescription] = field(default_factory=list)
    feather: int = 8

    def set_global_description(self, text: str, *, negative: str = "") -> None:
        self.global_description = (text or "").strip()
        if negative:
            self.global_negative = negative.strip()

    def add_local_description(
        self,
        prompt: str,
        *,
        box: Optional[tuple[float, float, float, float]] = None,
        anchor: Optional[str] = None,
        name: Optional[str] = None,
        negative: str = "",
        priority: int = 5,
        reference: str = "",
    ) -> None:
        if anchor:
            key = anchor.strip().lower().replace("-", "_").replace(" ", "_")
            if key not in OMOST_GRID:
                raise ValueError(f"unknown anchor {anchor!r}; choose from {list(OMOST_GRID)}")
            b = OMOST_GRID[key]
        elif box is not None:
            b = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        else:
            raise ValueError("provide box=(x1,y1,x2,y2) or anchor='center' etc.")
        idx = len(self.local_regions)
        self.local_regions.append(
            LocalDescription(
                prompt=(prompt or "").strip(),
                box=b,
                name=name or f"region_{idx}",
                negative=negative,
                priority=priority,
                reference=reference,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        regions: List[Dict[str, Any]] = []
        for loc in self.local_regions:
            r: Dict[str, Any] = {
                "name": loc.name,
                "box": list(loc.box),
                "prompt": loc.prompt,
                "priority": loc.priority,
            }
            if loc.negative:
                r["negative"] = loc.negative
            if loc.reference:
                r["reference"] = loc.reference
            regions.append(r)
        out: Dict[str, Any] = {
            "global_prompt": self.global_description,
            "feather": self.feather,
            "regions": regions,
        }
        if self.global_negative:
            out["global_negative"] = self.global_negative
        return out


def canvas_to_box_layout(canvas: OmostCanvas) -> Dict[str, Any]:
    """Export dict suitable for ``json.dump`` → ``--box-layout``."""
    if not canvas.local_regions:
        raise ValueError("canvas has no local descriptions")
    return canvas.to_dict()
