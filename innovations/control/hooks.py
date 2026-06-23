"""
Bridge fine control to spatial layout and sampling knobs.

Production paths:
  - Box + prompt regions: ``utils/generation/regional_box_prompting.py``, ``--box-layout``
  - Layout DSL: ``utils/generation/spatial_dsl/``
  - Scene composer: ``utils/generation/engine.py``
  - Content controls: ``utils/prompt/content_controls.py`` (people_layout, object_layout)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .engine import PrecisionControlSystem


def control_specs_from_box_layout(box_layout: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert ``--box-layout`` JSON regions into precision-control spatial specs."""
    regions = box_layout.get("regions", box_layout.get("boxes", [])) or []
    objects: List[Dict[str, Any]] = []
    for i, r in enumerate(regions):
        if not isinstance(r, Mapping):
            continue
        box = r.get("box", r.get("bbox", [0, 0, 1, 1]))
        if isinstance(box, Mapping):
            x1, y1 = float(box.get("x_min", box.get("x1", 0))), float(box.get("y_min", box.get("y1", 0)))
            x2, y2 = float(box.get("x_max", box.get("x2", 1))), float(box.get("y_max", box.get("y2", 1)))
        else:
            x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        objects.append(
            {
                "name": str(r.get("name", f"region_{i}")),
                "position": (cx, cy),
                "size": (max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)),
                "prompt": str(r.get("prompt", "")),
            }
        )
    return {"spatial": {"objects": objects}, "source": "box_layout"}


def apply_box_layout_controls(
    base_image,
    box_layout: Mapping[str, Any],
    *,
    system: Optional[PrecisionControlSystem] = None,
):
    """Apply precision-control routing using a box-layout dict."""
    sys = system or PrecisionControlSystem()
    specs = control_specs_from_box_layout(box_layout)
    return sys.apply_controls(base_image, specs)


__all__ = ["apply_box_layout_controls", "control_specs_from_box_layout"]
