"""
Auto-rig characters from a reference image.

Produces a box-layout JSON (regional prompting) with heuristic part regions.
Optional heavy path uses image dissection when models are available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CharacterRig",
    "RigPart",
    "auto_rig_character",
    "rig_to_box_layout",
    "write_rig_box_layout",
]


@dataclass(slots=True)
class RigPart:
    name: str
    box: Tuple[float, float, float, float]
    prompt: str = ""
    negative: str = ""
    lock: bool = False
    reference_image: str = ""
    reference_weight: float = 0.85


@dataclass(slots=True)
class CharacterRig:
    character_id: str
    source_image: str
    parts: List[RigPart] = field(default_factory=list)
    global_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# Default humanoid rig (normalized 0–1 boxes)
_HUMANOID_PARTS: Tuple[Tuple[str, Tuple[float, float, float, float], bool], ...] = (
    ("head", (0.32, 0.02, 0.68, 0.22), True),
    ("torso", (0.28, 0.20, 0.72, 0.55), True),
    ("left_arm", (0.08, 0.22, 0.32, 0.52), False),
    ("right_arm", (0.68, 0.22, 0.92, 0.52), False),
    ("legs", (0.30, 0.52, 0.70, 0.98), False),
)


def _center_of_mass_box(image_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """Rough subject box from luminance center (fallback when no ML)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    ys, xs = np.where(th > 0)
    if len(xs) < 50:
        ys, xs = np.where(gray < 240)
    if len(xs) < 50:
        return None
    h, w = gray.shape
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad_x = int((x1 - x0) * 0.08)
    pad_y = int((y1 - y0) * 0.05)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    return (x0 / w, y0 / h, x1 / w, y1 / h)


def _shift_parts_to_subject(
    parts: Sequence[Tuple[str, Tuple[float, float, float, float], bool]],
    subject: Tuple[float, float, float, float],
) -> List[RigPart]:
    sx0, sy0, sx1, sy1 = subject
    sw, sh = sx1 - sx0, sy1 - sy0
    out: List[RigPart] = []
    for name, box, lock in parts:
        x0, y0, x1, y1 = box
        out.append(
            RigPart(
                name=name,
                box=(sx0 + x0 * sw, sy0 + y0 * sh, sx0 + x1 * sw, sy0 + y1 * sh),
                lock=lock,
            )
        )
    return out


def auto_rig_character(
    character_id: str,
    image_path: str | Path,
    *,
    text_by_part: Optional[Dict[str, str]] = None,
    lock_parts: Optional[Sequence[str]] = None,
    use_dissection: bool = False,
) -> CharacterRig:
    """
    Build a character rig from one image.

    ``text_by_part``: e.g. {"head": "keep face", "torso": "red armor", "legs": "walking stride"}
    """
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"rig source not found: {p}")

    text_by_part = dict(text_by_part or {})
    lock_set = {x.lower() for x in (lock_parts or [])}

    subject = _center_of_mass_box(p) or (0.2, 0.05, 0.8, 0.95)
    parts = _shift_parts_to_subject(_HUMANOID_PARTS, subject)

    for part in parts:
        part.reference_image = str(p.resolve())
        part.prompt = text_by_part.get(part.name, text_by_part.get(part.name.replace("_", " "), ""))
        if part.name in lock_set or part.lock:
            part.lock = True
            if not part.prompt:
                part.prompt = f"preserve {part.name} from reference, identity locked"

    if use_dissection:
        try:
            from utils.generation.image_dissection import dissect_images_to_parts

            reqs, dparts, _ = dissect_images_to_parts(
                "; ".join(f"use {pt.name} from image 1" for pt in parts),
                [str(p)],
                output_dir=p.parent / f".rig_{character_id}",
                enable_heavy_models=True,
            )
            for pt in parts:
                for dp in dparts:
                    if dp.request.part.lower() in pt.name.lower() and dp.mask_path:
                        pt.reference_image = str(p)
                        if pt.lock:
                            pt.prompt = pt.prompt or f"preserve {pt.name}, locked"
        except Exception:
            pass

    return CharacterRig(
        character_id=character_id,
        source_image=str(p.resolve()),
        parts=parts,
        global_prompt=text_by_part.get("_global", ""),
        metadata={"subject_box": subject, "rig_type": "humanoid_heuristic"},
    )


def rig_to_box_layout(
    rig: CharacterRig,
    *,
    global_prompt: str = "",
    global_negative: str = "",
) -> Dict[str, Any]:
    regions: List[Dict[str, Any]] = []
    for i, pt in enumerate(rig.parts):
        reg: Dict[str, Any] = {
            "name": pt.name,
            "box": list(pt.box),
            "priority": 20 if pt.lock else 10 - i,
            "prompt": pt.prompt or f"{pt.name} detail",
            "negative": pt.negative,
        }
        if pt.reference_image:
            reg["reference"] = pt.reference_image
            reg["reference_weight"] = pt.reference_weight
            reg["reference_mode"] = "identity" if pt.lock else "style"
        regions.append(reg)
    return {
        "global_prompt": global_prompt or rig.global_prompt or "cohesive character, consistent anatomy",
        "global_negative": global_negative or "extra limbs, duplicate face, morphing",
        "feather": 8,
        "overlap_mode": "priority",
        "regions": regions,
    }


def write_rig_box_layout(
    rig: CharacterRig,
    out_path: str | Path,
    *,
    global_prompt: str = "",
    global_negative: str = "",
) -> Path:
    data = rig_to_box_layout(rig, global_prompt=global_prompt, global_negative=global_negative)
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return op
