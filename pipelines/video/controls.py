"""Fine-grained control modes for video entities and inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ControlMode",
    "EntityControlPlan",
    "InputBinding",
    "MediaInput",
    "ShotBinding",
    "build_sample_args_for_plan",
    "compile_shot_control_plan",
    "control_mode_help",
]


class ControlMode(str, Enum):
    """
    What text vs image are allowed to change.

    - LOCK: image pixels preserved; text adds metadata only
    - IDENTITY: face/body identity locked; wardrobe/pose may change
    - TRANSFORM: text-driven change; image is strong init reference
    - STYLE: image provides look; content from text
    - MOTION: image/video provides motion path only; pixels regenerated
    - INPAINT: text edits masked region only
    - GENERATE: ignore image for this entity; text only
    """

    LOCK = "lock"
    IDENTITY = "identity"
    TRANSFORM = "transform"
    STYLE = "style"
    MOTION = "motion"
    INPAINT = "inpaint"
    GENERATE = "generate"


def control_mode_help() -> Dict[str, str]:
    return {
        "lock": "Image frozen; text cannot redraw this entity",
        "identity": "Keep face/body identity; allow pose, outfit, lighting changes",
        "transform": "Image as init; text describes the change (default i2v)",
        "style": "Image provides palette/texture; text defines subject/scene",
        "motion": "Clip/image provides camera/pose motion only",
        "inpaint": "Text changes only inside mask/region",
        "generate": "No image constraint; text only",
    }


@dataclass(slots=True)
class MediaInput:
    """Scene-level image/video + text intent (what provides what)."""

    id: str
    image: str = ""
    video: str = ""
    provides: str = ""
    text_changes: str = ""
    control: ControlMode = ControlMode.TRANSFORM
    reference_strength: float = 0.8
    auto_rig: bool = False
    part: str = ""
    negative: str = ""


@dataclass(slots=True)
class ShotBinding:
    """Per-shot override linking an entity to an input + text action."""

    entity_id: str
    input_id: str = ""
    text: str = ""
    control: Optional[ControlMode] = None
    part: str = ""
    mask_path: str = ""
    region_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass(slots=True)
class InputBinding:
    """Resolved binding after compile (entity + input merged)."""

    entity_id: str
    image: str = ""
    video: str = ""
    image_role: str = ""
    text_role: str = ""
    control: ControlMode = ControlMode.TRANSFORM
    reference_strength: float = 0.8
    auto_rig: bool = False
    part: str = ""
    mask_path: str = ""
    region_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    negative: str = ""
    rig_json: str = ""


@dataclass(slots=True)
class EntityControlPlan:
    """Everything sample.py needs for one shot."""

    shot_id: str
    shot_index: int
    positive_prompt: str
    negative_prompt: str
    bindings: List[InputBinding] = field(default_factory=list)
    sample_extra_args: List[str] = field(default_factory=list)
    box_layout_path: str = ""
    dissect_prompt: str = ""
    init_image: str = ""
    init_strength: float = 0.65
    metadata: Dict[str, Any] = field(default_factory=dict)


def _mode_strength(mode: ControlMode) -> float:
    return {
        ControlMode.LOCK: 0.95,
        ControlMode.IDENTITY: 0.78,
        ControlMode.TRANSFORM: 0.62,
        ControlMode.STYLE: 0.45,
        ControlMode.MOTION: 0.35,
        ControlMode.INPAINT: 0.55,
        ControlMode.GENERATE: 0.25,
    }.get(mode, 0.62)


def _prompt_clause(binding: InputBinding) -> str:
    parts: List[str] = []
    if binding.image and binding.image_role:
        parts.append(f"[image:{binding.entity_id} provides {binding.image_role}]")
    elif binding.image:
        parts.append(f"[image:{binding.entity_id} reference]")
    if binding.text_role:
        parts.append(f"[text:{binding.text_role}]")
    if binding.control == ControlMode.LOCK:
        parts.append(f"[lock:{binding.entity_id}]")
    return " ".join(parts)


def compile_shot_control_plan(
    *,
    shot_id: str,
    shot_index: int,
    base_prompt: str,
    base_negative: str,
    bindings: Sequence[InputBinding],
    global_init_image: str = "",
) -> EntityControlPlan:
    """Merge image+text bindings → prompts and sample.py argv fragments."""
    pos_parts = [base_prompt]
    neg_parts = [base_negative] if base_negative else []
    extra: List[str] = []
    dissect_parts: List[str] = []
    init_image = global_init_image
    init_strength = 0.65
    box_layout = ""
    ref_images: List[str] = []
    ref_weights: List[str] = []

    for b in bindings:
        clause = _prompt_clause(b)
        if clause:
            pos_parts.append(clause)
        if b.negative:
            neg_parts.append(b.negative)
        if b.image:
            ref_images.append(b.image)
            ref_weights.append(f"{b.image}:{b.reference_strength:.2f}")
        if b.part and b.image:
            dissect_parts.append(f"use the {b.part} from image {len(ref_images)}")
        if b.control == ControlMode.LOCK:
            init_image = b.image or init_image
            init_strength = max(init_strength, _mode_strength(b.control))
        elif b.control in (ControlMode.IDENTITY, ControlMode.TRANSFORM) and b.image:
            init_image = init_image or b.image
            init_strength = _mode_strength(b.control)
        if b.rig_json:
            box_layout = b.rig_json
        if b.mask_path:
            extra.extend(["--mask", b.mask_path, "--inpaint-mode", "mdm"])

    if len(ref_images) == 1 and "--reference-image" not in extra:
        extra.extend(["--reference-image", ref_images[0], "--reference-strength", str(bindings[0].reference_strength)])
    elif len(ref_images) > 1:
        extra.extend(["--style-ref", ",".join(ref_weights)])

    if dissect_parts:
        extra.extend(["--dissect-refs", ",".join(ref_images), "--auto-init-from-dissection"])

    if box_layout:
        extra.extend(["--box-layout", box_layout])

    return EntityControlPlan(
        shot_id=shot_id,
        shot_index=shot_index,
        positive_prompt=", ".join(p for p in pos_parts if p.strip()),
        negative_prompt=", ".join(n for n in neg_parts if n.strip()),
        bindings=list(bindings),
        sample_extra_args=extra,
        box_layout_path=box_layout,
        dissect_prompt="; ".join(dissect_parts),
        init_image=init_image,
        init_strength=init_strength,
        metadata={"ref_count": len(ref_images), "binding_count": len(bindings)},
    )


def build_sample_args_for_plan(plan: EntityControlPlan) -> List[str]:
    args = list(plan.sample_extra_args)
    if plan.init_image and "--init-image" not in args:
        args.extend(["--init-image", plan.init_image, "--strength", str(plan.init_strength)])
    return args
