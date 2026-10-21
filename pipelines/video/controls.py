"""Fine-grained control modes for video entities and inputs.

Also holds what used to be: auto_rig, reference_sheet, elements.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

# ==================================================================================================
# controls
# ==================================================================================================


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


def control_mode_help() -> dict[str, str]:
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
    bindings: list[InputBinding] = field(default_factory=list)
    sample_extra_args: list[str] = field(default_factory=list)
    box_layout_path: str = ""
    dissect_prompt: str = ""
    init_image: str = ""
    init_strength: float = 0.65
    metadata: dict[str, Any] = field(default_factory=dict)


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
    parts: list[str] = []
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
    extra: list[str] = []
    dissect_parts: list[str] = []
    init_image = global_init_image
    init_strength = 0.65
    box_layout = ""
    ref_images: list[str] = []
    ref_weights: list[str] = []

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


def build_sample_args_for_plan(plan: EntityControlPlan) -> list[str]:
    args = list(plan.sample_extra_args)
    if plan.init_image and "--init-image" not in args:
        args.extend(["--init-image", plan.init_image, "--strength", str(plan.init_strength)])
    return args


# ==================================================================================================
# auto_rig: Auto-rig characters from a reference image.
#
# Produces a box-layout JSON (regional prompting) with heuristic part regions.
# Optional heavy path uses image dissection when models are available.
# ==================================================================================================


@dataclass(slots=True)
class RigPart:
    name: str
    box: tuple[float, float, float, float]
    prompt: str = ""
    negative: str = ""
    lock: bool = False
    reference_image: str = ""
    reference_weight: float = 0.85


@dataclass(slots=True)
class CharacterRig:
    character_id: str
    source_image: str
    parts: list[RigPart] = field(default_factory=list)
    global_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# Default humanoid rig (normalized 0–1 boxes)
_HUMANOID_PARTS: tuple[tuple[str, tuple[float, float, float, float], bool], ...] = (
    ("head", (0.32, 0.02, 0.68, 0.22), True),
    ("torso", (0.28, 0.20, 0.72, 0.55), True),
    ("left_arm", (0.08, 0.22, 0.32, 0.52), False),
    ("right_arm", (0.68, 0.22, 0.92, 0.52), False),
    ("legs", (0.30, 0.52, 0.70, 0.98), False),
)


def _center_of_mass_box(image_path: Path) -> tuple[float, float, float, float] | None:
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
    parts: Sequence[tuple[str, tuple[float, float, float, float], bool]],
    subject: tuple[float, float, float, float],
) -> list[RigPart]:
    sx0, sy0, sx1, sy1 = subject
    sw, sh = sx1 - sx0, sy1 - sy0
    out: list[RigPart] = []
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
    text_by_part: dict[str, str] | None = None,
    lock_parts: Sequence[str] | None = None,
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
) -> dict[str, Any]:
    regions: list[dict[str, Any]] = []
    for i, pt in enumerate(rig.parts):
        reg: dict[str, Any] = {
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


# ==================================================================================================
# reference_sheet: Auto reference sheet from one hero image (front / profile / full-body crops).
# ==================================================================================================


@dataclass(slots=True)
class ReferenceSheet:
    source: str
    views: list[str]
    boxes: list[tuple[str, tuple[float, float, float, float]]]
    manifest_path: str = ""


def _subject_box(rgb: np.ndarray) -> tuple[float, float, float, float]:

    tmp = Path("_tmp_rig_probe.png")
    try:
        save_frame_rgb(tmp, rgb)
        box = _center_of_mass_box(tmp)
        if box:
            return box
    finally:
        tmp.unlink(missing_ok=True)
    return (0.2, 0.05, 0.8, 0.95)


def _crop_norm(rgb: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    ix0 = int(max(0, min(w - 1, x0 * w)))
    iy0 = int(max(0, min(h - 1, y0 * h)))
    ix1 = int(max(ix0 + 1, min(w, x1 * w)))
    iy1 = int(max(iy0 + 1, min(h, y1 * h)))
    crop = rgb[iy0:iy1, ix0:ix1]
    try:
        import cv2

        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        return crop


def build_reference_sheet(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    target_size: tuple[int, int] | None = None,
) -> ReferenceSheet:
    """
    Heuristic multi-view sheet from a single image.

    Produces front (center), profile (horizontal offset), full-body (wider box) crops
    resized back to frame size — enough for multi-ref consistency like Runway/Kling.
    """
    src = Path(image_path)
    if not src.is_file():
        raise FileNotFoundError(src)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rgb = read_frame_rgb(src)
    h, w = rgb.shape[:2]
    if target_size:
        tw, th = target_size
        try:
            import cv2

            rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            pass

    subj = _subject_box(rgb)
    sx0, sy0, sx1, sy1 = subj
    sw, sh = sx1 - sx0, sy1 - sy0
    cx = (sx0 + sx1) * 0.5

    views_spec = [
        ("front", (sx0 + sw * 0.05, sy0, sx1 - sw * 0.05, sy1)),
        ("profile", (max(0.0, cx - sw * 0.55), sy0, min(1.0, cx + sw * 0.15), sy1)),
        (
            "full_body",
            (
                max(0.0, sx0 - sw * 0.12),
                max(0.0, sy0 - sh * 0.05),
                min(1.0, sx1 + sw * 0.12),
                min(1.0, sy1 + sh * 0.02),
            ),
        ),
    ]
    paths: list[str] = []
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for name, box in views_spec:
        view = _crop_norm(rgb, box)
        fp = out / f"{name}.png"
        save_frame_rgb(fp, view)
        paths.append(str(fp))
        boxes.append((name, box))

    manifest = out / "reference_sheet.json"
    write_reference_sheet_manifest(
        ReferenceSheet(source=str(src.resolve()), views=paths, boxes=boxes, manifest_path=str(manifest)),
        manifest,
    )
    return ReferenceSheet(source=str(src.resolve()), views=paths, boxes=boxes, manifest_path=str(manifest))


def write_reference_sheet_manifest(sheet: ReferenceSheet, path: str | Path) -> Path:
    p = Path(path)
    payload = {
        "source": sheet.source,
        "views": [{"name": n, "path": vp, "box": list(b)} for (n, b), vp in zip(sheet.boxes, sheet.views)],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


# ==================================================================================================
# elements: Kling/Runway-style Elements library — tagged multi-ref subjects.
# ==================================================================================================


@dataclass(slots=True)
class ElementDef:
    """Reusable visual DNA: character, location, style, or prop."""

    id: str
    images: list[str] = field(default_factory=list)
    video_ref: str = ""
    bind_subject: bool = False
    reference_sheet: bool = False
    role: str = "character"
    tags: list[str] = field(default_factory=list)
    reference_strength: float = 0.85
    negative: str = ""
    text_hint: str = ""


@dataclass(slots=True)
class ElementsLibrary:
    elements: dict[str, ElementDef] = field(default_factory=dict)

    def get(self, element_id: str) -> ElementDef | None:
        return self.elements.get(element_id)

    def ids(self) -> list[str]:
        return list(self.elements.keys())


def parse_elements(raw: Any) -> ElementsLibrary:
    lib = ElementsLibrary()
    if not isinstance(raw, Mapping):
        return lib
    for k, v in raw.items():
        eid = str(k)
        if isinstance(v, str):
            lib.elements[eid] = ElementDef(id=eid, images=[v], role="character")
            continue
        if not isinstance(v, Mapping):
            continue
        imgs = v.get("images") or v.get("refs") or []
        if isinstance(imgs, str):
            imgs = [imgs]
        single = str(v.get("image") or v.get("ref") or "")
        if single and single not in imgs:
            imgs = [single] + list(imgs)
        lib.elements[eid] = ElementDef(
            id=eid,
            images=[str(x) for x in imgs if str(x).strip()],
            video_ref=str(v.get("video_ref") or v.get("video") or v.get("clip") or ""),
            bind_subject=bool(v.get("bind_subject") or v.get("bind") or v.get("lock_subject")),
            reference_sheet=bool(v.get("reference_sheet") or v.get("sheet")),
            role=str(v.get("role") or v.get("type") or "character"),
            tags=[str(t) for t in (v.get("tags") or [])],
            reference_strength=float(v.get("reference_strength") or v.get("strength") or 0.85),
            negative=str(v.get("negative") or ""),
            text_hint=str(v.get("text") or v.get("hint") or ""),
        )
    return lib


def resolve_element_images(
    element: ElementDef,
    work_dir: str | Path,
    *,
    max_refs: int = 3,
) -> list[str]:
    """Return up to ``max_refs`` image paths; optionally build reference sheet."""
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    paths = [p for p in element.images if p and Path(p).is_file()]
    if element.reference_sheet and len(paths) <= 1 and paths:
        sheet = build_reference_sheet(paths[0], wd / f"sheet_{element.id}")
        paths = [str(p) for p in sheet.views if Path(p).is_file()]
    return paths[:max_refs]


def compile_element_refs(
    library: ElementsLibrary,
    element_ids: Sequence[str],
    work_dir: str | Path,
) -> tuple[list[str], list[str], list[str]]:
    """
    Compile element IDs → (image_paths, weights, video_refs).

    ``weights`` are ``path:strength`` strings for ``--style-ref``.
    """
    images: list[str] = []
    weights: list[str] = []
    videos: list[str] = []
    for eid in element_ids:
        el = library.get(eid)
        if not el:
            continue
        refs = resolve_element_images(el, work_dir)
        for rp in refs:
            if rp not in images:
                images.append(rp)
                weights.append(f"{rp}:{el.reference_strength:.2f}")
        if el.video_ref and Path(el.video_ref).is_file():
            videos.append(el.video_ref)
    return images, weights, videos


__all__ = [
    "ControlMode",
    "EntityControlPlan",
    "InputBinding",
    "MediaInput",
    "build_sample_args_for_plan",
    "compile_shot_control_plan",
    "control_mode_help",
    "CharacterRig",
    "RigPart",
    "auto_rig_character",
    "rig_to_box_layout",
    "write_rig_box_layout",
    "ReferenceSheet",
    "build_reference_sheet",
    "write_reference_sheet_manifest",
    "ElementDef",
    "ElementsLibrary",
    "compile_element_refs",
    "parse_elements",
    "resolve_element_images",
]
