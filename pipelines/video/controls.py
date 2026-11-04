"""Conditioning controls: entity control modes, rigs, reference sheets, elements, multimodal refs,
camera paths, in-gen scene cuts, motion intelligence, native audio and lip-sync.
"""

from __future__ import annotations

import json
import math
import wave
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pipelines.video.video_io import mux_audio_onto_video, read_frame_rgb, save_frame_rgb

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
            from utils.generation.editing import dissect_images_to_parts

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
        "views": [{"name": n, "path": vp, "box": list(b)} for (n, b), vp in zip(sheet.boxes, sheet.views, strict=False)],
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


class LeadersBundle:
    def __init__(self) -> None:
        self.opts: Any = None
        self.ref_positive: str = ""
        self.ref_negative: str = ""
        self.motion_positive: str = ""
        self.motion_negative: str = ""
        self.camera_prompt: str = ""
        self.audio_plan: Any = None
        self.cut_plan: Any = None
        self.camera_path: Any = None
        self.motion_plan: Any = None
        self.ref_pack: Any = None
        self.ops: list[str] = []


def compile_leaders_stack(
    opts: Any,
    *,
    prompt: str = "",
    duration_sec: float = 6.0,
    fps: float = 24.0,
    frame_count: int = 0,
    multimodal_refs: Any = None,
    enable_native_audio: bool = True,
    enable_scene_cuts: bool = True,
    enable_camera_path: bool = True,
    enable_motion_intel: bool = True,
) -> LeadersBundle:

    bundle = LeadersBundle()
    bundle.opts = opts

    pack = parse_multimodal_refs(multimodal_refs)
    bundle.ref_pack = pack
    ok, issues = ref_budget_ok(pack)
    if pack.refs:
        bundle.ref_positive, bundle.ref_negative = compile_ref_prompt_garnish(pack)
        bundle.ops.append(f"multimodal_refs:{pack.counts['total']}")
        if not ok:
            bundle.ops.append(f"ref_budget_warn:{','.join(issues)}")
        # Feed identity images into identity_bind refs
        id_paths = resolve_identity_paths(pack)
        if id_paths:
            existing = tuple(getattr(opts, "identity_refs", ()) or ())
            bundle.opts = replace(opts, identity_refs=existing + tuple(id_paths), identity_bind=True)
            opts = bundle.opts

    if enable_motion_intel:
        mplan = compile_motion_intelligence(prompt, style=str(getattr(opts, "motion_grammar", "") or ""))
        bundle.motion_plan = mplan
        bundle.motion_positive, bundle.motion_negative = plan_to_prompt_fragments(mplan)
        bundle.opts = apply_gate_boosts(opts, mplan)
        opts = bundle.opts
        bundle.ops.append(f"motion_intel:{mplan.primary_action}")

    if enable_camera_path:
        cpath = solve_camera_path(prompt, rig_movement=str(getattr(opts, "motion_grammar", "") or ""))
        bundle.camera_path = cpath
        bundle.camera_prompt = path_to_prompt(cpath)
        bundle.ops.append(f"camera_path:{cpath.preset}")

    n_frames = int(frame_count) if frame_count > 0 else max(8, int(duration_sec * fps))
    if enable_scene_cuts:
        cuts = plan_scene_cuts(prompt, frame_count=n_frames, fps=fps)
        bundle.cut_plan = cuts
        if cuts.cuts:
            bundle.ops.append(f"scene_cuts:{len(cuts.cuts)}")

    if enable_native_audio or bool(getattr(opts, "native_audio", False)):
        voice = resolve_audio_paths(pack) if pack else []
        bundle.audio_plan = plan_native_audio(prompt, duration_sec=duration_sec, voice_refs=voice)
        bundle.opts = replace(opts, native_audio=True)
        bundle.ops.append("native_audio:planned")

    # Merge prompt garnish onto motion_grammar fields
    pos = ", ".join(
        x
        for x in (
            getattr(bundle.opts, "motion_grammar_positive", "") or "",
            bundle.ref_positive,
            bundle.motion_positive,
            bundle.camera_prompt,
        )
        if x
    )
    neg = ", ".join(
        x
        for x in (
            getattr(bundle.opts, "motion_grammar_negative", "") or "",
            bundle.ref_negative,
            bundle.motion_negative,
        )
        if x
    )
    bundle.opts = replace(bundle.opts, motion_grammar_positive=pos, motion_grammar_negative=neg)
    return bundle


@dataclass(slots=True)
class CameraKeyframe:
    t: float  # 0..1
    pan: float = 0.0  # degrees, +right
    tilt: float = 0.0  # degrees, +up
    dolly: float = 0.0  # +push in
    roll: float = 0.0
    zoom: float = 1.0


@dataclass(slots=True)
class CameraPath:
    keys: list[CameraKeyframe] = field(default_factory=list)
    preset: str = ""
    notes: list[str] = field(default_factory=list)

    def sample(self, u: float) -> CameraKeyframe:
        if not self.keys:
            return CameraKeyframe(t=u)
        u = float(np_clip(u, 0.0, 1.0))
        if u <= self.keys[0].t:
            return self.keys[0]
        if u >= self.keys[-1].t:
            return self.keys[-1]
        for a, b in zip(self.keys[:-1], self.keys[1:], strict=False):
            if a.t <= u <= b.t:
                w = 0.0 if b.t <= a.t else (u - a.t) / (b.t - a.t)
                w = _smooth(w)
                return CameraKeyframe(
                    t=u,
                    pan=_lerp(a.pan, b.pan, w),
                    tilt=_lerp(a.tilt, b.tilt, w),
                    dolly=_lerp(a.dolly, b.dolly, w),
                    roll=_lerp(a.roll, b.roll, w),
                    zoom=_lerp(a.zoom, b.zoom, w),
                )
        return self.keys[-1]


def np_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _lerp(a: float, b: float, w: float) -> float:
    return a + (b - a) * w


def _smooth(t: float) -> float:
    t = np_clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


_PRESET_PATHS: dict[str, list[CameraKeyframe]] = {
    "dolly_in": [
        CameraKeyframe(0.0, dolly=0.0, zoom=1.0),
        CameraKeyframe(1.0, dolly=1.0, zoom=1.15),
    ],
    "orbit": [
        CameraKeyframe(0.0, pan=-25.0),
        CameraKeyframe(0.5, pan=0.0, dolly=0.3),
        CameraKeyframe(1.0, pan=25.0),
    ],
    "hitchcock": [
        CameraKeyframe(0.0, dolly=0.0, zoom=1.0),
        CameraKeyframe(1.0, dolly=1.2, zoom=0.75),
    ],
    "crane_up": [
        CameraKeyframe(0.0, tilt=-8.0, dolly=0.0),
        CameraKeyframe(1.0, tilt=12.0, dolly=0.4),
    ],
    "handheld": [
        CameraKeyframe(0.0, pan=-2.0, tilt=1.0, roll=-0.5),
        CameraKeyframe(0.33, pan=2.5, tilt=-1.5, roll=0.8),
        CameraKeyframe(0.66, pan=-1.5, tilt=2.0, roll=-0.4),
        CameraKeyframe(1.0, pan=1.0, tilt=-0.5, roll=0.2),
    ],
    "static": [CameraKeyframe(0.0), CameraKeyframe(1.0)],
}


def solve_camera_path(
    prompt: str = "",
    *,
    preset: str = "",
    rig_movement: str = "",
    duration_sec: float = 6.0,
) -> CameraPath:
    text = f"{prompt} {rig_movement} {preset}".lower()
    chosen = (preset or "").strip().lower().replace("-", "_")
    if not chosen or chosen not in _PRESET_PATHS:
        rules = [
            (("hitchcock", "vertigo", "dolly zoom"), "hitchcock"),
            (("orbit", "circle", "around"), "orbit"),
            (("crane", "jib", "rise"), "crane_up"),
            (("handheld", "docu", "shaky"), "handheld"),
            (("dolly", "push in", "push-in", "track in"), "dolly_in"),
            (("static", "locked off", "tripod"), "static"),
        ]
        for keys, name in rules:
            if any(k in text for k in keys):
                chosen = name
                break
        else:
            chosen = "dolly_in" if duration_sec >= 4 else "static"
    keys = [
        CameraKeyframe(**{f.name: getattr(kf, f.name) for f in fields(CameraKeyframe)}) for kf in _PRESET_PATHS[chosen]
    ]
    return CameraPath(keys=keys, preset=chosen, notes=[f"preset={chosen}"])


def path_to_prompt(path: CameraPath) -> str:
    if not path.keys:
        return ""
    mid = path.sample(0.5)
    end = path.sample(1.0)
    bits = [f"camera path:{path.preset}"]
    if abs(end.dolly - path.keys[0].dolly) > 0.1:
        bits.append("motivated dolly")
    if abs(end.pan - path.keys[0].pan) > 5:
        bits.append(f"pan {'right' if end.pan > 0 else 'left'}")
    if abs(end.tilt) > 4:
        bits.append("tilt accent")
    if path.preset == "hitchcock":
        bits.append("dolly zoom vertigo effect")
    if path.preset == "handheld":
        bits.append("subtle handheld breathing")
    bits.append(f"mid zoom {mid.zoom:.2f}")
    return ", ".join(bits)


def path_to_motion_deltas(path: CameraPath, *, frames: int) -> list[dict[str, float]]:
    """Per-frame deltas useful for motion templates / control."""
    n = max(1, int(frames))
    out: list[dict[str, float]] = []
    prev = path.sample(0.0)
    for i in range(n):
        u = i / max(1, n - 1)
        cur = path.sample(u)
        out.append(
            {
                "pan": cur.pan - prev.pan,
                "tilt": cur.tilt - prev.tilt,
                "dolly": cur.dolly - prev.dolly,
                "roll": cur.roll - prev.roll,
                "zoom": cur.zoom - prev.zoom,
            }
        )
        prev = cur
    return out


@dataclass(slots=True)
class CutBeat:
    frame_index: int
    kind: str = "hard"  # hard|smash|tempo
    note: str = ""


@dataclass(slots=True)
class SceneCutPlan:
    cuts: list[CutBeat] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def plan_scene_cuts(
    prompt: str,
    *,
    frame_count: int,
    fps: float = 24.0,
    max_cuts: int = 4,
) -> SceneCutPlan:
    text = (prompt or "").lower()
    n = max(1, int(frame_count))
    cuts: list[CutBeat] = []
    notes: list[str] = []

    # Explicit multi-shot language
    markers = 0
    for token in ("then ", "cut to", "wide shot", "close-up", "next,", "meanwhile", "later"):
        if token in text:
            markers += 1
    target = min(max_cuts, max(markers, 1 if "multi" in text or "montage" in text else 0))
    if target <= 0 and n >= int(fps * 8):
        # Long clip → at least one mid cut for cinematic pacing (Wan multi-shot default)
        target = 1
        notes.append("auto_mid_cut_long_clip")

    if target <= 0:
        return SceneCutPlan(notes=["no_cuts"])

    # Evenly space cuts away from edges
    span = n - 4
    for i in range(target):
        fi = 2 + int((i + 1) * span / (target + 1))
        kind = "smash" if "montage" in text or "flash" in text else "hard"
        cuts.append(CutBeat(frame_index=fi, kind=kind, note=f"beat_{i}"))
    notes.append(f"planned_cuts={len(cuts)}")
    return SceneCutPlan(cuts=cuts, notes=notes)


def apply_scene_cuts(
    frame_paths: list[Path],
    plan: SceneCutPlan,
    *,
    flash_frames: int = 1,
) -> list[Path]:
    """
    Emphasize planned cuts: optional 1-frame flash + micro hold on either side.
    Does not change frame count.
    """
    paths = list(frame_paths)
    if not plan.cuts or len(paths) < 6:
        return paths
    for cut in plan.cuts:
        i = int(np.clip(cut.frame_index, 1, len(paths) - 2))
        if cut.kind == "smash" and flash_frames > 0:
            rgb = read_frame_rgb(paths[i])
            flash = np.clip(rgb.astype(np.float32) * 1.8 + 40, 0, 255).astype(np.uint8)
            save_frame_rgb(paths[i], flash)
        # Micro hold: duplicate previous onto i-1 for snappier cut perception
        if i >= 2:
            prev = read_frame_rgb(paths[i - 2])
            save_frame_rgb(paths[i - 1], prev)
    return paths


@dataclass(slots=True)
class MotionPlan:
    primary_action: str = "idle"
    contacts: list[str] = field(default_factory=list)
    secondary: list[str] = field(default_factory=list)
    camera: str = ""
    physics_hints: list[str] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    gate_boosts: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


_ACTION_VERBS = (
    ("run", "running"),
    ("walk", "walking"),
    ("jump", "jumping"),
    ("fall", "falling"),
    ("dance", "dancing"),
    ("fight", "fighting"),
    ("drive", "driving"),
    ("fly", "flying"),
    ("swim", "swimming"),
    ("pour", "pouring liquid"),
    ("explode", "explosion"),
    ("turntable", "product turntable orbit"),
)


def compile_motion_intelligence(prompt: str, *, style: str = "") -> MotionPlan:
    text = f"{prompt} {style}".lower()
    plan = MotionPlan()

    for key, action in _ACTION_VERBS:
        if key in text:
            plan.primary_action = action
            break

    # Contacts
    if any(k in text for k in ("foot", "walk", "run", "stand", "ground", "floor")):
        plan.contacts.append("feet_to_ground")
        plan.gate_boosts["min_contact"] = 0.45
        plan.physics_hints.append("weight transfer through feet")
    if any(k in text for k in ("hand", "grab", "hold", "touch", "shake hands")):
        plan.contacts.append("hand_contact")
        plan.gate_boosts["min_extremity"] = 0.4
    if any(k in text for k in ("sit", "chair", "lean")):
        plan.contacts.append("body_to_furniture")

    # Secondary
    if any(k in text for k in ("hair", "cloth", "dress", "coat", "cape")):
        plan.secondary.append("cloth_follow_through")
        plan.physics_hints.append("cloth lag behind body")
        plan.forbid.append("cloth clipping through limbs")
    if any(k in text for k in ("water", "rain", "splash", "pour", "liquid")):
        plan.secondary.append("fluid_secondary")
        plan.physics_hints.append("gravity on droplets")
        plan.forbid.append("water floating upward")
        plan.gate_boosts["min_physics"] = 0.55
    if any(k in text for k in ("bag", "phone", "gun", "sword", "cup")):
        plan.secondary.append("prop_attachment")
        plan.gate_boosts["min_secondary"] = 0.4

    # Multi-body
    if any(k in text for k in ("crowd", "two people", "three ", "group", "together")):
        plan.physics_hints.append("independent body trajectories")
        plan.forbid.append("merged silhouettes")
        plan.gate_boosts["min_count"] = 0.45
        plan.gate_boosts["min_occlusion"] = 0.35

    # Camera from prompt
    for cam in ("orbit", "dolly", "handheld", "crane", "static", "hitchcock"):
        if cam in text:
            plan.camera = cam
            break

    plan.forbid.extend(
        [
            "teleporting limbs",
            "anti-gravity falls",
            "sliding on ice feet",
            "identity morph mid-action",
        ]
    )
    plan.notes.append(f"action={plan.primary_action}")
    return plan


def plan_to_prompt_fragments(plan: MotionPlan) -> tuple[str, str]:
    pos = [
        plan.primary_action,
        *plan.physics_hints,
        *plan.secondary,
        *plan.contacts,
    ]
    if plan.camera:
        pos.append(f"camera:{plan.camera}")
    neg = list(plan.forbid)
    return ", ".join(p for p in pos if p), ", ".join(neg)


def apply_gate_boosts(opts: Any, plan: MotionPlan) -> Any:
    """Raise ProcessOptions min_* thresholds from the motion plan."""
    from dataclasses import replace

    kw = {}
    for k, v in plan.gate_boosts.items():
        cur = float(getattr(opts, k, 0.0) or 0.0)
        kw[k] = max(cur, float(v))
    # Enable related repairs
    if "feet_to_ground" in plan.contacts:
        kw["contact_ground"] = True
    if "prop_attachment" in plan.secondary:
        kw["secondary_track"] = True
    if "fluid_secondary" in plan.secondary or plan.gate_boosts.get("min_physics"):
        kw["physics_gate"] = True
    if not kw:
        return opts
    return replace(opts, **kw)


RefRole = Literal[
    "identity",
    "product",
    "style",
    "camera",
    "voice",
    "motion",
    "layout",
    "scene",
    "wardrobe",
]


@dataclass(slots=True)
class ModalityRef:
    path: str
    role: RefRole = "identity"
    modality: Literal["image", "video", "audio"] = "image"
    strength: float = 0.85
    tag: str = ""
    notes: str = ""


@dataclass(slots=True)
class MultimodalRefPack:
    """Hailuo/Seedance-compatible reference budget."""

    refs: list[ModalityRef] = field(default_factory=list)
    max_images: int = 30
    max_videos: int = 10
    max_audios: int = 10

    def by_role(self, role: str) -> list[ModalityRef]:
        return [r for r in self.refs if r.role == role]

    def by_modality(self, modality: str) -> list[ModalityRef]:
        return [r for r in self.refs if r.modality == modality]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "image": len(self.by_modality("image")),
            "video": len(self.by_modality("video")),
            "audio": len(self.by_modality("audio")),
            "total": len(self.refs),
        }


_EXT_MOD = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".bmp": "image",
    ".mp4": "video",
    ".mov": "video",
    ".webm": "video",
    ".mkv": "video",
    ".wav": "audio",
    ".mp3": "audio",
    ".aac": "audio",
    ".m4a": "audio",
    ".flac": "audio",
}


def _guess_modality(path: str) -> str:
    return _EXT_MOD.get(Path(path).suffix.lower(), "image")


def _coerce_role(raw: str) -> RefRole:
    key = (raw or "identity").strip().lower().replace("-", "_")
    aliases = {
        "character": "identity",
        "face": "identity",
        "subject": "identity",
        "sku": "product",
        "logo": "product",
        "look": "style",
        "aesthetic": "style",
        "cam": "camera",
        "lens": "camera",
        "speech": "voice",
        "sfx": "voice",
        "music": "voice",
        "action": "motion",
        "mocap": "motion",
        "blockout": "layout",
        "3d": "layout",
        "location": "scene",
        "set": "scene",
        "outfit": "wardrobe",
        "costume": "wardrobe",
    }
    key = aliases.get(key, key)
    allowed: tuple[RefRole, ...] = (
        "identity",
        "product",
        "style",
        "camera",
        "voice",
        "motion",
        "layout",
        "scene",
        "wardrobe",
    )
    return key if key in allowed else "identity"  # type: ignore[return-value]


def parse_multimodal_refs(raw: Any) -> MultimodalRefPack:
    """
    Accepts:
      - list of paths / dicts
      - dict with keys images/videos/audios or role buckets
      - Seedance-like ``references: [{path, role, ...}]``
    """
    pack = MultimodalRefPack()
    if raw is None:
        return pack
    items: list[Any] = []
    if isinstance(raw, Mapping):
        if "references" in raw or "refs" in raw:
            items = list(raw.get("references") or raw.get("refs") or [])
        else:
            for key in ("images", "image", "videos", "video", "audios", "audio"):
                val = raw.get(key)
                if isinstance(val, str):
                    items.append({"path": val, "role": "identity" if "image" in key else key.rstrip("s")})
                elif isinstance(val, Sequence):
                    for v in val:
                        if isinstance(v, str):
                            items.append({"path": v, "role": "identity" if "image" in key else key.rstrip("s")})
                        elif isinstance(v, Mapping):
                            items.append(v)
            # Role buckets: {"identity": [...], "style": [...]}
            for role_key, val in raw.items():
                if role_key in ("images", "videos", "audios", "references", "refs", "image", "video", "audio"):
                    continue
                if isinstance(val, (str, Mapping)):
                    val = [val]
                if isinstance(val, Sequence):
                    for v in val:
                        if isinstance(v, str):
                            items.append({"path": v, "role": role_key})
                        elif isinstance(v, Mapping):
                            d = dict(v)
                            d.setdefault("role", role_key)
                            items.append(d)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = list(raw)
    else:
        return pack

    for it in items:
        if isinstance(it, str):
            path = it.strip()
            if not path:
                continue
            pack.refs.append(
                ModalityRef(path=path, role="identity", modality=_guess_modality(path))  # type: ignore[arg-type]
            )
            continue
        if not isinstance(it, Mapping):
            continue
        path = str(it.get("path") or it.get("url") or it.get("file") or it.get("src") or "").strip()
        if not path:
            continue
        modality = str(it.get("modality") or it.get("type") or _guess_modality(path)).lower()
        if modality not in ("image", "video", "audio"):
            modality = _guess_modality(path)
        pack.refs.append(
            ModalityRef(
                path=path,
                role=_coerce_role(str(it.get("role") or it.get("kind") or "identity")),
                modality=modality,  # type: ignore[arg-type]
                strength=float(it.get("strength") or it.get("weight") or 0.85),
                tag=str(it.get("tag") or it.get("id") or ""),
                notes=str(it.get("notes") or it.get("hint") or ""),
            )
        )
    return pack


def ref_budget_ok(pack: MultimodalRefPack) -> tuple[bool, list[str]]:
    """Seedance 2.5-ish budgets; Soft-warn rather than hard-fail."""
    c = pack.counts
    issues = []
    if c["image"] > pack.max_images:
        issues.append(f"images={c['image']}>{pack.max_images}")
    if c["video"] > pack.max_videos:
        issues.append(f"videos={c['video']}>{pack.max_videos}")
    if c["audio"] > pack.max_audios:
        issues.append(f"audios={c['audio']}>{pack.max_audios}")
    if c["total"] > 50:
        issues.append(f"total={c['total']}>50")
    return len(issues) == 0, issues


def compile_ref_prompt_garnish(pack: MultimodalRefPack) -> tuple[str, str]:
    """Positive/negative fragments describing bound roles (for keyframe prompts)."""
    pos_bits: list[str] = []
    neg_bits: list[str] = []
    for role in ("identity", "product", "wardrobe", "style", "scene", "camera", "motion"):
        rs = pack.by_role(role)
        if not rs:
            continue
        tags = ", ".join(r.tag or Path(r.path).stem for r in rs[:4])
        pos_bits.append(f"{role} locked to refs [{tags}]")
    if pack.by_role("identity"):
        neg_bits.append("identity drift, face morph, wardrobe change")
    if pack.by_role("product"):
        neg_bits.append("logo melt, warped label text, SKU morph")
    if pack.by_role("voice"):
        pos_bits.append("lip sync to voice reference")
    return "; ".join(pos_bits), ", ".join(neg_bits)


def resolve_identity_paths(pack: MultimodalRefPack) -> list[str]:
    out: list[str] = []
    for r in pack.by_role("identity") + pack.by_role("wardrobe") + pack.by_role("product"):
        if r.modality == "image" and Path(r.path).is_file():
            out.append(r.path)
    return out


def resolve_audio_paths(pack: MultimodalRefPack) -> list[str]:
    return [r.path for r in pack.by_modality("audio") if Path(r.path).is_file() or r.path.startswith("http")]


def resolve_motion_video_paths(pack: MultimodalRefPack) -> list[str]:
    vids = pack.by_role("motion") + pack.by_role("camera")
    return [r.path for r in vids if r.modality == "video"]


@dataclass(slots=True)
class AudioEvent:
    kind: str  # dialogue|sfx|ambience|music
    start_sec: float
    duration_sec: float
    hint: str = ""
    ref_path: str = ""


@dataclass(slots=True)
class AudioPlan:
    duration_sec: float
    sample_rate: int = 32000
    events: list[AudioEvent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def plan_native_audio(
    prompt: str,
    *,
    duration_sec: float = 6.0,
    voice_refs: list[str] | None = None,
) -> AudioPlan:
    text = (prompt or "").lower()
    dur = max(1.0, float(duration_sec))
    events: list[AudioEvent] = []
    notes: list[str] = []

    # Ambience always
    ambience = "room tone"
    for key, label in (
        ("rain", "rain ambience"),
        ("ocean", "ocean waves"),
        ("city", "city traffic bed"),
        ("forest", "forest wind"),
        ("crowd", "crowd murmur"),
        ("office", "office hush"),
    ):
        if key in text:
            ambience = label
            break
    events.append(AudioEvent(kind="ambience", start_sec=0.0, duration_sec=dur, hint=ambience))

    # Dialogue if speech verbs / quotes
    if any(k in text for k in ("says", "said", "speak", "talk", "dialogue", "sing", '"', "'")):
        events.append(
            AudioEvent(
                kind="dialogue",
                start_sec=min(0.4, dur * 0.1),
                duration_sec=min(dur * 0.6, dur - 0.4),
                hint="spoken dialogue",
                ref_path=(voice_refs or [""])[0] if voice_refs else "",
            )
        )
        notes.append("dialogue_planned")

    # SFX hits from action verbs
    for verb, when in (
        ("door", 0.2),
        ("gunshot", 0.5),
        ("explosion", 0.55),
        ("footstep", 0.15),
        ("whoosh", 0.3),
        ("splash", 0.4),
    ):
        if verb in text:
            events.append(
                AudioEvent(
                    kind="sfx",
                    start_sec=min(dur * when, dur - 0.3),
                    duration_sec=0.35,
                    hint=verb,
                )
            )

    if voice_refs:
        notes.append(f"voice_refs={len(voice_refs)}")
    return AudioPlan(duration_sec=dur, events=events, notes=notes)


def _write_wav(path: Path, samples: np.ndarray, *, sr: int = 32000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # stereo float -> int16
    if samples.ndim == 1:
        samples = np.stack([samples, samples], axis=1)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def synthesize_stereo_bed(plan: AudioPlan, out_path: str | Path) -> Path:
    """Procedural stereo bed — not a neural codec, but a real attachable track."""
    sr = int(plan.sample_rate)
    n = int(sr * plan.duration_sec)
    t = np.arange(n, dtype=np.float64) / sr
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    for ev in plan.events:
        i0 = int(ev.start_sec * sr)
        i1 = min(n, int((ev.start_sec + ev.duration_sec) * sr))
        if i1 <= i0:
            continue
        seg_t = t[i0:i1] - t[i0]
        if ev.kind == "ambience":
            # Soft noise bed + slow LFO
            rng = np.random.default_rng(abs(hash(ev.hint)) % (2**32))
            noise = rng.normal(0, 0.04, size=i1 - i0)
            lfo = 0.5 + 0.5 * np.sin(2 * math.pi * 0.15 * seg_t)
            left[i0:i1] += noise * lfo
            right[i0:i1] += noise * (1.0 - 0.3 * lfo)
        elif ev.kind == "dialogue":
            # Formant-ish buzz (placeholder voice)
            fund = 140.0 if "female" not in ev.hint else 220.0
            env = np.sin(np.pi * np.clip(seg_t / max(seg_t[-1], 1e-6), 0, 1)) ** 0.6
            sig = (
                0.12
                * env
                * (
                    np.sin(2 * math.pi * fund * seg_t)
                    + 0.4 * np.sin(2 * math.pi * fund * 2 * seg_t)
                    + 0.2 * np.sin(2 * math.pi * fund * 3 * seg_t)
                )
            )
            left[i0:i1] += sig
            right[i0:i1] += sig * 0.95
        elif ev.kind == "sfx":
            env = np.exp(-seg_t * 8.0)
            burst = 0.25 * env * np.sin(2 * math.pi * 400 * seg_t * (1 + seg_t))
            left[i0:i1] += burst
            right[i0:i1] += burst * 0.85
        elif ev.kind == "music":
            chord = sum(np.sin(2 * math.pi * f * seg_t) for f in (220.0, 277.0, 330.0)) / 3.0
            left[i0:i1] += 0.06 * chord
            right[i0:i1] += 0.06 * chord * 0.9

    peak = max(1e-6, float(np.max(np.abs(np.stack([left, right])))))
    left /= peak * 1.2
    right /= peak * 1.2
    return _write_wav(Path(out_path), np.stack([left, right], axis=1), sr=sr)


def energy_envelope(wav_path: str | Path, *, fps: float = 24.0, duration_sec: float = 0.0) -> np.ndarray:
    """Per-frame audio energy for lip-sync correlation."""
    path = Path(wav_path)
    if not path.is_file():
        return np.zeros(1, dtype=np.float32)
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
        sw = w.getsampwidth()
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    dur = duration_sec if duration_sec > 0 else len(data) / max(sr, 1)
    n_frames = max(1, int(round(dur * fps)))
    win = max(1, len(data) // n_frames)
    env = []
    for i in range(n_frames):
        chunk = data[i * win : (i + 1) * win]
        env.append(float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0)
    arr = np.asarray(env, dtype=np.float32)
    if arr.max() > 1e-6:
        arr = arr / arr.max()
    return arr


def attach_native_audio(
    video_path: str | Path,
    plan: AudioPlan,
    work_dir: str | Path,
    *,
    prefer_ref: str | Path | None = None,
) -> Path:
    """Synthesize or reuse ref audio and mux onto video."""
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    if prefer_ref and Path(prefer_ref).is_file():
        audio = Path(prefer_ref)
    else:
        audio = synthesize_stereo_bed(plan, wd / "native_audio.wav")
    return mux_audio_onto_video(video_path, audio, out_path=Path(video_path))


@dataclass(slots=True)
class LipSyncReport:
    score: float
    correlation: float = 0.0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _mouth_activity(rgb: np.ndarray) -> float:
    """Edge energy in lower-center face band."""
    h, w = rgb.shape[:2]
    y0, y1 = int(h * 0.55), int(h * 0.78)
    x0, x1 = int(w * 0.30), int(w * 0.70)
    band = rgb[y0:y1, x0:x1].astype(np.float32).mean(axis=2)
    if band.size < 4:
        return 0.0
    gx = np.abs(np.diff(band, axis=1)).mean()
    gy = np.abs(np.diff(band, axis=0)).mean()
    return float(gx + gy)


def score_lip_sync(
    frame_paths: list[Path] | list[str],
    audio_energy: np.ndarray | None = None,
    *,
    wav_path: str | Path | None = None,
    fps: float = 24.0,
) -> LipSyncReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return LipSyncReport(score=1.0)
    acts = np.asarray([_mouth_activity(read_frame_rgb(p)) for p in paths], dtype=np.float32)
    if acts.max() > 1e-6:
        acts = acts / acts.max()
    if audio_energy is None and wav_path:
        audio_energy = energy_envelope(wav_path, fps=fps, duration_sec=len(paths) / max(fps, 1e-6))
    if audio_energy is None or len(audio_energy) < 2:
        # No audio: penalize high mouth chatter (talking without sound)
        chatter = float(np.std(acts))
        score = float(np.clip(1.0 - chatter * 2.0, 0.0, 1.0))
        return LipSyncReport(score=score, correlation=0.0, notes=["no_audio_energy"])
    # Resample audio to frame count
    idx = np.linspace(0, len(audio_energy) - 1, len(acts)).astype(int)
    ae = audio_energy[idx]
    if ae.std() < 1e-6 or acts.std() < 1e-6:
        corr = 0.0
    else:
        corr = float(np.corrcoef(acts, ae)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    score = float(np.clip(0.5 + 0.5 * corr, 0.0, 1.0))
    notes = []
    if corr < 0.15:
        notes.append(f"lip_corr={corr:.2f}")
    return LipSyncReport(score=score, correlation=corr, notes=notes)


def apply_lip_sync_nudge(
    frame_paths: list[Path],
    audio_energy: np.ndarray,
    *,
    strength: float = 0.35,
) -> LipSyncReport:
    """
    Soft temporal blend of mouth band toward neighbors so activity tracks energy:
    high energy → prefer sharper (current) mouth; low energy → blend toward calmer neighbor.
    """
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 3 or audio_energy is None or len(audio_energy) < 2:
        return LipSyncReport(score=1.0)
    frames = [read_frame_rgb(p).astype(np.float32) for p in paths]
    idx = np.linspace(0, len(audio_energy) - 1, len(frames)).astype(int)
    ae = audio_energy[idx]
    if ae.max() > 1e-6:
        ae = ae / ae.max()
    a = float(np.clip(strength, 0.0, 1.0))
    repaired = 0
    for i in range(1, len(frames) - 1):
        h, w = frames[i].shape[:2]
        y0, y1 = int(h * 0.55), int(h * 0.78)
        x0, x1 = int(w * 0.30), int(w * 0.70)
        # Low energy → blend toward average of neighbors (closed-ish)
        calm = 0.5 * (frames[i - 1] + frames[i + 1])
        w_calm = (1.0 - float(ae[i])) * a
        if w_calm < 0.05:
            continue
        region = frames[i][y0:y1, x0:x1]
        frames[i][y0:y1, x0:x1] = region * (1.0 - w_calm) + calm[y0:y1, x0:x1] * w_calm
        save_frame_rgb(paths[i], np.clip(frames[i], 0, 255).astype(np.uint8))
        repaired += 1
    after = score_lip_sync(paths, audio_energy=ae)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"lip_nudge={repaired}")
    return after


@dataclass(slots=True)
class TimedEvent:
    """One authored beat on the master (or shot-local) clock."""

    at_sec: float = 0.0
    until_sec: float | None = None
    prompt: str = ""
    negative: str = ""
    effect: str = ""
    camera: str = ""
    action: str = ""
    refs: tuple[str, ...] = ()
    shot_id: str = ""  # empty = global timeline
    strength: float | None = None  # optional edit_strength override while active
    concurrent: bool = True  # stack with other events at same time
    tag: str = ""

    @property
    def end_sec(self) -> float:
        if self.until_sec is not None:
            return float(self.until_sec)
        return float(self.at_sec) + 0.4


@dataclass
class DirectorTimeline:
    events: list[TimedEvent] = field(default_factory=list)
    refs: list[dict[str, Any]] = field(default_factory=list)
    mode: str = ""  # t2v|i2v|v2v|i2i|cgi
    init_image: str = ""
    style_image: str = ""
    motion_video: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [asdict(e) for e in self.events],
            "refs": list(self.refs),
            "mode": self.mode,
            "init_image": self.init_image,
            "style_image": self.style_image,
            "motion_video": self.motion_video,
            "notes": list(self.notes),
        }


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_event(row: Mapping[str, Any]) -> TimedEvent | None:
    prompt = str(row.get("prompt") or row.get("text") or row.get("description") or "").strip()
    action = str(row.get("action") or row.get("do") or "").strip()
    effect = str(row.get("effect") or row.get("fx") or "").strip()
    camera = str(row.get("camera") or row.get("camera_move") or "").strip()
    if not any((prompt, action, effect, camera)):
        return None
    refs_raw = row.get("refs") or row.get("references") or row.get("images") or ()
    refs: list[str] = []
    if isinstance(refs_raw, str):
        refs = [refs_raw] if refs_raw.strip() else []
    elif isinstance(refs_raw, Sequence):
        refs = [str(x).strip() for x in refs_raw if str(x).strip()]
    until = row.get("until_sec")
    if until is None:
        until = row.get("end_sec")
    strength = row.get("strength")
    if strength is None:
        strength = row.get("edit_strength")
    return TimedEvent(
        at_sec=_as_float(row.get("at_sec") or row.get("t") or row.get("time") or 0.0),
        until_sec=_as_float(until) if until is not None else None,
        prompt=prompt,
        negative=str(row.get("negative") or ""),
        effect=effect,
        camera=camera,
        action=action,
        refs=tuple(refs),
        shot_id=str(row.get("shot_id") or row.get("shot") or ""),
        strength=float(strength) if strength is not None else None,
        concurrent=bool(row.get("concurrent", True)),
        tag=str(row.get("tag") or row.get("id") or ""),
    )


def parse_director_timeline(raw: Any = None, *, scene: Mapping[str, Any] | None = None) -> DirectorTimeline:
    """Parse ``events`` / ``timeline`` / media attach fields from scene or edit."""
    tl = DirectorTimeline()
    data: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        data.update(dict(raw))
    if isinstance(scene, Mapping):
        for k in (
            "events",
            "timeline",
            "references",
            "refs",
            "multimodal_refs",
            "mode",
            "init_image",
            "style_image",
            "style_ref",
            "motion_video",
            "motion_clip",
            "anchor_image",
        ):
            if k in scene and k not in data:
                data[k] = scene[k]
        if "events" not in data and isinstance(scene.get("frontier"), Mapping):
            fe = scene["frontier"].get("events") or scene["frontier"].get("timeline")
            if fe:
                data["events"] = fe

    events_raw = data.get("events") or data.get("timeline") or []
    if isinstance(events_raw, Mapping):
        converted: list[Any] = []
        for k, v in events_raw.items():
            if isinstance(v, str):
                converted.append({"at_sec": k, "prompt": v})
            elif isinstance(v, Mapping):
                converted.append({**dict(v), "at_sec": v.get("at_sec", k)})
        events_raw = converted
    if isinstance(events_raw, Sequence) and not isinstance(events_raw, (str, bytes)):
        for row in events_raw:
            if isinstance(row, str):
                s = row.strip()
                at = 0.0
                body = s
                if ":" in s:
                    head, rest = s.split(":", 1)
                    head = head.strip().lower().rstrip("s")
                    try:
                        at = float(head)
                        body = rest.strip()
                    except ValueError:
                        pass
                ev = _parse_event({"at_sec": at, "prompt": body})
            elif isinstance(row, Mapping):
                ev = _parse_event(row)
            else:
                ev = None
            if ev:
                tl.events.append(ev)

    tl.events.sort(key=lambda e: (e.at_sec, e.tag))

    refs_raw = data.get("references") or data.get("refs") or data.get("multimodal_refs")
    if refs_raw is not None:
        try:
            pack = parse_multimodal_refs(refs_raw)
            for r in pack.refs:
                tl.refs.append(
                    {
                        "path": r.path,
                        "role": r.role,
                        "modality": r.modality,
                        "strength": r.strength,
                        "tag": r.tag,
                        "notes": r.notes,
                    }
                )
        except Exception:
            if isinstance(refs_raw, Sequence):
                for x in refs_raw:
                    if isinstance(x, str):
                        tl.refs.append({"path": x, "role": "identity"})
                    elif isinstance(x, Mapping) and (x.get("path") or x.get("file")):
                        tl.refs.append(dict(x))

    tl.mode = str(data.get("mode") or "").strip().lower()
    tl.init_image = str(data.get("init_image") or data.get("anchor_image") or data.get("i2i_image") or "").strip()
    tl.style_image = str(data.get("style_image") or data.get("style_ref") or "").strip()
    tl.motion_video = str(data.get("motion_video") or data.get("motion_clip") or data.get("v2v_clip") or "").strip()

    for r in tl.refs:
        path = str(r.get("path") or "")
        role = str(r.get("role") or "")
        mod = str(r.get("modality") or "")
        if not tl.style_image and role == "style" and path:
            tl.style_image = path
        if not tl.motion_video and role == "motion" and path:
            tl.motion_video = path
        if not tl.init_image and role in ("identity", "scene") and mod == "image" and path:
            tl.init_image = path

    if tl.events:
        tl.notes.append(f"events={len(tl.events)}")
    if tl.refs:
        tl.notes.append(f"refs={len(tl.refs)}")
    return tl


def events_active_at(
    events: Sequence[TimedEvent],
    t_sec: float,
    *,
    shot_id: str = "",
) -> list[TimedEvent]:
    """Return events active at time ``t_sec`` (global + matching shot)."""
    active: list[TimedEvent] = []
    for e in events:
        if e.shot_id and shot_id and e.shot_id != shot_id:
            continue
        if e.shot_id and not shot_id:
            continue
        if float(e.at_sec) - 1e-6 <= float(t_sec) <= float(e.end_sec) + 1e-6:
            active.append(e)
    if not active:
        return []
    if all(e.concurrent for e in active):
        return active
    out: list[TimedEvent] = []
    last_exclusive: TimedEvent | None = None
    for e in active:
        if e.concurrent:
            out.append(e)
        else:
            last_exclusive = e
    if last_exclusive is not None:
        out.append(last_exclusive)
    return out


def prompt_for_keyframe(
    base_prompt: str,
    *,
    t_sec: float,
    events: Sequence[TimedEvent],
    shot_id: str = "",
    base_negative: str = "",
) -> tuple[str, str, float | None]:
    """Merge active timed events into keyframe prompt/negative; optional strength."""
    active = events_active_at(events, t_sec, shot_id=shot_id)
    if not active:
        return base_prompt, base_negative, None
    bits: list[str] = [base_prompt] if base_prompt else []
    neg_bits: list[str] = [base_negative] if base_negative else []
    strength: float | None = None
    for e in active:
        frag_parts = [x for x in (e.action, e.effect, e.camera, e.prompt) if x]
        frag = ", ".join(frag_parts)
        if frag:
            bits.append(frag)
        if e.negative:
            neg_bits.append(e.negative)
        if e.strength is not None:
            strength = float(e.strength)
    prompt = ", ".join(b for b in bits if b).strip(", ")
    negative = ", ".join(b for b in neg_bits if b).strip(", ")
    return prompt, negative, strength


def parse_cli_refs(raw_refs: Sequence[str] | None) -> list[dict[str, Any]]:
    """Parse ``--ref path`` or ``--ref path:role`` or ``--ref path:role:strength``."""
    out: list[dict[str, Any]] = []
    for raw in raw_refs or ():
        s = str(raw or "").strip()
        if not s:
            continue
        parts = s.split(":")
        path = s
        role = "identity"
        strength = 0.85
        # Windows drive: D:\a.png:style:0.9
        if len(parts) >= 3 and len(parts[0]) == 1 and parts[0].isalpha():
            path = f"{parts[0]}:{parts[1]}"
            role = parts[2]
            if len(parts) > 3:
                try:
                    strength = float(parts[3])
                except ValueError:
                    pass
        elif len(parts) >= 2 and not (len(parts[0]) == 1 and parts[0].isalpha()):
            path = parts[0]
            role = parts[1]
            if len(parts) > 2:
                try:
                    strength = float(parts[2])
                except ValueError:
                    pass
        out.append({"path": path, "role": role, "strength": strength})
    return out


def merge_refs_into_edit(edit: dict[str, Any], refs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold CLI/scene refs into ``edit.multimodal_refs``."""
    out = dict(edit or {})
    existing = out.get("multimodal_refs") or out.get("references") or out.get("refs") or []
    items: list[Any] = list(existing) if isinstance(existing, list) else []
    for r in refs:
        items.append(dict(r))
    if items:
        out["multimodal_refs"] = items
        out["leaders_stack"] = True
    return out


def media_mode_overrides(mode: str) -> dict[str, Any]:
    """ProcessOptions / assignment overrides for t2v|i2v|v2v|i2i|cgi."""
    m = (mode or "").strip().lower().replace("-", "_")
    if m in ("v2v", "video2video", "vid2vid"):
        return {
            "mode": "v2v",
            "motion_transfer": True,
            "motion_transfer_retrieved": True,
            "edit_strength": 0.48,
            "use_motion_only": False,
            "depth_interpolate": True,
        }
    if m in ("i2i", "img2img", "image2image"):
        return {
            "mode": "i2i",
            "edit_strength": 0.62,
            "keyframe_interval": 4,
            "motion_transfer": False,
            "use_motion_only": False,
        }
    if m in ("cgi", "3d_cgi", "cgi_3d"):
        return {
            "mode": "cgi",
            "motion_grammar": "cgi",
            "edit_strength": 0.50,
            "depth_interpolate": True,
        }
    if m == "i2v":
        return {"mode": "i2v", "edit_strength": 0.55}
    return {"mode": m or "t2v"}
