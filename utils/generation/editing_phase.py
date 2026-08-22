"""
Post-generation **editing phase** — the closed loop from the SDX quality flowchart.

Flow::

    image (+ prompt)
        → diagnose (CLIP/gates, missing prompt tokens, text/OCR, anatomy cues)
        → break into pieces (prompt entities → region masks)
        → plan edits (OCR / inpaint piece / img2img / RAG delta / art post)
        → apply (sample.py edit runner when ckpt set)
        → re-gate → loop until cohesive / natural or max_iters

Composes existing helpers; does not reimplement OCR/SAM/CLIP.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

_log = logging.getLogger(__name__)

ActionKind = Literal[
    "ocr_fix",
    "inpaint_region",
    "img2img",
    "rag_prompt_delta",
    "art_post",
    "prompt_realign",
]

_STOP = frozenset(
    """
    a an the and or of to in on at for with from by as is are was were be been being
    this that these those it its into over under near very more most less least
    image photo picture scene shot view style high quality detailed masterpiece
    """.split()
)

_ANATOMY_CUES = (
    "hand",
    "hands",
    "finger",
    "fingers",
    "face",
    "eyes",
    "arm",
    "legs",
    "anatomy",
    "portrait",
    "person",
    "woman",
    "man",
    "girl",
    "boy",
)


@dataclass(slots=True)
class EditAction:
    """One concrete edit to apply in the phase."""

    kind: ActionKind
    reason: str
    region: str | None = None  # face | hands | clothing | background | subject | full
    prompt_delta: str = ""
    negative_delta: str = ""
    strength: float = 0.55
    priority: int = 50  # lower runs first


@dataclass(slots=True)
class Diagnosis:
    """Signals from the verifier pass."""

    clip_score: float = 0.0
    sharpness: float = 0.0
    exposure: float = 0.0
    aesthetic: float = 0.0
    gate_passed: bool = False
    gate_failures: list[str] = field(default_factory=list)
    missing_tokens: list[str] = field(default_factory=list)
    expected_text: list[str] = field(default_factory=list)
    needs_ocr: bool = False
    needs_anatomy: bool = False
    piece_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EditingPhaseConfig:
    """Knobs for the editing-phase loop."""

    max_iters: int = 3
    min_clip: float = 0.28
    min_sharpness: float = 0.15
    min_exposure: float = 0.25
    min_aesthetic: float = 0.0
    enable_ocr: bool = True
    enable_pieces: bool = True
    enable_rag: bool = True
    enable_art_post: bool = True
    enable_anatomy: bool = True
    img2img_strength: float = 0.45
    inpaint_strength: float = 0.62
    refine_steps: int = 20
    scheduler: str = "ays_dit"
    solver: str = "dpmpp_2m"
    dry_run: bool = False  # plan only; never call sample.py
    device: str = "cuda"


@dataclass(slots=True)
class EditingPhaseResult:
    """Outcome of ``run_editing_phase``."""

    image_rgb: np.ndarray
    prompt: str
    negative_prompt: str
    diagnosis_history: list[Diagnosis] = field(default_factory=list)
    actions_applied: list[EditAction] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = ""
    output_path: str | None = None
    piece_dir: str | None = None


# ---------------------------------------------------------------------------
# Prompt / piece helpers
# ---------------------------------------------------------------------------


def extract_prompt_tokens(prompt: str, *, max_tokens: int = 24) -> list[str]:
    """Content-ish tokens from the prompt (for coverage / piece labels)."""
    raw = re.findall(r"[A-Za-z][A-Za-z0-9'\\-]{2,}", prompt.lower())
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        if t in _STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_tokens:
            break
    return out


def expected_render_text(prompt: str) -> list[str]:
    """Quoted / [text:] strings that should be readable in the image."""
    try:
        from utils.generation.text_rendering import TextRenderingEngine

        info = TextRenderingEngine().extract_text_requirements(prompt)
        return list(info.get("text_content") or [])
    except Exception:
        quoted = re.findall(r'"([^"]+)"', prompt)
        bracketed = re.findall(r"\[text:\s*([^\]]*)\]", prompt, flags=re.I)
        return [*(quoted or []), *(bracketed or [])]


def infer_piece_labels(prompt: str, *, max_pieces: int = 6) -> list[str]:
    """
    REVE-style piece list from the prompt.

    Maps common subject words onto heuristic inpaint regions; leftover nouns
    become ``subject`` pieces for documentation / future SAM grounding.
    """
    tokens = extract_prompt_tokens(prompt, max_tokens=32)
    labels: list[str] = []
    region_hits = {
        "face": ("face", "portrait", "head", "eyes", "smile"),
        "hands": ("hand", "hands", "finger", "fingers", "holding"),
        "clothing": ("dress", "shirt", "jacket", "coat", "outfit", "clothing", "armor"),
        "background": ("background", "sky", "landscape", "city", "forest", "room", "street"),
    }
    for region, keys in region_hits.items():
        if any(k in tokens or k in prompt.lower() for k in keys):
            labels.append(region)
    # Always ensure at least subject + background for decomposition UX
    if "subject" not in labels:
        labels.insert(0, "subject")
    if "background" not in labels and len(labels) < max_pieces:
        labels.append("background")
    return labels[:max_pieces]


def missing_tokens_heuristic(prompt: str, caption: str) -> list[str]:
    """Tokens in the prompt that do not appear in an image caption / description."""
    want = extract_prompt_tokens(prompt)
    cap = (caption or "").lower()
    return [t for t in want if t not in cap and not any(t in w for w in cap.split())][:12]


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------


def diagnose_image(
    image_rgb: np.ndarray,
    prompt: str,
    *,
    cfg: EditingPhaseConfig,
    caption: str = "",
) -> Diagnosis:
    """Verifier pass: gates + prompt gaps + text/anatomy cues."""
    from utils.superior.quality_gates import GateThresholds, QualityGateRunner

    d = Diagnosis()
    d.piece_labels = infer_piece_labels(prompt) if cfg.enable_pieces else []
    d.expected_text = expected_render_text(prompt)
    d.needs_ocr = bool(cfg.enable_ocr and d.expected_text)
    pl = prompt.lower()
    d.needs_anatomy = bool(cfg.enable_anatomy and any(c in pl for c in _ANATOMY_CUES))

    gates = QualityGateRunner(
        GateThresholds(
            min_sharpness=cfg.min_sharpness,
            min_exposure=cfg.min_exposure,
            min_clip=cfg.min_clip,
            min_aesthetic=cfg.min_aesthetic,
        )
    )
    try:
        gr = gates.evaluate(image_rgb, prompt=prompt, device=cfg.device if cfg.device != "cuda" else "cpu")
        d.gate_passed = bool(gr.passed)
        d.gate_failures = list(gr.failures)
        d.clip_score = float(gr.scores.get("clip", 0.0) or 0.0)
        d.sharpness = float(gr.scores.get("sharpness", 0.0) or 0.0)
        d.exposure = float(gr.scores.get("exposure", 0.0) or 0.0)
        d.aesthetic = float(gr.scores.get("aesthetic", 0.0) or 0.0)
    except Exception as e:
        d.notes.append(f"gate_error:{e}")
        d.gate_passed = False

    if caption.strip():
        d.missing_tokens = missing_tokens_heuristic(prompt, caption)
    elif not d.gate_passed and "clip" in d.gate_failures:
        # Without a caption, treat low CLIP as “global missing adherence”
        d.missing_tokens = extract_prompt_tokens(prompt, max_tokens=8)
        d.notes.append("low_clip_without_caption")

    return d


def maybe_caption_image(image_rgb: np.ndarray, *, device: str = "cpu") -> str:
    """Optional light VLM caption (moondream) for prompt-gap detection."""
    try:
        from pathlib import Path

        from PIL import Image

        from utils.modeling.model_paths import default_moondream_path

        path = default_moondream_path()
        local = Path(path)
        if not local.is_dir():
            return ""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(local), trust_remote_code=True)
        _ = tok  # tokenizer required for some moondream builds
        model = AutoModelForCausalLM.from_pretrained(str(local), trust_remote_code=True, low_cpu_mem_usage=True)
        import torch

        dev = device if device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(dev)
        model.eval()
        pil = Image.fromarray(image_rgb.astype(np.uint8))
        if hasattr(model, "encode_image") and hasattr(model, "answer_question"):
            enc = model.encode_image(pil)
            return str(model.answer_question(enc, "Describe the image in detail.") or "")
        return ""
    except Exception as e:
        _log.debug("caption skipped: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def plan_edits(diagnosis: Diagnosis, prompt: str, *, cfg: EditingPhaseConfig) -> list[EditAction]:
    """Turn diagnosis into a prioritized action list."""
    actions: list[EditAction] = []

    if diagnosis.needs_ocr and diagnosis.expected_text:
        actions.append(
            EditAction(
                kind="ocr_fix",
                reason=f"expected readable text: {diagnosis.expected_text[:3]}",
                region="subject",
                prompt_delta=", crisp readable typography, legible letters, correct spelling",
                negative_delta="gibberish text, misspelled words, warped letters",
                strength=cfg.inpaint_strength,
                priority=10,
            )
        )

    if diagnosis.needs_anatomy:
        region = "hands" if any(t in prompt.lower() for t in ("hand", "hands", "finger")) else "face"
        actions.append(
            EditAction(
                kind="inpaint_region",
                reason="anatomy / proportion risk from prompt cues",
                region=region,
                prompt_delta=", correct anatomy, natural proportions, fixed perspective",
                negative_delta="deformed hands, extra fingers, bad anatomy, warped face",
                strength=cfg.inpaint_strength,
                priority=20,
            )
        )

    for tok in diagnosis.missing_tokens[:4]:
        region = "background" if tok in ("sky", "forest", "city", "room", "street", "background") else "subject"
        actions.append(
            EditAction(
                kind="inpaint_region",
                reason=f"missing prompt token: {tok}",
                region=region,
                prompt_delta=f", clearly visible {tok}",
                strength=cfg.inpaint_strength,
                priority=30,
            )
        )

    if diagnosis.missing_tokens and cfg.enable_rag:
        join = ", ".join(diagnosis.missing_tokens[:6])
        actions.append(
            EditAction(
                kind="rag_prompt_delta",
                reason="retrieve / invent detail for missing entities",
                prompt_delta=f", rich authentic detail for: {join}",
                priority=35,
            )
        )

    if (not diagnosis.gate_passed) and ("clip" in diagnosis.gate_failures or diagnosis.clip_score < cfg.min_clip):
        actions.append(
            EditAction(
                kind="prompt_realign",
                reason="low prompt alignment",
                region="full",
                prompt_delta=", matches the prompt faithfully, coherent scene",
                strength=cfg.img2img_strength,
                priority=40,
            )
        )

    if not diagnosis.gate_passed and ("sharpness" in diagnosis.gate_failures or "exposure" in diagnosis.gate_failures):
        actions.append(
            EditAction(
                kind="img2img",
                reason="global quality (sharpness/exposure)",
                strength=min(0.35, cfg.img2img_strength),
                prompt_delta=", clean natural photo, cohesive lighting",
                negative_delta="blurry, overexposed, underexposed, artificial, plastic skin",
                priority=45,
            )
        )

    if cfg.enable_art_post and (
        not diagnosis.gate_passed or "aesthetic" in diagnosis.gate_failures or diagnosis.needs_anatomy
    ):
        actions.append(
            EditAction(
                kind="art_post",
                reason="composition / value / naturalize pass",
                priority=90,
            )
        )

    # Deduplicate by (kind, region, prompt_delta)
    seen: set[tuple] = set()
    uniq: list[EditAction] = []
    for a in sorted(actions, key=lambda x: x.priority):
        key = (a.kind, a.region, a.prompt_delta)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    return uniq


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _merge_prompt(base: str, delta: str) -> str:
    d = (delta or "").strip().strip(",")
    if not d:
        return base
    if d.lower() in base.lower():
        return base
    return f"{base.rstrip().rstrip(',')}, {d}"


def break_into_piece_masks(
    width: int,
    height: int,
    labels: list[str],
    out_dir: Path,
) -> dict[str, Path]:
    """Write heuristic piece masks (REVE-style separation scaffolding)."""
    from utils.generation.edit_masks import save_heuristic_mask

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for lab in labels:
        p = out_dir / f"piece_{lab}.png"
        save_heuristic_mask(p, width, height, lab)
        paths[lab] = p
    return paths


def apply_art_post(image_rgb: np.ndarray) -> np.ndarray:
    """Lightweight artistic naturalize pass."""
    try:
        from utils.quality.artistic_post_process import ArtisticPostConfig, apply_artistic_pipeline

        cfg = ArtisticPostConfig(
            composition_mode="rule_of_thirds",
            composition_strength=0.15,
            value_structure=True,
            value_midtone_contrast=0.08,
            asymmetry_strength=0.05,
            lost_found_strength=0.1,
        )
        return apply_artistic_pipeline(image_rgb, cfg)
    except Exception as e:
        _log.warning("art_post skipped: %s", e)
        return image_rgb


def apply_action(
    action: EditAction,
    *,
    image_rgb: np.ndarray,
    prompt: str,
    negative_prompt: str,
    ckpt: str | None,
    cfg: EditingPhaseConfig,
    work_dir: Path,
    piece_masks: dict[str, Path],
    seed: int | None = None,
) -> tuple[np.ndarray, str, str]:
    """
    Apply one action. Returns ``(image, prompt, negative)``.

    When ``cfg.dry_run`` or ``ckpt`` is missing, returns image unchanged (plan-only).
    """
    prompt2 = _merge_prompt(prompt, action.prompt_delta)
    neg2 = _merge_prompt(negative_prompt, action.negative_delta)

    if action.kind == "rag_prompt_delta":
        # Prompt-only enrichment; retrieval corpora are optional.
        return image_rgb, prompt2, neg2

    if action.kind == "art_post":
        if cfg.dry_run:
            return image_rgb, prompt2, neg2
        return apply_art_post(image_rgb), prompt2, neg2

    if cfg.dry_run or not ckpt:
        return image_rgb, prompt2, neg2

    from PIL import Image

    from utils.generation.sample_edit_runner import run_edit_with_pillow

    h, w = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    init = Image.fromarray(image_rgb.astype(np.uint8))
    mask = None
    strength = float(action.strength)
    if action.kind in ("ocr_fix", "inpaint_region") and action.region:
        mp = piece_masks.get(action.region) or piece_masks.get("subject")
        if mp and mp.is_file():
            mask = Image.open(mp).convert("L")
        else:
            from utils.generation.edit_masks import heuristic_inpaint_mask

            mask = heuristic_inpaint_mask(w, h, action.region or "subject")
    elif action.kind in ("img2img", "prompt_realign"):
        mask = None
        strength = float(action.strength)

    out = run_edit_with_pillow(
        ckpt=ckpt,
        prompt=prompt2,
        negative_prompt=neg2,
        base_image=init,
        mask_image=mask,
        width=w,
        height=h,
        steps=int(cfg.refine_steps),
        cfg_scale=7.0,
        seed=seed,
        img2img_strength=strength,
        device=cfg.device,
        scheduler=cfg.scheduler,
        solver=cfg.solver,
    )
    arr = np.asarray(out.convert("RGB"), dtype=np.uint8)
    return arr, prompt2, neg2


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_editing_phase(
    image: np.ndarray | str | Path,
    prompt: str,
    *,
    ckpt: str | None = None,
    negative_prompt: str = "",
    config: EditingPhaseConfig | None = None,
    work_dir: str | Path | None = None,
    seed: int | None = None,
    caption: str | None = None,
) -> EditingPhaseResult:
    """
    Run the post-gen editing loop until gates pass or ``max_iters``.

    Parameters
    ----------
    image:
        RGB uint8 array or path to an image.
    prompt:
        Positive prompt used for generation / realignment.
    ckpt:
        Checkpoint for img2img/inpaint via ``sample.py``. Optional in dry_run.
    """
    cfg = config or EditingPhaseConfig()
    if isinstance(image, (str, Path)):
        from PIL import Image

        image_rgb = np.asarray(Image.open(image).convert("RGB"), dtype=np.uint8)
        src_path = str(image)
    else:
        image_rgb = np.asarray(image, dtype=np.uint8)
        src_path = None

    wd = Path(work_dir) if work_dir else Path("outputs") / "editing_phase"
    wd.mkdir(parents=True, exist_ok=True)
    pieces_dir = wd / "pieces"

    prompt_cur = prompt
    neg_cur = negative_prompt
    hist: list[Diagnosis] = []
    applied: list[EditAction] = []
    cap = caption if caption is not None else ""

    piece_masks = break_into_piece_masks(
        int(image_rgb.shape[1]),
        int(image_rgb.shape[0]),
        infer_piece_labels(prompt_cur),
        pieces_dir,
    )

    stopped = "max_iters"
    for it in range(max(1, int(cfg.max_iters))):
        if not cap and it == 0 and not cfg.dry_run:
            cap = maybe_caption_image(image_rgb, device=cfg.device)

        diag = diagnose_image(image_rgb, prompt_cur, cfg=cfg, caption=cap)
        hist.append(diag)

        if diag.gate_passed and not diag.needs_ocr and not diag.missing_tokens:
            stopped = "gates_passed"
            break

        actions = plan_edits(diag, prompt_cur, cfg=cfg)
        if not actions:
            stopped = "no_actions"
            break

        # One primary action per iteration (plus optional art_post at end of plan)
        primary = [a for a in actions if a.kind != "art_post"]
        trailer = [a for a in actions if a.kind == "art_post"]
        step_actions = (primary[:1] + trailer[:1]) if primary else trailer[:1]

        for act in step_actions:
            image_rgb, prompt_cur, neg_cur = apply_action(
                act,
                image_rgb=image_rgb,
                prompt=prompt_cur,
                negative_prompt=neg_cur,
                ckpt=ckpt,
                cfg=cfg,
                work_dir=wd,
                piece_masks=piece_masks,
                seed=None if seed is None else int(seed) + it,
            )
            applied.append(act)

        # Refresh piece masks if canvas size unchanged (always true here)
        out_iter = wd / f"iter_{it:02d}.png"
        try:
            from PIL import Image

            Image.fromarray(image_rgb).save(out_iter)
        except Exception:
            pass
    else:
        stopped = "max_iters"

    # Final gate note
    if hist and hist[-1].gate_passed and stopped == "max_iters":
        stopped = "gates_passed"

    final_path = wd / "final.png"
    try:
        from PIL import Image

        Image.fromarray(image_rgb).save(final_path)
        out_p = str(final_path)
    except Exception:
        out_p = src_path

    return EditingPhaseResult(
        image_rgb=image_rgb,
        prompt=prompt_cur,
        negative_prompt=neg_cur,
        diagnosis_history=hist,
        actions_applied=applied,
        iterations=len(hist),
        stopped_reason=stopped,
        output_path=out_p,
        piece_dir=str(pieces_dir),
    )


__all__ = [
    "ActionKind",
    "Diagnosis",
    "EditAction",
    "EditingPhaseConfig",
    "EditingPhaseResult",
    "apply_action",
    "break_into_piece_masks",
    "diagnose_image",
    "expected_render_text",
    "extract_prompt_tokens",
    "infer_piece_labels",
    "missing_tokens_heuristic",
    "plan_edits",
    "run_editing_phase",
]
