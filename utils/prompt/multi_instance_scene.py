"""
Prompting and scoring hints for **multiple distinct visual instances** in one image.

Why this is hard for diffusion
------------------------------
One forward pass has **no latent memory** between spatial regions: the U-Net/DiT does
not "draw poster A, commit it, then draw poster B". Repetition, cloning, blended
faces, and duplicated textures are common failure modes.

When you need *true* consistency across views, prefer **reference conditioning**,
**img2img/inpaint**, or **compositing**.

Companion flags for ``sample.py``: :func:`multi_instance_auto_settings` with
``--multi-instance-auto``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import IO, Any, Dict, Literal, Optional, Tuple

MultiInstancePreset = Literal[
    "none",
    "distinct_objects",
    "stacked_media",
    "turnaround_sheet",
    "panel_strip",
    "group_portrait",
]


@dataclass(frozen=True, slots=True)
class MultiInstanceAugment:
    """Returned by :func:`apply_multi_instance_preset`."""

    prompt_suffix: str
    negative_suffix: str
    min_candidates: int
    suggested_expected_count: Optional[int]
    workflow_note: str


_PRESETS: dict[str, MultiInstanceAugment] = {
    "distinct_objects": MultiInstanceAugment(
        prompt_suffix=(
            "multiple separate framed or clearly spaced items, each with its own distinct "
            "graphic design, palette, and subject; readable separation between instances; "
            "no symmetry-driven duplication; frontal or mild perspective on each piece"
        ),
        negative_suffix=(
            "cloned identical posters, tiled repetition, mirrored duplicate artwork, "
            "merged faces across frames, copy-paste motifs, kaleidoscope symmetry, "
            "single repeating texture, melted together designs"
        ),
        min_candidates=8,
        suggested_expected_count=None,
        workflow_note=(
            "Use --pick-best combo_count + --multi-instance-count; --grid to compare; "
            "dissect/inpaint for per-poster control."
        ),
    ),
    "stacked_media": MultiInstanceAugment(
        prompt_suffix=(
            "orderly stack of volumes; each visible cover and spine belongs to a clearly "
            "different book with its own color blocking and title zone; believable "
            "foreshortening; page edges aligned; no smeared single rectangle of noise"
        ),
        negative_suffix=(
            "single illegible collage, fused covers, duplicated spine gibberish, melted "
            "stack, one blurry book repeated, mirrored spines, text soup"
        ),
        min_candidates=6,
        suggested_expected_count=None,
        workflow_note=(
            "Name titles/colors per book; combo_count + text constraints; inpaint spine rows."
        ),
    ),
    "turnaround_sheet": MultiInstanceAugment(
        prompt_suffix=(
            "character turnaround model sheet layout: labeled orthographic rows or columns "
            "for front view, three-quarter view, side view, back view; identical outfit "
            "and proportions in every panel; neutral studio lighting; flat neutral background"
        ),
        negative_suffix=(
            "different outfit per view, hairstyle changing between panels, mismatched shoes, "
            "asymmetric props swapping sides, mirrored clone artifacts, cropped missing views"
        ),
        min_candidates=8,
        suggested_expected_count=4,
        workflow_note=(
            "Prefer --reference-image + strength; otherwise render panels separately and composite."
        ),
    ),
    "panel_strip": MultiInstanceAugment(
        prompt_suffix=(
            "horizontal storyboard strip of clearly separated sequential panels "
            "(visible gutters or bezels); each panel shows a visually distinct beat; "
            "consistent line weight and lettering style across panels"
        ),
        negative_suffix=(
            "panels melting together, duplicated frame content, "
            "same screenshot repeated, ambiguous panel boundaries"
        ),
        min_candidates=8,
        suggested_expected_count=None,
        workflow_note=(
            "Set --multi-instance-count = panel count; --composition-brief auto for legibility."
        ),
    ),
    "group_portrait": MultiInstanceAugment(
        prompt_suffix=(
            "group shot with individually readable faces spaced apart; "
            "each person has distinct clothing, hairstyle, and body silhouette; "
            "no fused faces or shared jaws; proportional heads"
        ),
        negative_suffix=(
            "conjoined faces, fused heads, mirrored twins error, "
            "duplicate same person multiplied, anatomically merged limbs, "
            "extra limbs shared between subjects"
        ),
        min_candidates=8,
        suggested_expected_count=None,
        workflow_note=(
            "Set --multi-instance-count = people; widen --width; combo_count aligns with head count."
        ),
    ),
}


def describe_limitation_short() -> str:
    """One-line rationale for UX / stderr."""
    return (
        "Single-pass diffusion cannot 'remember' prior regions; multi-instance presets add "
        "structure, negatives, and candidate pressure — not true scene memory."
    )


def apply_multi_instance_preset(
    prompt: str,
    preset: str,
    *,
    user_expected_count: int = 0,
) -> Tuple[str, str, int, Optional[int], str]:
    """Returns ``( augmented_prompt, negative_fragment, min_candidates, expected_count_hint, workflow_note )``."""
    prompt = (prompt or "").strip()
    key = (preset or "none").lower().strip()
    if key in ("none", "") or not prompt:
        return prompt, "", 1, None, ""

    aug = _PRESETS.get(key)
    if aug is None:
        return prompt, "", 1, None, ""

    suf = aug.prompt_suffix
    fp = suf[:80].lower()
    if fp not in (prompt or "").lower():
        out_prompt = f"{prompt}, {suf}".strip().strip(",")
    else:
        out_prompt = prompt

    ec: Optional[int] = user_expected_count if user_expected_count > 0 else aug.suggested_expected_count
    neg = aug.negative_suffix

    return out_prompt, neg, aug.min_candidates, ec, aug.workflow_note


def multi_instance_auto_settings(preset_key: str) -> Dict[str, Any]:
    """
    Companion defaults for ``--multi-instance-auto`` in ``sample.py``.

    Keys: ``expected_count_target`` (``people`` | ``objects``), ``pick_best``, ``composition_brief_boost``.
    """
    pk = (preset_key or "").lower().strip()
    people_presets = {"turnaround_sheet", "group_portrait"}
    return {
        "expected_count_target": ("people" if pk in people_presets else "objects"),
        "pick_best": "combo_count",
        "composition_brief_boost": True,
    }


def print_multi_instance_hints(preset_key: str, *, stream: IO[str] | None = None) -> None:
    """stderr checklist for tougher multi-instance generations."""
    out = stream or sys.stderr
    aug = _PRESETS.get((preset_key or "").lower().strip())
    aug_line = aug.workflow_note if aug else ""
    print(
        "\nMulti-instance checklist:\n"
        "  • Reference lock: --reference-image + strength for repeated identity.\n"
        "  • Regions: --dissect-refs + --auto-init-from-dissection → composite init/mask.\n"
        "  • Search: high --num, --pick-best combo_count + --multi-instance-count / --expected-count.\n"
        "  • Per-entity pose/physics: --detailed-scene-boost auto (or on) + --detailed-scene-strength.\n"
        "  • Chain: refine pass, img2img on crops, or external compositing.\n"
        + (f"  • Preset: {aug_line}\n" if aug_line else ""),
        file=out,
    )


__all__ = [
    "MultiInstanceAugment",
    "MultiInstancePreset",
    "apply_multi_instance_preset",
    "describe_limitation_short",
    "multi_instance_auto_settings",
    "print_multi_instance_hints",
]
