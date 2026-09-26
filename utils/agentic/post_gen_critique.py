"""Post-gen critique questions → targeted region inpaint / refine plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CritiqueItem",
    "CritiquePlan",
    "build_post_gen_critique",
    "critique_answers_to_fix_plan",
]


@dataclass(slots=True)
class CritiqueItem:
    id: str
    question: str
    region: str  # face | hands | clothing | background | subject | full
    choices: list[str] = field(default_factory=list)
    answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CritiquePlan:
    image_path: str
    prompt: str
    items: list[CritiqueItem] = field(default_factory=list)
    fix_regions: list[str] = field(default_factory=list)
    mask_paths: dict[str, str] = field(default_factory=dict)
    refine_argv: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "prompt": self.prompt,
            "items": [i.to_dict() for i in self.items],
            "fix_regions": list(self.fix_regions),
            "mask_paths": dict(self.mask_paths),
            "refine_argv": list(self.refine_argv),
            "notes": list(self.notes),
        }


_BAD = frozenset(
    {
        "no",
        "n",
        "bad",
        "wrong",
        "fix",
        "redo",
        "broken",
        "ugly",
        "off",
        "fail",
        "0",
        "false",
    }
)


def build_post_gen_critique(image_path: str, prompt: str = "") -> CritiquePlan:
    """Standard critique checklist after a still is generated."""
    items = [
        CritiqueItem(
            id="face_ok",
            question="Is the face / identity correct?",
            region="face",
            choices=["yes", "no — fix face"],
        ),
        CritiqueItem(
            id="hands_ok",
            question="Are the hands / anatomy OK?",
            region="hands",
            choices=["yes", "no — fix hands"],
        ),
        CritiqueItem(
            id="outfit_ok",
            question="Is the outfit / attire right?",
            region="clothing",
            choices=["yes", "no — fix clothing"],
        ),
        CritiqueItem(
            id="bg_ok",
            question="Is the background / place right?",
            region="background",
            choices=["yes", "no — fix background"],
        ),
        CritiqueItem(
            id="overall_ok",
            question="Ship it, or full regenerate?",
            region="full",
            choices=["ship it", "full regenerate", "small refine only"],
        ),
    ]
    return CritiquePlan(image_path=str(image_path), prompt=str(prompt or ""), items=items)


def critique_answers_to_fix_plan(
    plan: CritiquePlan,
    answers: dict[str, str],
    *,
    work_dir: str | Path,
    width: int = 1024,
    height: int = 1024,
    strength: float = 0.45,
) -> CritiquePlan:
    """
    Map yes/no critique answers → heuristic masks + suggested edit_inpaint argv.
    Does not run GPU; writes masks under work_dir/masks/.
    """
    work = Path(work_dir)
    mask_dir = work / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for it in plan.items:
        if it.id in answers:
            it.answer = str(answers[it.id]).strip()

    regions: list[str] = []
    for it in plan.items:
        ans = it.answer.lower()
        if not ans:
            continue
        if it.id == "overall_ok":
            if "regenerate" in ans:
                plan.notes.append("user requested full regenerate")
            continue
        # Treat anything in _BAD or starting with "no" as fix request
        needs_fix = ans in _BAD or ans.startswith("no") or "fix" in ans
        if needs_fix and it.region != "full":
            regions.append(it.region)

    # Dedupe preserve order
    seen: set[str] = set()
    plan.fix_regions = []
    for r in regions:
        if r not in seen:
            seen.add(r)
            plan.fix_regions.append(r)

    if not plan.fix_regions:
        plan.notes.append("no region fixes requested")
        return plan

    try:
        from utils.generation.edit_masks import save_heuristic_mask
    except Exception as exc:
        plan.notes.append(f"mask helper unavailable: {exc}")
        return plan

    # Combined mask: union of regions (white = edit). For simplicity write per-region
    # and a primary mask for the first region; multi-region → subject if >1.
    primary = plan.fix_regions[0] if len(plan.fix_regions) == 1 else "subject"
    if len(plan.fix_regions) > 1:
        plan.notes.append(f"multi-region {plan.fix_regions} → primary mask '{primary}'")

    for region in plan.fix_regions:
        dest = mask_dir / f"mask_{region}.png"
        try:
            save_heuristic_mask(dest, width=width, height=height, region=region)
            plan.mask_paths[region] = str(dest)
        except Exception as exc:
            plan.notes.append(f"mask {region} failed: {exc}")

    mask = plan.mask_paths.get(primary) or next(iter(plan.mask_paths.values()), "")
    if mask and plan.image_path:
        plan.refine_argv = [
            "--init-image",
            plan.image_path,
            "--mask",
            mask,
            "--prompt",
            plan.prompt or "high quality, correct anatomy, detailed",
            "--strength",
            str(float(strength)),
        ]
        plan.notes.append(f"refine via inpaint region={primary} mask={mask}")
    return plan
