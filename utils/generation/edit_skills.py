"""Photoshop-like edit skills + Ideogram-style text/layout plan for stills.

Turns user intents (fix face, change outfit, add text, identity reference) into
sample.py / edit_inpaint argv without a full layer editor UI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from utils.generation.sample_cli_passthrough import sample_argv

__all__ = [
    "EditSkill",
    "EditSkillPlan",
    "plan_edit_skills",
    "plan_ideogram_layout",
    "IdeogramPlan",
]

SkillKind = Literal[
    "inpaint_region",
    "glyph_text",
    "identity_lock",
]


@dataclass
class EditSkill:
    kind: SkillKind
    prompt: str = ""
    region: str = ""  # face|hands|clothing|background|subject|full
    strength: float = 0.45
    mask_path: str = ""
    init_image: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EditSkillPlan:
    skills: list[EditSkill] = field(default_factory=list)
    sample_argv: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skills": [s.to_dict() for s in self.skills],
            "sample_argv": list(self.sample_argv),
            "notes": list(self.notes),
        }


@dataclass
class IdeogramPlan:
    """Text-in-image + layout compile (Ideogram-like)."""

    prompt: str
    glyph_texts: list[str] = field(default_factory=list)
    design_brief: dict[str, Any] = field(default_factory=dict)
    box_layout: dict[str, Any] | None = None
    palette_hex: list[str] = field(default_factory=list)
    sample_argv: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_edit_skills(
    *,
    init_image: str,
    prompt: str = "",
    fix_regions: list[str] | None = None,
    change_outfit: str = "",
    glyph_text: str = "",
    identity_ref: str = "",
    work_dir: str | Path = "",
    width: int = 1024,
    height: int = 1024,
) -> EditSkillPlan:
    """Build a sequenced edit plan (masks + argv hints)."""
    work = Path(work_dir or "outputs/edit_skills")
    work.mkdir(parents=True, exist_ok=True)
    plan = EditSkillPlan()
    argv_values: dict[str, Any] = {}
    regions = list(fix_regions or [])
    if change_outfit:
        regions.append("clothing")
        prompt = f"{prompt}, {change_outfit}".strip(", ")

    if regions and init_image:
        try:
            from utils.generation.edit_masks import save_heuristic_mask
        except Exception as exc:
            plan.notes.append(f"masks unavailable: {exc}")
            save_heuristic_mask = None  # type: ignore

        primary = regions[0] if len(regions) == 1 else "subject"
        for reg in regions:
            mask_path = work / f"mask_{reg}.png"
            if save_heuristic_mask is not None:
                try:
                    save_heuristic_mask(mask_path, width=width, height=height, region=reg)
                except Exception as exc:
                    plan.notes.append(f"mask {reg}: {exc}")
                    continue
            skill = EditSkill(
                kind="inpaint_region",
                prompt=prompt or f"fix {reg}, high quality, correct anatomy",
                region=reg,
                mask_path=str(mask_path),
                init_image=init_image,
            )
            plan.skills.append(skill)
        if plan.skills:
            s0 = plan.skills[0]
            argv_values.update(
                init_image=init_image, mask=s0.mask_path, prompt=s0.prompt, strength=s0.strength, inpaint_mode="mdm"
            )
            plan.notes.append(f"primary inpaint region={primary}")

    if glyph_text:
        plan.skills.append(EditSkill(kind="glyph_text", prompt=glyph_text, extra={"text": glyph_text}))
        argv_values.update(text_in_image=True, expected_text=glyph_text)

    if identity_ref:
        plan.skills.append(EditSkill(kind="identity_lock", init_image=identity_ref))
        argv_values["reference_image"] = identity_ref

    plan.sample_argv = sample_argv(argv_values)
    (work / "edit_plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return plan


def plan_ideogram_layout(
    prompt: str,
    *,
    texts: list[str] | None = None,
    palette_hex: list[str] | None = None,
    layout: str = "poster",  # poster | social | logo
) -> IdeogramPlan:
    """Compile Ideogram-like text + layout argv from prompt + explicit strings (the box layout is the caller's to save)."""
    from utils.generation.text_rendering import TextRenderingEngine

    found = TextRenderingEngine().extract_text_requirements(prompt or "")["text_content"]
    glyphs = list(texts or []) or [s.strip() for s in found if s.strip()]

    brief: dict[str, Any] = {
        "intent": layout,
        "prompt": prompt,
        "text_elements": [{"text": t, "role": "headline" if i == 0 else "body"} for i, t in enumerate(glyphs)],
    }
    # Simple vertical stack boxes for text regions
    regions = []
    for i, t in enumerate(glyphs[:4]):
        y0 = 0.08 + i * 0.18
        regions.append({"id": f"text_{i}", "prompt": f'text "{t}"', "box": [0.1, y0, 0.9, y0 + 0.14]})
    box = {"mode": "typography", "regions": regions, "global_prompt": prompt} if regions else None

    argv = sample_argv({"prompt": prompt, "text_in_image": True, "expected_text": ",".join(glyphs)})
    return IdeogramPlan(
        prompt=prompt,
        glyph_texts=glyphs,
        design_brief=brief,
        box_layout=box,
        palette_hex=list(palette_hex or []),
        sample_argv=argv,
        notes=[f"{len(glyphs)} glyph strings", f"layout={layout}"],
    )
