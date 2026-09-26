"""Photoshop-like edit skills + Ideogram-style text/layout plan for stills.

Turns user intents (fix face, change outfit, add text, outpaint) into
sample.py / edit_inpaint argv without a full layer editor UI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "EditSkill",
    "EditSkillPlan",
    "plan_edit_skills",
    "plan_ideogram_layout",
    "IdeogramPlan",
]

SkillKind = Literal[
    "inpaint_region",
    "img2img",
    "outpaint",
    "glyph_text",
    "palette_lock",
    "identity_lock",
    "corpus_ref",
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
    outpaint: bool = False,
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
            plan.sample_argv = [
                "--init-image",
                init_image,
                "--mask",
                s0.mask_path,
                "--prompt",
                s0.prompt,
                "--strength",
                str(s0.strength),
                "--inpaint-mode",
                "mdm",
            ]
            plan.notes.append(f"primary inpaint region={primary}")

    if outpaint and init_image:
        plan.skills.append(EditSkill(kind="outpaint", init_image=init_image, prompt=prompt, strength=0.55))
        plan.notes.append("outpaint: use compose_outpaint_canvas + img2img (see latent_edit_helpers)")

    if glyph_text:
        plan.skills.append(EditSkill(kind="glyph_text", prompt=glyph_text, extra={"text": glyph_text}))
        plan.sample_argv.extend(["--text-in-image", glyph_text, "--glyph-canvas"])
        plan.notes.append("ideogram-style glyph/text flags appended")

    if identity_ref:
        plan.skills.append(EditSkill(kind="identity_lock", init_image=identity_ref))
        plan.sample_argv.extend(["--reference-image", identity_ref, "--reference-style-mode", "instantstyle"])

    (work / "edit_plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return plan


def plan_ideogram_layout(
    prompt: str,
    *,
    texts: list[str] | None = None,
    palette_hex: list[str] | None = None,
    layout: str = "poster",  # poster | social | logo
) -> IdeogramPlan:
    """Compile Ideogram-like text + layout argv from prompt + explicit strings."""
    from utils.generation.glyph_canvas import extract_glyph_strings

    glyphs = list(texts or [])
    if not glyphs:
        try:
            from utils.generation.glyph_canvas import extract_glyph_strings

            glyphs = list(extract_glyph_strings(prompt) or [])
        except Exception:
            import re

            glyphs = re.findall(r'"([^"]+)"', prompt) or re.findall(r"'([^']+)'", prompt)

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

    argv = ["--prompt", prompt, "--text-in-image", "1"]
    if glyphs:
        argv.extend(["--glyph-canvas"])
    if palette_hex:
        argv.extend(["--palette-lock", ",".join(palette_hex)])

    # Persist brief for --design-brief consumers
    return IdeogramPlan(
        prompt=prompt,
        glyph_texts=glyphs,
        design_brief=brief,
        box_layout=box,
        palette_hex=list(palette_hex or []),
        sample_argv=argv,
        notes=[f"{len(glyphs)} glyph strings", f"layout={layout}"],
    )
