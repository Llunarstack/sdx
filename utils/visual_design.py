"""Visual-design prompt packs: UI, architecture, STEM, textbook, brand, infographic, packaging, wayfinding,
product, editorial layout, slides, technical drawings and fashion flats.

Every domain has a positive fragment and a negative addon per intensity tier (lite, standard, strong).
Used by ``sample.py`` (``--visual-design-*``) and the book pipeline.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

Intensity = Literal["lite", "standard", "strong"]
_TIERS: tuple[Intensity, ...] = ("lite", "standard", "strong")

DOMAIN_POSITIVES: dict[str, dict[str, str]] = {
    "ui_ux": {
        "lite": "clean UI hierarchy, intentional spacing scale, WCAG-conscious contrast",
        "standard": (
            "product-grade UI composition, consistent 8-point spacing rhythm, readable type scale, "
            "clear focal component, plausible device frame or canvas, restrained color system, subtle shadows"
        ),
        "strong": (
            "senior product designer level UI: grid discipline, articulate states (default/hover/active), "
            "coherent iconography, plausible information architecture, screenshot-ready polish, "
            "no cluttered chrome, unified corner radius language"
        ),
    },
    "architecture": {
        "lite": "photoreal architectural photograph, straight verticals, natural light",
        "standard": (
            "architectural visualization with coherent structure and materials, plausible scale figures, "
            "physically plausible sun/sky lighting, crisp edges on glass and concrete, no melting geometry"
        ),
        "strong": (
            "professional archviz: camera at believable height, linear perspective, material PBR read, "
            "site context that matches program, restrained post, editorial architectural photography framing"
        ),
    },
    "stem": {
        "lite": "clear STEM diagram with readable labels, vector-like precision where needed",
        "standard": (
            "scientific figure quality: labeled axes/units where applicable, consistent notation, "
            "high contrast ink on background, pedagogical clarity, no pseudo-equation gibberish"
        ),
        "strong": (
            "journal- or textbook-grade STEM illustration: disciplined layout, reproducible geometry, "
            "arrow hierarchy for causality/flow, color used for encoding (not decoration), caption-ready clarity"
        ),
    },
    "textbook": {
        "lite": "textbook illustration, friendly didactic clarity, stable proportions",
        "standard": (
            "educational publisher style: readable at print size, gentle instructional composition, "
            "consistent stylistic metaphor (realistic vs stylized), print-safe margins implied"
        ),
        "strong": (
            "K–12 / college textbook art direction: figure–ground clarity, callouts that match pedagogy, "
            "no busy clutter, diverse inclusive representation when people appear, assessment-ready labeling"
        ),
    },
    "brand": {
        "lite": "logo-ready vector discipline, simple memorable mark, works in monochrome",
        "standard": (
            "brand identity design: balanced lockups, scalable geometry, plausible wordmark + symbol harmony, "
            "print and digital safe zones implied, no illegible micro-details"
        ),
        "strong": (
            "award-show brand craft: distinctive silhouette, restrained palette (or intentional duotone), "
            "grid construction visible in forms, mascot or symbol with clear motion/rest states, merchandise-ready"
        ),
    },
    "infographic": {
        "lite": "clean infographic hierarchy, restrained chart junk",
        "standard": (
            "editorial infographic: numbered story flow, typographic tiers, proportional encodings where charts appear, "
            "icons that share one outline language"
        ),
        "strong": (
            "broadsheet / annual-report infographic density: disciplined grid, data-ink clarity, "
            "parallel legends, no deceptive perspective charts, accessible color pairs"
        ),
    },
    "packaging": {
        "lite": "retail packaging mockup, print registration awareness, plausible SKU panel",
        "standard": (
            "structural packaging design: dieline plausibility, flavor/category legibility from distance, "
            "barcode/ingredients margins implied (no invented compliance text as tiny illegible mush)"
        ),
        "strong": (
            "CPG design systems: die-cut alignment, spot-color discipline, flavor color coding, "
            "sustainable cues when appropriate (icons not legal claims), realistic shelf readability"
        ),
    },
    "wayfinding": {
        "lite": "clear pictogram signage, high contrast symbology",
        "standard": (
            "wayfinding graphics: ISO-inspired pictograms where relevant, glare-resistant contrast, "
            "consistent arrow grammar, accessible type size for viewing distance"
        ),
        "strong": (
            "transit-grade wayfinding system: modular sign family, multilingual hierarchy if labels exist, "
            "no ambiguous arrow junctions, tactile-map logic for floor plans"
        ),
    },
    "general_product": {
        "lite": "industrial design render, coherent CMF, soft studio lighting",
        "standard": (
            "product visualization: believable manufacturing seams, material transitions, "
            "hero angle with ground contact shadow, UI on device screens only when requested"
        ),
        "strong": (
            "designstudio hero shot: controlled HDRI, micro-scratch realism where appropriate, "
            "CMF story (color material finish), hero + detail callout composition"
        ),
    },
    "editorial_layout": {
        "lite": "print editorial layout with clear headline deck and body ladder",
        "standard": (
            "magazine or annual-report spread discipline: modular grid, pull-quote hierarchy, caption bands, "
            "bleed-aware composition, restrained ornament that supports reading order"
        ),
        "strong": (
            "art-director-grade editorial systems: dominant photo + supporting cluster, gutter rhythm, folio logic, "
            "color used for section navigation (not decoration noise), repro-ready sharpening discipline"
        ),
    },
    "presentation_slide": {
        "lite": "clean keynote-style slide layout, readable title hierarchy",
        "standard": (
            "boardroom slide design: headline + subtitle lockup, restrained bullets, coherent icon set, "
            "16:9 safe margins for projectors, one idea per slide"
        ),
        "strong": (
            "enterprise template fidelity: numbered story arc, comparative chart etiquette, presenter notes implied, "
            "no overcrowded KPI walls, cohesive theme tokens across placeholders"
        ),
    },
    "technical_blueprint": {
        "lite": "CAD-readable linework with consistent line weights",
        "standard": (
            "engineering drawing discipline: orthogonal views where appropriate, centerlines and leaders, "
            "callouts that read physically, sane dimension placement (no overlaps)"
        ),
        "strong": (
            "patent-grade technical figure: exploded axon optional, BOM-style leader grammar, sectional hatching logic, "
            "no contradictory hidden lines, reproducible machining story"
        ),
    },
    "fashion_flat": {
        "lite": "fashion technical flat illustration, symmetrical garment silhouette",
        "standard": (
            "apparel industry flat sketch: stitches and seams plausible, trims labeled by region, textile drape read, "
            "colorway chips implied for production handoff"
        ),
        "strong": (
            "tech-pack ready flats: graded size marks implied via proportion guides, stitching weight legend, hardware scale, "
            "no floating buttons, pattern piece alignment hints for CAD handoff"
        ),
    },
}

DOMAIN_NEGATIVES: dict[str, dict[str, str]] = {
    "ui_ux": {
        "lite": "holographic cliché chrome, random gradient soup",
        "standard": "lorem ipsum blocks as design, misaligned baselines, mixed light sources on flat UI",
        "strong": "six different corner radii, unreadable 6px body text, fake macOS window chrome mismatch",
    },
    "architecture": {
        "lite": "tilted verticals, melt architecture",
        "standard": "impossible structural spans, duplicated windows, warped glass grid",
        "strong": "Escher plumbing, random flying buttresses, vegetation clipping through solid walls",
    },
    "stem": {
        "lite": "blurry axis labels",
        "standard": "nonsense notation, dimensionless axes, illegible subscripts",
        "strong": "fake LaTeX, mirrored plots, arrows with no source or sink",
    },
    "textbook": {
        "lite": "busy comic clutter in a diagram",
        "standard": "illegible footnotes, inconsistent scale between insets",
        "strong": "offensive stereotype pedagogy, random clip-art mismatch",
    },
    "brand": {
        "lite": "generic swoosh pile",
        "standard": "illegible wordmark kerning, gradient banding on flat logo",
        "strong": "trademark collision silhouettes, rasterized logo blur",
    },
    "infographic": {
        "lite": "3d pie chart abuse",
        "standard": "truncated axis deception, ornament covering data",
        "strong": "illegible six-point type, rainbow without legend",
    },
    "packaging": {
        "lite": "warped label on bottle",
        "standard": "barcode on curved surface unreadable, illegal nutrition claims as text",
        "strong": "panel text crossing folds, impossible shrink-wrap physics",
    },
    "wayfinding": {
        "lite": "ambiguous pictogram",
        "standard": "conflicting arrow heads, low-contrast on safety background",
        "strong": "EXIT mispointing, tactile dots on screen UI",
    },
    "general_product": {
        "lite": "floating object with no shadow",
        "standard": "mismatched scale hands, plastic that reads as wax",
        "strong": "impossible undercuts, buttons that intersect shell",
    },
    "editorial_layout": {
        "lite": "random drop shadow stacks",
        "standard": "illegible micro-caption, misaligned column edges",
        "strong": "six competing focal photos, fold line through faces",
    },
    "presentation_slide": {
        "lite": "wall of ten bullet points",
        "standard": "chart without axis tick strategy, mixed icon styles",
        "strong": "4pt font claims, neon glow on body copy, conflicting master themes",
    },
    "technical_blueprint": {
        "lite": "wobbly hand sketch pretending to be CAD",
        "standard": "duplicate leader arrows, dimension text collision",
        "strong": "impossible section cuts, threads that intersect mid-air",
    },
    "fashion_flat": {
        "lite": "garment melting into body",
        "standard": "asymmetric buttons with no functional spacing",
        "strong": "impossible seam routes, zipper teeth on stretch panels wrong",
    },
}

# Named shortcuts: preset -> (domain, intensity).
VISUAL_DESIGN_PRESETS: dict[str, tuple[str, Intensity]] = {
    "saas_ui": ("ui_ux", "strong"),
    "arch_hero": ("architecture", "standard"),
    "stem_exam_figure": ("stem", "strong"),
    "textbook_spread": ("textbook", "standard"),
    "logo_lockup": ("brand", "standard"),
    "annual_report": ("editorial_layout", "strong"),
    "keynote_slide": ("presentation_slide", "standard"),
    "patent_drawing": ("technical_blueprint", "strong"),
    "tech_pack_flat": ("fashion_flat", "standard"),
}

_DOMAIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (domain, re.compile(pattern, re.IGNORECASE))
    for domain, pattern in (
        ("ui_ux", r"\b(ui|ux|gui|mockup|wireframe|figma|design system|component)\b"),
        ("brand", r"\b(logo|wordmark|brand identity|style guide|mascot|monogram)\b"),
        ("architecture", r"\b(architecture|archviz|facade|floor plan|brutalist|bauhaus)\b"),
        ("stem", r"\b(graph|equation|diagram|formula|physics|chemistry|vector field|chart)\b"),
        ("textbook", r"\b(textbook|didactic|workbook|primer|syllabus|student edition)\b"),
        ("infographic", r"\b(infographic|timeline|dashboard explainer|data viz)\b"),
        ("packaging", r"\b(packaging|label design|product box|sleeve|carton)\b"),
        ("wayfinding", r"\b(wayfinding|signage|pictogram|exit sign|map legend)\b"),
        ("editorial_layout", r"\b(editorial spread|magazine spread|annual report spread|masthead)\b"),
        ("presentation_slide", r"\b(keynote|powerpoint slide|pitch deck slide|speaker deck)\b"),
        (
            "technical_blueprint",
            r"\b(blueprint|patent drawing|engineering drawing|section view|exploded diagram|cad drawing)\b",
        ),
        ("fashion_flat", r"\b(fashion flat|technical flat|tech pack flat|garment flat)\b"),
    )
)


def design_pack_ids() -> tuple[str, ...]:
    return tuple(DOMAIN_POSITIVES)


def visual_design_cli_domain_choices() -> tuple[str, ...]:
    """Argparse ``choices`` for ``--visual-design-domain``."""
    return ("none", "auto", *DOMAIN_POSITIVES)


def preset_ids() -> tuple[str, ...]:
    return tuple(sorted(VISUAL_DESIGN_PRESETS))


def normalize_intensity(raw: object) -> Intensity:
    s = str(raw or "standard").strip().lower()
    return s if s in _TIERS else "standard"  # type: ignore[return-value]


def merge_visual_fragments(*parts: str) -> str:
    """Join non-empty prompt fragments with ``", "``."""
    return ", ".join(s for s in ((p or "").strip().strip(",") for p in parts) if s)


def prompt_suggests_domain(prompt: str) -> str | None:
    p = (prompt or "").strip()
    return next((domain for domain, rx in _DOMAIN_PATTERNS if rx.search(p)), None) if p else None


def build_visual_design_prompt_pair(prompt: str, domain: str, *, intensity: str = "standard") -> tuple[str, str]:
    """Return ``(positive, negative_addon)``; the fragment is not repeated when the prompt already has it."""
    base = (prompt or "").strip()
    d = (domain or "").strip().lower()
    if d not in DOMAIN_POSITIVES:
        raise ValueError(f"Unknown visual design domain {domain!r}; use one of: {design_pack_ids()}")
    tier = normalize_intensity(intensity)
    fragment = DOMAIN_POSITIVES[d][tier]
    positive = base if fragment.lower() in base.lower() else merge_visual_fragments(base, fragment)
    return positive, DOMAIN_NEGATIVES[d][tier].strip()


def apply_visual_design_preset_to_prompt(prompt: str, preset: str) -> tuple[str, str, str]:
    """``(prompt, domain, intensity)`` for a named preset; unknown presets give domain ``"none"``."""
    domain, intensity = VISUAL_DESIGN_PRESETS.get((preset or "").strip().lower(), ("none", "standard"))
    return (prompt or "").strip(), domain, intensity


@dataclass(frozen=True, slots=True)
class VisualDesignApplyResult:
    prompt: str
    negative_addon: str
    resolved_domain: str


def apply_visual_design_stage(
    prompt: str,
    *,
    cli_domain: str,
    intensity: str = "standard",
    use_negative_pack: bool,
    emit: Callable[[str], None] | None = None,
) -> VisualDesignApplyResult:
    """Apply ``--visual-design-domain`` (``auto`` guesses the domain from the prompt) as ``sample.py`` does."""
    base = (prompt or "").strip()
    skipped = VisualDesignApplyResult(prompt=base, negative_addon="", resolved_domain="none")
    resolved = str(cli_domain or "none").strip().lower()
    if resolved in {"", "none"} or not base:
        return skipped
    if resolved == "auto":
        guess = prompt_suggests_domain(base)
        if guess is None:
            if emit:
                emit("Note: visual-design-domain=auto did not match a domain; skipping pack.")
            return skipped
        resolved = guess
        if emit:
            emit(f"Visual design: auto-selected domain {resolved}")
    try:
        pos, neg = build_visual_design_prompt_pair(base, resolved, intensity=intensity)
    except ValueError:
        if emit:
            emit(f"Visual design pack skipped: unknown domain {resolved!r}")
        return skipped
    return VisualDesignApplyResult(
        prompt=pos, negative_addon=neg if use_negative_pack else "", resolved_domain=resolved
    )


def extend_sample_argv_visual_design(cmd: list[str], args: Any) -> None:
    """Append the ``--visual-design-*`` flags of ``args`` to a ``sample.py`` argv."""
    preset = str(getattr(args, "visual_design_preset", "") or "").strip()
    domain = str(getattr(args, "visual_design_domain", "none") or "none").strip().lower()
    if preset:
        cmd.extend(["--visual-design-preset", preset])
    elif domain != "none":
        intensity = normalize_intensity(getattr(args, "visual_design_intensity", "standard"))
        cmd.extend(["--visual-design-domain", domain, "--visual-design-intensity", intensity])
    else:
        return
    if getattr(args, "visual_design_negative_pack", False):
        cmd.append("--visual-design-negative-pack")


__all__ = [
    "DOMAIN_NEGATIVES",
    "DOMAIN_POSITIVES",
    "VISUAL_DESIGN_PRESETS",
    "Intensity",
    "VisualDesignApplyResult",
    "apply_visual_design_preset_to_prompt",
    "apply_visual_design_stage",
    "build_visual_design_prompt_pair",
    "design_pack_ids",
    "extend_sample_argv_visual_design",
    "merge_visual_fragments",
    "normalize_intensity",
    "preset_ids",
    "prompt_suggests_domain",
    "visual_design_cli_domain_choices",
]
