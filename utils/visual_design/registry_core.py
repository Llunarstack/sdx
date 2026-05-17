"""Primary visual-design fragments (domains most commonly used with DiT workflows)."""

from __future__ import annotations

from typing import Dict

DOMAIN_POSITIVES_CORE: Dict[str, Dict[str, str]] = {
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
}

DOMAIN_NEGATIVES_CORE: Dict[str, Dict[str, str]] = {
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
}

__all__ = ["DOMAIN_POSITIVES_CORE", "DOMAIN_NEGATIVES_CORE"]
