"""Additional design domains (editorial layouts, decks, flats, CAD-style reads)."""

from __future__ import annotations

from typing import Dict

DOMAIN_POSITIVES_EXTRA: Dict[str, Dict[str, str]] = {
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

DOMAIN_NEGATIVES_EXTRA: Dict[str, Dict[str, str]] = {
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

__all__ = ["DOMAIN_POSITIVES_EXTRA", "DOMAIN_NEGATIVES_EXTRA"]
