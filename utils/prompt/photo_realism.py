"""Photography / photorealism prompt helpers."""

from __future__ import annotations

from typing import Dict, Set, Tuple

PHOTO_REALISM_PACKS: Dict[str, Tuple[str, str]] = {
    "none": ("", ""),
    "documentary": (
        "photoreal documentary capture, natural lens perspective, candid realism, controlled dynamic range",
        "cgi render look, overprocessed hdr, waxy skin, synthetic microtexture",
    ),
    "cinematic": (
        "photoreal cinematic still, motivated practical lighting, filmic tonal separation, plausible optical depth",
        "clipped highlights, crushed blacks, fake bloom halos, inconsistent light motivation",
    ),
    "studio_portrait": (
        "studio portrait realism, key-fill-rim lighting discipline, realistic skin pores, subtle lens compression",
        "plastic skin retouch, uncanny symmetry, over-smoothing, artificial eye shine",
    ),
    "film_analog": (
        "analog film realism, organic grain cadence, natural highlight rolloff, gentle color response",
        "digital crunch artifacts, sterile noise pattern, harsh clipping, over-sharpened edges",
    ),
    "night_noir": (
        "night-time photorealism, practical-light contrast control, atmospheric depth cues, reflective surface realism",
        "flat night exposure, neon clipping, muddy shadows, fake reflective behavior",
    ),
    "product_catalog": (
        "commercial product photography realism, controlled studio reflections, clean edge definition, true material response",
        "warped product geometry, implausible reflections, blown specular hotspots, noisy gradients",
    ),
    "fashion_editorial": (
        "fashion editorial photorealism, high-end lighting styling, skin-texture fidelity, premium composition polish",
        "plastic beauty filter look, over-airbrushed skin, clipped fabric highlights, flat posing",
    ),
}

PHOTO_COLOR_GRADE_HINTS: Dict[str, str] = {
    "none": "",
    "natural": "natural color grade, restrained saturation, realistic white balance continuity",
    "teal_orange": "cinematic teal-orange split-tone with controlled skin-tone preservation",
    "kodak_portra": "kodak portra inspired color response, warm skin bias, soft contrast rolloff",
    "cinestill_800t": "cinestill 800T inspired tungsten-night palette, cool shadows with warm practical highlights",
    "noir_bw": "black-and-white film-noir grade, rich monochrome contrast and tonal separation",
    "fujifilm_eterna": "fujifilm eterna inspired subtle cinematic palette, restrained contrast and organic color separation",
}

PHOTO_LIGHTING_TECHNIQUES: Dict[str, str] = {
    "none": "",
    "three_point": "three-point lighting discipline with balanced key-fill-backlight ratio",
    "golden_hour": "golden-hour lighting technique with warm directional key and soft atmospheric fill",
    "overcast_soft": "overcast soft-light technique with low-contrast skin-friendly diffusion",
    "motivated_practical": "motivated practical-light technique, source-driven highlights and grounded shadow logic",
    "rim_backlight": "rim/backlight separation technique for clear subject silhouette depth",
    "butterfly": "butterfly portrait lighting technique with flattering facial-plane highlights",
    "rembrandt": "rembrandt portrait lighting technique with controlled triangle-light facial modeling",
}

PHOTO_FILTER_HINTS: Dict[str, str] = {
    "none": "",
    "pro_mist": "pro-mist diffusion filter feel, softened highlight halation without detail collapse",
    "polarizer": "polarizer filter behavior, controlled reflections, cleaner sky and surface contrast",
    "nd_long_exposure": "ND long-exposure photographic behavior with intentional motion blend and exposure control",
    "vintage_diffusion": "vintage diffusion filter character with gentle glow and softened micro-contrast",
    "clean_digital": "clean modern digital filter profile with neutral contrast and minimal color cast",
}

PHOTO_GRAIN_HINTS: Dict[str, str] = {
    "none": "",
    "fine_35mm": "fine 35mm-like grain structure with subtle luminance texture",
    "medium_35mm": "medium 35mm-like grain cadence, visible but balanced texture presence",
    "heavy_16mm": "heavy 16mm-like grain character for gritty analog texture",
    "clean_digital": "clean digital noise floor with minimal grain impression",
}

_PHOTO_INTENT_TERMS: Set[str] = {
    "photo",
    "photograph",
    "photoreal",
    "photorealistic",
    "dslr",
    "mirrorless",
    "35mm",
    "film",
    "cinematic still",
    "documentary",
    "editorial",
    "portrait",
    "studio lighting",
    "bokeh",
    "lens",
    "iso",
    "shutter",
    "aperture",
    "golden hour",
    "color grading",
}


def infer_photo_realism_controls(prompt: str) -> Dict[str, str]:
    p = str(prompt or "").lower()
    out: Dict[str, str] = {}
    if any(k in p for k in ("documentary", "photojournal", "street photo", "candid")):
        out["photo_realism_pack"] = "documentary"
    elif any(k in p for k in ("studio portrait", "headshot", "beauty photo")):
        out["photo_realism_pack"] = "studio_portrait"
    elif any(k in p for k in ("fashion editorial", "editorial photo")):
        out["photo_realism_pack"] = "fashion_editorial"
    elif any(k in p for k in ("product photo", "catalog photo", "packshot")):
        out["photo_realism_pack"] = "product_catalog"
    elif any(k in p for k in ("night noir", "night photo", "neon night")):
        out["photo_realism_pack"] = "night_noir"
    elif any(k in p for k in ("analog film", "35mm", "film photo", "portra", "cinestill")):
        out["photo_realism_pack"] = "film_analog"
    elif any(k in p for k in ("cinematic still", "cinematic photo")):
        out["photo_realism_pack"] = "cinematic"

    if any(k in p for k in ("teal orange", "teal-orange")):
        out["photo_color_grade"] = "teal_orange"
    elif "portra" in p:
        out["photo_color_grade"] = "kodak_portra"
    elif "cinestill" in p:
        out["photo_color_grade"] = "cinestill_800t"
    elif any(k in p for k in ("black and white", "black-and-white", "monochrome noir")):
        out["photo_color_grade"] = "noir_bw"

    if any(k in p for k in ("golden hour", "sunset photo")):
        out["photo_lighting_technique"] = "golden_hour"
    elif any(k in p for k in ("three point lighting", "three-point lighting")):
        out["photo_lighting_technique"] = "three_point"
    elif "overcast" in p:
        out["photo_lighting_technique"] = "overcast_soft"

    if "pro mist" in p or "promist" in p:
        out["photo_filter"] = "pro_mist"
    elif "polarizer" in p:
        out["photo_filter"] = "polarizer"
    elif "long exposure" in p or "nd filter" in p:
        out["photo_filter"] = "nd_long_exposure"

    return out


def is_photographic_prompt(prompt: str) -> bool:
    p = str(prompt or "").lower()
    if not p.strip():
        return False
    return any(term in p for term in _PHOTO_INTENT_TERMS)


def recommend_photo_post_profile(
    *,
    photo_realism_pack: str,
    photo_color_grade: str,
    photo_filter: str,
    photo_grain_style: str,
) -> Dict[str, str]:
    """
    Recommend stronger realism defaults for postprocess and ranking.
    """
    pack = str(photo_realism_pack or "none").lower().strip()
    grade = str(photo_color_grade or "none").lower().strip()
    filt = str(photo_filter or "none").lower().strip()
    grain = str(photo_grain_style or "none").lower().strip()

    post_strength = "0.62"
    pick_metric = "combo_realism"
    if pack in {"studio_portrait", "fashion_editorial", "product_catalog"}:
        post_strength = "0.58"
    elif pack in {"film_analog", "night_noir"}:
        post_strength = "0.72"
    elif pack in {"cinematic", "documentary"}:
        post_strength = "0.64"

    if grade in {"noir_bw", "cinestill_800t"}:
        post_strength = str(max(float(post_strength), 0.7))
    if filt in {"nd_long_exposure", "vintage_diffusion"}:
        post_strength = str(max(float(post_strength), 0.68))
    if grain == "clean_digital":
        post_strength = str(min(float(post_strength), 0.55))
    if grain == "none" and pack == "film_analog":
        grain = "fine_35mm"

    return {
        "photo_post_strength": post_strength,
        "pick_best_metric": pick_metric,
        "photo_grain_style": grain,
    }


def photo_realism_fragments(
    *,
    photo_realism_pack: str = "none",
    photo_color_grade: str = "none",
    photo_lighting_technique: str = "none",
    photo_filter: str = "none",
    photo_grain_style: str = "none",
    strength: float = 1.0,
) -> Tuple[str, str]:
    """
    Build positive/negative prompt fragments for photoreal/photography realism.
    """
    s = max(0.25, min(2.0, float(strength)))
    pos_pack, neg_pack = PHOTO_REALISM_PACKS.get((photo_realism_pack or "none").lower().strip(), ("", ""))
    pos = ", ".join(
        x
        for x in (
            pos_pack,
            PHOTO_COLOR_GRADE_HINTS.get((photo_color_grade or "none").lower().strip(), ""),
            PHOTO_LIGHTING_TECHNIQUES.get((photo_lighting_technique or "none").lower().strip(), ""),
            PHOTO_FILTER_HINTS.get((photo_filter or "none").lower().strip(), ""),
            PHOTO_GRAIN_HINTS.get((photo_grain_style or "none").lower().strip(), ""),
        )
        if str(x).strip()
    )
    neg = str(neg_pack or "").strip()
    if not pos and not neg:
        return "", ""
    if abs(s - 1.0) < 1e-6:
        return pos, neg
    # Light weighting syntax compatible with existing prompt emphasis parsing.
    weighted = f"({pos}:{s:.2f})" if pos else ""
    return weighted, neg

