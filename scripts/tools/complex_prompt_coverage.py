"""
Complex prompt coverage analysis for training manifests.

Checks that captions include the "hard to learn" ingredients (styles and weird/surreal vibes, NSFW
descriptors, wardrobe, anatomy details, weapons/props, food, text-in-image, foreground/background
framing, fine textures). ``prompt_gap_scout`` reuses the same categories for a single prompt.

Usage:
  python -m scripts.tools complex_prompt_coverage --manifest manifest.jsonl --out report.json
  python -m scripts.tools complex_prompt_coverage --manifest manifest.jsonl --thresholds "clothes_wardrobe:0.25"

Manifest format:
  JSONL rows with one of: `caption`, `text`, `prompt`
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path

from data.manifest_utils import caption_from_row, iter_manifest_rows
from utils.prompt.prompt_i18n import generic_suggestion

# NSFW detection is token-based so it matches the dataset's own vocabulary; its suggestion stays generic.
CATEGORIES: dict[str, list[str]] = {
    "nsfw_descriptors": [
        "nsfw",
        "adult",
        "erotic",
        "explicit",
        "porn",
        "nude",
        "lingerie",
        "underwear",
        "swimsuit",
        "bare",
        "revealing",
    ],
    "weird_strange": [
        "surreal",
        "abstract",
        "bizarre",
        "uncanny",
        "weird",
        "strange",
        "impossible",
        "glitch",
        "cyberpunk",
        "horror",
        "creepy",
        "grotesque",
        "eldritch",
        "non-euclidean",
    ],
    "clothes_wardrobe": [
        "dress",
        "coat",
        "jacket",
        "shirt",
        "t-shirt",
        "hoodie",
        "sweater",
        "uniform",
        "armor",
        "gloves",
        "boots",
        "scarf",
        "belt",
        "cape",
        "cloak",
        "robe",
        "kimono",
        "suit",
        "skirt",
        "leggings",
        "leotard",
        "lingerie",
        "underwear",
        "swimsuit",
        "corset",
        "tactical vest",
    ],
    "anatomy_details": [
        "tattoo",
        "scar",
        "freckles",
        "piercing",
        "mole",
        "beauty mark",
        "bandage",
        "wounds",
        "bruise",
        "veins",
        "muscle",
        "body hair",
    ],
    "weapons_props": [
        "sword",
        "gun",
        "rifle",
        "pistol",
        "knife",
        "blade",
        "bow",
        "arrow",
        "staff",
        "hammer",
        "axe",
        "spear",
        "katana",
        "revolver",
        "crossbow",
        "grenade",
        "blaster",
        "laser",
        "trident",
        "wand",
    ],
    "food_drink": [
        "pizza",
        "ramen",
        "sushi",
        "burger",
        "taco",
        "steak",
        "soup",
        "salad",
        "sandwich",
        "cake",
        "pie",
        "dessert",
        "chocolate",
        "coffee",
        "tea",
        "noodles",
        "dumplings",
        "bread",
        "cheese",
        "spaghetti",
    ],
    "text_in_image": [
        "legible text",
        "clear text",
        "readable text",
        "correct spelling",
        "sign",
        "signage",
        "label",
        "lettering",
        "words",
        "caption",
        "text that says",
        "sign that says",
    ],
    "foreground_framing": [
        "foreground",
        "in the foreground",
        "foreground bokeh",
        "near the camera",
        "closest",
        "close-up foreground",
    ],
    "background_framing": [
        "background",
        "in the background",
        "distant background",
        "far background",
        "backdrop",
        "sky",
        "horizon",
    ],
    "fine_details": [
        "intricate",
        "ornate",
        "embroidery",
        "stitched",
        "seams",
        "texture",
        "textured",
        "fabric",
        "weathered",
        "rust",
        "scuffed",
        "steam",
        "smoke",
        "fog",
        "dust",
        "particles",
        "micro details",
        "depth of field",
        "bokeh",
    ],
    "hands_fingers": ["hands", "correct hands", "five fingers", "hand focus", "visible hands"],
    "pose_actions": [
        "pose",
        "stance",
        "standing",
        "walking",
        "running",
        "dancing",
        "crouching",
        "kneeling",
        "sitting",
        "lying",
        "lying down",
        "holding",
        "gripping",
        "grabbing",
        "wielding",
        "carrying",
        "aiming",
        "shooting",
        "swinging",
        "attacking",
        "casting",
        "eating",
        "drinking",
    ],
    "character_scene": [
        "group",
        "crowd",
        "team",
        "duo",
        "multiple people",
        "room full of people",
        "many people",
        "several people",
        "gathering",
        "audience",
        "multiple characters",
    ],
    "clothes_materials": [
        "leather",
        "silk",
        "latex",
        "denim",
        "wool",
        "cotton",
        "chainmail",
        "lace",
        "metallic",
        "ribbon",
        "feather",
        "fur",
        "plaid",
        "stitched",
        "sewn",
    ],
    "art_styles": [
        "anime",
        "manga",
        "manhwa",
        "illustration",
        "digital painting",
        "oil painting",
        "watercolor",
        "cel shading",
        "comic",
        "cartoon",
        "cinematic",
        "baroque",
        "gothic",
        "cyberpunk",
        "steampunk",
        "noir",
        "retro",
        "grainy film",
        "photography",
        "macro",
    ],
}

SUGGESTIONS_EN: dict[str, str] = {
    "nsfw_descriptors": "If you want NSFW behavior, keep your NSFW descriptors consistent across the dataset and include them early in captions.",
    "weird_strange": "Add more surreal/weird descriptors so the model learns non-standard visuals consistently.",
    "clothes_wardrobe": "Add wardrobe/clothing details (garments, materials, accessories) so outfits are stable.",
    "anatomy_details": "Add anatomy micro-details (tattoos/scars/freckles/piercings) for higher consistency and specificity.",
    "weapons_props": "Add explicit weapon/prop words and (ideally) action verbs like 'holding' / 'gripping'.",
    "food_drink": "Add explicit food/drink words and eating/serving context if relevant.",
    "text_in_image": "If you need text in-image, include explicit 'legible/clear readable text' wording and spelling cues.",
    "foreground_framing": "Add foreground framing cues (foreground bokeh, nearby objects) early in captions.",
    "background_framing": "Add background framing cues (detailed background, backdrop, sky/horizon) early in captions.",
    "fine_details": "Add texture/detail descriptors (intricate, stitched, weathered, particles, micro details).",
    "hands_fingers": "If hands fail, add 'correct hands', 'five fingers', and 'hand focus' consistently.",
    "pose_actions": "Add explicit pose/action wording (standing/walking + holding/gripping/carrying verbs) so scenes don’t collapse.",
    "character_scene": "If you want multi-character scenes, include crowd/group words consistently (group/crowd/team/room full of people).",
    "clothes_materials": "Add clothing material terms (leather/silk/latex/denim/wool/cotton) so outfits are stable.",
    "art_styles": "Add explicit art-style tokens (anime/manga/manhwa/oil painting/cyberpunk/noir/etc.) early in captions.",
}


def term_present(caption_lower: str, term: str) -> bool:
    """Multi-word terms match as substrings; single words need word-ish boundaries."""
    t = term.strip().lower()
    if not t:
        return False
    if " " in t:
        return t in caption_lower
    return re.search(rf"(?<![a-z0-9_]){re.escape(t)}(?![a-z0-9_])", caption_lower) is not None


def _json_categories(path_str: str, what: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        raise SystemExit(f"Not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    cats = data.get("categories", data)
    if not isinstance(cats, dict):
        raise SystemExit(f'{what} must be {{"categories": {{...}}}} or a category mapping.')
    return cats


def load_custom_categories(path_str: str) -> dict[str, list[str]]:
    """``{"categories": {name: {"terms": [...]}}}`` (or the bare mapping) -> ``{name: terms}``."""
    out: dict[str, list[str]] = {}
    for name, spec in _json_categories(path_str, "custom-categories-json").items():
        terms = spec.get("terms") if isinstance(spec, dict) else None
        if isinstance(terms, list) and all(isinstance(x, str) for x in terms):
            out[str(name)] = terms
    return out


def suggestion_picker(lang: str, suggestions_json: str, categories: dict[str, list[str]]) -> Callable[[str], str]:
    """Per-category hint: ``--suggestions-json`` entry for ``lang``, else built-in English, else a localized template."""
    custom: dict[str, dict[str, str]] = {}
    if suggestions_json:
        for cat, spec in _json_categories(suggestions_json, "suggestions-json").items():
            if isinstance(spec, dict):
                custom[str(cat)] = {str(k).lower(): v for k, v in spec.items() if isinstance(v, str)}
    norm = (lang or "en").strip().lower()

    def pick(category: str) -> str:
        hit = custom.get(category, {}).get(norm)
        if hit:
            return hit
        if norm in ("en", "eng", "english"):
            return SUGGESTIONS_EN.get(category, "")
        return generic_suggestion(category, categories.get(category, []), norm)

    return pick


def _parse_thresholds(thresholds_str: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not thresholds_str.strip():
        return out
    parts = [p.strip() for p in thresholds_str.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except ValueError:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Complex prompt coverage for tricky categories.")
    ap.add_argument("--manifest", type=str, required=True, help="Input JSONL manifest")
    ap.add_argument("--out", type=str, default="", help="Optional JSON output report")
    ap.add_argument("--lang", type=str, default="en", help="Language code/name for suggestions (e.g. en, zh, es, fr)")
    ap.add_argument(
        "--suggestions-json",
        type=str,
        default="",
        help=(
            "Optional JSON file with localized suggestion text per category. "
            "Supported formats: "
            '1) {"{category}": {"en": "...", "es": "..."}} '
            '2) {"categories": {"{category}": {"en": "...", "es": "..."}}} '
            "If not provided (or missing language), tool falls back to built-in English."
        ),
    )
    ap.add_argument(
        "--thresholds",
        type=str,
        default="",
        help='Optional thresholds like "clothes_wardrobe:0.25,weapons_props:0.05"; only these categories are FAIL-checked.',
    )
    ap.add_argument("--min-rows", type=int, default=0, help="Stop after scanning N rows (0 = all)")
    ap.add_argument(
        "--custom-categories-json",
        type=str,
        default="",
        help='Optional JSON file with extra/override categories: {"categories": {name: {"terms": [...]}}}',
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Not found: {manifest_path}")

    categories = dict(CATEGORIES)
    if args.custom_categories_json:
        categories.update(load_custom_categories(args.custom_categories_json))
    thresholds = _parse_thresholds(args.thresholds)

    counts: dict[str, int] = dict.fromkeys(categories, 0)
    total = 0
    for rec in iter_manifest_rows(manifest_path):
        cap = caption_from_row(rec)
        if not cap:
            continue
        total += 1
        cap_lower = cap.lower()
        for name, terms in categories.items():
            if any(term_present(cap_lower, t) for t in terms):
                counts[name] += 1
        if args.min_rows and total >= args.min_rows:
            break
    if total == 0:
        raise SystemExit("No captions found in manifest.")

    fractions = {k: counts[k] / total for k in categories}
    failed: list[str] = []
    for name, frac in sorted(fractions.items(), key=lambda kv: kv[1]):
        mark = ""
        if name in thresholds and frac < thresholds[name]:
            failed.append(name)
            mark = "  <-- FAIL"
        print(f"- {name}: {frac:.3f} ({counts[name]}/{total}){mark}")

    report = {
        "manifest": str(manifest_path),
        "rows_scanned": total,
        "counts": counts,
        "fractions": fractions,
        "thresholds": thresholds,
    }
    if failed:
        print("\nFAIL categories:", ", ".join(failed))
        pick = suggestion_picker(args.lang, args.suggestions_json, categories)
        for name in failed:
            msg = pick(name)
            if msg:
                print("-", msg)
        report |= {"failed": failed, "suggestions_lang": args.lang}
    else:
        print("\nPASS: coverage computed (and thresholds passed, if provided).")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
