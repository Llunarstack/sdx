"""
Complex prompt coverage analysis for training manifests.

This helps you verify that your captions include the "hard to learn" ingredients:
- styles and "hard vibe" words (weird/strange/surreal)
- NSFW descriptors (detected, but suggestions are generic by default)
- clothes and wardrobe details
- anatomy details beyond the basic person descriptor list
- weapons/props
- food
- text-in-image
- foreground/background framing
- fine details / textures / particles

Usage:
  python -m scripts.tools.complex_prompt_coverage --manifest manifest.jsonl --out report.json
  python -m scripts.tools.complex_prompt_coverage --manifest manifest.jsonl --thresholds "clothes:0.25,weapons:0.05"

Manifest format:
  JSONL rows with one of: `caption`, `text`, `prompt`
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _get_caption(rec: Dict[str, Any]) -> str:
    for k in ("caption", "text", "prompt"):
        v = rec.get(k)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return ""


def _term_present(caption_lower: str, term: str) -> bool:
    """
    Basic term matcher:
    - multi-word terms: substring match
    - single-word terms: word-ish boundary match to reduce false positives
    """
    t = term.strip().lower()
    if not t:
        return False
    if " " in t:
        return t in caption_lower
    esc = re.escape(t)
    return re.search(rf"(?<![a-z0-9_]){esc}(?![a-z0-9_])", caption_lower) is not None


def _frac(n: int, total: int) -> float:
    return n / total if total else 0.0


def _parse_thresholds(thresholds_str: str) -> Dict[str, float]:
    # Format: cat:0.1,cat2:0.3
    out: Dict[str, float] = {}
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


def _load_custom_categories(custom_path: str) -> Dict[str, Dict[str, Any]]:
    p = Path(custom_path)
    if not p.exists():
        raise SystemExit(f"Custom categories file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    cats = data.get("categories", data)  # allow either wrapper or direct mapping
    if not isinstance(cats, dict):
        raise SystemExit('custom-categories-json must be {"categories": {name: {terms:[...]}}} or {name: {...}}')
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in cats.items():
        if not isinstance(spec, dict):
            continue
        terms = spec.get("terms", [])
        if isinstance(terms, list) and all(isinstance(x, str) for x in terms):
            out[str(name)] = {"terms": terms}
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
        help='Optional thresholds like "clothes:0.25,weapons:0.05". Only these categories are FAIL-checked.',
    )
    ap.add_argument("--min-rows", type=int, default=0, help="Stop after scanning N rows (0 = all)")
    ap.add_argument(
        "--custom-categories-json",
        type=str,
        default="",
        help='Optional JSON file with extra/override categories: {"categories": {name: {"terms": [...]}}}',
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.tools.prompt_i18n import generic_suggestion

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Not found: {manifest_path}")

    # Built-in categories.
    # Notes:
    # - NSFW detection is token-based (so it will match your dataset's chosen vocab).
    # - Suggestions for NSFW remain generic unless you include your own categories.
    built_in: Dict[str, Dict[str, Any]] = {
        "nsfw_descriptors": {
            "terms": [
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
            ]
        },
        "weird_strange": {
            "terms": [
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
            ]
        },
        "clothes_wardrobe": {
            "terms": [
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
            ]
        },
        "anatomy_details": {
            "terms": [
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
            ]
        },
        "weapons_props": {
            "terms": [
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
            ]
        },
        "food_drink": {
            "terms": [
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
            ]
        },
        "text_in_image": {
            "terms": [
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
            ]
        },
        "foreground_framing": {
            "terms": [
                "foreground",
                "in the foreground",
                "foreground bokeh",
                "near the camera",
                "closest",
                "close-up foreground",
            ]
        },
        "background_framing": {
            "terms": [
                "background",
                "in the background",
                "distant background",
                "far background",
                "backdrop",
                "sky",
                "horizon",
            ]
        },
        "fine_details": {
            "terms": [
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
            ]
        },
        "hands_fingers": {
            "terms": ["hands", "correct hands", "five fingers", "hand focus", "visible hands"],
        },
        "pose_actions": {
            "terms": [
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
        },
        "character_scene": {
            "terms": [
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
        },
        "clothes_materials": {
            "terms": [
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
        },
        "art_styles": {
            "terms": [
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
        },
    }

    if args.custom_categories_json:
        custom = _load_custom_categories(args.custom_categories_json)
        for name, spec in custom.items():
            built_in[name] = {"terms": list(spec.get("terms", []))}

    categories = built_in

    thresholds = _parse_thresholds(args.thresholds)

    counts: Dict[str, int] = {k: 0 for k in categories.keys()}
    total = 0

    for rec in _iter_jsonl(manifest_path):
        cap = _get_caption(rec)
        if not cap:
            continue
        total += 1
        cap_lower = cap.lower()
        for name, spec in categories.items():
            for t in spec.get("terms", []):
                if _term_present(cap_lower, t):
                    counts[name] += 1
                    break
        if args.min_rows and total >= args.min_rows:
            break

    if total == 0:
        raise SystemExit("No captions found in manifest.")

    fractions: Dict[str, float] = {k: _frac(counts[k], total) for k in categories.keys()}

    # Human suggestions language.
    sug_en: Dict[str, str] = {
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
        "clothes_materials": "Add clothing material terms (leather/silk/latexamples/denim/wool/cotton) so outfits are stable.",
        "art_styles": "Add explicit art-style tokens (anime/manga/manhwa/oil painting/cyberpunk/noir/etc.) early in captions.",
    }

    lang = str(args.lang or "en").strip()

    def _load_suggestions(path_str: str) -> Dict[str, Dict[str, str]]:
        p = Path(path_str)
        if not p.exists():
            raise SystemExit(f"Not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        cats = data.get("categories", data)
        if not isinstance(cats, dict):
            raise SystemExit('suggestions-json must be {"categories": {...}} or {...category...} structure.')
        out: Dict[str, Dict[str, str]] = {}
        for cat, spec in cats.items():
            if isinstance(spec, dict):
                out[str(cat)] = {str(k): str(v) for k, v in spec.items() if isinstance(v, str)}
        return out

    custom_suggestions: Dict[str, Dict[str, str]] = {}
    if args.suggestions_json:
        custom_suggestions = _load_suggestions(args.suggestions_json)

    normalized_lang = lang.lower()
    is_en = normalized_lang in ("en", "eng", "english")

    # Choose best available suggestion per category.
    def _pick_suggestion(category_name: str) -> str:
        if custom_suggestions:
            by_lang = custom_suggestions.get(category_name, {})
            # exact key first, then lowercase key match
            if lang in by_lang:
                return by_lang[lang]
            for k, v in by_lang.items():
                if str(k).lower() == normalized_lang:
                    return v
        if is_en:
            return sug_en.get(category_name, "")
        # unknown language: use localized generic template
        examples = categories.get(category_name, {}).get("terms", [])
        return generic_suggestion(category_name, examples, normalized_lang)

    # Print + optional FAIL gate.
    failed: List[str] = []
    for name, frac in sorted(fractions.items(), key=lambda kv: kv[1]):
        cnt = counts[name]
        mark = ""
        if name in thresholds and frac < thresholds[name]:
            failed.append(name)
            mark = "  <-- FAIL"
        print(f"- {name}: {frac:.3f} ({cnt}/{total}){mark}")

    if failed:
        print("\nFAIL categories:", ", ".join(failed))
        for name in failed:
            msg = _pick_suggestion(name)
            if msg:
                print("-", msg)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "manifest": str(manifest_path),
                        "rows_scanned": total,
                        "counts": counts,
                        "fractions": fractions,
                        "thresholds": thresholds,
                        "failed": failed,
                        "suggestions_lang": lang,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        raise SystemExit(2)

    print("\nPASS: coverage computed (and thresholds passed, if provided).")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "rows_scanned": total,
                    "counts": counts,
                    "fractions": fractions,
                    "thresholds": thresholds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
