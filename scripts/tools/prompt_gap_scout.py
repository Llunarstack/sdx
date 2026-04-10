"""
Prompt gap scout:
Given one prompt/caption, reports which tricky categories are missing (or weak),
and suggests what to add.

This is meant for dataset authors + inference experiments:
- complexamples/weird words
- NSFW descriptors (suggestions are generic unless you extend categories)
- clothes/wardrobe
- anatomy micro-details + hands
- weapons/props
- food/drink
- text-in-image cues
- foreground/background framing
- fine textures/details

Usage:
  python -m scripts.tools.prompt_gap_scout --prompt "..." --lang en
  python -m scripts.tools.prompt_gap_scout --prompt-file prompt.txt --lang zh
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

from scripts.tools.prompt_i18n import generic_suggestion


def _term_present(caption_lower: str, term: str) -> bool:
    t = term.strip().lower()
    if not t:
        return False
    if " " in t:
        return t in caption_lower
    esc = re.escape(t)
    return re.search(rf"(?<![a-z0-9_]){esc}(?![a-z0-9_])", caption_lower) is not None


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt and args.prompt.strip():
        return args.prompt.strip()
    if args.prompt_file and args.prompt_file.strip():
        p = Path(args.prompt_file)
        if not p.exists():
            raise SystemExit(f"Not found: {p}")
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    raise SystemExit("Provide --prompt or --prompt-file.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Find missing tricky categories in a prompt/caption.")
    ap.add_argument("--prompt", type=str, default="", help="Prompt/caption text")
    ap.add_argument("--prompt-file", type=str, default="", help="File containing prompt/caption")
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
            "If not provided (or missing language), tool falls back to built-in English/Chinese."
        ),
    )
    ap.add_argument(
        "--custom-categories-json",
        type=str,
        default="",
        help='Optional JSON file like {"categories": {"name": {"terms":[...]}}} to extend categories.',
    )
    ap.add_argument("--json", action="store_true", help="Output JSON only")
    args = ap.parse_args()

    prompt = _load_prompt(args)

    # Categories mirror complex_prompt_coverage defaults, but the tool outputs missing items.
    categories: Dict[str, Dict[str, Any]] = {
        "weird_strange": {
            "terms": ["surreal", "abstract", "bizarre", "uncanny", "weird", "strange", "impossible", "glitch"],
            "examples_en": "surreal, abstract, uncanny, glitchy non-standard geometry",
            "examples_zh": "超现实, 抽象, 离奇, 不自然感, glitch 风格的非标准几何",
            "suggest_en": "Add 1-3 weird/strange anchors early (surreal/uncanny/impossible/glitch words).",
            "suggest_zh": "尽量在前半段加入 1-3 个怪诞/离奇锚点词（surreal/uncanny/impossible/glitch）。",
        },
        "nsfw_descriptors": {
            "terms": ["nsfw", "adult", "erotic", "explicit", "porn", "nude"],
            "examples_en": "",
            "examples_zh": "",
            "suggest_en": "If you want NSFW behavior, keep your NSFW descriptors consistent across captions (use your dataset's exact tag set).",
            "suggest_zh": "如果你希望学到 NSFW，请保持数据集中 NSFW 描述词一致（使用你自己的标签集合）。",
        },
        "clothes_wardrobe": {
            "terms": [
                "dress",
                "coat",
                "jacket",
                "shirt",
                "hoodie",
                "uniform",
                "armor",
                "gloves",
                "boots",
                "belt",
                "cape",
                "cloak",
            ],
            "examples_en": "wearing a coat and gloves, belt, boots",
            "examples_zh": "穿着外套和手套，腰带，靴子",
            "suggest_en": "Add explicit wardrobe details (garments + material + accessories).",
            "suggest_zh": "补充明确的穿搭细节（服装类型 + 材质 + 配饰）。",
        },
        "anatomy_details": {
            "terms": ["tattoo", "scar", "freckles", "piercing", "mole", "beauty mark", "veins", "muscle"],
            "examples_en": "freckles, tattoo, subtle scars, defined veins",
            "examples_zh": "雀斑, 纹身, 细微疤痕, 明显静脉纹理",
            "suggest_en": "Add anatomy micro-details and keep them consistent in your captions.",
            "suggest_zh": "加入解剖微细节，并保证在 captions 里一致出现。",
        },
        "hands_fingers": {
            "terms": ["correct hands", "five fingers", "hand focus", "visible hands"],
            "examples_en": "correct hands, five fingers, hand focus",
            "examples_zh": "正确的手部, 五指, 手部聚焦",
            "suggest_en": "Hand failures: explicitly include correct hands / five fingers / hand focus.",
            "suggest_zh": "手部失败：明确加入 correct hands / five fingers / hand focus。",
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
            "examples_en": "standing, holding a sword, aiming a bow",
            "examples_zh": "站立，手持剑，瞄准弓箭",
            "suggest_en": "Add explicit pose/action words (holding/gripping/carrying + the verb that matches your scene).",
            "suggest_zh": "补充明确的姿势/动作词（holding/gripping/carrying + 与场景一致的动词）。",
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
            "examples_en": "a group of characters, a crowd, multiple characters in frame",
            "examples_zh": "一群角色，crowd，画面里多个角色",
            "suggest_en": "For multi-character scenes, include crowd/group/team words consistently (group/crowd/team/room full of people).",
            "suggest_zh": "多角色场景时，保持群体词一致出现（group/crowd/team/room full of people）。",
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
            "examples_en": "leather jacket, silk scarf, stitched seams",
            "examples_zh": "皮革夹克，丝绸围巾，缝线/stitched seam",
            "suggest_en": "Add clothing material terms (leather/silk/latexamples/denim/wool/cotton) so outfits don’t drift.",
            "suggest_zh": "加入服装材质词（leather/silk/latexamples/denim/wool/cotton），让穿搭更稳定不漂移。",
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
            "examples_en": "cyberpunk illustration, noir lighting, grainy film look",
            "examples_zh": "赛博朋克插画，noir 光效，grainy film 风格",
            "suggest_en": "Add explicit art-style tokens (anime/manga/oil painting/cyberpunk/noir/etc.) early in captions.",
            "suggest_zh": "前置明确的艺术风格词（anime/manga/oil painting/cyberpunk/noir 等）。",
        },
        "weapons_props": {
            "terms": [
                "sword",
                "gun",
                "rifle",
                "pistol",
                "knife",
                "bow",
                "arrow",
                "staff",
                "hammer",
                "axe",
                "katana",
                "blaster",
                "laser",
            ],
            "examples_en": "holding a sword, gripping a rifle, carrying a pistol",
            "examples_zh": "手持剑，抓握步枪，携带手枪",
            "suggest_en": "Add the exact weapon/prop word + (ideally) holding/gripping/carrying action.",
            "suggest_zh": "补充精确的武器/道具词，并尽量加上 holding/gripping/carrying 这类动作。",
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
                "cake",
                "pie",
                "coffee",
                "tea",
                "noodles",
                "dumplings",
            ],
            "examples_en": "holding ramen bowl, slice of pizza, coffee cup",
            "examples_zh": "拿着拉面碗，披萨一片，咖啡杯",
            "suggest_en": "Add explicit food/drink words and serving/holding context if relevant.",
            "suggest_zh": "补充具体食物/饮品词，并在需要时加上摆盘/拿取语境。",
        },
        "text_in_image": {
            "terms": ["legible text", "clear text", "readable text", "correct spelling", "sign", "label", "lettering"],
            "examples_en": "legible readable text, clear lettering, correct spelling",
            "examples_zh": "清晰可读文字，正确拼写，清楚的字样",
            "suggest_en": "Need text in image: add legible/clear/readable text + spelling cues early.",
            "suggest_zh": "需要图中文字时：尽量在前段加入 legible/clear/readable text + 拼写相关提示。",
        },
        "foreground_framing": {
            "terms": ["foreground", "in the foreground", "foreground bokeh", "near the camera", "closest"],
            "examples_en": "foreground bokeh, detailed foreground objects",
            "examples_zh": "前景散景，细节丰富的前景物体",
            "suggest_en": "Add explicit foreground framing (foreground bokeh / near the camera).",
            "suggest_zh": "补充明确的前景构图词（foreground bokeh / near the camera）。",
        },
        "background_framing": {
            "terms": ["background", "in the background", "distant background", "backdrop", "sky", "horizon"],
            "examples_en": "detailed background, backdrop, sky and horizon",
            "examples_zh": "详细背景，幕布式背景，天空与地平线",
            "suggest_en": "Add explicit background framing (detailed background/backdrop/sky/horizon).",
            "suggest_zh": "补充明确的背景构图词（detailed background/backdrop/sky/horizon）。",
        },
        "fine_details": {
            "terms": [
                "intricate",
                "ornate",
                "embroidery",
                "stitched",
                "seams",
                "texture",
                "weathered",
                "steam",
                "smoke",
                "fog",
                "dust",
                "particles",
                "micro details",
                "depth of field",
                "bokeh",
            ],
            "examples_en": "intricate textures, stitched seams, micro details, particles, depth of field",
            "examples_zh": "精细纹理，缝线细节，微细节，粒子效果，景深",
            "suggest_en": "Add fine detail anchors (textures/material + particles/DOF/bokeh).",
            "suggest_zh": "加入细节锚点（纹理/材质 + 粒子/景深/散景）。",
        },
    }

    if args.custom_categories_json:
        p = Path(args.custom_categories_json)
        if not p.exists():
            raise SystemExit(f"Not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        cats = data.get("categories", data)
        if isinstance(cats, dict):
            for name, spec in cats.items():
                if isinstance(spec, dict) and isinstance(spec.get("terms", []), list):
                    categories[str(name)] = {"terms": spec["terms"]}

    cap_lower = prompt.lower()

    found: Dict[str, bool] = {}
    for name, spec in categories.items():
        terms = spec.get("terms", [])
        found_any = False
        for t in terms:
            if _term_present(cap_lower, t):
                found_any = True
                break
        found[name] = found_any

    # Missing categories = not present
    missing = [name for name, ok in sorted(found.items(), key=lambda kv: kv[0]) if not ok]
    present = [name for name, ok in sorted(found.items(), key=lambda kv: kv[0]) if ok]

    suggestions_en: Dict[str, str] = {
        name: categories[name].get("suggest_en", "") for name in categories.keys() if name in categories
    }
    out: Dict[str, Any] = {
        "prompt": prompt,
        "present_categories": present,
        "missing_categories": missing,
    }

    # Optional external i18n
    def _load_suggestions(path_str: str) -> Dict[str, Dict[str, str]]:
        p = Path(path_str)
        if not p.exists():
            raise SystemExit(f"Not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        cats = data.get("categories", data)
        if not isinstance(cats, dict):
            raise SystemExit('suggestions-json must be {"categories": {...}} or {...category...} structure.')
        out2: Dict[str, Dict[str, str]] = {}
        for cat, spec in cats.items():
            if isinstance(spec, dict):
                out2[str(cat)] = {str(k): str(v) for k, v in spec.items() if isinstance(v, str)}
        return out2

    lang = str(args.lang or "en").strip()
    normalized_lang = lang.lower()
    is_en = normalized_lang in ("en", "eng", "english")

    custom_suggestions: Dict[str, Dict[str, str]] = {}
    if args.suggestions_json:
        custom_suggestions = _load_suggestions(args.suggestions_json)

    def _pick_suggestion(category_name: str) -> str:
        if custom_suggestions:
            by_lang = custom_suggestions.get(category_name, {})
            if lang in by_lang:
                return by_lang[lang]
            for k, v in by_lang.items():
                if str(k).lower() == normalized_lang:
                    return v
        if is_en:
            return suggestions_en.get(category_name, "")
        # Unknown language: localized generic template
        examples = categories.get(category_name, {}).get("terms", [])
        return generic_suggestion(category_name, examples, normalized_lang)

    if missing:
        out["missing_suggestions"] = {name: _pick_suggestion(name) for name in missing}

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print("Prompt gap scout")
    print(f"- Present: {', '.join(present) if present else '(none)'}")
    print(f"- Missing: {', '.join(missing) if missing else '(none)'}")
    if missing:
        print("\nSuggestions:")
        for name in missing:
            msg = _pick_suggestion(name)
            if msg:
                print(f"- {name}: {msg}")

    if not missing:
        print("\nPASS: prompt covers most tricky categories (by keyword heuristic).")


if __name__ == "__main__":
    main()
