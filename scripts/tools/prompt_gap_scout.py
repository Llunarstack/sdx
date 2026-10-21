"""
Prompt gap scout: report which tricky categories (see ``complex_prompt_coverage.CATEGORIES``) a single
prompt/caption misses, with a suggestion for each.

Usage:
  python -m scripts.tools prompt_gap_scout --prompt "..." --lang en
  python -m scripts.tools prompt_gap_scout --prompt-file prompt.txt --lang zh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.tools.complex_prompt_coverage import (
    CATEGORIES,
    load_custom_categories,
    suggestion_picker,
    term_present,
)


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
            "Optional JSON file with localized suggestion text per category: "
            '{"{category}": {"en": "...", "es": "..."}} or {"categories": {...}}. '
            "Missing languages fall back to built-in English or a localized generic template."
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
    categories = dict(CATEGORIES)
    if args.custom_categories_json:
        categories.update(load_custom_categories(args.custom_categories_json))

    cap_lower = prompt.lower()
    present = sorted(name for name, terms in categories.items() if any(term_present(cap_lower, t) for t in terms))
    missing = sorted(set(categories) - set(present))
    pick = suggestion_picker(args.lang, args.suggestions_json, categories)

    out: dict[str, Any] = {"prompt": prompt, "present_categories": present, "missing_categories": missing}
    if missing:
        out["missing_suggestions"] = {name: pick(name) for name in missing}
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print("Prompt gap scout")
    print(f"- Present: {', '.join(present) if present else '(none)'}")
    print(f"- Missing: {', '.join(missing) if missing else '(none)'}")
    if missing:
        print()
        print("Suggestions:")
        for name, msg in out["missing_suggestions"].items():
            if msg:
                print(f"- {name}: {msg}")
    else:
        print()
        print("PASS: prompt covers most tricky categories (by keyword heuristic).")


if __name__ == "__main__":
    main()
