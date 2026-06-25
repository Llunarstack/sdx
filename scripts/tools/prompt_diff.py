#!/usr/bin/env python3
"""Semantic diff between two prompts (subjects, negations, layout tokens)."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict


def _parse(prompt: str) -> Dict[str, Any]:
    from models.prompt_adherence import PromptParser

    parsed = PromptParser().parse(prompt)
    subjects = [getattr(s, "text", str(s)) for s in getattr(parsed, "subjects", []) or []]
    negations = [getattr(n, "text", str(n)) for n in getattr(parsed, "negations", []) or []]
    return {
        "subjects": subjects,
        "negations": negations,
        "raw": prompt,
    }


def diff_prompts(a: str, b: str) -> Dict[str, Any]:
    pa, pb = _parse(a), _parse(b)
    sa, sb = set(pa["subjects"]), set(pb["subjects"])
    na, nb = set(pa["negations"]), set(pb["negations"])
    return {
        "only_in_a": sorted(sa - sb),
        "only_in_b": sorted(sb - sa),
        "shared_subjects": sorted(sa & sb),
        "negations_only_in_a": sorted(na - nb),
        "negations_only_in_b": sorted(nb - na),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt_a", type=str)
    p.add_argument("prompt_b", type=str, nargs="?", default="")
    p.add_argument("--file-b", type=str, default="", help="Read second prompt from file")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    pb = args.prompt_b
    if args.file_b:
        from pathlib import Path

        pb = Path(args.file_b).read_text(encoding="utf-8").strip()
    result = diff_prompts(args.prompt_a, pb)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, val in result.items():
            if val:
                print(f"{key}:")
                for line in val:
                    print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
