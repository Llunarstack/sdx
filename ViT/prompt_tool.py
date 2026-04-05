#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from ViT.prompt_system import build_prompt_plan

    p = argparse.ArgumentParser(description="ViT prompt breakdown: add vs avoid (negative-in-positive)")
    p.add_argument("--prompt", type=str, default="", help="Prompt string to decompose")
    p.add_argument("--json-in", type=str, default="", help="Optional input JSONL with caption/text fields")
    p.add_argument("--json-out", type=str, default="", help="Optional output JSONL with prompt plan fields")
    p.add_argument("--no-default-avoid", action="store_true", help="Disable built-in avoid defaults")
    args = p.parse_args()

    inject_default = not args.no_default_avoid
    if args.prompt:
        plan = build_prompt_plan(args.prompt, inject_default_avoid=inject_default)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if args.json_in:
        inp = Path(args.json_in)
        outp = Path(args.json_out) if args.json_out else inp.with_name(f"{inp.stem}.prompt_plan.jsonl")
        outp.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with inp.open("r", encoding="utf-8") as rf, outp.open("w", encoding="utf-8") as wf:
            for line in rf:
                t = line.strip()
                if not t:
                    continue
                try:
                    row = json.loads(t)
                except Exception:
                    continue
                cap = row.get("caption") or row.get("text") or ""
                plan = build_prompt_plan(str(cap), inject_default_avoid=inject_default)
                row["vit_prompt_add"] = plan["add"]
                row["vit_prompt_avoid"] = plan["avoid"]
                row["vit_prompt_composed"] = plan["composed_prompt"]
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(f"[ViT] wrote {n} rows -> {outp}")
        return 0

    print("Provide either --prompt or --json-in", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
