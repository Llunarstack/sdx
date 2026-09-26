#!/usr/bin/env python3
"""Invention Lab CLI — diagnose failures + apply novel T2I modules."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SDX Invention Lab")
    p.add_argument("--prompt", default="")
    p.add_argument("--negative", default="")
    p.add_argument("--enable", default="auto", help="auto|all|off|bindlock,negatron,...")
    p.add_argument("--work-dir", default="outputs/inventions")
    p.add_argument("--diagnose-only", action="store_true")
    p.add_argument("--self-heal", action="store_true", help="Emit #100 self-healing plan")
    p.add_argument("--atlas-out", default="", help="Write FailureAtlas JSONL path")
    p.add_argument("--inventory", action="store_true", help="Print 100-invention status counts")
    p.add_argument("--print-argv", action="store_true")
    p.add_argument("--list-ideas", action="store_true", help="Print path to 100-idea bank")
    args = p.parse_args(argv)

    if args.list_ideas:
        root = Path(__file__).resolve().parents[3]
        doc = root / "docs" / "research" / "INVENTION_100.md"
        print(doc)
        return 0

    if args.inventory:
        from utils.generation.inventions.registry import INVENTIONS, inventory

        print(json.dumps({"count": len(INVENTIONS), "by_status": inventory()}, indent=2))
        return 0

    if args.atlas_out:
        from utils.generation.inventions.composition_ext import save_failure_atlas_jsonl

        path = save_failure_atlas_jsonl(args.atlas_out)
        print(f"FailureAtlas -> {path}")
        return 0

    from utils.generation.inventions.failure_oracle import diagnose_failures, repair_plan_from_failures
    from utils.generation.inventions.stack import apply_invention_stack

    if args.self_heal:
        if not args.prompt:
            p.error("--prompt required for --self-heal")
        from utils.generation.inventions.systems_ext import plan_self_healing, run_self_healing_plan

        plan = plan_self_healing(args.prompt, args.negative)
        path = run_self_healing_plan(plan, work_dir=args.work_dir)
        print(json.dumps(plan.to_dict(), indent=2)[:2000])
        print(f"wrote {path}")
        if args.print_argv:
            print("sample argv:")
            print(" ", " ".join(shlex.quote(a) for a in plan.sample_argv))
        return 0

    if args.diagnose_only:
        if not args.prompt:
            p.error("--prompt required for diagnose")
        rep = repair_plan_from_failures(diagnose_failures(args.prompt, negative=args.negative))
        print(json.dumps(rep.to_dict(), indent=2))
        return 0

    if not args.prompt:
        p.error("--prompt is required unless --inventory / --atlas-out / --list-ideas")

    res = apply_invention_stack(
        args.prompt,
        args.negative,
        enable=args.enable,
        work_dir=args.work_dir,
    )
    out = Path(args.work_dir) / "invention_stack.json"
    print(f"positive: {res.positive[:240]}...")
    print(f"negative: {res.negative[:200]}...")
    print(f"risks: {[r for r in (res.reports.get('oracle') or {}).get('risks', [])]}")
    print(f"wrote: {out}")
    if args.print_argv:
        argv_list = ["--prompt", res.positive]
        if res.negative:
            argv_list.extend(["--negative-prompt", res.negative])
        if res.box_layout:
            box = Path(args.work_dir) / "invention_box_layout.json"
            argv_list.extend(["--box-layout", str(box), "--anti-bleed"])
        print("sample argv:")
        print(" ", " ".join(shlex.quote(a) for a in argv_list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
