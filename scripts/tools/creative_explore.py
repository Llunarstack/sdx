#!/usr/bin/env python3
"""Preview creative mutations or run creative auto-refine via sample.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--mutate", type=int, default=6, help="Number of prompt variants to show or run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-constraint", action="store_true")
    p.add_argument("--json", action="store_true", help="Print imagination plan as JSON")
    p.add_argument("--run", action="store_true", help="Run sample.py with creative auto-refine")
    p.add_argument("--out", type=str, default="creative_best.png")
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("sample_args", nargs=argparse.REMAINDER)
    args = p.parse_args()

    from frontier.imagination import analyze_imagination

    plan = analyze_imagination(
        args.prompt,
        mutate_count=args.mutate,
        mutate_seed=args.seed,
        random_constraint_seed=args.seed if args.random_constraint else None,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "augmented_prompt": plan.augmented_prompt,
                    "serendipity_dial": plan.serendipity_dial,
                    "cfg_multiplier": plan.cfg_multiplier,
                    "creative_trace": plan.creative_trace,
                    "mutations": plan.mutations,
                },
                indent=2,
            )
        )
    else:
        print(f"Augmented: {plan.augmented_prompt[:200]}...")
        print(f"Serendipity: {plan.serendipity_dial:.3f}  CFG mult: {plan.cfg_multiplier:.3f}")
        print(f"Trace: {', '.join(plan.creative_trace)}")
        print("Mutations:")
        for i, m in enumerate(plan.mutations):
            print(f"  [{i}] {m}")

    if not args.run:
        return 0

    sample_args = list(args.sample_args)
    if sample_args and sample_args[0] == "--":
        sample_args = sample_args[1:]
    cmd = [
        sys.executable,
        str(_REPO / "sample.py"),
        "--frontier-creative",
        "--prompt",
        args.prompt,
        *sample_args,
    ]
    import shutil

    from utils.generation.sample_features import run_creative_refine

    res = run_creative_refine(
        sample_cmd=cmd,
        base_prompt=args.prompt,
        num_candidates=args.candidates,
        seed_base=args.seed,
        out_stem="creative_explore",
        mutate_count=args.mutate,
    )
    shutil.copy(res.outputs[res.best_index], args.out)
    print(f"Best {res.best_index} score={res.scores[res.best_index]:.3f} -> {args.out}")
    if res.prompts:
        print(f"Winning prompt: {res.prompts[res.best_index]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
