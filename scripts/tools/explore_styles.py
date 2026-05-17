#!/usr/bin/env python3
"""
Invent style genomes and write an explore manifest for batch generation.

Examples::

    python -m scripts.tools.explore_styles \\
        --prompt "a lone samurai at sunset" \\
        --genomes 3 --mutations 1

    python -m scripts.tools explore_styles --prompt "void priest" --mode apocalypse --chaos 1.0

    python -m scripts.tools explore_styles --list-presets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "native" / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Invent StyleGenomes and build explore manifests.")
    parser.add_argument("--prompt", type=str, default="", help="Base scene/subject prompt")
    parser.add_argument("--negative", type=str, default="", help="Base negative prompt")
    parser.add_argument("--genomes", type=int, default=3, help="Number of invented genomes")
    parser.add_argument(
        "--mutations",
        type=int,
        default=0,
        help="Extra prompt mutations per genome (utils.prompt.prompt_mutation)",
    )
    parser.add_argument("--creativity", type=float, default=0.75, help="0–1 invention novelty")
    parser.add_argument(
        "--chaos",
        type=float,
        default=0.0,
        help="0–1 chaos spice (wild axis injection)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=("normal", "insane", "apocalypse", "chimera", "glitch", "eldritch", "cyberpunk"),
        help="Invention mode",
    )
    parser.add_argument("--preset", type=str, default="", help="Named insane preset (see --list-presets)")
    parser.add_argument("--fusion", action="store_true", help="Add chimera fusion candidates")
    parser.add_argument(
        "--hypermutate",
        type=int,
        default=0,
        metavar="N",
        help="Add N hypermutated siblings per genome",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-qwen", action="store_true", help="Deterministic fallback inventor only")
    parser.add_argument(
        "--out",
        type=str,
        default="data/style_genomes/explore_manifest.jsonl",
        help="JSONL manifest (caption, negative, style, genome metadata)",
    )
    parser.add_argument(
        "--genomes-json",
        type=str,
        default="",
        help="Optional path to write full genome JSON array",
    )
    parser.add_argument(
        "--preview-stack",
        action="store_true",
        help="Run PromptStack preview on first candidate (no image generation)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print insane preset ids and exit",
    )
    parser.add_argument(
        "--insane",
        action="store_true",
        help="Shorthand: --mode apocalypse --chaos 0.95 --genomes 4 --fusion --hypermutate 1",
    )
    parser.add_argument(
        "--native-stats",
        action="store_true",
        help="After write, print Go explore-stats (requires built sdx-manifest).",
    )
    parser.add_argument(
        "--dedupe-out",
        type=str,
        default="",
        help="Dedupe manifest by --dedupe-key via Go (requires sdx-manifest).",
    )
    parser.add_argument(
        "--dedupe-key",
        type=str,
        default="style_genome_id",
        help="Field for --dedupe-out",
    )
    parser.add_argument(
        "--show-native",
        action="store_true",
        help="Print Rust/CUDA/Go/Mojo style native discovery JSON and exit.",
    )
    args = parser.parse_args()

    if args.show_native:
        import json

        from utils.prompt.style_native import native_stack_status

        print(json.dumps(native_stack_status(), indent=2))
        return 0

    if args.list_presets:
        from utils.prompt.style_genome_chaos import list_insane_presets

        for pid in list_insane_presets():
            print(pid)
        return 0

    if not (args.prompt or "").strip():
        parser.error("--prompt is required (unless --list-presets)")

    if args.insane:
        args.mode = "apocalypse"
        args.chaos = max(args.chaos, 0.95)
        args.genomes = max(args.genomes, 4)
        args.fusion = True
        args.hypermutate = max(args.hypermutate, 1)

    from utils.prompt.style_explore import StyleExplorePlanner

    planner = StyleExplorePlanner(device="cpu", use_qwen=not args.no_qwen)
    plan = planner.plan(
        args.prompt,
        n_genomes=max(1, args.genomes),
        mutations_per_genome=max(0, args.mutations),
        base_negative=args.negative,
        creativity_level=max(0.0, min(1.0, args.creativity)),
        seed=args.seed,
        invention_mode=args.mode,  # type: ignore[arg-type]
        chaos_level=max(0.0, min(1.0, args.chaos)),
        fusion_pairs=bool(args.fusion),
        hypermutate_siblings=max(0, args.hypermutate),
        preset=args.preset,
    )

    out_path = Path(args.out)
    plan.write_manifest(out_path)
    print(f"Wrote {len(plan.candidates)} candidates -> {out_path} [{args.mode} chaos={args.chaos:.2f}]")

    if args.native_stats:
        from utils.prompt.style_native import run_go_explore_stats

        stats = run_go_explore_stats(out_path)
        if stats:
            print(stats)
        else:
            print("Go explore-stats unavailable (build native/go/sdx-manifest)", file=sys.stderr)

    if args.dedupe_out:
        from utils.prompt.style_native import run_go_explore_dedupe

        ok = run_go_explore_dedupe(out_path, Path(args.dedupe_out), key=args.dedupe_key)
        if not ok:
            print("Go explore-dedupe failed or sdx-manifest not built", file=sys.stderr)
        else:
            print(f"Deduped -> {args.dedupe_out}")

    if args.genomes_json:
        plan.write_genomes_json(Path(args.genomes_json))
        print(f"Wrote genomes -> {args.genomes_json}")

    for i, cand in enumerate(plan.candidates[:12]):
        print(
            f"\n[{i}] {cand.genome.name} ({cand.candidate_kind}, overlap={cand.catalog_overlap:.2f}) "
            f"mut={cand.mutation_strategy or '-'}"
        )
        print(f"  style: {cand.style[:110]}")
        print(f"  pos:   {cand.positive[:130]}...")

    if args.preview_stack and plan.candidates:
        from types import SimpleNamespace

        from utils.prompt.stack import PromptContext, StackMode, run_prompt_stack

        c0 = plan.candidates[0]
        ns = SimpleNamespace(
            prompt=c0.positive,
            negative_prompt=c0.negative,
            style=c0.style,
            invent_styles=0,
            style_genome_file="",
            style_genome_enabled=True,
            _active_style_genome=c0.genome,
            style_chaos_level=args.chaos,
            shortcomings_mitigation="none",
            art_guidance_mode="none",
            anatomy_guidance="none",
            style_guidance_mode="none",
            auto_content_fix=False,
            prompt_stack_intelligence=True,
            prompt_stack_auto_quality=True,
            prompt_clauses="style.chaos,style.surreal" if args.chaos > 0.5 else "",
            prompt_special_helpers="off",
            one_shot_boost=False,
            text_in_image=False,
            no_neg_filter=False,
        )
        ctx = PromptContext(positive=c0.positive, negative=c0.negative, mode=StackMode.PREVIEW, args=ns)
        result = run_prompt_stack(ctx)
        print("\nPreview stack trace:", " -> ".join(result.trace))
        print("Preview positive:", result.positive[:220])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
