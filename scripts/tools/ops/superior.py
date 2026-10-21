#!/usr/bin/env python3
"""Superior Stack commands: generate, ensemble, curate, report, dpo, auto-loop, flywheel.

    python -m scripts.tools superior generate --ckpt best.pt --prompt "cyberpunk alley at night" --num 4
    python -m scripts.tools superior ensemble --checkpoints a.pt b.pt --prompt "..."
    python -m scripts.tools superior curate data/train.jsonl --out data/curated.jsonl
    python -m scripts.tools superior report bench_out/
    python -m scripts.tools superior dpo --benchmark-json bench/results.json --ckpt best.pt
    python -m scripts.tools superior auto-loop --base-ckpt best.pt --iterations 2
    python -m scripts.tools superior flywheel --base-ckpt best.pt --manifest-in data/train.jsonl

The old names (``superior_generate``, ``superior_ensemble``, ``superior_curate``, ``superior_eval_report``,
``superior_dpo_loop``, ``superior_auto_loop``, ``run_flywheel``) still work as aliases.
For art self-improvement with a held-out gate, use ``rsi_loop``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _generate(args: argparse.Namespace) -> int:
    from utils.superior.inference_pipeline import SuperiorInferenceConfig, build_superior_sample_argv

    cfg = SuperiorInferenceConfig(
        num_candidates=max(1, args.num),
        local_rag_jsonl=args.local_rag_jsonl or None,
        rag_top_k=args.local_rag_top_k,
        self_correct_clip=bool(args.self_correct),
        compile_inference=bool(args.compile_inference),
        use_composite_rank=True,
    )
    argv = build_superior_sample_argv(ckpt=args.ckpt, prompt=args.prompt, out=args.out, config=cfg)
    if args.expand_prompt:
        argv.append("--expand-prompt")
    cmd = [sys.executable, str(_REPO / "sample.py"), *argv]
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(_REPO), check=False).returncode


def _ensemble(args: argparse.Namespace) -> int:
    from utils.superior.inference_pipeline import EnsembleConfig, generate_ensemble

    cfg = EnsembleConfig(
        checkpoints=args.checkpoints,
        num_per_ckpt=max(1, args.num_per_ckpt),
        vit_ckpt=args.vit_ckpt,
        local_rag_jsonl=args.local_rag_jsonl,
    )
    best, listing = generate_ensemble(args.prompt, cfg, out=args.out, repo_root=_REPO, dry_run=bool(args.dry_run))
    if best:
        print(f"Best: {best}")
        for ckpt, _path, score in listing:
            print(f"  {ckpt}: {score:.4f}")
    return 0


def _curate(args: argparse.Namespace) -> int:
    from utils.data_quality import FilterConfig, filter_jsonl_file

    cfg = FilterConfig(
        dedup=args.dedup,
        min_caption_len=args.min_caption_len,
        max_caption_len=args.max_caption_len,
        min_weight=args.min_weight,
        min_clip_sim=args.min_clip_sim,
        min_aesthetic_proxy=args.min_aesthetic_proxy,
        bad_words=tuple(x.strip().lower() for x in args.bad_words.split(",") if x.strip()),
        image_root=Path(args.image_root) if args.image_root else None,
    )
    _rows, stats = filter_jsonl_file(args.input, config=cfg, output_path=args.out)
    print(
        f"kept={stats.kept} dropped={stats.dropped_total} "
        f"(dup={stats.dropped_dup} caption={stats.dropped_caption} weight={stats.dropped_weight})"
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    from utils.superior.auto_loop import build_markdown_report, write_report

    bench = Path(args.bench_dir)
    out = Path(args.out) if args.out else bench / "REPORT.md"
    write_report(bench, out)
    print(build_markdown_report(bench))
    print(f"Wrote {out}")
    return 0


def _dpo(args: argparse.Namespace) -> int:
    from utils.superior.auto_loop import DPOStageConfig, MinePairsConfig, run_alignment_loop

    summary = run_alignment_loop(
        benchmark_json=args.benchmark_json,
        base_ckpt=args.ckpt,
        pairs_jsonl=args.pairs_jsonl,
        dpo_out=args.out,
        mine=MinePairsConfig(
            benchmark_json=args.benchmark_json, out_jsonl=args.pairs_jsonl, min_margin=args.min_margin
        ),
        dpo=DPOStageConfig(ckpt=args.ckpt, preference_jsonl=args.pairs_jsonl, out=args.out, steps=args.steps),
        repo_root=_REPO,
        dry_run=args.dry_run,
    )
    print(summary)
    return int(summary.get("dpo_exit_code", 0))


def _auto_loop(args: argparse.Namespace) -> int:
    from utils.superior.auto_loop import AutoImproveConfig, run_auto_improve

    cfg = AutoImproveConfig(
        base_ckpt=args.base_ckpt,
        work_dir=args.work_dir,
        vit_ckpt=args.vit_ckpt,
        local_rag_jsonl=args.local_rag_jsonl,
        iterations=args.iterations,
        model_soup=bool(args.model_soup),
        pick_best="superior_composite",
        num=4,
    )
    return run_auto_improve(cfg, repo_root=_REPO, dry_run=bool(args.dry_run))


def _flywheel(args: argparse.Namespace) -> int:
    from config.defaults.superior_stack import FlywheelPlan, SuperiorStackDefaults
    from utils.superior.auto_loop import run_flywheel

    plan = FlywheelPlan(
        base_ckpt=args.base_ckpt,
        work_dir=args.work_dir,
        manifest_in=args.manifest_in or None,
        manifest_out=args.manifest_out or None,
        local_rag_jsonl=args.local_rag_jsonl or None,
        vit_ckpt=args.vit_ckpt or None,
        promote_path=args.promote_path or str(Path(args.work_dir) / "best.pt"),
        skip_curate=bool(args.skip_curate) or not args.manifest_in,
        skip_align=bool(args.skip_align),
        defaults=SuperiorStackDefaults(auto_loop_iterations=max(1, args.iterations)),
    )
    summary = run_flywheel(plan, repo_root=_REPO, dry_run=bool(args.dry_run))
    print(summary)
    return 0 if summary.get("status") != "align_failed" else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="sample.py with local RAG + multi-candidate + composite pick-best")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="superior_out.png")
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--local-rag-jsonl", default="", help="JSONL fact corpus for TF-IDF RAG")
    p.add_argument("--local-rag-top-k", type=int, default=8)
    p.add_argument("--self-correct", action="store_true")
    p.add_argument("--compile-inference", action="store_true")
    p.add_argument("--expand-prompt", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=_generate)

    p = sub.add_parser("ensemble", help="Generate from several checkpoints; keep the global best")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="ensemble_best.png")
    p.add_argument("--num-per-ckpt", type=int, default=2)
    p.add_argument("--vit-ckpt", default="")
    p.add_argument("--local-rag-jsonl", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=_ensemble)

    p = sub.add_parser("curate", help="Filter a training JSONL with utils/data_quality.py")
    p.add_argument("input", help="Input manifest JSONL")
    p.add_argument("--out", required=True)
    p.add_argument("--dedup", choices=["", "phash", "md5"], default="phash")
    p.add_argument("--min-caption-len", type=int, default=8)
    p.add_argument("--max-caption-len", type=int, default=512)
    p.add_argument("--min-weight", type=float, default=0.0)
    p.add_argument("--min-clip-sim", type=float, default=0.0)
    p.add_argument("--min-aesthetic-proxy", type=float, default=0.0)
    p.add_argument("--bad-words", default="")
    p.add_argument("--image-root", default="")
    p.set_defaults(fn=_curate)

    p = sub.add_parser("report", help="Markdown report from a benchmark_suite output directory")
    p.add_argument("bench_dir", help="Directory with results.json + leaderboard.json")
    p.add_argument("--out", default="", help="Output .md (default: bench_dir/REPORT.md)")
    p.set_defaults(fn=_report)

    p = sub.add_parser("dpo", help="Mine benchmark pairs, then train Diffusion-DPO")
    p.add_argument("--benchmark-json", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pairs-jsonl", default="prefs_mined.jsonl")
    p.add_argument("--out", default="dpo_policy.pt")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--min-margin", type=float, default=0.08)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=_dpo)

    p = sub.add_parser("auto-loop", help="auto_improve_loop with Superior defaults (ViT mine, RAG, soup)")
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--work-dir", default="superior_auto_loop")
    p.add_argument("--vit-ckpt", default="", help="ViT quality checkpoint for mining/scoring")
    p.add_argument("--local-rag-jsonl", default="")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--model-soup", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=_auto_loop)

    p = sub.add_parser("flywheel", help="Curate + align + promote")
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--work-dir", default="flywheel_run")
    p.add_argument("--manifest-in", default="", help="Training JSONL to curate")
    p.add_argument("--manifest-out", default="", help="Curated JSONL output")
    p.add_argument("--local-rag-jsonl", default="")
    p.add_argument("--vit-ckpt", default="")
    p.add_argument("--promote-path", default="")
    p.add_argument("--skip-curate", action="store_true")
    p.add_argument("--skip-align", action="store_true")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=_flywheel)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
