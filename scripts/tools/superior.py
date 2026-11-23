#!/usr/bin/env python3
"""Self-improvement stack: Superior generation, preference mining, DPO loops, RSI and human feedback.

    python -m scripts.tools superior generate --ckpt best.pt --prompt "cyberpunk alley at night" --num 4
    python -m scripts.tools superior ensemble --checkpoints a.pt b.pt --prompt "..."
    python -m scripts.tools superior curate data/train.jsonl --out data/curated.jsonl
    python -m scripts.tools superior report bench_out/
    python -m scripts.tools superior dpo --benchmark-json bench/results.json --ckpt best.pt
    python -m scripts.tools superior auto-loop --base-ckpt best.pt --iterations 2
    python -m scripts.tools superior flywheel --base-ckpt best.pt --manifest-in data/train.jsonl
    python -m scripts.tools superior auto-improve --base-ckpt best.pt --iterations 2 --promote-best
    python -m scripts.tools superior mine-pairs --results-json bench/results.json --out-jsonl prefs.jsonl
    python -m scripts.tools superior prefs --from-pick-best runs/pick_best.json --out data/prefs.jsonl
    python -m scripts.tools superior feedback like outputs/out.png --prompt "a red cube"
    python -m scripts.tools superior rsi --base-ckpt results/run/best.pt --rounds 5

Old names still work: ``superior_generate``, ``superior_ensemble``, ``superior_curate``, ``superior_eval_report``,
``superior_dpo_loop``, ``superior_auto_loop``, ``run_flywheel``, ``auto_improve_loop``, ``mine_preference_pairs``,
``preference_flywheel``, ``feedback`` / ``feedback_cli``, ``rsi_loop``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- superior subcommands


def _generate(args: argparse.Namespace) -> int:
    from utils.agentic.quality_loop import SuperiorInferenceConfig, build_superior_sample_argv

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
    cmd = [sys.executable, str(ROOT / "sample.py"), *argv]
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def _ensemble(args: argparse.Namespace) -> int:
    from utils.agentic.quality_loop import EnsembleConfig, generate_ensemble

    cfg = EnsembleConfig(
        checkpoints=args.checkpoints,
        num_per_ckpt=max(1, args.num_per_ckpt),
        vit_ckpt=args.vit_ckpt,
        local_rag_jsonl=args.local_rag_jsonl,
    )
    best, listing = generate_ensemble(args.prompt, cfg, out=args.out, repo_root=ROOT, dry_run=bool(args.dry_run))
    if best:
        print(f"Best: {best}")
        for ckpt, _path, score in listing:
            print(f"  {ckpt}: {score:.4f}")
    return 0


def _curate(args: argparse.Namespace) -> int:
    from utils.quality.critics import FilterConfig, filter_jsonl_file

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
    from utils.agentic.quality_loop import build_markdown_report, write_report

    bench = Path(args.bench_dir)
    out = Path(args.out) if args.out else bench / "REPORT.md"
    write_report(bench, out)
    print(build_markdown_report(bench))
    print(f"Wrote {out}")
    return 0


def _dpo(args: argparse.Namespace) -> int:
    from utils.agentic.quality_loop import DPOStageConfig, MinePairsConfig, run_alignment_loop

    summary = run_alignment_loop(
        benchmark_json=args.benchmark_json,
        base_ckpt=args.ckpt,
        pairs_jsonl=args.pairs_jsonl,
        dpo_out=args.out,
        mine=MinePairsConfig(
            benchmark_json=args.benchmark_json, out_jsonl=args.pairs_jsonl, min_margin=args.min_margin
        ),
        dpo=DPOStageConfig(ckpt=args.ckpt, preference_jsonl=args.pairs_jsonl, out=args.out, steps=args.steps),
        repo_root=ROOT,
        dry_run=args.dry_run,
    )
    print(summary)
    return int(summary.get("dpo_exit_code", 0))


def _auto_loop(args: argparse.Namespace) -> int:
    from utils.agentic.quality_loop import AutoImproveConfig, run_auto_improve

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
    return run_auto_improve(cfg, repo_root=ROOT, dry_run=bool(args.dry_run))


def _flywheel(args: argparse.Namespace) -> int:
    from config.presets import FlywheelPlan, SuperiorStackDefaults
    from utils.agentic.quality_loop import run_flywheel

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
    summary = run_flywheel(plan, repo_root=ROOT, dry_run=bool(args.dry_run))
    print(summary)
    return 0 if summary.get("status") != "align_failed" else 1


# ---------------------------------------------------------------- mine-pairs (benchmark results -> DPO JSONL)


def mine_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    min_margin: float = 0.08,
    max_pairs_per_case: int = 2,
    require_existing_files: bool = True,
    hard_case_names: set[str] | None = None,
    hardcase_extra_pairs: int = 0,
    hardcase_min_margin_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Per (case, prompt) group, pair the top-scored output with the lowest ones while the margin holds.

    Rows are benchmark ``results.json`` entries (``case``, ``prompt``, ``output``, ``composite``); output rows
    are ``win_image_path`` / ``lose_image_path`` / ``caption`` for ``train_diffusion_dpo``.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = (str(r.get("case", "") or "").strip(), str(r.get("prompt", "") or "").strip())
        if key[1]:
            groups.setdefault(key, []).append(r)

    out: list[dict[str, Any]] = []
    for (case, prompt), items in groups.items():
        is_hardcase = case in (hard_case_names or set())
        eff_max_pairs = int(max_pairs_per_case) + (int(hardcase_extra_pairs) if is_hardcase else 0)
        eff_min_margin = float(min_margin) * (float(hardcase_min_margin_scale) if is_hardcase else 1.0)
        scored = []
        for x in items:
            try:
                s = float(x.get("composite", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            p = str(x.get("output", "") or "").strip()
            if not p or (require_existing_files and not Path(p).is_file()):
                continue
            scored.append((s, x))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda t: t[0], reverse=True)
        hi = scored[0]
        used = 0
        for lo in reversed(scored[1:]):
            margin = float(hi[0] - lo[0])
            if margin < eff_min_margin:
                continue
            out.append(
                {
                    "win_image_path": str(hi[1]["output"]),
                    "lose_image_path": str(lo[1]["output"]),
                    "caption": prompt,
                    "case": case,
                    "win_score": float(hi[0]),
                    "lose_score": float(lo[0]),
                    "margin": margin,
                    "source": "benchmark_suite",
                    "winner_model": str(hi[1].get("model", "") or ""),
                    "loser_model": str(lo[1].get("model", "") or ""),
                    "is_hardcase": bool(is_hardcase),
                }
            )
            used += 1
            if used >= eff_max_pairs:
                break
    return out


def _read_hardcase_names(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and str(row.get("case", "") or "").strip():
            out.add(str(row["case"]).strip())
    return out


def _mine_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="superior mine-pairs", description="Mine DPO preference pairs from benchmark results.json."
    )
    ap.add_argument("--results-json", type=str, required=True, help="benchmark_suite results.json path")
    ap.add_argument("--out-jsonl", type=str, default="data/prefs_from_benchmark.jsonl")
    ap.add_argument("--min-margin", type=float, default=0.08)
    ap.add_argument("--max-pairs-per-case", type=int, default=2)
    ap.add_argument("--allow-missing-files", action="store_true")
    ap.add_argument("--hardcases-jsonl", type=str, default="", help="Optional hard-case JSONL from benchmark_suite.")
    ap.add_argument("--hardcase-extra-pairs", type=int, default=0, help="Additional pairs per hard-case group.")
    ap.add_argument(
        "--hardcase-min-margin-scale",
        type=float,
        default=1.0,
        help="Scale min margin for hard-cases (e.g. 0.75 mines more hard-case pairs).",
    )
    args = ap.parse_args(argv)

    data = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("results json must be a list")
    hard = str(args.hardcases_jsonl).strip()
    rows = mine_pairs(
        data,
        min_margin=float(args.min_margin),
        max_pairs_per_case=int(args.max_pairs_per_case),
        require_existing_files=not bool(args.allow_missing_files),
        hard_case_names=_read_hardcase_names(Path(hard)) if hard else set(),
        hardcase_extra_pairs=int(args.hardcase_extra_pairs),
        hardcase_min_margin_scale=float(args.hardcase_min_margin_scale),
    )
    out_path = Path(args.out_jsonl)
    _write_jsonl(out_path, rows)
    print(f"[mine_preference_pairs] wrote {len(rows)} pairs -> {out_path}")
    return 0


# ---------------------------------------------------------------- prefs (preference flywheel)


def _rows_from_pick_best(path: Path) -> list[dict]:
    """Normalize pick-best dumps (``{"prompt", "candidates": [{"path", "score"}]}`` or a list) into mine rows."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for i, obj in enumerate(raw if isinstance(raw, list) else [raw]):
        if not isinstance(obj, dict):
            continue
        prompt = str(obj.get("prompt") or obj.get("caption") or "").strip()
        case = str(obj.get("case") or f"pick_{i}")
        for c in obj.get("candidates") or obj.get("images") or []:
            if not isinstance(c, dict):
                continue
            out = str(c.get("path") or c.get("output") or c.get("image") or "").strip()
            score = c.get("score", c.get("composite", c.get("metric")))
            if out and score is not None:
                rows.append({"case": case, "prompt": prompt, "output": out, "composite": float(score)})
    return rows


def _prefs_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="superior prefs",
        description="Mine preference pairs for Diffusion-DPO from feedback, pick-best dumps and/or benchmark results.",
    )
    ap.add_argument("--from-pick-best", type=str, default="", help="pick-best JSON dump")
    ap.add_argument("--from-feedback", type=str, default="", help="feedback JSONL (human pairs merged first)")
    ap.add_argument("--from-benchmark", type=str, default="", help="benchmark results.json")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--min-margin", type=float, default=0.08)
    ap.add_argument("--max-pairs-per-case", type=int, default=2)
    args = ap.parse_args(argv)

    pairs: list[dict] = []
    if args.from_feedback:
        from utils.training.preference import pairs_from_feedback

        pairs.extend(pairs_from_feedback(Path(args.from_feedback)))
    rows: list[dict] = []
    if args.from_pick_best:
        rows.extend(_rows_from_pick_best(Path(args.from_pick_best)))
    if args.from_benchmark:
        data = json.loads(Path(args.from_benchmark).read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict) and "results" in data:
            rows.extend(data["results"])
    if not rows and not pairs:
        print("No rows loaded. Pass --from-feedback, --from-pick-best, and/or --from-benchmark.", file=sys.stderr)
        return 2
    if rows:
        pairs.extend(mine_pairs(rows, min_margin=args.min_margin, max_pairs_per_case=args.max_pairs_per_case))
    if not pairs:
        print("No preference pairs produced.", file=sys.stderr)
        return 2
    out = Path(args.out)
    _write_jsonl(out, pairs)
    print(f"Wrote {len(pairs)} pairs → {out}", file=sys.stderr)
    print(
        f"Next: python -m scripts.tools train_diffusion_dpo --ckpt <ckpt> --preference-jsonl {out}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------- auto-improve (benchmark -> prefs -> DPO -> benchmark)


def _read_leaderboard(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def _best_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    return max(rows, key=lambda r: float(r.get("mean_composite", 0.0) or 0.0)) if rows else None


def _best_model_tag(rows: list[dict[str, object]]) -> str:
    best = _best_row(rows)
    return str(best.get("model", "") or "").strip() if best else ""


def _resolve_ckpt_by_tag(ckpts: list[Path], tag: str) -> Path | None:
    t = (tag or "").strip().lower()
    return next((p for p in ckpts if t and p.stem.lower() == t), None)


def _iter_dir(work: Path, it: int, total: int) -> Path:
    return work if total <= 1 else work / f"iter_{it:03d}"


def _bench_argv(args: argparse.Namespace, ckpts: list[Path], out_dir: Path) -> list[str]:
    argv = ["--ckpt", *[str(p) for p in ckpts], "--out-dir", str(out_dir)]
    argv += ["--preset", str(args.preset), "--device", str(args.device), "--suite-pack", str(args.suite_pack)]
    argv += ["--steps", str(int(args.steps)), "--seed-list", str(args.seed_list), "--num", str(int(args.num))]
    argv += ["--pick-best", str(args.pick_best), "--robustness-penalty", str(float(args.robustness_penalty))]
    if str(args.local_rag_jsonl or "").strip():
        argv += ["--local-rag-jsonl", str(args.local_rag_jsonl).strip()]
    return argv


def _auto_improve_main(argv: list[str]) -> int:
    from utils.agentic.quality_loop import run_tool

    ap = argparse.ArgumentParser(
        prog="superior auto-improve",
        description="Benchmark -> mine preferences -> Diffusion-DPO -> re-benchmark (base vs DPO), "
        "optionally over several iterations, then promote the best checkpoint.",
    )
    ap.add_argument("--base-ckpt", type=str, required=True, help="Starting checkpoint path.")
    ap.add_argument("--work-dir", type=str, default="auto_improve_loop")
    ap.add_argument("--preset", type=str, default="sdxl")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--suite-pack", type=str, default="top_contender_proxy_v1")
    ap.add_argument("--steps", type=int, default=30, help="Benchmark sampling steps.")
    ap.add_argument("--seed-list", type=str, default="", help="Comma-separated seeds used by benchmark_suite.")
    ap.add_argument(
        "--robustness-penalty",
        type=float,
        default=0.15,
        help="Penalty weight for seed variance in leaderboard ranking.",
    )
    ap.add_argument("--num", type=int, default=3, help="Benchmark candidates per case.")
    ap.add_argument("--pick-best", type=str, default="auto")
    ap.add_argument("--preference-min-margin", type=float, default=0.08)
    ap.add_argument("--preference-max-pairs-per-case", type=int, default=2)
    ap.add_argument(
        "--export-hardcases",
        action="store_true",
        help="Export iteration hard-case JSONL from the pre-DPO benchmark stage.",
    )
    ap.add_argument("--hardcase-threshold", type=float, default=0.60)
    ap.add_argument("--hardcase-max-rows", type=int, default=200)
    ap.add_argument(
        "--hardcase-preference-boost",
        type=int,
        default=2,
        help="Additional mined preference pairs per hard-case group.",
    )
    ap.add_argument(
        "--hardcase-margin-scale",
        type=float,
        default=0.75,
        help="Mine more hard-case pairs by scaling min margin (e.g. 0.75).",
    )
    ap.add_argument("--dpo-steps", type=int, default=500)
    ap.add_argument("--dpo-batch-size", type=int, default=2)
    ap.add_argument("--dpo-lr", type=float, default=1e-6)
    ap.add_argument("--dpo-beta", type=float, default=500.0)
    ap.add_argument("--iterations", type=int, default=1, help="How many full loop rounds to run.")
    ap.add_argument("--promote-best", action="store_true", help="Copy best loop checkpoint to --promote-path.")
    ap.add_argument("--promote-path", type=str, default="auto_improve_loop/best_auto.pt")
    ap.add_argument(
        "--vit-ckpt",
        type=str,
        default="",
        help="ViT quality checkpoint for vit-weighted preference mining and benchmark scoring.",
    )
    ap.add_argument(
        "--local-rag-jsonl",
        type=str,
        default="",
        help="JSONL corpus passed to sample.py during benchmark (--local-rag-jsonl).",
    )
    ap.add_argument(
        "--model-soup",
        action="store_true",
        help="After DPO, blend base + DPO checkpoints into soup_policy.pt and benchmark that too.",
    )
    ap.add_argument(
        "--use-hard-negatives",
        action="store_true",
        help="Mine failure tags from pre-DPO benchmark and pass --negative-prompt to post-DPO benchmark.",
    )
    ap.add_argument("--soup-weight-base", type=float, default=0.35)
    ap.add_argument("--soup-weight-dpo", type=float, default=0.65)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    dry = bool(args.dry_run)
    iterations = int(args.iterations)
    if iterations < 1:
        print("--iterations must be >= 1", file=sys.stderr)
        return 2
    work = Path(args.work_dir)
    if not dry:
        work.mkdir(parents=True, exist_ok=True)
    vit = str(args.vit_ckpt or "").strip()

    current_base = Path(args.base_ckpt).resolve()
    best_overall_ckpt: Path | None = current_base
    best_overall_score = float("-inf")

    for it in range(1, iterations + 1):
        it_dir = _iter_dir(work, it, iterations).resolve()
        if not dry:
            it_dir.mkdir(parents=True, exist_ok=True)
        dpo_ckpt = it_dir / "dpo_policy.pt"
        pref_jsonl = it_dir / "prefs.jsonl"
        hardcase_jsonl = it_dir / "hardcases.jsonl"
        bench_before = it_dir / "bench_before"
        bench_after = it_dir / "bench_after"
        print(f"[auto_improve_loop] iteration {it}/{iterations} base={current_base}")

        bench_1 = _bench_argv(args, [current_base], bench_before)
        bench_1 += ["--export-preference-jsonl", str(pref_jsonl)]
        bench_1 += ["--preference-min-margin", str(float(args.preference_min_margin))]
        bench_1 += ["--preference-max-pairs-per-case", str(int(args.preference_max_pairs_per_case))]
        if args.export_hardcases:
            bench_1 += ["--export-hardcases-jsonl", str(hardcase_jsonl)]
            bench_1 += ["--hardcase-threshold", str(float(args.hardcase_threshold))]
            bench_1 += ["--hardcase-max-rows", str(int(args.hardcase_max_rows))]
        if vit:
            bench_1 += ["--pick-vit-ckpt", vit, "--vit-score", "--vit-mine-preferences"]
        rc = run_tool("benchmark_suite", bench_1, repo_root=ROOT, dry_run=dry)
        if rc != 0:
            return rc

        if args.export_hardcases:  # re-mine with extra, looser-margin pairs for hard cases
            remine = ["--results-json", str(bench_before / "results.json"), "--out-jsonl", str(pref_jsonl)]
            remine += ["--min-margin", str(float(args.preference_min_margin))]
            remine += ["--max-pairs-per-case", str(int(args.preference_max_pairs_per_case)), "--allow-missing-files"]
            remine += ["--hardcases-jsonl", str(hardcase_jsonl)]
            remine += ["--hardcase-extra-pairs", str(int(args.hardcase_preference_boost))]
            remine += ["--hardcase-min-margin-scale", str(float(args.hardcase_margin_scale))]
            rc = run_tool("mine_preference_pairs", remine, repo_root=ROOT, dry_run=dry)
            if rc != 0:
                return rc

        dpo_cmd = ["--ckpt", str(current_base), "--preference-jsonl", str(pref_jsonl), "--out", str(dpo_ckpt)]
        dpo_cmd += ["--steps", str(int(args.dpo_steps)), "--batch-size", str(int(args.dpo_batch_size))]
        dpo_cmd += ["--lr", str(float(args.dpo_lr)), "--dpo-beta", str(float(args.dpo_beta))]
        dpo_cmd += ["--device", str(args.device)]
        rc = run_tool("train_diffusion_dpo", dpo_cmd, repo_root=ROOT, dry_run=dry)
        if rc != 0:
            return rc

        ckpts_compare = [current_base, dpo_ckpt]
        if args.model_soup:
            soup_ckpt = it_dir / "soup_policy.pt"
            if not dry:
                from utils.agentic.quality_loop import save_soup_checkpoint, soup_checkpoints

                avg = soup_checkpoints(
                    [current_base, dpo_ckpt], weights=[float(args.soup_weight_base), float(args.soup_weight_dpo)]
                )
                save_soup_checkpoint(avg, template_path=current_base, out_path=soup_ckpt)
                print(f"[auto_improve_loop] model soup -> {soup_ckpt}")
            ckpts_compare.append(soup_ckpt)

        bench_2 = _bench_argv(args, ckpts_compare, bench_after)
        if vit:
            bench_2 += ["--pick-vit-ckpt", vit, "--vit-score"]
        results_path = bench_before / "results.json"
        if args.use_hard_negatives and results_path.is_file() and not dry:
            from utils.agentic.quality_loop import (
                benchmark_sample_args_for_negatives,
                load_hard_negatives_from_results,
            )

            bundle = load_hard_negatives_from_results(results_path, threshold=float(args.hardcase_threshold))
            for tok in benchmark_sample_args_for_negatives(bundle):
                bench_2 += ["--sample-arg", tok]
            if bundle.negative_suffix:
                print(f"[auto_improve_loop] hard negatives: {bundle.negative_suffix[:120]}...")
        rc = run_tool("benchmark_suite", bench_2, repo_root=ROOT, dry_run=dry)
        if rc != 0:
            return rc

        if not dry:
            rows = _read_leaderboard(bench_after / "leaderboard.json")
            br = _best_row(rows)
            iter_best_ckpt = _resolve_ckpt_by_tag(ckpts_compare, _best_model_tag(rows))
            if br is not None and iter_best_ckpt is not None:
                iter_best_score = float(br.get("mean_composite", 0.0) or 0.0)
                print(f"[auto_improve_loop] iter best: {iter_best_ckpt} score={iter_best_score:.4f}")
                current_base = iter_best_ckpt
                if iter_best_score > best_overall_score:
                    best_overall_score = iter_best_score
                    best_overall_ckpt = iter_best_ckpt
            else:
                current_base = dpo_ckpt

    if args.promote_best:
        if dry:
            print(f"Dry-run: would read leaderboard and promote to {args.promote_path}")
            return 0
        best_ckpt = best_overall_ckpt or current_base
        promote_path = Path(args.promote_path)
        promote_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, promote_path)
        print(f"Promoted best checkpoint: {best_ckpt} -> {promote_path}")
    return 0


# ---------------------------------------------------------------- feedback (human ratings for RSI / DPO)


def _feedback_main(argv: list[str]) -> int:
    """like / dislike / pair / pick / export-dpo / sync-taste / status on the append-only feedback log."""
    from utils.training.preference import (
        default_feedback_log,
        export_dpo_jsonl,
        iter_feedback,
        pairs_from_feedback,
        record_dislike,
        record_like,
        record_pair,
        record_pick,
        update_user_taste_from_feedback,
    )

    ap = argparse.ArgumentParser(
        prog="superior feedback", description="Rate generated images/video frames for RSI preference learning."
    )
    ap.add_argument("--log", type=str, default=str(default_feedback_log()), help="Append-only feedback JSONL path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--prompt", type=str, default="")
        p.add_argument("--ckpt", type=str, default="")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--run-id", type=str, default="")
        p.add_argument("--media-type", type=str, default="image", choices=("image", "video"))

    like = sub.add_parser("like", help="Record a liked generation")
    like.add_argument("image", type=str)
    like.add_argument("--score", type=float, default=1.0)
    like.add_argument("--notes", type=str, default="")
    _common(like)

    dislike = sub.add_parser("dislike", help="Record a disliked generation")
    dislike.add_argument("image", type=str)
    dislike.add_argument("--notes", type=str, default="")
    _common(dislike)

    pair = sub.add_parser("pair", help="Record an explicit win/lose pair")
    pair.add_argument("--win", type=str, required=True)
    pair.add_argument("--lose", type=str, required=True)
    _common(pair)

    pick = sub.add_parser("pick", help="User picked one winner among candidates")
    pick.add_argument("--win", type=str, required=True)
    pick.add_argument("--lose", action="append", default=[], help="Loser path (repeatable)")
    _common(pick)

    exp = sub.add_parser("export-dpo", help="Export DPO preference JSONL from feedback")
    exp.add_argument("--out", type=str, required=True)
    exp.add_argument("--allow-cross-prompt", action="store_true")
    exp.add_argument("--no-synthetic", action="store_true", help="Only explicit pair/pick events")

    sub.add_parser("sync-taste", help="Update UserTaste likes_notes/hates from feedback")
    sub.add_parser("status", help="Summarize feedback log")

    args = ap.parse_args(argv)
    log = args.log
    if args.cmd in ("like", "dislike", "pair", "pick"):
        meta = dict(prompt=args.prompt, ckpt=args.ckpt, seed=args.seed, run_id=args.run_id, log_path=log)
        meta["media_type"] = args.media_type
    if args.cmd == "like":
        print(f"like → {record_like(args.image, score=args.score, notes=args.notes, **meta)}")
        return 0
    if args.cmd == "dislike":
        print(f"dislike → {record_dislike(args.image, notes=args.notes, **meta)}")
        return 0
    if args.cmd == "pair":
        print(f"pair → {record_pair(args.win, args.lose, **meta)}")
        return 0
    if args.cmd == "pick":
        paths = record_pick(args.win, list(args.lose or []), **meta)
        print(f"pick → {len(paths)} pairs (+ like) in {log}")
        return 0
    if args.cmd == "export-dpo":
        n = export_dpo_jsonl(
            args.out,
            log_path=log,
            same_prompt_only=not args.allow_cross_prompt,
            include_synthetic=not args.no_synthetic,
        )
        print(f"Wrote {n} pairs → {args.out}")
        print(
            f"Next: python -m scripts.tools train_diffusion_dpo --ckpt <ckpt> --preference-jsonl {args.out}\n"
            "  or let rsi_loop mix your feedback into self-improvement rounds automatically.",
            file=sys.stderr,
        )
        return 0 if n else 2
    if args.cmd == "sync-taste":
        print(f"UserTaste updated → {update_user_taste_from_feedback(log_path=log)}")
        return 0
    counts: dict[str, int] = defaultdict(int)
    n = 0
    for row in iter_feedback(log):
        n += 1
        counts[str(row.get("event") or "?")] += 1
    ready = len(pairs_from_feedback(log))
    print(json.dumps({"log": log, "events": n, "by_event": dict(counts), "dpo_pairs_ready": ready}, indent=2))
    return 0


# ---------------------------------------------------------------- rsi (self-improvement loop)
#
# Every round: build art prompts (medium x subject x light x framing) weighted toward the medium families
# the champion scores worst on; paint --candidates images per prompt through sample.py --jobs-jsonl (one
# model load per round); judge each image (HF reward panel + art critic + ai_look); train a DPO candidate
# on best-vs-worst pairs (+ human feedback pairs); gate it on held-out prompts/seeds. The candidate replaces
# the champion only if the held-out mean rises by --min-gain and it wins >= half the head-to-heads; a
# rejected candidate is deleted and the DPO lr halves. State lives in <work-dir>/state.json (resumable).
# The judge sets HF_HUB_OFFLINE=1 unless already set, so an unattended loop never starts a big download.

RSI_DESCRIPTION = """RSI: an SDX checkpoint teaches itself to make better art (paint -> judge -> DPO -> held-out gate).

    python -m scripts.tools rsi_loop --base-ckpt results/run/best.pt --rounds 5
    python -m scripts.tools rsi_loop --rounds 3                  # resume outputs/rsi
    python -m scripts.tools rsi_loop --status                    # round history
    python -m scripts.tools rsi_loop --base-ckpt ... --dry-run   # print round 1's prompts and commands
"""

TRAIN_SUBJECTS = (
    "a lighthouse on a storm-battered cliff",
    "an old fisherman mending nets",
    "a fox curled up in fresh snow",
    "a crowded night market in the rain",
    "a cathedral interior with stained light",
    "a dancer mid-leap",
    "a still life of pears and a copper jug",
    "a mountain lake at dawn",
    "a dragon coiled around a tower",
    "a cyclist on a coastal road",
    "a cat asleep on a windowsill",
    "a ruined temple overgrown with vines",
    "a jazz trio in a smoky club",
    "a field of sunflowers under a thunderstorm",
    "a portrait of an elderly woman laughing",
    "a sailing ship in heavy seas",
    "a greenhouse full of tropical plants",
    "a knight resting beside a campfire",
    "a koi pond with lily pads",
    "a desert caravan at sunset",
    "a busy train station platform",
    "an owl perched on a mossy branch",
    "a child flying a kite on a hill",
    "a bowl of ramen on a wooden counter",
)
# Held-out subjects never appear in training prompts, so the gate measures art skill, not memorised prompts.
EVAL_SUBJECTS = (
    "a heron wading through reeds",
    "a violinist on a city rooftop",
    "a stone bridge over a misty river",
    "a bakery window at closing time",
    "a wolf howling under the aurora",
    "a lantern-lit alley in an old town",
)
LIGHT = (
    "golden hour light",
    "soft overcast light",
    "dramatic chiaroscuro",
    "cool moonlight",
    "warm candlelight",
    "bright midday sun",
    "neon glow",
    "foggy diffused light",
)
FRAMING = (
    "rule of thirds composition",
    "centered symmetrical composition",
    "wide establishing shot",
    "intimate close-up",
    "low angle view",
    "high angle view",
    "leading lines toward the subject",
    "generous negative space",
)
JUDGE_WEIGHTS = {"reward": 0.45, "art": 0.30, "natural": 0.25}


def medium_families() -> dict[str, list[str]]:
    """Medium family -> medium phrases, straight from ``config.styles``."""
    from config.styles import MEDIUM_SPECS, detect_medium

    out: dict[str, list[str]] = {}
    for spec in MEDIUM_SPECS:
        phrase = spec.keywords[0]
        out.setdefault(detect_medium(phrase).family, []).append(phrase)
    return out


def art_prompt(rng: random.Random, medium: str, subjects: tuple[str, ...]) -> str:
    return f"{rng.choice(subjects)}, {medium}, {rng.choice(LIGHT)}, {rng.choice(FRAMING)}"


def train_prompts(n: int, weights: dict[str, float], rng: random.Random) -> list[tuple[str, str]]:
    """``n`` (family, prompt) rows, families drawn in proportion to ``weights``."""
    fams = medium_families()
    names = sorted(f for f in fams if weights.get(f, 0) > 0)
    picks = rng.choices(names, [weights[f] for f in names], k=n)
    return [(f, art_prompt(rng, rng.choice(fams[f]), TRAIN_SUBJECTS)) for f in picks]


def eval_prompts(per_family: int) -> list[tuple[str, str]]:
    """Fixed held-out set: same prompts every round, so champion and candidate scores compare."""
    rng = random.Random(1234)
    return [
        (fam, art_prompt(rng, rng.choice(media), EVAL_SUBJECTS))
        for fam, media in sorted(medium_families().items())
        for _ in range(per_family)
    ]


def curriculum_weights(family_scores: dict[str, float]) -> dict[str, float]:
    """Weaker families get more practice; unknown families start at full weight."""
    return {f: max(0.05, 1.05 - family_scores.get(f, 0.0)) for f in medium_families()}


class ArtJudge:
    """One art score in [0, 1] per image: reward panel + art critic + generated-look naturalness."""

    def __init__(self, device: str = "cuda") -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from utils.modeling.hf_loaders import HFRewardPanel, HFRewardWeights

        zero = {k: 0.0 for k in HFRewardWeights.__dataclass_fields__}
        self.panel = HFRewardPanel(
            HFRewardWeights(**{**zero, "pickscore": 0.45, "clip_h14": 0.20, "cafe_aesthetic": 0.35}),
            device=device,
        )

    def __call__(self, path: str | Path, prompt: str) -> tuple[float, dict[str, float]]:
        import numpy as np
        from PIL import Image
        from utils.quality.critics import ai_look_score, score_art

        rgb = np.asarray(Image.open(path).convert("RGB"))
        reward = self.panel.score_breakdown(rgb, prompt=prompt)
        parts = {"art": score_art(rgb, prompt=prompt).score, "natural": ai_look_score(rgb, prompt=prompt).natural}
        if reward.parts:  # panel falls back to a flat 0.5 when no scorer loaded; leave it out then
            parts["reward"] = reward.composite
        score = sum(JUDGE_WEIGHTS[k] * v for k, v in parts.items()) / sum(JUDGE_WEIGHTS[k] for k in parts)
        return float(score), {**parts, **reward.parts}

    @staticmethod
    def release() -> None:
        """Free scorer VRAM before the DPO subprocess needs it."""
        from utils.modeling import hf_loaders

        hf_loaders._CACHE.store.clear()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def mine_self_pairs(scored: list[dict[str, Any]], min_margin: float) -> list[dict[str, Any]]:
    """Per prompt, pair best with worst, 2nd best with 2nd worst, ... while the score gap is >= ``min_margin``."""
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_prompt[row["prompt"]].append(row)
    pairs = []
    for prompt, rows in by_prompt.items():
        rows = sorted(rows, key=lambda r: r["score"], reverse=True)
        for i in range(len(rows) // 2):
            win, lose = rows[i], rows[-1 - i]
            margin = win["score"] - lose["score"]
            if margin < min_margin:
                break
            pairs.append(
                {
                    "win_image_path": win["path"],
                    "lose_image_path": lose["path"],
                    "caption": prompt,
                    "source": "rsi_self",
                    "margin": round(margin, 4),
                }
            )
    return pairs


def human_pairs(feedback_log: str | Path) -> list[dict[str, Any]]:
    """Your thumbs/pairs from the feedback bus (image media, files still on disk)."""
    from utils.training.preference import pairs_from_feedback

    if not Path(feedback_log).is_file():
        return []
    return [
        p
        for p in pairs_from_feedback(feedback_log)
        if Path(p["win_image_path"]).is_file() and Path(p["lose_image_path"]).is_file()
    ]


def gate(champion: dict[str, float], candidate: dict[str, float], min_gain: float) -> dict[str, Any]:
    """Paired comparison on identical (seed, prompt) keys."""
    keys = sorted(champion.keys() & candidate.keys())
    if not keys:
        return {"gain": 0.0, "win_rate": 0.0, "n": 0, "promote": False}
    deltas = [candidate[k] - champion[k] for k in keys]
    gain, win_rate = fmean(deltas), fmean(d > 0 for d in deltas)
    return {
        "gain": round(gain, 5),
        "win_rate": round(win_rate, 3),
        "n": len(keys),
        "promote": gain >= min_gain and win_rate >= 0.5,
    }


def family_means(scored: list[dict[str, Any]]) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        by[r["family"]].append(r["score"])
    return {f: round(fmean(v), 4) for f, v in by.items()}


def sdx_generate(ckpt: str, jobs: list[dict[str, Any]], out_dir: Path, opts: argparse.Namespace) -> None:
    """Run every job through ``sample.py`` with one model load. Jobs that fail just leave no file."""
    jobs_path = out_dir / "jobs.jsonl"
    _write_jsonl(jobs_path, jobs)
    cmd = [sys.executable, str(ROOT / "sample.py"), "--ckpt", ckpt, "--jobs-jsonl", str(jobs_path), "--num", "1"]
    cmd += ["--steps", str(opts.steps), "--device", opts.device]
    cmd += ["--no-lora-bank", *opts.sample_arg]  # train the model itself, not whatever adapters the bank would pick
    subprocess.run(cmd, cwd=str(ROOT), check=False)


def sdx_train(ckpt: str, prefs: Path, out: Path, lr: float, opts: argparse.Namespace) -> bool:
    from utils.agentic.quality_loop import run_tool

    argv = ["--ckpt", ckpt, "--preference-jsonl", str(prefs), "--out", str(out), "--steps", str(opts.dpo_steps)]
    argv += ["--batch-size", "2", "--lr", str(lr), "--dpo-beta", str(opts.dpo_beta), "--save-every", "0"]
    argv += ["--device", opts.device]
    return run_tool("train_diffusion_dpo", argv, repo_root=ROOT) == 0 and out.is_file()


def paint_and_judge(
    ckpt: str,
    rows: list[tuple[str, str]],
    seeds: list[int],
    out_dir: Path,
    *,
    generate: Callable[[str, list[dict[str, Any]], Path], None],
    judge: Callable[[str | Path, str], tuple[float, dict[str, float]]],
) -> list[dict[str, Any]]:
    """Paint every (row, seed) and score what came out."""
    jobs = [
        {"prompt": prompt, "seed": seed, "out": str(out_dir / f"p{i:03d}_s{seed}.png"), "family": fam}
        for i, (fam, prompt) in enumerate(rows)
        for seed in seeds
    ]
    generate(ckpt, [{k: j[k] for k in ("prompt", "seed", "out")} for j in jobs], out_dir)
    scored = []
    for j in jobs:
        if Path(j["out"]).is_file():
            score, parts = judge(j["out"], j["prompt"])
            scored.append({**j, "path": j["out"], "score": round(score, 5), "parts": parts})
    return scored


def eval_scores(scored: list[dict[str, Any]]) -> dict[str, float]:
    return {f"{r['seed']}|{r['prompt']}": r["score"] for r in scored}


def load_state(work: Path) -> dict[str, Any]:
    path = work / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def save_state(work: Path, state: dict[str, Any]) -> None:
    tmp = work / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, work / "state.json")


def rsi_run(
    opts: argparse.Namespace,
    *,
    generate: Callable[[str, list[dict[str, Any]], Path], None] | None = None,
    train: Callable[[str, Path, Path, float], bool] | None = None,
    judge: Any = None,
) -> dict[str, Any]:
    """Run ``opts.rounds`` RSI rounds; ``generate`` / ``train`` / ``judge`` are injectable for tests."""
    work = Path(opts.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    state = load_state(work)
    if not state:
        if not opts.base_ckpt:
            raise SystemExit(f"No state in {work}; pass --base-ckpt to start.")
        state = {"base_ckpt": opts.base_ckpt, "champion": opts.base_ckpt, "round": 0, "lr": opts.dpo_lr, "history": []}
    generate = generate or (lambda c, j, d: sdx_generate(c, j, d, opts))
    train = train or (lambda c, p, o, lr: sdx_train(c, p, o, lr, opts))
    judge = judge or ArtJudge(opts.device)
    release = getattr(judge, "release", lambda: None)
    held_out = eval_prompts(opts.eval_per_family)
    eval_seeds = list(range(9000, 9000 + opts.eval_seeds))

    if "champion_eval" not in state:
        print(f"[rsi] baseline: judging champion {state['champion']} on {len(held_out)} held-out prompts")
        scored = paint_and_judge(
            state["champion"], held_out, eval_seeds, work / "round_000" / "eval", generate=generate, judge=judge
        )
        if not scored:
            raise SystemExit("Baseline produced no images; check that sample.py runs with this checkpoint.")
        state["champion_eval"] = eval_scores(scored)
        state["family_scores"] = family_means(scored)
        save_state(work, state)

    for _ in range(opts.rounds):
        rnd = state["round"] + 1
        rdir = work / f"round_{rnd:03d}"
        rng = random.Random(opts.seed + rnd)
        weights = curriculum_weights(state["family_scores"])
        rows = train_prompts(opts.prompts, weights, rng)
        seeds = [rng.randrange(1, 2**31) for _ in range(opts.candidates)]
        print(f"[rsi] round {rnd}: {len(rows)} prompts x {len(seeds)} candidates, weights={weights}")

        scored = paint_and_judge(state["champion"], rows, seeds, rdir / "train", generate=generate, judge=judge)
        pairs = mine_self_pairs(scored, opts.min_margin)
        if opts.from_feedback:
            pairs += human_pairs(opts.from_feedback)
        prefs = rdir / "prefs.jsonl"
        prefs.write_text("".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
        report: dict[str, Any] = {"round": rnd, "lr": state["lr"], "pairs": len(pairs), "images": len(scored)}

        candidate = rdir / "candidate.pt"
        release()
        if len(pairs) < 2:
            report["result"] = "skipped: fewer than 2 pairs cleared --min-margin"
        elif not train(state["champion"], prefs, candidate, state["lr"]):
            report["result"] = "skipped: DPO training failed"
        else:
            cand_scored = paint_and_judge(
                str(candidate), held_out, eval_seeds, rdir / "eval", generate=generate, judge=judge
            )
            verdict = gate(state["champion_eval"], eval_scores(cand_scored), opts.min_gain)
            report.update(verdict)
            if verdict["promote"]:
                champ = work / "champion.pt"
                if champ.is_file():
                    os.replace(champ, work / "champion.prev.pt")
                shutil.move(str(candidate), champ)
                state.update(
                    champion=str(champ),
                    champion_eval=eval_scores(cand_scored),
                    family_scores=family_means(cand_scored),
                    lr=opts.dpo_lr,
                )
                report["result"] = "promoted"
            else:
                candidate.unlink(missing_ok=True)
                state["lr"] = max(state["lr"] * 0.5, opts.dpo_lr / 64)
                report["result"] = "rejected"
        state["round"] = rnd
        state["history"].append(report)
        (rdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        save_state(work, state)
        print(f"[rsi] round {rnd}: {report['result']} {json.dumps({k: report.get(k) for k in ('gain', 'win_rate')})}")
    return state


def _rsi_status(work: Path) -> int:
    state = load_state(work)
    if not state:
        print(f"No RSI state in {work}.")
        return 1
    print(f"champion: {state['champion']}  (base: {state['base_ckpt']})")
    print(f"family scores: {state.get('family_scores')}")
    for r in state["history"]:
        print(f"  round {r['round']:>3}: {r['result']:<10} gain={r.get('gain', '-')} win_rate={r.get('win_rate', '-')}")
    return 0


def build_rsi_parser() -> argparse.ArgumentParser:
    from utils.training.preference import default_feedback_log

    ap = argparse.ArgumentParser(
        prog="superior rsi", description=RSI_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-ckpt", default="", help="SDX checkpoint to start from (only needed for a new work dir)")
    ap.add_argument("--work-dir", default="outputs/rsi")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--prompts", type=int, default=16, help="Training prompts per round")
    ap.add_argument("--candidates", type=int, default=4, help="Images per training prompt")
    ap.add_argument("--eval-per-family", type=int, default=3, help="Held-out prompts per medium family")
    ap.add_argument("--eval-seeds", type=int, default=2)
    ap.add_argument("--min-margin", type=float, default=0.03, help="Min judge-score gap for a self pair")
    ap.add_argument("--min-gain", type=float, default=0.005, help="Min held-out mean gain to promote")
    ap.add_argument("--steps", type=int, default=30, help="Sampling steps")
    ap.add_argument("--dpo-steps", type=int, default=200)
    ap.add_argument("--dpo-lr", type=float, default=1e-6)
    ap.add_argument("--dpo-beta", type=float, default=500.0)
    ap.add_argument("--from-feedback", default=str(default_feedback_log()), help="Human feedback log ('' = none)")
    ap.add_argument("--sample-arg", action="append", default=[], help="Extra sample.py token (repeatable)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--status", action="store_true", help="Print round history and exit")
    ap.add_argument("--dry-run", action="store_true", help="Print round 1's prompts and commands, run nothing")
    return ap


def _rsi_main(argv: list[str]) -> int:
    opts = build_rsi_parser().parse_args(argv)
    if opts.status:
        return _rsi_status(Path(opts.work_dir))
    if opts.dry_run:
        for fam, prompt in train_prompts(opts.prompts, curriculum_weights({}), random.Random(opts.seed + 1)):
            print(f"[{fam}] {prompt}")
        print(f"held-out: {len(eval_prompts(opts.eval_per_family))} prompts x {opts.eval_seeds} seeds")
        sdx_cmd = f"sample.py --ckpt {opts.base_ckpt or '<champion>'} --jobs-jsonl <round>/jobs.jsonl"
        print(f"generate: {sdx_cmd} --steps {opts.steps}")
        print(f"train:    train_diffusion_dpo --steps {opts.dpo_steps} --lr {opts.dpo_lr} --dpo-beta {opts.dpo_beta}")
        return 0
    rsi_run(opts)
    return 0


# ---------------------------------------------------------------- CLI

# Commands with their own full parsers; dispatched before the subparser below sees argv.
RAW_COMMANDS: dict[str, tuple[Callable[[list[str]], int], str]] = {
    "auto-improve": (_auto_improve_main, "Benchmark -> prefs -> DPO -> benchmark loop with promotion"),
    "mine-pairs": (_mine_main, "Mine DPO pairs from benchmark results.json"),
    "prefs": (_prefs_main, "Mine DPO pairs from feedback / pick-best / benchmark dumps"),
    "feedback": (_feedback_main, "Record like/dislike/pair/pick ratings; export DPO pairs"),
    "rsi": (_rsi_main, "Self-improvement loop with held-out gate"),
}


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

    p = sub.add_parser("curate", help="Filter a training JSONL with the data-quality filter")
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

    p = sub.add_parser("auto-loop", help="auto-improve with Superior defaults (ViT mine, RAG, soup)")
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

    for name, (_fn, help_text) in RAW_COMMANDS.items():
        sub.add_parser(name, help=help_text, add_help=False)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in RAW_COMMANDS:
        return int(RAW_COMMANDS[argv[0]][0](argv[1:]) or 0)
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
