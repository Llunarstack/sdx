#!/usr/bin/env python3
"""
Automated improvement loop:
1) benchmark base checkpoint
2) mine preference pairs
3) run diffusion DPO stage-2
4) benchmark base vs DPO checkpoint
5) optionally promote best checkpoint by leaderboard
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _run(cmd: List[str], *, cwd: Path, dry_run: bool) -> int:
    print("Running:", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def _read_leaderboard(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    out: List[Dict[str, object]] = []
    for r in data:
        if isinstance(r, dict):
            out.append(r)
    return out


def _best_model_tag(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return ""
    best = sorted(rows, key=lambda r: float(r.get("mean_composite", 0.0) or 0.0), reverse=True)[0]
    return str(best.get("model", "") or "").strip()


def _resolve_ckpt_by_tag(ckpts: List[Path], tag: str) -> Optional[Path]:
    t = (tag or "").strip().lower()
    if not t:
        return None
    for p in ckpts:
        if p.stem.lower() == t:
            return p
    return None


def _iter_dir(work: Path, it: int, total: int) -> Path:
    if total <= 1:
        return work
    return work / f"iter_{it:03d}"


def _best_row(rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not rows:
        return None
    return sorted(rows, key=lambda r: float(r.get("mean_composite", 0.0) or 0.0), reverse=True)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run benchmark->preference->DPO->benchmark loop.")
    ap.add_argument("--base-ckpt", type=str, required=True, help="Starting checkpoint path.")
    ap.add_argument("--work-dir", type=str, default="auto_improve_loop")
    ap.add_argument("--preset", type=str, default="sdxl")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--suite-pack", type=str, default="top_contender_proxy_v1")
    ap.add_argument("--steps", type=int, default=30, help="Benchmark sampling steps.")
    ap.add_argument("--seed-list", type=str, default="", help="Comma-separated seeds used by benchmark_suite.")
    ap.add_argument("--robustness-penalty", type=float, default=0.15, help="Penalty weight for seed variance in leaderboard ranking.")
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    work = Path(args.work_dir)
    if not args.dry_run:
        work.mkdir(parents=True, exist_ok=True)
    if int(args.iterations) < 1:
        print("--iterations must be >= 1", file=sys.stderr)
        return 2

    current_base = Path(args.base_ckpt).resolve()
    best_overall_ckpt: Optional[Path] = current_base
    best_overall_score = float("-inf")

    for it in range(1, int(args.iterations) + 1):
        it_dir = _iter_dir(work, it, int(args.iterations)).resolve()
        if not args.dry_run:
            it_dir.mkdir(parents=True, exist_ok=True)
        dpo_ckpt = (it_dir / "dpo_policy.pt").resolve()
        pref_jsonl = (it_dir / "prefs.jsonl").resolve()
        bench_before = (it_dir / "bench_before").resolve()
        bench_after = (it_dir / "bench_after").resolve()

        print(f"[auto_improve_loop] iteration {it}/{int(args.iterations)} base={current_base}")

        # 1) Benchmark base + export preferences.
        bench_cmd_1 = [
            sys.executable,
            "-m",
            "scripts.tools.benchmark_suite",
            "--ckpt",
            str(current_base),
            "--out-dir",
            str(bench_before),
            "--preset",
            str(args.preset),
            "--device",
            str(args.device),
            "--suite-pack",
            str(args.suite_pack),
            "--steps",
            str(int(args.steps)),
            "--seed-list",
            str(args.seed_list),
            "--num",
            str(int(args.num)),
            "--pick-best",
            str(args.pick_best),
            "--robustness-penalty",
            str(float(args.robustness_penalty)),
            "--export-preference-jsonl",
            str(pref_jsonl),
            "--preference-min-margin",
            str(float(args.preference_min_margin)),
            "--preference-max-pairs-per-case",
            str(int(args.preference_max_pairs_per_case)),
        ]
        if bool(args.export_hardcases):
            bench_cmd_1.extend(
                [
                    "--export-hardcases-jsonl",
                    str((it_dir / "hardcases.jsonl").resolve()),
                    "--hardcase-threshold",
                    str(float(args.hardcase_threshold)),
                    "--hardcase-max-rows",
                    str(int(args.hardcase_max_rows)),
                ]
            )
        rc = _run(bench_cmd_1, cwd=root, dry_run=bool(args.dry_run))
        if rc != 0:
            return rc

        # Optional hard-case aware remine: converts hard-case JSONL into extra weighted preference pairs.
        if bool(args.export_hardcases):
            hardcase_jsonl = (it_dir / "hardcases.jsonl").resolve()
            remine_cmd = [
                sys.executable,
                "-m",
                "scripts.tools.training.mine_preference_pairs",
                "--results-json",
                str((bench_before / "results.json").resolve()),
                "--out-jsonl",
                str(pref_jsonl),
                "--min-margin",
                str(float(args.preference_min_margin)),
                "--max-pairs-per-case",
                str(int(args.preference_max_pairs_per_case)),
                "--allow-missing-files",
                "--hardcases-jsonl",
                str(hardcase_jsonl),
                "--hardcase-extra-pairs",
                str(int(args.hardcase_preference_boost)),
                "--hardcase-min-margin-scale",
                str(float(args.hardcase_margin_scale)),
            ]
            rc = _run(remine_cmd, cwd=root, dry_run=bool(args.dry_run))
            if rc != 0:
                return rc

        # 2) DPO stage-2.
        dpo_cmd = [
            sys.executable,
            str(root / "scripts" / "tools" / "training" / "train_diffusion_dpo.py"),
            "--ckpt",
            str(current_base),
            "--preference-jsonl",
            str(pref_jsonl),
            "--out",
            str(dpo_ckpt),
            "--steps",
            str(int(args.dpo_steps)),
            "--batch-size",
            str(int(args.dpo_batch_size)),
            "--lr",
            str(float(args.dpo_lr)),
            "--dpo-beta",
            str(float(args.dpo_beta)),
            "--device",
            str(args.device),
        ]
        rc = _run(dpo_cmd, cwd=root, dry_run=bool(args.dry_run))
        if rc != 0:
            return rc

        # 3) Benchmark base vs DPO.
        bench_cmd_2 = [
            sys.executable,
            "-m",
            "scripts.tools.benchmark_suite",
            "--ckpt",
            str(current_base),
            str(dpo_ckpt),
            "--out-dir",
            str(bench_after),
            "--preset",
            str(args.preset),
            "--device",
            str(args.device),
            "--suite-pack",
            str(args.suite_pack),
            "--steps",
            str(int(args.steps)),
            "--seed-list",
            str(args.seed_list),
            "--num",
            str(int(args.num)),
            "--pick-best",
            str(args.pick_best),
            "--robustness-penalty",
            str(float(args.robustness_penalty)),
        ]
        rc = _run(bench_cmd_2, cwd=root, dry_run=bool(args.dry_run))
        if rc != 0:
            return rc

        if not args.dry_run:
            lb_path = bench_after / "leaderboard.json"
            rows = _read_leaderboard(lb_path)
            br = _best_row(rows)
            best_tag = _best_model_tag(rows)
            iter_best_ckpt = _resolve_ckpt_by_tag([current_base, dpo_ckpt], best_tag)
            if br is not None and iter_best_ckpt is not None:
                iter_best_score = float(br.get("mean_composite", 0.0) or 0.0)
                print(f"[auto_improve_loop] iter best: {iter_best_ckpt} score={iter_best_score:.4f}")
                current_base = iter_best_ckpt
                if iter_best_score > best_overall_score:
                    best_overall_score = iter_best_score
                    best_overall_ckpt = iter_best_ckpt
            else:
                current_base = dpo_ckpt

    # Optional promotion at end.
    if bool(args.promote_best):
        if args.dry_run:
            print(f"Dry-run: would read leaderboard and promote to {args.promote_path}")
            return 0
        best_ckpt = best_overall_ckpt or current_base
        if best_ckpt is None:
            print("Could not resolve best checkpoint path for promotion.", file=sys.stderr)
            return 2
        promote_path = Path(args.promote_path)
        promote_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, promote_path)
        print(f"Promoted best checkpoint: {best_ckpt} -> {promote_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
