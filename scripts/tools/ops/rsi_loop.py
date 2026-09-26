#!/usr/bin/env python3
"""RSI loop: user feedback (+ optional benchmark) → DPO → promote-gated improve.

Closes the recursive self-improvement path for SDX image models using the
existing Diffusion-DPO trainer and auto_improve / scorecard gates.

    # From accumulated thumbs
    python -m scripts.tools rsi_loop \\
        --base-ckpt results/run/best.pt \\
        --from-feedback outputs/feedback/feedback.jsonl \\
        --work-dir outputs/rsi

    # Merge user pairs with last benchmark mine
    python -m scripts.tools rsi_loop \\
        --base-ckpt results/run/best.pt \\
        --from-feedback outputs/feedback/feedback.jsonl \\
        --from-benchmark auto_improve_loop/results.json \\
        --run-auto-improve

Dry-run plans the pair merge without training::

    python -m scripts.tools rsi_loop --base-ckpt ... --from-feedback ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    print("Running:", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def _merge_pair_jsonl(paths: list[Path], out: Path) -> int:
    seen: set[tuple[str, str, str]] = set()
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in paths:
            if not p.is_file():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                win = str(obj.get("win_image_path") or "").strip()
                lose = str(obj.get("lose_image_path") or "").strip()
                cap = str(obj.get("caption") or obj.get("prompt") or "").strip()
                key = (win, lose, cap)
                if not win or not lose or key in seen:
                    continue
                seen.add(key)
                # Prefer human pairs: stamp source if missing
                obj.setdefault("source", "merged")
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    from utils.training.feedback_bus import default_feedback_log, export_dpo_jsonl, update_user_taste_from_feedback

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-ckpt", type=str, required=True)
    ap.add_argument("--work-dir", type=str, default="outputs/rsi")
    ap.add_argument("--from-feedback", type=str, default=str(default_feedback_log()))
    ap.add_argument("--from-pairs", type=str, default="", help="Existing DPO JSONL to merge")
    ap.add_argument("--from-benchmark", type=str, default="", help="benchmark results.json → mine pairs")
    ap.add_argument("--from-pick-best", type=str, default="", help="pick-best JSON → mine pairs")
    ap.add_argument("--min-user-pairs", type=int, default=4, help="Abort if fewer user pairs")
    ap.add_argument("--dpo-steps", type=int, default=200)
    ap.add_argument("--dpo-beta", type=float, default=300.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--run-auto-improve", action="store_true", help="Delegate to auto_improve_loop after merge")
    ap.add_argument("--sync-taste", action="store_true", default=True)
    ap.add_argument("--no-sync-taste", action="store_false", dest="sync_taste")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-cross-prompt", action="store_true")
    args = ap.parse_args(argv)

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    user_pairs = work / "user_prefs.jsonl"
    mined = work / "mined_prefs.jsonl"
    merged = work / "merged_prefs.jsonl"

    # 1) Export user feedback → DPO
    n_user = export_dpo_jsonl(
        user_pairs,
        log_path=args.from_feedback,
        same_prompt_only=not args.allow_cross_prompt,
        include_synthetic=True,
    )
    print(f"User feedback pairs: {n_user} → {user_pairs}", flush=True)
    if n_user < int(args.min_user_pairs) and not args.from_pairs and not args.from_benchmark and not args.from_pick_best:
        print(
            f"Need at least --min-user-pairs {args.min_user_pairs} (have {n_user}). "
            "Rate more images: python -m scripts.tools feedback like|dislike|pair ...",
            file=sys.stderr,
        )
        return 2

    if args.sync_taste:
        tp = update_user_taste_from_feedback(log_path=args.from_feedback)
        print(f"Synced UserTaste → {tp}", flush=True)

    # 2) Optional mine from benchmark / pick-best
    mine_inputs: list[str] = []
    if args.from_benchmark or args.from_pick_best:
        cmd = [
            sys.executable,
            "-m",
            "scripts.tools",
            "preference_flywheel",
            "--out",
            str(mined),
        ]
        if args.from_benchmark:
            cmd.extend(["--from-benchmark", args.from_benchmark])
        if args.from_pick_best:
            cmd.extend(["--from-pick-best", args.from_pick_best])
        rc = _run(cmd, cwd=ROOT, dry_run=args.dry_run)
        if rc != 0 and not args.dry_run:
            return rc
        if mined.is_file() or args.dry_run:
            mine_inputs.append(str(mined))

    # 3) Merge
    sources = [user_pairs]
    if args.from_pairs:
        sources.append(Path(args.from_pairs))
    if mined.is_file():
        sources.append(mined)
    n_merged = _merge_pair_jsonl(sources, merged) if not args.dry_run else n_user
    if args.dry_run:
        print(f"[dry-run] would merge → {merged} (user={n_user})", flush=True)
    else:
        print(f"Merged {n_merged} unique pairs → {merged}", flush=True)
        (work / "rsi_plan.json").write_text(
            json.dumps(
                {
                    "base_ckpt": args.base_ckpt,
                    "user_pairs": n_user,
                    "merged_pairs": n_merged,
                    "dpo_steps": args.dpo_steps,
                    "feedback_log": args.from_feedback,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.run_auto_improve:
        # Copy merged prefs into auto_improve work and run full loop
        ai_work = work / "auto_improve"
        ai_work.mkdir(parents=True, exist_ok=True)
        if not args.dry_run and merged.is_file():
            shutil.copy2(merged, ai_work / "prefs.jsonl")
        cmd = [
            sys.executable,
            "-m",
            "scripts.tools",
            "auto_improve_loop",
            "--base-ckpt",
            args.base_ckpt,
            "--work-dir",
            str(ai_work),
            "--device",
            args.device,
            "--dpo-steps",
            str(args.dpo_steps),
            "--dpo-beta",
            str(args.dpo_beta),
            "--iterations",
            "1",
        ]
        return _run(cmd, cwd=ROOT, dry_run=args.dry_run)

    # Direct DPO stage-2 on merged pairs
    out_ckpt = work / "rsi_dpo.pt"
    cmd = [
        sys.executable,
        "-m",
        "scripts.tools",
        "train_diffusion_dpo",
        "--ckpt",
        args.base_ckpt,
        "--preference-jsonl",
        str(merged if merged.is_file() or not args.dry_run else user_pairs),
        "--out",
        str(out_ckpt),
        "--steps",
        str(args.dpo_steps),
        "--dpo-beta",
        str(args.dpo_beta),
        "--device",
        args.device,
    ]
    rc = _run(cmd, cwd=ROOT, dry_run=args.dry_run)
    if rc == 0:
        print(
            f"RSI DPO ckpt → {out_ckpt}\n"
            "Gate with: python -m scripts.tools eval_scorecard ... then promote if promote_ok.\n"
            "Inference taste (no train): perfect_gen / sample already bias from UserTaste after sync-taste.",
            flush=True,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
