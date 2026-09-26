#!/usr/bin/env python3
"""Rate generated images/video frames for RSI preference learning.

Examples::

    python -m scripts.tools feedback like outputs/out.png --prompt "a red cube"
    python -m scripts.tools feedback dislike outputs/out.png --prompt "a red cube"
    python -m scripts.tools feedback pair --win good.png --lose bad.png --prompt "..."
    python -m scripts.tools feedback pick --win good.png --lose a.png --lose b.png
    python -m scripts.tools feedback export-dpo --out data/user_prefs.jsonl
    python -m scripts.tools feedback sync-taste
    python -m scripts.tools feedback status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    from utils.training.feedback_bus import (
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

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--log",
        type=str,
        default=str(default_feedback_log()),
        help="Append-only feedback JSONL path",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    like = sub.add_parser("like", help="Record a liked generation")
    like.add_argument("image", type=str)
    like.add_argument("--prompt", type=str, default="")
    like.add_argument("--score", type=float, default=1.0)
    like.add_argument("--ckpt", type=str, default="")
    like.add_argument("--seed", type=int, default=None)
    like.add_argument("--run-id", type=str, default="")
    like.add_argument("--media-type", type=str, default="image", choices=("image", "video"))
    like.add_argument("--notes", type=str, default="")

    dislike = sub.add_parser("dislike", help="Record a disliked generation")
    dislike.add_argument("image", type=str)
    dislike.add_argument("--prompt", type=str, default="")
    dislike.add_argument("--ckpt", type=str, default="")
    dislike.add_argument("--seed", type=int, default=None)
    dislike.add_argument("--run-id", type=str, default="")
    dislike.add_argument("--media-type", type=str, default="image", choices=("image", "video"))
    dislike.add_argument("--notes", type=str, default="")

    pair = sub.add_parser("pair", help="Record an explicit win/lose pair")
    pair.add_argument("--win", type=str, required=True)
    pair.add_argument("--lose", type=str, required=True)
    pair.add_argument("--prompt", type=str, default="")
    pair.add_argument("--ckpt", type=str, default="")
    pair.add_argument("--seed", type=int, default=None)
    pair.add_argument("--run-id", type=str, default="")
    pair.add_argument("--media-type", type=str, default="image", choices=("image", "video"))

    pick = sub.add_parser("pick", help="User picked one winner among candidates")
    pick.add_argument("--win", type=str, required=True)
    pick.add_argument("--lose", action="append", default=[], help="Loser path (repeatable)")
    pick.add_argument("--prompt", type=str, default="")
    pick.add_argument("--ckpt", type=str, default="")
    pick.add_argument("--seed", type=int, default=None)
    pick.add_argument("--run-id", type=str, default="")
    pick.add_argument("--media-type", type=str, default="image", choices=("image", "video"))

    exp = sub.add_parser("export-dpo", help="Export DPO preference JSONL from feedback")
    exp.add_argument("--out", type=str, required=True)
    exp.add_argument("--allow-cross-prompt", action="store_true")
    exp.add_argument("--no-synthetic", action="store_true", help="Only explicit pair/pick events")

    sub.add_parser("sync-taste", help="Update UserTaste likes_notes/hates from feedback")
    sub.add_parser("status", help="Summarize feedback log")

    args = ap.parse_args(argv)
    log = args.log

    if args.cmd == "like":
        p = record_like(
            args.image,
            prompt=args.prompt,
            score=args.score,
            ckpt=args.ckpt,
            seed=args.seed,
            run_id=args.run_id,
            media_type=args.media_type,
            log_path=log,
            notes=args.notes,
        )
        print(f"like → {p}")
        return 0
    if args.cmd == "dislike":
        p = record_dislike(
            args.image,
            prompt=args.prompt,
            ckpt=args.ckpt,
            seed=args.seed,
            run_id=args.run_id,
            media_type=args.media_type,
            log_path=log,
            notes=args.notes,
        )
        print(f"dislike → {p}")
        return 0
    if args.cmd == "pair":
        p = record_pair(
            args.win,
            args.lose,
            prompt=args.prompt,
            ckpt=args.ckpt,
            seed=args.seed,
            run_id=args.run_id,
            media_type=args.media_type,
            log_path=log,
        )
        print(f"pair → {p}")
        return 0
    if args.cmd == "pick":
        paths = record_pick(
            args.win,
            list(args.lose or []),
            prompt=args.prompt,
            ckpt=args.ckpt,
            seed=args.seed,
            run_id=args.run_id,
            media_type=args.media_type,
            log_path=log,
        )
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
            "Next: python -m scripts.tools rsi_loop --pairs "
            f"{args.out} --base-ckpt <ckpt>   # or preference_flywheel / train_diffusion_dpo",
            file=sys.stderr,
        )
        return 0 if n else 2
    if args.cmd == "sync-taste":
        tp = update_user_taste_from_feedback(log_path=log)
        print(f"UserTaste updated → {tp}")
        return 0
    if args.cmd == "status":
        counts: dict[str, int] = {}
        n = 0
        for row in iter_feedback(log):
            n += 1
            ev = str(row.get("event") or "?")
            counts[ev] = counts.get(ev, 0) + 1
        pairs = pairs_from_feedback(log)
        print(json.dumps({"log": log, "events": n, "by_event": counts, "dpo_pairs_ready": len(pairs)}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
