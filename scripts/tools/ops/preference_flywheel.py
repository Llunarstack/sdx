#!/usr/bin/env python3
"""
Preference flywheel helper: mine win/lose pairs from pick-best JSON → DPO JSONL.

    python -m scripts.tools preference_flywheel \\
        --from-pick-best runs/pick_best.json --out data/prefs.jsonl

Also accepts benchmark_suite results.json (same schema as mine_preference_pairs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rows_from_pick_best(path: Path) -> list[dict]:
    """
    Normalize pick-best dumps into mine_preference_pairs rows.

    Expected shapes:
      {"prompt": "...", "candidates": [{"path": "...", "score": 0.8}, ...]}
      or a list of such objects.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else [raw]
    rows: list[dict] = []
    for i, obj in enumerate(items):
        if not isinstance(obj, dict):
            continue
        prompt = str(obj.get("prompt") or obj.get("caption") or "").strip()
        cands = obj.get("candidates") or obj.get("images") or []
        case = str(obj.get("case") or f"pick_{i}")
        for c in cands:
            if not isinstance(c, dict):
                continue
            out = str(c.get("path") or c.get("output") or c.get("image") or "").strip()
            score = c.get("score", c.get("composite", c.get("metric")))
            if not out or score is None:
                continue
            rows.append({"case": case, "prompt": prompt, "output": out, "composite": float(score)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine preference pairs for Diffusion-DPO")
    ap.add_argument("--from-pick-best", type=str, default="", help="pick-best JSON dump")
    ap.add_argument("--from-feedback", type=str, default="", help="feedback JSONL (human pairs merged first)")
    ap.add_argument("--from-benchmark", type=str, default="", help="benchmark results.json")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--min-margin", type=float, default=0.08)
    ap.add_argument("--max-pairs-per-case", type=int, default=2)
    args = ap.parse_args()

    from scripts.tools.training.mine_preference_pairs import mine_pairs

    pairs: list[dict] = []
    if args.from_feedback:
        from utils.training.feedback_bus import pairs_from_feedback

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
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} pairs → {out}", file=sys.stderr)
    print(
        f"Next: python -m scripts.tools train_diffusion_dpo --pairs {out}  (see train_diffusion_dpo --help)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
