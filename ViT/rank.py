#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _score(row: dict, quality_weight: float, adherence_weight: float) -> float:
    q = float(row.get("vit_quality_prob", 0.0))
    a = float(row.get("vit_adherence_score", 0.0))
    return quality_weight * q + adherence_weight * a


def main() -> int:
    p = argparse.ArgumentParser(description="Rank JSONL rows by ViT quality/adherence scores")
    p.add_argument("--input", required=True, help="Input JSONL with vit_quality_prob/vit_adherence_score")
    p.add_argument("--output", required=True, help="Ranked output JSONL")
    p.add_argument("--quality-weight", type=float, default=0.6)
    p.add_argument("--adherence-weight", type=float, default=0.4)
    p.add_argument("--top-k", type=int, default=0, help="0 = keep all")
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    rows = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            try:
                row = json.loads(t)
            except Exception:
                continue
            row["vit_final_score"] = _score(row, args.quality_weight, args.adherence_weight)
            rows.append(row)

    rows.sort(key=lambda x: float(x.get("vit_final_score", 0.0)), reverse=True)
    if args.top_k > 0:
        rows = rows[: args.top_k]

    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as wf:
        for r in rows:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ViT] ranked {len(rows)} rows -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
