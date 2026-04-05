#!/usr/bin/env python3
"""
Mine DPO-style preference pairs from benchmark results.

Input: benchmark_suite results.json rows with at least:
  - prompt
  - output (image path)
  - composite (quality score)
  - case (grouping key)

Output: JSONL rows consumable by train_diffusion_dpo.py / PreferenceImageDataset:
  - win_image_path
  - lose_image_path
  - caption
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


def _group_key(row: Dict[str, Any]) -> Tuple[str, str]:
    case = str(row.get("case", "") or "").strip()
    prompt = str(row.get("prompt", "") or "").strip()
    return case, prompt


def mine_pairs(
    rows: Iterable[Dict[str, Any]],
    *,
    min_margin: float = 0.08,
    max_pairs_per_case: int = 2,
    require_existing_files: bool = True,
    hard_case_names: Set[str] | None = None,
    hardcase_extra_pairs: int = 0,
    hardcase_min_margin_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = _group_key(r)
        if not key[1]:
            continue
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
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
            if not p:
                continue
            if require_existing_files and not Path(p).is_file():
                continue
            scored.append((s, x))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda t: t[0], reverse=True)
        hi = scored[0]
        used = 0
        for lo in reversed(scored[1:]):
            margin = float(hi[0] - lo[0])
            if margin < float(eff_min_margin):
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
            if used >= int(eff_max_pairs):
                break
    return out


def _read_hardcase_names(path: Path) -> Set[str]:
    out: Set[str] = set()
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        c = str(row.get("case", "") or "").strip()
        if c:
            out.add(c)
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
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
    args = ap.parse_args()

    in_path = Path(args.results_json)
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("results json must be a list")
    hard_case_names: Set[str] = set()
    if str(args.hardcases_jsonl).strip():
        hard_case_names = _read_hardcase_names(Path(str(args.hardcases_jsonl).strip()))
    rows = mine_pairs(
        data,
        min_margin=float(args.min_margin),
        max_pairs_per_case=int(args.max_pairs_per_case),
        require_existing_files=not bool(args.allow_missing_files),
        hard_case_names=hard_case_names,
        hardcase_extra_pairs=int(args.hardcase_extra_pairs),
        hardcase_min_margin_scale=float(args.hardcase_min_margin_scale),
    )
    out_path = Path(args.out_jsonl)
    _write_jsonl(out_path, rows)
    print(f"[mine_preference_pairs] wrote {len(rows)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
