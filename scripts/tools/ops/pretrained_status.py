#!/usr/bin/env python3
"""
Report SDX pretrained model resolution and local availability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.modeling.model_paths import pretrained_catalog, verify_gen_searcher_8b_local


def _is_local_path(path_s: str) -> bool:
    p = Path(path_s)
    return p.is_absolute() and p.exists()


def _folder_size_gb(path_s: str) -> float:
    p = Path(path_s)
    if not p.is_dir():
        return 0.0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += int(f.stat().st_size)
            except OSError:
                pass
    return float(total) / float(1024**3)


def build_report() -> dict:
    rows = pretrained_catalog()
    for r in rows:
        resolved = str(r["resolved"])
        is_local = _is_local_path(resolved)
        r["is_local"] = bool(is_local)
        r["size_gb"] = round(_folder_size_gb(resolved), 3) if is_local else 0.0
        if str(r["name"]) == "GenSearcher-8B" and is_local:
            gs = verify_gen_searcher_8b_local(resolved)
            r["gen_searcher_ok"] = bool(gs.get("all_required_present", False))
            r["gen_searcher_missing"] = list(gs.get("missing", []))
    return {"models": rows}


def _print_human(report: dict) -> None:
    rows = list(report.get("models", []))
    print("SDX pretrained resolution status:")
    for r in rows:
        src = "local" if r.get("is_local") else "hf"
        line = f"- {r['name']}: {src} -> {r['resolved']}"
        if r.get("is_local"):
            line += f" ({r.get('size_gb', 0.0):.3f} GB)"
        if r.get("name") == "GenSearcher-8B":
            line += f" | gen_searcher_ok={bool(r.get('gen_searcher_ok', False))}"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Show SDX pretrained model wiring status.")
    ap.add_argument("--out-json", type=str, default="", help="Optional JSON path for machine-readable report.")
    args = ap.parse_args()

    report = build_report()
    _print_human(report)
    if str(args.out_json).strip():
        p = Path(str(args.out_json).strip())
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[pretrained_status] wrote report -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
