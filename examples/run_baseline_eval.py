#!/usr/bin/env python3
"""
Run `sample.py` over `examples/eval_prompts_baseline.json` (or another pack).

Dry-run default (prints commands). Pass ``--execute`` to actually invoke sample.

Pack schema: a JSON array of prompt strings, or of objects with a ``"prompt"`` string and an optional ``"id"``.

Usage (repo root)::

    python examples/run_baseline_eval.py --ckpt results/your_run/best.pt
    python examples/run_baseline_eval.py --ckpt results/your_run/best.pt --execute
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_eval_prompts(path: Path | str) -> list[tuple[str, str]]:
    """``(id, prompt)`` per pack entry; ids default to ``p<index>``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("eval prompt pack root must be a JSON array")
    out: list[tuple[str, str]] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            if item.strip():
                out.append((f"p{i}", item.strip()))
        elif isinstance(item, dict):
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"item {i}: missing non-empty string 'prompt'")
            eid = item.get("id")
            out.append((eid.strip() if isinstance(eid, str) and eid.strip() else f"p{i}", prompt.strip()))
        else:
            raise ValueError(f"item {i}: expected str or object, got {type(item).__name__}")
    if not out:
        raise ValueError("eval prompt pack is empty")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True, help="Checkpoint for sample.py")
    p.add_argument(
        "--pack",
        type=Path,
        default=_REPO_ROOT / "examples" / "eval_prompts_baseline.json",
        help="JSON prompt pack",
    )
    p.add_argument("--out-dir", type=Path, default=Path("eval_runs/baseline"), help="Output directory")
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional args after -- passed to sample.py (e.g. -- --steps 30 --preset auto)",
    )
    p.add_argument("--execute", action="store_true", help="Run sample.py; default is dry-run print only")
    args = p.parse_args()

    pack = args.pack.resolve()
    ckpt = args.ckpt.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    for eid, prompt in load_eval_prompts(pack):
        dest = out_dir / f"{eid}.png"
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "sample.py"),
            "--ckpt",
            str(ckpt),
            "--prompt",
            prompt,
            "--out",
            str(dest),
            *extra,
        ]
        if args.execute:
            print("RUN:", " ".join(cmd[:6]), "...")
            subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
        else:
            print("WOULD RUN:", " ".join(cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
