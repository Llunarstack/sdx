#!/usr/bin/env python3
"""
Run `sample.py` over `examples/eval_prompts_baseline.json` (or another pack).

Dry-run default (prints commands). Pass ``--execute`` to actually invoke sample.

We load ``eval_prompt_pack`` via ``importlib`` so this script does **not** import
``utils.generation``'s package ``__init__`` (which re-exports heavy modules and
pulls hundreds of transitive types into static analysis / CI).

Usage (repo root)::

    python examples/run_baseline_eval.py --ckpt results/your_run/best.pt
    python examples/run_baseline_eval.py --ckpt results/your_run/best.pt --execute
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_eval_prompt_records_fn() -> Callable[..., Any]:
    path = _REPO_ROOT / "utils" / "generation" / "eval_prompt_pack.py"
    name = "_sdx_eval_prompt_pack_standalone"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load eval prompt pack from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "load_eval_prompt_records", None)
    if fn is None or not callable(fn):
        raise ImportError("load_eval_prompt_records missing from eval_prompt_pack")
    return fn


load_eval_prompt_records = _load_eval_prompt_records_fn()


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

    records = load_eval_prompt_records(pack)
    for r in records:
        eid = r.id
        prompt = r.prompt
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
