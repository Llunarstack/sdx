#!/usr/bin/env python3
"""
Export VP-DDPM-style noise schedule CSV via Rust ``sdx-noise-schedule`` (from ``build_native``).

Usage::

  python -m scripts.tools noise_schedule_export linear --steps 1000 --out sched.csv
  python -m scripts.tools noise_schedule_export cosine --steps 1000 --s 0.008 -o cos.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_NATIVE = ROOT / "native" / "python"
if str(_NATIVE) not in sys.path:
    sys.path.insert(0, str(_NATIVE))


def main() -> int:
    p = argparse.ArgumentParser(description="Noise schedule CSV (Rust sdx-noise-schedule).")
    p.add_argument("mode", choices=("linear", "cosine"), help="Schedule family")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4, help="linear only")
    p.add_argument("--beta-end", type=float, default=2e-2, help="linear only")
    p.add_argument("--s", type=float, default=0.008, help="cosine only (offset s)")
    p.add_argument("-o", "--out", type=str, default="", help="Write CSV (default: stdout)")
    args = p.parse_args()

    from sdx_native.native_tools import run_rust_noise_schedule, rust_noise_schedule_exe

    if rust_noise_schedule_exe() is None:
        print(
            "sdx-noise-schedule not built. Run: .\\scripts\\tools\\native\\build_native.ps1 "
            "or: (cd native/rust/sdx-noise-schedule && cargo build --release)",
            file=sys.stderr,
        )
        return 2

    if args.mode == "linear":
        r = run_rust_noise_schedule(
            [
                "linear",
                "--steps",
                str(args.steps),
                "--beta-start",
                str(args.beta_start),
                "--beta-end",
                str(args.beta_end),
            ]
        )
    else:
        r = run_rust_noise_schedule(["cosine", "--steps", str(args.steps), "--s", str(args.s)])
    if r.returncode != 0:
        print(r.stderr or r.stdout or "noise-schedule failed", file=sys.stderr)
        return r.returncode or 1
    text = r.stdout or ""
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
