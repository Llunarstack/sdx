#!/usr/bin/env python3
"""Scan trained LoRA adapters and build ``lora_bank/index.json`` for inference.

    python setup/build_lora_bank_index.py \\
        --bank-root /workspace/data/lora_bank \\
        --out /workspace/data/lora_bank/index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.lora.lora_bank import LoRABank, LoRAEntry, slugify_lora_key  # noqa: E402

_LORA_SUFFIXES = ("_lora.pt", "_lora.safetensors", ".safetensors", ".pt")


def _find_adapter_file(folder: Path) -> Path | None:
    for name in ("best_lora.pt", "latest_lora.pt", "adapter_lora.pt"):
        p = folder / name
        if p.is_file():
            return p
    for p in sorted(folder.glob("*")):
        if p.is_file() and any(p.name.endswith(s) for s in _LORA_SUFFIXES):
            return p
    return None


def build_index(bank_root: Path, *, default_scale: float = 0.75) -> LoRABank:
    bank = LoRABank(root=bank_root)

    for kind, role, dest in (
        ("artist", "style", "artists"),
        ("style", "style", "styles"),
    ):
        base = bank_root / kind
        if not base.is_dir():
            continue
        bucket = getattr(bank, dest)
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            adapter = _find_adapter_file(sub)
            if adapter is None:
                continue
            key = slugify_lora_key(sub.name)
            rel = adapter.relative_to(bank_root)
            bucket[key] = LoRAEntry(
                lora=str(rel).replace("\\", "/"),
                default_scale=default_scale,
                role=role,
            )

    return bank


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build LoRA bank index.json.")
    p.add_argument("--bank-root", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--default-scale", type=float, default=0.75)
    args = p.parse_args(argv)

    root = Path(args.bank_root or REPO_ROOT / "data" / "lora_bank")
    out = Path(args.out or root / "index.json")
    bank = build_index(root, default_scale=args.default_scale)
    bank.save(out)
    print(f"LoRA bank index: {len(bank.artists)} artists, {len(bank.styles)} styles -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
