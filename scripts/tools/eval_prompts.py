"""
Evaluate a checkpoint on a small suite of canonical prompts.

Usage:
    python -m scripts.tools.eval_prompts --ckpt results/.../best.pt --preset sdxl --op-mode portrait

This will:
- Run a fixed set of prompts (portrait, full-body, 3D, interior, exterior, anime) through sample.py.
- Save outputs and a simple index.json you can compare across checkpoints.

Intended for quick qualitative regression testing when you tweak training configs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_PROMPTS: Dict[str, str] = {
    "portrait_real": "portrait of a woman, soft window light, shallow depth of field, sharp eyes, 50mm lens",
    "fullbody_char": "1girl, standing, long dress, legs, shoes, full body, outdoors, sunset",
    "3d_car": "3d render of a red car parked next to a blue house, sunset, detailed reflections",
    "interior_room": "modern living room interior, sofa, coffee table, large window, natural light, detailed background",
    "exterior_city": "street view, urban city at night, neon signs, rain, reflections",
    "anime_char": "1girl, school uniform, anime style, dynamic lighting, detailed hair",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on a small suite of prompts.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--op-mode", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="eval_prompts")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None, help="Override preset steps for evaluation")
    args, unknown = ap.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, str]] = []
    repo_root = Path(__file__).resolve().parents[2]
    sample_py = repo_root / "sample.py"
    for key, prompt in DEFAULT_PROMPTS.items():
        out_path = out_dir / f"{key}.png"
        cli_args = [
            "--ckpt",
            args.ckpt,
            "--prompt",
            prompt,
            "--out",
            str(out_path),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--num",
            "1",
        ]
        if args.preset:
            cli_args.extend(["--preset", args.preset])
        if args.op_mode:
            cli_args.extend(["--op-mode", args.op_mode])
        if args.steps is not None:
            cli_args.extend(["--steps", str(args.steps)])
        cli_args.extend(unknown)
        print(f"[eval_prompts] Running sample.py for '{key}'")
        subprocess.run([sys.executable, str(sample_py), *cli_args], check=True)
        results.append({"name": key, "prompt": prompt, "path": str(out_path)})

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved eval index: {index_path}")


if __name__ == "__main__":
    main()

