#!/usr/bin/env python3
"""Manage taste profiles (likes/dislikes) and export DPO preference pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_profile(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"likes": [], "dislikes": [], "notes": ""}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("profile must be a JSON object")
    data.setdefault("likes", [])
    data.setdefault("dislikes", [])
    return data


def save_profile(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def add_entry(data: Dict[str, Any], *, bucket: str, image: str, prompt: str, score: float = 1.0) -> None:
    row = {"image": image, "prompt": prompt, "score": float(score)}
    data.setdefault(bucket, []).append(row)


def export_pairs(data: Dict[str, Any], *, min_prompt_match: bool = True) -> List[Dict[str, str]]:
    """Pair each like against dislikes with the same prompt when possible."""
    likes = list(data.get("likes") or [])
    dislikes = list(data.get("dislikes") or [])
    pairs: List[Dict[str, str]] = []
    for win in likes:
        wp = str(win.get("prompt", "") or "")
        for lose in dislikes:
            lp = str(lose.get("prompt", "") or "")
            if min_prompt_match and wp and lp and wp != lp:
                continue
            pairs.append(
                {
                    "win_image_path": str(win.get("image", "")),
                    "lose_image_path": str(lose.get("image", "")),
                    "caption": wp or lp,
                }
            )
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, default=Path("taste_profile.json"))
    sub = p.add_subparsers(dest="cmd", required=True)

    like = sub.add_parser("like", help="Record a liked image")
    like.add_argument("image", type=str)
    like.add_argument("--prompt", type=str, default="")
    like.add_argument("--score", type=float, default=1.0)

    dislike = sub.add_parser("dislike", help="Record a disliked image")
    dislike.add_argument("image", type=str)
    dislike.add_argument("--prompt", type=str, default="")

    sub.add_parser("show", help="Print profile JSON")

    exp = sub.add_parser("export-dpo", help="Write preference pairs JSONL")
    exp.add_argument("--out", type=Path, default=Path("taste_pairs.jsonl"))
    exp.add_argument("--allow-cross-prompt", action="store_true")

    args = p.parse_args()
    prof = load_profile(args.profile)

    if args.cmd == "like":
        add_entry(prof, bucket="likes", image=args.image, prompt=args.prompt, score=args.score)
        save_profile(args.profile, prof)
        print(f"Added like -> {args.profile}")
        return 0
    if args.cmd == "dislike":
        add_entry(prof, bucket="dislikes", image=args.image, prompt=args.prompt)
        save_profile(args.profile, prof)
        print(f"Added dislike -> {args.profile}")
        return 0
    if args.cmd == "show":
        print(json.dumps(prof, indent=2))
        return 0
    if args.cmd == "export-dpo":
        pairs = export_pairs(prof, min_prompt_match=not args.allow_cross_prompt)
        lines = [json.dumps(row) for row in pairs if row["win_image_path"] and row["lose_image_path"]]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"Wrote {len(lines)} pairs -> {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
