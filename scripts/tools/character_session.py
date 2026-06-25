#!/usr/bin/env python3
"""Create and inspect character session lock JSON for sample.py --character-session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from utils.generation.sample_features import load_character_session, save_character_session


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=Path("character_session.json"))
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create empty session file")
    init.add_argument("--name", type=str, default="character")

    setp = sub.add_parser("set", help="Set a session field")
    setp.add_argument("field", choices=["prompt_additions", "negative_prompt", "reference_images"])
    setp.add_argument("value", type=str)

    sub.add_parser("show", help="Print session JSON")

    args = p.parse_args()
    path = args.path

    if args.cmd == "init":
        data: Dict[str, Any] = {
            "name": args.name,
            "prompt_additions": "",
            "negative_prompt": "",
            "reference_images": [],
        }
        save_character_session(path, data)
        print(f"Created {path}")
        return 0

    data = load_character_session(path) if path.is_file() else {}
    if args.cmd == "set":
        if args.field == "reference_images":
            data[args.field] = [v.strip() for v in args.value.split(",") if v.strip()]
        else:
            data[args.field] = args.value
        save_character_session(path, data)
        print(f"Updated {args.field} in {path}")
        return 0
    if args.cmd == "show":
        print(json.dumps(data, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
