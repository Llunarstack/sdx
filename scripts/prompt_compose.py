#!/usr/bin/env python3
"""Preview composed prompts without running sample.py.

    python scripts/prompt_compose.py "@wlop +character: 1girl, silver hair +car: red sports car"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.prompt.prompt_composer import compose_prompt  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compose @artist / +category prompts for SDX.")
    p.add_argument("prompt", help="Prompt with @artist and/or +category blocks.")
    p.add_argument("--artist-strength", type=float, default=1.0)
    p.add_argument("--artist-index", default="", help="artist_index.json path")
    args = p.parse_args(argv)
    idx = args.artist_index.strip() or None
    cp = compose_prompt(args.prompt, artist_strength=args.artist_strength, artist_index=idx)
    print(cp.positive)
    if cp.artists:
        print(f"# artists: {', '.join(cp.artists)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
