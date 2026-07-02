#!/usr/bin/env python3
"""Profile one image — test the caption fusion pipeline.

    python scripts/profile_image_cli.py path/to/image.png
    python scripts/profile_image_cli.py path/to/image.png --json
    python scripts/profile_image_cli.py path/to/image.png --no-vlm --no-reverse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.caption.image_profiler import profile_image  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Profile one image (booru + SauceNAO + VLM).")
    p.add_argument("image", help="Image path")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--no-reverse", action="store_true")
    p.add_argument("--no-saucenao", action="store_true")
    p.add_argument("--no-tineye", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    prof = profile_image(
        args.image,
        use_vlm=not args.no_vlm,
        use_reverse_search=not args.no_reverse,
        use_saucenao=not args.no_saucenao,
        use_tineye=not args.no_tineye,
        device=args.device,
    )
    if args.json:
        print(json.dumps(prof.to_manifest_patch() | {"sources": prof.sources}, ensure_ascii=False, indent=2))
    else:
        print("caption:", prof.caption)
        print("scene_summary:", prof.scene_summary)
        print("character:", prof.character_tags)
        print("copyright:", prof.copyright_tags)
        print("artist:", prof.artist_tags)
        print("is_oc:", prof.is_original_character)
        print("confidence:", prof.confidence)
        print("sources:", prof.sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
