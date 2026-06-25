#!/usr/bin/env python3
"""Export Style Genome runs as a browsable HTML gallery."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List


def _rows_from_dir(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for img in sorted(root.rglob("*.png")):
        meta = img.with_suffix(".json")
        prompt = ""
        genome = ""
        if meta.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                prompt = str(data.get("prompt", "") or "")
                genome = str(data.get("style_genome", "") or data.get("genome", "") or "")
            except Exception:
                pass
        rows.append({"path": str(img), "prompt": prompt, "genome": genome})
    return rows


def render_html(rows: List[Dict[str, Any]], *, title: str = "SDX Style Gallery") -> str:
    cards = []
    for r in rows:
        rel = html.escape(r["path"])
        prompt = html.escape(r.get("prompt", ""))
        genome = html.escape(r.get("genome", ""))
        cards.append(
            f'<article class="card"><img src="{rel}" loading="lazy" />'
            f'<p class="prompt">{prompt}</p>'
            f'<pre class="genome">{genome}</pre></article>'
        )
    body = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #0d0d12; color: #e8e8f0; margin: 0; padding: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }}
.card {{ background: #16161f; border-radius: 12px; overflow: hidden; border: 1px solid #2a2a38; }}
.card img {{ width: 100%; display: block; aspect-ratio: 1; object-fit: cover; }}
.prompt {{ font-size: 13px; padding: 12px; margin: 0; line-height: 1.4; }}
.genome {{ font-size: 11px; padding: 0 12px 12px; margin: 0; color: #9aa0b5; white-space: pre-wrap; }}
h1 {{ font-weight: 600; letter-spacing: -0.02em; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="grid">
{body}
</div>
</body>
</html>"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images_dir", type=Path, help="Directory of PNG outputs (optional sidecar JSON)")
    p.add_argument("--out", type=Path, default=Path("style_gallery.html"))
    p.add_argument("--title", type=str, default="SDX Style Gallery")
    args = p.parse_args()
    rows = _rows_from_dir(args.images_dir)
    html_doc = render_html(rows, title=args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {len(rows)} cards -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
