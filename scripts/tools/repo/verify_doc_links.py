#!/usr/bin/env python3
"""
Verify relative markdown links in key docs point to existing files.

Usage (repo root):
    python -m scripts.tools verify_doc_links
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# (file_path, content) globs: markdown files to scan
SCAN_GLOBS = [
    "README.md",
    "PROJECT_STRUCTURE.md",
    "docs/**/*.md",
    "pipelines/**/*.md",
    "ViT/**/*.md",
    "scripts/**/*.md",
]

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _targets(path: str) -> list[str]:
    """Extract path from markdown link target; ignore URLs and anchors-only."""
    path = path.strip()
    if not path or path.startswith(("#", "http://", "https://", "mailto:")):
        return []
    # strip "title" in angle quotes not common; strip fragment for file check
    base = path.split("#", 1)[0].strip()
    if not base:
        return []
    return [base]


def main() -> int:
    bad: list[tuple[str, str, str]] = []
    scanned = 0
    for pattern in SCAN_GLOBS:
        for md in REPO_ROOT.glob(pattern):
            if not md.is_file():
                continue
            scanned += 1
            text = md.read_text(encoding="utf-8", errors="replace")
            for m in LINK_RE.finditer(text):
                for rel in _targets(m.group(1)):
                    # Windows path in md often uses forward slashes
                    target = (md.parent / rel.replace("/", "\\") if "\\" in rel else md.parent / rel).resolve()
                    # Normalize: relative to repo
                    try:
                        target.relative_to(REPO_ROOT)
                    except ValueError:
                        # link goes outside repo — skip
                        continue
                    if not target.exists():
                        bad.append((str(md.relative_to(REPO_ROOT)), rel, str(target.relative_to(REPO_ROOT))))

    print(f"Scanned {scanned} markdown files under {REPO_ROOT.name}/", file=sys.stderr)
    if bad:
        print("Broken relative links:", file=sys.stderr)
        for src, link, _ in bad:
            print(f"  {src}: [{link}]", file=sys.stderr)
        return 1
    print("All relative markdown links OK.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
