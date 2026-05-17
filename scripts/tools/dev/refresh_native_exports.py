#!/usr/bin/env python3
"""
Recompute ``_NATIVE_EXPORTS`` in ``utils/native/__init__.py`` from the legacy
``import *`` chain (seven modules; last duplicate wins).

Print the assignment to stdout (default), or rewrite the marked block::

    python scripts/tools/dev/refresh_native_exports.py --write

Verify the file is current (exit 1 if stale)::

    python scripts/tools/dev/refresh_native_exports.py --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_NATIVE_INIT = _REPO / "utils" / "native" / "__init__.py"

_EXEC_CHAIN = (
    "from sdx_native import *",
    "from sdx_native.cuda_image_metrics_native import *",
    "from sdx_native.image_metrics_native import *",
    "from sdx_native.latent_geometry import *",
    "from sdx_native.native_tools import *",
    "from sdx_native.score_ops_native import *",
    "from sdx_native.text_hygiene import *",
)

_BEGIN = "# BEGIN_NATIVE_EXPORTS"
_END = "# END_NATIVE_EXPORTS"


def _merged_public_names() -> list[str]:
    np = _REPO / "native" / "python"
    sys.path.insert(0, str(np))
    ns: dict[str, object] = {}
    for stmt in _EXEC_CHAIN:
        exec(compile(stmt, "<refresh_native_exports>", "exec"), ns)
    return sorted(k for k in ns if not k.startswith("_"))


def _render_block(names: list[str]) -> str:
    lines = [
        _BEGIN,
        "_NATIVE_EXPORTS: frozenset[str] = frozenset(",
        "    {",
    ]
    lines.extend(f'        {repr(n)},' for n in names)
    lines.extend(["    }", ")", _END, ""])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--write",
        action="store_true",
        help=f"Patch {_NATIVE_INIT.relative_to(_REPO)} between {_BEGIN!r} / {_END!r}.",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Exit 0 if file matches computed exports, else 1.",
    )
    args = ap.parse_args()
    if args.write and args.check:
        ap.error("use only one of --write / --check")

    names = _merged_public_names()
    block = _render_block(names)

    if not args.write and not args.check:
        print(block, end="")
        return 0

    text = _NATIVE_INIT.read_text(encoding="utf-8")
    pat = re.compile(
        rf"{re.escape(_BEGIN)}.*?{re.escape(_END)}\s*\n",
        re.DOTALL,
    )
    if not pat.search(text):
        print(f"error: {_BEGIN} … {_END} block not found in {_NATIVE_INIT}", file=sys.stderr)
        return 1

    if args.check:
        current = pat.search(text)
        assert current is not None
        if current.group(0).rstrip("\n") == block.rstrip("\n"):
            return 0
        print("error: _NATIVE_EXPORTS is stale; run with --write", file=sys.stderr)
        return 1

    new_text, n = pat.subn(block + "\n", text, count=1)
    if n != 1:
        print("error: replacement failed", file=sys.stderr)
        return 1
    _NATIVE_INIT.write_text(new_text, encoding="utf-8", newline="\n")
    print(f"wrote {_NATIVE_INIT.relative_to(_REPO)} ({len(names)} names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
