#!/usr/bin/env python3
"""Deprecated alias for ``preview_prompt_stack``.

Prefer::

  python -m scripts.tools preview_prompt_stack --prompt "..."
"""

from __future__ import annotations

import sys
import warnings


def main() -> int:
    warnings.warn(
        "preview_generation_prompt is deprecated; use preview_prompt_stack",
        DeprecationWarning,
        stacklevel=2,
    )
    # Re-dispatch to the canonical tool so argv/help stay consistent.
    from pathlib import Path

    here = Path(__file__).resolve().parent
    sys.argv = [str(here / "preview_prompt_stack.py"), *sys.argv[1:]]
    import runpy

    runpy.run_path(str(here / "preview_prompt_stack.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
