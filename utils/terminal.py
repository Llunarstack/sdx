"""Terminal helpers for cross-platform CLI output."""

from __future__ import annotations

import os
import sys


def configure_stdio_for_console() -> None:
    """
    Avoid UnicodeEncodeError on Windows terminals using legacy encodings (e.g. cp1252).

    Sets UTF-8 with ``errors="replace"`` so ``--help`` and logging stay robust when
    output includes non-ASCII text (typographic arrows, emoji, etc.).
    """
    if os.name != "nt":
        return
    try:
        stdout_rc = getattr(sys.stdout, "reconfigure", None)
        if callable(stdout_rc):
            stdout_rc(encoding="utf-8", errors="replace")
        stderr_rc = getattr(sys.stderr, "reconfigure", None)
        if callable(stderr_rc):
            stderr_rc(encoding="utf-8", errors="replace")
    except Exception:
        pass
