"""
Run Mojo sources from Python when the Modular CLI is installed (``mojo`` or ``magic`` on PATH).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

from sdx_native.native_tools import mojo_cli_path


def run_mojo_file(
    mojo_src: Path,
    extra_args: Optional[Sequence[str]] = None,
    *,
    cwd: Optional[Path] = None,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    cli = mojo_cli_path()
    if not cli:
        raise FileNotFoundError("Install Modular Mojo and add `mojo` or `magic` to PATH")
    if not mojo_src.is_file():
        raise FileNotFoundError(mojo_src)
    # Current Modular docs: ``mojo hello.mojo``. Some builds also accept ``mojo run <file>``.
    base: List[str] = [cli, str(mojo_src)]
    if extra_args:
        base.extend(extra_args)
    r = subprocess.run(
        base,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
    )
    if r.returncode != 0 and "run" not in base[1:2]:
        alt: List[str] = [cli, "run", str(mojo_src)]
        if extra_args:
            alt.extend(extra_args)
        r = subprocess.run(
            alt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
    return r
