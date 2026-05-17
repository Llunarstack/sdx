"""Argv fragments for subprocess / ``sample.py`` parity across tools."""

from __future__ import annotations

from typing import Any, List

from .sampling import normalize_intensity


def extend_sample_argv_visual_design(cmd: List[str], args: Any) -> None:
    """Append ``--visual-design-*`` flags when preset or domain is active."""
    vp = str(getattr(args, "visual_design_preset", "") or "").strip()
    if vp:
        cmd.extend(["--visual-design-preset", vp])
        if bool(getattr(args, "visual_design_negative_pack", False)):
            cmd.append("--visual-design-negative-pack")
        return

    vdd = str(getattr(args, "visual_design_domain", "none") or "none").strip().lower()
    if vdd == "none":
        return
    cmd.extend(["--visual-design-domain", vdd])
    vdi = normalize_intensity(getattr(args, "visual_design_intensity", "standard"))
    cmd.extend(["--visual-design-intensity", vdi])
    if bool(getattr(args, "visual_design_negative_pack", False)):
        cmd.append("--visual-design-negative-pack")


__all__ = ["extend_sample_argv_visual_design"]
