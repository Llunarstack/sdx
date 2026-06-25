"""Virtual camera rig presets — lens, body, movement, fps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

__all__ = ["CameraRig", "parse_camera_rig", "rig_to_prompt"]


@dataclass(slots=True)
class CameraRig:
    body: str = ""
    lens_mm: str = ""
    movement: str = ""
    focus: str = ""
    fps: float = 24.0
    aspect: str = ""
    filter: str = ""
    operator_style: str = ""


_PRESETS: Dict[str, CameraRig] = {
    "arri_alexa": CameraRig(body="ARRI Alexa look", lens_mm="50", movement="steadicam", fps=24.0),
    "imax": CameraRig(body="IMAX large format", lens_mm="24", movement="slow crane", fps=24.0),
    "handheld_doc": CameraRig(
        body="documentary handheld", lens_mm="35", movement="handheld", operator_style="documentary"
    ),
    "drone": CameraRig(body="aerial drone", lens_mm="24", movement="orbit", fps=24.0),
    "gopro": CameraRig(body="GoPro wide", lens_mm="16", movement="body mount", fps=60.0),
    "vhs": CameraRig(body="VHS camcorder", lens_mm="wide", movement="handheld", filter="VHS", fps=29.97),
    "security": CameraRig(body="security CCTV", lens_mm="fixed wide", movement="static", fps=15.0),
    "anime_cam": CameraRig(
        body="anime dramatic camera", lens_mm="varied", movement="snap zoom", operator_style="anime"
    ),
}


def parse_camera_rig(raw: Any) -> CameraRig:
    if not raw:
        return CameraRig()
    if isinstance(raw, str):
        key = raw.strip().lower().replace(" ", "_")
        return _PRESETS.get(key, CameraRig(body=raw))
    if not isinstance(raw, Mapping):
        return CameraRig()
    preset = str(raw.get("preset") or raw.get("template") or "")
    base = _PRESETS.get(preset.lower(), CameraRig()) if preset else CameraRig()
    return CameraRig(
        body=str(raw.get("body") or base.body),
        lens_mm=str(raw.get("lens") or raw.get("lens_mm") or base.lens_mm),
        movement=str(raw.get("movement") or raw.get("camera_move") or base.movement),
        focus=str(raw.get("focus") or base.focus),
        fps=float(raw.get("fps") or base.fps or 24.0),
        aspect=str(raw.get("aspect") or base.aspect),
        filter=str(raw.get("filter") or base.filter),
        operator_style=str(raw.get("operator") or raw.get("style") or base.operator_style),
    )


def rig_to_prompt(r: CameraRig) -> str:
    parts = []
    if r.body:
        parts.append(r.body)
    if r.lens_mm:
        parts.append(f"{r.lens_mm}mm lens")
    if r.movement:
        parts.append(f"{r.movement} camera")
    if r.focus:
        parts.append(f"{r.focus} focus")
    if r.filter:
        parts.append(f"{r.filter} aesthetic")
    if r.operator_style:
        parts.append(f"{r.operator_style} camera operator")
    return ", ".join(parts)
