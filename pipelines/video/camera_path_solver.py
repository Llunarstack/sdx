"""
Camera path solver — Runway/Seedance choreography as a keyframed trajectory.

Instead of prompt-only verbs, we emit a continuous path (pan/tilt/dolly/roll)
that can drive motion templates, control images, and velocity ease.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

__all__ = [
    "CameraKeyframe",
    "CameraPath",
    "solve_camera_path",
    "path_to_prompt",
    "path_to_motion_deltas",
]


@dataclass(slots=True)
class CameraKeyframe:
    t: float  # 0..1
    pan: float = 0.0  # degrees, +right
    tilt: float = 0.0  # degrees, +up
    dolly: float = 0.0  # +push in
    roll: float = 0.0
    zoom: float = 1.0


@dataclass(slots=True)
class CameraPath:
    keys: list[CameraKeyframe] = field(default_factory=list)
    preset: str = ""
    notes: list[str] = field(default_factory=list)

    def sample(self, u: float) -> CameraKeyframe:
        if not self.keys:
            return CameraKeyframe(t=u)
        u = float(np_clip(u, 0.0, 1.0))
        if u <= self.keys[0].t:
            return self.keys[0]
        if u >= self.keys[-1].t:
            return self.keys[-1]
        for a, b in zip(self.keys[:-1], self.keys[1:]):
            if a.t <= u <= b.t:
                w = 0.0 if b.t <= a.t else (u - a.t) / (b.t - a.t)
                w = _smooth(w)
                return CameraKeyframe(
                    t=u,
                    pan=_lerp(a.pan, b.pan, w),
                    tilt=_lerp(a.tilt, b.tilt, w),
                    dolly=_lerp(a.dolly, b.dolly, w),
                    roll=_lerp(a.roll, b.roll, w),
                    zoom=_lerp(a.zoom, b.zoom, w),
                )
        return self.keys[-1]


def np_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _lerp(a: float, b: float, w: float) -> float:
    return a + (b - a) * w


def _smooth(t: float) -> float:
    t = np_clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


_PRESET_PATHS: dict[str, list[CameraKeyframe]] = {
    "dolly_in": [
        CameraKeyframe(0.0, dolly=0.0, zoom=1.0),
        CameraKeyframe(1.0, dolly=1.0, zoom=1.15),
    ],
    "orbit": [
        CameraKeyframe(0.0, pan=-25.0),
        CameraKeyframe(0.5, pan=0.0, dolly=0.3),
        CameraKeyframe(1.0, pan=25.0),
    ],
    "hitchcock": [
        CameraKeyframe(0.0, dolly=0.0, zoom=1.0),
        CameraKeyframe(1.0, dolly=1.2, zoom=0.75),
    ],
    "crane_up": [
        CameraKeyframe(0.0, tilt=-8.0, dolly=0.0),
        CameraKeyframe(1.0, tilt=12.0, dolly=0.4),
    ],
    "handheld": [
        CameraKeyframe(0.0, pan=-2.0, tilt=1.0, roll=-0.5),
        CameraKeyframe(0.33, pan=2.5, tilt=-1.5, roll=0.8),
        CameraKeyframe(0.66, pan=-1.5, tilt=2.0, roll=-0.4),
        CameraKeyframe(1.0, pan=1.0, tilt=-0.5, roll=0.2),
    ],
    "static": [CameraKeyframe(0.0), CameraKeyframe(1.0)],
}


def solve_camera_path(
    prompt: str = "",
    *,
    preset: str = "",
    rig_movement: str = "",
    duration_sec: float = 6.0,
) -> CameraPath:
    text = f"{prompt} {rig_movement} {preset}".lower()
    chosen = (preset or "").strip().lower().replace("-", "_")
    if not chosen or chosen not in _PRESET_PATHS:
        rules = [
            (("hitchcock", "vertigo", "dolly zoom"), "hitchcock"),
            (("orbit", "circle", "around"), "orbit"),
            (("crane", "jib", "rise"), "crane_up"),
            (("handheld", "docu", "shaky"), "handheld"),
            (("dolly", "push in", "push-in", "track in"), "dolly_in"),
            (("static", "locked off", "tripod"), "static"),
        ]
        for keys, name in rules:
            if any(k in text for k in keys):
                chosen = name
                break
        else:
            chosen = "dolly_in" if duration_sec >= 4 else "static"
    keys = [
        CameraKeyframe(**{f.name: getattr(kf, f.name) for f in fields(CameraKeyframe)}) for kf in _PRESET_PATHS[chosen]
    ]
    return CameraPath(keys=keys, preset=chosen, notes=[f"preset={chosen}"])


def path_to_prompt(path: CameraPath) -> str:
    if not path.keys:
        return ""
    mid = path.sample(0.5)
    end = path.sample(1.0)
    bits = [f"camera path:{path.preset}"]
    if abs(end.dolly - path.keys[0].dolly) > 0.1:
        bits.append("motivated dolly")
    if abs(end.pan - path.keys[0].pan) > 5:
        bits.append(f"pan {'right' if end.pan > 0 else 'left'}")
    if abs(end.tilt) > 4:
        bits.append("tilt accent")
    if path.preset == "hitchcock":
        bits.append("dolly zoom vertigo effect")
    if path.preset == "handheld":
        bits.append("subtle handheld breathing")
    bits.append(f"mid zoom {mid.zoom:.2f}")
    return ", ".join(bits)


def path_to_motion_deltas(path: CameraPath, *, frames: int) -> list[dict[str, float]]:
    """Per-frame deltas useful for motion templates / control."""
    n = max(1, int(frames))
    out: list[dict[str, float]] = []
    prev = path.sample(0.0)
    for i in range(n):
        u = i / max(1, n - 1)
        cur = path.sample(u)
        out.append(
            {
                "pan": cur.pan - prev.pan,
                "tilt": cur.tilt - prev.tilt,
                "dolly": cur.dolly - prev.dolly,
                "roll": cur.roll - prev.roll,
                "zoom": cur.zoom - prev.zoom,
            }
        )
        prev = cur
    return out
