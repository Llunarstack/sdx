"""Shot list planner: user prompt → timed shots with cinema grammar."""

from __future__ import annotations

import math
import re
from typing import List

from .types import MasterTimeline, ShotSpec, VideoMode, VideoPlan

__all__ = ["plan_video_from_prompt", "split_prompt_into_beats"]


def split_prompt_into_beats(prompt: str) -> List[str]:
    """Split on sentence/clause boundaries for multi-shot planning."""
    p = (prompt or "").strip()
    if not p:
        return ["cinematic scene"]
    parts = re.split(r"[.;]\s+|\s+then\s+|\s+and then\s+", p, flags=re.I)
    parts = [x.strip() for x in parts if x.strip()]
    return parts or [p]


def _infer_shot_type(chunk: str) -> tuple[str, str, str]:
    from frontier.cinema.shot_grammar import ShotGrammar

    pos, neg = ShotGrammar().fragments(chunk)
    low = chunk.lower()
    if "establishing" in low or "wide" in low:
        st = "establishing"
    elif "close" in low or "portrait" in low:
        st = "close_up"
    elif "over the shoulder" in low or "ots" in low:
        st = "over_shoulder"
    elif "pov" in low or "point of view" in low:
        st = "pov"
    else:
        st = "medium"
    return st, pos, neg


def plan_video_from_prompt(
    prompt: str,
    *,
    mode: VideoMode = VideoMode.T2V,
    duration_sec: float = 6.0,
    fps: float = 24.0,
    width: int = 1280,
    height: int = 720,
    max_shots: int = 5,
    anchor_image_note: str = "",
) -> VideoPlan:
    """
    Heuristic shot planner (no LLM required).

    Splits prompt into beats; assigns duration proportionally; adds shot grammar fragments.
    """
    beats = split_prompt_into_beats(prompt)
    beats = beats[: max(1, int(max_shots))]
    n = len(beats)
    base = max(0.8, float(duration_sec) / n)
    shots: List[ShotSpec] = []
    for i, beat in enumerate(beats):
        st, pos, neg = _infer_shot_type(beat)
        seg_prompt = beat
        if pos and pos.lower() not in beat.lower():
            seg_prompt = f"{beat}, {pos}"
        if mode == VideoMode.I2V and i == 0 and anchor_image_note:
            seg_prompt = f"{seg_prompt}, {anchor_image_note}"
        dur = base
        if i == 0 and st == "establishing":
            dur = base * 1.15
        elif st == "close_up":
            dur = base * 0.9
        shots.append(
            ShotSpec(
                index=i,
                prompt=seg_prompt.strip(),
                duration_sec=round(dur, 3),
                shot_type=st,
                lens_hint="50mm",
                negative=neg,
                motion_hint="slow pan" if st == "establishing" else "natural motion",
            )
        )
    total = sum(s.duration_sec for s in shots)
    scale = float(duration_sec) / max(0.1, total)
    for s in shots:
        s.duration_sec = round(s.duration_sec * scale, 3)
    ar = "16:9"
    if height > width:
        ar = "9:16"
    elif abs(width / max(1, height) - 1.0) < 0.05:
        ar = "1:1"
    timeline = MasterTimeline(fps=fps, width=width, height=height, duration_sec=float(duration_sec), aspect_ratio=ar)
    return VideoPlan(
        mode=mode,
        user_prompt=prompt,
        timeline=timeline,
        shots=shots,
        global_negative="flicker, morphing faces, warped anatomy, watermark",
        metadata={"beat_count": n, "planner": "heuristic"},
    )


def estimate_keyframe_count(duration_sec: float, fps: float, interval: int) -> int:
    frames = max(1, int(math.ceil(duration_sec * fps)))
    return max(2, len(range(0, frames, max(1, interval))) + 1)
