"""Witness Lens — per-shot embodied viewpoint (who is watching changes the frame)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

__all__ = ["WitnessShotPlan", "parse_witness_config", "plan_witness_lens"]

_LENS_PROMPTS: Dict[str, tuple[str, str]] = {
    "child": ("low camera height child POV, wide-eyed framing, wonder", "adult eye-level detached"),
    "cctv": ("CCTV surveillance angle, slight fish-eye, timestamp overlay feel", "cinematic beauty lighting"),
    "lover": ("intimate close proximity, soft bokeh, tender framing", "wide detached documentary"),
    "victim": ("dutch angle unease, tight claustrophobic framing", "heroic low angle power"),
    "stranger": ("candid off-center voyeur framing, subject unaware", "posed symmetrical portrait"),
    "archivist": ("flat documentary light, neutral observational distance", "dramatic stylized noir"),
    "deity": ("god's eye high angle, vast scale, subjects small", "ground level immersive"),
    "drone": ("aerial witness, detached overview, geometric composition", "handheld intimacy"),
}


@dataclass(slots=True)
class WitnessShotPlan:
    shot_id: str
    lens: str
    prompt_suffix: str
    negative_suffix: str


def parse_witness_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, Mapping):
        return {"enabled": bool(raw.get("enabled", True)), "default": str(raw.get("default") or "neutral")}
    return {"enabled": bool(raw)}


def _shot_witness(shot: Any, default: str) -> str:
    w = str(getattr(shot, "witness", "") or getattr(shot, "pov_witness", "") or "").lower()
    if w:
        return w
    p = str(getattr(shot, "prompt", "")).lower()
    for key in _LENS_PROMPTS:
        if key in p:
            return key
    return default if default in _LENS_PROMPTS else ""


def plan_witness_lens(
    shots: Sequence[Any],
    *,
    config: Mapping[str, Any],
) -> List[WitnessShotPlan]:
    if not config.get("enabled"):
        return []
    default = str(config.get("default") or "")
    plans: List[WitnessShotPlan] = []
    for sh in shots:
        sid = str(getattr(sh, "id", ""))
        lens = _shot_witness(sh, default)
        if not lens:
            continue
        pos, neg = _LENS_PROMPTS.get(lens, (f"{lens} witness perspective", ""))
        plans.append(WitnessShotPlan(shot_id=sid, lens=lens, prompt_suffix=pos, negative_suffix=neg))
    return plans
