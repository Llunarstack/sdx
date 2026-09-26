"""Director timeline — timed events, concurrent actions, media attach helpers.

Lets users schedule *what happens when* (and attach uploads) without a new
pipeline: events inject into keyframe prompts; refs feed identity/style/motion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "TimedEvent",
    "DirectorTimeline",
    "parse_director_timeline",
    "events_active_at",
    "prompt_for_keyframe",
    "parse_cli_refs",
    "merge_refs_into_edit",
    "media_mode_overrides",
]


@dataclass(slots=True)
class TimedEvent:
    """One authored beat on the master (or shot-local) clock."""

    at_sec: float = 0.0
    until_sec: float | None = None
    prompt: str = ""
    negative: str = ""
    effect: str = ""
    camera: str = ""
    action: str = ""
    refs: tuple[str, ...] = ()
    shot_id: str = ""  # empty = global timeline
    strength: float | None = None  # optional edit_strength override while active
    concurrent: bool = True  # stack with other events at same time
    tag: str = ""

    @property
    def end_sec(self) -> float:
        if self.until_sec is not None:
            return float(self.until_sec)
        return float(self.at_sec) + 0.4


@dataclass
class DirectorTimeline:
    events: list[TimedEvent] = field(default_factory=list)
    refs: list[dict[str, Any]] = field(default_factory=list)
    mode: str = ""  # t2v|i2v|v2v|i2i|cgi
    init_image: str = ""
    style_image: str = ""
    motion_video: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [asdict(e) for e in self.events],
            "refs": list(self.refs),
            "mode": self.mode,
            "init_image": self.init_image,
            "style_image": self.style_image,
            "motion_video": self.motion_video,
            "notes": list(self.notes),
        }


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_event(row: Mapping[str, Any]) -> TimedEvent | None:
    prompt = str(row.get("prompt") or row.get("text") or row.get("description") or "").strip()
    action = str(row.get("action") or row.get("do") or "").strip()
    effect = str(row.get("effect") or row.get("fx") or "").strip()
    camera = str(row.get("camera") or row.get("camera_move") or "").strip()
    if not any((prompt, action, effect, camera)):
        return None
    refs_raw = row.get("refs") or row.get("references") or row.get("images") or ()
    refs: list[str] = []
    if isinstance(refs_raw, str):
        refs = [refs_raw] if refs_raw.strip() else []
    elif isinstance(refs_raw, Sequence):
        refs = [str(x).strip() for x in refs_raw if str(x).strip()]
    until = row.get("until_sec")
    if until is None:
        until = row.get("end_sec")
    strength = row.get("strength")
    if strength is None:
        strength = row.get("edit_strength")
    return TimedEvent(
        at_sec=_as_float(row.get("at_sec") or row.get("t") or row.get("time") or 0.0),
        until_sec=_as_float(until) if until is not None else None,
        prompt=prompt,
        negative=str(row.get("negative") or ""),
        effect=effect,
        camera=camera,
        action=action,
        refs=tuple(refs),
        shot_id=str(row.get("shot_id") or row.get("shot") or ""),
        strength=float(strength) if strength is not None else None,
        concurrent=bool(row.get("concurrent", True)),
        tag=str(row.get("tag") or row.get("id") or ""),
    )


def parse_director_timeline(raw: Any = None, *, scene: Mapping[str, Any] | None = None) -> DirectorTimeline:
    """Parse ``events`` / ``timeline`` / media attach fields from scene or edit."""
    tl = DirectorTimeline()
    data: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        data.update(dict(raw))
    if isinstance(scene, Mapping):
        for k in (
            "events",
            "timeline",
            "references",
            "refs",
            "multimodal_refs",
            "mode",
            "init_image",
            "style_image",
            "style_ref",
            "motion_video",
            "motion_clip",
            "anchor_image",
        ):
            if k in scene and k not in data:
                data[k] = scene[k]
        if "events" not in data and isinstance(scene.get("frontier"), Mapping):
            fe = scene["frontier"].get("events") or scene["frontier"].get("timeline")
            if fe:
                data["events"] = fe

    events_raw = data.get("events") or data.get("timeline") or []
    if isinstance(events_raw, Mapping):
        converted: list[Any] = []
        for k, v in events_raw.items():
            if isinstance(v, str):
                converted.append({"at_sec": k, "prompt": v})
            elif isinstance(v, Mapping):
                converted.append({**dict(v), "at_sec": v.get("at_sec", k)})
        events_raw = converted
    if isinstance(events_raw, Sequence) and not isinstance(events_raw, (str, bytes)):
        for row in events_raw:
            if isinstance(row, str):
                s = row.strip()
                at = 0.0
                body = s
                if ":" in s:
                    head, rest = s.split(":", 1)
                    head = head.strip().lower().rstrip("s")
                    try:
                        at = float(head)
                        body = rest.strip()
                    except ValueError:
                        pass
                ev = _parse_event({"at_sec": at, "prompt": body})
            elif isinstance(row, Mapping):
                ev = _parse_event(row)
            else:
                ev = None
            if ev:
                tl.events.append(ev)

    tl.events.sort(key=lambda e: (e.at_sec, e.tag))

    refs_raw = data.get("references") or data.get("refs") or data.get("multimodal_refs")
    if refs_raw is not None:
        try:
            from pipelines.video.multimodal_ref_bus import parse_multimodal_refs

            pack = parse_multimodal_refs(refs_raw)
            for r in pack.refs:
                tl.refs.append(
                    {
                        "path": r.path,
                        "role": r.role,
                        "modality": r.modality,
                        "strength": r.strength,
                        "tag": r.tag,
                        "notes": r.notes,
                    }
                )
        except Exception:
            if isinstance(refs_raw, Sequence):
                for x in refs_raw:
                    if isinstance(x, str):
                        tl.refs.append({"path": x, "role": "identity"})
                    elif isinstance(x, Mapping) and (x.get("path") or x.get("file")):
                        tl.refs.append(dict(x))

    tl.mode = str(data.get("mode") or "").strip().lower()
    tl.init_image = str(data.get("init_image") or data.get("anchor_image") or data.get("i2i_image") or "").strip()
    tl.style_image = str(data.get("style_image") or data.get("style_ref") or "").strip()
    tl.motion_video = str(data.get("motion_video") or data.get("motion_clip") or data.get("v2v_clip") or "").strip()

    for r in tl.refs:
        path = str(r.get("path") or "")
        role = str(r.get("role") or "")
        mod = str(r.get("modality") or "")
        if not tl.style_image and role == "style" and path:
            tl.style_image = path
        if not tl.motion_video and role == "motion" and path:
            tl.motion_video = path
        if not tl.init_image and role in ("identity", "scene") and mod == "image" and path:
            tl.init_image = path

    if tl.events:
        tl.notes.append(f"events={len(tl.events)}")
    if tl.refs:
        tl.notes.append(f"refs={len(tl.refs)}")
    return tl


def events_active_at(
    events: Sequence[TimedEvent],
    t_sec: float,
    *,
    shot_id: str = "",
) -> list[TimedEvent]:
    """Return events active at time ``t_sec`` (global + matching shot)."""
    active: list[TimedEvent] = []
    for e in events:
        if e.shot_id and shot_id and e.shot_id != shot_id:
            continue
        if e.shot_id and not shot_id:
            continue
        if float(e.at_sec) - 1e-6 <= float(t_sec) <= float(e.end_sec) + 1e-6:
            active.append(e)
    if not active:
        return []
    if all(e.concurrent for e in active):
        return active
    out: list[TimedEvent] = []
    last_exclusive: TimedEvent | None = None
    for e in active:
        if e.concurrent:
            out.append(e)
        else:
            last_exclusive = e
    if last_exclusive is not None:
        out.append(last_exclusive)
    return out


def prompt_for_keyframe(
    base_prompt: str,
    *,
    t_sec: float,
    events: Sequence[TimedEvent],
    shot_id: str = "",
    base_negative: str = "",
) -> tuple[str, str, float | None]:
    """Merge active timed events into keyframe prompt/negative; optional strength."""
    active = events_active_at(events, t_sec, shot_id=shot_id)
    if not active:
        return base_prompt, base_negative, None
    bits: list[str] = [base_prompt] if base_prompt else []
    neg_bits: list[str] = [base_negative] if base_negative else []
    strength: float | None = None
    for e in active:
        frag_parts = [x for x in (e.action, e.effect, e.camera, e.prompt) if x]
        frag = ", ".join(frag_parts)
        if frag:
            bits.append(frag)
        if e.negative:
            neg_bits.append(e.negative)
        if e.strength is not None:
            strength = float(e.strength)
    prompt = ", ".join(b for b in bits if b).strip(", ")
    negative = ", ".join(b for b in neg_bits if b).strip(", ")
    return prompt, negative, strength


def parse_cli_refs(raw_refs: Sequence[str] | None) -> list[dict[str, Any]]:
    """Parse ``--ref path`` or ``--ref path:role`` or ``--ref path:role:strength``."""
    out: list[dict[str, Any]] = []
    for raw in raw_refs or ():
        s = str(raw or "").strip()
        if not s:
            continue
        parts = s.split(":")
        path = s
        role = "identity"
        strength = 0.85
        # Windows drive: D:\a.png:style:0.9
        if len(parts) >= 3 and len(parts[0]) == 1 and parts[0].isalpha():
            path = f"{parts[0]}:{parts[1]}"
            role = parts[2]
            if len(parts) > 3:
                try:
                    strength = float(parts[3])
                except ValueError:
                    pass
        elif len(parts) >= 2 and not (len(parts[0]) == 1 and parts[0].isalpha()):
            path = parts[0]
            role = parts[1]
            if len(parts) > 2:
                try:
                    strength = float(parts[2])
                except ValueError:
                    pass
        out.append({"path": path, "role": role, "strength": strength})
    return out


def merge_refs_into_edit(edit: dict[str, Any], refs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold CLI/scene refs into ``edit.multimodal_refs``."""
    out = dict(edit or {})
    existing = out.get("multimodal_refs") or out.get("references") or out.get("refs") or []
    items: list[Any] = list(existing) if isinstance(existing, list) else []
    for r in refs:
        items.append(dict(r))
    if items:
        out["multimodal_refs"] = items
        out["leaders_stack"] = True
    return out


def media_mode_overrides(mode: str) -> dict[str, Any]:
    """ProcessOptions / assignment overrides for t2v|i2v|v2v|i2i|cgi."""
    m = (mode or "").strip().lower().replace("-", "_")
    if m in ("v2v", "video2video", "vid2vid"):
        return {
            "mode": "v2v",
            "motion_transfer": True,
            "motion_transfer_retrieved": True,
            "edit_strength": 0.48,
            "use_motion_only": False,
            "depth_interpolate": True,
        }
    if m in ("i2i", "img2img", "image2image"):
        return {
            "mode": "i2i",
            "edit_strength": 0.62,
            "keyframe_interval": 4,
            "motion_transfer": False,
            "use_motion_only": False,
        }
    if m in ("cgi", "3d_cgi", "cgi_3d"):
        return {
            "mode": "cgi",
            "motion_grammar": "cgi",
            "edit_strength": 0.50,
            "depth_interpolate": True,
        }
    if m == "i2v":
        return {"mode": "i2v", "edit_strength": 0.55}
    return {"mode": m or "t2v"}
