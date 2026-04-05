"""
Structured scene blueprint -> prompt controls.

The goal is maximum controllability without relying on brittle free-text prompts.
Users can define actors, scene layout, camera, pose intent, object placement, and
size/relationship constraints in JSON; this module compiles that into positive and
negative prompt additions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, list):
        out: List[str] = []
        for x in value:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def _dedupe(tokens: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tokens:
        key = t.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(t.strip())
    return out


def _actor_tokens(actor: Dict[str, Any], index: int) -> Tuple[List[str], List[str]]:
    pos: List[str] = []
    neg: List[str] = []
    label = str(actor.get("id", f"actor_{index + 1}")).strip() or f"actor_{index + 1}"
    role = str(actor.get("role", "")).strip()
    anchor = str(actor.get("spatial_anchor", "") or actor.get("screen_position", "")).strip()
    if anchor:
        pos.append(f"{label} position: {anchor}")

    if role:
        pos.append(f"{label} role: {role}")
    else:
        pos.append(f"{label} subject")

    for k in (
        "identity",
        "appearance",
        "body",
        "wardrobe",
        "pose",
        "expression",
        "props",
        "style",
        "locks",
    ):
        pos.extend(_as_list(actor.get(k)))

    neg.extend(_as_list(actor.get("avoid")))
    return pos, neg


def _relation_tokens(rel: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    pos: List[str] = []
    neg: List[str] = []
    a = str(rel.get("a", "")).strip()
    b = str(rel.get("b", "")).strip()
    kind = str(rel.get("kind", "")).strip()
    detail = str(rel.get("detail", "")).strip()
    if a and b and kind:
        base = f"{a} {kind} {b}"
        if detail:
            base = f"{base}, {detail}"
        pos.append(base)
    neg.extend(_as_list(rel.get("avoid")))
    return pos, neg


def _compile_blueprint_dict(data: Dict[str, Any], strength: float = 1.0) -> Tuple[str, str]:
    pos: List[str] = []
    neg: List[str] = []

    # Global scene controls
    for key in (
        "subject",
        "scene",
        "composition",
        "camera",
        "lighting",
        "style",
        "color_script",
        "background",
        "objects",
        "constraints",
        "quality",
    ):
        pos.extend(_as_list(data.get(key)))

    neg.extend(_as_list(data.get("avoid")))

    # Actors
    for i, actor in enumerate(data.get("actors", []) or []):
        if isinstance(actor, dict):
            p, n = _actor_tokens(actor, i)
            pos.extend(p)
            neg.extend(n)

    # Relations / spatial constraints
    for rel in data.get("relations", []) or []:
        if isinstance(rel, dict):
            p, n = _relation_tokens(rel)
            pos.extend(p)
            neg.extend(n)

    # Optional scale directives
    for scale in data.get("scale_directives", []) or []:
        if isinstance(scale, str) and scale.strip():
            pos.append(scale.strip())

    # Optional anti-artifact directives
    if data.get("anti_artifacts", False):
        neg.extend(
            [
                "warped anatomy",
                "merged limbs",
                "floating objects",
                "inconsistent perspective",
                "duplicate face",
                "extra limbs",
            ]
        )

    # Strength controls how many high-signal scene controls we duplicate with emphasis.
    s = max(0.5, min(float(strength), 2.0))
    if s > 1.0:
        for tag in _as_list(data.get("constraints"))[:6]:
            pos.append(f"({tag})")

    pos = _dedupe(pos)
    neg = _dedupe(neg)
    return ", ".join(pos), ", ".join(neg)


def load_scene_blueprint(path: str, strength: float = 1.0) -> Tuple[str, str]:
    """
    Load scene blueprint JSON and return `(positive_additions, negative_additions)`.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"scene-blueprint not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(data, dict):
        raise ValueError("scene-blueprint must be a JSON object")
    return _compile_blueprint_dict(data, strength=strength)
