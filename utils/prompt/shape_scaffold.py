"""
Auto shape/composition scaffold inferred from free-text prompts.

This bridges "human art learning" ideas (big shapes -> relations -> detail)
into existing prompt controls by synthesizing a scene blueprint dict.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .scene_blueprint import compile_scene_blueprint_dict

_ACTOR_HINTS = (
    "man",
    "woman",
    "girl",
    "boy",
    "person",
    "character",
    "warrior",
    "knight",
    "mage",
    "robot",
    "dragon",
    "cat",
    "dog",
    "car",
)

_CAMERA_HINTS = (
    "close-up",
    "close up",
    "wide shot",
    "full body",
    "portrait",
    "overhead",
    "low angle",
    "high angle",
    "cinematic",
)

_LIGHT_HINTS = ("rim light", "soft light", "hard light", "dramatic lighting", "sunset", "neon", "moonlight")

_RELATION_PATTERNS = (
    (re.compile(r"\bleft of\b", flags=re.IGNORECASE), "left_of"),
    (re.compile(r"\bright of\b", flags=re.IGNORECASE), "right_of"),
    (re.compile(r"\bin front of\b", flags=re.IGNORECASE), "in_front_of"),
    (re.compile(r"\bbehind\b", flags=re.IGNORECASE), "behind"),
    (re.compile(r"\bon top of\b", flags=re.IGNORECASE), "on_top_of"),
    (re.compile(r"\bholding\b", flags=re.IGNORECASE), "holding"),
    (re.compile(r"\bnear\b", flags=re.IGNORECASE), "near"),
)


def _extract_phrases(prompt: str) -> List[str]:
    parts = [p.strip() for p in str(prompt).split(",")]
    return [p for p in parts if p]


def _extract_actors(prompt: str, max_actors: int = 4) -> List[Dict[str, Any]]:
    phrases = _extract_phrases(prompt)
    actors: List[Dict[str, Any]] = []
    for ph in phrases:
        low = ph.lower()
        if any(h in low for h in _ACTOR_HINTS):
            actors.append(
                {
                    "id": f"actor_{len(actors) + 1}",
                    "role": ph,
                    "appearance": [ph],
                    "locks": ["consistent proportions", "clean silhouette"],
                }
            )
        if len(actors) >= max(1, int(max_actors)):
            break
    if not actors:
        actors.append(
            {
                "id": "actor_1",
                "role": "main subject",
                "appearance": ["clear primary silhouette"],
                "locks": ["stable anatomy"],
            }
        )
    return actors


def _extract_relations(prompt: str) -> List[Dict[str, Any]]:
    low = str(prompt).lower()
    rels: List[Dict[str, Any]] = []
    for rx, tag in _RELATION_PATTERNS:
        if rx.search(low):
            rels.append({"a": "actor_1", "kind": tag, "b": "actor_2" if "actor_2" in low else "subject_2"})
    return rels


def infer_shape_blueprint(prompt: str, max_actors: int = 4) -> Dict[str, Any]:
    low = str(prompt).lower()
    phrases = _extract_phrases(prompt)
    camera = [p for p in phrases if any(h in p.lower() for h in _CAMERA_HINTS)]
    lighting = [p for p in phrases if any(h in p.lower() for h in _LIGHT_HINTS)]
    constraints = [
        "clear foreground/midground/background separation",
        "anatomically plausible proportions",
        "coherent perspective and horizon",
        "no merged limbs, no duplicate body parts",
    ]
    if "text" in low or "title" in low or "logo" in low:
        constraints.append("legible text with correct spelling")
    return {
        "subject": phrases[:2] or [prompt],
        "composition": ["strong silhouette readability", "shape-first composition"],
        "camera": camera or ["balanced camera framing"],
        "lighting": lighting or ["coherent single-key lighting"],
        "style": ["high-fidelity, intentional brushwork/texture"],
        "constraints": constraints,
        "actors": _extract_actors(prompt, max_actors=max_actors),
        "relations": _extract_relations(prompt),
        "anti_artifacts": True,
    }


def compile_shape_scaffold(prompt: str, strength: float = 1.0, max_actors: int = 4) -> Tuple[str, str, Dict[str, Any]]:
    blueprint = infer_shape_blueprint(prompt, max_actors=max_actors)
    pos, neg = compile_scene_blueprint_dict(blueprint, strength=strength)
    return pos, neg, blueprint
