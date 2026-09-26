"""Composition inventions 5–19: SAT-layout, scene graph, priors, atlas, neuro-CFG."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Constraint",
    "SatLayoutReport",
    "check_sat_layout",
    "SceneGraph",
    "compile_scene_graph",
    "anti_centering_addon",
    "zorder_addon",
    "occlusion_grammar_expand",
    "neuro_symbolic_cfg_scale",
    "FailureAtlas",
    "load_failure_atlas",
    "synthetic_constraint_caption",
    "binding_contrastive_pair",
    "multiplicity_curriculum_weight",
]


@dataclass(slots=True)
class Constraint:
    kind: str  # count|left_of|right_of|above|below|negation|binding
    a: str = ""
    b: str = ""
    n: int = 0
    raw: str = ""


@dataclass
class SatLayoutReport:
    constraints: list[Constraint] = field(default_factory=list)
    satisfied: list[bool] = field(default_factory=list)
    score: float = 1.0
    reject: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraints": [asdict(c) for c in self.constraints],
            "satisfied": list(self.satisfied),
            "score": self.score,
            "reject": self.reject,
            "notes": list(self.notes),
        }


def _parse_constraints(prompt: str) -> list[Constraint]:
    p = str(prompt or "")
    out: list[Constraint] = []
    for m in re.finditer(r"\b(\d+)\s+([a-z]+)", p, re.I):
        out.append(Constraint(kind="count", n=int(m.group(1)), a=m.group(2).lower(), raw=m.group(0)))
    for rel in ("left of", "right of", "above", "below", "behind", "in front of"):
        for m in re.finditer(rf"([a-z0-9_ -]+?)\s+{re.escape(rel)}\s+([a-z0-9_ -]+?)(?:,|\.|$)", p, re.I):
            out.append(
                Constraint(
                    kind=rel.replace(" ", "_"),
                    a=m.group(1).strip().lower(),
                    b=m.group(2).strip().lower(),
                    raw=m.group(0),
                )
            )
    for m in re.finditer(r"\b(?:no|without)\s+([a-z0-9][\w -]{1,30})", p, re.I):
        out.append(Constraint(kind="negation", a=m.group(1).strip().lower(), raw=m.group(0)))
    return out


def check_sat_layout(
    prompt: str,
    *,
    detections: list[dict[str, Any]] | None = None,
    reject_threshold: float = 0.45,
) -> SatLayoutReport:
    """
    CSP-lite: if detections provided (name, cx, cy, count), score constraints;
    else return unevaluated constraints for a VLM/detector judge later.
    """
    cons = _parse_constraints(prompt)
    if not cons:
        return SatLayoutReport(notes=["no constraints"])
    if not detections:
        return SatLayoutReport(
            constraints=cons, satisfied=[False] * len(cons), score=0.0, notes=["awaiting detections"]
        )

    # Index detections by label substring
    sat: list[bool] = []
    for c in cons:
        ok = True
        if c.kind == "count":
            n = sum(1 for d in detections if c.a in str(d.get("label", "")).lower())
            ok = n == c.n
        elif c.kind in ("left_of", "right_of", "above", "below"):
            as_ = [d for d in detections if c.a.split()[-1] in str(d.get("label", "")).lower()]
            bs_ = [d for d in detections if c.b.split()[-1] in str(d.get("label", "")).lower()]
            if not as_ or not bs_:
                ok = False
            else:
                ax, ay = float(as_[0].get("cx", 0.5)), float(as_[0].get("cy", 0.5))
                bx, by = float(bs_[0].get("cx", 0.5)), float(bs_[0].get("cy", 0.5))
                if c.kind == "left_of":
                    ok = ax < bx
                elif c.kind == "right_of":
                    ok = ax > bx
                elif c.kind == "above":
                    ok = ay < by
                else:
                    ok = ay > by
        elif c.kind == "negation":
            ok = not any(c.a in str(d.get("label", "")).lower() for d in detections)
        sat.append(ok)
    score = float(sum(sat) / max(len(sat), 1))
    return SatLayoutReport(
        constraints=cons,
        satisfied=sat,
        score=score,
        reject=score < reject_threshold,
        notes=[f"sat score={score:.2f}"],
    )


@dataclass
class SceneGraph:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}


def compile_scene_graph(graph: SceneGraph | dict[str, Any]) -> dict[str, Any]:
    """Scene graph JSON → box layout + relation prompt shard."""
    if isinstance(graph, dict):
        g = SceneGraph(nodes=list(graph.get("nodes") or []), edges=list(graph.get("edges") or []))
    else:
        g = graph
    n = max(1, len(g.nodes))
    regions = []
    for i, node in enumerate(g.nodes):
        box = node.get("box")
        if not box:
            # fan out horizontally
            w = 0.85 / n
            box = [0.05 + i * w, 0.15, 0.05 + (i + 1) * w - 0.02, 0.9]
        regions.append(
            {
                "id": str(node.get("id") or f"n{i}"),
                "prompt": str(node.get("label") or node.get("prompt") or f"object {i}"),
                "box": box,
            }
        )
    rel_bits = []
    for e in g.edges:
        rel_bits.append(f"{e.get('src')} {e.get('rel', 'near')} {e.get('dst')}")
    return {
        "mode": "scene_graph",
        "anti_bleed": True,
        "regions": regions,
        "global_prompt": ", ".join(rel_bits),
        "edges": g.edges,
    }


def anti_centering_addon() -> tuple[str, str]:
    return (
        "off-center subject, rule of thirds composition, asymmetric framing",
        "perfectly centered subject, bullseye composition, dead-center framing",
    )


def zorder_addon(prompt: str) -> tuple[str, str]:
    """Inject front/mid/back layer language when depth words appear or always lightly."""
    p = str(prompt or "").lower()
    pos = "foreground subject, midground secondary, background environment, clear depth layers"
    neg = "flat cardboard depth, everything same distance, z-fighting"
    if any(k in p for k in ("behind", "in front", "foreground", "background", "depth")):
        pos = f"explicit z-order, {pos}"
    return pos, neg


_OCCLUSION_TEMPLATES = [
    "{a} behind a fence, {b} visible through gaps",
    "{a} partially occluded by {occluder}, {b} fully visible",
    "{a} in front of {occluder}, casting soft contact shadow",
]


def occlusion_grammar_expand(a: str = "person", b: str = "person", occluder: str = "fence") -> str:
    t = random.choice(_OCCLUSION_TEMPLATES)
    return t.format(a=a, b=b, occluder=occluder)


def neuro_symbolic_cfg_scale(
    base_cfg: float, sat_score: float, *, min_mult: float = 0.75, max_mult: float = 1.25
) -> float:
    """Lower CFG when constraints already satisfied; raise when failing."""
    # sat_score 1 → min_mult, 0 → max_mult
    s = float(max(0.0, min(1.0, sat_score)))
    mult = max_mult + (min_mult - max_mult) * s
    return float(base_cfg) * float(mult)


@dataclass
class FailureAtlas:
    modes: list[dict[str, Any]] = field(default_factory=list)

    def prompts_for(self, mode_id: str, n: int = 5) -> list[str]:
        for m in self.modes:
            if m.get("id") == mode_id:
                pool = list(m.get("prompts") or [])
                return pool[:n]
        return []


_ATLAS_MODES = [
    ("count_under", "Wrong count (too few)", ["exactly four red apples on a table", "3girls standing in a row"]),
    ("count_over", "Wrong count (too many)", ["exactly two cats, no more", "a single bicycle"]),
    ("bind_color", "Attribute binding", ["red cube left of blue sphere", "green backpack on a brown dog"]),
    ("spatial_left_right", "Left/right", ["the lamp is left of the chair", "person on the right side of frame"]),
    ("negation", "Negation ignored", ["a cat with no hat", "portrait without watermark or text"]),
    ("hands", "Hand anatomy", ["close-up of hands holding a coffee cup", "counting on five fingers"]),
    ("face_far", "Distant face smear", ["crowd scene with recognizable faces", "two people talking across a room"]),
    ("glyph", "Text rendering", ['poster that says "OPEN"', "street sign reading MAIN ST"]),
    ("multi_bleed", "Character bleed", ["2girls different outfits, not twins", "boy in red, girl in blue, no swap"]),
    ("occlusion", "Occlusion", ["person behind chain-link fence", "cat partially behind a vase"]),
    ("reflection", "Reflection", ["person reflected in wet street", "mirror selfie with correct chirality"]),
    ("long_prompt", "Long prompt tail", ["masterpiece, " + ", ".join([f"detail{i}" for i in range(40)])]),
    ("plastic", "Plastic skin", ["natural skin texture portrait, pores visible"]),
    ("extra_limb", "Extra limbs", ["yoga pose, anatomically correct", "dancer mid-leap"]),
    ("object_fusion", "Merged objects", ["two distinct wine glasses side by side"]),
    ("style_mud", "Style mud", ["in the style of ukiyo-e, sharp linework"]),
    ("lighting_physics", "Lighting", ["single hard side light, long cast shadow"]),
    ("material_metal", "Materials", ["brushed aluminum cylinder, sharp reflections"]),
    ("transparency", "Glass/transparency", ["clear glass cup with water and straw"]),
    ("text_small", "Small text", ["business card with tiny crisp letters"]),
    ("rare_count", "Rare counts", ["seven candles on a cake", "eight matching chairs"]),
    ("negation_color", "Negation color", ["a car that is not red", "flowers without yellow"]),
    ("relation_stack", "Stacked relations", ["cup on book on table under lamp"]),
    ("action_multi", "Multi-action", ["A waving while B sits reading"]),
    ("crop_fullbody", "Full body crop", ["full body standing person, head to toes in frame"]),
    ("background_amnesia", "BG consistency", ["detailed kitchen background, sharp cabinets"]),
    ("cfg_burn", "CFG burn", ["soft natural colors, no oversat neon skin"]),
]


def load_failure_atlas() -> FailureAtlas:
    modes = [{"id": a, "name": b, "prompts": c} for a, b, c in _ATLAS_MODES]
    return FailureAtlas(modes=modes)


def save_failure_atlas_jsonl(path: str | Path) -> Path:
    atlas = load_failure_atlas()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for m in atlas.modes:
            for pr in m["prompts"]:
                f.write(json.dumps({"mode": m["id"], "name": m["name"], "prompt": pr}) + "\n")
    return p


def synthetic_constraint_caption(rng: random.Random | None = None) -> str:
    """Generate synthetic count/spatial/negation captions for training (#77)."""
    r = rng or random.Random()
    colors = ["red", "blue", "green", "yellow", "black", "white"]
    objs = ["cube", "sphere", "cat", "mug", "book", "apple"]
    n = r.randint(2, 5)
    c1, c2 = r.sample(colors, 2)
    o1, o2 = r.sample(objs, 2)
    rel = r.choice(["left of", "right of", "above", "below"])
    neg = r.choice(["no text", "without watermark", "no hat"])
    return f"{n} {c1} {o1}s, one {c2} {o2} {rel} them, {neg}"


def binding_contrastive_pair(caption: str) -> tuple[str, str]:
    """Return (win_caption, lose_caption_with_swapped_colors) for DPO (#16)."""
    colors = re.findall(r"\b(red|blue|green|yellow|black|white|orange|purple)\b", caption, re.I)
    if len(colors) >= 2:
        lose = caption
        a, b = colors[0], colors[1]
        lose = re.sub(rf"\b{a}\b", "§TMP§", lose, count=1, flags=re.I)
        lose = re.sub(rf"\b{b}\b", a, lose, count=1, flags=re.I)
        lose = lose.replace("§TMP§", b, 1)
        return caption, lose
    return caption, caption + ", attribute swap error"


def multiplicity_curriculum_weight(num_subjects: int, step: int, total_steps: int) -> float:
    """Ramp weight for multi-subject examples over training (#15)."""
    prog = float(step) / float(max(total_steps, 1))
    # Early: favor 1 subject; late: upweight 3+
    if num_subjects <= 1:
        return 1.0 - 0.3 * prog
    if num_subjects == 2:
        return 0.6 + 0.4 * prog
    return 0.3 + 0.9 * prog
