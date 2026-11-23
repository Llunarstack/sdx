"""Invention stack: prompt passes for known T2I failure modes (counting, binding, negation, anatomy, plastic look), multi-character cast, corpus retrieval and the ``sample.py`` wiring."""

from __future__ import annotations

import json
import math
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config.styles import detect_medium

from utils.prompt.scene import (
    MULTI_SUBJECT_NEGATIVE_EXTRAS,
    MULTI_SUBJECT_POSITIVE_EXTRAS,
    merge_character_sheet_positives,
)
from utils.prompt.text import merge_csv_unique

_COLOR = (
    r"red|blue|green|yellow|orange|purple|pink|black|white|brown|gray|grey|"
    r"cyan|magenta|gold|silver|teal|navy|beige"
)
_OBJECT = (
    r"cube|sphere|ball|box|cat|dog|girl|boy|man|woman|car|book|cup|bottle|"
    r"apple|bird|horse|chair|table|lamp|tree|flower|sword|shield|hat|dress|"
    r"hoodie|jacket|catgirl|fox|dragon"
)
_SPATIAL = (
    r"left of|right of|above|below|under|over|next to|beside|behind|in front of|"
    r"on top of|inside|outside"
)


@dataclass(slots=True)
class BindTriple:
    subject: str
    attributes: list[str] = field(default_factory=list)
    relation: str = ""
    other: str = ""

    def ownership_phrase(self) -> str:
        attrs = ", ".join(self.attributes) if self.attributes else ""
        core = f"{attrs} {self.subject}".strip() if attrs else self.subject
        if self.relation and self.other:
            return f"{core} {self.relation} {self.other}".strip()
        return core


@dataclass
class BindLockPlan:
    triples: list[BindTriple] = field(default_factory=list)
    positive_addon: str = ""
    negative_addon: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "triples": [asdict(t) for t in self.triples],
            "positive_addon": self.positive_addon,
            "negative_addon": self.negative_addon,
            "notes": list(self.notes),
        }


def plan_bindlock(prompt: str) -> BindLockPlan:
    text = str(prompt or "")
    triples: list[BindTriple] = []
    # Pattern: <color> <object> <spatial> <color> <object>
    pat = re.compile(
        rf"\b((?:{_COLOR})\s+)?({_OBJECT})\b(?:\s+({_SPATIAL})\s+((?:{_COLOR})\s+)?({_OBJECT}))?",
        re.IGNORECASE,
    )
    for m in pat.finditer(text):
        c1, o1, rel, c2, o2 = m.groups()
        attrs = [c1.strip()] if c1 else []
        other = ""
        if o2:
            other_attrs = [c2.strip()] if c2 else []
            other = f"{' '.join(other_attrs)} {o2}".strip()
        triples.append(
            BindTriple(
                subject=o1.lower(),
                attributes=[a.lower() for a in attrs if a],
                relation=(rel or "").lower(),
                other=other.lower(),
            )
        )

    # Also catch "N red apples" style
    count_pat = re.compile(rf"\b(\d+)\s+((?:{_COLOR})\s+)?({_OBJECT})s?\b", re.IGNORECASE)
    for m in count_pat.finditer(text):
        n, c, o = m.groups()
        attrs = [c.strip().lower()] if c else []
        attrs.append(f"count:{n}")
        triples.append(BindTriple(subject=o.lower(), attributes=attrs))

    # Dedup by subject+attrs
    seen: set[str] = set()
    uniq: list[BindTriple] = []
    for t in triples:
        key = f"{t.subject}|{','.join(t.attributes)}|{t.relation}|{t.other}"
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)

    pos_bits = [t.ownership_phrase() for t in uniq]
    # Explicit binding language
    pos_bits.extend(
        [
            "correct attribute binding",
            "each object keeps its own color",
            "no attribute swap between objects",
        ]
    )
    neg_bits = [
        "attribute swap",
        "color swap",
        "wrong color on wrong object",
        "swapped positions",
        "merged objects",
        "incorrect binding",
    ]
    # Cross-negatives: don't put A's color on B
    colors = []
    subjects = []
    for t in uniq:
        subjects.append(t.subject)
        for a in t.attributes:
            if re.fullmatch(_COLOR, a, re.IGNORECASE):
                colors.append((t.subject, a))
    for subj, col in colors:
        for other in subjects:
            if other != subj:
                neg_bits.append(f"{col} {other}")

    return BindLockPlan(
        triples=uniq,
        positive_addon=", ".join(dict.fromkeys(pos_bits)),
        negative_addon=", ".join(dict.fromkeys(neg_bits)),
        notes=[f"{len(uniq)} bind triples"],
    )


def apply_bindlock_to_prompts(positive: str, negative: str = "", plan: BindLockPlan | None = None) -> tuple[str, str]:
    plan = plan or plan_bindlock(positive)
    pos = str(positive or "").rstrip(",")
    neg = str(negative or "").rstrip(",")
    if plan.positive_addon and plan.positive_addon.lower() not in pos.lower():
        pos = f"{pos}, {plan.positive_addon}"
    if plan.negative_addon:
        neg = f"{neg}, {plan.negative_addon}".strip(", ")
    return pos.strip(", "), neg.strip(", ")


_WORD_COUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "a pair of": 2,
    "a couple of": 2,
}

_COUNT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|a pair of|a couple of)\s+"
    r"([a-z][a-z0-9_-]{1,24})s?\b",
    re.IGNORECASE,
)
_BOORU_COUNT = re.compile(r"\b(\d+)(girl|boy|girls|boys|people|cats|dogs)\b", re.IGNORECASE)


@dataclass(slots=True)
class CountSpec:
    n: int
    noun: str


@dataclass
class CountGatePlan:
    specs: list[CountSpec] = field(default_factory=list)
    positive_addon: str = ""
    negative_addon: str = ""
    boxes: list[list[float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "specs": [asdict(s) for s in self.specs],
            "positive_addon": self.positive_addon,
            "negative_addon": self.negative_addon,
            "boxes": self.boxes,
            "notes": list(self.notes),
        }


def _parse_n(raw: str) -> int:
    r = raw.lower().strip()
    if r.isdigit():
        return int(r)
    return int(_WORD_COUNTS.get(r, 0))


def plan_countgate(prompt: str) -> CountGatePlan:
    text = str(prompt or "")
    specs: list[CountSpec] = []
    for m in _COUNT_RE.finditer(text):
        n = _parse_n(m.group(1))
        noun = m.group(2).lower().rstrip("s")
        if n >= 1:
            specs.append(CountSpec(n=n, noun=noun))
    for m in _BOORU_COUNT.finditer(text):
        n = int(m.group(1))
        noun = m.group(2).lower().rstrip("s")
        specs.append(CountSpec(n=n, noun=noun))

    # Dedup by noun keep max n
    by_noun: dict[str, int] = {}
    for s in specs:
        by_noun[s.noun] = max(by_noun.get(s.noun, 0), s.n)
    specs = [CountSpec(n=n, noun=k) for k, n in by_noun.items()]

    pos = []
    neg = [
        "wrong number of subjects",
        "extra subjects",
        "missing subjects",
        "merged duplicates",
        "cloned identical copies overlapping",
    ]
    primary_n = 0
    primary_noun = "subject"
    for s in specs:
        pos.append(f"exactly {s.n} distinct {s.noun}{'s' if s.n != 1 else ''}")
        pos.append(f"count:{s.n} {s.noun}")
        if s.n > primary_n:
            primary_n = s.n
            primary_noun = s.noun
        if s.n >= 2:
            pos.append("clearly separated instances")
            neg.append(f"{s.n + 1} {s.noun}s")
            if s.n > 1:
                neg.append(f"{s.n - 1} {s.noun}s")

    boxes = count_grid_boxes(primary_n) if primary_n >= 2 else []
    return CountGatePlan(
        specs=specs,
        positive_addon=", ".join(dict.fromkeys(pos)),
        negative_addon=", ".join(dict.fromkeys(neg)),
        boxes=boxes,
        notes=[f"primary {primary_n}×{primary_noun}"],
    )


def count_grid_boxes(n: int) -> list[list[float]]:
    """Axis-aligned non-overlap priors for regional binding."""
    n = max(0, min(int(n), 8))
    if n <= 1:
        return [[0.1, 0.1, 0.9, 0.9]] if n == 1 else []
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols
    boxes: list[list[float]] = []
    for i in range(n):
        r, c = divmod(i, cols)
        x0 = 0.05 + c * (0.9 / cols) + 0.02
        x1 = 0.05 + (c + 1) * (0.9 / cols) - 0.02
        y0 = 0.08 + r * (0.85 / rows) + 0.02
        y1 = 0.08 + (r + 1) * (0.85 / rows) - 0.02
        boxes.append([x0, y0, min(x1, 0.95), min(y1, 0.95)])
    return boxes


def apply_countgate_prompts(positive: str, negative: str = "", plan: CountGatePlan | None = None) -> tuple[str, str]:
    plan = plan or plan_countgate(positive)
    pos = str(positive or "").rstrip(",")
    neg = str(negative or "").rstrip(",")
    if plan.positive_addon:
        pos = f"{pos}, {plan.positive_addon}"
    if plan.negative_addon:
        neg = f"{neg}, {plan.negative_addon}".strip(", ")
    return pos.strip(", "), neg.strip(", ")


@dataclass
class HydraSlot:
    slot_id: str
    prompt: str
    box: list[float]


@dataclass
class HydraSlotPlan:
    slots: list[HydraSlot] = field(default_factory=list)
    global_prompt: str = ""
    box_layout: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_hydra_slots(prompt: str, *, max_slots: int = 6) -> HydraSlotPlan:
    cg = plan_countgate(prompt)
    if not cg.specs:
        return HydraSlotPlan(global_prompt=prompt)
    primary = max(cg.specs, key=lambda s: s.n)
    n, noun = min(int(primary.n), int(max_slots)), primary.noun
    if n < 2:
        return HydraSlotPlan(global_prompt=prompt)
    boxes = cg.boxes or count_grid_boxes(n)
    slots = [
        HydraSlot(
            slot_id=f"{noun}_{i + 1}",
            prompt=f"distinct {noun} instance {i + 1}, unique pose, unique silhouette",
            box=list(boxes[i] if i < len(boxes) else [0.1, 0.1, 0.9, 0.9]),
        )
        for i in range(n)
    ]
    layout = {
        "mode": "hydra_slots",
        "anti_bleed": True,
        "global_prompt": prompt,
        "regions": [{"id": s.slot_id, "prompt": s.prompt, "box": s.box} for s in slots],
    }
    return HydraSlotPlan(slots=slots, global_prompt=prompt, box_layout=layout)


@dataclass
class FailureOracleReport:
    risks: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_failures(prompt: str, *, negative: str = "") -> FailureOracleReport:
    p = str(prompt or "").lower()
    scores: dict[str, float] = {}

    # Counting risk
    if re.search(r"\b([2-9]|two|three|four|five|six)\s+\w+", p) or re.search(r"\b[2-9](girl|boy|cat)", p):
        scores["count"] = 0.85
    # Binding / spatial
    if re.search(r"\b(left of|right of|above|below|next to)\b", p) or re.search(
        r"\b(red|blue|green)\s+\w+\b.+\b(red|blue|green)\s+\w+", p
    ):
        scores["binding"] = 0.8
    # Negation
    if re.search(r"\b(no|without|not|n't)\s+\w+", p):
        scores["negation"] = 0.9
    # Hands / anatomy
    if re.search(r"\b(hand|hands|fingers|holding|grasp|pose)\b", p):
        scores["anatomy"] = 0.7
    # Text / glyph
    if '"' in prompt or "'" in prompt or re.search(r"\b(text|sign|logo|poster|typography)\b", p):
        scores["glyph"] = 0.75
    # Multi-character bleed
    if re.search(r"\b(2girl|3girl|two girls|multiple|both wearing)\b", p):
        scores["multi_char"] = 0.8
    # Plastic / beauty
    if re.search(r"\b(portrait|skin|beauty|face|close-?up)\b", p):
        scores["plastic"] = 0.55
    # Long prompt truncation
    if len(p) > 400:
        scores["long_prompt"] = 0.65

    risks = [k for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v >= 0.5]
    return FailureOracleReport(risks=risks, scores=scores, notes=[f"{len(risks)} elevated risks"])


def repair_plan_from_failures(report: FailureOracleReport) -> FailureOracleReport:
    repairs: list[dict[str, Any]] = []
    for risk in report.risks:
        if risk == "count":
            repairs.append({"module": "countgate", "action": "apply_countgate_prompts"})
        elif risk == "binding":
            repairs.append({"module": "bindlock", "action": "apply_bindlock_to_prompts"})
        elif risk == "negation":
            repairs.append({"module": "negatron", "action": "apply_negatron"})
        elif risk == "anatomy":
            repairs.append({"module": "edit_skills", "action": "fix_region=hands,face", "post": True})
        elif risk == "glyph":
            repairs.append({"module": "ideogram", "action": "plan_ideogram_layout"})
        elif risk == "multi_char":
            repairs.append({"module": "multi_char_cast", "action": "compile_cast_scene"})
        elif risk == "plastic":
            repairs.append({"module": "friction", "action": "apply_friction_to_prompts"})
        elif risk == "long_prompt":
            repairs.append({"module": "prompt_breakdown", "action": "enable prompt_breakdown=auto"})
    report.repairs = repairs
    return report


# (mode id, name, eval prompts that tend to trigger it)
FAILURE_ATLAS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("count_under", "Wrong count (too few)", ("exactly four red apples on a table", "3girls standing in a row")),
    ("count_over", "Wrong count (too many)", ("exactly two cats, no more", "a single bicycle")),
    ("bind_color", "Attribute binding", ("red cube left of blue sphere", "green backpack on a brown dog")),
    ("spatial_left_right", "Left/right", ("the lamp is left of the chair", "person on the right side of frame")),
    ("negation", "Negation ignored", ("a cat with no hat", "portrait without watermark or text")),
    ("hands", "Hand anatomy", ("close-up of hands holding a coffee cup", "counting on five fingers")),
    ("face_far", "Distant face smear", ("crowd scene with recognizable faces", "two people talking across a room")),
    ("glyph", "Text rendering", ('poster that says "OPEN"', "street sign reading MAIN ST")),
    ("multi_bleed", "Character bleed", ("2girls different outfits, not twins", "boy in red, girl in blue, no swap")),
    ("occlusion", "Occlusion", ("person behind chain-link fence", "cat partially behind a vase")),
    ("reflection", "Reflection", ("person reflected in wet street", "mirror selfie with correct chirality")),
    ("long_prompt", "Long prompt tail", ("masterpiece, " + ", ".join(f"detail{i}" for i in range(40)),)),
    ("plastic", "Plastic skin", ("natural skin texture portrait, pores visible",)),
    ("extra_limb", "Extra limbs", ("yoga pose, anatomically correct", "dancer mid-leap")),
    ("object_fusion", "Merged objects", ("two distinct wine glasses side by side",)),
    ("style_mud", "Style mud", ("in the style of ukiyo-e, sharp linework",)),
    ("lighting_physics", "Lighting", ("single hard side light, long cast shadow",)),
    ("material_metal", "Materials", ("brushed aluminum cylinder, sharp reflections",)),
    ("transparency", "Glass/transparency", ("clear glass cup with water and straw",)),
    ("text_small", "Small text", ("business card with tiny crisp letters",)),
    ("rare_count", "Rare counts", ("seven candles on a cake", "eight matching chairs")),
    ("negation_color", "Negation color", ("a car that is not red", "flowers without yellow")),
    ("relation_stack", "Stacked relations", ("cup on book on table under lamp",)),
    ("action_multi", "Multi-action", ("A waving while B sits reading",)),
    ("crop_fullbody", "Full body crop", ("full body standing person, head to toes in frame",)),
    ("background_amnesia", "BG consistency", ("detailed kitchen background, sharp cabinets",)),
    ("cfg_burn", "CFG burn", ("soft natural colors, no oversat neon skin",)),
)


def save_failure_atlas_jsonl(path: str | Path) -> Path:
    """One ``{"mode", "name", "prompt"}`` row per atlas prompt (an eval pack for failure-mode regressions)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for mode, name, prompts in FAILURE_ATLAS:
            for prompt in prompts:
                f.write(json.dumps({"mode": mode, "name": name, "prompt": prompt}) + "\n")
    return p


_NEG_PATTERNS = [
    re.compile(r"\b(?:no|without|not|n't)\s+([a-z0-9][\w\s-]{1,40}?)(?=,|\.|$)", re.IGNORECASE),
    re.compile(r"\b(?:exclude|avoid|ban)\s+([a-z0-9][\w\s-]{1,40}?)(?=,|\.|$)", re.IGNORECASE),
    re.compile(r"\b(?:free of|absent of)\s+([a-z0-9][\w\s-]{1,40}?)(?=,|\.|$)", re.IGNORECASE),
]


@dataclass
class NegatronPlan:
    forbidden: list[str] = field(default_factory=list)
    cleaned_positive: str = ""
    negative_addon: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_negatron(prompt: str) -> NegatronPlan:
    text = str(prompt or "")
    forbidden: list[str] = []
    cleaned = text
    for pat in _NEG_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(1).strip(" ,.")
            if len(phrase) < 2:
                continue
            forbidden.append(phrase)
            # Remove the negation clause from positive so model isn't confused
            cleaned = cleaned.replace(m.group(0), " ")
    # Cleanup commas
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")

    # Expand common negation targets
    expanded: list[str] = []
    for f in forbidden:
        expanded.append(f)
        fl = f.lower()
        if "text" in fl or "watermark" in fl:
            expanded.extend(["watermark", "signature", "logo", "subtitle", "caption overlay"])
        if "hat" in fl:
            expanded.extend(["hat", "cap", "beanie"])
        if "smile" in fl:
            expanded.extend(["smiling", "grin", "toothy smile"])

    uniq = list(dict.fromkeys(expanded))
    neg = ", ".join(uniq)
    if uniq:
        neg = f"{neg}, presence of forbidden concepts"
    return NegatronPlan(
        forbidden=uniq,
        cleaned_positive=cleaned or text,
        negative_addon=neg,
        notes=[f"{len(uniq)} negation targets"],
    )


def apply_negatron(positive: str, negative: str = "") -> tuple[str, str, NegatronPlan]:
    plan = plan_negatron(positive)
    neg = str(negative or "").rstrip(",")
    if plan.negative_addon:
        neg = f"{neg}, {plan.negative_addon}".strip(", ")
    return plan.cleaned_positive, neg, plan


HUMANISH = re.compile(
    r"\b(1girl|1boy|girl|boy|woman|man|person|hands?|fingers?|pose|standing|sitting|"
    r"holding|arm|legs?|body|portrait|selfie)\b",
    re.IGNORECASE,
)
_HANDS = re.compile(r"\b(hand|hands|fingers|holding|grasp|typing|peace sign)\b", re.IGNORECASE)
_DEPTH = re.compile(r"\b(behind|in front|foreground|background|depth)\b", re.IGNORECASE)

_ANATOMY_POS = (
    "anatomically correct limbs",
    "five fingers per hand",
    "correct joint articulation",
    "plausible bone lengths",
    "shoulders connected to arms",
    "no extra limbs",
)
_ANATOMY_NEG = (
    "extra fingers",
    "fused fingers",
    "broken wrists",
    "inverted elbows",
    "disconnected limbs",
    "melted hands",
    "extra arms",
    "bad anatomy",
)

_FRICTION_POS = (
    "natural skin pores",
    "subtle film grain",
    "asymmetric detail",
    "fabric weave visible",
    "imperfect real-world texture",
    "lived-in surfaces",
)
_FRICTION_NEG = (
    "plastic skin",
    "waxy skin",
    "oversmooth",
    "airbrushed",
    "porcelain doll skin",
    "CGI sheen",
    "beauty filter blur",
)

_PEOPLE_PRIORS: tuple[tuple[str, str], ...] = (
    (
        "subtle natural eye asymmetry, lived-in gaze, uneven catchlights",
        "perfectly symmetric doll eyes, identical mirrored pupils",
    ),
    (
        "natural teeth, correct gum line, coherent mouth interior",
        "broken teeth mesh, shark mouth, melted lips, extra teeth",
    ),
    ("feet planted with soft contact shadows on ground", "floating feet, missing contact shadow, hovering subject"),
    ("unfiltered skin, visible pores and peach fuzz", "Instagram beauty filter, Facetune smooth, poreless plastic"),
)
_HAND_PRIOR = (
    "detailed correct hands, five fingers, natural knuckles",
    "extra fingers, fused fingers, melted hands, poorly drawn hands",
)
_COMPOSITION_PRIOR = (
    "off-center subject, rule of thirds composition, asymmetric framing",
    "perfectly centered subject, bullseye composition, dead-center framing",
)
_DEPTH_PRIOR = (
    "explicit z-order, foreground subject, midground secondary, background environment, clear depth layers",
    "flat cardboard depth, everything same distance, z-fighting",
)
_CAMERA_PRIORS: tuple[tuple[str, str], ...] = (
    (
        "filmic highlight rolloff, soft clipped whites, no RGB burn",
        "harsh digital highlight clipping, neon burned whites",
    ),
    (
        "50mm lens look, f/2.8, optical vignetting subtle, Kodak Portra 400 color science, soft skin rolloff, "
        "EXIF-plausible photography",
        "",
    ),
)


@dataclass
class AnatomonPlan:
    active: bool = False
    positive_addon: str = ""
    negative_addon: str = ""
    fix_regions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_anatomon(prompt: str) -> AnatomonPlan:
    """Joint-chain anatomy language for prompts that show a person."""
    if not HUMANISH.search(prompt or ""):
        return AnatomonPlan()
    return AnatomonPlan(
        active=True,
        positive_addon=", ".join(_ANATOMY_POS),
        negative_addon=", ".join(_ANATOMY_NEG),
        fix_regions=["hands", "face"],
    )


def _append(positive: str, negative: str, pos_addon: str, neg_addon: str) -> tuple[str, str]:
    pos, neg = str(positive or "").rstrip(", "), str(negative or "").rstrip(", ")
    if pos_addon and pos_addon.lower() not in pos.lower():
        pos = f"{pos}, {pos_addon}".strip(", ")
    if neg_addon:
        neg = f"{neg}, {neg_addon}".strip(", ")
    return pos, neg


def apply_anatomon_prompts(positive: str, negative: str = "") -> tuple[str, str, AnatomonPlan]:
    plan = plan_anatomon(positive)
    if not plan.active:
        return positive, negative, plan
    pos, neg = _append(positive, negative, plan.positive_addon, plan.negative_addon)
    return pos, neg, plan


def apply_friction_to_prompts(positive: str, negative: str = "", *, strength: float = 0.55) -> tuple[str, str]:
    """Anti-plastic microtexture language; ``strength`` in 0..1 picks how many cues are used."""
    if strength <= 0:
        return positive, negative
    k = max(1, round(len(_FRICTION_POS) * float(strength)))
    return _append(positive, negative, ", ".join(_FRICTION_POS[:k]), ", ".join(_FRICTION_NEG[: max(2, k)]))


def scene_priors(prompt: str) -> list[tuple[str, str]]:
    """``(positive, negative)`` priors that fit ``prompt``; hands are skipped when ANATOMON already covers them."""
    p = str(prompt or "")
    out = [_COMPOSITION_PRIOR]
    if _DEPTH.search(p):
        out.append(_DEPTH_PRIOR)
    if HUMANISH.search(p):
        out.extend(_PEOPLE_PRIORS)
    if _HANDS.search(p) and "five fingers per hand" not in p.lower():
        out.append(_HAND_PRIOR)
    if detect_medium(p).family in ("photo", "generic"):
        out.extend(_CAMERA_PRIORS)
    return out


# Oracle repair module -> stack pass (``ideogram`` has no prompt pass).
_ORACLE_PASS = {"edit_skills": "anatomon", "multi_char_cast": "hydra", "ideogram": ""}


@dataclass
class InventionStackResult:
    positive: str
    negative: str
    box_layout: dict[str, Any] | None = None
    reports: dict[str, Any] = field(default_factory=dict)
    repairs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_invention_stack(
    prompt: str,
    negative: str = "",
    *,
    enable: str = "auto",
    work_dir: str | Path = "",
) -> InventionStackResult:
    """
    ``enable``:
      - ``off``: no-op
      - ``auto``: failure oracle picks passes, plus the scene priors that fit the prompt
      - ``all``: bindlock, countgate, negatron, friction, anatomon, hydra + scene priors
      - comma list: e.g. ``bindlock,negatron,friction`` (``texture`` / ``compose`` add the scene priors)
    """
    pos, neg = str(prompt or ""), str(negative or "")
    mode = str(enable or "auto").strip().lower()
    if mode in ("off", "none", "0", "false"):
        return InventionStackResult(positive=pos, negative=neg)

    oracle = repair_plan_from_failures(diagnose_failures(pos, negative=neg))
    if mode == "all":
        wanted = {"bindlock", "countgate", "negatron", "friction", "anatomon", "hydra"}
    elif mode == "auto":
        wanted = {_ORACLE_PASS.get(m, m) for r in oracle.repairs if (m := str(r.get("module") or ""))} - {""}
        wanted = wanted or {"friction"}
    else:
        wanted = {x.strip() for x in mode.split(",") if x.strip()}

    reports: dict[str, Any] = {"oracle": oracle.to_dict()}
    box_layout = None

    if "negatron" in wanted:
        pos, neg, plan = apply_negatron(pos, neg)
        reports["negatron"] = plan.to_dict()

    if "bindlock" in wanted:
        plan = plan_bindlock(pos)
        pos, neg = apply_bindlock_to_prompts(pos, neg, plan)
        reports["bindlock"] = plan.to_dict()

    if "countgate" in wanted:
        plan = plan_countgate(pos)
        pos, neg = apply_countgate_prompts(pos, neg, plan)
        reports["countgate"] = plan.to_dict()
        if plan.boxes:
            box_layout = {
                "mode": "countgate",
                "anti_bleed": True,
                "global_prompt": pos,
                "regions": [
                    {"id": f"inst_{i}", "prompt": f"instance {i + 1}", "box": b} for i, b in enumerate(plan.boxes)
                ],
            }

    if "hydra" in wanted:
        hp = plan_hydra_slots(pos)
        reports["hydra"] = hp.to_dict()
        if hp.box_layout and hp.slots:
            box_layout = hp.box_layout

    if "anatomon" in wanted:
        pos, neg, plan = apply_anatomon_prompts(pos, neg)
        reports["anatomon"] = plan.to_dict()

    if "friction" in wanted:
        pos, neg = apply_friction_to_prompts(pos, neg)
        reports["friction"] = {"applied": True}

    if mode in ("auto", "all") or "texture" in wanted or "compose" in wanted:
        priors = scene_priors(pos)
        for ap, an in priors:
            if ap and ap.lower() not in pos.lower():
                pos = f"{pos}, {ap}"
            if an:
                neg = f"{neg}, {an}".strip(", ")
        reports["priors"] = [ap for ap, _ in priors]

    result = InventionStackResult(
        positive=pos,
        negative=neg,
        box_layout=box_layout,
        reports=reports,
        repairs=list(oracle.repairs),
    )
    if work_dir:
        w = Path(work_dir)
        w.mkdir(parents=True, exist_ok=True)
        (w / "invention_stack.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        if box_layout:
            (w / "invention_box_layout.json").write_text(json.dumps(box_layout, indent=2), encoding="utf-8")
    return result


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'-]{1,}", re.IGNORECASE)

# Known board / site tags that may appear in paths or metadata (documentation + filters).
BOARD_SOURCE_HINTS: tuple[str, ...] = (
    "danbooru",
    "gelbooru",
    "e621",
    "rule34",
    "rule34xyz",
    "deviantart",
    "artstation",
    "pixiv",
    "safebooru",
    "anime-pictures",
    "konachan",
    "yande.re",
)


@dataclass(slots=True)
class CorpusHit:
    path: str
    caption: str
    score: float
    source: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusIndex:
    """In-memory caption/tag index with optional on-disk image paths."""

    paths: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    tags: list[list[str]] = field(default_factory=list)
    doc_freq: dict[str, int] = field(default_factory=dict)
    doc_tfidf: list[dict[str, float]] = field(default_factory=list)
    root: str = ""

    def __len__(self) -> int:
        return len(self.paths)

    def _build_tfidf(self) -> None:
        n = len(self.captions)
        self.doc_freq = {}
        tf_rows: list[Counter[str]] = []
        for cap, tag_list in zip(self.captions, self.tags, strict=False):
            blob = cap
            if tag_list:
                blob = f"{cap} {' '.join(tag_list)}"
            toks = [t.lower() for t in _TOKEN_RE.findall(blob or "")]
            c = Counter(toks)
            tf_rows.append(c)
            for tok in c:
                self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1
        self.doc_tfidf = []
        for c in tf_rows:
            denom = float(sum(c.values()) or 1)
            row: dict[str, float] = {}
            for tok, cnt in c.items():
                idf = math.log((1.0 + n) / (1.0 + self.doc_freq.get(tok, 0))) + 1.0
                row[tok] = (cnt / denom) * idf
            self.doc_tfidf.append(row)

    def query_tfidf(self, text: str, *, top_k: int = 8) -> list[tuple[int, float]]:
        if not self.doc_tfidf:
            self._build_tfidf()
        q_toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
        if not q_toks or not self.doc_tfidf:
            return []
        qc = Counter(q_toks)
        q_denom = float(sum(qc.values()) or 1)
        n = max(1, len(self.doc_tfidf))
        q_vec: dict[str, float] = {}
        for tok, cnt in qc.items():
            idf = math.log((1.0 + n) / (1.0 + self.doc_freq.get(tok, 0))) + 1.0
            q_vec[tok] = (cnt / q_denom) * idf
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()) or 1e-12)
        scored: list[tuple[int, float]] = []
        for i, row in enumerate(self.doc_tfidf):
            dot = 0.0
            for tok, qv in q_vec.items():
                dot += qv * row.get(tok, 0.0)
            r_norm = math.sqrt(sum(v * v for v in row.values()) or 1e-12)
            scored.append((i, float(dot / (q_norm * r_norm + 1e-12))))
        scored.sort(key=lambda x: -x[1])
        return scored[: max(1, int(top_k))]


def _infer_source(path: str, meta: dict[str, Any]) -> str:
    s = str(meta.get("source") or meta.get("site") or meta.get("board") or "").lower()
    if s:
        return s
    low = path.replace("\\", "/").lower()
    for hint in BOARD_SOURCE_HINTS:
        if hint in low:
            return hint
    return ""


def _row_path_caption(obj: dict[str, Any], *, root: Path | None) -> tuple[str, str, list[str]] | None:
    path = obj.get("image") or obj.get("file_name") or obj.get("path") or obj.get("file") or obj.get("img") or ""
    path = str(path).strip()
    if not path:
        return None
    p = Path(path)
    if not p.is_file() and root is not None:
        cand = root / path
        if cand.is_file():
            path = str(cand)
    cap = obj.get("text") or obj.get("caption") or obj.get("prompt") or obj.get("tag_string") or obj.get("tags") or ""
    tags: list[str] = []
    if isinstance(obj.get("tags"), list):
        tags = [str(t).strip() for t in obj["tags"] if str(t).strip()]
        if not cap:
            cap = ", ".join(tags)
    elif isinstance(cap, list):
        tags = [str(t).strip() for t in cap if str(t).strip()]
        cap = ", ".join(tags)
    cap = str(cap).strip()
    if not cap:
        return None
    return path, cap, tags


def build_corpus_index_from_jsonl(
    jsonl_path: str | Path,
    *,
    root: str | Path | None = None,
    max_rows: int = 0,
    require_existing_file: bool = False,
) -> CorpusIndex:
    """Scan a training/manifest JSONL into a CorpusIndex."""
    path = Path(jsonl_path)
    root_p = Path(root) if root else path.parent
    idx = CorpusIndex(root=str(root_p))
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            parsed = _row_path_caption(obj, root=root_p)
            if parsed is None:
                continue
            img_path, cap, tags = parsed
            if require_existing_file and not Path(img_path).is_file():
                continue
            idx.paths.append(img_path)
            idx.captions.append(cap)
            idx.tags.append(tags)
            idx.sources.append(_infer_source(img_path, obj))
            n += 1
            if max_rows > 0 and n >= max_rows:
                break
    idx._build_tfidf()
    return idx


def save_corpus_index_meta(index: CorpusIndex, out_path: str | Path) -> Path:
    """Save lightweight meta (paths+captions) for fast reload without re-scan."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(index)):
        rows.append(
            {
                "image": index.paths[i],
                "text": index.captions[i],
                "source": index.sources[i] if i < len(index.sources) else "",
                "tags": index.tags[i] if i < len(index.tags) else [],
            }
        )
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def load_corpus_index(jsonl_path: str | Path, *, root: str | Path | None = None) -> CorpusIndex:
    return build_corpus_index_from_jsonl(jsonl_path, root=root, require_existing_file=False)


def retrieve_corpus_refs(
    index: CorpusIndex,
    query: str,
    *,
    top_k: int = 6,
    source_filter: str = "",
    prefer_existing: bool = True,
) -> list[CorpusHit]:
    """TF-IDF retrieve corpus hits for a prompt (subject/style/attire tags work well)."""
    ranked = index.query_tfidf(query, top_k=max(top_k * 4, top_k))
    hits: list[CorpusHit] = []
    filt = str(source_filter or "").lower().strip()
    for i, score in ranked:
        src = index.sources[i] if i < len(index.sources) else ""
        if filt and filt not in src and filt not in index.paths[i].lower():
            continue
        path = index.paths[i]
        hits.append(
            CorpusHit(
                path=path,
                caption=index.captions[i],
                score=float(score),
                source=src,
                tags=list(index.tags[i]) if i < len(index.tags) else [],
            )
        )
        if len(hits) >= top_k:
            break
    return hits


def hits_to_moodboard_payload(hits: list[CorpusHit]) -> dict[str, Any]:
    """JSON payload compatible with ``--moodboard-json`` / InstantStyle wiring."""
    images = [h.path for h in hits if h.path and Path(h.path).is_file()]
    return {
        "images": images,
        "paths": images,
        "captions": [h.caption for h in hits],
        "scores": [h.score for h in hits],
        "sources": [h.source for h in hits],
        "by_facet": {"corpus": images},
    }


@dataclass
class CastMember:
    id: str
    appearance: str = ""
    outfit: str = ""
    pose: str = ""
    action: str = ""
    spatial: str = ""  # left / right / center / bbox hint
    face_ref: str = ""
    sheet_path: str = ""
    avoid: list[str] = field(default_factory=list)

    def prompt_block(self) -> str:
        bits = [b for b in (self.appearance, self.outfit, self.pose, self.action, self.spatial) if b]
        return ", ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CastScene:
    members: list[CastMember] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    setting: str = ""
    quality: list[str] = field(default_factory=list)
    anti_blend: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "members": [m.to_dict() for m in self.members],
            "relations": list(self.relations),
            "setting": self.setting,
            "quality": list(self.quality),
            "anti_blend": self.anti_blend,
        }


@dataclass
class CastCompileResult:
    positive: str
    negative: str
    count_tag: str
    box_layout: dict[str, Any] | None
    sheet_paths: list[str]
    face_refs: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    return [s] if s else []


def _member_from_obj(obj: dict[str, Any], fallback_id: str) -> CastMember:
    wardrobe = ", ".join(_as_list(obj.get("wardrobe") or obj.get("outfit") or obj.get("clothing")))
    pose = ", ".join(_as_list(obj.get("pose") or obj.get("poses")))
    action = ", ".join(_as_list(obj.get("action") or obj.get("actions") or obj.get("doing")))
    hair = ", ".join(_as_list(obj.get("hair") or obj.get("hair_features")))
    face = ", ".join(_as_list(obj.get("face") or obj.get("face_features") or obj.get("identity_tokens")))
    appearance = ", ".join(x for x in (_as_list(obj.get("appearance") or obj.get("description")) + [hair, face]) if x)
    spatial = str(obj.get("spatial_anchor") or obj.get("spatial") or obj.get("position") or "").strip()
    return CastMember(
        id=str(obj.get("id") or obj.get("name") or fallback_id),
        appearance=appearance,
        outfit=wardrobe,
        pose=pose,
        action=action,
        spatial=spatial,
        face_ref=str(obj.get("face_ref") or obj.get("reference_image") or "").strip(),
        sheet_path=str(obj.get("character_sheet") or obj.get("sheet") or "").strip(),
        avoid=_as_list(obj.get("avoid")),
    )


def load_cast_scene(path_or_dict: str | Path | dict[str, Any]) -> CastScene:
    if isinstance(path_or_dict, dict):
        data = path_or_dict
    else:
        data = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    actors = data.get("actors") or data.get("members") or data.get("characters") or []
    members = [_member_from_obj(a, f"character_{i + 1}") for i, a in enumerate(actors) if isinstance(a, dict)]
    rels: list[str] = []
    for r in data.get("relations") or []:
        if isinstance(r, str):
            rels.append(r)
        elif isinstance(r, dict):
            a, b = r.get("a"), r.get("b")
            kind = r.get("kind") or r.get("relation") or "interacting"
            detail = r.get("detail") or ""
            rels.append(f"{a} and {b}: {kind}" + (f", {detail}" if detail else ""))
    return CastScene(
        members=members,
        relations=rels,
        setting=str(data.get("setting") or data.get("place") or "").strip(),
        quality=_as_list(data.get("quality")),
        anti_blend=bool(data.get("anti_blend", data.get("anti_artifacts", True))),
    )


def _default_bboxes(n: int) -> list[tuple[float, float, float, float]]:
    """Non-overlapping horizontal strips for n characters."""
    if n <= 0:
        return []
    if n == 1:
        return [(0.15, 0.1, 0.85, 0.95)]
    boxes = []
    w = 0.9 / n
    gap = 0.02
    for i in range(n):
        x0 = 0.05 + i * w + gap
        x1 = 0.05 + (i + 1) * w - gap
        boxes.append((x0, 0.08, min(x1, 0.95), 0.95))
    return boxes


def compile_cast_scene(scene: CastScene, *, base_prompt: str = "") -> CastCompileResult:
    n = len(scene.members)
    notes: list[str] = []
    if n == 0:
        return CastCompileResult(
            positive=base_prompt,
            negative="",
            count_tag="",
            box_layout=None,
            sheet_paths=[],
            face_refs=[],
            notes=["empty cast"],
        )

    # Count tags (booru-friendly)
    if n == 1:
        count = "1girl" if "girl" in (scene.members[0].appearance + scene.members[0].outfit).lower() else "1person"
    elif n == 2:
        count = "2girls"
    else:
        count = f"{n}girls" if n <= 6 else "multiple girls"
    # Prefer generic if mixed — keep simple count
    if n >= 2:
        count = f"{n}people" if n > 2 else "2people"
        # Also emit classic tags
        count = f"{n}girls" if n <= 4 else "multiple girls"

    labels = [m.id for m in scene.members]
    blocks = []
    for m in scene.members:
        block = m.prompt_block()
        # Explicit ownership phrasing fights clothing bleed
        owned = []
        if m.outfit:
            owned.append(f"{m.id} wears {m.outfit}")
        if m.action:
            owned.append(f"{m.id} is {m.action}")
        if m.pose:
            owned.append(f"{m.id} pose: {m.pose}")
        if owned:
            block = f"{block}, " + ", ".join(owned) if block else ", ".join(owned)
        blocks.append(block or m.id)

    labeled = merge_character_sheet_positives(blocks, labels=labels)
    parts = [count]
    if scene.quality:
        parts.append(", ".join(scene.quality))
    if base_prompt.strip():
        parts.append(base_prompt.strip().rstrip(","))
    parts.append(labeled)
    if scene.relations:
        parts.append(", ".join(scene.relations))
    if scene.setting:
        parts.append(scene.setting)
    if scene.anti_blend:
        parts.extend(MULTI_SUBJECT_POSITIVE_EXTRAS)
        notes.append("anti-blend positives on")

    positive = ", ".join(p for p in parts if p)

    neg_bits = list(MULTI_SUBJECT_NEGATIVE_EXTRAS) if scene.anti_blend else []
    neg_bits.extend(
        [
            "character blending",
            "face swap",
            "identity mix",
            "attributes swapped between characters",
            "shared clothing",
        ]
    )
    for m in scene.members:
        for a in m.avoid:
            neg_bits.append(a)
        # Explicit: don't put others' outfits on this character
        others_outfits = [o.outfit for o in scene.members if o.id != m.id and o.outfit]
        for oo in others_outfits[:2]:
            neg_bits.append(f"{m.id} wearing {oo}")
    negative = merge_csv_unique(*neg_bits)

    # Box layout for regional CFG / layout attn
    boxes = _default_bboxes(n)
    regions = []
    for m, box in zip(scene.members, boxes, strict=False):
        regions.append(
            {
                "id": m.id,
                "prompt": m.prompt_block() or m.appearance or m.id,
                "box": [box[0], box[1], box[2], box[3]],
            }
        )
    box_layout = {
        "mode": "multi_character",
        "anti_bleed": True,
        "regions": regions,
        "global_prompt": scene.setting or base_prompt,
    }

    sheets = [m.sheet_path for m in scene.members if m.sheet_path]
    faces = [m.face_ref for m in scene.members if m.face_ref]
    notes.append(f"{n} characters compiled with ownership phrases")
    return CastCompileResult(
        positive=positive,
        negative=negative,
        count_tag=count,
        box_layout=box_layout,
        sheet_paths=sheets,
        face_refs=faces,
        notes=notes,
    )


_OFF = ("off", "none", "0", "false", "")


@dataclass
class InventionRuntime:
    positive: str
    negative: str
    enable: str = "off"
    box_layout: dict[str, Any] | None = None
    box_layout_path: str = ""
    steps: int | None = None
    reports: dict[str, Any] = field(default_factory=dict)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelfHealingPlan:
    """Oracle -> stack -> APEX sampling -> critique-region redo, as a ``sample.py`` argv plus redo hints."""

    prompt: str
    negative: str = ""
    steps: int = 28
    sample_argv: list[str] = field(default_factory=list)
    critique_hints: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_json(obj: Any, work_dir: str | Path, name: str) -> str:
    """Write *obj* to ``work_dir/name`` (or ``$TMP/sdx_name`` without a work dir); return the path."""
    path = Path(work_dir) / name if work_dir else Path(tempfile.gettempdir()) / f"sdx_{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return str(path)


def adaptive_nfe(prompt: str, *, base_steps: int = 28, min_steps: int = 12, max_steps: int = 40) -> int:
    """Step budget from oracle risk: +3 steps per elevated risk, -2 for a risk-free prompt."""
    risks = diagnose_failures(prompt).risks
    steps = base_steps + 3 * len(risks) - (0 if risks else 2)
    return int(max(min_steps, min(max_steps, steps)))


def _redo_regions(risks: list[str]) -> list[str]:
    regions: list[str] = []
    for r in risks:
        if r == "anatomy":
            regions.extend(["hands", "face"])
        elif r == "glyph":
            regions.append("subject")
        elif r == "plastic":
            regions.append("face")
    return list(dict.fromkeys(regions))


def plan_self_healing(prompt: str, negative: str = "") -> SelfHealingPlan:
    inv = apply_invention_stack(prompt, negative, enable="auto")
    steps = adaptive_nfe(inv.positive)
    risks = (inv.reports.get("oracle") or {}).get("risks") or []
    argv = [
        "--prompt",
        inv.positive,
        "--negative-prompt",
        inv.negative or negative,
        "--steps",
        str(steps),
        "--scheduler",
        "apex",
        "--solver",
        "apex_adaptive",
        "--invention-stack",
        "off",  # already applied
    ]
    return SelfHealingPlan(
        prompt=inv.positive,
        negative=inv.negative,
        steps=steps,
        sample_argv=argv,
        critique_hints=_redo_regions(risks),
        notes=[f"risks={risks}"],
    )


def write_self_healing_plan(plan: SelfHealingPlan, work_dir: str | Path) -> str:
    return _write_json(plan.to_dict(), work_dir, "self_healing_plan.json")


def prepare_invention_runtime(
    prompt: str,
    negative: str = "",
    *,
    enable: str = "auto",
    base_steps: int | None = None,
    adaptive_steps: bool = False,
    work_dir: str | Path = "",
) -> InventionRuntime:
    """Run the invention stack and compute optional runtime knobs."""
    mode = str(enable or "off").strip().lower()
    pos, neg = str(prompt or ""), str(negative or "")
    if mode in _OFF:
        return InventionRuntime(positive=pos, negative=neg, enable="off")

    res = apply_invention_stack(pos, neg, enable=mode, work_dir=work_dir)
    notes: list[str] = []
    steps: int | None = None
    if adaptive_steps and base_steps is not None:
        steps = adaptive_nfe(res.positive, base_steps=int(base_steps))
        notes.append(f"adaptive_nfe={steps}")
    return InventionRuntime(
        positive=res.positive,
        negative=res.negative,
        enable=mode,
        box_layout=res.box_layout,
        box_layout_path=_write_json(res.box_layout, work_dir, "invention_box_layout.json") if res.box_layout else "",
        steps=steps,
        reports=dict(res.reports),
        repairs=list(res.repairs),
        notes=notes,
    )


def apply_invention_to_namespace(args: Any, *, work_dir: str = "outputs/_invention") -> InventionRuntime | None:
    """Mutate a ``sample.py`` namespace in place for ``--invention-stack``; None when it is off."""
    enable = str(getattr(args, "invention_stack", "off") or "off").strip().lower()
    if enable in _OFF:
        return None

    prompt = str(getattr(args, "prompt", "") or "")
    negative = str(getattr(args, "negative_prompt", "") or "")
    adaptive = bool(getattr(args, "invention_adaptive_steps", False))
    rt = prepare_invention_runtime(
        prompt,
        negative,
        enable=enable,
        base_steps=int(getattr(args, "steps", 28) or 28),
        adaptive_steps=adaptive,
        work_dir=work_dir,
    )
    args.prompt = rt.positive
    if rt.negative:
        args.negative_prompt = rt.negative
    if rt.box_layout_path and not str(getattr(args, "box_layout", "") or "").strip():
        args.box_layout = rt.box_layout_path
        args.anti_bleed = True
    if rt.steps is not None:
        args.steps = int(rt.steps)

    # Guidance knobs for the risks the oracle found; only fills values the user left unset.
    risks = set((rt.reports.get("oracle") or {}).get("risks") or [])
    if (
        risks & {"binding", "count", "negation", "multi_char"}
        and not str(getattr(args, "guidance_schedule", None) or "").strip()
        and not getattr(args, "holy_grail", False)  # gaussian_diffusion rejects a schedule with Holy Grail
    ):
        args.guidance_schedule = "piecewise"
    if risks & {"plastic", "binding"} and float(getattr(args, "apg_parallel_eta", -1.0)) < 0:
        args.apg_parallel_eta = 0.0

    if bool(getattr(args, "invention_self_heal", False)):
        rt.notes.append(f"self_heal_plan={write_self_healing_plan(plan_self_healing(prompt, negative), work_dir)}")
    return rt


def apply_cast_and_corpus_to_namespace(args: Any, *, work_dir: str | Path = "") -> None:
    """``--cast-json`` (ownership phrases, anti-blend negatives, box layout, sheets) and ``--corpus-jsonl``."""
    cast_json = str(getattr(args, "cast_json", "") or "").strip()
    if cast_json:
        cast = compile_cast_scene(load_cast_scene(cast_json), base_prompt=str(getattr(args, "prompt", "") or ""))
        args.prompt = cast.positive
        args.negative_prompt = ", ".join(n for n in (getattr(args, "negative_prompt", ""), cast.negative) if n)
        if cast.box_layout and not str(getattr(args, "box_layout", "") or "").strip():
            args.box_layout = _write_json(cast.box_layout, work_dir, "cast_box_layout.json")
            args.anti_bleed = True
        if cast.sheet_paths and not str(getattr(args, "character_sheet", "") or "").strip():
            args.character_sheet = ",".join(cast.sheet_paths)
            args.label_multi_character_sheets = True

    corpus = str(getattr(args, "corpus_jsonl", "") or "").strip()
    if corpus:
        from utils.prompt.rag import merge_facts_into_prompt

        index = load_corpus_index(corpus, root=str(getattr(args, "corpus_root", "") or "") or None)
        hits = retrieve_corpus_refs(
            index,
            str(getattr(args, "prompt", "") or ""),
            top_k=max(1, int(getattr(args, "corpus_top_k", 4) or 4)),
            source_filter=str(getattr(args, "corpus_source", "") or ""),
        )
        facts = [h.caption for h in hits if h.caption]
        if facts:
            args.prompt = merge_facts_into_prompt(str(getattr(args, "prompt", "") or ""), facts)
        images = hits_to_moodboard_payload(hits)["images"]
        if images and not str(getattr(args, "moodboard_images", "") or "").strip():
            args.moodboard_images = ",".join(images)
