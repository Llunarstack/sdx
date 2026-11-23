"""Photorealism shot spec (camera, shutter, ISO chosen to agree with the planned lens and light), hard-prompt
plans (exact counts, spatial relations, attribute binding, text in the image) and weak-domain prompt packs.

Everything here turns on from the prompt; ``--realism``, ``--shot-spec`` and ``--hard-prompts`` override.
Imports of other ``utils`` modules stay inside functions: ``content_controls`` imports this module.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

_OFF = ("off", "none", "0", "false", "")

# ---------------------------------------------------------------------------------------------------------
# Shot spec
# ---------------------------------------------------------------------------------------------------------

# Scene brightness as EV at ISO 100 (the sunny-16 family of exposure tables).
_TIME_EV = {
    "midday": 15.0,
    "morning": 14.0,
    "afternoon": 14.0,
    "golden_hour": 12.0,
    "dawn": 11.0,
    "overcast": 12.0,
    "blue_hour": 9.0,
    "night": 5.0,
    "interior": 7.0,
    "unspecified": 12.0,
}
_SOURCE_EV = {
    "window": 9.0,
    "lamp": 6.0,
    "screen": 5.0,
    "streetlight": 5.0,
    "neon": 6.0,
    "fire": 5.0,
    "candle": 4.0,
    "moon": -2.0,
}
_DAYLIGHT = ("dawn", "morning", "midday", "afternoon", "golden_hour", "overcast")
_ISO_STOPS = (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600)
_SHUTTERS = (1 / 8000, 1 / 4000, 1 / 2000, 1 / 1000, 1 / 500, 1 / 250, 1 / 125, 1 / 60, 1 / 30, 1 / 15)
_SHUTTERS += (1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0, 15.0, 20.0, 30.0)

_ISO_RE = re.compile(r"\biso[ -]?(\d{2,6})\b", re.I)
_SHUTTER_RE = re.compile(
    r"\b1\s*/\s*(\d{1,5})\s*(?:s\b|sec\b|second\b|shutter\b)|\bshutter(?: speed)?(?: of)?\s+1\s*/\s*(\d{1,5})\b"
    r"|\b(\d{1,2}(?:\.\d)?)\s*(?:s|sec|seconds?)\s+(?:exposure|shutter)\b",
    re.I,
)
_BODY_RE = re.compile(
    r"\b(?:dslr|mirrorless|full[- ]frame|medium[- ]format|large[- ]format|rangefinder|aps-c|"
    r"micro four thirds|cinema camera)\b",
    re.I,
)
_GRAIN_WORD_RE = re.compile(r"\b(?:grain\w*|noise|noisy|clean files?)\b", re.I)
_ASTRO_RE = re.compile(r"\b(?:milky way|starry sky|night sky|astrophoto\w*|star trails?|aurora\w*)\b", re.I)
_LONG_RE = re.compile(
    r"\b(?:long[- ]exposure|light trails?|silky water|slow shutter|tripod|motion[- ]blurred? water)\b", re.I
)
_ACTION_RE = re.compile(
    r"\b(?:running|sprint\w*|jumping|leaping|mid-?air|in flight|flying|splash\w*|sports?|racing|race car|"
    r"dancing|skateboard\w*|surf\w*|action shot|frozen motion|freeze motion|galloping)\b",
    re.I,
)


@dataclass(slots=True)
class ShotSpec:
    """One coherent camera setting. ``explicit`` names the fields the user stated (never re-captioned)."""

    camera: str = "dslr"  # dslr | phone | compact | film
    focal_mm: int = 50
    aperture: float = 5.6
    iso: int = 100  # 0 = let the film stock decide
    shutter_s: float = 1 / 125
    ev: float = 12.0
    time_of_day: str = "unspecified"
    motion: str = "handheld"  # handheld | action | long | astro | studio
    dof: str = "medium"
    explicit: frozenset[str] = field(default_factory=frozenset)

    @property
    def shutter_text(self) -> str:
        if self.shutter_s >= 1.0:
            return f"{self.shutter_s:g} s exposure"
        return f"1/{round(1.0 / self.shutter_s)} s shutter"

    def caption(self) -> str:
        """Prompt text for the settings the user left open (the art-direction caption already has lens and light)."""
        ex = self.explicit
        parts: list[str] = []
        if "camera" not in ex and self.camera == "dslr":
            parts.append("full-frame camera")
        if "shutter" not in ex:
            parts.append(self.shutter_text)
        if "iso" not in ex and self.iso:
            parts.append(f"ISO {self.iso}")
        if "grain" not in ex:
            if self.iso >= 1600:
                parts.append("visible fine luminance grain")
            elif 0 < self.iso <= 200 and self.camera != "film":
                parts.append("clean low-noise detail")
        parts.append(
            {
                "action": "motion frozen crisply",
                "long": "long-exposure blur on moving elements",
                "astro": "pinpoint stars",
            }.get(self.motion, "")
        )
        return ", ".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["explicit"] = sorted(self.explicit)
        d["shutter"] = self.shutter_text
        d["caption"] = self.caption()
        return d


def _nearest(value: float, stops: tuple) -> Any:
    return min(stops, key=lambda s: abs(math.log2(max(value, 1e-9) / s)))


def _scene_ev(ad: Any, text: str, motion: str) -> float:
    if motion == "astro":
        return -6.0
    tod = str(getattr(ad, "time_of_day", "unspecified"))
    sources = tuple(getattr(ad, "sources", ()) or ())
    key = sources[0] if sources else ""
    if key in _SOURCE_EV and (tod not in _DAYLIGHT or key == "window"):
        return _SOURCE_EV[key]
    return _TIME_EV.get(tod, 12.0)


def plan_shot_spec(prompt: str, ad: Any = None, *, seed: int = 0) -> ShotSpec:
    """Camera, shutter and ISO solved from the art-direction plan's lens and light (EV at ISO 100).

    Handheld shutter is limited to 1/focal (1/125 with people, 1/500 for animals), action to 1/1000; the ISO
    then makes up the exposure: ``ISO = 100 * N^2 / (t * 2^EV)`` rounded to a full stop. Studio work is
    strobe-lit at ISO 100, 1/160 s. Values the prompt states (ISO, shutter, camera) are kept.
    """
    text = str(prompt or "")
    if ad is None:
        from utils.prompt.art_direction import plan_art_direction

        ad = plan_art_direction(text, seed=seed)
    from utils.quality.finishing import _COMPACT_RE, _FILM_RE, _PHONE_RE

    explicit: set[str] = set()
    camera = "dslr"
    if _FILM_RE.search(text):
        camera, _ = "film", explicit.add("camera")
    elif _PHONE_RE.search(text):
        camera, _ = "phone", explicit.add("camera")
    elif _COMPACT_RE.search(text):
        camera, _ = "compact", explicit.add("camera")
    elif _BODY_RE.search(text):
        explicit.add("camera")
    if _GRAIN_WORD_RE.search(text):
        explicit.add("grain")

    focal = int(getattr(ad, "focal_mm", 50) or 50)
    n = float(getattr(ad, "aperture", 5.6) or 5.6)
    tod = str(getattr(ad, "time_of_day", "unspecified"))
    if _ASTRO_RE.search(text):
        motion = "astro"
    elif _LONG_RE.search(text):
        motion = "long"
    elif _ACTION_RE.search(text):
        motion = "action"
    elif tod == "studio":
        motion = "studio"
    else:
        motion = "handheld"
    ev = _scene_ev(ad, text, motion)

    if motion == "studio":
        t, iso = 1 / 160, 100
    else:
        open_ok = "aperture" not in getattr(ad, "explicit", frozenset())
        if motion == "astro" and open_ok:
            n = min(n, 2.8)  # stars need a fast lens wide open
        t_max = 1.0 / max(focal, 30)
        if getattr(ad, "has_people", False):
            t_max = min(t_max, 1 / 125)
        if getattr(ad, "subject", "") == "animal":
            t_max = min(t_max, 1 / 500)
        if motion == "action":
            t_max = min(t_max, 1 / 1000)
        while True:
            t = n * n / 2.0**ev  # correct exposure at ISO 100
            if motion == "astro":
                t = 20.0
            elif motion == "long":
                t = min(max(t, 0.5), 30.0)
            else:
                t = max(min(t, t_max), 1 / 8000)
            iso = 100.0 * n * n / (t * 2.0**ev)
            # Past ISO 6400 a photographer opens up first (down to f/2.8) unless the aperture was asked for.
            if iso <= 6400 or not open_ok or n <= 2.8:
                break
            n = max((s for s in (2.8, 4.0, 5.6, 8.0, 11.0, 16.0) if s < n), default=2.8)  # one stop wider
        iso = int(_nearest(min(max(iso, 100.0), 25600.0), _ISO_STOPS))
        t = float(_nearest(t, _SHUTTERS))

    m = _SHUTTER_RE.search(text)
    if m:
        explicit.add("shutter")
        if m.group(1) or m.group(2):
            t = 1.0 / max(1, int(m.group(1) or m.group(2)))
        else:
            t = float(m.group(3))
    m = _ISO_RE.search(text)
    if m:
        iso, _ = int(min(max(int(m.group(1)), 25), 102400)), explicit.add("iso")
    elif camera == "film":
        iso = 0  # the emulsion sets it (utils.quality.finishing.film_stock_for_prompt)
    elif camera == "phone":
        iso = int(min(iso, 3200))
    return ShotSpec(
        camera=camera,
        focal_mm=focal,
        aperture=n,
        iso=int(iso),
        shutter_s=float(t),
        ev=float(ev),
        time_of_day=tod,
        motion=motion,
        dof=str(getattr(ad, "dof", "medium")),
        explicit=frozenset(explicit),
    )


def realism_mode(args: Any) -> str:
    """``--realism``: off | auto | max (a namespace without the flag counts as off)."""
    mode = str(getattr(args, "realism", "off") or "off").strip().lower()
    return "off" if mode in _OFF else mode if mode in ("auto", "max") else "auto"


def apply_shot_spec_override(args: Any) -> str:
    """``--shot-spec "85mm f/1.4 ISO 6400 1/60s"``: the user's camera text joins the prompt (before planning)."""
    text = str(getattr(args, "shot_spec", "") or "").strip()
    if text.lower() in (*_OFF, "auto"):
        return ""
    for name in ("prompt", "_user_prompt"):
        cur = str(getattr(args, name, "") or "")
        if hasattr(args, name) and text.lower() not in cur.lower():
            setattr(args, name, f"{cur}, {text}" if cur.strip() else text)
    return text


def apply_realism_to_args(args: Any) -> ShotSpec | None:
    """Plan the shot spec for a photographic prompt (after art direction).

    Sets ``args._shot_spec`` (read by the capture finish), ``args.shot_spec_plan`` (report) and
    ``args._shot_caption`` (prompt text; empty with ``--shot-spec off``). ``--realism max`` also ranks
    at least four candidates with ``combo_realism`` when no pick metric was asked for.
    """
    args._shot_spec, args.shot_spec_plan, args._shot_caption = None, None, ""
    mode = realism_mode(args)
    if mode == "off":
        return None
    text = str(getattr(args, "_user_prompt", "") or getattr(args, "prompt", "") or "")
    if not text.strip():
        return None
    ad = getattr(args, "_art_direction", None)
    if ad is None:
        from utils.prompt.art_direction import plan_art_direction

        ad = plan_art_direction(text, seed=int(getattr(args, "seed", 0) or 0))
    if not ad.photographic:
        return None
    spec = plan_shot_spec(text, ad)
    if spec.aperture != ad.aperture:  # opened up for low light: keep the art-direction caption in agreement
        ad.aperture = spec.aperture
        if spec.aperture <= 2.8 and spec.motion != "astro" and "dof" not in ad.explicit:
            ad.dof = spec.dof = "shallow"
    args._shot_spec, args.shot_spec_plan = spec, spec.to_dict()
    if str(getattr(args, "shot_spec", "") or "").strip().lower() not in _OFF[:-1]:  # "" = auto
        args._shot_caption = spec.caption()
    if mode == "max" and str(getattr(args, "pick_best", "none") or "none").lower() == "none":
        args.pick_best = "combo_realism"
        args.num = max(int(getattr(args, "num", 1) or 1), 4)
    return spec


# ---------------------------------------------------------------------------------------------------------
# Hard prompts: counts, spatial relations, attribute binding, text
# ---------------------------------------------------------------------------------------------------------

_NUM_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "a pair of": 2,
    "a dozen": 12,
}
_COLORS = (
    "red|blue|green|yellow|orange|purple|violet|pink|black|white|brown|gr[ae]y|cyan|magenta|gold|"
    "silver|teal|navy|beige|turquoise|crimson|maroon|lime"
)
_COLOR_RE = re.compile(rf"\b(?:{_COLORS})\b", re.I)
# Relation phrase -> canonical relation (longest first so "to the left of" wins over "left of").
_RELATIONS = (
    ("to the left of", "left_of"),
    ("on the left of", "left_of"),
    ("left of", "left_of"),
    ("to the right of", "right_of"),
    ("on the right of", "right_of"),
    ("right of", "right_of"),
    ("on top of", "on"),
    ("in front of", "in_front_of"),
    ("next to", "beside"),
    ("beside", "beside"),
    ("behind", "behind"),
    ("above", "above"),
    ("below", "below"),
    ("beneath", "below"),
    ("under", "below"),
    ("underneath", "below"),
    ("on", "on"),
)
_TRIGGER_RELATIONS = {"left_of", "right_of", "above", "below", "in_front_of", "behind", "beside"}
_REL_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p, _ in _RELATIONS) + r")\b", re.I)
_REL_NAME = dict(_RELATIONS)
_SPLIT_RE = re.compile(r"\s*(?:,|;|\.\s|\band\b|\bwhile\b|\bwith\b(?=\s+(?:a|an|the|\d|two|three)\b))\s*", re.I)
_DET_RE = re.compile(r"^(?:(?:a|an|the|some|exactly|one|its|their|is|are|sits?|sitting|placed|standing)\s+)+", re.I)
# Nouns that name looks, light or framing, not objects (never become a layout region).
_NOT_OBJECT = re.compile(
    r"\b(?:light\w*|hour|sunset|sunrise|dawn|dusk|night|background|backdrop|sky|tones?|palette|hues?|glow|colou?rs?|theme|grade|style|aesthetic|"
    r"lens|bokeh|shot|photo\w*|render\w*|quality|detail\w*|8k|4k|hdr|composition|lighting|atmosphere|mood)\b",
    re.I,
)
# Counted nouns that are body parts, units or idioms, not instances to lay out.
_NOT_COUNTED = frozenset(
    "hand hands eye eyes arm arms leg legs finger fingers foot feet ear ears wing wings horn horns tail tails "
    "year years day days hour hours minute minutes second seconds time times mm percent way ways tone tones "
    "story stories storey storeys side sides point points step steps color colors colour colours".split()
)
_IRREGULAR = {"people": "person", "men": "man", "women": "woman", "children": "child", "mice": "mouse"}
_IRREGULAR_PLURAL = {v: k for k, v in _IRREGULAR.items()}
_PEOPLE_NOUNS = frozenset(
    "person man woman child girl boy kid friend people men women children girls boys kids friends".split()
)
_ING_NOUNS = frozenset("building painting clothing ceiling earring dumpling pudding stocking railing".split())
_STOP = frozenset("of and with on in at the a an to near by from for while is are that which wearing holding".split())
_COUNT_RE = re.compile(
    r"\b(?:exactly\s+)?(\d{1,2}|" + "|".join(re.escape(w) for w in _NUM_WORDS) + r")\s+([a-z][a-z' -]{1,40})",
    re.I,
)
_TEXT_CUE = r"(?:says|saying|reads|reading|labell?ed|titled|the words?|text|lettering|spelling(?: out)?|inscribed)"
_QUOTED_RE = re.compile(r'"([^"\n]{1,80})"|“([^”\n]{1,80})”|' + _TEXT_CUE + r"\s+['‘]([^'’\n]{1,60})['’]", re.I)
_UNQUOTED_RE = re.compile(
    r"\b(?:that|which)?\s*" + _TEXT_CUE + r"\s*:?\s+([A-Z0-9][A-Z0-9&'!.-]*(?:\s+[A-Z0-9][A-Z0-9&'!.-]*){0,3})\b"
)
_HAND_RE = re.compile(
    r"\b(?:hands?|fingers?|holding|grip\w*|grasp\w*|waving|pointing|clapping|typing|"
    r"playing (?:the )?(?:piano|guitar|violin)|thumbs up|peace sign)\b",
    re.I,
)
_GROUP_RE = re.compile(r"\b(?:group of|crowd|team of|family of|friends|band of|audience)\b", re.I)


def _singular(word: str) -> str:
    w = word.lower()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if re.search(r"(?:ches|shes|sses|xes)$", w):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _plural(word: str) -> str:
    if word in _IRREGULAR_PLURAL:
        return _IRREGULAR_PLURAL[word]
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + ("es" if re.search(r"(?:ch|sh|ss|x)$", word) else "s")


def extract_expected_texts(prompt: str) -> list[str]:
    """Words the image should spell: double / curly quotes, quotes after a text cue ("labeled 'X'"), and
    unquoted capitals after a cue ("a sign that says OPEN", "reading MAIN ST")."""
    text = str(prompt or "")
    out: list[str] = []
    for m in _QUOTED_RE.finditer(text):
        out.append(next(g for g in m.groups() if g))
    for m in _UNQUOTED_RE.finditer(text):
        out.append(m.group(1))
    seen: set[str] = set()
    keep = []
    for t in (s.strip() for s in out):
        if t and re.search(r"[A-Za-z0-9]", t) and t.lower() not in seen:
            seen.add(t.lower())
            keep.append(t)
    return keep[:4]


@dataclass(slots=True)
class Subject:
    phrase: str
    noun: str
    count: int = 1
    colors: tuple[str, ...] = ()


@dataclass
class HardPromptPlan:
    subjects: list[Subject] = field(default_factory=list)
    relations: list[tuple[int, str, int]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    hands: bool = False
    people: int = 0
    positive: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)
    layout: dict[str, Any] | None = None
    pick_metric: str = ""

    @property
    def count(self) -> tuple[int, str]:
        best = max(self.subjects, key=lambda s: s.count, default=None)
        return (best.count, best.noun) if best and best.count >= 2 else (0, "")

    @property
    def binding(self) -> bool:
        colored = [s for s in self.subjects if s.colors]
        return len(colored) >= 2 and len({s.colors for s in colored}) >= 2

    @property
    def spatial(self) -> bool:
        return any(r in _TRIGGER_RELATIONS for _, r, _ in self.relations)

    @property
    def active(self) -> bool:
        return bool(self.positive or self.negative or self.layout or self.texts or self.pick_metric)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["count"], d["binding"], d["spatial"] = list(self.count), self.binding, self.spatial
        return d


def _subject(segment: str) -> Subject | None:
    seg = _DET_RE.sub("", re.sub(r"\s+", " ", segment.strip(" .,;:!?\"'"))).strip()
    words = seg.split()
    if not words or len(words) > 7 or _NOT_OBJECT.search(seg):
        return None
    count = 1
    m = re.match(r"^(\d{1,2}|" + "|".join(re.escape(w) for w in _NUM_WORDS) + r")\s+(.+)$", seg, re.I)
    if m:
        raw = m.group(1).lower()
        count = int(raw) if raw.isdigit() else _NUM_WORDS[raw]
        seg = m.group(2)
        words = seg.split()
    # The head noun is the last word before a stop word or participle ("man in a red shirt", "cats sitting").
    head = []
    for w in words:
        lw = w.lower()
        if lw in _STOP or (head and lw.endswith("ing") and len(lw) > 5 and lw not in _ING_NOUNS):
            break
        head.append(w)
    noun = _singular(re.sub(r"[^a-z-]", "", (head or words)[-1].lower()))
    if not noun or noun in _NOT_COUNTED or not (2 <= count <= 12):
        count = 1
    colors = tuple(sorted({c.lower().replace("grey", "gray") for c in _COLOR_RE.findall(seg)}))
    return Subject(phrase=seg, noun=noun, count=count, colors=colors)


def _parse_subjects(text: str) -> tuple[list[Subject], list[tuple[int, str, int]]]:
    """Clauses split at commas / "and" / relations; relations link the clauses on either side."""
    clean = re.sub(r'"[^"]*"|“[^”]*”', " ", text)
    # "black and white cat" is one two-tone subject, not two.
    clean = re.sub(rf"\b({_COLORS})\s+and\s+({_COLORS})\b", r"\1-\2", clean, flags=re.I)
    pieces = _REL_RE.split(clean)  # [clause, rel, clause, rel, ...]
    subjects: list[Subject] = []
    relations: list[tuple[int, str, int]] = []
    prev_last: int | None = None
    pending: str | None = None
    for i, piece in enumerate(pieces):
        if i % 2 == 1:
            pending = _REL_NAME[piece.lower()]
            continue
        segs = [s for s in _SPLIT_RE.split(piece) if s and s.strip()]
        idx: list[int | None] = []
        for s in segs:
            sub = _subject(s)
            if sub is None:
                idx.append(None)
                continue
            subjects.append(sub)
            idx.append(len(subjects) - 1)
        first = idx[0] if idx else None
        if pending and prev_last is not None and first is not None:
            relations.append((prev_last, pending, first))
        pending = None
        prev_last = idx[-1] if idx else None
    return subjects, relations


def _solve_boxes(subjects: list[Subject], relations: list[tuple[int, str, int]]) -> list[list[float]]:
    """Boxes (x0, y0, x1, y1) that satisfy the relations; mention order sets the default left-to-right."""
    k = len(subjects)
    size = min(0.6, 0.9 / k + 0.12)
    w = [size] * k
    h = [size] * k
    cx = [(i + 1) / (k + 1) for i in range(k)]
    cy = [0.55] * k
    for a, rel, b in relations:  # supports are wide and low; front objects bigger
        if rel == "on":
            w[b], h[b] = min(0.9, max(w[b], 1.6 * w[a])), max(0.25, 0.7 * h[b])
            if subjects[a].count > 1:  # a group on a support spreads along it in shallow rows
                w[a], h[a] = max(w[a], 0.8 * w[b]), min(h[a], 0.3)
        elif rel == "in_front_of":
            w[a], h[a], w[b], h[b] = w[a] * 1.15, h[a] * 1.15, w[b] * 0.85, h[b] * 0.85
        elif rel == "behind":
            w[a], h[a], w[b], h[b] = w[a] * 0.85, h[a] * 0.85, w[b] * 1.15, h[b] * 1.15
    for _ in range(12):  # relaxation: each violated relation moves both ends halfway
        for a, rel, b in relations:
            sep_x = 0.5 * (w[a] + w[b]) * 0.9
            sep_y = 0.5 * (h[a] + h[b]) * 0.9
            if rel in ("left_of", "right_of", "beside"):
                lo, hi = (a, b) if rel == "left_of" or (rel == "beside" and cx[a] <= cx[b]) else (b, a)
                gap = sep_x - (cx[hi] - cx[lo])
                if gap > 0:
                    cx[lo] -= gap / 2
                    cx[hi] += gap / 2
            elif rel == "on":
                # Resting on: a's bottom edge lands on b's top surface (upper third of b's box), not above it.
                gap = (0.15 * h[b] + 0.5 * h[a]) - (cy[b] - cy[a])
                cy[a] -= gap / 2
                cy[b] += gap / 2
                cx[a] = cx[b]
            elif rel in ("above", "below"):
                top, bot = (a, b) if rel == "above" else (b, a)
                gap = sep_y - (cy[bot] - cy[top])
                if gap > 0:
                    cy[top] -= gap / 2
                    cy[bot] += gap / 2
            elif rel in ("in_front_of", "behind"):
                front, back = (a, b) if rel == "in_front_of" else (b, a)
                gap = 0.15 - (cy[front] - cy[back])
                if gap > 0:
                    cy[front] += gap / 2
                    cy[back] -= gap / 2
        for i in range(k):  # keep every box inside the frame
            cx[i] = min(max(cx[i], 0.02 + w[i] / 2), 0.98 - w[i] / 2)
            cy[i] = min(max(cy[i], 0.02 + h[i] / 2), 0.98 - h[i] / 2)
    return [
        [round(cx[i] - w[i] / 2, 3), round(cy[i] - h[i] / 2, 3), round(cx[i] + w[i] / 2, 3), round(cy[i] + h[i] / 2, 3)]
        for i in range(k)
    ]


def _split_box(box: list[float], n: int) -> list[list[float]]:
    """``n`` instance boxes inside ``box`` (a row up to four, else a grid)."""
    from utils.generation.inventions import count_grid_boxes

    x0, y0, x1, y1 = box
    if n <= 4:
        cells = [[i / n, 0.0, (i + 1) / n, 1.0] for i in range(n)]
    else:
        cells = count_grid_boxes(n)
    pad = 0.08
    return [
        [
            round(x0 + (x1 - x0) * (c[0] + pad * (c[2] - c[0])), 3),
            round(y0 + (y1 - y0) * (c[1] + pad * (c[3] - c[1])), 3),
            round(x0 + (x1 - x0) * (c[2] - pad * (c[2] - c[0])), 3),
            round(y0 + (y1 - y0) * (c[3] - pad * (c[3] - c[1])), 3),
        ]
        for c in cells
    ]


def plan_hard_prompt(prompt: str) -> HardPromptPlan:
    """Counts, relations, colour binding, text, hands and multi-person cues -> prompt packs, a box layout
    (one region per subject / counted instance, other subjects' colours in each region's negative) and the
    best-of-N metric that checks the hardest constraint."""
    text = str(prompt or "")
    plan = HardPromptPlan(texts=extract_expected_texts(text))
    subjects, relations = _parse_subjects(text)
    # A subject only matters if it is counted, coloured or related to another.
    related = {i for a, _, b in relations for i in (a, b)}
    keep = [i for i, s in enumerate(subjects) if s.count >= 2 or s.colors or i in related]
    remap = {old: new for new, old in enumerate(keep)}
    plan.subjects = [subjects[i] for i in keep]
    plan.relations = [(remap[a], r, remap[b]) for a, r, b in relations if a in remap and b in remap]
    plan.hands = bool(_HAND_RE.search(text))
    n_people = sum(s.count for s in plan.subjects if s.noun in _PEOPLE_NOUNS)
    plan.people = n_people if n_people >= 2 else (3 if _GROUP_RE.search(text) else 0)

    n, noun = plan.count
    if n:
        many = _plural(noun)
        plan.positive += [f"exactly {n} {many}", f"{n} separate {many}, each fully visible"]
        plan.negative += [f"{n + 1} {many}", f"{n - 1} {many if n - 1 != 1 else noun}", f"merged {many}"]
    if plan.binding:
        plan.positive.append("each object keeps its own color")
        plan.negative.append("color bleeding between objects")
        for s in plan.subjects:
            for o in plan.subjects:
                if o is not s:
                    plan.negative += [f"{c} {s.noun}" for c in o.colors if c not in s.colors]
    if plan.spatial:
        side = {"left_of": "on the left", "right_of": "on the right"}
        for a, rel, b in plan.relations:
            if rel in side:
                other = "on the right" if rel == "left_of" else "on the left"
                plan.positive.append(f"{plan.subjects[a].phrase} {side[rel]}, {plan.subjects[b].phrase} {other}")
        plan.negative.append("swapped positions")
    if plan.people >= 2:
        plan.positive.append("each person with a distinct face and body")
        plan.negative += ["cloned faces", "merged bodies"]

    regional = len(plan.subjects) >= 2 and (plan.binding or plan.spatial or n)
    if regional or (n and len(plan.subjects) == 1 and n <= 8):
        boxes = _solve_boxes(plan.subjects, plan.relations)
        regions = []
        for i, (s, box) in enumerate(zip(plan.subjects, boxes, strict=True)):
            others = sorted({c for o in plan.subjects if o is not s for c in o.colors} - set(s.colors))
            neg = ", ".join(f"{c} {s.noun}" for c in others)
            if 2 <= s.count <= 8:
                attrs = " ".join(s.colors)
                for j, b in enumerate(_split_box(box, s.count) if len(plan.subjects) > 1 else _full_grid(s.count)):
                    regions.append(
                        {
                            "id": f"{s.noun}_{j + 1}",
                            "prompt": f"one {attrs} {s.noun}".replace("  ", " "),
                            "box": b,
                            "negative": neg,
                        }
                    )
            else:
                regions.append({"id": f"{s.noun}_{i + 1}", "prompt": s.phrase, "box": box, "negative": neg})
        plan.layout = {"mode": "hard_prompt", "anti_bleed": True, "global_prompt": text, "regions": regions}

    if n:
        plan.pick_metric = "combo_count"
    elif plan.binding and plan.spatial:
        plan.pick_metric = "combo_spatial_bind"
    elif plan.hands:
        plan.pick_metric = "combo_hand"
    elif plan.people >= 2:
        plan.pick_metric = "combo_human"
    plan.positive = list(dict.fromkeys(plan.positive))
    plan.negative = list(dict.fromkeys(plan.negative))
    return plan


def _full_grid(n: int) -> list[list[float]]:
    from utils.generation.inventions import count_grid_boxes

    return count_grid_boxes(n)


def apply_hard_prompts_to_args(args: Any, *, work_dir: str = "outputs/_invention") -> HardPromptPlan | None:
    """``--hard-prompts`` (auto by default): packs into the prompt, a box layout unless the user or the
    invention stack gave one, the expected count / text for the critics, the text-friendly negative and
    the pick metric. ``max`` also turns on best-of-4. Skipped on OCR-repair re-runs."""
    mode = str(getattr(args, "hard_prompts", "off") or "off").strip().lower()
    if mode in _OFF or int(getattr(args, "ocr_repair_iter", 0) or 0) > 0:
        return None
    text = str(getattr(args, "_user_prompt", "") or getattr(args, "prompt", "") or "")
    plan = plan_hard_prompt(text)
    if not plan.active:
        return plan
    pos = str(getattr(args, "prompt", "") or "")
    add = [p for p in plan.positive if p.lower() not in pos.lower()]
    if add:
        args.prompt = ", ".join([pos.rstrip(", "), *add]) if pos.strip() else ", ".join(add)
    if plan.negative:
        neg = str(getattr(args, "negative_prompt", "") or "").rstrip(", ")
        args.negative_prompt = ", ".join(p for p in (neg, ", ".join(plan.negative)) if p)
    user_layout = (
        str(getattr(args, "box_layout", "") or "").strip()
        or getattr(args, "_box_layout_spec", None) is not None
        or getattr(args, "_used_prompt_layout", False)
    )
    if plan.layout and not user_layout:
        from utils.generation.inventions import _write_json

        args.box_layout = _write_json(plan.layout, work_dir, "hard_prompt_box_layout.json")
        args.anti_bleed = True
    n, noun = plan.count
    if n and not int(getattr(args, "expected_count", 0) or 0):
        args.expected_count = n
        args.expected_count_target = "people" if noun in _PEOPLE_NOUNS else "objects"
        args.expected_count_object = "" if noun in _PEOPLE_NOUNS else noun
    if plan.texts:
        args.text_in_image = True
        if not str(getattr(args, "expected_text", "") or "").strip():
            args.expected_text = json.dumps(plan.texts)
    args._hard_pick_metric = plan.pick_metric
    if mode == "max" and str(getattr(args, "pick_best", "none") or "none").lower() == "none":
        args.pick_best = "auto"
        args.num = max(int(getattr(args, "num", 1) or 1), 4)
    args.hard_prompt_plan = plan.to_dict()
    return plan


# ---------------------------------------------------------------------------------------------------------
# Weak domains: packs for utils.prompt.content_controls (scene_domain) and the hand / people inference
# ---------------------------------------------------------------------------------------------------------

# domain -> (positive, negative). Medium-neutral where possible; photo terms are gated off for paintings.
WEAK_DOMAIN_PACKS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "product": (
        (
            "accurate product proportions",
            "crisp clean edges",
            "controlled specular highlights",
            "true-to-life material finish",
            "soft contact shadow grounding the product",
            "uncluttered backdrop",
        ),
        (
            "warped product shape",
            "melted edges",
            "garbled label text",
            "floating product without shadow",
            "smeared reflections",
            "dust and fingerprints",
        ),
    ),
    "food": (
        (
            "appetizing fresh texture",
            "natural food colors",
            "visible moisture and crumb detail",
            "realistic portion size",
            "plates and cutlery at believable scale",
            "soft directional side light",
        ),
        (
            "plastic-looking food",
            "waxy fruit",
            "oversaturated food",
            "melted indistinct ingredients",
            "impossible food textures",
            "floating garnish",
        ),
    ),
    "interior": (
        (
            "coherent room layout",
            "furniture at human scale",
            "vertical lines kept straight",
            "light consistent with the windows",
            "realistic floor and wall materials",
            "believable ceiling height",
        ),
        (
            "furniture merging into walls",
            "warped door frames",
            "impossible room geometry",
            "inconsistent shadows",
            "floating furniture",
            "mismatched perspective",
        ),
    ),
    "machinery": (
        (
            "mechanically plausible parts",
            "bolts and joints that connect",
            "functional hydraulics and gears",
            "consistent part symmetry",
            "believable wear on metal",
            "correct scale against surroundings",
        ),
        (
            "parts connected to nothing",
            "melted mechanical detail",
            "impossible linkages",
            "mismatched wheels",
            "random greebles",
            "warped panels",
        ),
    ),
    "nature_macro": (
        (
            "accurate anatomy of the organism",
            "fine natural texture",
            "focal plane on the subject",
            "true-to-species coloration",
            "natural habitat context",
            "fine hairs and dew resolved",
        ),
        (
            "extra legs on insects",
            "fused petals",
            "plastic-looking leaves",
            "wrong number of wings",
            "cartoonish proportions",
            "oversaturated foliage",
        ),
    ),
    "technical": (
        (
            "accurate technical illustration",
            "clean precise linework",
            "correct structure and proportions",
            "consistent line weight",
            "clear labeled callouts",
            "neutral plain background",
        ),
        (
            "structurally wrong parts",
            "nonsense mechanisms",
            "gibberish labels",
            "inconsistent scale",
            "wobbly sketchy lines",
            "decorative clutter",
        ),
    ),
    "diagram": (
        (
            "clear readable layout",
            "consistent legend and symbols",
            "aligned grid",
            "limited color coding",
            "legible labels",
            "flat orthographic view",
        ),
        (
            "gibberish text",
            "garbled labels",
            "overlapping labels",
            "inconsistent symbols",
            "perspective distortion",
            "cluttered layout",
        ),
    ),
}

# Prompt keywords -> domain (substring match, first hit wins; placed ahead of the older buckets).
WEAK_DOMAIN_BUCKETS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "technical illustration",
            "scientific illustration",
            "medical illustration",
            "anatomical illustration",
            "botanical illustration",
            "cutaway",
            "exploded view",
            "cross-section",
            "cross section",
            "schematic",
            "blueprint",
            "patent drawing",
        ),
        "technical",
    ),
    (
        (
            "infographic",
            "diagram",
            "flowchart",
            "flow chart",
            "map of",
            "world map",
            "city map",
            "fantasy map",
            "subway map",
            "floor plan",
            "bar chart",
            "pie chart",
        ),
        "diagram",
    ),
    (
        (
            "product photo",
            "product shot",
            "packshot",
            "e-commerce",
            "ecommerce",
            "catalog photo",
            "advertising shot",
            "commercial photo",
            "product render",
        ),
        "product",
    ),
    (
        (
            "food",
            "meal",
            "dessert",
            "cuisine",
            "pastry",
            "sushi",
            "burger",
            "pizza",
            "ramen",
            "steak",
            "salad",
            "breakfast",
            "cocktail",
            "bakery",
        ),
        "food",
    ),
    (
        (
            "excavator",
            "bulldozer",
            "tractor",
            "crane",
            "jet engine",
            "steam engine",
            "engine bay",
            "turbine",
            "machinery",
            "robotic arm",
            "robot arm",
            "locomotive",
            "forklift",
            "gearbox",
        ),
        "machinery",
    ),
    (
        (
            "macro",
            "insect",
            "butterfly",
            "beetle",
            "dragonfly",
            "spider",
            "dewdrop",
            "dew drop",
            "mushroom",
            "moss",
            "wildflower",
            "pollen",
        ),
        "nature_macro",
    ),
    (
        (
            "interior",
            "living room",
            "bedroom",
            "kitchen",
            "bathroom",
            "dining room",
            "hallway",
            "lobby",
            "office space",
        ),
        "interior",
    ),
)


def infer_hard_controls(prompt_lower: str) -> dict[str, str]:
    """Hand and people-layout modes for prompts the older keyword lists miss."""
    p = prompt_lower
    out: dict[str, str] = {}
    if re.search(r"\b(?:holding|grip\w*|grasp\w*|carrying|clutching)\b", p):
        out["hand_mode"] = "grip"
    elif _HAND_RE.search(p):
        out["hand_mode"] = "stable"
    people = plan_people_count(p)
    if people >= 6 or "crowd" in p:
        out["people_layout"] = "group_large"
    elif people >= 3 or _GROUP_RE.search(p):
        out["people_layout"] = "group_small"
    return out


def plan_people_count(text: str) -> int:
    """People asked for by number ("three women", "5 friends"); 0 when none is stated."""
    total = 0
    for m in _COUNT_RE.finditer(text):
        raw = m.group(1).lower()
        if any(_singular(w) in _PEOPLE_NOUNS for w in m.group(2).split()[:3]):
            total += int(raw) if raw.isdigit() else _NUM_WORDS[raw]
    return total
