"""Imagination engine (``--frontier-creative``): creative analyzers that emit prompt fragments AND diffusion knobs
(serendipity, CFG, step curves) without duplicating the art-medium / shortcomings prompt stack.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from config.prompts import merge_csv_unique

# ==================================================================================================
# archetype: Jungian-adjacent symbols as compositional anchors — artistic, not preachy.
# ==================================================================================================


class ArchetypeSymbol(StrEnum):
    THRESHOLD = "threshold"
    MIRROR = "mirror"
    LABYRINTH = "labyrinth"
    CHALICE = "chalice"
    MASK = "mask"
    BRIDGE = "bridge"
    NONE = "none"


@dataclass(frozen=True)
class ArchetypePlan:
    symbol: ArchetypeSymbol
    positive: str
    negative: str


_MAP: tuple[tuple[re.Pattern, ArchetypeSymbol, str, str], ...] = (
    (
        re.compile(r"\b(threshold|doorway|gate|portal|liminal)\b", re.I),
        ArchetypeSymbol.THRESHOLD,
        "liminal threshold framing, passage between worlds, light spill from beyond",
        "random door pasted, no spatial logic",
    ),
    (
        re.compile(r"\b(mirror|reflection|doppelganger|twin)\b", re.I),
        ArchetypeSymbol.MIRROR,
        "meaningful mirror symmetry, identity doubling, reflective narrative",
        "broken mirror noise, accidental clone face",
    ),
    (
        re.compile(r"\b(labyrinth|maze|minotaur|lost paths)\b", re.I),
        ArchetypeSymbol.LABYRINTH,
        "recursive path geometry, center-oriented composition, maze readability",
        "random wall maze, no exit logic",
    ),
    (
        re.compile(r"\b(chalice|grail|offering|ritual cup)\b", re.I),
        ArchetypeSymbol.CHALICE,
        "symbolic vessel focal point, ceremonial light, reverent scale",
        "generic cup without symbolic weight",
    ),
    (
        re.compile(r"\b(mask|masked|masquerade|hidden identity)\b", re.I),
        ArchetypeSymbol.MASK,
        "mask as identity boundary, eye holes readable, material truth",
        "face blur pretending to be mask",
    ),
    (
        re.compile(r"\b(bridge|crossing|span|connecting shores)\b", re.I),
        ArchetypeSymbol.BRIDGE,
        "bridge as narrative connector, perspective along span, destination implied",
        "floating bridge without support",
    ),
)


class SymbolMapEngine:
    def plan(self, prompt: str) -> ArchetypePlan:
        text = prompt or ""
        for pat, sym, pos, neg in _MAP:
            if pat.search(text):
                return ArchetypePlan(sym, pos, neg)
        return ArchetypePlan(ArchetypeSymbol.NONE, "", "")

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


# ==================================================================================================
# cinema: Cinema shot grammar — ECU, OTS, worm's eye, establishing, insert cut.
#
# Film-director vocabulary; pairs with ``narrative.WitnessPerspectiveAnalyzer``.
# ==================================================================================================


class ShotType(StrEnum):
    ESTABLISHING = "establishing"
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE = "extreme_close"
    OVER_SHOULDER = "over_shoulder"
    POV = "pov"
    WORMS_EYE = "worms_eye"
    BIRDS_EYE = "birds_eye"
    INSERT = "insert"
    TWO_SHOT = "two_shot"


@dataclass(frozen=True)
class ShotPlan:
    shot: ShotType
    positive: str
    negative: str
    lens_hint: str


_SHOTS: tuple[tuple[re.Pattern, ShotType, str, str, str], ...] = (
    (
        re.compile(r"\b(establishing shot|extreme wide|city establishing)\b", re.I),
        ShotType.ESTABLISHING,
        "geography readable in one frame, story context, tiny human scale optional",
        "tight crop without context",
        "24mm environmental",
    ),
    (
        re.compile(r"\b(over the shoulder|OTS|over-shoulder)\b", re.I),
        ShotType.OVER_SHOULDER,
        "foreground shoulder soft blur, subject beyond, dialogue staging",
        "flat single plane, no depth staging",
        "50mm shallow",
    ),
    (
        re.compile(r"\b(POV|point of view|first person view)\b", re.I),
        ShotType.POV,
        "first-person hands optional, horizon at eye line, immersive",
        "third person floating camera",
        "wide POV lens",
    ),
    (
        re.compile(r"\b(worm'?s eye|from below|low angle hero)\b", re.I),
        ShotType.WORMS_EYE,
        "extreme low angle, towering subject, converging verticals",
        "eye-level default",
        "ultrawide low",
    ),
    (
        re.compile(r"\b(bird'?s eye|top down|aerial directly above)\b", re.I),
        ShotType.BIRDS_EYE,
        "plan view geometry, pattern read, map-like clarity",
        "tilted aerial confusion",
        "top-down orthographic feel",
    ),
    (
        re.compile(r"\b(insert shot|detail cut|macro insert)\b", re.I),
        ShotType.INSERT,
        "narrative object isolated, shallow depth, story prop clarity",
        "wide scene without insert focus",
        "macro insert",
    ),
    (
        re.compile(r"\b(two shot|two-shot|couple frame)\b", re.I),
        ShotType.TWO_SHOT,
        "two subjects balanced in frame, relationship staging",
        "orphan single subject when two intended",
        "50mm two-shot",
    ),
    (
        re.compile(r"\b(extreme close[- ]?up|ECU|eyes only)\b", re.I),
        ShotType.EXTREME_CLOSE,
        "skin pore scale, iris detail, crop at brow or lip",
        "medium shot distance",
        "100mm macro portrait",
    ),
)


class ShotGrammar:
    def plan(self, prompt: str) -> ShotPlan:
        text = prompt or ""
        for pat, shot, pos, neg, lens in _SHOTS:
            if pat.search(text):
                return ShotPlan(shot, pos, neg, lens)
        if re.search(r"\b(cinematic|film still|movie frame)\b", text, re.I):
            return ShotPlan(
                ShotType.MEDIUM,
                "cinematic framing, headroom discipline, lookroom toward action",
                "amateur snapshot framing",
                "35mm film still",
            )
        return ShotPlan(ShotType.MEDIUM, "", "", "")

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        pos = ", ".join(x for x in (p.positive, p.lens_hint) if x)
        return pos, p.negative


# ==================================================================================================
# collective: Many people — diversity and staging, not duplicate subjects.
# ==================================================================================================


class CrowdScale(StrEnum):
    SMALL_GROUP = "small_group"
    CROWD = "crowd"
    MASS = "mass"
    NONE = "none"


@dataclass(frozen=True)
class CrowdPlan:
    scale: CrowdScale
    positive: str
    negative: str


_RULES: tuple[tuple[re.Pattern, CrowdScale, str, str], ...] = (
    (
        re.compile(r"\b(crowd|packed|stadium|concert audience|protest march)\b", re.I),
        CrowdScale.CROWD,
        "varied faces and silhouettes, staggered depth rows, no identical clones",
        "duplicate faces, clone army, same person repeated",
    ),
    (
        re.compile(r"\b(two people|couple|pair|duo|three friends)\b", re.I),
        CrowdScale.SMALL_GROUP,
        "distinct individuals, separate styling, clear spatial relationship",
        "merged bodies, twin duplication",
    ),
    (
        re.compile(r"\b(army|horde|thousands|sea of people|massive crowd)\b", re.I),
        CrowdScale.MASS,
        "mass read via silhouette tiers, atmospheric perspective, individual only in foreground",
        "every face sharp and identical",
    ),
)


class CrowdGrammar:
    def plan(self, prompt: str) -> CrowdPlan:
        text = prompt or ""
        for pat, scale, pos, neg in _RULES:
            if pat.search(text):
                return CrowdPlan(scale, pos, neg)
        if re.search(r"\b(people|group|family|team)\b", text, re.I):
            return CrowdPlan(
                CrowdScale.SMALL_GROUP,
                "each person visually distinct, natural group staging",
                "duplicate subjects, extra heads",
            )
        return CrowdPlan(CrowdScale.NONE, "", "")

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


# ==================================================================================================
# constraint: Creative constraints — art-school limits that beat "add more detail".
#
# Examples: one light source, three colors only, silhouettes only, no faces shown.
# ==================================================================================================


class LimitRule(StrEnum):
    MONOCHROME = "monochrome"
    TRIADIC_ONLY = "triadic_only"
    SINGLE_LIGHT = "single_light"
    SILHOUETTE = "silhouette"
    NO_FACE = "no_face"
    CENTER_SUBJECT = "center_subject"
    HORIZON_LOW = "horizon_low"
    TEXTURE_ONLY = "texture_only"


@dataclass(frozen=True)
class ConstraintPack:
    rules: tuple[LimitRule, ...]
    positive: str
    negative: str


_TRIGGERS: tuple[tuple[re.Pattern, LimitRule, str, str], ...] = (
    (
        re.compile(r"\b(monochrome only|black and white only|single hue)\b", re.I),
        LimitRule.MONOCHROME,
        "strict monochrome value design, no chroma distraction",
        "color accents, rainbow noise",
    ),
    (
        re.compile(r"\b(three colors? only|limited palette|3-color)\b", re.I),
        LimitRule.TRIADIC_ONLY,
        "only three hues plus neutrals, disciplined palette",
        "extra accent colors, palette drift",
    ),
    (
        re.compile(r"\b(one light|single light source|one lamp)\b", re.I),
        LimitRule.SINGLE_LIGHT,
        "exactly one motivated light source, coherent shadow fan",
        "multiple conflicting keys, flat ambient mush",
    ),
    (
        re.compile(r"\b(silhouette only|silhouettes?|backlit shape)\b", re.I),
        LimitRule.SILHOUETTE,
        "readable silhouette, interior detail suppressed, rim edge clarity",
        "interior detail in silhouette, gray fill",
    ),
    (
        re.compile(r"\b(faceless|no face|hidden face|turned away)\b", re.I),
        LimitRule.NO_FACE,
        "face obscured or turned away, identity through body language",
        "direct eye contact, visible facial features",
    ),
    (
        re.compile(r"\b(low horizon|horizon at bottom)\b", re.I),
        LimitRule.HORIZON_LOW,
        "sky-dominant composition, low horizon line, vast ceiling of atmosphere",
        "centered horizon, cramped sky",
    ),
    (
        re.compile(r"\b(texture study|macro texture only|surface abstract)\b", re.I),
        LimitRule.TEXTURE_ONLY,
        "abstract texture read, material truth, minimal figurative content",
        "busy scene narrative, random objects",
    ),
)


class CreativeConstraintEngine:
    def detect(self, prompt: str) -> list[LimitRule]:
        text = prompt or ""
        return [rule for pat, rule, _, _ in _TRIGGERS if pat.search(text)]

    def pack(self, prompt: str) -> ConstraintPack:
        text = prompt or ""
        hits = [(rule, pos, neg) for pat, rule, pos, neg in _TRIGGERS if pat.search(text)]
        return ConstraintPack(
            tuple(rule for rule, _, _ in hits),
            ", ".join(pos for _, pos, _ in hits),
            ", ".join(neg for _, _, neg in hits),
        )

    def suggest_random(self, *, seed: int = 0) -> ConstraintPack:
        """Pick a creative limit (deterministic in ``seed``) when the user wants surprise."""
        rules = list(LimitRule)
        r = rules[seed % len(rules)]
        for _, rule, pos, neg in _TRIGGERS:
            if rule == r:
                return ConstraintPack((r,), pos, neg)
        return ConstraintPack((r,), "creative restriction discipline", "generic clutter")


# ==================================================================================================
# focal: DOF and focus intent — isolation, discovery, overwhelm.
# ==================================================================================================


class FocusIntent(StrEnum):
    ISOLATE = "isolate"
    DISCOVER = "discover"
    OVERWHELM = "overwhelm"
    LAYERED = "layered"
    NONE = "none"


@dataclass(frozen=True)
class FocalStory:
    intent: FocusIntent
    positive: str
    negative: str
    cfg_hint: float


_focal_RULES: tuple[tuple[re.Pattern, FocusIntent, str, str, float], ...] = (
    (
        re.compile(r"\b(shallow dof|bokeh|blurred background|subject isolation)\b", re.I),
        FocusIntent.ISOLATE,
        "shallow depth isolates subject emotionally, creamy bokeh separation",
        "subject and background equally sharp",
        1.05,
    ),
    (
        re.compile(r"\b(rack focus|focus pull|revealing detail)\b", re.I),
        FocusIntent.DISCOVER,
        "foreground blur yields to sharp discovery plane, narrative reveal",
        "uniform focus everywhere",
        1.08,
    ),
    (
        re.compile(r"\b(everything sharp|deep focus|hyperfocal|landscape clarity)\b", re.I),
        FocusIntent.OVERWHELM,
        "deep focus overwhelming detail, environmental complexity readable",
        "fake tilt-shift on wide scene",
        1.0,
    ),
    (
        re.compile(r"\b(foreground blur|layers of depth|through the reeds)\b", re.I),
        FocusIntent.LAYERED,
        "multiple depth layers, peek-through framing, voyeur discovery",
        "single plane cardboard",
        1.06,
    ),
)


class FocalStoryteller:
    def plan(self, prompt: str) -> FocalStory:
        text = prompt or ""
        for pat, intent, pos, neg, cfg in _focal_RULES:
            if pat.search(text):
                return FocalStory(intent, pos, neg, cfg)
        return FocalStory(FocusIntent.NONE, "", "", 1.0)

    def fragments(self, prompt: str) -> tuple[str, str, float]:
        p = self.plan(prompt)
        return p.positive, p.negative, p.cfg_hint


# ==================================================================================================
# fusion: Genre fusion — steampunk+cyberpunk, folk+sci-fi without muddy soup.
#
# Picks a *dominant* genre and treats the other as accent (30% rule).
# ==================================================================================================


class GenrePair(StrEnum):
    STEAMPUNK_CYBER = "steampunk_cyber"
    FOLK_SCIFI = "folk_scifi"
    ANIME_OIL = "anime_oil"
    BRUTALIST_ORGANIC = "brutalist_organic"
    WESTERN_FANTASY = "western_fantasy"
    NONE = "none"


@dataclass(frozen=True)
class FusionPlan:
    pair: GenrePair
    dominant: str
    accent: str
    positive: str
    negative: str


_DETECT: tuple[tuple[str, str, GenrePair, str, str, str, str], ...] = (
    (
        r"steampunk",
        r"cyberpunk",
        GenrePair.STEAMPUNK_CYBER,
        "steampunk",
        "cyberpunk",
        "brass Victorian engineering dominant, neon cyber accents at 30%, unified grime palette",
        "equal mix mush, chrome on brass without logic",
    ),
    (
        r"folk",
        r"sci[- ]?fi|space|astronaut",
        GenrePair.FOLK_SCIFI,
        "folk art",
        "sci-fi",
        "folk pattern and handcraft dominant, sci-fi element as single focal artifact",
        "random spacesuit in medieval tapestry",
    ),
    (
        r"anime",
        r"oil painting|impasto",
        GenrePair.ANIME_OIL,
        "anime",
        "oil painting",
        "anime linework with oil impasto fill, cel edges over painterly body",
        "flat anime with fake oil filter",
    ),
    (
        r"brutalist|concrete",
        r"organic|biomorphic|vine",
        GenrePair.BRUTALIST_ORGANIC,
        "brutalist",
        "organic",
        "concrete mass dominant, biomorphic breach as singular rupture",
        "ivy wallpaper on brutalism",
    ),
    (
        r"western|cowboy",
        r"fantasy|dragon|magic",
        GenrePair.WESTERN_FANTASY,
        "western",
        "fantasy",
        "frontier western dominant, one fantasy element as mythic intrusion",
        "elf cowboy costume party",
    ),
)


class GenreCollisionEngine:
    def detect(self, prompt: str) -> FusionPlan:
        text = prompt or ""
        for a_pat, b_pat, pair, dom, acc, pos, neg in _DETECT:
            if re.search(a_pat, text, re.I) and re.search(b_pat, text, re.I):
                return FusionPlan(pair, dom, acc, pos, neg)
        return FusionPlan(GenrePair.NONE, "", "", "", "")

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.detect(prompt)
        return p.positive, p.negative


# ==================================================================================================
# glitch: Glitch as aesthetic choice — VHS, datamosh, scanlines.
# ==================================================================================================


class GlitchAesthetic(StrEnum):
    VHS = "vhs"
    DATAMOSH = "datamosh"
    SCANLINE = "scanline"
    RGB_SPLIT = "rgb_split"
    PIXEL_SORT = "pixel_sort"
    NONE = "none"


@dataclass(frozen=True)
class GlitchPlan:
    aesthetic: GlitchAesthetic
    positive: str
    negative: str
    serendipity_boost: float


_glitch_RULES: tuple[tuple[re.Pattern, GlitchAesthetic, str, str, float], ...] = (
    (
        re.compile(r"\b(VHS|tape warp|tracking error|analog video)\b", re.I),
        GlitchAesthetic.VHS,
        "VHS head-switch noise, chroma bleed, tape curvature",
        "clean digital 4k on VHS request",
        0.1,
    ),
    (
        re.compile(r"\b(datamosh|compression artifact|mpeg glitch)\b", re.I),
        GlitchAesthetic.DATAMOSH,
        "intentional macroblock smear, motion-coded glitch bands",
        "random jpeg blocks",
        0.15,
    ),
    (
        re.compile(r"\b(scanlines?|CRT|tube TV)\b", re.I),
        GlitchAesthetic.SCANLINE,
        "CRT aperture grille, phosphor glow, barrel edge",
        "flat LCD on CRT request",
        0.05,
    ),
    (
        re.compile(r"\b(chromatic aberration|RGB split|glitch offset)\b", re.I),
        GlitchAesthetic.RGB_SPLIT,
        "channel offset at high contrast edges, prismatic fringe",
        "uniform blur",
        0.08,
    ),
    (
        re.compile(r"\b(pixel sort|sorted pixels|glitch art)\b", re.I),
        GlitchAesthetic.PIXEL_SORT,
        "directional pixel sort streaks along luminance edges",
        "random salt noise",
        0.12,
    ),
)


class GlitchPlanner:
    def plan(self, prompt: str) -> GlitchPlan:
        text = prompt or ""
        for pat, aes, pos, neg, boost in _glitch_RULES:
            if pat.search(text):
                return GlitchPlan(aes, pos, neg, boost)
        return GlitchPlan(GlitchAesthetic.NONE, "", "", 0.0)

    def fragments(self, prompt: str) -> tuple[str, str, float]:
        p = self.plan(prompt)
        return p.positive, p.negative, p.serendipity_boost


# ==================================================================================================
# mutation: Prompt mutator: deterministic creative variants of a prompt for best-of-N, auto-refine and explore mode.
#
# Each variant swaps or adds one axis (time, weather, mood, medium, scale, palette, detail) instead of
# appending static guidance tags.
# ==================================================================================================


class MutationAxis(StrEnum):
    TIME_SHIFT = "time_shift"
    WEATHER_SWAP = "weather_swap"
    SCALE_FLIP = "scale_flip"
    PALETTE_INVERT = "palette_invert"
    MEDIUM_SWAP = "medium_swap"
    MOOD_FLIP = "mood_flip"
    DETAIL_ZOOM = "detail_zoom"


@dataclass(frozen=True)
class PromptMutation:
    axis: MutationAxis
    original: str
    mutated: str
    seed_salt: int


# Swap pairs per axis: a term found in the prompt is replaced by its partner, otherwise one partner is added.
_SWAP_PAIRS: dict[MutationAxis, tuple[tuple[str, str], ...]] = {
    MutationAxis.TIME_SHIFT: (("dawn", "midnight"), ("noon", "blue hour"), ("summer", "winter")),
    MutationAxis.WEATHER_SWAP: (("clear sky", "sudden rain"), ("calm", "windstorm"), ("fog", "harsh sun")),
    MutationAxis.MOOD_FLIP: (("serene", "ominous"), ("joyful", "melancholic"), ("chaotic", "still")),
    MutationAxis.MEDIUM_SWAP: (
        ("oil painting", "charcoal sketch"),
        ("photograph", "watercolor"),
        ("3d render", "linocut"),
    ),
    MutationAxis.SCALE_FLIP: (("wide establishing shot", "extreme close-up"), ("tiny figure", "colossal figure")),
}
_CLOSE_UP = re.compile(r"\bclose[- ]?up\b", re.I)


def _append(text: str, fragment: str) -> str:
    return f"{text}, {fragment}" if text else fragment


class PromptMutator:
    """Deterministic creative mutations from prompt + seed."""

    def __init__(self, axes: Sequence[MutationAxis] | None = None) -> None:
        self.axes = tuple(axes or MutationAxis)

    def mutate_batch(self, prompt: str, *, seed: int = 0, count: int = 4) -> list[PromptMutation]:
        """Up to ``count`` mutations; no-op mutations are dropped except the first."""
        text = (prompt or "").strip()
        out: list[PromptMutation] = []
        for i in range(max(1, count)):
            ax = self._pick_axis(text, seed + i * 997)
            mutated = self._apply(text, ax, seed + i)
            if mutated != text or i == 0:
                out.append(PromptMutation(ax, text, mutated, (seed + i) & 0xFFFF))
        return out[:count]

    def _pick_axis(self, prompt: str, seed: int) -> MutationAxis:
        h = int(hashlib.md5(f"{prompt}:{seed}".encode()).hexdigest()[:8], 16)
        return self.axes[h % len(self.axes)]

    def _apply(self, text: str, axis: MutationAxis, seed: int) -> str:
        if axis in _SWAP_PAIRS:
            return self._swap_pair(text, _SWAP_PAIRS[axis], seed)
        if axis == MutationAxis.PALETTE_INVERT:
            if "inverted" in text.lower():
                return text
            return _append(text, "inverted color palette, complementary shadow hues")
        # DETAIL_ZOOM: pull a close-up back out, otherwise push in.
        if _CLOSE_UP.search(text):
            return _CLOSE_UP.sub("wide environmental context", text, count=1)
        return _append(text, "extreme macro detail on focal texture")

    @staticmethod
    def _swap_pair(text: str, pairs: Sequence[tuple[str, str]], seed: int) -> str:
        for a, b in pairs:
            for src, dst in ((a, b), (b, a)):
                swapped, n = re.subn(rf"\b{re.escape(src)}\b", dst, text, count=1, flags=re.I)
                if n:
                    return swapped
        partner = pairs[seed % len(pairs)][1]
        return f"{text}, {partner} atmosphere" if text else partner


# ==================================================================================================
# paradox: Beautiful paradox — Escher, infinite loops, day-night in one frame.
#
# Pairs with ``logic.ContradictionScanner``: a *requested* paradox suppresses contradiction auto-resolve.
# ==================================================================================================


class BeautifulParadox(StrEnum):
    ESCHER = "escher"
    INFINITE_LOOP = "infinite_loop"
    DUAL_TIME = "dual_time"
    INSIDE_OUTSIDE = "inside_outside"
    NONE = "none"


@dataclass(frozen=True)
class ParadoxPlan:
    kind: BeautifulParadox
    positive: str
    negative: str
    suppress_contradiction_resolve: bool


_PATTERNS: tuple[tuple[re.Pattern, BeautifulParadox, str, str], ...] = (
    (
        re.compile(r"\b(escher|impossible stairs|penrose|relativity)\b", re.I),
        BeautifulParadox.ESCHER,
        "consistent impossible geometry, paradox architecture with single light source",
        "broken perspective without internal logic",
    ),
    (
        re.compile(r"\b(infinite loop|ouroboros|perpetual waterfall|droste)\b", re.I),
        BeautifulParadox.INFINITE_LOOP,
        "recursive visual loop, seamless self-reference, hypnotic repetition",
        "obvious seam in loop, broken recursion",
    ),
    (
        re.compile(r"\b(day and night|sun and moon same sky|dusk and dawn)\b", re.I),
        BeautifulParadox.DUAL_TIME,
        "split-time sky, coherent horizon, two times in one frame by design",
        "muddy gray without time read",
    ),
    (
        re.compile(r"\b(inside outside|interior landscape|room with sky)\b", re.I),
        BeautifulParadox.INSIDE_OUTSIDE,
        "interior-exterior boundary dissolve, architectural paradox",
        "random sky pasted in room",
    ),
)


class ParadoxKeeper:
    def plan(self, prompt: str) -> ParadoxPlan:
        text = prompt or ""
        for pat, kind, pos, neg in _PATTERNS:
            if pat.search(text):
                return ParadoxPlan(kind, pos, neg, suppress_contradiction_resolve=True)
        return ParadoxPlan(BeautifulParadox.NONE, "", "", False)

    def fragments(self, prompt: str) -> tuple[str, str, bool]:
        p = self.plan(prompt)
        return p.positive, p.negative, p.suppress_contradiction_resolve


# ==================================================================================================
# rhythm: Visual rhythm and pattern beat — music for the eye.
# ==================================================================================================


class VisualRhythm(StrEnum):
    REPETITION = "repetition"
    RADIAL = "radial"
    GRID = "grid"
    SPIRAL = "spiral"
    BROKEN = "broken"
    NONE = "none"


@dataclass(frozen=True)
class RhythmPlan:
    rhythm: VisualRhythm
    positive: str
    negative: str


_rhythm_RULES: tuple[tuple[re.Pattern, VisualRhythm, str, str], ...] = (
    (
        re.compile(r"\b(repeating|pattern|rhythm|motif|series of)\b", re.I),
        VisualRhythm.REPETITION,
        "motif repetition with variation, visual beat every third element",
        "clone stamp spam, identical duplicates",
    ),
    (
        re.compile(r"\b(radial|mandala|sunburst|concentric)\b", re.I),
        VisualRhythm.RADIAL,
        "radial symmetry with organic deviation, center-weighted energy",
        "perfect mathematical mandala filter",
    ),
    (
        re.compile(r"\b(grid|checker|tessellat|tiling pattern)\b", re.I),
        VisualRhythm.GRID,
        "disciplined grid with one broken cell, op-art restraint",
        "warped grid without intent",
    ),
    (
        re.compile(r"\b(spiral|vortex|whirlpool|fibonacci)\b", re.I),
        VisualRhythm.SPIRAL,
        "logarithmic spiral flow, eye travels inward",
        "random swirl smear",
    ),
    (
        re.compile(r"\b(broken rhythm|syncopat|irregular pattern)\b", re.I),
        VisualRhythm.BROKEN,
        "syncopated visual rhythm, deliberate off-beat element",
        "chaotic noise without pattern anchor",
    ),
)


class RhythmPlanner:
    def plan(self, prompt: str) -> RhythmPlan:
        text = prompt or ""
        for pat, rhythm, pos, neg in _rhythm_RULES:
            if pat.search(text):
                return RhythmPlan(rhythm, pos, neg)
        return RhythmPlan(VisualRhythm.NONE, "", "")

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


# ==================================================================================================
# scale: Scale relationships — make size contrast readable and intentional.
# ==================================================================================================


class ScaleRelation(StrEnum):
    TITAN = "titan"
    MINIATURE = "miniature"
    COSMIC = "cosmic"
    INTIMATE = "intimate"
    NONE = "none"


@dataclass(frozen=True)
class MagnitudePlan:
    relation: ScaleRelation
    positive: str
    negative: str


_scale_RULES: tuple[tuple[re.Pattern, ScaleRelation, str, str], ...] = (
    (
        re.compile(r"\b(colossal|giant|titan|kaiju|massive creature|towering)\b", re.I),
        ScaleRelation.TITAN,
        "scale cue via tiny foreground reference, atmospheric haze on upper mass",
        "giant without scale reference, floating size",
    ),
    (
        re.compile(r"\b(miniature|tiny world|microcosm|diorama scale)\b", re.I),
        ScaleRelation.MINIATURE,
        "forced perspective miniature read, shallow depth, toy-like material honesty",
        "real scale objects in miniature scene",
    ),
    (
        re.compile(r"\b(cosmic|galaxy|planet|astronomical|nebula backdrop)\b", re.I),
        ScaleRelation.COSMIC,
        "figure against cosmic scale, rim light from nebula, humility framing",
        "studio backdrop pretending to be space",
    ),
    (
        re.compile(r"\b(intimate scale|close world|macro universe|hands only)\b", re.I),
        ScaleRelation.INTIMATE,
        "micro-narrative in small frame, tactile nearness, world implied off-frame",
        "wide shot when intimacy requested",
    ),
)


class MagnitudePlanner:
    def plan(self, prompt: str) -> MagnitudePlan:
        text = prompt or ""
        for pat, rel, pos, neg in _scale_RULES:
            if pat.search(text):
                return MagnitudePlan(rel, pos, neg)
        return MagnitudePlan(ScaleRelation.NONE, "", "")

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


# ==================================================================================================
# surreal: Dream logic: when the prompt wants surrealism, stop fighting it.
#
# ``logic.ContradictionScanner`` resolves conflicts; this module *amplifies* intentional impossibility.
# ==================================================================================================


class SurrealMode(StrEnum):
    NONE = "none"
    DREAM = "dream"
    METAMORPHOSIS = "metamorphosis"
    SCALE_SHIFT = "scale_shift"
    FLOATING = "floating"
    MELTING = "melting"
    VAST_TINY = "vast_tiny"


@dataclass(frozen=True)
class DreamLogicPlan:
    mode: SurrealMode
    positive: str
    negative: str
    serendipity_boost: float  # add to base dial


_surreal_RULES: tuple[tuple[re.Pattern, SurrealMode, str, str, float], ...] = (
    (
        re.compile(r"\b(surreal|surrealism|dreamlike|dreamscape|oneiric)\b", re.I),
        SurrealMode.DREAM,
        "oneiric logic, symbolic juxtaposition, soft impossible transitions",
        "literal stock photo realism, mundane documentary",
        0.15,
    ),
    (
        re.compile(r"\b(morph|metamorph|transforming into|becoming)\b", re.I),
        SurrealMode.METAMORPHOSIS,
        "readable in-between form, coherent metamorphosis midpoint, intentional hybrid silhouette",
        "hard cut paste collage, incoherent limb soup",
        0.2,
    ),
    (
        re.compile(r"\b(giant tiny|colossal|miniature person|wrong scale)\b", re.I),
        SurrealMode.SCALE_SHIFT,
        "deliberate scale paradox, clear size relationship, toy-world or titan-world read",
        "accidental scale drift without intent",
        0.18,
    ),
    (
        re.compile(r"\b(floating|levitat|anti[- ]gravity|weightless)\b", re.I),
        SurrealMode.FLOATING,
        "weightless suspension with shadow or tether cue, dream physics",
        "random float without composition reason",
        0.12,
    ),
    (
        re.compile(r"\b(melting|dali|soft clock|viscous form)\b", re.I),
        SurrealMode.MELTING,
        "viscous deformation along gravity suggestion, molten form rhythm",
        "jpeg smear, random liquify artifact",
        0.2,
    ),
    (
        re.compile(r"\b(vast tiny|infinite|endless|impossibly large)\b", re.I),
        SurrealMode.VAST_TINY,
        "awe scale contrast, tiny figure against immense architecture or void",
        "flat backdrop without depth",
        0.1,
    ),
)


class DreamLogicPlanner:
    def plan(self, prompt: str) -> DreamLogicPlan:
        text = prompt or ""
        for pat, mode, pos, neg, boost in _surreal_RULES:
            if pat.search(text):
                return DreamLogicPlan(mode, pos, neg, boost)
        return DreamLogicPlan(SurrealMode.NONE, "", "", 0.0)

    def fragments(self, prompt: str) -> tuple[str, str, float]:
        p = self.plan(prompt)
        return p.positive, p.negative, p.serendipity_boost


# ==================================================================================================
# synesthesia: Synesthesia engine: map sound/music words to generation knobs, not just tags.
#
# Outputs a serendipity dial offset, a CFG bias, and color / rhythm prompt hints.
# ==================================================================================================


@dataclass(frozen=True)
class SynesthesiaTuning:
    mood: str
    serendipity_offset: float
    cfg_bias: float
    color_hint: str
    rhythm_hint: str


@dataclass(frozen=True)
class CrossModalMap:
    """Bundled tuning from audio/mood vocabulary."""

    tunings: tuple[SynesthesiaTuning, ...]
    merged_color: str
    merged_rhythm: str


_AUDIO_MOOD: tuple[tuple[re.Pattern, str, float, float, str, str], ...] = (
    (
        re.compile(r"\b(jazz|swing|brass|smoky club)\b", re.I),
        "jazz",
        0.12,
        1.05,
        "amber highlights, blue shadow jazz",
        "syncopated visual rhythm, off-beat accents",
    ),
    (
        re.compile(r"\b(synthwave|retrowave|80s electronic|vaporwave)\b", re.I),
        "synthwave",
        0.08,
        1.0,
        "magenta-cyan gradient sky, neon grid",
        "pulse repetition, horizon line beat",
    ),
    (
        re.compile(r"\b(classical|orchestral|symphony|strings)\b", re.I),
        "classical",
        0.05,
        1.08,
        "balanced warm-cool orchestral grade",
        "crescendo toward focal point",
    ),
    (
        re.compile(r"\b(techno|industrial|distorted bass|rave)\b", re.I),
        "techno",
        0.18,
        0.95,
        "high contrast strobe accents",
        "staccato geometric repetition",
    ),
    (
        re.compile(r"\b(lullaby|gentle|soft piano|ambient)\b", re.I),
        "ambient",
        -0.05,
        1.02,
        "desaturated pastels, low contrast",
        "slow breathing negative space",
    ),
    (
        re.compile(r"\b(chaos|cacophony|dissonance|noise)\b", re.I),
        "chaos",
        0.25,
        0.92,
        "clashing hues with one anchor",
        "broken grid, visual dissonance",
    ),
)


class SynesthesiaEngine:
    def map_prompt(self, prompt: str) -> CrossModalMap:
        text = prompt or ""
        hits = [SynesthesiaTuning(*fields) for pat, *fields in _AUDIO_MOOD if pat.search(text)]
        return CrossModalMap(
            tuple(hits),
            ", ".join(h.color_hint for h in hits[:2]),
            ", ".join(h.rhythm_hint for h in hits[:2]),
        )

    def diffusion_knobs(self, prompt: str) -> tuple[float, float]:
        """Return (serendipity_offset, cfg_multiplier)."""
        m = self.map_prompt(prompt)
        if not m.tunings:
            return 0.0, 1.0
        ser = sum(t.serendipity_offset for t in m.tunings) / len(m.tunings)
        return ser, math.prod(t.cfg_bias for t in m.tunings)


# ==================================================================================================
# vibe: Mood physics: emotional adjectives become numeric sampler knobs (serendipity, CFG, step emphasis).
# ==================================================================================================


@dataclass(frozen=True)
class MoodVector:
    name: str
    serendipity: float  # -0.1 .. +0.3 offset
    cfg_mult: float
    step_emphasis_early: float  # weight early vs late structure
    fragment: str


@dataclass(frozen=True)
class MoodPhysicsPlan:
    vectors: tuple[MoodVector, ...]
    serendipity_offset: float
    cfg_mult: float
    prompt_fragment: str


_VECTORS: tuple[tuple[re.Pattern, str, float, float, float, str], ...] = (
    (
        re.compile(r"\b(ominous|dread|foreboding|eerie)\b", re.I),
        "ominous",
        0.05,
        1.1,
        0.7,
        "low-key tension, withheld reveal",
    ),
    (
        re.compile(r"\b(joyful|euphoric|celebration|festive)\b", re.I),
        "joyful",
        0.15,
        1.0,
        0.4,
        "bright energy, open gesture",
    ),
    (
        re.compile(r"\b(melancholic|lonely|wistful|sad)\b", re.I),
        "melancholic",
        -0.02,
        1.05,
        0.55,
        "quiet emptiness, soft desaturation",
    ),
    (
        re.compile(r"\b(manic|frenetic|chaotic energy)\b", re.I),
        "manic",
        0.22,
        0.94,
        0.35,
        "visual stutter, high entropy",
    ),
    (re.compile(r"\b(serene|calm|peaceful|zen)\b", re.I), "serene", -0.08, 1.03, 0.6, "stillness, balanced masses"),
    (re.compile(r"\b(romantic|tender|intimate)\b", re.I), "romantic", 0.03, 1.06, 0.5, "soft proximity, warm falloff"),
    (
        re.compile(r"\b(awe|sublime|transcendent|majestic)\b", re.I),
        "awe",
        0.06,
        1.12,
        0.75,
        "scale overwhelm, vastness",
    ),
)


class MoodPhysics:
    def analyze(self, prompt: str) -> MoodPhysicsPlan:
        text = prompt or ""
        hits = [MoodVector(*fields) for pat, *fields in _VECTORS if pat.search(text)]
        if not hits:
            return MoodPhysicsPlan((), 0.0, 1.0, "")
        return MoodPhysicsPlan(
            tuple(hits),
            serendipity_offset=sum(h.serendipity for h in hits) / len(hits),
            cfg_mult=math.prod(h.cfg_mult for h in hits),
            prompt_fragment=", ".join(h.fragment for h in hits[:2]),
        )

    def step_emphasis_curve(self, plan: MoodPhysicsPlan, num_steps: int) -> tuple[float, ...]:
        """Per-step emphasis (at least 4 steps) blending from early-heavy to late-heavy by mood."""
        n = max(4, num_steps)
        if not plan.vectors:
            return (1.0,) * n
        early_w = sum(v.step_emphasis_early for v in plan.vectors) / len(plan.vectors)
        ts = (i / (n - 1) for i in range(n))
        return tuple(0.85 + 0.3 * ((1.0 - t) * early_w + t * (1.0 - early_w)) for t in ts)


# ==================================================================================================
# weathering: Weathering / patina — rust, wear, repair, graffiti layers as narrative.
#
# Different from era/period: this is *object biography*, not calendar year.
# ==================================================================================================


class WearLevel(StrEnum):
    PRISTINE = "pristine"
    WORN = "worn"
    WEATHERED = "weathered"
    RUINED = "ruined"
    REPAIRED = "repaired"
    LAYERED = "layered"


@dataclass(frozen=True)
class PatinaPlan:
    level: WearLevel
    positive: str
    negative: str


_weathering_RULES: tuple[tuple[re.Pattern, WearLevel, str, str], ...] = (
    (
        re.compile(r"\b(rust|rusted|oxidized|corroded|patina)\b", re.I),
        WearLevel.WEATHERED,
        "honest oxidation, rust streak direction, metal decay at edges",
        "plastic rust paint, uniform orange",
    ),
    (
        re.compile(r"\b(cracked|peeling paint|flaking|sun-bleached)\b", re.I),
        WearLevel.WORN,
        "UV-faded paint, crack network, substrate showing through",
        "random noise cracks",
    ),
    (
        re.compile(r"\b(ruins|collapsed|decay|abandoned|post-apocalyptic)\b", re.I),
        WearLevel.RUINED,
        "structural failure logic, vegetation reclaim, debris scatter",
        "pristine building labeled abandoned",
    ),
    (
        re.compile(r"\b(repaired|patched|duct tape|makeshift fix)\b", re.I),
        WearLevel.REPAIRED,
        "visible repair history, mismatched materials, human fix marks",
        "invisible restoration, museum perfect",
    ),
    (
        re.compile(r"\b(graffiti layers|sticker bomb|wheatpaste|urban decay)\b", re.I),
        WearLevel.LAYERED,
        "overlapping poster tear, graffiti palimpsest, urban time layers",
        "single flat graffiti sticker",
    ),
    (
        re.compile(r"\b(brand new|factory fresh|pristine|mint condition)\b", re.I),
        WearLevel.PRISTINE,
        "unworn edges, clean manufacturing, no patina",
        "scuffs on new object, fake wear",
    ),
)


class PatinaStoryteller:
    def plan(self, prompt: str) -> PatinaPlan:
        text = prompt or ""
        for pat, level, pos, neg in _weathering_RULES:
            if pat.search(text):
                return PatinaPlan(level, pos, neg)
        return PatinaPlan(WearLevel.PRISTINE, "", "")

    def fragments(self, prompt: str) -> tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


# ==================================================================================================
# imagination
# ==================================================================================================


@dataclass
class ImaginationPlan:
    prompt: str
    augmented_prompt: str = ""
    merged_positive: str = ""
    merged_negative: str = ""
    serendipity_dial: float = 0.25
    cfg_multiplier: float = 1.0
    step_emphasis: tuple[float, ...] | None = None
    suppress_contradiction_resolve: bool = False
    mutations: list[str] = field(default_factory=list)
    creative_trace: list[str] = field(default_factory=list)


def analyze_imagination(
    prompt: str,
    *,
    num_steps: int = 28,
    base_serendipity: float = 0.25,
    mutate_count: int = 0,
    mutate_seed: int = 0,
    random_constraint_seed: int | None = None,
) -> ImaginationPlan:
    """Run the creative analyzers; ``creative_trace`` names each one that contributed a positive fragment."""
    trace: list[str] = []
    pos_parts: list[str] = []
    neg_parts: list[str] = []

    def add(name: str, pos: str, neg: str = "") -> None:
        if pos:
            pos_parts.append(pos)
            trace.append(name)
        if neg:
            neg_parts.append(neg)

    dream_pos, dream_neg, dream_ser = DreamLogicPlanner().fragments(prompt)
    add("surreal", dream_pos, dream_neg)
    paradox_pos, paradox_neg, suppress = ParadoxKeeper().fragments(prompt)
    add("paradox", paradox_pos, paradox_neg)

    constraint = CreativeConstraintEngine().pack(prompt)
    if constraint.rules:
        add("constraint", constraint.positive, constraint.negative)
    elif random_constraint_seed is not None:
        rnd = CreativeConstraintEngine().suggest_random(seed=random_constraint_seed)
        add(f"constraint_random:{rnd.rules[0].value}", rnd.positive, rnd.negative)

    syn = SynesthesiaEngine()
    syn_map = syn.map_prompt(prompt)
    ser_off, syn_cfg = syn.diffusion_knobs(prompt)
    add("synesthesia", merge_csv_unique(syn_map.merged_color, syn_map.merged_rhythm))

    add("cinema", *ShotGrammar().fragments(prompt))
    mood = MoodPhysics().analyze(prompt)
    add("mood_physics", mood.prompt_fragment)
    add("weathering", *PatinaStoryteller().fragments(prompt))
    glitch_pos, glitch_neg, glitch_ser = GlitchPlanner().fragments(prompt)
    add("glitch", glitch_pos, glitch_neg)
    add("fusion", *GenreCollisionEngine().fragments(prompt))
    add("archetype", *SymbolMapEngine().fragments(prompt))
    add("rhythm", *RhythmPlanner().fragments(prompt))
    focal_pos, focal_neg, focal_cfg = FocalStoryteller().fragments(prompt)
    add("focal", focal_pos, focal_neg)
    add("collective", *CrowdGrammar().fragments(prompt))
    add("scale", *MagnitudePlanner().fragments(prompt))

    merged_pos = merge_csv_unique(*pos_parts)
    ser = float(base_serendipity) + dream_ser + glitch_ser + ser_off + mood.serendipity_offset

    mutations: list[str] = []
    if mutate_count > 0:
        muts = PromptMutator().mutate_batch(prompt, seed=mutate_seed, count=mutate_count)
        mutations = [m.mutated for m in muts if m.mutated.strip()]
        trace.append(f"mutations:{len(mutations)}")

    return ImaginationPlan(
        prompt=prompt,
        augmented_prompt=f"{prompt}, {merged_pos}" if merged_pos and prompt and len(merged_pos) < 350 else prompt,
        merged_positive=merged_pos,
        merged_negative=merge_csv_unique(*neg_parts),
        serendipity_dial=max(0.0, min(0.85, ser)),
        cfg_multiplier=mood.cfg_mult * syn_cfg * focal_cfg,
        step_emphasis=MoodPhysics().step_emphasis_curve(mood, num_steps) if mood.vectors else None,
        suppress_contradiction_resolve=suppress,
        mutations=mutations,
        creative_trace=trace,
    )


def imagination_sample_kwargs(plan: ImaginationPlan, *, base_negative: str = "") -> dict[str, Any]:
    kw: dict[str, Any] = {
        "prompt": plan.augmented_prompt or plan.prompt,
        "negative_prompt": merge_csv_unique(base_negative, plan.merged_negative),
        "cfg_scale_multiplier": plan.cfg_multiplier,
        "frontier_serendipity_dial": plan.serendipity_dial,
        "frontier_creative": True,
        "creative_trace": list(plan.creative_trace),
        "suppress_contradiction_resolve": plan.suppress_contradiction_resolve,
    }
    if plan.step_emphasis:
        kw["frontier_step_emphasis"] = list(plan.step_emphasis)
    if plan.mutations:
        kw["creative_prompt_mutations"] = list(plan.mutations)
    return kw


__all__ = [
    "ArchetypePlan",
    "ArchetypeSymbol",
    "SymbolMapEngine",
    "ShotGrammar",
    "ShotPlan",
    "ShotType",
    "CrowdGrammar",
    "CrowdPlan",
    "CrowdScale",
    "ConstraintPack",
    "CreativeConstraintEngine",
    "LimitRule",
    "FocalStory",
    "FocalStoryteller",
    "FocusIntent",
    "FusionPlan",
    "GenreCollisionEngine",
    "GenrePair",
    "GlitchAesthetic",
    "GlitchPlan",
    "GlitchPlanner",
    "MutationAxis",
    "PromptMutation",
    "PromptMutator",
    "BeautifulParadox",
    "ParadoxKeeper",
    "ParadoxPlan",
    "RhythmPlan",
    "RhythmPlanner",
    "VisualRhythm",
    "MagnitudePlan",
    "MagnitudePlanner",
    "ScaleRelation",
    "DreamLogicPlan",
    "DreamLogicPlanner",
    "SurrealMode",
    "CrossModalMap",
    "SynesthesiaEngine",
    "SynesthesiaTuning",
    "MoodPhysics",
    "MoodPhysicsPlan",
    "MoodVector",
    "PatinaPlan",
    "PatinaStoryteller",
    "WearLevel",
    "ImaginationPlan",
    "analyze_imagination",
    "imagination_sample_kwargs",
]
