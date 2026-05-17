"""
Style genome chaos layer — presets, fusion, hypermutation, apocalypse mode.

Turn creativity past 1.0 without breaking the StyleGenome compile path.
"""

from __future__ import annotations

import random
import uuid
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from .style_genome import StyleGenome

InventionMode = Literal["normal", "insane", "apocalypse", "chimera", "glitch", "eldritch", "cyberpunk"]

# ---------------------------------------------------------------------------
# Named presets (instant wild identities)
# ---------------------------------------------------------------------------
INSANE_PRESETS: Dict[str, Dict[str, object]] = {
    "glitch_cathedral": {
        "name": "Glitch Cathedral",
        "palette": "corrupted RGB channel split, sacramental gold bleeding into void black",
        "line": "datamoshed contour tearing, voxel-snapped edges",
        "surface": "CRT phosphor burn-in, interlaced scanline halos",
        "camera": "fisheye vertigo, horizon buckled 18 degrees",
        "lighting": "strobing practical neon through incense smoke",
        "signature": "sacred geometry dissolving into compression artifacts",
        "positive_fragments": (
            "ritual scale composition",
            "chromatic aberration as design element",
            "intentional JPEG ruin",
        ),
        "negative_fragments": (
            "clean corporate stock",
            "flat mobile snapshot",
            "sterile minimalism",
        ),
    },
    "biolume_abyss": {
        "name": "Biolume Abyss",
        "palette": "abyssal indigo with toxic lime bioluminescent veins",
        "line": "gelatinous silhouette edges, no hard outline",
        "surface": "wet membrane subsurface scatter, mucous specular",
        "camera": "macro plunge into cavern scale, subject dwarfed",
        "lighting": "only diegetic glow, zero ambient fill",
        "signature": "deep-sea pressure made visible as color",
        "positive_fragments": ("pelagic atmosphere", "particle murk", "alien ecology"),
        "negative_fragments": ("daylight scene", "cheerful palette", "dry desert"),
    },
    "rusted_sky_myth": {
        "name": "Rusted Sky Myth",
        "palette": "oxidized iron sky, iodine yellow horizon bruise",
        "line": "monumental silhouette carving, epic negative space",
        "surface": "granular rust dust on every plane, pitting",
        "camera": "tilted epic wide, tiny figures against titan ruins",
        "lighting": "dust-choked god rays, single apocalyptic sun",
        "signature": "post-collapse myth painted on industrial decay",
        "positive_fragments": ("megalith scale", "particulate air", "weathered myth"),
        "negative_fragments": ("modern clean city", "cheerful picnic", "studio seamless"),
    },
    "porcelain_fracture": {
        "name": "Porcelain Fracture",
        "palette": "bone white, hairline cobalt crack network, dried blood accent",
        "line": "kintsugi logic but violent, sharp fracture vectors",
        "surface": "glazed ceramic skin, craquelure everywhere",
        "camera": "clinical close portrait, uncomfortable proximity",
        "lighting": "cold top light, no fill, surgical shadow",
        "signature": "beauty through controlled breakage",
        "positive_fragments": ("micro fracture detail", "tactile glaze", "tension stillness"),
        "negative_fragments": ("cartoon smooth", "rubbery skin", "warm golden hour cliché"),
    },
    "neon_kaiju_ink": {
        "name": "Neon Kaiju Ink",
        "palette": "hot magenta and venom green on sumi black",
        "line": "explosive calligraphic stroke, splatter trajectory",
        "surface": "wet ink bleed into rice paper fiber",
        "camera": "dynamic low angle, motion implied in static frame",
        "lighting": "impossible neon rim on traditional ink wash",
        "signature": "ukiyo-e violence meeting arcade cabinet glow",
        "positive_fragments": ("ink splash physics", "scale drama", "electric atmosphere"),
        "negative_fragments": ("muted pastel", "western comic flat", "3d plastic render"),
    },
    "eldritch_taxonomy": {
        "name": "Eldritch Taxonomy",
        "palette": "fungal purple, bile yellow, wet charcoal",
        "line": "anatomical diagram wrongness, too many joints",
        "surface": "chitinous gloss, spore haze, mucous membrane",
        "camera": "specimen plate overhead, clinical horror",
        "lighting": "greenish morgue fluorescent, specular slime",
        "signature": "naturalist illustration of something that should not exist",
        "positive_fragments": ("specimen labeling aesthetic", "organic horror", "wet depth"),
        "negative_fragments": ("cute chibi", "safe fantasy", "stock vampire"),
    },
    "solar_punk_ruin": {
        "name": "Solarpunk Ruin",
        "palette": "verdigris copper, algae green, sun-bleached concrete",
        "line": "botanical wireframe over brutalist mass",
        "surface": "moss concrete, photovoltaic patina, rainwater streak",
        "camera": "elevated drone ruin gaze, nature reclaiming grid",
        "lighting": "harsh noon with chlorophyll bounce fill",
        "signature": "utopia already rotting beautifully",
        "positive_fragments": ("ecological entanglement", "structural decay detail", "hopeful rot"),
        "negative_fragments": ("sterile sci-fi hallway", "gray mud only", "empty greybox"),
    },
    "vhs_prophecy": {
        "name": "VHS Prophecy",
        "palette": "faded NTSC warm lift, magenta shadow cast",
        "line": "soft analog blur, tracking error displacement",
        "surface": "magnetic tape grain, chroma noise swarm",
        "camera": "handheld prophecy, date stamp energy",
        "lighting": "single on-camera flash in darkness",
        "signature": "found footage omen, retro dread",
        "positive_fragments": ("tracking lines", "timestamp mood", "analog horror"),
        "negative_fragments": ("crisp 4k digital", "clean HDR", "modern phone look"),
    },
}

WILD_PALETTES: Tuple[str, ...] = (
    "molten chrome sunset bleeding into tar black",
    "radioactive honeycomb lattice over bruised violet",
    "bleach-white heat shimmer with singed edge orange",
    "toxic lagoon teal eating dusty rose flesh tones",
    "static snow noise palette with one screaming red accent",
    "petroleum iridescence on rain-black pavement",
    "decomposing fruit magentas and bile chartreuse",
    "quantum frost blue with fractal gold interference",
    "candle soot monochrome except one arterial crimson",
    "holographic foil rainbow trapped under scratched acrylic",
)
WILD_LINES: Tuple[str, ...] = (
    "contour lines vibrating like heat distortion",
    "woodcut violence with digital stair-stepping",
    "hair-thin spirograph filigree over blunt masses",
    "torn paper collage edge, misregistered print layers",
    "calligraphy blade stroke with aerosol overspray halo",
    "blueprint schematic overlay on organic form",
    "kinetic blur smear on extremities only",
    "topographic map lines treating flesh as terrain",
)
WILD_SURFACES: Tuple[str, ...] = (
    "liquid mercury skin catching environment",
    "crystalline frost growing on velvet",
    "burnt sugar glass crackle over wax",
    "rusted razor mesh catching specular blood",
    "soap bubble thin film interference",
    "sandblasted hologram laminate peeling",
    "wet latex with micro bead sweat",
    "volcanic pumice porous matte",
)
WILD_CAMERAS: Tuple[str, ...] = (
    "peephole distortion, claustrophobic frame",
    "infrared trail cam wrongness",
    "tilt-shift miniature faking epic scale",
    "split diopter two focal planes at war",
    "mirror kaleidoscope multiplication",
    "security cam corner, brutal perspective",
    "underwater housing dome bend",
    "anamorphic oval bokeh crushing background",
)
WILD_LIGHT: Tuple[str, ...] = (
    "lightning freeze frame mixed with long exposure trails",
    "blacklight UV revealing hidden ink patterns",
    "multiple colored gels fighting one shadow",
    "lens flare as primary subject",
    "bioluminescent fill from subject itself",
    "firelight strobe through spinning slats",
    "polarized sky sucked dead, ground hyper-saturated",
)
WILD_POSITIVE: Tuple[str, ...] = (
    "hallucinatory detail density",
    "impossible material honesty",
    "dream logic scale shift",
    "aggressive negative space violence",
    "tactile temperature you can feel",
    "synesthetic color sound",
    "liminal threshold atmosphere",
    "designed accident composition",
    "maximalist micro clutter zones",
    "sacred profane tension",
)
WILD_NEGATIVE: Tuple[str, ...] = (
    "boring centered stock composition",
    "AI slop smoothness",
    "instagram filter pack",
    "generic deviantart lighting",
    "floating without weight",
    "plastic doll skin",
    "sameface syndrome",
    "watermark template",
    "low effort background void",
    "safe corporate illustration",
)

CHAOS_CLAUSE_POSITIVE: Tuple[str, ...] = (
    "visually unforgettable",
    "deliberately uncanny",
    "high concept art direction",
)
CHAOS_CLAUSE_NEGATIVE: Tuple[str, ...] = (
    "forgettable average",
    "template AI look",
    "boring safe composition",
)


def preset_genome(preset_id: str, *, seed_suffix: str = "") -> Optional[StyleGenome]:
    data = INSANE_PRESETS.get(preset_id.strip().lower().replace("-", "_"))
    if not data:
        return None
    d = dict(data)
    d["id"] = f"preset_{preset_id}_{seed_suffix or uuid.uuid4().hex[:6]}"
    return StyleGenome.from_dict(d)


def list_insane_presets() -> List[str]:
    return sorted(INSANE_PRESETS.keys())


def apply_chaos_level(genome: StyleGenome, chaos_level: float, *, rng: Optional[random.Random] = None) -> StyleGenome:
    """Push an existing genome toward maximum unhinged (0 = noop, 1 = full spice)."""
    level = max(0.0, min(1.0, float(chaos_level)))
    if level <= 0.01:
        return genome
    r = rng or random.Random()

    extra_pos = list(genome.positive_fragments)
    extra_neg = list(genome.negative_fragments)
    n_pos = max(1, int(level * 4))
    n_neg = max(1, int(level * 3))
    extra_pos.extend(r.sample(WILD_POSITIVE, k=min(n_pos, len(WILD_POSITIVE))))
    extra_neg.extend(r.sample(WILD_NEGATIVE, k=min(n_neg, len(WILD_NEGATIVE))))

    palette = genome.palette
    if level > 0.35 and r.random() < level:
        palette = merge_axis(palette, r.choice(WILD_PALETTES), r)

    line = genome.line
    if level > 0.45 and r.random() < level:
        line = merge_axis(line, r.choice(WILD_LINES), r)

    signature = genome.signature
    if level > 0.6:
        signature = f"{signature}; chaos injection {level:.0%}".strip("; ")

    return StyleGenome(
        id=genome.id,
        name=f"{genome.name} [chaos {level:.0%}]",
        palette=palette,
        line=line,
        surface=genome.surface or (r.choice(WILD_SURFACES) if level > 0.5 else ""),
        camera=genome.camera or (r.choice(WILD_CAMERAS) if level > 0.5 else ""),
        lighting=genome.lighting or (r.choice(WILD_LIGHT) if level > 0.5 else ""),
        signature=signature,
        anti_clone=genome.anti_clone,
        positive_fragments=tuple(extra_pos),
        negative_fragments=tuple(extra_neg),
        reasoning=genome.reasoning,
    )


def merge_axis(a: str, b: str, rng: random.Random) -> str:
    if not a.strip():
        return b
    if not b.strip():
        return a
    if rng.random() < 0.5:
        return f"{a}, {b}"
    return f"{b} overlaid on {a}"


def fuse_genomes(
    a: StyleGenome,
    b: StyleGenome,
    *,
    name: str = "",
    ratio: float = 0.5,
) -> StyleGenome:
    """Chimera: splice two genomes into one unstable identity."""
    r = max(0.0, min(1.0, ratio))
    pick_a = r >= 0.5

    def _pick(fa: str, fb: str) -> str:
        if fa and fb:
            return fa if (pick_a and r > 0.4) or (not pick_a and r < 0.6) else fb
        return fa or fb

    pos = tuple(dict.fromkeys(list(a.positive_fragments) + list(b.positive_fragments)))[:8]
    neg = tuple(dict.fromkeys(list(a.negative_fragments) + list(b.negative_fragments)))[:10]
    anti = tuple(dict.fromkeys(list(a.anti_clone) + list(b.anti_clone)))

    return StyleGenome(
        id=f"chimera_{uuid.uuid4().hex[:10]}",
        name=name or f"{a.name} x {b.name}",
        palette=_pick(a.palette, b.palette),
        line=_pick(a.line, b.line),
        surface=_pick(a.surface, b.surface),
        camera=merge_axis(a.camera, b.camera, random.Random(7)),
        lighting=_pick(a.lighting, b.lighting),
        signature=f"Chimera: {a.signature[:60]} // {b.signature[:60]}",
        anti_clone=anti,
        positive_fragments=pos + ("hybrid aesthetic", "visual tension between schools"),
        negative_fragments=neg,
        reasoning=f"Fusion of {a.id} and {b.id} at ratio {r:.2f}",
    )


def hypermutate(
    genome: StyleGenome,
    *,
    intensity: float = 0.85,
    seed: int = 0,
) -> StyleGenome:
    """Aggressive random walk in style space — sibling genome, not a copy."""
    r = random.Random(seed ^ hash(genome.id))
    level = max(0.0, min(1.0, intensity))

    def maybe_replace(current: str, pool: Sequence[str]) -> str:
        if r.random() < level:
            return r.choice(pool)
        if r.random() < level * 0.5:
            return merge_axis(current, r.choice(pool), r)
        return current

    return StyleGenome(
        id=f"mut_{uuid.uuid4().hex[:8]}",
        name=f"{genome.name} (mutant)",
        palette=maybe_replace(genome.palette, WILD_PALETTES),
        line=maybe_replace(genome.line, WILD_LINES),
        surface=maybe_replace(genome.surface, WILD_SURFACES),
        camera=maybe_replace(genome.camera, WILD_CAMERAS),
        lighting=maybe_replace(genome.lighting, WILD_LIGHT),
        signature=maybe_replace(genome.signature, WILD_POSITIVE),
        anti_clone=genome.anti_clone,
        positive_fragments=tuple(
            dict.fromkeys(
                list(genome.positive_fragments)
                + list(r.sample(WILD_POSITIVE, k=min(3, len(WILD_POSITIVE))))
            )
        ),
        negative_fragments=tuple(
            dict.fromkeys(
                list(genome.negative_fragments)
                + list(r.sample(WILD_NEGATIVE, k=min(4, len(WILD_NEGATIVE))))
            )
        ),
        reasoning=f"Hypermutation of {genome.id} @ {level:.0%}",
    )


def invent_insane_batch(
    prompt: str,
    n: int,
    *,
    seed: int = 42,
    mode: InventionMode = "insane",
    chaos_level: float = 0.85,
) -> List[StyleGenome]:
    """Deterministic wild invention without LLM."""
    r = random.Random(seed ^ hash(prompt))
    presets = list(INSANE_PRESETS.keys())
    genomes: List[StyleGenome] = []

    if mode == "glitch":
        g = preset_genome("glitch_cathedral", seed_suffix=str(seed))
        if g:
            genomes.append(apply_chaos_level(g, chaos_level, rng=r))
        mode = "insane"

    if mode == "eldritch":
        preset_ids = ["eldritch_taxonomy"] * n
    elif mode == "cyberpunk":
        pool = ["neon_kaiju_ink", "glitch_cathedral", "vhs_prophecy"]
        preset_ids = [r.choice(pool) for _ in range(n)]
    else:
        preset_ids = r.sample(presets, k=min(n, len(presets)))
        while len(preset_ids) < n:
            preset_ids.append(r.choice(presets))

    for i, pid in enumerate(preset_ids[:n]):
        if mode in ("insane", "apocalypse", "eldritch", "cyberpunk"):
            g = preset_genome(pid, seed_suffix=f"{seed}_{i}")
            if g is None:
                continue
            level = chaos_level if mode != "apocalypse" else 1.0
            if mode == "apocalypse":
                g = apply_chaos_level(g, 1.0, rng=r)
                g = hypermutate(g, intensity=0.9, seed=seed + i)
            else:
                g = apply_chaos_level(g, level, rng=r)
            if r.random() < 0.4:
                g = hypermutate(g, intensity=0.5 + level * 0.4, seed=seed + i + 7)
            genomes.append(g)
        elif mode == "chimera" and len(genomes) >= 2:
            break

    if mode == "chimera":
        base = invent_insane_batch(prompt, max(n * 2, 4), seed=seed, mode="insane", chaos_level=chaos_level)
        chimeras: List[StyleGenome] = []
        for i in range(n):
            if len(base) < 2:
                break
            a, b = r.sample(base, 2)
            chimeras.append(fuse_genomes(a, b, ratio=r.random()))
        return chimeras[:n]

    while len(genomes) < n:
        g = StyleGenome(
            id=f"wild_{uuid.uuid4().hex[:8]}",
            name=f"Wild strain {len(genomes) + 1}",
            palette=r.choice(WILD_PALETTES),
            line=r.choice(WILD_LINES),
            surface=r.choice(WILD_SURFACES),
            camera=r.choice(WILD_CAMERAS),
            lighting=r.choice(WILD_LIGHT),
            signature=r.choice(WILD_POSITIVE),
            anti_clone=("not generic", "not stock photo", "not AI slop"),
            positive_fragments=r.sample(WILD_POSITIVE, k=3),
            negative_fragments=r.sample(WILD_NEGATIVE, k=4),
            reasoning=f"Pure chaos strain for: {prompt[:80]}",
        )
        genomes.append(apply_chaos_level(g, chaos_level, rng=r))

    # Dedupe by preset/base id stem (hypermutate keeps new ids but same name stem)
    seen_names: set[str] = set()
    unique: List[StyleGenome] = []
    for g in genomes:
        key = g.name.split("[")[0].strip().lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        unique.append(g)
    while len(unique) < n and len(unique) < len(presets) + 3:
        extra = invent_insane_batch(
            prompt, 1, seed=seed + len(unique) * 31, mode="insane", chaos_level=chaos_level
        )
        for g in extra:
            key = g.name.split("[")[0].strip().lower()
            if key not in seen_names:
                seen_names.add(key)
                unique.append(g)
                break
    return unique[:n]


def auto_chaos_clauses(chaos_level: float) -> Tuple[str, ...]:
    """PromptStack clause names to auto-attach at high chaos."""
    if chaos_level < 0.35:
        return ()
    if chaos_level < 0.7:
        return ("style.surreal",)
    return ("style.surreal", "style.chaos")


__all__ = [
    "InventionMode",
    "INSANE_PRESETS",
    "apply_chaos_level",
    "auto_chaos_clauses",
    "fuse_genomes",
    "hypermutate",
    "invent_insane_batch",
    "list_insane_presets",
    "preset_genome",
]
