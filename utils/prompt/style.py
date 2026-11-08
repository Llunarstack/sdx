"""Style genomes: native fast paths, the genome type, chaos presets, the inventor, prompt mutation and the explore planner."""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import textwrap
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from utils.modeling.model_paths import default_qwen_path
from utils.prompt.text import merge_csv_unique, token_set


def style_native_status() -> dict[str, Any]:
    """Style-genome native backend availability (``explore_styles --show-native``)."""
    out: dict[str, Any] = {"rust_style_ops": False, "cuda_style_pick": False, "go_explore": False}
    try:
        from sdx_native.ffi import cuda_available, rust_available
        from sdx_native.native_tools import go_sdx_manifest_exe

        out["rust_style_ops"] = rust_available()
        out["cuda_style_pick"] = cuda_available()
        out["go_explore"] = go_sdx_manifest_exe() is not None
    except Exception:
        pass
    return out


def text_overlap(a: str, b: str) -> float:
    from sdx_native.ffi import maybe_token_jaccard

    j = maybe_token_jaccard(a, b)
    if j is not None:
        return j
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def pick_best_embedding_index(
    query: np.ndarray,
    candidates: np.ndarray,
) -> tuple[int, float]:
    from sdx_native.ffi import maybe_pick_best_style_embedding

    picked = maybe_pick_best_style_embedding(query, candidates)
    if picked is not None:
        return picked
    qn = query / (np.linalg.norm(query) + 1e-8)
    cn = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
    scores = cn @ qn
    idx = int(np.argmax(scores))
    return idx, float(scores[idx])


def explore_stats_python(manifest_path: Path) -> str:
    """Pure-Python explore manifest stats (same fields as Go ``explore-stats``)."""
    import json
    from collections import Counter

    by_genome: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    captions: set[str] = set()
    total = 0
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            gid = str(obj.get("style_genome_id") or "(none)")
            by_genome[gid] += 1
            kind = str(obj.get("candidate_kind") or "base")
            by_kind[kind] += 1
            cap = str(obj.get("caption") or "")
            if cap:
                captions.add(cap)
    lines = [
        f"explore_manifest: {manifest_path}",
        f"  rows: {total}",
        f"  unique captions: {len(captions)}",
        "  by candidate_kind:",
    ]
    for k in sorted(by_kind):
        lines.append(f"    {k}: {by_kind[k]}")
    lines.append("  by style_genome_id:")
    for g in sorted(by_genome):
        lines.append(f"    {g}: {by_genome[g]}")
    return "\n".join(lines) + "\n"


def explore_dedupe_python(in_path: Path, out_path: Path, *, key: str = "style_genome_id") -> int:
    """Dedupe JSONL by *key*; returns number of rows written."""
    import json

    seen: dict[str, dict] = {}
    order: list[str] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = str(obj.get(key) or "")
            if not k or k in seen:
                continue
            seen[k] = obj
            order.append(k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for k in order:
            f.write(json.dumps(seen[k], ensure_ascii=False) + "\n")
    return len(order)


def run_go_explore_stats(manifest_path: Path) -> str | None:
    try:
        from sdx_native.native_tools import go_sdx_manifest_exe

        exe = go_sdx_manifest_exe()
        if exe is not None:
            r = subprocess.run(
                [str(exe), "explore-stats", str(manifest_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode == 0:
                return r.stdout
    except Exception:
        pass
    if manifest_path.is_file():
        return explore_stats_python(manifest_path)
    return None


def run_go_explore_dedupe(
    in_path: Path,
    out_path: Path,
    *,
    key: str = "style_genome_id",
) -> bool:
    try:
        from sdx_native.native_tools import go_sdx_manifest_exe

        exe = go_sdx_manifest_exe()
        if exe is not None:
            r = subprocess.run(
                [str(exe), "explore-dedupe", "-o", str(out_path), "--key", key, str(in_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r.returncode == 0:
                return True
    except Exception:
        pass
    if not in_path.is_file():
        return False
    try:
        explore_dedupe_python(in_path, out_path, key=key)
        return True
    except Exception:
        return False


@dataclass
class StyleGenome:
    """Invented style identity (not a single artist name clone)."""

    id: str
    name: str
    palette: str = ""
    line: str = ""
    surface: str = ""
    camera: str = ""
    lighting: str = ""
    signature: str = ""
    anti_clone: tuple[str, ...] = ()
    positive_fragments: tuple[str, ...] = ()
    negative_fragments: tuple[str, ...] = ()
    reasoning: str = ""

    def axis_tokens(self) -> list[str]:
        out: list[str] = []
        for part in (self.palette, self.line, self.surface, self.camera, self.lighting, self.signature):
            p = (part or "").strip()
            if p:
                out.append(p)
        out.extend(x for x in self.positive_fragments if (x or "").strip())
        return out

    def style_head_string(self) -> str:
        """T5 style channel text (medium + look, not full scene)."""
        parts = self.axis_tokens()[:8]
        if not parts:
            return self.name.strip()
        return ", ".join(parts)

    def compile_positive(self, base_prompt: str) -> str:
        base = (base_prompt or "").strip()
        extras = ", ".join(self.axis_tokens())
        if extras:
            return merge_csv_unique(base, extras)
        return base

    def compile_negative(self, base_negative: str) -> str:
        neg = (base_negative or "").strip()
        chunks: list[str] = []
        if self.negative_fragments:
            chunks.append(", ".join(self.negative_fragments))
        if self.anti_clone:
            chunks.append(", ".join(self.anti_clone))
        if chunks:
            return merge_csv_unique(neg, ", ".join(chunks))
        return neg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["anti_clone"] = list(self.anti_clone)
        d["positive_fragments"] = list(self.positive_fragments)
        d["negative_fragments"] = list(self.negative_fragments)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleGenome:
        def _tuple(key: str) -> tuple[str, ...]:
            raw = data.get(key) or ()
            if isinstance(raw, str):
                return tuple(x.strip() for x in raw.split(",") if x.strip())
            return tuple(str(x).strip() for x in raw if str(x).strip())

        gid = str(data.get("id") or "").strip() or f"genome_{uuid.uuid4().hex[:10]}"
        return cls(
            id=gid,
            name=str(data.get("name") or "Invented style").strip(),
            palette=str(data.get("palette") or "").strip(),
            line=str(data.get("line") or "").strip(),
            surface=str(data.get("surface") or "").strip(),
            camera=str(data.get("camera") or "").strip(),
            lighting=str(data.get("lighting") or "").strip(),
            signature=str(data.get("signature") or "").strip(),
            anti_clone=_tuple("anti_clone"),
            positive_fragments=_tuple("positive_fragments"),
            negative_fragments=_tuple("negative_fragments"),
            reasoning=str(data.get("reasoning") or "").strip(),
        )


def nearest_catalog_style_overlap(genome: StyleGenome) -> tuple[str, float]:
    """
    Return (best_matching_style_id, overlap_score 0–1) against built-in STYLE_SPECS keywords.
    Higher = closer to an existing catalog style (less novel).
    """
    try:
        from config.styles import STYLE_SPECS
    except ImportError:
        return "", 0.0

    text = " ".join(genome.axis_tokens()).lower()
    if not text:
        return "", 0.0

    best_id = ""
    best_score = 0.0
    tokens = set(re.findall(r"[a-z0-9]+", text))
    if not tokens:
        return "", 0.0

    for spec in STYLE_SPECS:
        kw_text = ", ".join(spec.keywords)
        try:
            overlap = text_overlap(" ".join(genome.axis_tokens()), kw_text)
        except Exception:
            kw_tokens: set[str] = set()
            for kw in spec.keywords:
                kw_tokens.update(re.findall(r"[a-z0-9]+", kw.lower()))
            if not kw_tokens:
                continue
            overlap = len(tokens & kw_tokens) / max(len(kw_tokens), 1)
        if overlap > best_score:
            best_score = overlap
            best_id = spec.id

    return best_id, min(1.0, best_score)


def is_genome_novel_enough(genome: StyleGenome, *, max_catalog_overlap: float = 0.55) -> bool:
    _, score = nearest_catalog_style_overlap(genome)
    return score < max_catalog_overlap


InventionMode = Literal["normal", "insane", "apocalypse", "chimera", "glitch", "eldritch", "cyberpunk"]

# Named presets (instant wild identities)
INSANE_PRESETS: dict[str, dict[str, object]] = {
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

WILD_PALETTES: tuple[str, ...] = (
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
WILD_LINES: tuple[str, ...] = (
    "contour lines vibrating like heat distortion",
    "woodcut violence with digital stair-stepping",
    "hair-thin spirograph filigree over blunt masses",
    "torn paper collage edge, misregistered print layers",
    "calligraphy blade stroke with aerosol overspray halo",
    "blueprint schematic overlay on organic form",
    "kinetic blur smear on extremities only",
    "topographic map lines treating flesh as terrain",
)
WILD_SURFACES: tuple[str, ...] = (
    "liquid mercury skin catching environment",
    "crystalline frost growing on velvet",
    "burnt sugar glass crackle over wax",
    "rusted razor mesh catching specular blood",
    "soap bubble thin film interference",
    "sandblasted hologram laminate peeling",
    "wet latex with micro bead sweat",
    "volcanic pumice porous matte",
)
WILD_CAMERAS: tuple[str, ...] = (
    "peephole distortion, claustrophobic frame",
    "infrared trail cam wrongness",
    "tilt-shift miniature faking epic scale",
    "split diopter two focal planes at war",
    "mirror kaleidoscope multiplication",
    "security cam corner, brutal perspective",
    "underwater housing dome bend",
    "anamorphic oval bokeh crushing background",
)
WILD_LIGHT: tuple[str, ...] = (
    "lightning freeze frame mixed with long exposure trails",
    "blacklight UV revealing hidden ink patterns",
    "multiple colored gels fighting one shadow",
    "lens flare as primary subject",
    "bioluminescent fill from subject itself",
    "firelight strobe through spinning slats",
    "polarized sky sucked dead, ground hyper-saturated",
)
WILD_POSITIVE: tuple[str, ...] = (
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
WILD_NEGATIVE: tuple[str, ...] = (
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


def preset_genome(preset_id: str, *, seed_suffix: str = "") -> StyleGenome | None:
    data = INSANE_PRESETS.get(preset_id.strip().lower().replace("-", "_"))
    if not data:
        return None
    d = dict(data)
    d["id"] = f"preset_{preset_id}_{seed_suffix or uuid.uuid4().hex[:6]}"
    return StyleGenome.from_dict(d)


def list_insane_presets() -> list[str]:
    return sorted(INSANE_PRESETS.keys())


def apply_chaos_level(genome: StyleGenome, chaos_level: float, *, rng: random.Random | None = None) -> StyleGenome:
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
            dict.fromkeys(list(genome.positive_fragments) + list(r.sample(WILD_POSITIVE, k=min(3, len(WILD_POSITIVE)))))
        ),
        negative_fragments=tuple(
            dict.fromkeys(list(genome.negative_fragments) + list(r.sample(WILD_NEGATIVE, k=min(4, len(WILD_NEGATIVE)))))
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
) -> list[StyleGenome]:
    """Deterministic wild invention without LLM."""
    r = random.Random(seed ^ hash(prompt))
    presets = list(INSANE_PRESETS.keys())
    genomes: list[StyleGenome] = []

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
        chimeras: list[StyleGenome] = []
        for _ in range(n):
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
    unique: list[StyleGenome] = []
    for g in genomes:
        key = g.name.split("[")[0].strip().lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        unique.append(g)
    while len(unique) < n and len(unique) < len(presets) + 3:
        extra = invent_insane_batch(prompt, 1, seed=seed + len(unique) * 31, mode="insane", chaos_level=chaos_level)
        for g in extra:
            key = g.name.split("[")[0].strip().lower()
            if key not in seen_names:
                seen_names.add(key)
                unique.append(g)
                break
    return unique[:n]


def auto_chaos_clauses(chaos_level: float) -> tuple[str, ...]:
    """PromptStack clause names to auto-attach at high chaos."""
    if chaos_level < 0.35:
        return ()
    if chaos_level < 0.7:
        return ("style.surreal",)
    return ("style.surreal", "style.chaos")


# Resolved at import time so local Qwen3-14B (or legacy Qwen2.5) folders win.
_QWEN_PATH = Path(default_qwen_path())

_FALLBACK_PALETTES = (
    "volcanic glass highlights on desaturated umber base",
    "bioluminescent cyan accents over charcoal neutrals",
    "sun-bleached terracotta and oxidized copper",
    "frosted lavender mist with pearl grey shadows",
    "electric magenta rim light on deep indigo fill",
)
_FALLBACK_LINES = (
    "variable-weight ink strokes with dry-brush breaks",
    "clean vector contours with one weighted accent edge",
    "scratchy gestural linework, incomplete outlines",
    "engraved cross-hatching with selective solid blacks",
)
_FALLBACK_SURFACES = (
    "visible paper tooth, matte gouache layering",
    "subsurface wax resist on rough cotton",
    "anisotropic brushed metal with micro-scratches",
    "chalky pigment bloom at edge transitions",
)
_FALLBACK_CAMERAS = (
    "wide angle with slight dutch tilt",
    "telephoto compression, shallow depth of field",
    "low hero angle, asymmetric negative space",
    "overhead tableau framing, graphic flatness",
)
_FALLBACK_LIGHT = (
    "single hard key with colored bounce fill",
    "overcast skylight, soft wrap, no blown highlights",
    "rim-only silhouette with fog scatter",
    "practical lamp pools in otherwise dim scene",
)
_ANTI_CLONE = (
    "not studio ghibli",
    "not miyazaki",
    "not disney",
    "not pixar",
    "not marvel comic",
    "not generic anime",
)


def _extract_json_array(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    return []


def _qwen_invent_genomes(
    prompt: str,
    n: int,
    creativity_level: float,
    *,
    device: str = "cpu",
    max_new_tokens: int = 1024,
    invention_mode: InventionMode = "normal",
) -> list[StyleGenome]:
    if not _QWEN_PATH.exists() or n < 1:
        return []
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return []

    try:
        tok = AutoTokenizer.from_pretrained(str(_QWEN_PATH), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(_QWEN_PATH),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            device_map=device if device != "cpu" else None,
        )
        model.eval()

        insane = invention_mode in ("insane", "apocalypse", "chimera", "glitch", "eldritch", "cyberpunk")
        novelty = (
            "GO UNHINGED: impossible materials, sacred-profane tension, visual danger, synesthetic color. "
            "No famous artist names. Make statisticians weep."
            if insane or creativity_level > 0.85
            else "Maximize originality — avoid naming famous artists or trademarked franchises."
            if creativity_level > 0.55
            else "Balance novelty with clarity; still avoid artist name drops."
        )

        system_msg = textwrap.dedent("""
            You invent NEW visual styles for an AI image model. Output only valid JSON.
            Each style must be a fresh aesthetic identity built from orthogonal axes, not a copy
            of a known artist or franchise.
        """).strip()
        if insane:
            system_msg += (
                " You are allowed to be extreme: glitch sacrament, biolume abyss, rusted sky myth, "
                "porcelain fracture, eldritch taxonomy — but always original phrasing."
            )

        user_msg = textwrap.dedent(f"""
            Scene / subject prompt: "{prompt}"
            Invention mode: {invention_mode}

            Invent exactly {n} distinct style genomes as a JSON array. {novelty}

            Each object must have:
            - "name": short title (2–5 words)
            - "palette", "line", "surface", "camera", "lighting": one vivid phrase each
            - "signature": one sentence tying the look together
            - "positive_fragments": array of 2–4 extra comma-ready tags
            - "negative_fragments": array of 3–6 anti-generic terms
            - "anti_clone": array of 2–4 "not X" phrases blocking famous style clones
            - "reasoning": one sentence

            Do not repeat the same palette or line approach across genomes.
            Respond with ONLY the JSON array, no markdown fences.
        """).strip()

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=creativity_level > 0.25,
                temperature=max(0.2, min(1.1, 0.35 + creativity_level * 0.75)),
                top_p=0.92,
                repetition_penalty=1.08,
                pad_token_id=tok.eos_token_id,
            )

        response = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        raw_items = _extract_json_array(response)
        genomes: list[StyleGenome] = []
        for i, item in enumerate(raw_items[:n]):
            item = dict(item)
            item.setdefault("id", f"qwen_{uuid.uuid4().hex[:8]}")
            item.setdefault("name", f"Invented style {i + 1}")
            g = StyleGenome.from_dict(item)
            if is_genome_novel_enough(g):
                genomes.append(g)
        return genomes
    except Exception:
        return []


def _fallback_invent_genomes(
    prompt: str,
    n: int,
    *,
    seed: int = 42,
    creativity_level: float = 0.7,
) -> list[StyleGenome]:
    h = hashlib.sha256(f"{prompt}|{seed}|{creativity_level:.2f}".encode()).hexdigest()
    rng = random.Random(int(h[:8], 16))
    genomes: list[StyleGenome] = []
    used_sigs: set[str] = set()

    for i in range(n):
        palette = rng.choice(_FALLBACK_PALETTES)
        line = rng.choice(_FALLBACK_LINES)
        surface = rng.choice(_FALLBACK_SURFACES)
        camera = rng.choice(_FALLBACK_CAMERAS)
        lighting = rng.choice(_FALLBACK_LIGHT)
        signature = (
            f"{palette.split(',')[0]} meets {line.split(',')[0]} — "
            f"intentional {rng.choice(('asymmetry', 'tension', 'stillness', 'rhythm'))} for the scene"
        )
        if signature in used_sigs:
            signature += f" (variant {i})"
        used_sigs.add(signature)

        anti = tuple(rng.sample(_ANTI_CLONE, k=min(3, len(_ANTI_CLONE))))
        pos_extra = (
            "original composition grammar",
            "material-accurate rendering",
            "cohesive color script",
        )
        neg_extra = (
            "generic stock photo",
            "template composition",
            "muddy midtones",
            "overprocessed HDR",
        )

        genomes.append(
            StyleGenome(
                id=f"fb_{uuid.uuid4().hex[:8]}",
                name=f"Synthetic genome {i + 1}",
                palette=palette,
                line=line,
                surface=surface,
                camera=camera,
                lighting=lighting,
                signature=signature,
                anti_clone=anti,
                positive_fragments=pos_extra,
                negative_fragments=neg_extra,
                reasoning=f"Deterministic axis blend for prompt intent (creativity={creativity_level:.2f}).",
            )
        )
    return genomes


class StyleInventor:
    """Propose N novel style genomes for a base prompt."""

    def __init__(self, *, device: str = "cpu", use_qwen: bool = True) -> None:
        self.device = device
        self.use_qwen = use_qwen

    def invent(
        self,
        prompt: str,
        n: int = 3,
        *,
        creativity_level: float = 0.75,
        seed: int = 42,
        max_catalog_overlap: float = 0.55,
        invention_mode: InventionMode = "normal",
        chaos_level: float = 0.0,
    ) -> list[StyleGenome]:
        prompt = (prompt or "").strip()
        if not prompt or n < 1:
            return []

        mode: InventionMode = invention_mode  # type: ignore[assignment]
        chaos = max(0.0, min(1.0, float(chaos_level)))
        insane_modes = ("insane", "apocalypse", "chimera", "glitch", "eldritch", "cyberpunk")
        if mode in insane_modes:
            max_catalog_overlap = max(max_catalog_overlap, 0.92)

        genomes: list[StyleGenome] = []
        if mode in insane_modes:
            genomes = invent_insane_batch(
                prompt,
                n,
                seed=seed,
                mode=mode,
                chaos_level=max(chaos, 0.65 if mode == "insane" else 0.9),
            )

        if self.use_qwen and mode not in insane_modes:
            genomes = _qwen_invent_genomes(
                prompt,
                n,
                creativity_level,
                device=self.device,
                invention_mode=mode,
            )
        elif self.use_qwen and len(genomes) < n:
            extra = _qwen_invent_genomes(
                prompt,
                n - len(genomes),
                max(creativity_level, 0.9),
                device=self.device,
                invention_mode=mode,
            )
            genomes.extend(extra)

        if len(genomes) < n and mode not in insane_modes:
            need = n - len(genomes)
            genomes.extend(_fallback_invent_genomes(prompt, need, seed=seed, creativity_level=creativity_level))
        elif len(genomes) < n:
            genomes.extend(
                invent_insane_batch(
                    prompt,
                    n - len(genomes),
                    seed=seed + 17,
                    mode="insane",
                    chaos_level=max(chaos, 0.8),
                )
            )

        if chaos > 0.01:
            chaos_rng = random.Random(seed + 9001)
            genomes = [apply_chaos_level(g, chaos, rng=chaos_rng) for g in genomes]

        # Filter near-duplicate catalog styles; dedupe by signature
        filtered: list[StyleGenome] = []
        seen: set[str] = set()
        for g in genomes:
            sig = g.signature.lower()[:80]
            if sig in seen:
                continue
            if not is_genome_novel_enough(g, max_catalog_overlap=max_catalog_overlap):
                continue
            seen.add(sig)
            filtered.append(g)
            if len(filtered) >= n:
                break

        if len(filtered) < n:
            for g in genomes:
                if g not in filtered:
                    filtered.append(g)
                if len(filtered) >= n:
                    break

        return filtered[:n]


# ---------------------------------------------------------------------------
# Vocabulary tables for mutations
# ---------------------------------------------------------------------------

_SYNONYM_MAP: dict[str, list[str]] = {
    # Subjects
    "woman": ["female figure", "young woman", "girl", "lady", "female character"],
    "man": ["male figure", "young man", "guy", "gentleman", "male character"],
    "person": ["figure", "individual", "character", "human"],
    "girl": ["young woman", "female figure", "young girl"],
    "boy": ["young man", "male figure", "young boy"],
    # Environments
    "forest": ["woodland", "dense forest", "ancient forest", "wooded area", "tree canopy"],
    "city": ["urban landscape", "cityscape", "metropolitan area", "downtown"],
    "beach": ["shoreline", "coastal scene", "sandy beach", "ocean shore"],
    "mountain": ["mountain range", "alpine landscape", "rocky peaks", "highland"],
    "room": ["interior space", "indoor setting", "chamber", "living space"],
    # Lighting
    "sunset": ["golden hour", "dusk", "twilight", "evening light", "setting sun"],
    "sunrise": ["dawn", "early morning light", "golden hour morning", "first light"],
    "night": ["nighttime", "after dark", "nocturnal scene", "moonlit"],
    "dark": ["shadowy", "dimly lit", "low light", "moody darkness"],
    "bright": ["well-lit", "luminous", "radiant", "brilliantly lit"],
    # Styles
    "realistic": ["photorealistic", "lifelike", "true-to-life", "naturalistic"],
    "anime": ["anime style", "Japanese animation style", "manga-inspired"],
    "painting": ["painted", "painterly", "fine art painting", "artistic rendering"],
    "beautiful": ["stunning", "gorgeous", "striking", "captivating", "breathtaking"],
    "detailed": ["highly detailed", "intricate", "fine detail", "meticulously rendered"],
    # Actions
    "standing": ["posed standing", "upright stance", "standing figure"],
    "sitting": ["seated", "sitting pose", "resting seated"],
    "walking": ["in motion", "mid-stride", "walking pose"],
    "looking": ["gazing", "staring", "glancing", "peering"],
    "wearing": ["dressed in", "clothed in", "adorned with"],
}

_SPECIFICITY_ADDITIONS: dict[str, list[str]] = {
    "sunset": ["warm amber and orange tones", "long shadows", "golden rim light"],
    "forest": ["dappled light through canopy", "moss-covered ground", "ancient trees"],
    "portrait": ["catchlights in eyes", "natural skin texture", "three-quarter view"],
    "city": ["wet reflective streets", "neon signs", "urban depth of field"],
    "night": ["moonlight", "star-filled sky", "ambient city glow"],
    "rain": ["rain-soaked surfaces", "water droplets", "reflective puddles"],
    "snow": ["fresh powder", "snow-laden branches", "cold blue light"],
    "fire": ["warm flickering light", "ember glow", "dancing flames"],
    "water": ["caustic light patterns", "surface reflections", "underwater clarity"],
    "face": ["pore-level skin detail", "natural asymmetry", "expressive micro-expressions"],
    "hands": ["natural finger curl", "skin texture", "believable joint structure"],
    "hair": ["individual strand detail", "natural flow", "light catching highlights"],
    "clothing": ["fabric texture", "natural drape", "material-specific sheen"],
    "background": ["atmospheric depth", "environmental storytelling", "coherent perspective"],
}

_STYLE_ANCHORS: dict[str, list[str]] = {
    "photorealistic": [
        "shot on Canon EOS R5, 85mm f/1.4",
        "Hasselblad medium format, natural light",
        "documentary photography style, Leica M11",
        "editorial photography, professional lighting setup",
    ],
    "oil painting": [
        "in the style of classical oil painting, visible brushwork",
        "old master technique, glazing layers",
        "impressionist oil on canvas, palette knife texture",
    ],
    "watercolor": [
        "loose watercolor washes, wet-on-wet technique",
        "transparent watercolor, granulation texture",
        "plein air watercolor sketch",
    ],
    "digital art": [
        "professional digital illustration, Procreate",
        "concept art, Photoshop, industry standard",
        "digital painting, ArtStation quality",
    ],
    "anime": [
        "high-quality anime key visual",
        "anime production art, clean linework",
        "anime illustration, vibrant colors, sharp lines",
    ],
    "3d render": [
        "Octane render, subsurface scattering, global illumination",
        "Blender Cycles, physically-based materials",
        "Unreal Engine 5, photorealistic 3D",
    ],
}

_QUALITY_ANCHORS = [
    "masterpiece, best quality",
    "award-winning, professional quality",
    "museum quality, fine art",
    "highly detailed, sharp focus",
    "8k resolution, ultra-detailed",
]

_REORDER_PRIORITY_PATTERNS = [
    # Subject patterns that should come first
    r"\b(\d+)?(girl|boy|woman|man|person|character|figure|warrior|mage|knight)\b",
    r"\b(portrait|full body|close.?up|bust)\b",
    r"\b(solo|duo|group)\b",
]

_CULTURAL_CONTEXTS: dict[str, list[str]] = {
    "samurai": ["Edo period Japan", "feudal Japanese aesthetic", "bushido warrior tradition"],
    "knight": ["medieval European", "chivalric tradition", "Gothic armor style"],
    "wizard": ["arcane tradition", "mystical scholarly aesthetic", "ancient magical order"],
    "viking": ["Norse mythology", "Scandinavian warrior culture", "Age of Vikings"],
    "geisha": ["traditional Japanese culture", "Meiji era aesthetic", "classical Japanese beauty"],
    "cowboy": ["American frontier", "Wild West aesthetic", "19th century American West"],
    "pharaoh": ["ancient Egyptian", "New Kingdom period", "divine ruler aesthetic"],
    "ninja": ["feudal Japanese", "shinobi tradition", "covert warrior aesthetic"],
}


# ---------------------------------------------------------------------------
# Mutation strategies
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MutationResult:
    """A single prompt mutation with metadata."""

    prompt: str
    strategy: str
    changes: list[str] = field(default_factory=list)
    confidence: float = 1.0  # how likely this mutation is to improve results


class PromptMutationEngine:
    """
    Generates semantically-varied mutations of a prompt.

    Each mutation preserves the core intent while varying vocabulary,
    specificity, emphasis, and structure to explore the model's response space.
    """

    STRATEGIES = [
        "synonym",
        "specificity",
        "style_anchor",
        "reorder",
        "emphasis",
        "quality_anchor",
        "cultural_context",
        "technical_spec",
        "negative_inversion",
    ]

    def __init__(
        self,
        n_mutations: int = 4,
        strategies: list[str] | None = None,
        seed: int | None = None,
        max_length: int = 300,
        preserve_emphasis: bool = True,
    ):
        self.n_mutations = int(n_mutations)
        self.strategies = strategies or self.STRATEGIES
        self.seed = seed
        self.max_length = int(max_length)
        self.preserve_emphasis = preserve_emphasis

    def _get_rng(self, prompt: str) -> random.Random:
        if self.seed is not None:
            return random.Random(self.seed)
        h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return random.Random(int(h, 16))

    def mutate(self, prompt: str) -> list[MutationResult]:
        """
        Generate n_mutations varied versions of the prompt.

        Returns list of MutationResult, always including the original as first entry.
        """
        rng = self._get_rng(prompt)
        results = [MutationResult(prompt=prompt, strategy="original", confidence=1.0)]

        # Apply each strategy in rotation
        strategy_cycle = list(self.strategies) * (self.n_mutations // len(self.strategies) + 1)
        rng.shuffle(strategy_cycle)

        for strategy in strategy_cycle[: self.n_mutations]:
            try:
                mutation = self._apply_strategy(prompt, strategy, rng)
                if mutation and mutation.prompt != prompt:
                    results.append(mutation)
            except Exception:
                continue

        # Deduplicate
        seen = {prompt}
        deduped = [results[0]]
        for r in results[1:]:
            if r.prompt not in seen:
                seen.add(r.prompt)
                deduped.append(r)

        return deduped[: self.n_mutations + 1]

    def _apply_strategy(
        self,
        prompt: str,
        strategy: str,
        rng: random.Random,
    ) -> MutationResult | None:
        """Apply a single mutation strategy."""
        if strategy == "synonym":
            return self._mutate_synonym(prompt, rng)
        elif strategy == "specificity":
            return self._mutate_specificity(prompt, rng)
        elif strategy == "style_anchor":
            return self._mutate_style_anchor(prompt, rng)
        elif strategy == "reorder":
            return self._mutate_reorder(prompt, rng)
        elif strategy == "emphasis":
            return self._mutate_emphasis(prompt, rng)
        elif strategy == "quality_anchor":
            return self._mutate_quality_anchor(prompt, rng)
        elif strategy == "cultural_context":
            return self._mutate_cultural_context(prompt, rng)
        elif strategy == "technical_spec":
            return self._mutate_technical_spec(prompt, rng)
        elif strategy == "negative_inversion":
            return self._mutate_negative_inversion(prompt, rng)
        return None

    def _mutate_synonym(self, prompt: str, rng: random.Random) -> MutationResult | None:
        """Replace key words with synonyms."""
        parts = [p.strip() for p in prompt.split(",")]
        changes = []
        new_parts = []

        for part in parts:
            new_part = part
            for word, synonyms in _SYNONYM_MAP.items():
                pattern = r"\b" + re.escape(word) + r"\b"
                if re.search(pattern, part, re.IGNORECASE) and rng.random() < 0.6:
                    synonym = rng.choice(synonyms)
                    new_part = re.sub(pattern, synonym, new_part, flags=re.IGNORECASE)
                    changes.append(f"{word} → {synonym}")
                    break  # one substitution per part
            new_parts.append(new_part)

        if not changes:
            return None

        return MutationResult(
            prompt=", ".join(new_parts)[: self.max_length],
            strategy="synonym",
            changes=changes,
            confidence=0.8,
        )

    def _mutate_specificity(self, prompt: str, rng: random.Random) -> MutationResult | None:
        """Add specific details to vague elements."""
        additions = []
        prompt_lower = prompt.lower()

        for keyword, details in _SPECIFICITY_ADDITIONS.items():
            if keyword in prompt_lower and rng.random() < 0.7:
                detail = rng.choice(details)
                if detail.lower() not in prompt_lower:
                    additions.append(detail)
                if len(additions) >= 2:
                    break

        if not additions:
            return None

        new_prompt = prompt + ", " + ", ".join(additions)
        return MutationResult(
            prompt=new_prompt[: self.max_length],
            strategy="specificity",
            changes=additions,
            confidence=0.9,
        )

    def _mutate_style_anchor(self, prompt: str, rng: random.Random) -> MutationResult | None:
        """Add style-specific technical vocabulary."""
        prompt_lower = prompt.lower()

        for style_key, anchors in _STYLE_ANCHORS.items():
            if style_key in prompt_lower:
                anchor = rng.choice(anchors)
                if anchor.lower() not in prompt_lower:
                    new_prompt = f"{prompt}, {anchor}"
                    return MutationResult(
                        prompt=new_prompt[: self.max_length],
                        strategy="style_anchor",
                        changes=[anchor],
                        confidence=0.85,
                    )

        return None

    def _mutate_reorder(self, prompt: str, rng: random.Random) -> MutationResult | None:
        """Reorder elements to put the most important first."""
        parts = [p.strip() for p in prompt.split(",") if p.strip()]
        if len(parts) <= 2:
            return None

        # Find subject-like elements
        subject_indices = []
        for i, part in enumerate(parts):
            for pattern in _REORDER_PRIORITY_PATTERNS:
                if re.search(pattern, part, re.IGNORECASE):
                    subject_indices.append(i)
                    break

        if not subject_indices or subject_indices[0] == 0:
            return None

        # Move subject to front
        new_parts = []
        for i in subject_indices:
            new_parts.append(parts[i])
        for i, part in enumerate(parts):
            if i not in subject_indices:
                new_parts.append(part)

        new_prompt = ", ".join(new_parts)
        if new_prompt == prompt:
            return None

        return MutationResult(
            prompt=new_prompt[: self.max_length],
            strategy="reorder",
            changes=["moved subject to front"],
            confidence=0.75,
        )

    def _mutate_emphasis(self, prompt: str, rng: random.Random) -> MutationResult | None:
        """Add (word) emphasis to key elements."""
        if "(" in prompt and self.preserve_emphasis:
            return None  # Already has emphasis

        parts = [p.strip() for p in prompt.split(",") if p.strip()]
        if not parts:
            return None

        # Emphasize the first 1-2 important elements
        new_parts = []
        emphasized = 0
        for i, part in enumerate(parts):
            if emphasized < 2 and i < 3 and rng.random() < 0.6:
                # Check if it's a meaningful element (not just quality tags)
                quality_words = {"masterpiece", "best quality", "high quality", "detailed", "8k", "4k"}
                if not any(qw in part.lower() for qw in quality_words):
                    new_parts.append(f"({part})")
                    emphasized += 1
                    continue
            new_parts.append(part)

        if emphasized == 0:
            return None

        return MutationResult(
            prompt=", ".join(new_parts)[: self.max_length],
            strategy="emphasis",
            changes=[f"emphasized {emphasized} elements"],
            confidence=0.7,
        )

    def _mutate_quality_anchor(self, prompt: str, rng: random.Random) -> MutationResult | None:
        """Add quality anchor tags if missing."""
        prompt_lower = prompt.lower()
        quality_present = any(
            q in prompt_lower for q in ["masterpiece", "best quality", "award", "professional", "8k", "4k"]
        )

        if quality_present:
            return None

        anchor = rng.choice(_QUALITY_ANCHORS)
        new_prompt = f"{anchor}, {prompt}"

        return MutationResult(
            prompt=new_prompt[: self.max_length],
            strategy="quality_anchor",
            changes=[anchor],
            confidence=0.85,
        )

    def _mutate_cultural_context(self, prompt: str, rng: random.Random) -> MutationResult | None:
        """Add cultural/historical grounding."""
        prompt_lower = prompt.lower()

        for keyword, contexts in _CULTURAL_CONTEXTS.items():
            if keyword in prompt_lower:
                context = rng.choice(contexts)
                if context.lower() not in prompt_lower:
                    new_prompt = f"{prompt}, {context}"
                    return MutationResult(
                        prompt=new_prompt[: self.max_length],
                        strategy="cultural_context",
                        changes=[context],
                        confidence=0.8,
                    )

        return None

    def _mutate_technical_spec(self, prompt: str, rng: random.Random) -> MutationResult | None:
        """Add technical photography/art terms."""
        prompt_lower = prompt.lower()

        # Only add if not already heavily technical
        technical_count = sum(
            1 for t in ["depth of field", "bokeh", "f/", "mm", "iso", "aperture", "shutter"] if t in prompt_lower
        )
        if technical_count >= 2:
            return None

        # Detect if it's a photographic prompt
        photo_indicators = ["photo", "photograph", "realistic", "real", "camera", "shot", "dslr"]
        is_photo = any(p in prompt_lower for p in photo_indicators)

        if is_photo:
            specs = [
                "shallow depth of field, bokeh background",
                "natural light, golden hour",
                "35mm film grain, authentic texture",
                "professional color grading, cinematic",
                "sharp focus, high dynamic range",
            ]
        else:
            specs = [
                "dramatic lighting, chiaroscuro",
                "atmospheric perspective, depth",
                "color harmony, limited palette",
                "compositional balance, rule of thirds",
                "textural contrast, material authenticity",
            ]

        spec = rng.choice(specs)
        if spec.lower() not in prompt_lower:
            new_prompt = f"{prompt}, {spec}"
            return MutationResult(
                prompt=new_prompt[: self.max_length],
                strategy="technical_spec",
                changes=[spec],
                confidence=0.75,
            )

        return None

    def _mutate_negative_inversion(self, prompt: str, rng: random.Random) -> MutationResult | None:
        """
        Rephrase to avoid common failure modes.

        Instead of "a woman with long hair", use "a woman, long flowing hair, clearly defined"
        to avoid the model ignoring the hair attribute.
        """
        # Split compound descriptions into separate elements
        parts = [p.strip() for p in prompt.split(",") if p.strip()]
        new_parts = []
        changes = []

        for part in parts:
            # Split "X with Y" into "X, Y"
            with_match = re.match(r"^(.+?)\s+with\s+(.+)$", part, re.IGNORECASE)
            if with_match and rng.random() < 0.5:
                main = with_match.group(1).strip()
                attr = with_match.group(2).strip()
                new_parts.append(main)
                new_parts.append(attr)
                changes.append(f"split '{part}' into separate elements")
                continue

            # Split "X and Y" into "X, Y" for non-subject elements
            and_match = re.match(r"^(.+?)\s+and\s+(.+)$", part, re.IGNORECASE)
            if and_match and rng.random() < 0.4:
                new_parts.append(and_match.group(1).strip())
                new_parts.append(and_match.group(2).strip())
                changes.append(f"split '{part}'")
                continue

            new_parts.append(part)

        if not changes:
            return None

        return MutationResult(
            prompt=", ".join(new_parts)[: self.max_length],
            strategy="negative_inversion",
            changes=changes,
            confidence=0.7,
        )


# ---------------------------------------------------------------------------
# Full mutation + generation + pick pipeline
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MutationPipelineResult:
    """Result of the full mutation + generation + pick pipeline."""

    best_image: Any  # numpy array (H, W, 3) uint8
    best_prompt: str
    best_score: float
    original_prompt: str
    all_mutations: list[MutationResult]
    all_scores: list[float]
    pick_metric: str
    winner_index: int


class MutationGenerationPipeline:
    """
    Full pipeline: mutate prompt → generate all variants → pick best.

    This is the "prompt-level beam search" — instead of searching over
    latent trajectories, we search over prompt formulations.

    Usage:
        def my_generate(prompt: str, seed: int) -> np.ndarray:
            # Your generation function
            return image_array

        pipeline = MutationGenerationPipeline(
            engine=PromptMutationEngine(n_mutations=3),
            pick_metric="combo_vit_hq",
        )
        result = pipeline.run(
            original_prompt="a samurai at sunset",
            generate_fn=my_generate,
            seed=42,
        )
        print(f"Best prompt: {result.best_prompt}")
        print(f"Best score: {result.best_score:.3f}")
    """

    def __init__(
        self,
        engine: PromptMutationEngine | None = None,
        pick_metric: str = "combo",
        device: str = "cuda",
        clip_model_id: str = "openai/clip-vit-base-patch32",
    ):
        self.engine = engine or PromptMutationEngine(n_mutations=3)
        self.pick_metric = str(pick_metric)
        self.device = str(device)
        self.clip_model_id = str(clip_model_id)

    def run(
        self,
        original_prompt: str,
        generate_fn: Callable[[str, int], Any],
        seed: int = 42,
        negative_prompt: str = "",
        expected_text: str = "",
        expected_count: int = 0,
        vit_ckpt: str = "",
    ) -> MutationPipelineResult:
        """
        Run the full mutation + generation + pick pipeline.

        Args:
            original_prompt: The user's original prompt
            generate_fn: Function(prompt, seed) → numpy RGB image
            seed: Base seed (each mutation gets seed + mutation_index)
            negative_prompt: Negative prompt (same for all mutations)
            expected_text: Expected OCR text for text-in-image scoring
            expected_count: Expected object count for count scoring
            vit_ckpt: Optional ViT quality checkpoint path

        Returns:
            MutationPipelineResult with best image and metadata
        """
        # Generate mutations
        mutations = self.engine.mutate(original_prompt)

        # Generate one image per mutation
        images = []
        for i, mutation in enumerate(mutations):
            try:
                img = generate_fn(mutation.prompt, seed + i)
                images.append(img)
            except Exception as e:
                import sys

                print(f"Generation failed for mutation {i} ({mutation.strategy}): {e}", file=sys.stderr)
                images.append(None)

        # Filter out failed generations
        valid_pairs = [(img, mut) for img, mut in zip(images, mutations) if img is not None]
        if not valid_pairs:
            raise RuntimeError("All mutations failed to generate")

        valid_images, valid_mutations = zip(*valid_pairs)
        valid_images = list(valid_images)

        # Score all images
        try:
            from utils.quality.test_time_pick import pick_best_indices

            best_idx, scores = pick_best_indices(
                valid_images,
                original_prompt,  # Score against original intent
                self.pick_metric,
                self.device,
                expected_text,
                self.clip_model_id,
                expected_count,
                "auto",
                "",
                vit_ckpt,
                False,
                -1,
            )
        except Exception:
            # Fallback: use first image
            best_idx = 0
            scores = [1.0] + [0.0] * (len(valid_images) - 1)

        return MutationPipelineResult(
            best_image=valid_images[best_idx],
            best_prompt=valid_mutations[best_idx].prompt,
            best_score=float(scores[best_idx]) if scores else 0.0,
            original_prompt=original_prompt,
            all_mutations=list(valid_mutations),
            all_scores=[float(s) for s in scores] if scores else [],
            pick_metric=self.pick_metric,
            winner_index=int(best_idx),
        )


@dataclass
class ExploreCandidate:
    """One render-ready prompt bundle from a genome."""

    genome: StyleGenome
    positive: str
    negative: str
    style: str
    mutation_strategy: str = ""
    catalog_overlap: float = 0.0
    candidate_kind: str = "base"

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "caption": self.positive,
            "negative_caption": self.negative,
            "style": self.style,
            "style_genome_id": self.genome.id,
            "style_genome_name": self.genome.name,
            "mutation_strategy": self.mutation_strategy,
            "catalog_overlap": self.catalog_overlap,
            "candidate_kind": self.candidate_kind,
        }


@dataclass
class ExplorePlan:
    """Full explore session output."""

    base_prompt: str
    candidates: list[ExploreCandidate] = field(default_factory=list)
    reasoning: str = ""

    def write_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in self.candidates:
                f.write(json.dumps(row.to_manifest_row(), ensure_ascii=False) + "\n")

    def write_genomes_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [c.genome.to_dict() for c in self.candidates]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def compile_genome_pair(
    genome: StyleGenome,
    base_prompt: str,
    base_negative: str = "",
) -> tuple[str, str, str]:
    pos = genome.compile_positive(base_prompt)
    neg = genome.compile_negative(base_negative)
    style = genome.style_head_string()
    return pos, neg, style


def resolve_style_genome_for_args(args: Any, base_prompt: str) -> StyleGenome | None:
    """
    Load or invent a genome and attach to *args* for PromptStack / sample.py.

    Sets: ``_active_style_genome``, ``style`` (if empty), ``_style_genome_catalog_overlap``.
    """
    genome: StyleGenome | None = None
    path = str(getattr(args, "style_genome_file", "") or "").strip()
    if path:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, list):
            idx = int(getattr(args, "style_genome_index", 0) or 0)
            data = data[idx] if data else {}
        genome = StyleGenome.from_dict(data if isinstance(data, dict) else {})

    invent_n = int(getattr(args, "invent_styles", 0) or 0)
    if genome is None and invent_n > 0:
        inventor = StyleInventor(
            device=str(getattr(args, "style_inventor_device", "cpu") or "cpu"),
            use_qwen=not bool(getattr(args, "no_style_inventor_qwen", False)),
        )
        mode = str(getattr(args, "style_inventor_mode", "normal") or "normal").strip().lower()
        genomes = inventor.invent(
            base_prompt,
            n=invent_n,
            creativity_level=float(getattr(args, "style_inventor_creativity", 0.75) or 0.75),
            seed=int(getattr(args, "seed", 42) or 42),
            invention_mode=mode,  # type: ignore[arg-type]
            chaos_level=float(getattr(args, "style_chaos_level", 0.0) or 0.0),
        )
        preset = str(getattr(args, "style_genome_preset", "") or "").strip()
        if preset:
            pg = preset_genome(preset)
            if pg is not None:
                genomes = [pg] + [g for g in genomes if g.id != pg.id]
        if bool(getattr(args, "style_genome_hypermutate", False)) and genomes:
            genomes = [
                hypermutate(
                    g,
                    intensity=float(getattr(args, "style_chaos_level", 0.85) or 0.85),
                    seed=int(getattr(args, "seed", 42)),
                )
                for g in genomes
            ]
        clauses = str(getattr(args, "prompt_clauses", "") or "")
        auto = auto_chaos_clauses(float(getattr(args, "style_chaos_level", 0) or 0))
        if auto and not clauses:
            args.prompt_clauses = ",".join(auto)
        elif auto:
            args.prompt_clauses = f"{clauses},{','.join(auto)}"
        idx = int(getattr(args, "style_genome_index", 0) or 0)
        if genomes:
            genome = genomes[min(idx, len(genomes) - 1)]
            args._invented_style_genomes = genomes

    if genome is None:
        return None

    args._active_style_genome = genome
    _, overlap = nearest_catalog_style_overlap(genome)
    args._style_genome_catalog_overlap = overlap

    if not (getattr(args, "style", None) or "").strip():
        args.style = genome.style_head_string()

    return genome


class StyleExplorePlanner:
    """Build a manifest of genome × optional prompt mutations."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        use_qwen: bool = True,
    ) -> None:
        self.inventor = StyleInventor(device=device, use_qwen=use_qwen)

    def plan(
        self,
        base_prompt: str,
        *,
        n_genomes: int = 3,
        mutations_per_genome: int = 0,
        base_negative: str = "",
        creativity_level: float = 0.75,
        seed: int = 42,
        mutation_strategies: Sequence[str] | None = None,
        invention_mode: InventionMode = "normal",
        chaos_level: float = 0.0,
        fusion_pairs: bool = False,
        hypermutate_siblings: int = 0,
        preset: str = "",
    ) -> ExplorePlan:
        base_prompt = (base_prompt or "").strip()
        genomes = self.inventor.invent(
            base_prompt,
            n=n_genomes,
            creativity_level=creativity_level,
            seed=seed,
            invention_mode=invention_mode,
            chaos_level=chaos_level,
        )
        if preset:
            pg = preset_genome(preset)
            if pg is not None:
                genomes.insert(0, pg)
        candidates: list[ExploreCandidate] = []
        mutator = None
        if mutations_per_genome > 0:
            try:
                mutator = PromptMutationEngine(
                    n_mutations=mutations_per_genome,
                    strategies=list(mutation_strategies) if mutation_strategies else None,
                    seed=seed,
                )
            except ImportError:
                mutator = None

        for genome in genomes:
            pos, neg, style = compile_genome_pair(genome, base_prompt, base_negative)
            _, overlap = nearest_catalog_style_overlap(genome)
            candidates.append(
                ExploreCandidate(
                    genome=genome,
                    positive=pos,
                    negative=neg,
                    style=style,
                    catalog_overlap=overlap,
                    candidate_kind="base",
                )
            )
            if fusion_pairs and len(genomes) >= 2:
                import random

                r = random.Random(seed)
                others = [g for g in genomes if g.id != genome.id and g.name != genome.name]
                if len(others) >= 1:
                    partner = r.choice(others)
                    chimera = fuse_genomes(genome, partner, ratio=r.random())
                    cpos, cneg, cstyle = compile_genome_pair(chimera, base_prompt, base_negative)
                    _, cov = nearest_catalog_style_overlap(chimera)
                    candidates.append(
                        ExploreCandidate(
                            genome=chimera,
                            positive=cpos,
                            negative=cneg,
                            style=cstyle,
                            catalog_overlap=cov,
                            candidate_kind="chimera",
                        )
                    )
            if hypermutate_siblings > 0:
                for j in range(hypermutate_siblings):
                    sib = hypermutate(genome, intensity=max(0.5, chaos_level, 0.75), seed=seed + j + 3)
                    spos, sneg, sstyle = compile_genome_pair(sib, base_prompt, base_negative)
                    _, sov = nearest_catalog_style_overlap(sib)
                    candidates.append(
                        ExploreCandidate(
                            genome=sib,
                            positive=spos,
                            negative=sneg,
                            style=sstyle,
                            catalog_overlap=sov,
                            candidate_kind="hypermutate",
                        )
                    )
            if mutator is not None:
                for mut in mutator.mutate(pos)[:mutations_per_genome]:
                    mpos, mneg, mstyle = compile_genome_pair(genome, mut.prompt, neg)
                    candidates.append(
                        ExploreCandidate(
                            genome=genome,
                            positive=mpos,
                            negative=mneg,
                            style=mstyle,
                            mutation_strategy=mut.strategy,
                            catalog_overlap=overlap,
                        )
                    )

        return ExplorePlan(
            base_prompt=base_prompt,
            candidates=candidates,
            reasoning=f"Invented {len(genomes)} genomes; {len(candidates)} total candidates.",
        )
