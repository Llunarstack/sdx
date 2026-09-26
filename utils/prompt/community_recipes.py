"""Community prompt recipes (PixAI + Civitai) for prompt research.

Live sites share working prompts. We store the *text* (never the image), classify
simple / intermediate / hard the way those UIs actually split work, then copy only
**structural** tokens into the user prompt — not their character, LoRA, or face.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from utils.modeling.model_paths import repo_root

Lane = Literal["simple", "intermediate", "hard"]

# Public PixAI docs/blog/API examples (https://docs.pixai.art , https://blog.pixai.art).
# These are the site's own teaching prompts, not scraped private gens.
PIXAI_PUBLIC_RECIPES: tuple[dict[str, str], ...] = (
    {
        "source": "pixai",
        "source_id": "docs-basics-nl",
        "prompt": (
            "pixai_mio wearing a flowing white summer dress, turning around as the breeze "
            "catches her skirt and makes it billow outward. She stands in a sunlit sunflower "
            "field with golden petals stretching endlessly toward the horizon, afternoon light "
            "warming the scene."
        ),
        "negative": "",
        "note": "PixAI basic formula: subject + action + scene (natural language).",
    },
    {
        "source": "pixai",
        "source_id": "docs-api-tags",
        "prompt": "1girl, green hair, solo, standing",
        "negative": "",
        "note": "PixAI API v2 sample: short tag prompt.",
    },
    {
        "source": "pixai",
        "source_id": "docs-api-quality",
        "prompt": "1girl, green hair, masterpiece, best quality, solo, standing, flower",
        "negative": "",
        "note": "PixAI first-api-call example with quality last.",
    },
    {
        "source": "pixai",
        "source_id": "blog-six-slot",
        "prompt": (
            "1girl, black hair, blue eyes, looking at viewer, slight smile, standing, "
            "outdoors, ginkgo leaves, autumn, anime style, soft illustration, soft lighting, "
            "cinematic light, depth of field, high detail, masterpiece, best quality"
        ),
        "negative": (
            "lowres, bad anatomy, bad hands, missing fingers, extra digits, cropped, "
            "worst quality, low quality, jpeg artifacts, watermark, signature"
        ),
        "note": "PixAI 6-slot formula (subject, action, place, style, lighting, quality).",
    },
    {
        "source": "pixai",
        "source_id": "blog-sdxl-order",
        "prompt": ("1boy, black hair, short hair, blue eyes, upper body, anime style, masterpiece, best quality"),
        "negative": "lowres, bad anatomy, bad hands, text, error, missing fingers, blurry",
        "note": "PixAI SDXL order: core subject, details, camera, style, quality.",
    },
    {
        "source": "pixai",
        "source_id": "docs-layered-nl",
        "prompt": (
            "pixai_mio wearing a flowing white summer dress, turning around as the breeze "
            "catches her skirt. She stands in a sunlit sunflower field. The scene glows with "
            "golden hour warmth, soft light filtering through the flowers. Rendered in a "
            "painterly anime style with soft pastel tones and cinematic shallow depth of field."
        ),
        "negative": "",
        "note": "PixAI advanced NL: subject + action + scene + mood + style.",
    },
    {
        "source": "pixai",
        "source_id": "docs-hard-two-character",
        "prompt": (
            "soft light, masterpiece. A full illustration, cowboy shot. Character 1: short white "
            "hair with pink tones, cat ears, cat tail, sleeveless black shirt with sailor collar, "
            "upper body, light smile, from side, closed eyes. BREAK Character 2: much smaller "
            "full-body girl in the lower right, short bob, cat ears, facing away from the viewer, "
            "one arm raised. Two distinct people, no identity merge, pastoral flowers, soft bokeh."
        ),
        "negative": "bad anatomy, fused bodies, extra limbs, extra heads, watermark",
        "note": "PixAI hard: two characters, shot lock, BREAK, details last.",
    },
)

# Extra public teaching prompts: editing (PixAI Reference Pro), POV/camera, SFW vs adult NSFW.
# These are structure templates, not identity copies.
EXTRA_PUBLIC_RECIPES: tuple[dict[str, str], ...] = (
    {
        "source": "pixai",
        "source_id": "refpro-sleeping",
        "prompt": "Change it to a sleeping pose. Keep identity, keep outfit, keep composition, same character.",
        "negative": "different person, extra faces, identity swap",
        "note": "kind=edit | PixAI Reference Pro: one-sentence pose edit.",
        "lane": "simple",
    },
    {
        "source": "pixai",
        "source_id": "refpro-pose-from-image",
        "prompt": (
            "Recreate the exact pose from image 2, but use my character from image 1. "
            "Keep identity, keep lighting, keep proportions, change pose, inpaint."
        ),
        "negative": "face from pose photo, merged identity, extra limbs",
        "note": "kind=edit,pov | PixAI Reference Pro: pose from one image, identity from another.",
        "lane": "hard",
    },
    {
        "source": "pixai",
        "source_id": "refpro-outfit-swap",
        "prompt": "Dress them in the outfit from image 2. Keep identity, keep face, keep hairstyle, keep composition.",
        "negative": "wrong face, extra outfits, tangled clothes",
        "note": "kind=edit | PixAI Reference Pro: wardrobe from a reference.",
        "lane": "intermediate",
    },
    {
        "source": "sdx",
        "source_id": "simple-sfw-solo",
        "prompt": "1girl, standing, looking at viewer, simple background, fully clothed",
        "negative": "nsfw, nude, extra fingers, watermark",
        "note": "kind=sfw | Simple SFW tag prompt.",
        "lane": "simple",
    },
    {
        "source": "sdx",
        "source_id": "simple-sfw-nl",
        "prompt": "A man sitting on a bench in a sunny park, looking at viewer, photorealistic",
        "negative": "nsfw, extra limbs, watermark",
        "note": "kind=sfw | Simple SFW natural language.",
        "lane": "simple",
    },
    {
        "source": "sdx",
        "source_id": "simple-nsfw-adult",
        "prompt": "1girl, nsfw, uncensored, standing, looking at viewer, simple background",
        "negative": "child, loli, extra fingers, mosaic censor",
        "note": "kind=nsfw | Simple adult NSFW tags. Not a person copy.",
        "lane": "simple",
    },
    {
        "source": "sdx",
        "source_id": "pov-first-person",
        "prompt": "pov, first-person view, looking at viewer, cowboy shot, arms in frame",
        "negative": "third-person full body of the viewer, extra heads, extra arms",
        "note": "kind=pov,camera | First-person camera lock.",
        "lane": "intermediate",
    },
    {
        "source": "sdx",
        "source_id": "pov-from-below",
        "prompt": "from below, low angle, looking down at viewer, dutch angle, foreshortening",
        "negative": "eye-level default, extra legs, broken perspective",
        "note": "kind=pov,camera | Low viewpoint / from below.",
        "lane": "intermediate",
    },
    {
        "source": "sdx",
        "source_id": "pov-over-shoulder",
        "prompt": "over the shoulder, from behind, three-quarter view, looking at viewer",
        "negative": "two faces of the same person, extra heads",
        "note": "kind=pov,camera | Over-the-shoulder staging.",
        "lane": "intermediate",
    },
    {
        "source": "sdx",
        "source_id": "hard-pov-two-adults",
        "prompt": (
            "1girl, 1boy, nsfw, pov, first-person view, looking at viewer, cowboy shot, "
            "two distinct people, no identity merge, uncensored"
        ),
        "negative": "child, extra arms, extra genitals, fused bodies, viewer face in frame",
        "note": "kind=nsfw,pov | Adult two-person POV. Face of POV character stays out of frame.",
        "lane": "hard",
    },
    {
        "source": "sdx",
        "source_id": "hard-fisheye",
        "prompt": "fisheye, worm's eye view, extreme perspective, full body, dynamic pose, foreshortening",
        "negative": "flat perspective, extra limbs, broken anatomy",
        "note": "kind=camera | Extreme perspective.",
        "lane": "hard",
    },
    {
        "source": "sdx",
        "source_id": "sfw-portrait",
        "prompt": "close-up, upper body, looking at viewer, soft lighting, fully clothed, rating_safe",
        "negative": "nsfw, nude, extra fingers",
        "note": "kind=sfw,camera | SFW portrait crop.",
        "lane": "simple",
    },
)

# Generic structure only — never copy names, LoRAs, or clothing story from a neighbor.
_STRUCTURAL = frozenset(
    {
        "looking at viewer",
        "slight smile",
        "standing",
        "sitting",
        "walking",
        "solo",
        "cowboy shot",
        "upper body",
        "full body",
        "close-up",
        "closeup",
        "from below",
        "from above",
        "from side",
        "outdoors",
        "indoors",
        "simple background",
        "white background",
        "depth of field",
        "bokeh",
        "soft lighting",
        "cinematic light",
        "cinematic lighting",
        "rim light",
        "golden hour",
        "anime style",
        "illustration",
        "photorealistic",
        "masterpiece",
        "best quality",
        "high detail",
        "highly detailed",
        "soft bokeh",
        "blurry background",
        "two distinct people",
        "no identity merge",
        "score_9",
        "score_8_up",
        "score_7_up",
        "source_anime",
        "rating_safe",
        "rating_explicit",
        "pov",
        "first-person view",
        "first person",
        "from behind",
        "low angle",
        "high angle",
        "over the shoulder",
        "dutch angle",
        "three-quarter view",
        "worm's eye view",
        "bird's eye view",
        "fisheye",
        "foreshortening",
        "arms in frame",
        "looking down at viewer",
        "keep identity",
        "keep same face",
        "keep face",
        "keep hairstyle",
        "same character",
        "keep composition",
        "keep lighting",
        "keep proportions",
        "keep outfit",
        "change pose",
        "inpaint",
        "uncensored",
        "fully clothed",
        "nsfw",
    }
)

_MINOR_RE = re.compile(
    r"\b(child|children|kid|kids|toddler|infant|baby|minor|underage|"
    r"loli|shota|preteen|pre-teen)\b",
    re.I,
)
_LORA_RE = re.compile(r"<lora:[^>]+>", re.I)
_EMB_RE = re.compile(r"embedding:[^\s,]+", re.I)
_WEIGHT_RE = re.compile(r"\([^):]+:\s*[\d.]+\s*\)")
_SCORE_RE = re.compile(r"\bscore_\d", re.I)
_COMMA_SPLIT = re.compile(r"[,;\n]+")
_NSFW_USER_RE = re.compile(
    r"\b(nsfw|nude|naked|uncensored|explicit|sex|pov sex|cowgirl|hentai|"
    r"penis|cock|pussy|handjob|blowjob|paizuri|creampie|ahegao|"
    r"tentacle|milking|sexbot|sex\s*bot|fuck(?:ing)?|lewd|ecchi)\b",
    re.I,
)
_POV_RE = re.compile(
    r"\b(pov|first-person|first person|from below|from behind|from above|over the shoulder|"
    r"dutch angle|worm'?s eye|bird'?s eye|fisheye|foreshortening|low angle|high angle)\b",
    re.I,
)
_EDIT_RE = re.compile(
    r"\b(inpaint|img2img|keep identity|keep (the )?face|change pose|change the|"
    r"replace the|edit the|recreate the|outfit from image|pose from image)\b",
    re.I,
)


@dataclass(slots=True)
class CommunityRecipe:
    source: str
    source_id: str
    prompt: str
    negative: str = ""
    lane: Lane = "simple"
    note: str = ""
    kinds: tuple[str, ...] = ()

    def fact_line(self) -> str:
        kinds = ",".join(self.kinds)
        bits = [f"lane={self.lane}", f"source={self.source}"]
        if kinds:
            bits.append(f"kind={kinds}")
        bits.append(self.prompt.strip())
        if self.negative.strip():
            bits.append(f"negative: {self.negative.strip()[:240]}")
        if self.note.strip():
            bits.append(self.note.strip())
        return " | ".join(bits)


@dataclass(slots=True)
class CommunityResearchResult:
    prompt: str
    negative: str = ""
    lane: Lane = "simple"
    added: list[str] = field(default_factory=list)
    neighbors: list[CommunityRecipe] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    kinds: tuple[str, ...] = ()


def bundled_corpus_path() -> Path:
    return repo_root() / "config" / "defaults" / "community_prompt_recipes.jsonl"


def detect_kinds(prompt: str, *, is_edit: bool = False) -> tuple[str, ...]:
    p = prompt or ""
    kinds: list[str] = []
    if _POV_RE.search(p):
        kinds.append("pov")
        kinds.append("camera")
    if is_edit or _EDIT_RE.search(p):
        kinds.append("edit")
    if _NSFW_USER_RE.search(p):
        kinds.append("nsfw")
    else:
        kinds.append("sfw")
    if any(k in p.lower() for k in ("cowboy shot", "close-up", "full body", "upper body", "perspective")):
        if "camera" not in kinds:
            kinds.append("camera")
    return tuple(kinds)


def is_blocked_prompt(text: str) -> bool:
    return bool(_MINOR_RE.search(text or ""))


def classify_lane(prompt: str) -> Lane:
    """Match PixAI (slot count) + Civitai (LoRA / BREAK / regional / Pony scores)."""
    raw = (prompt or "").strip()
    if not raw:
        return "simple"
    tags = [t.strip() for t in _COMMA_SPLIT.split(raw) if t.strip()]
    n = len(tags)
    n_char = len(raw)
    n_lora = len(_LORA_RE.findall(raw))
    low = raw.lower()
    two_bodies = bool(re.search(r"\b(2girls|2boys|couple|1girl.{0,48}1boy|1boy.{0,48}1girl)\b", raw, re.I))
    hard = (
        n_lora >= 3
        or n_char >= 700
        or n >= 45
        or bool(re.search(r"\bBREAK\b|\b\[SEP\]\b", raw))
        or bool(re.search(r"(?:^|[,;]\s+)AND(?:\s*[,;]|$)", raw))
        or ("character 1" in low and "character 2" in low)
        or (n_lora >= 1 and two_bodies)
        or (two_bodies and bool(_POV_RE.search(raw)))
        or bool(_EDIT_RE.search(raw))
        or bool(re.search(r"\b(fisheye|worm'?s eye|extreme perspective)\b", raw, re.I))
    )
    if hard:
        return "hard"
    mid = (
        n_lora >= 1
        or bool(_WEIGHT_RE.search(raw))
        or bool(_SCORE_RE.search(raw))
        or n >= 12
        or n_char >= 220
        or bool(_POV_RE.search(raw))
        or any(k in low for k in ("masterpiece", "best quality", "cowboy shot", "depth of field"))
    )
    return "intermediate" if mid else "simple"


def recipe_from_mapping(row: dict[str, str]) -> CommunityRecipe | None:
    prompt = str(row.get("prompt") or "").strip()
    if not prompt or is_blocked_prompt(prompt):
        return None
    neg = str(row.get("negative") or row.get("negative_prompt") or "").strip()
    if is_blocked_prompt(neg):
        neg = ""
    lane = str(row.get("lane") or "").strip().lower()
    if lane not in ("simple", "intermediate", "hard"):
        lane = classify_lane(prompt)
    note = str(row.get("note") or "").strip()
    raw_kinds = str(row.get("kinds") or "").strip()
    if not raw_kinds and "kind=" in note.lower():
        # note like "kind=edit,pov | ..."
        m = re.search(r"kind=([a-z0-9,]+)", note, re.I)
        raw_kinds = m.group(1) if m else ""
    kinds = tuple(k.strip() for k in raw_kinds.split(",") if k.strip()) or detect_kinds(prompt)
    return CommunityRecipe(
        source=str(row.get("source") or "unknown").strip() or "unknown",
        source_id=str(row.get("source_id") or row.get("id") or "").strip(),
        prompt=prompt[:2500],
        negative=neg[:800],
        lane=lane,  # type: ignore[arg-type]
        note=note,
        kinds=kinds,
    )


def pixai_seed_recipes() -> list[CommunityRecipe]:
    """PixAI docs/blog examples only (used by tests)."""
    out: list[CommunityRecipe] = []
    for row in PIXAI_PUBLIC_RECIPES:
        rec = recipe_from_mapping(dict(row))
        if rec is not None:
            out.append(rec)
    return out


def all_seed_recipes() -> list[CommunityRecipe]:
    """PixAI + POV/edit/SFW/NSFW teaching templates."""
    out = pixai_seed_recipes()
    seen = {(r.source, r.source_id) for r in out}
    for row in EXTRA_PUBLIC_RECIPES:
        rec = recipe_from_mapping(dict(row))
        if rec is None:
            continue
        key = (rec.source, rec.source_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def load_recipes(path: str | Path | None = None) -> list[CommunityRecipe]:
    recipes = all_seed_recipes()
    seen = {(r.source, r.source_id, r.prompt[:80].lower()) for r in recipes}
    p = Path(path) if path else bundled_corpus_path()
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            rec = recipe_from_mapping(row)
            if rec is None:
                continue
            key = (rec.source, rec.source_id, rec.prompt[:80].lower())
            if key in seen:
                continue
            seen.add(key)
            recipes.append(rec)
    return recipes


def retrieve_recipes(
    query: str,
    recipes: Sequence[CommunityRecipe],
    *,
    lane: Lane | None = None,
    kinds: Sequence[str] | None = None,
    top_k: int = 4,
) -> list[CommunityRecipe]:
    from utils.superior.retrieval import TfidfFactIndex

    pool = list(recipes)
    if lane is not None:
        lane_pool = [r for r in pool if r.lane == lane]
        if lane_pool:
            pool = lane_pool
    want = {k for k in (kinds or []) if k and k != "sfw"}
    if want:
        kind_pool = [r for r in pool if want.intersection(r.kinds)]
        if kind_pool:
            pool = kind_pool
    if not pool:
        pool = list(recipes)
    if not pool or top_k <= 0:
        return []
    index = TfidfFactIndex(facts=[r.fact_line() for r in pool])
    hits = index.query(query, top_k=min(top_k, len(pool)))
    by_line = {r.fact_line(): r for r in pool}
    out: list[CommunityRecipe] = []
    for line in hits:
        rec = by_line.get(line)
        if rec is not None:
            out.append(rec)
    return out[:top_k]


def _clauses(prompt: str) -> list[str]:
    return [c.strip() for c in _COMMA_SPLIT.split(prompt or "") if c.strip()]


def _norm(text: str) -> str:
    t = _LORA_RE.sub("", text)
    t = _EMB_RE.sub("", t)
    return " ".join(t.lower().split())


def structural_additions(
    user_prompt: str,
    neighbors: Sequence[CommunityRecipe],
    *,
    lane: Lane,
    kinds: Sequence[str] = (),
    is_edit: bool = False,
) -> list[str]:
    """Tokens we are allowed to borrow from community prompts."""
    blob = _norm(user_prompt)
    nsfw = "nsfw" in kinds or bool(_NSFW_USER_RE.search(user_prompt))
    found: list[str] = []
    for rec in neighbors:
        for clause in _clauses(rec.prompt):
            key = _norm(clause)
            if key not in _STRUCTURAL or key in blob or key in {_norm(x) for x in found}:
                continue
            if not nsfw and key in ("nsfw", "uncensored", "rating_explicit"):
                continue
            if nsfw and key in ("fully clothed", "rating_safe"):
                continue
            found.append(clause.strip())
    quality_last = (
        "masterpiece",
        "best quality",
        "high detail",
        "highly detailed",
        "score_9",
        "score_8_up",
        "score_7_up",
    )
    rest = [x for x in found if _norm(x) not in quality_last]
    quality = [x for x in found if _norm(x) in quality_last]
    if lane == "simple":
        rest = [x for x in rest if _norm(x) not in ("masterpiece", "best quality")]
        quality = []
        rest = rest[:4]
    elif lane == "intermediate":
        rest = rest[:8]
        quality = quality[:2]
    else:
        rest = rest[:10]
        quality = quality[:3]
        if "two distinct people" not in blob and any(
            k in blob for k in ("1girl", "1boy", "couple", "2girls", "catgirl", "catboy")
        ):
            n_subjects = sum(
                1 for k in ("1girl", "1boy", "2girls", "2boys", "catgirl", "catboy", "couple") if k in blob
            )
            if n_subjects >= 2 and "no identity merge" not in blob:
                rest.append("two distinct people")
                rest.append("no identity merge")
    forced: list[str] = []
    if "pov" in kinds:
        for tok in ("pov", "first-person view"):
            if tok not in blob and tok not in {_norm(x) for x in rest}:
                forced.append(tok)
    if is_edit or "edit" in kinds:
        for tok in ("keep identity", "keep composition"):
            if tok not in blob and tok not in {_norm(x) for x in rest}:
                forced.append(tok)
    rest = forced + rest
    pony = bool(_SCORE_RE.search(user_prompt)) or any(_SCORE_RE.search(n.prompt) for n in neighbors)
    if pony and lane != "simple":
        for tag in ("score_9", "score_8_up"):
            if tag not in blob and tag not in {_norm(x) for x in quality}:
                quality.insert(0, tag)
                break
    return rest + quality


def neighbor_negative_bits(
    user_negative: str,
    neighbors: Sequence[CommunityRecipe],
    *,
    lane: Lane,
    nsfw: bool = False,
) -> str:
    if lane == "simple":
        return ""
    have = _norm(user_negative)
    bits: list[str] = []
    for rec in neighbors:
        for clause in _clauses(rec.negative):
            key = _norm(clause)
            if not key or key in have:
                continue
            if nsfw and any(w in key for w in ("nsfw", "nude", "naked", "explicit", "uncensored", "lewd")):
                continue
            if any(
                w in key
                for w in (
                    "bad anatomy",
                    "bad hands",
                    "missing fingers",
                    "extra digits",
                    "watermark",
                    "signature",
                    "worst quality",
                    "lowres",
                    "fused",
                    "extra limbs",
                )
            ):
                bits.append(clause.strip())
            if len(bits) >= (6 if lane == "hard" else 4):
                return ", ".join(bits)
    return ", ".join(bits)


def apply_community_research(
    prompt: str,
    *,
    negative: str = "",
    corpus_path: str | Path | None = None,
    recipes: Sequence[CommunityRecipe] | None = None,
    top_k: int = 4,
    is_edit: bool = False,
) -> CommunityResearchResult:
    user = (prompt or "").strip()
    if not user:
        return CommunityResearchResult(prompt="")
    lane = classify_lane(user)
    kinds = detect_kinds(user, is_edit=is_edit)
    pool = list(recipes) if recipes is not None else load_recipes(corpus_path)
    neighbors = retrieve_recipes(user, pool, lane=lane, top_k=max(2, top_k // 2))
    extra = retrieve_recipes(user, pool, lane=None, kinds=kinds, top_k=top_k)
    seen: set[str] = set()
    merged: list[CommunityRecipe] = []
    for rec in neighbors + extra:
        key = rec.source_id or rec.prompt[:80]
        if key in seen:
            continue
        seen.add(key)
        merged.append(rec)
        if len(merged) >= top_k + 2:
            break
    if not merged:
        merged = retrieve_recipes(user, pool, lane=None, top_k=top_k)
    added = structural_additions(user, merged, lane=lane, kinds=kinds, is_edit=is_edit)
    out = user
    if added:
        from utils.prompt.fast_paths import append_unique

        out = append_unique(out, added)
    extra_neg = neighbor_negative_bits(negative, merged, lane=lane, nsfw="nsfw" in kinds)
    merged_neg = negative
    if extra_neg:
        from utils.prompt.fast_paths import merge_fragments

        merged_neg = merge_fragments(negative, extra_neg)
    sources = [f"{n.source}:{n.source_id or n.lane}" for n in merged]
    sources.insert(0, f"lane:{lane}")
    sources.append("kinds:" + ",".join(kinds))
    return CommunityResearchResult(
        prompt=out,
        negative=merged_neg,
        lane=lane,
        added=added,
        neighbors=list(merged),
        sources=sources,
        kinds=kinds,
    )


def recipes_to_jsonl(recipes: Sequence[CommunityRecipe], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in recipes:
            if rec is None or not rec.prompt.strip():
                continue
            f.write(
                json.dumps(
                    {
                        "source": rec.source,
                        "source_id": rec.source_id,
                        "lane": rec.lane,
                        "kinds": ",".join(rec.kinds),
                        "prompt": rec.prompt,
                        "negative": rec.negative,
                        "note": rec.note,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


__all__ = [
    "CommunityRecipe",
    "CommunityResearchResult",
    "PIXAI_PUBLIC_RECIPES",
    "EXTRA_PUBLIC_RECIPES",
    "all_seed_recipes",
    "apply_community_research",
    "bundled_corpus_path",
    "classify_lane",
    "detect_kinds",
    "is_blocked_prompt",
    "load_recipes",
    "pixai_seed_recipes",
    "recipe_from_mapping",
    "recipes_to_jsonl",
    "retrieve_recipes",
]
