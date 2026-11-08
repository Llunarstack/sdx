"""Prompt grounding: RAG facts, the creative RAG engine and the advanced prompt analyzer / optimizer."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.modeling.model_paths import default_moondream2_path, default_qwen_path


def merge_facts_into_prompt(
    user_prompt: str,
    facts: Sequence[str],
    *,
    prefix: str = "Context (user-supplied facts, treat as authoritative):\n",
    sep: str = "\n- ",
    suffix: str = "\n\nTask:\n",
    max_chars: int = 6000,
) -> str:
    """
    Prepend structured facts so the diffusion prompt conditions on explicit grounding.

    Truncates joined facts to ``max_chars`` (keeps the start). Empty ``facts`` returns
    ``user_prompt`` unchanged.
    """
    user_prompt = (user_prompt or "").strip()
    lines = [str(f).strip() for f in facts if f is not None and str(f).strip()]
    if not lines:
        return user_prompt
    block = prefix + sep + sep.join(lines)
    if len(block) > max_chars:
        block = block[: max_chars - 3] + "..."
    return (block + suffix + user_prompt).strip()


def load_facts_from_jsonl(
    path: str | Path,
    *,
    text_key: str = "text",
    max_entries: int = 32,
) -> list[str]:
    """
    Load up to ``max_entries`` string fields from a JSONL file (one JSON object per line).

    Each line should be like ``{\"text\": \"...\"}`` or use ``text_key`` for the fact body.
    Lines without ``text_key`` are skipped.
    """
    p = Path(path)
    if not p.is_file():
        return []
    out: list[str] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get(text_key)
            if t is None:
                continue
            out.append(str(t).strip())
            if len(out) >= max_entries:
                break
    return out


def _append_if_text(out: list[str], seen: set, value: Any, *, max_item_chars: int = 280) -> None:
    s = str(value or "").strip()
    if not s:
        return
    if len(s) > max_item_chars:
        s = s[: max_item_chars - 3].rstrip() + "..."
    k = s.lower()
    if k in seen:
        return
    seen.add(k)
    out.append(s)


def _extract_text_lines(obj: Any, out: list[str], seen: set, *, max_item_chars: int = 280) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        _append_if_text(out, seen, obj, max_item_chars=max_item_chars)
        return
    if isinstance(obj, dict):
        # Preferred keys commonly seen in search outputs.
        for k in ("text", "snippet", "summary", "title", "caption", "description", "content", "query"):
            if k in obj:
                _append_if_text(out, seen, obj.get(k), max_item_chars=max_item_chars)
        # Also recursively walk nested values.
        for v in obj.values():
            if isinstance(v, (dict, list, tuple, str)):
                _extract_text_lines(v, out, seen, max_item_chars=max_item_chars)
        return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            _extract_text_lines(x, out, seen, max_item_chars=max_item_chars)


def facts_from_gen_searcher_payload(
    payload: Any,
    *,
    max_entries: int = 24,
    max_item_chars: int = 280,
) -> list[str]:
    """
    Extract text facts from Gen-Searcher-like outputs (JSON object/list).

    Works with common fields from agentic-search style outputs, while remaining
    robust to schema drift by recursively scanning nested text-bearing fields.
    """
    out: list[str] = []
    seen: set = set()
    if payload is None:
        return out

    if isinstance(payload, dict):
        preferred_keys = (
            "reasoning_summary",
            "search_summary",
            "final_answer",
            "retrieved_facts",
            "evidence",
            "web_evidence",
            "search_results",
            "image_results",
            "references",
            "sources",
            "results",
        )
        for k in preferred_keys:
            if k in payload:
                _extract_text_lines(payload.get(k), out, seen, max_item_chars=max_item_chars)
                if len(out) >= max_entries:
                    return out[:max_entries]
    _extract_text_lines(payload, out, seen, max_item_chars=max_item_chars)
    return out[:max_entries]


def load_facts_from_gen_searcher_json(
    path: str | Path,
    *,
    max_entries: int = 24,
    max_item_chars: int = 280,
) -> list[str]:
    """
    Load Gen-Searcher-style facts from either .json or .jsonl file.

    - JSON: object or list
    - JSONL: one object per line (facts extracted from each row)
    """
    p = Path(path)
    if not p.is_file():
        return []
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        out: list[str] = []
        seen: set = set()
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for fact in facts_from_gen_searcher_payload(
                    row, max_entries=max_entries, max_item_chars=max_item_chars
                ):
                    k = fact.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(fact)
                    if len(out) >= max_entries:
                        return out
        return out

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return facts_from_gen_searcher_payload(payload, max_entries=max_entries, max_item_chars=max_item_chars)


def retrieve_facts_for_query_local(
    query: str,
    corpus_path: str | Path,
    *,
    top_k: int = 8,
) -> list[str]:
    """TF-IDF retrieval from a JSONL corpus (see ``utils.agentic.quality_loop``)."""
    from utils.agentic.quality_loop import build_tfidf_index_from_jsonl

    return build_tfidf_index_from_jsonl(corpus_path).query(query, top_k=top_k)


_MOONDREAM_PATH = Path(default_moondream2_path())
_QWEN_PATH = Path(default_qwen_path())

# Aligned with multi-reference ingestion (API-style caps); dissect/RAG facts use full list.
MAX_REFERENCE_IMAGES_RAG = 16
# Moondream runs per-image; keep a lower cap so enrichment stays tractable on one GPU.
MAX_MOONDREAM_REFERENCE_IMAGES = 8


# Result dataclass
@dataclass(slots=True)
class RAGEnrichmentResult:
    """Output of CreativeRAGEngine.enrich()."""

    original_prompt: str
    enriched_prompt: str
    negative_additions: str
    reasoning: str
    image_description: str = ""
    retrieved_facts: list[str] = field(default_factory=list)
    concept_layers: dict[str, str] = field(default_factory=dict)
    novelty_score: float = 0.0  # 0–1: how much was added vs original
    fallback_used: bool = False  # True when heavy models were unavailable


# Lightweight semantic decomposition (no heavy models required)

# Intent keywords → semantic layer
_LAYER_PATTERNS: dict[str, list[str]] = {
    "subject": [
        r"\b(\d+)?(girl|boy|woman|man|person|character|figure|warrior|samurai|knight|"
        r"wizard|mage|elf|dragon|creature|robot|android|alien|cat|dog|animal)\b",
        r"\b(portrait|bust|full body|close.?up)\b",
    ],
    "mood": [
        r"\b(melancholy|joyful|serene|tense|mysterious|ethereal|dark|bright|hopeful|"
        r"lonely|epic|intimate|nostalgic|surreal|dreamlike|ominous|peaceful|chaotic)\b",
        r"\b(sunset|dawn|dusk|midnight|golden hour|blue hour|storm|fog|mist|rain)\b",
    ],
    "style": [
        r"\b(anime|manga|realistic|photorealistic|oil painting|watercolor|sketch|"
        r"concept art|digital art|illustration|3d render|pixel art|impressionist|"
        r"baroque|art nouveau|cyberpunk|steampunk|fantasy|sci.?fi)\b",
        r"\bby\s+\w+\b",  # "by [artist]"
        r"\bstyle of\s+\w+\b",
    ],
    "narrative": [
        r"\b(battle|journey|discovery|reunion|farewell|ritual|ceremony|hunt|escape|"
        r"meditation|celebration|mourning|creation|destruction|transformation)\b",
        r"\b(ancient|futuristic|medieval|modern|post.?apocalyptic|mythological|historical)\b",
    ],
    "technical": [
        r"\b(depth of field|bokeh|lens flare|film grain|chromatic aberration|"
        r"vignette|motion blur|tilt.?shift|macro|wide angle|telephoto|fisheye)\b",
        r"\b(8k|4k|uhd|hdr|raw photo|dslr|cinematic|anamorphic)\b",
    ],
}

# Per-layer enrichment vocabulary (used in fallback path)
_LAYER_ENRICHMENTS: dict[str, list[str]] = {
    "subject": [
        "intricate costume details",
        "expressive micro-expressions",
        "weight and physicality",
        "authentic material textures",
        "believable anatomy",
        "character-defining silhouette",
        "story-worn details",
        "lived-in appearance",
    ],
    "mood": [
        "atmospheric depth",
        "emotional resonance",
        "tonal coherence",
        "mood-consistent color temperature",
        "environmental storytelling",
        "subtle narrative tension",
        "layered emotional subtext",
        "immersive atmosphere",
    ],
    "style": [
        "medium-authentic mark-making",
        "style-consistent edge language",
        "period-accurate visual vocabulary",
        "coherent aesthetic grammar",
        "intentional stylistic choices",
        "unified visual language",
    ],
    "narrative": [
        "implied story context",
        "before-and-after tension",
        "environmental narrative cues",
        "symbolic visual elements",
        "cultural authenticity",
        "historical grounding",
        "mythological resonance",
    ],
    "technical": [
        "optically accurate rendering",
        "physically plausible lighting",
        "sensor-authentic noise",
        "lens-characteristic distortion",
        "natural exposure balance",
    ],
}

# Cross-domain creative bridges: when two layers co-occur, add these
_CROSS_DOMAIN_BRIDGES: dict[tuple[str, str], list[str]] = {
    ("mood", "style"): [
        "style-mood unity",
        "medium reinforces emotion",
        "technique serves narrative",
    ],
    ("subject", "narrative"): [
        "character embodies story",
        "pose tells the moment",
        "costume reflects history",
    ],
    ("mood", "narrative"): [
        "atmosphere amplifies stakes",
        "environment mirrors inner state",
        "light as metaphor",
    ],
    ("style", "technical"): [
        "medium-authentic rendering",
        "technique-consistent artifacts",
        "optically coherent style",
    ],
    ("subject", "mood"): [
        "emotional authenticity",
        "psychological depth",
        "inner life visible",
    ],
}

# Novelty injection: concepts that push beyond statistical defaults
_NOVELTY_INJECTIONS: dict[str, list[str]] = {
    "composition": [
        "unexpected negative space",
        "asymmetric visual weight",
        "foreground-background dialogue",
        "frame within frame",
        "diagonal tension",
        "rule-of-thirds subversion",
        "golden spiral implied",
        "visual rhythm through repetition",
    ],
    "light": [
        "motivated practical light source",
        "subsurface scattering on skin",
        "caustic light patterns",
        "volumetric god rays",
        "reflected fill from environment",
        "color temperature contrast",
        "chiaroscuro drama",
        "ambient occlusion in crevices",
    ],
    "texture": [
        "micro-surface detail",
        "material memory and wear",
        "layered patina",
        "tactile surface quality",
        "age-appropriate weathering",
        "material-specific reflectance",
    ],
    "color": [
        "limited palette with one accent",
        "analogous harmony with tension note",
        "simultaneous contrast",
        "color temperature narrative",
        "desaturated base with saturated focal point",
        "split complementary scheme",
    ],
}


def _decompose_prompt(prompt: str) -> dict[str, list[str]]:
    """Extract semantic layers from a prompt using regex patterns."""
    p = prompt.lower()
    found: dict[str, list[str]] = {}
    for layer, patterns in _LAYER_PATTERNS.items():
        matches = []
        for pat in patterns:
            for m in re.finditer(pat, p, re.IGNORECASE):
                t = m.group(0).strip()
                if t and t not in matches:
                    matches.append(t)
        if matches:
            found[layer] = matches
    return found


def _build_fallback_enrichment(
    prompt: str,
    layers: dict[str, list[str]],
    creativity_level: float,
    rng_seed: int,
) -> tuple[str, str, dict[str, str]]:
    """
    Pure-Python enrichment when heavy models are unavailable.
    Returns (additions_csv, negative_additions, concept_layers_dict).
    """
    import random

    rng = random.Random(rng_seed)

    additions: list[str] = []
    concept_layers: dict[str, str] = {}

    # Per-layer enrichments
    for layer in layers:
        pool = _LAYER_ENRICHMENTS.get(layer, [])
        if not pool:
            continue
        k = max(1, int(round(creativity_level * 2)))
        k = min(k, len(pool))
        chosen = rng.sample(pool, k)
        concept_layers[layer] = ", ".join(chosen)
        additions.extend(chosen)

    # Cross-domain bridges
    layer_keys = list(layers.keys())
    for i, la in enumerate(layer_keys):
        for lb in layer_keys[i + 1 :]:
            key = (la, lb) if (la, lb) in _CROSS_DOMAIN_BRIDGES else (lb, la)
            bridge = _CROSS_DOMAIN_BRIDGES.get(key, [])
            if bridge and rng.random() < creativity_level:
                additions.append(rng.choice(bridge))

    # Novelty injections (scale with creativity_level)
    novelty_categories = list(_NOVELTY_INJECTIONS.keys())
    n_novelty = max(1, int(round(creativity_level * len(novelty_categories))))
    chosen_cats = rng.sample(novelty_categories, min(n_novelty, len(novelty_categories)))
    for cat in chosen_cats:
        pool = _NOVELTY_INJECTIONS[cat]
        additions.append(rng.choice(pool))
        concept_layers[f"novelty_{cat}"] = additions[-1]

    # Deduplicate while preserving order
    seen: set = set()
    deduped: list[str] = []
    for a in additions:
        k = a.lower().strip()
        if k not in seen and k not in prompt.lower():
            seen.add(k)
            deduped.append(a)

    negative_additions = (
        "generic composition, predictable framing, stock photo feel, "
        "statistically average, templated result, derivative concept"
        if creativity_level > 0.5
        else ""
    )

    return ", ".join(deduped), negative_additions, concept_layers


# Heavy model helpers (lazy-loaded, gracefully degraded)


def _moondream_describe_many(
    image_paths: Sequence[str],
    prompt_context: str,
    *,
    device: str = "cpu",
    max_images: int = MAX_MOONDREAM_REFERENCE_IMAGES,
) -> str:
    """
    Run moondream2 once per reference (up to ``max_images``), single model load.

    Covers multi-reference grounding similar to proprietary "many input images"
    workflows: each chunk is labeled [Reference i] for downstream Qwen synthesis.
    """
    paths = [str(p) for p in image_paths if p and Path(str(p)).is_file()]
    paths = paths[: max(0, int(max_images))]
    if not paths or not _MOONDREAM_PATH.exists():
        return ""
    try:
        from PIL import Image as _PIL_Image
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(_MOONDREAM_PATH), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(_MOONDREAM_PATH),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()

        n = len(paths)
        chunks: list[str] = []
        for i, path in enumerate(paths):
            img = _PIL_Image.open(path).convert("RGB")
            question = (
                f"The user may combine multiple references (this is image {i + 1} of {n}). "
                f"Overall generation intent: '{prompt_context}'. "
                "Describe THIS image only: subjects, pose, clothing, materials, "
                "composition, lighting direction, color palette, and any text or logos visible. "
                "Be concrete; note what should be preserved if used as a reference."
            )
            enc = model.encode_image(img)
            answer = model.answer_question(enc, question, tok)
            text = str(answer or "").strip()
            if text:
                chunks.append(f"[Reference {i + 1}]\n{text}")

        try:
            model.to("cpu")
            del model
            import gc

            gc.collect()
            if device != "cpu":
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass

        return "\n\n".join(chunks).strip()
    except Exception:
        return ""


def _qwen_synthesize(
    prompt: str,
    image_description: str,
    facts: list[str],
    layers: dict[str, list[str]],
    creativity_level: float,
    *,
    device: str = "cpu",
    max_new_tokens: int = 512,
) -> tuple[str, str, str]:
    """
    Use Qwen2.5-14B to synthesize novel creative directions.
    Returns (enriched_additions, negative_additions, reasoning).
    """
    if not _QWEN_PATH.exists():
        return "", "", ""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(_QWEN_PATH), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(_QWEN_PATH),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            device_map=device if device != "cpu" else None,
        )
        model.eval()

        # Build a structured reasoning prompt
        facts_block = ""
        if facts:
            facts_block = "\n\nRetrieved context:\n" + "\n".join(f"- {f}" for f in facts[:12])

        img_block = ""
        if image_description:
            img_block = f"\n\nReference image analysis:\n{image_description}"

        layers_block = ""
        if layers:
            layer_strs = [f"  {k}: {', '.join(v[:3])}" for k, v in layers.items()]
            layers_block = "\n\nDetected prompt layers:\n" + "\n".join(layer_strs)

        creativity_instruction = (
            "Push hard for genuinely novel, unexpected creative directions."
            if creativity_level > 0.7
            else "Balance novelty with coherence — enhance without overwhelming."
            if creativity_level > 0.4
            else "Make subtle, targeted improvements that enhance quality."
        )

        system_msg = textwrap.dedent("""
            You are a world-class art director and creative consultant for an AI image generation system.
            Your role is to transform user prompts into richer, more specific, more visually compelling
            descriptions that will produce genuinely novel, high-quality images — not generic, statistically
            average outputs.

            You understand: composition theory, color science, lighting physics, art history, cultural
            context, material properties, and narrative storytelling through visual elements.

            You do NOT add random words. Every addition must serve the image's creative intent.
        """).strip()

        user_msg = textwrap.dedent(f"""
            User's image generation prompt: "{prompt}"
            {facts_block}{img_block}{layers_block}

            Task: {creativity_instruction}

            Analyze the prompt deeply and provide:
            1. ENRICHED_ADDITIONS: A comma-separated list of specific visual/creative additions that will
               make this image more compelling, novel, and high-quality. Focus on:
               - Specific lighting conditions (not just "good lighting")
               - Precise material and texture descriptions
               - Compositional decisions that serve the narrative
               - Color relationships that reinforce mood
               - Unique details that make this image feel intentionally crafted
               - Cross-domain connections (e.g., how the setting reflects the character's psychology)
               Do NOT repeat what's already in the prompt. Max 15 additions.

            2. NEGATIVE_ADDITIONS: Comma-separated terms to avoid that would make this image generic
               or miss the creative intent. Max 8 terms.

            3. REASONING: One paragraph explaining the creative logic behind your additions.

            Respond in this exact format:
            ENRICHED_ADDITIONS: [your additions here]
            NEGATIVE_ADDITIONS: [your negatives here]
            REASONING: [your reasoning here]
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
                do_sample=creativity_level > 0.3,
                temperature=max(0.1, min(1.2, 0.4 + creativity_level * 0.8)),
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tok.eos_token_id,
            )

        response = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

        # Parse structured response
        enriched = _extract_field(response, "ENRICHED_ADDITIONS")
        negatives = _extract_field(response, "NEGATIVE_ADDITIONS")
        reasoning = _extract_field(response, "REASONING")

        return enriched, negatives, reasoning
    except Exception:
        return "", "", ""


def _extract_field(text: str, field_name: str) -> str:
    """Extract a named field from structured LLM output."""
    pattern = rf"{re.escape(field_name)}:\s*(.+?)(?=\n[A-Z_]+:|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("[]")
    return ""


# Main engine


class CreativeRAGEngine:
    """
    Multimodal RAG engine for creative prompt enrichment.

    Lazy-loads heavy models on first use. Falls back gracefully to the
    lightweight semantic enrichment path if models are unavailable.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        cache_size: int = 64,
    ) -> None:
        self._device = device
        self._cache: dict[str, RAGEnrichmentResult] = {}
        self._cache_size = cache_size

    def _cache_key(self, prompt: str, ref_paths_seq: Sequence[str], creativity_level: float) -> str:
        refs = "|".join(str(p) for p in ref_paths_seq)
        raw = f"{prompt}|{refs}|{creativity_level:.2f}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def enrich(
        self,
        prompt: str,
        *,
        reference_image_path: str | None = None,
        reference_image_paths: Sequence[str] | None = None,
        facts: Sequence[str] | None = None,
        creativity_level: float = 0.7,
        seed: int = 42,
        use_qwen: bool = True,
        use_moondream: bool = True,
        use_image_dissection: bool = True,
        max_additions: int = 12,
    ) -> RAGEnrichmentResult:
        """
        Enrich a prompt using multimodal RAG.

        Args:
            prompt: The user's original prompt.
            reference_image_path: Optional path to a reference image. moondream2
                will describe it in correlation with the prompt intent.
            reference_image_paths: More reference paths (merged with the single-path arg;
                capped at ``MAX_REFERENCE_IMAGES_RAG`` for dissection facts; Moondream uses
                the first ``MAX_MOONDREAM_REFERENCE_IMAGES`` existing files).
            facts: Optional list of retrieved facts (from GenSearcher, JSONL, etc.)
                to ground the creative synthesis.
            creativity_level: 0–1. Higher = more novel/unexpected additions.
                0.3 = subtle quality improvements
                0.6 = balanced novelty + coherence
                0.9 = push hard for genuinely unexpected directions
            seed: Deterministic seed for reproducible enrichment.
            use_qwen: Whether to attempt Qwen synthesis (prefers pretrained/Qwen3-14B).
            use_moondream: Whether to attempt moondream image understanding (prefers moondream3).
            max_additions: Maximum number of additions to append to the prompt.

        Returns:
            RAGEnrichmentResult with enriched_prompt and metadata.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            return RAGEnrichmentResult(
                original_prompt="",
                enriched_prompt="",
                negative_additions="",
                reasoning="Empty prompt.",
            )

        # Normalize reference images: keep backwards compatibility with single path.
        ref_paths: list[str] = []
        if reference_image_paths:
            ref_paths = [str(p) for p in reference_image_paths if p and str(p).strip()]
        if reference_image_path and str(reference_image_path).strip():
            if str(reference_image_path) not in ref_paths:
                ref_paths.insert(0, str(reference_image_path))
        ref_paths = ref_paths[:MAX_REFERENCE_IMAGES_RAG]

        cache_key = self._cache_key(prompt, ref_paths, creativity_level)
        if cache_key in self._cache:
            return self._cache[cache_key]

        facts_list = [str(f).strip() for f in (facts or []) if f and str(f).strip()]
        rng_seed = int(hashlib.sha256(f"{prompt}|{seed}".encode()).hexdigest()[:8], 16)

        # Step 1: Semantic decomposition (always runs)
        layers = _decompose_prompt(prompt)

        # Step 1b: Reference-image dissection facts (prompt-driven region constraints)
        if use_image_dissection and ref_paths:
            try:
                from utils.generation.editing import dissect_images_to_parts

                _, _, dissection_facts = dissect_images_to_parts(
                    prompt,
                    ref_paths,
                    output_dir=Path(".") / "runs" / "rag_dissection",
                    default_source_index=0,
                    enable_heavy_models=False,  # facts-only by default (disk-safe)
                )
                # Prepend so they are treated as high-priority constraints.
                facts_list = list(dissection_facts) + facts_list
            except Exception:
                pass

        # Step 2: Image understanding (moondream2) — one description per reference (capped).
        image_description = ""
        if use_moondream and ref_paths:
            image_description = _moondream_describe_many(
                ref_paths,
                prompt,
                device=self._device,
                max_images=MAX_MOONDREAM_REFERENCE_IMAGES,
            )

        # Step 3: Creative synthesis (Qwen2.5 or fallback)
        enriched_additions = ""
        negative_additions = ""
        reasoning = ""
        fallback_used = False
        concept_layers: dict[str, str] = {}

        if use_qwen and _QWEN_PATH.exists():
            enriched_additions, negative_additions, reasoning = _qwen_synthesize(
                prompt,
                image_description,
                facts_list,
                layers,
                creativity_level,
                device=self._device,
            )

        if not enriched_additions:
            # Fallback: lightweight semantic enrichment
            fallback_used = True
            enriched_additions, negative_additions, concept_layers = _build_fallback_enrichment(
                prompt, layers, creativity_level, rng_seed
            )
            reasoning = (
                f"Semantic enrichment applied across {len(layers)} detected layers "
                f"({', '.join(layers.keys())}). "
                + (f"Image context: {image_description[:120]}..." if image_description else "")
                + (f" Grounded in {len(facts_list)} retrieved facts." if facts_list else "")
            )

        # Step 4: Merge facts into additions if Qwen didn't use them
        if facts_list and not (use_qwen and _QWEN_PATH.exists()):
            # Summarize facts as context tokens
            fact_tokens = _facts_to_tokens(facts_list, max_tokens=4)
            if fact_tokens:
                enriched_additions = f"{enriched_additions}, {fact_tokens}" if enriched_additions else fact_tokens

        # Step 5: Trim to max_additions
        if enriched_additions:
            parts = [p.strip() for p in enriched_additions.split(",") if p.strip()]
            # Remove anything already in the prompt
            prompt_lower = prompt.lower()
            parts = [p for p in parts if p.lower() not in prompt_lower]
            parts = parts[:max_additions]
            enriched_additions = ", ".join(parts)

        # Step 6: Build enriched prompt
        if enriched_additions:
            enriched_prompt = f"{prompt}, {enriched_additions}"
        else:
            enriched_prompt = prompt

        # Compute novelty score: fraction of tokens added vs original
        orig_tokens = len(prompt.split(","))
        new_tokens = len(enriched_prompt.split(","))
        novelty_score = min(1.0, (new_tokens - orig_tokens) / max(1, orig_tokens))

        result = RAGEnrichmentResult(
            original_prompt=prompt,
            enriched_prompt=enriched_prompt,
            negative_additions=negative_additions,
            reasoning=reasoning,
            image_description=image_description,
            retrieved_facts=facts_list,
            concept_layers=concept_layers,
            novelty_score=novelty_score,
            fallback_used=fallback_used,
        )

        # Cache management
        if len(self._cache) >= self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[cache_key] = result

        return result

    def enrich_from_jsonl_facts(
        self,
        prompt: str,
        facts_path: str,
        *,
        reference_image_path: str | None = None,
        creativity_level: float = 0.7,
        max_facts: int = 16,
        seed: int = 42,
    ) -> RAGEnrichmentResult:
        """Convenience: load facts from JSONL then enrich."""

        facts = load_facts_from_jsonl(facts_path, max_entries=max_facts)
        return self.enrich(
            prompt,
            reference_image_path=reference_image_path,
            facts=facts,
            creativity_level=creativity_level,
            seed=seed,
        )

    def enrich_from_gen_searcher(
        self,
        prompt: str,
        searcher_json_path: str,
        *,
        reference_image_path: str | None = None,
        creativity_level: float = 0.7,
        max_facts: int = 16,
        seed: int = 42,
    ) -> RAGEnrichmentResult:
        """Convenience: load Gen-Searcher output then enrich."""

        facts = load_facts_from_gen_searcher_json(searcher_json_path, max_entries=max_facts)
        return self.enrich(
            prompt,
            reference_image_path=reference_image_path,
            facts=facts,
            creativity_level=creativity_level,
            seed=seed,
        )


# Helpers


def _facts_to_tokens(facts: list[str], max_tokens: int = 4) -> str:
    """
    Distill a list of facts into a small set of prompt-compatible tokens.
    Extracts the most concrete, visually-relevant nouns and adjectives.
    """
    # Simple heuristic: extract capitalized nouns and distinctive adjectives
    tokens: list[str] = []
    seen: set = set()
    for fact in facts[: max_tokens * 3]:
        # Extract short phrases (1-3 words) that look like visual descriptors
        words = re.findall(r"\b[A-Za-z][a-z]{2,}\b", fact)
        for w in words:
            wl = w.lower()
            if wl in seen or len(wl) < 4:
                continue
            # Skip common stop words
            if wl in {
                "that",
                "this",
                "with",
                "from",
                "have",
                "been",
                "were",
                "they",
                "their",
                "there",
                "which",
                "when",
                "where",
                "what",
                "also",
                "some",
                "more",
                "than",
                "into",
                "over",
                "after",
                "about",
                "through",
                "during",
                "before",
                "between",
            }:
                continue
            seen.add(wl)
            tokens.append(w.lower())
            if len(tokens) >= max_tokens:
                break
        if len(tokens) >= max_tokens:
            break
    return ", ".join(tokens)


# Module-level convenience function

_default_engine: CreativeRAGEngine | None = None


def get_default_engine(device: str = "cpu") -> CreativeRAGEngine:
    """Get or create the module-level default engine."""
    global _default_engine
    if _default_engine is None or _default_engine._device != device:
        _default_engine = CreativeRAGEngine(device=device)
    return _default_engine


def enrich_prompt(
    prompt: str,
    *,
    reference_image_path: str | None = None,
    reference_image_paths: Sequence[str] | None = None,
    facts: Sequence[str] | None = None,
    creativity_level: float = 0.7,
    seed: int = 42,
    device: str = "cpu",
) -> RAGEnrichmentResult:
    """Module-level convenience wrapper around CreativeRAGEngine.enrich()."""
    return get_default_engine(device).enrich(
        prompt,
        reference_image_path=reference_image_path,
        reference_image_paths=reference_image_paths,
        facts=facts,
        creativity_level=creativity_level,
        seed=seed,
    )


# Prompt intent classification

_INTENT_PATTERNS: dict[str, list[str]] = {
    "portrait": [
        r"\b(portrait|headshot|bust|face|close.?up|selfie|profile)\b",
        r"\b(1girl|1boy|1woman|1man|solo)\b",
    ],
    "character": [
        r"\b(character|oc|original character|design|full body|standing|pose)\b",
        r"\b(warrior|mage|knight|wizard|hero|villain|anime character)\b",
    ],
    "landscape": [
        r"\b(landscape|scenery|environment|nature|forest|mountain|ocean|sky|city)\b",
        r"\b(wide shot|establishing shot|panorama|vista)\b",
    ],
    "scene": [
        r"\b(scene|setting|interior|exterior|room|street|battle|crowd)\b",
        r"\b(multiple|group|together|interaction)\b",
    ],
    "abstract": [
        r"\b(abstract|surreal|dreamlike|conceptual|symbolic|metaphor)\b",
        r"\b(pattern|texture|fractal|geometric|non.?representational)\b",
    ],
    "product": [
        r"\b(product|object|item|still life|isolated|white background|studio)\b",
    ],
}

# Semantic conflict pairs: (term_a, term_b, conflict_description)
# These are mutually contradictory regardless of which category they appear in.
_SEMANTIC_CONFLICTS: list[tuple[str, str, str]] = [
    ("photorealistic", "anime", "realism vs stylization"),
    ("photorealistic", "cartoon", "realism vs stylization"),
    ("photorealistic", "illustration", "realism vs stylization"),
    ("photorealistic", "pixel art", "realism vs stylization"),
    ("photorealistic", "oil painting", "realism vs stylization"),
    ("photorealistic", "watercolor", "realism vs stylization"),
    ("realistic", "anime", "realism vs stylization"),
    ("realistic", "cartoon", "realism vs stylization"),
    ("3d render", "watercolor", "3d vs traditional media"),
    ("3d render", "oil painting", "3d vs traditional media"),
    ("3d render", "sketch", "3d vs traditional media"),
    ("day", "night", "time of day"),
    ("daytime", "nighttime", "time of day"),
    ("sunny", "dark", "lighting contradiction"),
    ("bright", "dark", "lighting contradiction"),
    ("indoor", "outdoor", "location contradiction"),
    ("interior", "exterior", "location contradiction"),
    ("smiling", "crying", "expression contradiction"),
    ("happy", "sad", "emotion contradiction"),
    ("young", "old", "age contradiction"),
    ("simple", "detailed", "detail level contradiction"),
    ("minimalist", "detailed", "detail level contradiction"),
    ("minimalist", "complex", "detail level contradiction"),
    ("black and white", "colorful", "color contradiction"),
    ("monochrome", "vibrant colors", "color contradiction"),
    ("standing", "sitting", "pose contradiction"),
    ("standing", "lying", "pose contradiction"),
    ("sitting", "running", "pose contradiction"),
    ("close up", "wide shot", "framing contradiction"),
    ("close up", "full body", "framing contradiction"),
    ("portrait", "wide shot", "framing contradiction"),
    ("from above", "from below", "camera angle contradiction"),
    ("bird eye", "worm eye", "camera angle contradiction"),
    ("soft light", "harsh light", "lighting quality contradiction"),
    ("natural light", "studio lighting", "lighting source contradiction"),
    ("warm", "cold", "color temperature contradiction"),
    ("warm colors", "cool colors", "color temperature contradiction"),
]

# Intent → recommended quality additions
_INTENT_QUALITY_BOOSTS: dict[str, list[str]] = {
    "portrait": [
        "detailed facial features",
        "natural skin texture",
        "expressive eyes",
        "believable subsurface scattering",
    ],
    "character": [
        "consistent character design",
        "clear silhouette",
        "story-worn details",
        "weight and physicality",
    ],
    "landscape": [
        "atmospheric perspective",
        "environmental depth",
        "coherent light source",
        "foreground interest",
    ],
    "scene": [
        "spatial coherence",
        "consistent scale",
        "environmental storytelling",
        "motivated lighting",
    ],
    "abstract": [
        "intentional composition",
        "visual rhythm",
        "color harmony",
        "conceptual clarity",
    ],
    "product": [
        "clean edges",
        "accurate material rendering",
        "neutral background",
        "studio-quality lighting",
    ],
}


def classify_prompt_intent(prompt: str) -> str:
    """Classify the dominant intent of a prompt."""
    p = prompt.lower()
    scores: dict[str, int] = {}
    for intent, patterns in _INTENT_PATTERNS.items():
        score = sum(1 for pat in patterns if re.search(pat, p, re.IGNORECASE))
        if score:
            scores[intent] = score
    return max(scores, key=lambda k: scores[k]) if scores else "scene"


def detect_semantic_conflicts(prompt: str) -> list[dict[str, Any]]:
    """
    Detect semantic contradictions in a prompt regardless of category.
    Returns list of conflict dicts: term_a, term_b, description, severity.
    """
    p = prompt.lower()
    return [
        {"term_a": ta, "term_b": tb, "description": desc, "severity": "high"}
        for ta, tb, desc in _SEMANTIC_CONFLICTS
        if ta in p and tb in p
    ]


def resolve_semantic_conflicts(prompt: str) -> tuple[str, list[str]]:
    """
    Remove the lower-priority (later-occurring) term from each conflicting pair.
    Returns (resolved_prompt, list_of_removed_parts).
    """
    conflicts = detect_semantic_conflicts(prompt)
    if not conflicts:
        return prompt, []

    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    removed: list[str] = []

    for conflict in conflicts:
        term_a, term_b = conflict["term_a"], conflict["term_b"]
        pos_a = next((i for i, p in enumerate(parts) if term_a in p.lower()), -1)
        pos_b = next((i for i, p in enumerate(parts) if term_b in p.lower()), -1)
        if pos_a == -1 or pos_b == -1:
            continue
        # Remove the later-occurring term (user intent = earlier = higher priority)
        remove_idx = pos_b if pos_a <= pos_b else pos_a
        if 0 <= remove_idx < len(parts):
            removed.append(parts[remove_idx])
            parts.pop(remove_idx)

    return ", ".join(parts), removed


@dataclass(slots=True)
class PromptElement:
    """Represents a single element in a prompt."""

    content: str
    category: str
    priority: float
    weight: float = 1.0
    conflicts: tuple | None = field(default=None)

    def __post_init__(self):
        if self.conflicts is None:
            self.conflicts = ()

    def __hash__(self):
        return hash((self.content, self.category, self.priority, self.weight, self.conflicts))

    def __eq__(self, other):
        if not isinstance(other, PromptElement):
            return False
        return (
            self.content == other.content
            and self.category == other.category
            and self.priority == other.priority
            and self.weight == other.weight
            and self.conflicts == other.conflicts
        )


class PromptAnalyzer:
    """Analyze and categorize prompt elements."""

    def __init__(self):
        self.categories = {
            "subject": {
                "keywords": [
                    "person",
                    "woman",
                    "man",
                    "girl",
                    "boy",
                    "character",
                    "figure",
                    "animal",
                    "cat",
                    "dog",
                    "bird",
                    "creature",
                    "object",
                ],
                "priority_base": 0.9,
            },
            "action": {
                "keywords": [
                    "standing",
                    "sitting",
                    "walking",
                    "running",
                    "dancing",
                    "jumping",
                    "holding",
                    "wearing",
                    "looking",
                    "smiling",
                    "crying",
                ],
                "priority_base": 0.8,
            },
            "style": {
                "keywords": [
                    "realistic",
                    "anime",
                    "cartoon",
                    "photorealistic",
                    "artistic",
                    "oil painting",
                    "watercolor",
                    "digital art",
                    "concept art",
                ],
                "priority_base": 0.7,
            },
            "composition": {
                "keywords": [
                    "portrait",
                    "full body",
                    "close up",
                    "wide shot",
                    "from above",
                    "from below",
                    "side view",
                    "front view",
                    "three quarter",
                ],
                "priority_base": 0.6,
            },
            "environment": {
                "keywords": [
                    "background",
                    "forest",
                    "city",
                    "beach",
                    "mountain",
                    "indoor",
                    "outdoor",
                    "studio",
                    "landscape",
                    "room",
                ],
                "priority_base": 0.5,
            },
            "lighting": {
                "keywords": [
                    "bright",
                    "dark",
                    "soft light",
                    "harsh light",
                    "natural light",
                    "dramatic lighting",
                    "sunset",
                    "sunrise",
                    "golden hour",
                ],
                "priority_base": 0.4,
            },
            "quality": {
                "keywords": [
                    "masterpiece",
                    "best quality",
                    "high quality",
                    "detailed",
                    "sharp",
                    "clear",
                    "professional",
                    "8k",
                    "4k",
                ],
                "priority_base": 0.3,
            },
            "technical": {
                "keywords": [
                    "depth of field",
                    "bokeh",
                    "lens flare",
                    "motion blur",
                    "film grain",
                    "chromatic aberration",
                    "vignette",
                ],
                "priority_base": 0.2,
            },
        }

        self.conflict_groups = {
            "style_conflicts": [
                ["realistic", "anime", "cartoon"],
                ["photorealistic", "artistic", "stylized"],
                ["detailed", "simple", "minimalist"],
            ],
            "composition_conflicts": [
                ["portrait", "full body", "close up"],
                ["from above", "from below", "eye level"],
                ["wide shot", "close up", "medium shot"],
            ],
            "lighting_conflicts": [
                ["bright", "dark", "dim"],
                ["soft light", "harsh light", "dramatic lighting"],
                ["natural light", "artificial light", "studio lighting"],
            ],
        }

        # Prompt length optimization
        self.length_thresholds = {
            "short": 50,  # Under 50 chars - too short
            "optimal": 200,  # 50-200 chars - optimal range
            "long": 400,  # 200-400 chars - acceptable
            "too_long": 400,  # Over 400 chars - likely over-specified
        }

    def analyze_prompt(self, prompt: str) -> dict[str, Any]:
        """Analyze a prompt and return detailed breakdown."""
        elements = self._parse_prompt_elements(prompt)
        categorized = self._categorize_elements(elements)
        # Use semantic conflict detection (cross-category, not just within-category)
        conflicts = detect_semantic_conflicts(prompt)
        # Also run the category-level conflict check for intra-category issues
        category_conflicts = self._detect_conflicts(categorized)
        all_conflicts = conflicts + [c for c in category_conflicts if c not in conflicts]
        complexity = self._assess_complexity(prompt, categorized)
        intent = classify_prompt_intent(prompt)

        return {
            "original_prompt": prompt,
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "elements": categorized,
            "conflicts": all_conflicts,
            "semantic_conflicts": conflicts,
            "complexity_score": complexity,
            "intent": intent,
            "optimization_needed": complexity > 0.7 or len(all_conflicts) > 0,
            "recommendations": self._generate_recommendations(prompt, categorized, all_conflicts, complexity, intent),
        }

    def _parse_prompt_elements(self, prompt: str) -> list[tuple[str, float]]:
        """Parse prompt into individual elements."""
        # Split by commas and clean up
        elements = [elem.strip() for elem in prompt.split(",")]

        # Remove empty elements
        elements = [elem for elem in elements if elem]

        # Handle parentheses and brackets (emphasis)
        processed_elements = []
        for elem in elements:
            # Extract emphasis level
            emphasis_level = 1.0
            clean_elem = elem

            # Count parentheses for emphasis
            open_parens = elem.count("(")
            close_parens = elem.count(")")
            if open_parens > 0 and close_parens > 0:
                emphasis_level = 1.0 + (min(open_parens, close_parens) * 0.2)
                clean_elem = re.sub(r"[()]", "", elem).strip()

            # Count brackets for de-emphasis
            open_brackets = elem.count("[")
            close_brackets = elem.count("]")
            if open_brackets > 0 and close_brackets > 0:
                emphasis_level = 1.0 - (min(open_brackets, close_brackets) * 0.2)
                clean_elem = re.sub(r"[\[\]]", "", elem).strip()

            if clean_elem:
                processed_elements.append((clean_elem, emphasis_level))

        return processed_elements

    def _categorize_elements(self, elements: list[tuple[str, float]]) -> dict[str, list[PromptElement]]:
        """Categorize prompt elements by type."""
        categorized = {category: [] for category in self.categories.keys()}

        for element_text, emphasis in elements:
            element_lower = element_text.lower()
            categorized_flag = False

            # Check each category
            for category, info in self.categories.items():
                for keyword in info["keywords"]:
                    if keyword in element_lower:
                        priority = info["priority_base"] * emphasis

                        prompt_element = PromptElement(
                            content=element_text, category=category, priority=priority, weight=emphasis
                        )

                        categorized[category].append(prompt_element)
                        categorized_flag = True
                        break

                if categorized_flag:
                    break

            if not categorized_flag:
                # Default category based on position and content
                default_category = self._determine_default_category(element_text)
                priority = 0.5 * emphasis

                prompt_element = PromptElement(
                    content=element_text, category=default_category, priority=priority, weight=emphasis
                )

                categorized[default_category].append(prompt_element)

        return categorized

    def _determine_default_category(self, element: str) -> str:
        """Determine default category for uncategorized elements."""
        element_lower = element.lower()

        # Simple heuristics
        if any(word in element_lower for word in ["color", "red", "blue", "green", "black", "white"]):
            return "style"
        elif any(word in element_lower for word in ["beautiful", "pretty", "handsome", "cute"]):
            return "quality"
        elif len(element.split()) == 1:
            return "subject"  # Single words often describe subjects
        else:
            return "environment"  # Multi-word phrases often describe environment

    def _detect_conflicts(self, categorized: dict[str, list[PromptElement]]) -> list[dict[str, Any]]:
        """Detect conflicting elements in the prompt."""
        conflicts = []

        for conflict_type, conflict_groups in self.conflict_groups.items():
            for conflict_group in conflict_groups:
                found_conflicts = []

                # Check all categories for conflicting terms
                for category, elements in categorized.items():
                    for element in elements:
                        element_lower = element.content.lower()
                        for conflict_term in conflict_group:
                            if conflict_term in element_lower:
                                found_conflicts.append(
                                    {
                                        "element": element.content,
                                        "category": category,
                                        "conflict_term": conflict_term,
                                        "priority": element.priority,
                                    }
                                )

                # If we found multiple conflicting terms, report the conflict
                if len(found_conflicts) > 1:
                    conflicts.append(
                        {
                            "type": conflict_type,
                            "conflicting_elements": found_conflicts,
                            "severity": "high" if len(found_conflicts) > 2 else "medium",
                        }
                    )

        return conflicts

    def _assess_complexity(self, prompt: str, categorized: dict[str, list[PromptElement]]) -> float:
        """Assess prompt complexity score (0-1)."""
        complexity_factors = {
            "length": min(len(prompt) / 500, 1.0) * 0.3,  # Length factor
            "element_count": min(sum(len(elements) for elements in categorized.values()) / 20, 1.0) * 0.2,
            "category_diversity": (
                len([cat for cat, elements in categorized.items() if elements]) / len(self.categories) * 0.2
            )
            if self.categories
            else 0.0,
            "emphasis_usage": self._count_emphasis_markers(prompt) / 10 * 0.1,
            "technical_terms": self._count_technical_terms(prompt) / 5 * 0.2,
        }

        return sum(complexity_factors.values())

    def _count_emphasis_markers(self, prompt: str) -> int:
        """Count emphasis markers in prompt."""
        return prompt.count("(") + prompt.count("[")

    def _count_technical_terms(self, prompt: str) -> int:
        """Count technical photography/art terms."""
        technical_terms = [
            "depth of field",
            "bokeh",
            "aperture",
            "iso",
            "shutter speed",
            "composition",
            "rule of thirds",
            "golden ratio",
            "leading lines",
            "chromatic aberration",
            "vignette",
            "film grain",
            "lens flare",
        ]

        count = 0
        prompt_lower = prompt.lower()
        for term in technical_terms:
            if term in prompt_lower:
                count += 1

        return count

    def _generate_recommendations(
        self,
        prompt: str,
        categorized: dict[str, list[PromptElement]],
        conflicts: list[dict[str, Any]],
        complexity: float,
        intent: str = "scene",
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Length recommendations
        prompt_length = len(prompt)
        if prompt_length < self.length_thresholds["short"]:
            recommendations.append("Prompt is too short - add more descriptive details")
        elif prompt_length > self.length_thresholds["too_long"]:
            recommendations.append("Prompt is too long - consider simplifying or using multi-stage generation")

        # Semantic conflict recommendations (cross-category)
        semantic = [c for c in conflicts if "term_a" in c]
        if semantic:
            recommendations.append(
                f"Found {len(semantic)} semantic contradiction(s) - these will fight each other during generation"
            )
            for c in semantic:
                recommendations.append(
                    f"Contradiction: '{c['term_a']}' vs '{c['term_b']}' ({c['description']}) — "
                    f"keep the one that matches your intent, remove the other"
                )

        # Category conflict recommendations
        category_conflicts = [c for c in conflicts if "type" in c]
        if category_conflicts:
            recommendations.append(f"Found {len(category_conflicts)} category conflicts - resolve contradictory terms")
            for conflict in category_conflicts:
                if conflict.get("severity") == "high":
                    recommendations.append(f"High priority: resolve {conflict['type']} conflicts")

        # Complexity recommendations
        if complexity > 0.8:
            recommendations.append("Very high complexity - consider breaking into multiple prompts")
        elif complexity > 0.6:
            recommendations.append("High complexity - prioritize most important elements")

        # Category balance recommendations
        element_counts = {cat: len(elements) for cat, elements in categorized.items() if elements}

        if "subject" not in element_counts or element_counts["subject"] == 0:
            recommendations.append("Add clear subject description")

        if element_counts.get("quality", 0) == 0:
            recommendations.append("Consider adding quality tags (masterpiece, best quality)")

        if element_counts.get("style", 0) == 0:
            recommendations.append("Consider specifying art style")

        # Intent-specific recommendations
        intent_boosts = _INTENT_QUALITY_BOOSTS.get(intent, [])
        if intent_boosts and element_counts.get("quality", 0) < 2:
            recommendations.append(f"For {intent} intent, consider adding: {', '.join(intent_boosts[:2])}")

        # Over-specification warnings
        if element_counts.get("technical", 0) > 3:
            recommendations.append("Too many technical terms - may confuse the model")

        if sum(element_counts.values()) > 15:
            recommendations.append("Too many elements - focus on the most important ones")

        return recommendations


class PromptOptimizer:
    """Optimize prompts for better generation results."""

    def __init__(self):
        self.analyzer = PromptAnalyzer()

        # Optimization strategies
        self.optimization_strategies = {
            "simplify": self._simplify_prompt,
            "prioritize": self._prioritize_elements,
            "resolve_conflicts": self._resolve_conflicts,
            "balance_categories": self._balance_categories,
            "enhance_clarity": self._enhance_clarity,
        }

        # Quality enhancers by category
        self.quality_enhancers = {
            "general": ["masterpiece", "best quality", "high quality"],
            "realistic": ["photorealistic", "detailed", "sharp focus"],
            "artistic": ["artistic", "creative", "expressive"],
            "anime": ["anime style", "clean lines", "vibrant colors"],
            "portrait": ["detailed face", "clear eyes", "natural expression"],
            "landscape": ["scenic", "atmospheric", "wide view"],
        }

    def optimize_prompt(
        self, prompt: str, optimization_level: str = "balanced", target_style: str = None, max_length: int = 300
    ) -> dict[str, Any]:
        """Optimize a prompt using various strategies."""
        # Analyze original prompt
        analysis = self.analyzer.analyze_prompt(prompt)

        # Apply optimization strategies based on level
        if optimization_level == "minimal":
            strategies = ["resolve_conflicts"]
        elif optimization_level == "balanced":
            strategies = ["resolve_conflicts", "prioritize", "enhance_clarity"]
        elif optimization_level == "aggressive":
            strategies = ["simplify", "resolve_conflicts", "prioritize", "balance_categories"]
        else:
            strategies = ["prioritize", "enhance_clarity"]

        # Apply strategies
        optimized_elements = analysis["elements"]
        optimization_log = []

        for strategy in strategies:
            if strategy in self.optimization_strategies:
                optimized_elements, log_entry = self.optimization_strategies[strategy](
                    optimized_elements, analysis, target_style
                )
                optimization_log.append(log_entry)

        # Build optimized prompt
        optimized_prompt = self._build_optimized_prompt(optimized_elements, max_length, target_style)

        # Analyze optimized prompt
        optimized_analysis = self.analyzer.analyze_prompt(optimized_prompt)

        return {
            "original_prompt": prompt,
            "optimized_prompt": optimized_prompt,
            "original_analysis": analysis,
            "optimized_analysis": optimized_analysis,
            "optimization_log": optimization_log,
            "improvement_score": self._calculate_improvement_score(analysis, optimized_analysis),
        }

    def _simplify_prompt(
        self, elements: dict[str, list[PromptElement]], analysis: dict[str, Any], target_style: str
    ) -> tuple[dict[str, list[PromptElement]], str]:
        """Simplify prompt by removing low-priority elements."""
        simplified = {}
        removed_count = 0

        for category, element_list in elements.items():
            # Keep only high-priority elements
            simplified[category] = [
                elem for elem in element_list if elem.priority > 0.5 or category in ["subject", "action"]
            ]
            removed_count += len(element_list) - len(simplified[category])

        log_entry = f"Simplified: removed {removed_count} low-priority elements"
        return simplified, log_entry

    def _prioritize_elements(
        self, elements: dict[str, list[PromptElement]], analysis: dict[str, Any], target_style: str
    ) -> tuple[dict[str, list[PromptElement]], str]:
        """Prioritize elements by importance and relevance."""
        prioritized = {}

        # Priority order for categories
        category_priority = {
            "subject": 1.0,
            "action": 0.9,
            "style": 0.8,
            "composition": 0.7,
            "environment": 0.6,
            "lighting": 0.5,
            "quality": 0.4,
            "technical": 0.3,
        }

        for category, element_list in elements.items():
            # Adjust priorities based on category importance
            category_multiplier = category_priority.get(category, 0.5)

            prioritized[category] = []
            for elem in element_list:
                new_priority = elem.priority * category_multiplier

                # Boost priority if matches target style
                if target_style and target_style.lower() in elem.content.lower():
                    new_priority *= 1.3

                # Create a copy to avoid mutating the shared element
                updated_elem = copy.copy(elem)
                updated_elem.priority = new_priority
                prioritized[category].append(updated_elem)

            # Sort by priority within category
            prioritized[category].sort(key=lambda x: x.priority, reverse=True)

        log_entry = "Prioritized: adjusted element priorities by category and relevance"
        return prioritized, log_entry

    def _resolve_conflicts(
        self, elements: dict[str, list[PromptElement]], analysis: dict[str, Any], target_style: str
    ) -> tuple[dict[str, list[PromptElement]], str]:
        """Resolve conflicting elements by keeping highest priority ones."""
        resolved = {}
        conflicts_resolved = 0

        for category, element_list in elements.items():
            resolved[category] = []

            # Group potentially conflicting elements
            conflict_groups = self._identify_element_conflicts(element_list)

            for group in conflict_groups:
                if len(group) > 1:
                    # Keep only the highest priority element from conflicting group
                    best_element = max(group, key=lambda x: x.priority)
                    resolved[category].append(best_element)
                    conflicts_resolved += len(group) - 1
                else:
                    resolved[category].extend(group)

            # Add non-conflicting elements
            all_conflicting = set()
            for group in conflict_groups:
                all_conflicting.update(group)

            for elem in element_list:
                if elem not in all_conflicting:
                    resolved[category].append(elem)

        log_entry = f"Resolved conflicts: removed {conflicts_resolved} conflicting elements"
        return resolved, log_entry

    def _balance_categories(
        self, elements: dict[str, list[PromptElement]], analysis: dict[str, Any], target_style: str
    ) -> tuple[dict[str, list[PromptElement]], str]:
        """Balance elements across categories."""
        balanced = {}

        # Target distribution (max elements per category)
        max_per_category = {
            "subject": 3,
            "action": 2,
            "style": 2,
            "composition": 2,
            "environment": 2,
            "lighting": 1,
            "quality": 2,
            "technical": 1,
        }

        total_trimmed = 0

        for category, element_list in elements.items():
            max_allowed = max_per_category.get(category, 2)

            if len(element_list) > max_allowed:
                # Keep top elements by priority
                balanced[category] = sorted(element_list, key=lambda x: x.priority, reverse=True)[:max_allowed]
                total_trimmed += len(element_list) - max_allowed
            else:
                balanced[category] = element_list

        log_entry = f"Balanced categories: trimmed {total_trimmed} elements for better distribution"
        return balanced, log_entry

    def _enhance_clarity(
        self, elements: dict[str, list[PromptElement]], analysis: dict[str, Any], target_style: str
    ) -> tuple[dict[str, list[PromptElement]], str]:
        """Enhance prompt clarity by adding missing essential elements."""
        enhanced = {cat: list(elem_list) for cat, elem_list in elements.items()}  # Deep-enough copy
        additions = 0

        # Check for missing essential elements
        element_counts = {cat: len(elem_list) for cat, elem_list in elements.items()}

        # Add quality tags if missing
        if element_counts.get("quality", 0) == 0:
            quality_tags = self.quality_enhancers["general"][:2]
            for tag in quality_tags:
                enhanced["quality"].append(PromptElement(content=tag, category="quality", priority=0.4, weight=1.0))
                additions += 1

        # Add style clarity if target style specified
        if target_style and element_counts.get("style", 0) == 0:
            enhanced["style"].append(PromptElement(content=target_style, category="style", priority=0.8, weight=1.0))
            additions += 1

        log_entry = f"Enhanced clarity: added {additions} essential elements"
        return enhanced, log_entry

    def _identify_element_conflicts(self, elements: list[PromptElement]) -> list[list[PromptElement]]:
        """Identify conflicting elements within a category."""
        # This is simplified - would be more sophisticated in practice
        conflict_keywords = [
            ["realistic", "anime", "cartoon"],
            ["detailed", "simple", "minimalist"],
            ["bright", "dark", "dim"],
            ["large", "small", "tiny"],
            ["old", "young", "new"],
        ]

        conflict_groups = []
        used_elements = set()

        for keywords in conflict_keywords:
            conflicting_elements = []

            for element in elements:
                if element in used_elements:
                    continue

                element_lower = element.content.lower()
                for keyword in keywords:
                    if keyword in element_lower:
                        conflicting_elements.append(element)
                        used_elements.add(element)
                        break

            if conflicting_elements:
                conflict_groups.append(conflicting_elements)

        # Add non-conflicting elements as individual groups
        for element in elements:
            if element not in used_elements:
                conflict_groups.append([element])

        return conflict_groups

    def _build_optimized_prompt(
        self, elements: dict[str, list[PromptElement]], max_length: int, target_style: str
    ) -> str:
        """Build the final optimized prompt."""
        # Collect all elements and sort by priority
        all_elements = []
        for element_list in elements.values():
            all_elements.extend(element_list)

        all_elements.sort(key=lambda x: x.priority, reverse=True)

        # Build prompt respecting length limit
        prompt_parts = []
        current_length = 0

        for element in all_elements:
            element_text = element.content

            # Add emphasis markers based on weight
            if element.weight > 1.2:
                element_text = f"({element_text})"
            elif element.weight < 0.8:
                element_text = f"[{element_text}]"

            # Check if adding this element would exceed length limit
            if current_length + len(element_text) + 2 > max_length:  # +2 for ", "
                break

            prompt_parts.append(element_text)
            current_length += len(element_text) + 2

        return ", ".join(prompt_parts)

    def _calculate_improvement_score(
        self, original_analysis: dict[str, Any], optimized_analysis: dict[str, Any]
    ) -> float:
        """Calculate improvement score between original and optimized prompts."""
        improvements = 0.0

        # Conflict reduction
        original_conflicts = len(original_analysis.get("conflicts", []))
        optimized_conflicts = len(optimized_analysis.get("conflicts", []))
        if original_conflicts > optimized_conflicts:
            improvements += 0.3

        # Complexity reduction
        original_complexity = original_analysis.get("complexity_score", 0)
        optimized_complexity = optimized_analysis.get("complexity_score", 0)
        if original_complexity > optimized_complexity:
            improvements += 0.3

        # Length optimization
        original_length = original_analysis.get("length", 0)
        optimized_length = optimized_analysis.get("length", 0)
        if 50 <= optimized_length <= 300 and abs(optimized_length - 200) < abs(original_length - 200):
            improvements += 0.2

        # Recommendation reduction
        original_recs = len(original_analysis.get("recommendations", []))
        optimized_recs = len(optimized_analysis.get("recommendations", []))
        if original_recs > optimized_recs:
            improvements += 0.2

        return min(improvements, 1.0)


# CreativeRAGOptimizer: full pipeline combining all systems


class CreativeRAGOptimizer:
    """
    Full creative pipeline:
    1. Semantic conflict resolution (cross-category, not just within-category)
    2. Intent classification → strategy selection
    3. PromptOptimizer (structural optimization)
    4. Intent-specific quality boosts
    5. CreativeRAGEngine enrichment (novelty + grounding via moondream2 + Qwen2.5)

    This is the recommended entry point for maximum output quality.
    """

    def __init__(self, *, device: str = "cpu") -> None:
        self.analyzer = PromptAnalyzer()
        self.optimizer = PromptOptimizer()
        self._device = device
        self._rag_engine: Any | None = None  # lazy-loaded

    def _get_rag_engine(self) -> Any | None:
        if self._rag_engine is None:
            try:
                self._rag_engine = CreativeRAGEngine(device=self._device)
            except Exception:
                self._rag_engine = None
        return self._rag_engine

    def optimize(
        self,
        prompt: str,
        *,
        reference_image_path: str | None = None,
        reference_image_paths: Sequence[str] | None = None,
        facts: Sequence[str] | None = None,
        creativity_level: float = 0.7,
        optimization_level: str = "balanced",
        max_length: int = 350,
        seed: int = 42,
        use_rag: bool = True,
    ) -> dict[str, Any]:
        """
        Full creative optimization pipeline.

        Args:
            prompt: User's original prompt.
            reference_image_path: Optional reference image for moondream2 understanding.
            reference_image_paths: Optional multiple references (combined with Creative RAGEngine caps).
            facts: Optional retrieved facts for grounding.
            creativity_level: 0-1. Controls novelty injection strength.
            optimization_level: "minimal" | "balanced" | "aggressive"
            max_length: Maximum character length of final prompt.
            seed: Deterministic seed.
            use_rag: Whether to run the CreativeRAG enrichment step.

        Returns:
            Dict with keys: optimized_prompt, negative_additions, analysis,
            semantic_conflicts_resolved, rag_result, intent, improvement_score.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            return {"optimized_prompt": "", "negative_additions": "", "intent": "scene"}

        # Step 1: Resolve semantic conflicts before anything else
        resolved_prompt, removed_terms = resolve_semantic_conflicts(prompt)
        conflicts_resolved = len(removed_terms)

        # Step 2: Classify intent
        intent = classify_prompt_intent(resolved_prompt)

        # Step 3: Structural optimization
        opt_result = self.optimizer.optimize_prompt(
            resolved_prompt,
            optimization_level=optimization_level,
            max_length=max_length,
        )
        structurally_optimized = opt_result["optimized_prompt"]

        # Step 4: Intent-specific quality boosts (add if not already present)
        intent_boosts = _INTENT_QUALITY_BOOSTS.get(intent, [])
        prompt_lower = structurally_optimized.lower()
        missing_boosts = [b for b in intent_boosts[:2] if b.lower() not in prompt_lower]
        if missing_boosts:
            structurally_optimized = f"{structurally_optimized}, {', '.join(missing_boosts)}"

        # Step 5: Creative RAG enrichment
        rag_result = None
        negative_additions = ""
        final_prompt = structurally_optimized

        if use_rag:
            engine = self._get_rag_engine()
            if engine is not None:
                try:
                    rag_result = engine.enrich(
                        structurally_optimized,
                        reference_image_path=reference_image_path,
                        reference_image_paths=reference_image_paths,
                        facts=facts,
                        creativity_level=creativity_level,
                        seed=seed,
                    )
                    final_prompt = rag_result.enriched_prompt
                    negative_additions = rag_result.negative_additions
                except Exception:
                    pass

        # Enforce max_length
        if len(final_prompt) > max_length:
            parts = [p.strip() for p in final_prompt.split(",") if p.strip()]
            trimmed: list[str] = []
            length = 0
            for p in parts:
                if length + len(p) + 2 > max_length:
                    break
                trimmed.append(p)
                length += len(p) + 2
            final_prompt = ", ".join(trimmed)

        # Compute improvement score
        analysis = self.analyzer.analyze_prompt(final_prompt)
        original_analysis = self.analyzer.analyze_prompt(prompt)
        improvement = self.optimizer._calculate_improvement_score(original_analysis, analysis)

        return {
            "original_prompt": prompt,
            "optimized_prompt": final_prompt,
            "negative_additions": negative_additions,
            "intent": intent,
            "semantic_conflicts_resolved": conflicts_resolved,
            "removed_terms": removed_terms,
            "analysis": analysis,
            "rag_result": rag_result,
            "improvement_score": improvement,
            "optimization_log": opt_result.get("optimization_log", []),
        }
