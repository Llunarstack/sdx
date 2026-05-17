"""
Creative RAG (Retrieval-Augmented Generation) engine for SDX.

Bridges the gap between "statistically probable" and "genuinely novel" by:

1. **Image understanding** — uses moondream2 (pretrained/moondream2) to deeply
   describe a reference image in correlation with the user's prompt intent.
2. **Concept synthesis** — uses Qwen2.5-14B (pretrained/Qwen2.5-14B-Instruct) to
   reason about the prompt + image description and generate novel creative directions
   that go beyond what the model would produce by default.
3. **Fact grounding** — merges retrieved facts (from GenSearcher or any JSONL source)
   into the prompt via the existing rag_prompt.py pipeline.
4. **Semantic decomposition** — breaks the prompt into intent layers (subject, mood,
   style, narrative, technical) and enriches each layer independently.

All heavy models are loaded lazily and cached — if they're not present in pretrained/,
the engine degrades gracefully to the lightweight fallback path.

Usage (inference, wired into sample.py via --creative-rag):
    from utils.prompt.creative_rag import CreativeRAGEngine
    engine = CreativeRAGEngine()
    result = engine.enrich(
        prompt="a lone samurai at sunset",
        reference_image_path="ref.jpg",   # optional
        facts=["Edo period Japan", "wabi-sabi aesthetic"],  # optional
        creativity_level=0.8,
    )
    # result.enriched_prompt  → richer, more novel prompt
    # result.negative_additions → suggested negative additions
    # result.reasoning        → why these additions were made
"""

from __future__ import annotations

import hashlib
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Pretrained model paths (mirrors utils/modeling/model_paths.py conventions)
# ---------------------------------------------------------------------------
_PRETRAINED_ROOT = Path(__file__).resolve().parents[2] / "pretrained"
_MOONDREAM_PATH = _PRETRAINED_ROOT / "moondream2"
_QWEN_PATH = _PRETRAINED_ROOT / "Qwen2.5-14B-Instruct"
_GENSEARCHER_PATH = _PRETRAINED_ROOT / "GenSearcher-8B"

# Aligned with multi-reference ingestion (API-style caps); dissect/RAG facts use full list.
MAX_REFERENCE_IMAGES_RAG = 16
# Moondream runs per-image; keep a lower cap so enrichment stays tractable on one GPU.
MAX_MOONDREAM_REFERENCE_IMAGES = 8


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class RAGEnrichmentResult:
    """Output of CreativeRAGEngine.enrich()."""

    original_prompt: str
    enriched_prompt: str
    negative_additions: str
    reasoning: str
    image_description: str = ""
    retrieved_facts: List[str] = field(default_factory=list)
    concept_layers: Dict[str, str] = field(default_factory=dict)
    novelty_score: float = 0.0  # 0–1: how much was added vs original
    fallback_used: bool = False  # True when heavy models were unavailable


# ---------------------------------------------------------------------------
# Lightweight semantic decomposition (no heavy models required)
# ---------------------------------------------------------------------------

# Intent keywords → semantic layer
_LAYER_PATTERNS: Dict[str, List[str]] = {
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
_LAYER_ENRICHMENTS: Dict[str, List[str]] = {
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
_CROSS_DOMAIN_BRIDGES: Dict[Tuple[str, str], List[str]] = {
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
_NOVELTY_INJECTIONS: Dict[str, List[str]] = {
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


def _decompose_prompt(prompt: str) -> Dict[str, List[str]]:
    """Extract semantic layers from a prompt using regex patterns."""
    p = prompt.lower()
    found: Dict[str, List[str]] = {}
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
    layers: Dict[str, List[str]],
    creativity_level: float,
    rng_seed: int,
) -> Tuple[str, str, Dict[str, str]]:
    """
    Pure-Python enrichment when heavy models are unavailable.
    Returns (additions_csv, negative_additions, concept_layers_dict).
    """
    import random

    rng = random.Random(rng_seed)

    additions: List[str] = []
    concept_layers: Dict[str, str] = {}

    # Per-layer enrichments
    for layer, matches in layers.items():
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
    deduped: List[str] = []
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


# ---------------------------------------------------------------------------
# Heavy model helpers (lazy-loaded, gracefully degraded)
# ---------------------------------------------------------------------------


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
        chunks: List[str] = []
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


def _moondream_describe(
    image_path: str,
    prompt_context: str,
    *,
    device: str = "cpu",
) -> str:
    """Describe a single reference via moondream2 (wrapper over ``_moondream_describe_many``)."""
    return _moondream_describe_many([image_path], prompt_context, device=device, max_images=1)


def _qwen_synthesize(
    prompt: str,
    image_description: str,
    facts: List[str],
    layers: Dict[str, List[str]],
    creativity_level: float,
    *,
    device: str = "cpu",
    max_new_tokens: int = 512,
) -> Tuple[str, str, str]:
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


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


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
        self._cache: Dict[str, RAGEnrichmentResult] = {}
        self._cache_size = cache_size

    def _cache_key(self, prompt: str, ref_paths_seq: Sequence[str], creativity_level: float) -> str:
        refs = "|".join(str(p) for p in ref_paths_seq)
        raw = f"{prompt}|{refs}|{creativity_level:.2f}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def enrich(
        self,
        prompt: str,
        *,
        reference_image_path: Optional[str] = None,
        reference_image_paths: Optional[Sequence[str]] = None,
        facts: Optional[Sequence[str]] = None,
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
            use_qwen: Whether to attempt Qwen2.5 synthesis (requires pretrained/Qwen2.5-14B-Instruct).
            use_moondream: Whether to attempt moondream2 image understanding (requires pretrained/moondream2).
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
        ref_paths: List[str] = []
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
                from utils.generation.image_dissection import dissect_images_to_parts

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
        concept_layers: Dict[str, str] = {}

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
        reference_image_path: Optional[str] = None,
        creativity_level: float = 0.7,
        max_facts: int = 16,
        seed: int = 42,
    ) -> RAGEnrichmentResult:
        """Convenience: load facts from JSONL then enrich."""
        from utils.prompt.rag_prompt import load_facts_from_jsonl

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
        reference_image_path: Optional[str] = None,
        creativity_level: float = 0.7,
        max_facts: int = 16,
        seed: int = 42,
    ) -> RAGEnrichmentResult:
        """Convenience: load Gen-Searcher output then enrich."""
        from utils.prompt.rag_prompt import load_facts_from_gen_searcher_json

        facts = load_facts_from_gen_searcher_json(searcher_json_path, max_entries=max_facts)
        return self.enrich(
            prompt,
            reference_image_path=reference_image_path,
            facts=facts,
            creativity_level=creativity_level,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _facts_to_tokens(facts: List[str], max_tokens: int = 4) -> str:
    """
    Distill a list of facts into a small set of prompt-compatible tokens.
    Extracts the most concrete, visually-relevant nouns and adjectives.
    """
    # Simple heuristic: extract capitalized nouns and distinctive adjectives
    tokens: List[str] = []
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


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_engine: Optional[CreativeRAGEngine] = None


def get_default_engine(device: str = "cpu") -> CreativeRAGEngine:
    """Get or create the module-level default engine."""
    global _default_engine
    if _default_engine is None or _default_engine._device != device:
        _default_engine = CreativeRAGEngine(device=device)
    return _default_engine


def enrich_prompt(
    prompt: str,
    *,
    reference_image_path: Optional[str] = None,
    reference_image_paths: Optional[Sequence[str]] = None,
    facts: Optional[Sequence[str]] = None,
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
