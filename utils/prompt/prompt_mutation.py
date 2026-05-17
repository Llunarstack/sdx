"""
Prompt mutation engine: generate N semantically-varied mutations of a prompt,
generate all of them, and pick the best result.

This is "test-time compute" applied at the PROMPT level rather than the latent level.
Instead of generating N images from the same prompt and picking the best (which is
what --pick-best does), this generates N DIFFERENT prompts that all describe the same
intent, generates one image per prompt, and picks the best.

Why this is better than just --num N:
- Different prompts activate different parts of the model's knowledge
- A prompt that's slightly more specific often produces dramatically better results
- Catches cases where the original prompt is ambiguous or uses vocabulary the model
  doesn't respond well to
- Combines with --pick-best for a 2D search: N prompts × M candidates each

Mutation strategies:
1. SYNONYM: Replace key words with synonyms (woman → female figure, forest → woodland)
2. SPECIFICITY: Add specific details (sunset → golden hour sunset, warm amber light)
3. STYLE_ANCHOR: Add style-specific vocabulary (photorealistic → shot on Canon 5D, f/1.8)
4. REORDER: Put the most important elements first (T5 truncates at 300 tokens)
5. EMPHASIS: Add (word) emphasis to key elements
6. NEGATIVE_INVERSION: Rephrase to avoid common failure modes
7. CULTURAL_CONTEXT: Add cultural/historical grounding for better authenticity
8. TECHNICAL_SPEC: Add technical photography/art terms that activate quality circuits

Usage:
    engine = PromptMutationEngine(n_mutations=4, strategies=["synonym", "specificity"])
    mutations = engine.mutate("a beautiful woman in a forest at sunset")
    # Returns list of 4 varied prompts

    # Full pipeline with generation:
    pipeline = MutationGenerationPipeline(engine, pick_metric="combo_vit_hq")
    best_image, best_prompt, scores = pipeline.run(
        original_prompt="a beautiful woman in a forest at sunset",
        generate_fn=my_generate_function,
    )
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Vocabulary tables for mutations
# ---------------------------------------------------------------------------

_SYNONYM_MAP: Dict[str, List[str]] = {
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

_SPECIFICITY_ADDITIONS: Dict[str, List[str]] = {
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

_STYLE_ANCHORS: Dict[str, List[str]] = {
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

_CULTURAL_CONTEXTS: Dict[str, List[str]] = {
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
    changes: List[str] = field(default_factory=list)
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
        strategies: Optional[List[str]] = None,
        seed: Optional[int] = None,
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

    def mutate(self, prompt: str) -> List[MutationResult]:
        """
        Generate n_mutations varied versions of the prompt.

        Returns list of MutationResult, always including the original as first entry.
        """
        rng = self._get_rng(prompt)
        results = [MutationResult(prompt=prompt, strategy="original", confidence=1.0)]

        # Apply each strategy in rotation
        strategy_cycle = list(self.strategies) * (self.n_mutations // len(self.strategies) + 1)
        rng.shuffle(strategy_cycle)

        for strategy in strategy_cycle[:self.n_mutations]:
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

        return deduped[:self.n_mutations + 1]

    def _apply_strategy(
        self,
        prompt: str,
        strategy: str,
        rng: random.Random,
    ) -> Optional[MutationResult]:
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

    def _mutate_synonym(self, prompt: str, rng: random.Random) -> Optional[MutationResult]:
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
            prompt=", ".join(new_parts)[:self.max_length],
            strategy="synonym",
            changes=changes,
            confidence=0.8,
        )

    def _mutate_specificity(self, prompt: str, rng: random.Random) -> Optional[MutationResult]:
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
            prompt=new_prompt[:self.max_length],
            strategy="specificity",
            changes=additions,
            confidence=0.9,
        )

    def _mutate_style_anchor(self, prompt: str, rng: random.Random) -> Optional[MutationResult]:
        """Add style-specific technical vocabulary."""
        prompt_lower = prompt.lower()

        for style_key, anchors in _STYLE_ANCHORS.items():
            if style_key in prompt_lower:
                anchor = rng.choice(anchors)
                if anchor.lower() not in prompt_lower:
                    new_prompt = f"{prompt}, {anchor}"
                    return MutationResult(
                        prompt=new_prompt[:self.max_length],
                        strategy="style_anchor",
                        changes=[anchor],
                        confidence=0.85,
                    )

        return None

    def _mutate_reorder(self, prompt: str, rng: random.Random) -> Optional[MutationResult]:
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
            prompt=new_prompt[:self.max_length],
            strategy="reorder",
            changes=["moved subject to front"],
            confidence=0.75,
        )

    def _mutate_emphasis(self, prompt: str, rng: random.Random) -> Optional[MutationResult]:
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
            prompt=", ".join(new_parts)[:self.max_length],
            strategy="emphasis",
            changes=[f"emphasized {emphasized} elements"],
            confidence=0.7,
        )

    def _mutate_quality_anchor(self, prompt: str, rng: random.Random) -> Optional[MutationResult]:
        """Add quality anchor tags if missing."""
        prompt_lower = prompt.lower()
        quality_present = any(
            q in prompt_lower
            for q in ["masterpiece", "best quality", "award", "professional", "8k", "4k"]
        )

        if quality_present:
            return None

        anchor = rng.choice(_QUALITY_ANCHORS)
        new_prompt = f"{anchor}, {prompt}"

        return MutationResult(
            prompt=new_prompt[:self.max_length],
            strategy="quality_anchor",
            changes=[anchor],
            confidence=0.85,
        )

    def _mutate_cultural_context(self, prompt: str, rng: random.Random) -> Optional[MutationResult]:
        """Add cultural/historical grounding."""
        prompt_lower = prompt.lower()

        for keyword, contexts in _CULTURAL_CONTEXTS.items():
            if keyword in prompt_lower:
                context = rng.choice(contexts)
                if context.lower() not in prompt_lower:
                    new_prompt = f"{prompt}, {context}"
                    return MutationResult(
                        prompt=new_prompt[:self.max_length],
                        strategy="cultural_context",
                        changes=[context],
                        confidence=0.8,
                    )

        return None

    def _mutate_technical_spec(self, prompt: str, rng: random.Random) -> Optional[MutationResult]:
        """Add technical photography/art terms."""
        prompt_lower = prompt.lower()

        # Only add if not already heavily technical
        technical_count = sum(
            1 for t in ["depth of field", "bokeh", "f/", "mm", "iso", "aperture", "shutter"]
            if t in prompt_lower
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
                prompt=new_prompt[:self.max_length],
                strategy="technical_spec",
                changes=[spec],
                confidence=0.75,
            )

        return None

    def _mutate_negative_inversion(self, prompt: str, rng: random.Random) -> Optional[MutationResult]:
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
            prompt=", ".join(new_parts)[:self.max_length],
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
    best_image: Any                         # numpy array (H, W, 3) uint8
    best_prompt: str
    best_score: float
    original_prompt: str
    all_mutations: List[MutationResult]
    all_scores: List[float]
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
        engine: Optional[PromptMutationEngine] = None,
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


__all__ = [
    "PromptMutationEngine",
    "MutationResult",
    "MutationGenerationPipeline",
    "MutationPipelineResult",
]
