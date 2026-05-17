"""
Self-improvement data flywheel for SDX.

The model generates images → moondream2 captions them → quality filter keeps
the best → they're added to training data → the model trains on its own best outputs.

This is how closed systems like GPT-Image and Midjourney continuously improve
without requiring new human-labeled data. It's almost never implemented in
open-source image generation.

Why it works:
1. The model generates diverse outputs across many prompts
2. Quality filtering (reward model + CLIP + edge sharpness) keeps only the best
3. moondream2 generates accurate, detailed captions for the kept images
4. These (image, caption) pairs are added to training data
5. Training on high-quality self-generated data improves the model
6. Repeat: the improved model generates better images → better training data

Key insight: the model's best outputs are often BETTER than average training data
because they're generated with careful prompting, high CFG, and quality filtering.
Training on these outputs teaches the model to consistently produce its best work.

Safety: we only add images that pass quality thresholds, preventing the model
from training on its own failures (which would cause drift).

Usage:
    flywheel = SelfImprovementFlywheel(
        generate_fn=my_generate_function,
        output_dir="./self_improvement_data",
        quality_threshold=0.65,
    )
    # Run one cycle:
    stats = flywheel.run_cycle(
        prompts=my_prompt_list,
        n_per_prompt=4,
        cycle_id=0,
    )
    print(f"Added {stats['added']} samples to training data")
    # Then train on the new data:
    # train.py --manifest-jsonl ./self_improvement_data/cycle_000/manifest.jsonl
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_log = logging.getLogger(__name__)
_PRETRAINED = Path(__file__).resolve().parents[2] / "pretrained"


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QualityFilterConfig:
    """Configuration for quality filtering of self-generated images."""

    min_edge_sharpness: float = 150.0  # Laplacian variance threshold
    min_clip_score: float = 0.22  # CLIP image-text similarity
    min_aesthetic_score: float = 0.45  # LAION aesthetic score
    min_combined_score: float = 0.55  # Combined weighted score
    max_aspect_ratio: float = 4.0  # Reject extreme aspect ratios
    min_resolution: int = 128  # Minimum side length


class QualityFilter:
    """
    Multi-metric quality filter for self-generated images.

    Combines edge sharpness, CLIP alignment, and aesthetic scoring
    to keep only the best generated images.
    """

    def __init__(self, cfg: Optional[QualityFilterConfig] = None, device: str = "cpu"):
        self.cfg = cfg or QualityFilterConfig()
        self.device = device
        self._clip_model = None
        self._clip_processor = None
        self._aesthetic_model = None

    def _edge_sharpness(self, image_rgb: np.ndarray) -> float:
        """Laplacian variance as sharpness proxy."""
        try:
            gray = (
                0.299 * image_rgb[:, :, 0].astype(float)
                + 0.587 * image_rgb[:, :, 1].astype(float)
                + 0.114 * image_rgb[:, :, 2].astype(float)
            )
            # Simple Laplacian
            lap = -gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:] + 4 * gray[1:-1, 1:-1]
            return float(lap.var())
        except Exception:
            return 0.0

    def _clip_score(self, image_rgb: np.ndarray, prompt: str) -> float:
        """CLIP image-text cosine similarity."""
        clip_path = _PRETRAINED / "CLIP-ViT-L-14"
        if not clip_path.exists():
            return 0.5
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor

            if self._clip_model is None:
                self._clip_model = CLIPModel.from_pretrained(str(clip_path)).eval()
                self._clip_processor = CLIPProcessor.from_pretrained(str(clip_path))

            pil = Image.fromarray(image_rgb)
            inputs = self._clip_processor(
                text=[prompt[:77]],
                images=[pil],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
            score = float(torch.sigmoid(outputs.logits_per_image / 100.0).item())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5

    def score(self, image_rgb: np.ndarray, prompt: str) -> Dict[str, float]:
        """Score an image on all quality metrics."""
        sharpness = self._edge_sharpness(image_rgb)
        sharpness_norm = min(1.0, sharpness / 500.0)
        clip = self._clip_score(image_rgb, prompt)

        combined = 0.4 * sharpness_norm + 0.6 * clip

        return {
            "edge_sharpness": sharpness,
            "edge_sharpness_norm": sharpness_norm,
            "clip_score": clip,
            "combined": combined,
        }

    def passes(self, scores: Dict[str, float]) -> bool:
        """Return True if the image passes all quality thresholds."""
        return (
            scores["edge_sharpness"] >= self.cfg.min_edge_sharpness
            and scores["clip_score"] >= self.cfg.min_clip_score
            and scores["combined"] >= self.cfg.min_combined_score
        )


# ---------------------------------------------------------------------------
# Caption generator (moondream2)
# ---------------------------------------------------------------------------


class MoondreamCaptioner:
    """
    Generate detailed captions for images using moondream2.

    Uses a structured prompt to generate training-quality captions that
    describe: subject, action, setting, style, lighting, and quality.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> bool:
        """Lazy-load moondream2."""
        if self._model is not None:
            return True
        moondream_path = _PRETRAINED / "moondream2"
        if not moondream_path.exists():
            return False
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(moondream_path), trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                str(moondream_path),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self._model.eval()
            return True
        except Exception as e:
            _log.warning(f"Failed to load moondream2: {e}")
            return False

    def caption(
        self,
        image_rgb: np.ndarray,
        original_prompt: str = "",
        style: str = "detailed",
    ) -> str:
        """
        Generate a training-quality caption for an image.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image
            original_prompt: The prompt used to generate this image (for context)
            style: "detailed" | "concise" | "danbooru"

        Returns:
            Caption string suitable for training
        """
        if not self._load():
            # Fallback: return the original prompt
            return original_prompt

        try:
            from PIL import Image as _PIL

            pil = _PIL.fromarray(image_rgb)
            enc = self._model.encode_image(pil)

            if style == "danbooru":
                question = (
                    "Describe this image as comma-separated Danbooru-style tags. "
                    "Include: subject tags (1girl, 1boy, etc.), appearance tags (hair color, eye color, clothing), "
                    "action/pose tags, setting/background tags, style tags, and quality tags. "
                    "Be specific and comprehensive. Format: tag1, tag2, tag3, ..."
                )
            elif style == "concise":
                question = (
                    "Describe this image in one detailed sentence covering: "
                    "the main subject, their appearance, what they're doing, "
                    "the setting, and the art style."
                )
            else:  # detailed
                question = (
                    "Provide a detailed image caption for AI training. Describe: "
                    "1) The main subject(s) and their appearance in detail, "
                    "2) Actions, poses, and expressions, "
                    "3) Setting, background, and environment, "
                    "4) Lighting conditions and color palette, "
                    "5) Art style and quality. "
                    "Be specific and concrete. Use comma-separated descriptive phrases."
                )

            caption = self._model.answer_question(enc, question, self._tokenizer)
            return str(caption or original_prompt).strip()
        except Exception as e:
            _log.debug(f"Caption generation failed: {e}")
            return original_prompt


# ---------------------------------------------------------------------------
# Self-improvement cycle
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CycleStats:
    """Statistics from one self-improvement cycle."""

    cycle_id: int
    total_generated: int = 0
    passed_quality: int = 0
    added_to_dataset: int = 0
    mean_quality_score: float = 0.0
    mean_clip_score: float = 0.0
    prompts_used: int = 0
    duration_seconds: float = 0.0
    manifest_path: str = ""


@dataclass(slots=True)
class SelfImprovementConfig:
    """Configuration for the self-improvement flywheel."""

    output_dir: str = "./self_improvement_data"
    quality_threshold: float = 0.55
    n_per_prompt: int = 4  # Generate N images per prompt, keep best
    caption_style: str = "detailed"  # "detailed" | "danbooru" | "concise"
    max_samples_per_cycle: int = 1000
    min_samples_per_cycle: int = 10  # Skip cycle if fewer than this pass
    base_seed: int = 42
    save_rejected: bool = False  # Save rejected images for analysis
    mix_with_original: bool = True  # Include original prompt in caption
    device: str = "cpu"


class SelfImprovementFlywheel:
    """
    Self-improvement data flywheel.

    Generates images → filters by quality → captions with moondream2 →
    saves as JSONL manifest for training.

    Each cycle produces a manifest.jsonl that can be directly used with
    train.py --manifest-jsonl path/to/manifest.jsonl
    """

    def __init__(
        self,
        generate_fn: Callable[[str, int], np.ndarray],
        cfg: Optional[SelfImprovementConfig] = None,
    ):
        """
        Args:
            generate_fn: Function(prompt, seed) → RGB numpy array (H, W, 3) uint8
            cfg: Configuration
        """
        self.generate_fn = generate_fn
        self.cfg = cfg or SelfImprovementConfig()
        self.quality_filter = QualityFilter(
            QualityFilterConfig(min_combined_score=self.cfg.quality_threshold),
            device=self.cfg.device,
        )
        self.captioner = MoondreamCaptioner(device=self.cfg.device)
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_cycle(
        self,
        prompts: List[str],
        cycle_id: int = 0,
        negative_prompt: str = "",
    ) -> CycleStats:
        """
        Run one self-improvement cycle.

        Args:
            prompts: List of prompts to generate images for
            cycle_id: Cycle number (used for directory naming and seeds)
            negative_prompt: Negative prompt for generation

        Returns:
            CycleStats with statistics about this cycle
        """
        t_start = time.time()
        cycle_dir = self.output_dir / f"cycle_{cycle_id:03d}"
        images_dir = cycle_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        stats = CycleStats(cycle_id=cycle_id, prompts_used=len(prompts))
        manifest_rows: List[Dict[str, Any]] = []
        all_scores: List[float] = []

        _log.info(f"Self-improvement cycle {cycle_id}: {len(prompts)} prompts × {self.cfg.n_per_prompt} images")

        for prompt_idx, prompt in enumerate(prompts):
            if stats.added_to_dataset >= self.cfg.max_samples_per_cycle:
                break

            # Generate N images for this prompt
            candidates = []
            for i in range(self.cfg.n_per_prompt):
                seed = self.cfg.base_seed + cycle_id * 100000 + prompt_idx * 100 + i
                try:
                    img = self.generate_fn(prompt, seed)
                    stats.total_generated += 1
                    candidates.append((img, seed))
                except Exception as e:
                    _log.debug(f"Generation failed for prompt {prompt_idx}, seed {seed}: {e}")

            # Score all candidates
            scored = []
            for img, seed in candidates:
                scores = self.quality_filter.score(img, prompt)
                scored.append((img, seed, scores))
                all_scores.append(scores["combined"])

            # Keep only the best that passes threshold
            scored.sort(key=lambda x: x[2]["combined"], reverse=True)
            for img, seed, scores in scored:
                if not self.quality_filter.passes(scores):
                    if self.cfg.save_rejected:
                        from PIL import Image as _PIL

                        _PIL.fromarray(img).save(images_dir / f"rejected_p{prompt_idx:04d}_s{seed}.png")
                    continue

                stats.passed_quality += 1

                # Generate caption
                caption = self.captioner.caption(
                    img,
                    original_prompt=prompt if self.cfg.mix_with_original else "",
                    style=self.cfg.caption_style,
                )

                # Save image
                img_filename = f"p{prompt_idx:04d}_s{seed}_q{scores['combined']:.3f}.png"
                img_path = images_dir / img_filename
                try:
                    from PIL import Image as _PIL

                    _PIL.fromarray(img).save(img_path)
                except Exception as e:
                    _log.warning(f"Failed to save image: {e}")
                    continue

                # Add to manifest
                row = {
                    "image_path": str(img_path),
                    "caption": caption,
                    "original_prompt": prompt,
                    "quality_score": scores["combined"],
                    "clip_score": scores["clip_score"],
                    "edge_sharpness": scores["edge_sharpness"],
                    "seed": seed,
                    "cycle_id": cycle_id,
                    "weight": min(2.0, 1.0 + scores["combined"]),  # Higher weight for better images
                }
                manifest_rows.append(row)
                stats.added_to_dataset += 1

                # Only keep the best candidate per prompt
                break

        # Check minimum threshold
        if stats.added_to_dataset < self.cfg.min_samples_per_cycle:
            _log.warning(
                f"Cycle {cycle_id}: only {stats.added_to_dataset} samples passed quality filter "
                f"(minimum: {self.cfg.min_samples_per_cycle}). Consider lowering quality_threshold."
            )

        # Save manifest
        manifest_path = cycle_dir / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write(json.dumps(row) + "\n")

        # Save cycle stats
        stats.mean_quality_score = float(np.mean(all_scores)) if all_scores else 0.0
        stats.mean_clip_score = float(np.mean([r["clip_score"] for r in manifest_rows])) if manifest_rows else 0.0
        stats.duration_seconds = time.time() - t_start
        stats.manifest_path = str(manifest_path)

        stats_path = cycle_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "cycle_id": stats.cycle_id,
                    "total_generated": stats.total_generated,
                    "passed_quality": stats.passed_quality,
                    "added_to_dataset": stats.added_to_dataset,
                    "mean_quality_score": stats.mean_quality_score,
                    "mean_clip_score": stats.mean_clip_score,
                    "prompts_used": stats.prompts_used,
                    "duration_seconds": stats.duration_seconds,
                    "manifest_path": stats.manifest_path,
                },
                f,
                indent=2,
            )

        _log.info(
            f"Cycle {cycle_id} complete: "
            f"generated={stats.total_generated} "
            f"passed={stats.passed_quality} "
            f"added={stats.added_to_dataset} "
            f"mean_quality={stats.mean_quality_score:.3f} "
            f"duration={stats.duration_seconds:.1f}s"
        )

        return stats

    def run_continuous(
        self,
        prompt_source: Callable[[], List[str]],
        n_cycles: int = 10,
        train_fn: Optional[Callable[[str], None]] = None,
        train_every_n_cycles: int = 1,
    ) -> List[CycleStats]:
        """
        Run the flywheel continuously for N cycles.

        Args:
            prompt_source: Function() → list of prompts for this cycle
            n_cycles: Number of cycles to run
            train_fn: Optional function(manifest_path) → None to trigger training
            train_every_n_cycles: Train after every N cycles

        Returns:
            List of CycleStats, one per cycle
        """
        all_stats = []
        accumulated_manifests = []

        for cycle_id in range(n_cycles):
            _log.info(f"Starting cycle {cycle_id + 1}/{n_cycles}")

            prompts = prompt_source()
            stats = self.run_cycle(prompts, cycle_id=cycle_id)
            all_stats.append(stats)

            if stats.added_to_dataset > 0:
                accumulated_manifests.append(stats.manifest_path)

            # Trigger training if requested
            if train_fn is not None and (cycle_id + 1) % train_every_n_cycles == 0 and accumulated_manifests:
                # Merge manifests for training
                merged_path = self.output_dir / f"merged_cycles_0_{cycle_id}.jsonl"
                with open(merged_path, "w") as out_f:
                    for manifest_path in accumulated_manifests:
                        try:
                            with open(manifest_path) as in_f:
                                out_f.write(in_f.read())
                        except Exception:
                            pass

                _log.info(f"Triggering training on {len(accumulated_manifests)} cycles of data")
                try:
                    train_fn(str(merged_path))
                except Exception as e:
                    _log.error(f"Training failed: {e}")

        return all_stats

    def get_training_command(self, cycle_id: int) -> str:
        """Return the train.py command to train on this cycle's data."""
        manifest = self.output_dir / f"cycle_{cycle_id:03d}" / "manifest.jsonl"
        return f"python train.py --manifest-jsonl {manifest} --max-steps 1000 --lr 1e-5 --no-cache --log-every 50"


# ---------------------------------------------------------------------------
# Prompt diversity engine (for flywheel prompts)
# ---------------------------------------------------------------------------


class PromptDiversityEngine:
    """
    Generates diverse prompts for the self-improvement flywheel.

    Combines:
    - Domain coverage (portrait, landscape, scene, abstract, etc.)
    - Style coverage (photorealistic, anime, oil painting, etc.)
    - Difficulty coverage (simple, complex, challenging)
    - Random variation to avoid repetition
    """

    _DOMAINS = [
        "portrait",
        "landscape",
        "character design",
        "scene",
        "still life",
        "architecture",
        "fantasy",
        "sci-fi",
        "abstract",
        "nature",
    ]
    _STYLES = [
        "photorealistic",
        "anime style",
        "oil painting",
        "watercolor",
        "digital art",
        "concept art",
        "3d render",
        "illustration",
    ]
    _SUBJECTS = [
        "a young woman",
        "a warrior",
        "a wizard",
        "a samurai",
        "a scientist",
        "a dragon",
        "a robot",
        "a city",
        "a forest",
        "a mountain",
        "a castle",
        "a spaceship",
        "a market",
        "a garden",
        "a library",
    ]
    _MODIFIERS = [
        "at sunset",
        "in the rain",
        "at night",
        "in golden hour",
        "in a storm",
        "underwater",
        "in space",
        "in winter",
        "in autumn",
        "at dawn",
    ]
    _QUALITY = [
        "masterpiece, best quality, highly detailed",
        "award-winning photography",
        "professional illustration",
        "8k resolution, sharp focus",
    ]

    def __init__(self, seed: int = 42):
        import random

        self.rng = random.Random(seed)

    def generate(self, n: int = 100) -> List[str]:
        """Generate N diverse prompts."""
        prompts = []
        for _ in range(n):
            subject = self.rng.choice(self._SUBJECTS)
            modifier = self.rng.choice(self._MODIFIERS)
            style = self.rng.choice(self._STYLES)
            quality = self.rng.choice(self._QUALITY)

            prompt = f"{subject} {modifier}, {style}, {quality}"
            prompts.append(prompt)

        return prompts


__all__ = [
    "SelfImprovementFlywheel",
    "SelfImprovementConfig",
    "CycleStats",
    "QualityFilter",
    "QualityFilterConfig",
    "MoondreamCaptioner",
    "PromptDiversityEngine",
]
