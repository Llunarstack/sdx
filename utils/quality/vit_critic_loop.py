"""
In-loop ViT/CLIP critic for steering the denoising trajectory mid-generation.

Unlike post-hoc pick-best (which scores finished images), this module applies
a lightweight critic DURING the denoising loop to:

1. Detect when the generation is going wrong early (wrong composition, missing subject)
2. Apply a latent-space correction to steer back toward the prompt
3. Boost CFG dynamically when alignment drops
4. Optionally rewind to a saved checkpoint and branch

Uses pretrained models from pretrained/:
- pretrained/DINOv2-Large: rich visual features, good for composition/structure
- pretrained/SigLIP-SO400M: strong image-text alignment, good for prompt adherence
- pretrained/CLIP-ViT-L-14: fast, widely compatible

All models are loaded lazily and cached. If none are available, the module
degrades gracefully to a no-op.

Integration with sample.py:
    critic = ViTCriticLoop(device=device, vae=vae, latent_scale=latent_scale)
    # In the sampling loop callback:
    def step_callback(step_i, x0_pred, t_current):
        action = critic.evaluate(x0_pred, prompt, step_i, total_steps)
        if action.should_boost_cfg:
            cfg_scale *= action.cfg_multiplier
        if action.should_rewind:
            return saved_latent  # rewind
        return x0_pred

Why this is better than CLIP guard (which already exists):
- CLIP guard runs ONCE after generation and does a full re-denoise
- This runs EVERY N steps and applies LIGHTWEIGHT corrections
- Catches problems at step 5-10 when they're cheap to fix, not step 50
- Uses DINOv2 (structural) + SigLIP (semantic) for richer signal than CLIP alone
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_PRETRAINED = Path(__file__).resolve().parents[2] / "pretrained"
_DINOV2_PATH = _PRETRAINED / "DINOv2-Large"
_SIGLIP_PATH = _PRETRAINED / "SigLIP-SO400M"
_CLIP_L_PATH = _PRETRAINED / "CLIP-ViT-L-14"


# ---------------------------------------------------------------------------
# Critic action: what to do based on the evaluation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CriticAction:
    """Action recommended by the critic after evaluating a denoising step."""

    step: int
    alignment_score: float  # 0-1: how well x0_pred matches the prompt
    structure_score: float  # 0-1: how coherent the structure is
    combined_score: float  # weighted combination

    should_boost_cfg: bool = False  # boost CFG on next steps
    cfg_multiplier: float = 1.0  # how much to boost (e.g. 1.15)

    should_rewind: bool = False  # rewind to a saved checkpoint
    rewind_to_step: int = -1  # which step to rewind to

    should_inject_noise: bool = False  # add small noise to escape local minimum
    noise_scale: float = 0.0

    details: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------


class _ModelCache:
    """Lazy-loads and caches vision models."""

    def __init__(self):
        self._dinov2 = None
        self._siglip = None
        self._clip = None
        self._clip_processor = None
        self._siglip_processor = None

    def get_dinov2(self, device: torch.device) -> Optional[Any]:
        if self._dinov2 is not None:
            return self._dinov2
        if not _DINOV2_PATH.exists():
            return None
        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(str(_DINOV2_PATH))
            model = model.to(device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._dinov2 = model
            return model
        except Exception:
            return None

    def get_siglip(self, device: torch.device) -> Tuple[Optional[Any], Optional[Any]]:
        if self._siglip is not None:
            return self._siglip, self._siglip_processor
        if not _SIGLIP_PATH.exists():
            return None, None
        try:
            from transformers import AutoModel, AutoProcessor

            model = AutoModel.from_pretrained(str(_SIGLIP_PATH))
            model = model.to(device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            processor = AutoProcessor.from_pretrained(str(_SIGLIP_PATH))
            self._siglip = model
            self._siglip_processor = processor
            return model, processor
        except Exception:
            return None, None

    def get_clip(self, device: torch.device) -> Tuple[Optional[Any], Optional[Any]]:
        if self._clip is not None:
            return self._clip, self._clip_processor
        if not _CLIP_L_PATH.exists():
            return None, None
        try:
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(str(_CLIP_L_PATH))
            model = model.to(device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            processor = CLIPProcessor.from_pretrained(str(_CLIP_L_PATH))
            self._clip = model
            self._clip_processor = processor
            return model, processor
        except Exception:
            return None, None


_GLOBAL_CACHE = _ModelCache()


# ---------------------------------------------------------------------------
# Latent → image decoder (fast approximate decode)
# ---------------------------------------------------------------------------


def _decode_latent_fast(
    latent: torch.Tensor,
    vae: Any,
    latent_scale: float,
    ae_type: str = "kl",
    rae_bridge: Any = None,
    max_size: int = 224,
) -> Optional[np.ndarray]:
    """
    Fast approximate decode of a latent to a small RGB image for critic scoring.
    Decodes only the first sample in the batch.
    """
    try:
        with torch.no_grad():
            z = latent[:1].float()
            if ae_type == "kl":
                z = z / latent_scale
            elif ae_type == "rae" and rae_bridge is not None:
                z = rae_bridge.dit_to_rae(z)

            decoded = vae.decode(z).sample
            decoded = (decoded * 0.5 + 0.5).clamp(0, 1)

            # Resize to small size for fast scoring
            if decoded.shape[-1] > max_size or decoded.shape[-2] > max_size:
                decoded = F.interpolate(decoded, size=(max_size, max_size), mode="bilinear", align_corners=False)

            img = decoded[0].permute(1, 2, 0).cpu().numpy()
            return (img * 255).round().astype(np.uint8)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------


@torch.no_grad()
def _score_clip_alignment(
    image_rgb: np.ndarray,
    prompt: str,
    device: torch.device,
) -> float:
    """CLIP image-text cosine similarity."""
    clip_model, clip_processor = _GLOBAL_CACHE.get_clip(device)
    if clip_model is None or clip_processor is None:
        return 0.5

    try:
        from PIL import Image

        pil = Image.fromarray(image_rgb)
        inputs = clip_processor(
            text=[prompt[:77]],
            images=[pil],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image
        # Convert logits to similarity score [0, 1]
        score = float(torch.sigmoid(logits / 100.0).item())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


@torch.no_grad()
def _score_siglip_alignment(
    image_rgb: np.ndarray,
    prompt: str,
    device: torch.device,
) -> float:
    """SigLIP image-text alignment (better calibrated than CLIP for generation)."""
    siglip_model, siglip_processor = _GLOBAL_CACHE.get_siglip(device)
    if siglip_model is None or siglip_processor is None:
        return 0.5

    try:
        from PIL import Image

        pil = Image.fromarray(image_rgb)
        inputs = siglip_processor(
            text=[prompt[:64]],
            images=[pil],
            return_tensors="pt",
            padding="max_length",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = siglip_model(**inputs)
        logits = outputs.logits_per_image
        score = float(torch.sigmoid(logits).item())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


@torch.no_grad()
def _score_dinov2_structure(
    image_rgb: np.ndarray,
    device: torch.device,
) -> float:
    """
    DINOv2 structural coherence score.

    Uses the CLS token norm as a proxy for structural coherence:
    - High norm = rich, structured features = coherent image
    - Low norm = weak features = incoherent/noisy image

    This is a heuristic but works well in practice.
    """
    dinov2 = _GLOBAL_CACHE.get_dinov2(device)
    if dinov2 is None:
        return 0.5

    try:
        import torchvision.transforms as T
        from PIL import Image

        pil = Image.fromarray(image_rgb).resize((224, 224))
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_t = transform(pil).unsqueeze(0).to(device)

        outputs = dinov2(pixel_values=img_t)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        norm = float(cls_token.norm(dim=-1).item())

        # Normalize: typical range is 20-80 for coherent images
        score = min(1.0, max(0.0, (norm - 10.0) / 60.0))
        return score
    except Exception:
        return 0.5


def _score_latent_coherence(latent: torch.Tensor) -> float:
    """
    Fast latent-space coherence score (no decode needed).

    Measures:
    1. Spatial frequency distribution (coherent images have structured frequencies)
    2. Channel correlation (coherent images have correlated channels)
    3. Absence of extreme values (no NaN/Inf, reasonable range)
    """
    try:
        with torch.no_grad():
            z = latent.float()

            # Check for NaN/Inf
            if not torch.isfinite(z).all():
                return 0.0

            # Spatial frequency: coherent images have more low-frequency energy
            fft = torch.fft.rfft2(z, dim=(-2, -1))
            power = fft.abs().pow(2)
            h, w = z.shape[-2], z.shape[-1]
            # Low-frequency region (center 25%)
            lf_h, lf_w = max(1, h // 4), max(1, w // 4)
            lf_power = power[..., :lf_h, :lf_w].mean()
            total_power = power.mean().clamp(min=1e-8)
            freq_score = float((lf_power / total_power).clamp(0, 1).item())

            # Value range: should be roughly [-4, 4] for well-trained VAE latents
            std = float(z.std().item())
            range_score = float(torch.exp(-torch.tensor(max(0.0, std - 3.0))).item())

            return 0.6 * freq_score + 0.4 * range_score
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Main critic class
# ---------------------------------------------------------------------------


class ViTCriticLoop:
    """
    In-loop ViT/CLIP critic for steering the denoising trajectory.

    Evaluates the current x0 prediction every N steps and recommends
    actions (CFG boost, rewind, noise injection) to improve alignment.

    Usage:
        critic = ViTCriticLoop(
            device=device,
            vae=vae,
            latent_scale=latent_scale,
            eval_every=5,           # evaluate every 5 steps
            alignment_threshold=0.3, # boost CFG if below this
            rewind_threshold=0.15,  # rewind if below this
        )

        # In your sampling loop:
        for step_i, (x0_pred, t) in enumerate(denoising_steps):
            action = critic.step(step_i, total_steps, x0_pred, prompt)
            if action.should_boost_cfg:
                cfg_scale = cfg_scale * action.cfg_multiplier
    """

    def __init__(
        self,
        device: torch.device,
        vae: Any = None,
        latent_scale: float = 0.18215,
        ae_type: str = "kl",
        rae_bridge: Any = None,
        eval_every: int = 5,
        alignment_threshold: float = 0.28,
        rewind_threshold: float = 0.15,
        cfg_boost_factor: float = 1.15,
        max_cfg_boost: float = 1.5,
        use_dinov2: bool = True,
        use_siglip: bool = True,
        use_clip: bool = True,
        use_latent_coherence: bool = True,
        min_step_for_eval: int = 3,
        max_step_for_rewind: float = 0.4,  # fraction of total steps
        history_window: int = 3,
    ):
        self.device = device
        self.vae = vae
        self.latent_scale = float(latent_scale)
        self.ae_type = str(ae_type)
        self.rae_bridge = rae_bridge
        self.eval_every = int(eval_every)
        self.alignment_threshold = float(alignment_threshold)
        self.rewind_threshold = float(rewind_threshold)
        self.cfg_boost_factor = float(cfg_boost_factor)
        self.max_cfg_boost = float(max_cfg_boost)
        self.use_dinov2 = use_dinov2
        self.use_siglip = use_siglip
        self.use_clip = use_clip
        self.use_latent_coherence = use_latent_coherence
        self.min_step_for_eval = int(min_step_for_eval)
        self.max_step_for_rewind = float(max_step_for_rewind)
        self.history_window = int(history_window)

        # State
        self._score_history: List[float] = []
        self._saved_checkpoints: Dict[int, torch.Tensor] = {}
        self._current_cfg_multiplier: float = 1.0
        self._total_boosts: int = 0

    def reset(self) -> None:
        """Reset state for a new generation."""
        self._score_history.clear()
        self._saved_checkpoints.clear()
        self._current_cfg_multiplier = 1.0
        self._total_boosts = 0

    def save_checkpoint(self, step: int, latent: torch.Tensor) -> None:
        """Save a latent checkpoint for potential rewind."""
        # Keep only last 3 checkpoints to save memory
        if len(self._saved_checkpoints) >= 3:
            oldest = min(self._saved_checkpoints.keys())
            del self._saved_checkpoints[oldest]
        self._saved_checkpoints[step] = latent.detach().cpu()

    def get_checkpoint(self, step: int) -> Optional[torch.Tensor]:
        """Retrieve a saved checkpoint."""
        ckpt = self._saved_checkpoints.get(step)
        if ckpt is not None:
            return ckpt.to(self.device)
        return None

    def step(
        self,
        step_i: int,
        total_steps: int,
        x0_pred: torch.Tensor,
        prompt: str,
        current_latent: Optional[torch.Tensor] = None,
    ) -> CriticAction:
        """
        Evaluate the current denoising step and recommend an action.

        Args:
            step_i: Current step index (0-based)
            total_steps: Total number of denoising steps
            x0_pred: Current x0 prediction (B, C, H, W)
            prompt: The text prompt
            current_latent: Current noisy latent (for checkpointing)

        Returns:
            CriticAction with recommendations
        """
        # Save checkpoint periodically for potential rewind
        if current_latent is not None and step_i % (self.eval_every * 2) == 0:
            self.save_checkpoint(step_i, current_latent)

        # Only evaluate every N steps and after min_step
        if step_i < self.min_step_for_eval or step_i % self.eval_every != 0:
            return CriticAction(
                step=step_i,
                alignment_score=0.5,
                structure_score=0.5,
                combined_score=0.5,
            )

        # Decode latent to image for scoring
        image_rgb = None
        if self.vae is not None:
            image_rgb = _decode_latent_fast(x0_pred, self.vae, self.latent_scale, self.ae_type, self.rae_bridge)

        # Compute scores
        scores = {}

        # Latent coherence (always available, no decode needed)
        if self.use_latent_coherence:
            scores["latent_coherence"] = _score_latent_coherence(x0_pred)

        if image_rgb is not None:
            # CLIP alignment
            if self.use_clip:
                scores["clip"] = _score_clip_alignment(image_rgb, prompt, self.device)

            # SigLIP alignment (better calibrated)
            if self.use_siglip:
                scores["siglip"] = _score_siglip_alignment(image_rgb, prompt, self.device)

            # DINOv2 structure
            if self.use_dinov2:
                scores["dinov2"] = _score_dinov2_structure(image_rgb, self.device)

        # Compute combined score
        weights = {
            "siglip": 0.35,
            "clip": 0.25,
            "dinov2": 0.25,
            "latent_coherence": 0.15,
        }
        total_weight = sum(weights[k] for k in scores if k in weights)
        if total_weight > 0:
            combined = sum(scores[k] * weights[k] for k in scores if k in weights) / total_weight
        else:
            combined = 0.5

        alignment = scores.get("siglip", scores.get("clip", 0.5))
        structure = scores.get("dinov2", scores.get("latent_coherence", 0.5))

        # Update history
        self._score_history.append(combined)
        if len(self._score_history) > self.history_window:
            self._score_history.pop(0)

        # Determine action
        action = CriticAction(
            step=step_i,
            alignment_score=alignment,
            structure_score=structure,
            combined_score=combined,
            details=scores,
        )

        step_frac = step_i / max(total_steps - 1, 1)

        # Check for rewind (only in early steps)
        if (
            combined < self.rewind_threshold
            and step_frac < self.max_step_for_rewind
            and len(self._saved_checkpoints) > 0
        ):
            # Find the best checkpoint to rewind to
            rewind_step = max(s for s in self._saved_checkpoints if s < step_i)
            action.should_rewind = True
            action.rewind_to_step = rewind_step

        # Check for CFG boost
        elif (
            combined < self.alignment_threshold
            and self._current_cfg_multiplier < self.max_cfg_boost
            and self._total_boosts < 3  # limit total boosts
        ):
            # Boost more aggressively if score is very low
            boost = self.cfg_boost_factor
            if combined < self.alignment_threshold * 0.6:
                boost = min(self.cfg_boost_factor * 1.3, self.max_cfg_boost)

            action.should_boost_cfg = True
            action.cfg_multiplier = boost
            self._current_cfg_multiplier *= boost
            self._total_boosts += 1

        # Check for noise injection (if score is stagnating)
        elif (
            len(self._score_history) >= self.history_window
            and max(self._score_history) - min(self._score_history) < 0.02
            and combined < 0.4
        ):
            # Score is stagnating at a low value — inject small noise to escape
            action.should_inject_noise = True
            action.noise_scale = 0.05

        return action

    def apply_action(
        self,
        action: CriticAction,
        current_latent: torch.Tensor,
        cfg_scale: float,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply the recommended action to the current latent and CFG scale.

        Returns:
            (modified_latent, new_cfg_scale)
        """
        new_latent = current_latent
        new_cfg = cfg_scale

        if action.should_rewind and action.rewind_to_step >= 0:
            ckpt = self.get_checkpoint(action.rewind_to_step)
            if ckpt is not None:
                new_latent = ckpt
                # Reset CFG multiplier after rewind
                self._current_cfg_multiplier = 1.0
                self._total_boosts = 0

        elif action.should_boost_cfg:
            new_cfg = cfg_scale * action.cfg_multiplier

        elif action.should_inject_noise:
            noise = torch.randn_like(current_latent) * action.noise_scale
            new_latent = current_latent + noise

        return new_latent, new_cfg


# ---------------------------------------------------------------------------
# Trajectory monitor: track score evolution across the full generation
# ---------------------------------------------------------------------------


class TrajectoryMonitor:
    """
    Monitors the full denoising trajectory and provides post-hoc analysis.

    Useful for:
    - Understanding why a generation failed
    - Identifying the optimal number of steps for a given prompt
    - Building training data for better critics
    """

    def __init__(self):
        self.steps: List[int] = []
        self.scores: List[float] = []
        self.actions: List[CriticAction] = []

    def record(self, action: CriticAction) -> None:
        self.steps.append(action.step)
        self.scores.append(action.combined_score)
        self.actions.append(action)

    def summary(self) -> Dict[str, Any]:
        if not self.scores:
            return {}
        return {
            "final_score": self.scores[-1] if self.scores else 0.0,
            "peak_score": max(self.scores),
            "min_score": min(self.scores),
            "mean_score": sum(self.scores) / len(self.scores),
            "total_boosts": sum(1 for a in self.actions if a.should_boost_cfg),
            "total_rewinds": sum(1 for a in self.actions if a.should_rewind),
            "score_trajectory": list(zip(self.steps, self.scores)),
        }

    def optimal_early_stop_step(self, threshold: float = 0.6) -> Optional[int]:
        """Find the earliest step where score exceeded threshold."""
        for step, score in zip(self.steps, self.scores):
            if score >= threshold:
                return step
        return None


__all__ = [
    "ViTCriticLoop",
    "CriticAction",
    "TrajectoryMonitor",
]
