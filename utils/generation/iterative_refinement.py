"""
Iterative refinement session manager for multi-turn artistic collaboration.

Enables treating the model as an art director you can give corrections to:
    "make the lighting more dramatic"
    "the face looks off, fix it"
    "swap the red dress for a blue one"
    "add more depth to the background"

Architecture:
- RefinementSession: maintains state across turns (latent, prompt history, edits)
- EditInstruction: structured representation of a user edit request
- EditRouter: classifies edit instructions and routes to the right strategy
- PromptDeltaEngine: computes prompt changes from natural language instructions
- LatentEditStrategy: applies edits at the latent level (img2img, inpaint, etc.)

This is a pure Python/inference-time system — no retraining required.
It wraps the existing sample.py infrastructure.

Usage:
    session = RefinementSession.from_prompt(
        "a lone samurai at sunset",
        checkpoint_path="results/best.pt",
        device="cuda",
    )
    result = session.generate()
    result.save("output_v1.png")

    # User gives feedback
    result2 = session.refine("make the lighting more dramatic, add rim light")
    result2.save("output_v2.png")

    result3 = session.refine("the face looks too smooth, add more texture")
    result3.save("output_v3.png")

    session.save_history("session.json")
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edit instruction types
# ---------------------------------------------------------------------------

EDIT_TYPES = {
    "lighting": [
        r"\b(light|lighting|shadow|shadows|bright|dark|dramatic|rim light|backlight|"
        r"golden hour|sunset|sunrise|moody|atmospheric|glow|luminous|dim|harsh|soft light)\b"
    ],
    "color": [
        r"\b(color|colour|hue|saturation|palette|warm|cool|tint|tone|vibrant|muted|"
        r"red|blue|green|yellow|orange|purple|pink|black|white|gray)\b"
    ],
    "style": [
        r"\b(style|artistic|painterly|realistic|anime|sketch|render|3d|illustration|"
        r"watercolor|oil painting|digital art|concept art)\b"
    ],
    "composition": [
        r"\b(composition|framing|angle|perspective|zoom|close.?up|wide|crop|"
        r"rule of thirds|center|off.?center|portrait|landscape)\b"
    ],
    "subject": [
        r"\b(face|hair|eyes|expression|pose|body|hands|clothing|outfit|dress|"
        r"swap|change|replace|add|remove|fix)\b"
    ],
    "detail": [
        r"\b(detail|texture|sharp|blur|focus|quality|resolution|noise|grain|"
        r"smooth|rough|fine|coarse|crisp|soft)\b"
    ],
    "background": [
        r"\b(background|environment|setting|scene|sky|ground|floor|wall|"
        r"depth|foreground|bokeh|blur background)\b"
    ],
    "anatomy": [
        r"\b(anatomy|proportion|limb|arm|leg|hand|finger|face|head|body|"
        r"deformed|wrong|fix|correct|realistic anatomy)\b"
    ],
}


@dataclass(slots=True)
class EditInstruction:
    """A parsed edit instruction from the user."""

    raw_text: str
    edit_type: str  # one of EDIT_TYPES keys
    confidence: float  # 0-1
    prompt_additions: List[str] = field(default_factory=list)
    prompt_removals: List[str] = field(default_factory=list)
    negative_additions: List[str] = field(default_factory=list)
    strength: float = 0.5  # img2img strength for this edit
    use_inpaint: bool = False  # whether to use inpainting
    inpaint_region: Optional[str] = None  # "face", "background", "full"


@dataclass(slots=True)
class GenerationResult:
    """Result of one generation or refinement step."""

    image_np: Any  # (H, W, 3) uint8 numpy array
    prompt: str
    negative_prompt: str
    seed: int
    step: int
    edit_instruction: Optional[EditInstruction] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save the image to disk."""
        import numpy as np
        from PIL import Image

        img = np.asarray(self.image_np, dtype=np.uint8)
        Image.fromarray(img).save(path)
        print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Edit router: classify and parse edit instructions
# ---------------------------------------------------------------------------


class EditRouter:
    """
    Classifies natural language edit instructions and routes them to the
    appropriate refinement strategy.
    """

    # Strength hints from language
    _STRENGTH_HINTS = {
        "subtle": 0.25,
        "slightly": 0.25,
        "a bit": 0.3,
        "little": 0.3,
        "more": 0.5,
        "stronger": 0.6,
        "much more": 0.65,
        "dramatically": 0.7,
        "completely": 0.8,
        "totally": 0.8,
        "entirely": 0.8,
    }

    # Inpaint region hints
    _INPAINT_REGIONS = {
        "face": ["face", "eyes", "nose", "mouth", "expression", "skin"],
        "background": ["background", "sky", "environment", "setting", "scene"],
        "hands": ["hands", "fingers", "hand"],
        "clothing": ["clothing", "outfit", "dress", "shirt", "pants", "jacket"],
    }

    def classify(self, instruction: str) -> EditInstruction:
        """Parse a natural language instruction into an EditInstruction."""
        text = instruction.strip()
        text_lower = text.lower()

        # Classify edit type
        best_type = "detail"
        best_score = 0
        for etype, patterns in EDIT_TYPES.items():
            score = sum(len(re.findall(p, text_lower, re.IGNORECASE)) for p in patterns)
            if score > best_score:
                best_score = score
                best_type = etype

        confidence = min(1.0, best_score / 3.0)

        # Estimate strength from language
        strength = 0.45  # default
        for hint, s in self._STRENGTH_HINTS.items():
            if hint in text_lower:
                strength = s
                break

        # Check for inpainting hints
        use_inpaint = False
        inpaint_region = None
        for region, keywords in self._INPAINT_REGIONS.items():
            if any(kw in text_lower for kw in keywords):
                use_inpaint = True
                inpaint_region = region
                break

        # Generate prompt additions/removals
        additions, removals, neg_additions = self._extract_prompt_changes(text, best_type)

        return EditInstruction(
            raw_text=text,
            edit_type=best_type,
            confidence=confidence,
            prompt_additions=additions,
            prompt_removals=removals,
            negative_additions=neg_additions,
            strength=strength,
            use_inpaint=use_inpaint,
            inpaint_region=inpaint_region,
        )

    def _extract_prompt_changes(
        self,
        text: str,
        edit_type: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Extract prompt additions, removals, and negative additions from instruction."""
        additions: List[str] = []
        removals: List[str] = []
        neg_additions: List[str] = []
        t = text.lower()

        # Lighting edits
        if edit_type == "lighting":
            if any(w in t for w in ["dramatic", "moody", "dark"]):
                additions.extend(["dramatic lighting", "chiaroscuro", "deep shadows"])
                neg_additions.append("flat lighting")
            if any(w in t for w in ["rim", "backlight"]):
                additions.extend(["rim lighting", "backlit", "edge light"])
            if any(w in t for w in ["soft", "gentle", "natural"]):
                additions.extend(["soft natural lighting", "diffused light"])
                neg_additions.append("harsh lighting")
            if any(w in t for w in ["golden", "warm", "sunset"]):
                additions.extend(["golden hour lighting", "warm light", "amber glow"])
            if any(w in t for w in ["bright", "luminous", "glowing"]):
                additions.extend(["bright lighting", "luminous", "well-lit"])

        # Color edits
        elif edit_type == "color":
            if any(w in t for w in ["vibrant", "saturated", "vivid"]):
                additions.extend(["vibrant colors", "saturated palette"])
                neg_additions.append("desaturated")
            if any(w in t for w in ["muted", "desaturated", "subtle"]):
                additions.extend(["muted palette", "desaturated colors"])
            if any(w in t for w in ["warm", "orange", "golden"]):
                additions.extend(["warm color palette", "warm tones"])
                neg_additions.append("cool tones")
            if any(w in t for w in ["cool", "blue", "cold"]):
                additions.extend(["cool color palette", "cool tones"])

        # Style edits
        elif edit_type == "style":
            if "painterly" in t:
                additions.extend(["painterly style", "visible brushstrokes", "artistic"])
            if "realistic" in t:
                additions.extend(["photorealistic", "detailed", "sharp focus"])
            if "anime" in t:
                additions.extend(["anime style", "clean lines"])

        # Detail edits
        elif edit_type == "detail":
            if any(w in t for w in ["more detail", "sharper", "crisp"]):
                additions.extend(["highly detailed", "sharp focus", "fine detail"])
                neg_additions.extend(["blurry", "soft focus"])
            if any(w in t for w in ["texture", "rough", "surface"]):
                additions.extend(["detailed texture", "surface detail", "tactile"])
            if any(w in t for w in ["smooth", "clean", "polished"]):
                additions.extend(["smooth", "clean", "polished"])
                neg_additions.append("rough texture")

        # Subject/anatomy edits
        elif edit_type in ("subject", "anatomy"):
            if any(w in t for w in ["fix", "correct", "better"]):
                additions.extend(["correct anatomy", "natural proportions"])
                neg_additions.extend(["deformed", "bad anatomy", "wrong proportions"])
            if "face" in t:
                additions.extend(["detailed face", "natural expression"])
                neg_additions.extend(["blurry face", "deformed face"])
            if "hands" in t:
                additions.extend(["correct hands", "five fingers", "natural hands"])
                neg_additions.extend(["bad hands", "extra fingers", "deformed hands"])

        # Background edits
        elif edit_type == "background":
            if any(w in t for w in ["depth", "bokeh", "blur"]):
                additions.extend(["depth of field", "bokeh background", "blurred background"])
            if any(w in t for w in ["detailed", "rich", "complex"]):
                additions.extend(["detailed background", "rich environment"])
                neg_additions.append("plain background")

        return additions, removals, neg_additions


# ---------------------------------------------------------------------------
# Prompt delta engine
# ---------------------------------------------------------------------------


class PromptDeltaEngine:
    """
    Computes the new prompt from the current prompt + an edit instruction.

    Handles:
    - Adding new elements
    - Removing contradicted elements
    - Resolving conflicts between old and new elements
    """

    def apply(
        self,
        current_prompt: str,
        current_negative: str,
        edit: EditInstruction,
    ) -> Tuple[str, str]:
        """
        Apply an edit instruction to the current prompt.

        Returns (new_prompt, new_negative).
        """
        parts = [p.strip() for p in current_prompt.split(",") if p.strip()]
        neg_parts = [p.strip() for p in current_negative.split(",") if p.strip()]

        # Remove elements that conflict with additions
        for removal in edit.prompt_removals:
            parts = [p for p in parts if removal.lower() not in p.lower()]

        # Remove elements that conflict with additions (auto-detect)
        for addition in edit.prompt_additions:
            # Remove obvious contradictions
            parts = self._remove_contradictions(parts, addition)

        # Add new elements (avoid duplicates)
        existing_lower = {p.lower() for p in parts}
        for addition in edit.prompt_additions:
            if addition.lower() not in existing_lower:
                parts.append(addition)
                existing_lower.add(addition.lower())

        # Add negative elements
        neg_existing = {p.lower() for p in neg_parts}
        for neg in edit.negative_additions:
            if neg.lower() not in neg_existing:
                neg_parts.append(neg)
                neg_existing.add(neg.lower())

        return ", ".join(parts), ", ".join(neg_parts)

    def _remove_contradictions(self, parts: List[str], addition: str) -> List[str]:
        """Remove prompt elements that contradict the addition."""
        # Simple contradiction pairs
        contradictions = {
            "dramatic lighting": ["flat lighting", "even lighting", "soft light"],
            "soft natural lighting": ["harsh lighting", "dramatic lighting"],
            "warm color palette": ["cool tones", "cold colors"],
            "cool color palette": ["warm tones", "warm colors"],
            "vibrant colors": ["desaturated", "muted"],
            "muted palette": ["vibrant", "saturated"],
            "photorealistic": ["anime", "cartoon", "illustration"],
            "anime style": ["photorealistic", "realistic"],
            "blurred background": ["sharp background", "detailed background"],
            "detailed background": ["blurred background", "bokeh"],
        }
        to_remove = contradictions.get(addition.lower(), [])
        return [p for p in parts if not any(r.lower() in p.lower() for r in to_remove)]


# ---------------------------------------------------------------------------
# Refinement session
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SessionConfig:
    """Configuration for a refinement session."""

    checkpoint_path: str
    device: str = "cuda"
    steps: int = 50
    cfg_scale: float = 7.5
    width: int = 0
    height: int = 0
    seed: int = 42
    scheduler: str = "ddim"
    solver: str = "ddim"
    inpaint_mode: str = "mdm"
    # Refinement defaults
    default_refine_strength: float = 0.45
    max_refine_strength: float = 0.75
    # Post-processing
    use_artistic_post: bool = False
    artistic_post_config: Optional[Any] = None


class RefinementSession:
    """
    Multi-turn image generation session.

    Maintains the full history of prompts, edits, and generated images
    so the user can iterate toward their vision.
    """

    def __init__(
        self,
        prompt: str,
        config: SessionConfig,
        negative_prompt: str = "",
    ) -> None:
        self.config = config
        self.router = EditRouter()
        self.delta_engine = PromptDeltaEngine()

        self.current_prompt = prompt
        self.current_negative = negative_prompt
        self.current_image_np: Optional[Any] = None
        self.current_latent_path: Optional[str] = None

        self.history: List[GenerationResult] = []
        self.step = 0

        # Temp dir for intermediate latents
        self._tmp_dir = Path(f"_session_{int(time.time())}")
        self._tmp_dir.mkdir(exist_ok=True)

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        checkpoint_path: str,
        *,
        device: str = "cuda",
        negative_prompt: str = "",
        steps: int = 50,
        cfg_scale: float = 7.5,
        seed: int = 42,
        width: int = 0,
        height: int = 0,
    ) -> "RefinementSession":
        """Create a new session from a prompt."""
        cfg = SessionConfig(
            checkpoint_path=checkpoint_path,
            device=device,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            width=width,
            height=height,
        )
        return cls(prompt, cfg, negative_prompt)

    def generate(self) -> GenerationResult:
        """Generate the initial image from the current prompt."""
        image_np = self._run_generation(
            prompt=self.current_prompt,
            negative_prompt=self.current_negative,
            seed=self.config.seed,
            strength=None,  # full generation
            init_image_path=None,
        )

        result = GenerationResult(
            image_np=image_np,
            prompt=self.current_prompt,
            negative_prompt=self.current_negative,
            seed=self.config.seed,
            step=self.step,
            metadata={"type": "initial"},
        )
        self.history.append(result)
        self.current_image_np = image_np

        # Save intermediate latent for future refinements
        self._save_current_image()
        self.step += 1
        return result

    def refine(
        self,
        instruction: str,
        *,
        strength: Optional[float] = None,
        seed: Optional[int] = None,
        mask_path: Optional[str] = None,
        segment_target: Optional[str] = None,
        segment_feather: float = 4.0,
        use_segmentation_models: bool = True,
        use_heuristic_inpaint_mask: bool = True,
    ) -> GenerationResult:
        """
        Apply a natural language edit instruction to the current image.

        Args:
            instruction: Natural language edit (e.g. "make the lighting more dramatic")
            strength: img2img strength override (0-1). If None, auto-determined from instruction.
            seed: Seed override. If None, uses current seed + step.
            mask_path: Optional inpaint mask (white=paint). Highest priority mask source.
            segment_target: If set (and ``mask_path`` is not), run ``segmentation_to_mask`` on the
                current image (Grounding DINO + SAM2 when present under ``pretrained/``, else phrases).
            segment_feather: Gaussian feather on the inferred mask (pixels-ish radius).
            use_segmentation_models: Allow loading DINO/SAM; if False, phrase heuristics only.
            use_heuristic_inpaint_mask: When True and the router sets ``edit.use_inpaint``, build a
                coarse region mask only if neither ``mask_path`` nor ``segment_target`` is set.

        Returns:
            GenerationResult with the refined image.
        """
        if self.current_image_np is None:
            raise RuntimeError("No image to refine. Call generate() first.")

        # Parse the instruction
        edit = self.router.classify(instruction)
        print(
            f"Refinement step {self.step}: type={edit.edit_type} "
            f"strength={edit.strength:.2f} inpaint={edit.use_inpaint}",
        )

        # Update prompt
        new_prompt, new_negative = self.delta_engine.apply(
            self.current_prompt,
            self.current_negative,
            edit,
        )

        # Determine strength
        refine_strength = (
            strength
            if strength is not None
            else min(
                edit.strength,
                self.config.max_refine_strength,
            )
        )

        # Determine seed
        use_seed = seed if seed is not None else self.config.seed + self.step

        inpaint_mask_file: Optional[str] = mask_path
        mask_source = "none"
        seg_extra: Dict[str, Any] = {}

        seg_t = (segment_target or "").strip()
        if inpaint_mask_file is None and seg_t:
            from PIL import Image

            from utils.generation.segmentation_to_mask import build_segmentation_mask_for_edit

            pil = Image.open(self._current_image_path()).convert("RGB")
            res = build_segmentation_mask_for_edit(
                pil,
                seg_t,
                feather_radius=float(segment_feather),
                use_vision_models=bool(use_segmentation_models),
            )
            inpaint_mask_file = str(self._tmp_dir / f"segment_mask_{self.step}.png")
            res.mask.save(inpaint_mask_file)
            mask_source = "segmentation"
            seg_extra = {"segmentation_mode": res.mode, "segmentation_notes": res.notes, "segment_target": seg_t}
        elif inpaint_mask_file is None and use_heuristic_inpaint_mask and edit.use_inpaint:
            hh, ww = (
                int(self.current_image_np.shape[0]),
                int(self.current_image_np.shape[1]),
            )
            from utils.generation.edit_masks import normalize_heuristic_region, save_heuristic_mask

            inpaint_mask_file = str(self._tmp_dir / f"heuristic_mask_{self.step}.png")
            _reg = normalize_heuristic_region(edit.inpaint_region or "subject")
            save_heuristic_mask(
                inpaint_mask_file,
                ww,
                hh,
                _reg,
            )
            mask_source = "heuristic"
        elif inpaint_mask_file is not None:
            mask_source = "user"

        # Run refinement
        image_np = self._run_generation(
            prompt=new_prompt,
            negative_prompt=new_negative,
            seed=use_seed,
            strength=refine_strength,
            init_image_path=self._current_image_path(),
            mask_path=inpaint_mask_file,
        )

        result = GenerationResult(
            image_np=image_np,
            prompt=new_prompt,
            negative_prompt=new_negative,
            seed=use_seed,
            step=self.step,
            edit_instruction=edit,
            metadata={
                "type": "refinement",
                "instruction": instruction,
                "strength": refine_strength,
                "prompt_additions": edit.prompt_additions,
                "prompt_removals": edit.prompt_removals,
                "mask_path": inpaint_mask_file,
                "mask_source": mask_source,
                "inpaint_region_hint": edit.inpaint_region,
                **seg_extra,
            },
        )

        self.history.append(result)
        self.current_prompt = new_prompt
        self.current_negative = new_negative
        self.current_image_np = image_np
        self._save_current_image()
        self.step += 1

        return result

    def undo(self) -> Optional[GenerationResult]:
        """Revert to the previous generation."""
        if len(self.history) < 2:
            print("Nothing to undo.")
            return None

        self.history.pop()
        prev = self.history[-1]
        self.current_prompt = prev.prompt
        self.current_negative = prev.negative_prompt
        self.current_image_np = prev.image_np
        self.step = prev.step + 1
        self._save_current_image()
        print(f"Reverted to step {prev.step}.")
        return prev

    def save_history(self, path: str) -> None:
        """Save the session history to a JSON file."""
        history_data = []
        for r in self.history:
            history_data.append(
                {
                    "step": r.step,
                    "prompt": r.prompt,
                    "negative_prompt": r.negative_prompt,
                    "seed": r.seed,
                    "edit_type": r.edit_instruction.edit_type if r.edit_instruction else None,
                    "instruction": r.edit_instruction.raw_text if r.edit_instruction else None,
                    "metadata": r.metadata,
                }
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"history": history_data, "steps": self.step}, f, indent=2)
        print(f"Session history saved: {path}")

    def cleanup(self) -> None:
        """Remove temporary files."""
        import shutil

        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _current_image_path(self) -> str:
        return str(self._tmp_dir / "current.png")

    def _save_current_image(self) -> None:
        if self.current_image_np is not None:
            import numpy as np
            from PIL import Image

            img = np.asarray(self.current_image_np, dtype=np.uint8)
            Image.fromarray(img).save(self._current_image_path())

    def _run_generation(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        strength: Optional[float],
        init_image_path: Optional[str],
        mask_path: Optional[str] = None,
    ) -> Any:
        """
        Run generation via subprocess call to sample.py.

        This keeps the session manager decoupled from the model loading —
        sample.py handles all the model/VAE/diffusion setup.
        """
        import subprocess
        import sys

        import numpy as np
        from PIL import Image

        from utils.generation.sample_edit_runner import resolve_repo_root, resolve_sample_py

        repo = resolve_repo_root()
        out_path = str(self._tmp_dir / f"step_{self.step}.png")

        cmd = [
            sys.executable,
            str(resolve_sample_py(repo)),
            "--ckpt",
            self.config.checkpoint_path,
            "--prompt",
            prompt,
            "--out",
            out_path,
            "--seed",
            str(seed),
            "--steps",
            str(self.config.steps),
            "--cfg-scale",
            str(self.config.cfg_scale),
            "--device",
            self.config.device,
            "--scheduler",
            self.config.scheduler,
            "--solver",
            self.config.solver,
        ]

        if negative_prompt.strip():
            cmd += ["--negative-prompt", negative_prompt]

        if self.config.width > 0:
            cmd += ["--width", str(self.config.width)]
        if self.config.height > 0:
            cmd += ["--height", str(self.config.height)]

        if init_image_path and strength is not None:
            cmd += ["--init-image", init_image_path, "--strength", str(strength)]

        if mask_path:
            cmd += ["--mask", mask_path, "--inpaint-mode", str(self.config.inpaint_mode or "mdm")]

        result = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
        if result.returncode != 0:
            err_tail = (result.stderr or "").strip()
            out_tail = (result.stdout or "").strip()
            detail = ((err_tail + "\n" + out_tail).strip())[-3500:] or "(no subprocess output captured)"
            _log.error(
                "sample.py failed step=%s returncode=%s ckpt=%s",
                self.step,
                result.returncode,
                self.config.checkpoint_path,
            )
            _log.debug("sample.py subprocess detail:\n%s", detail)
            raise RuntimeError(f"Generation failed (step {self.step}):\n{detail}") from None

        img = Image.open(out_path).convert("RGB")
        return np.array(img, dtype=np.uint8)


__all__ = [
    "RefinementSession",
    "SessionConfig",
    "GenerationResult",
    "EditInstruction",
    "EditRouter",
    "PromptDeltaEngine",
    "EDIT_TYPES",
]
