#!/usr/bin/env python3
"""
Enhanced Inference Script for Advanced DiT Models
Generate images using models with built-in precision control, anatomy awareness, text rendering, and consistency.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from diffusion import create_diffusion
from models.enhanced_dit import EnhancedDiT_models
from utils.enhanced_utils import SimpleAnatomyValidator, SimpleConsistencyManager, SimpleSceneComposer, SimpleTextEngine


def load_enhanced_model(checkpoint_path: str, device: str = "cuda"):
    """Load enhanced DiT model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    # Get model function
    model_name = getattr(config, "model_name", "EnhancedDiT-XL/2")
    model_fn = EnhancedDiT_models.get(model_name)
    if model_fn is None:
        raise ValueError(f"Unknown enhanced model: {model_name}")

    # Create model with enhanced features
    model = model_fn(
        input_size=config.image_size // 8,
        enable_spatial_control=getattr(config, "enable_spatial_control", True),
        enable_anatomy_awareness=getattr(config, "enable_anatomy_awareness", True),
        enable_text_rendering=getattr(config, "enable_text_rendering", True),
        enable_consistency=getattr(config, "enable_consistency", True),
    )

    # Load weights (prefer EMA if available)
    state_dict = checkpoint.get("ema", checkpoint.get("model"))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    return model, config


def prepare_enhanced_inputs(prompt: str, args, device: str = "cuda"):
    """Prepare enhanced feature inputs from prompt and arguments."""
    enhanced_inputs = {}

    # Initialize feature processors
    text_engine = SimpleTextEngine()
    scene_composer = SimpleSceneComposer()
    _anatomy_validator = SimpleAnatomyValidator()
    _consistency_manager = SimpleConsistencyManager()

    # Process spatial control
    if args.enable_spatial_control:
        # Extract objects from prompt
        objects = extract_objects_from_prompt(prompt)
        if objects:
            scene_objects = scene_composer.create_scene_layout(
                objects[:10],  # Max 10 objects
                layout_type=args.layout_type,
                constraints=args.spatial_constraints.split(",") if args.spatial_constraints else [],
            )

            # Convert to tensor
            spatial_layout = torch.zeros(1, 10, 4, device=device)
            for i, obj in enumerate(scene_objects):
                if i < 10:
                    spatial_layout[0, i] = torch.tensor(
                        [obj.position[0], obj.position[1], obj.size[0], obj.size[1]], device=device
                    )

            enhanced_inputs["spatial_layout"] = spatial_layout

    # Process anatomy awareness
    if args.enable_anatomy_awareness:
        # Check if prompt contains humans
        if detect_human_in_prompt(prompt):
            # Create simple anatomy mask (would be more sophisticated)
            anatomy_mask = torch.zeros(1, (args.image_size // 8) ** 2, device=device)
            # Mark center region as potential human area
            mask_size = args.image_size // 8
            center = mask_size // 2
            anatomy_mask[0, center * mask_size + center - 10 : center * mask_size + center + 10] = 1.0
            enhanced_inputs["anatomy_mask"] = anatomy_mask

    # Process text rendering
    if args.enable_text_rendering:
        text_info = text_engine.extract_text_requirements(prompt)
        if text_info["has_text"] or args.text_content:
            # Prepare text tokens
            text_content = args.text_content.split(",") if args.text_content else text_info.get("text_content", [])
            if text_content:
                # Simple tokenization (would use actual tokenizer)
                text_tokens = torch.zeros(1, 50, dtype=torch.long, device=device)
                text_combined = " ".join(text_content)[:50]
                for i, char in enumerate(text_combined):
                    if i < 50:
                        text_tokens[0, i] = ord(char) % 1000

                enhanced_inputs["text_tokens"] = text_tokens

                # Text positions
                text_positions = torch.zeros(1, 50, 2, device=device)
                for i in range(min(len(text_content), 50)):
                    x = 0.3 + 0.4 * (i / max(1, len(text_content) - 1))
                    y = 0.5
                    text_positions[0, i] = torch.tensor([x, y], device=device)

                enhanced_inputs["text_positions"] = text_positions

                # Typography style
                typography_styles = {"modern": 0, "classic": 1, "display": 2, "script": 3}
                style_id = typography_styles.get(args.typography_style, 0)
                enhanced_inputs["typography_style"] = torch.tensor([style_id], dtype=torch.long, device=device)

    # Process consistency
    if args.enable_consistency:
        if args.character_name:
            # Would load from consistency manager
            character_id = hash(args.character_name) % 100  # Simplified
            enhanced_inputs["character_id"] = torch.tensor([character_id], dtype=torch.long, device=device)

        if args.style_name:
            # Would load from consistency manager
            style_id = hash(args.style_name) % 50  # Simplified
            enhanced_inputs["style_id"] = torch.tensor([style_id], dtype=torch.long, device=device)

    return enhanced_inputs


def extract_objects_from_prompt(prompt: str):
    """Extract objects from prompt for spatial layout."""
    common_objects = [
        "person",
        "woman",
        "man",
        "girl",
        "boy",
        "child",
        "car",
        "house",
        "tree",
        "flower",
        "chair",
        "table",
        "cat",
        "dog",
        "bird",
        "book",
        "cup",
        "bottle",
    ]

    objects = []
    prompt_lower = prompt.lower()

    for obj in common_objects:
        if obj in prompt_lower:
            objects.append(obj)

    return objects


def detect_human_in_prompt(prompt: str):
    """Detect if prompt describes humans."""
    human_keywords = [
        "person",
        "woman",
        "man",
        "girl",
        "boy",
        "child",
        "people",
        "human",
        "figure",
        "character",
        "portrait",
    ]

    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in human_keywords)


@torch.no_grad()
def sample_enhanced(model, diffusion, prompt, enhanced_inputs, args, device="cuda"):
    """Sample from enhanced DiT model."""
    # Prepare inputs
    batch_size = 1

    # Create noise
    shape = (batch_size, 4, args.image_size // 8, args.image_size // 8)
    z = torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(args.seed))

    # Create timesteps
    timesteps = torch.linspace(1000, 0, args.steps + 1, device=device)[:-1]

    # Class conditioning (simplified)
    y = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Sampling loop
    for i, t in enumerate(timesteps):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Model prediction with enhanced features
        noise_pred = model(z, t_batch, y, **enhanced_inputs)

        # DDIM step (simplified)
        alpha_t = torch.cos(t / 1000 * np.pi / 2) ** 2
        alpha_t_prev = torch.cos(timesteps[min(i + 1, len(timesteps) - 1)] / 1000 * np.pi / 2) ** 2

        # Predict x0
        x0_pred = (z - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        # Compute previous sample
        if i < len(timesteps) - 1:
            z = torch.sqrt(alpha_t_prev) * x0_pred + torch.sqrt(1 - alpha_t_prev) * noise_pred
        else:
            z = x0_pred

    return z


def main():
    """Main enhanced inference function."""
    parser = argparse.ArgumentParser(description="Enhanced DiT Inference")
    parser.add_argument("prompt", type=str, help="Text prompt")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=str, default="enhanced_output.png", help="Output path")

    # Generation parameters
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image-size", type=int, default=512, help="Image size")

    # Enhanced feature flags
    parser.add_argument("--enable-spatial-control", action="store_true", default=True)
    parser.add_argument("--enable-anatomy-awareness", action="store_true", default=True)
    parser.add_argument("--enable-text-rendering", action="store_true", default=True)
    parser.add_argument("--enable-consistency", action="store_true", default=True)

    # Spatial control parameters
    parser.add_argument(
        "--layout-type", type=str, default="balanced", choices=["grid", "circle", "line", "pyramid", "balanced"]
    )
    parser.add_argument("--spatial-constraints", type=str, default="", help="Comma-separated spatial constraints")

    # Text rendering parameters
    parser.add_argument("--text-content", type=str, default="", help="Comma-separated text content to render")
    parser.add_argument(
        "--typography-style", type=str, default="modern", choices=["modern", "classic", "display", "script"]
    )

    # Consistency parameters
    parser.add_argument("--character-name", type=str, default="", help="Character name for consistency")
    parser.add_argument("--style-name", type=str, default="", help="Style name for consistency")

    args = parser.parse_args()
    prompt_aug = args.prompt

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading enhanced model from {args.checkpoint}...")
    model, config = load_enhanced_model(args.checkpoint, device)
    print(f"Model loaded: {config.model_name}")

    # Create diffusion
    diffusion = create_diffusion(timestep_respacing="")

    # Prepare enhanced inputs
    print("Preparing enhanced features...")
    enhanced_inputs = prepare_enhanced_inputs(args.prompt, args, device)

    print("Enhanced features enabled:")
    for feature, tensor in enhanced_inputs.items():
        print(f"  {feature}: {tensor.shape}")

    # Generate image
    args.prompt = prompt_aug
    print(f"Generating image with prompt: '{args.prompt}'")
    print(f"Using {args.steps} steps, seed {args.seed}")

    latents = sample_enhanced(model, diffusion, args.prompt, enhanced_inputs, args, device)

    # Decode latents (would use actual VAE)
    # For now, convert latents to image format
    latents = latents.cpu()

    # Simple conversion to image (placeholder)
    image_array = latents[0].permute(1, 2, 0).numpy()
    image_array = (image_array + 1) * 127.5  # Denormalize
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # Create PIL image
    if image_array.shape[-1] == 4:  # 4 channels from latent
        # Convert to RGB (simplified)
        image_array = image_array[:, :, :3]

    image = Image.fromarray(image_array)
    image = image.resize((args.image_size, args.image_size), Image.LANCZOS)

    # Save image
    image.save(args.output)
    print(f"✅ Enhanced image saved to {args.output}")

    # Print feature summary
    print("\n🎨 Generation Summary:")
    print(f"  Model: {config.model_name}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Enhanced features used: {len(enhanced_inputs)}")

    if "spatial_layout" in enhanced_inputs:
        objects = extract_objects_from_prompt(args.prompt)
        print(f"  Objects detected: {objects}")

    if "text_tokens" in enhanced_inputs:
        print(f"  Text content: {args.text_content}")
        print(f"  Typography style: {args.typography_style}")

    if "character_id" in enhanced_inputs:
        print(f"  Character: {args.character_name}")

    if "style_id" in enhanced_inputs:
        print(f"  Style: {args.style_name}")


if __name__ == "__main__":
    main()
