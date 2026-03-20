#!/usr/bin/env python3
"""
Style Harmonization System Example
Demonstrates how to handle mixed styles to prevent weird-looking images.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.style_harmonization import create_style_harmonization_system  # noqa: E402


def demonstrate_style_conflict_detection():
    """Demonstrate detection of style conflicts."""
    print("Style Conflict Detection Demonstration")
    print("=" * 50)

    harmonizer = create_style_harmonization_system()

    # Example 1: Severe 2D/3D conflict
    print("Example 1: Severe 2D/3D Conflict")
    prompt1 = "photorealistic 3d render of anime girl with cartoon eyes and manga style hair"
    loras1 = [
        {"name": "realistic_vision_v2", "strength": 1.0},
        {"name": "anime_diffusion", "strength": 0.8},
        {"name": "cartoon_style_lora", "strength": 0.6},
    ]

    result1 = harmonizer.harmonize_styles(prompt=prompt1, lora_configs=loras1)

    print(f"  Original Prompt: {prompt1}")
    print(f"  Detected Styles: {len(result1['detected_styles'])}")
    for style in result1["detected_styles"]:
        print(f"    - {style['type']} ({style['source']}): {style['strength']:.2f}")

    print(f"  Conflict Level: {result1['style_analysis']['conflict_level']}")
    print(f"  Dominant Style: {result1['style_analysis']['dominant_style']}")
    print(f"  Conflicts Found: {len(result1['conflicts'])}")

    for conflict in result1["conflicts"]:
        print(f"    - {conflict['style1']} <-> {conflict['style2']}: {conflict['severity']}")

    print()


def demonstrate_style_harmonization():
    """Demonstrate style harmonization in action."""
    print("Style Harmonization Demonstration")
    print("=" * 50)

    harmonizer = create_style_harmonization_system()

    # Scenario 1: Mixed 2D/3D with harmonization
    print("Scenario 1: Mixed 2D/3D Harmonization")
    prompt = "realistic 3d anime character portrait"
    loras = [
        {"name": "realistic_vision", "strength": 1.0},
        {"name": "anime_style", "strength": 0.8},
    ]

    result = harmonizer.harmonize_styles(
        prompt=prompt,
        lora_configs=loras,
        user_preferences={
            "harmonization_mode": "balanced",
            "allow_prompt_modification": True,
            "max_strength_reduction": 0.4,
        },
    )

    print(f"  Original: {prompt}")
    print(f"  Harmonized: {result['harmonized_prompt']}")
    print("  Changes Made:")
    for change in result["changes_made"]:
        print(f"    - {change}")

    print("  Original LoRAs:")
    for lora_cfg in loras:
        print(f"    - {lora_cfg['name']}: {lora_cfg['strength']}")

    print("  Harmonized LoRAs:")
    for lora_cfg in result["harmonized_loras"]:
        print(f"    - {lora_cfg['name']}: {lora_cfg['strength']}")

    print()


def main():
    demonstrate_style_conflict_detection()
    demonstrate_style_harmonization()


if __name__ == "__main__":
    main()

