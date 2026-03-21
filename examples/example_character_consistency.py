#!/usr/bin/env python3
"""
Character Consistency System Example
Demonstrates how to use the character consistency features for maintaining character identity across generations.
"""

import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.enhanced_dit import EnhancedDiT  # noqa: E402
from training.enhanced_trainer import EnhancedTrainer, EnhancedTrainingBatch  # noqa: E402
from utils.character_consistency import (  # noqa: E402
    CharacterDatabase,
    PhysicalFeatures,
    StylePreferences,
)


def create_sample_reference_images(character_name: str, output_dir: str = "./sample_references"):
    """Create sample reference images for demonstration."""
    os.makedirs(output_dir, exist_ok=True)

    # Create simple colored reference images with text labels
    colors = [(255, 200, 200), (200, 255, 200), (200, 200, 255)]  # Light red, green, blue
    reference_paths = []

    for i, color in enumerate(colors):
        # Create image
        img = Image.new("RGB", (256, 256), color)
        draw = ImageDraw.Draw(img)

        # Add character name and reference number
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        text = f"{character_name}\nRef {i + 1}"
        if font:
            draw.text((10, 10), text, fill=(0, 0, 0), font=font)
        else:
            draw.text((10, 10), text, fill=(0, 0, 0))

        # Add some simple "facial features" (circles for eyes, etc.)
        # Eyes
        draw.ellipse([80, 80, 100, 100], fill=(0, 0, 0))  # Left eye
        draw.ellipse([156, 80, 176, 100], fill=(0, 0, 0))  # Right eye

        # Nose
        draw.ellipse([120, 120, 136, 140], fill=(100, 100, 100))

        # Mouth
        draw.arc([100, 160, 156, 180], 0, 180, fill=(0, 0, 0), width=3)

        # Save image
        img_path = os.path.join(output_dir, f"{character_name.lower().replace(' ', '_')}_ref_{i + 1}.png")
        img.save(img_path)
        reference_paths.append(img_path)
        print(f"Created reference image: {img_path}")

    return reference_paths


def demonstrate_character_creation():
    """Demonstrate creating character profiles."""
    print("Character Creation Demonstration")
    print("=" * 50)

    # Initialize character database
    db = CharacterDatabase("./demo_character_database")

    # Create sample reference images
    elena_refs = create_sample_reference_images("Elena Rodriguez")
    marcus_refs = create_sample_reference_images("Marcus Chen")

    # Create first character - Elena Rodriguez
    elena_features = PhysicalFeatures(
        face_shape="oval",
        eye_color="hazel",
        eye_shape="almond",
        hair_color="dark_brown",
        hair_style="long_wavy",
        height="average",
        build="athletic",
        distinctive_marks=["small_scar_left_eyebrow"],
    )

    elena_style = StylePreferences(
        clothing_style="casual_modern",
        color_palette=["navy", "white", "gold"],
        accessories=["silver_watch", "small_hoop_earrings"],
    )

    elena_profile = db.create_character(
        name="Elena Rodriguez",
        reference_images=elena_refs,
        physical_features=elena_features,
        style_preferences=elena_style,
    )

    print(f"Created character: {elena_profile.name}")
    print(f"  Character ID: {elena_profile.character_id}")
    print(f"  Reference Images: {len(elena_profile.reference_images)}")
    print(f"  Has Face Embedding: {'Yes' if elena_profile.face_embedding is not None else 'No'}")
    print(f"  Has Body Embedding: {'Yes' if elena_profile.body_embedding is not None else 'No'}")

    # Create second character - Marcus Chen
    marcus_features = PhysicalFeatures(
        face_shape="square",
        eye_color="brown",
        eye_shape="round",
        hair_color="black",
        hair_style="short",
        height="tall",
        build="slim",
        facial_hair="goatee",
    )

    marcus_style = StylePreferences(
        clothing_style="formal",
        color_palette=["black", "gray", "blue"],
        accessories=["glasses", "leather_watch"],
    )

    marcus_profile = db.create_character(
        name="Marcus Chen",
        reference_images=marcus_refs,
        physical_features=marcus_features,
        style_preferences=marcus_style,
    )

    print(f"Created character: {marcus_profile.name}")
    print(f"  Character ID: {marcus_profile.character_id}")

    return db, elena_profile, marcus_profile


def demonstrate_consistency_validation(db, character_profile):
    """Demonstrate character consistency validation."""
    print(f"\nConsistency Validation for {character_profile.name}")
    print("=" * 50)

    # Create a test image (in practice, this would be a generated image)
    test_image = torch.randn(3, 256, 256)

    # Validate consistency
    scores = db.validate_consistency(test_image, character_profile.character_id)

    print("Consistency Results:")
    print(f"  Face Similarity: {scores['face_similarity']:.3f}")
    print(f"  Body Similarity: {scores['body_similarity']:.3f}")
    print(f"  Color Consistency: {scores['color_consistency']:.3f}")
    print(f"  Overall Score: {scores['overall_consistency']:.3f}")
    print(f"  Consistency Level: {scores['consistency_level']}")

    return scores


def demonstrate_training_integration():
    """Demonstrate training integration with character consistency."""
    print("\nTraining Integration Demonstration")
    print("=" * 50)

    # Create a minimal Enhanced DiT model for demonstration
    model = EnhancedDiT(
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=10,
    )

    # Create trainer with character consistency
    trainer = EnhancedTrainer(model, device="cpu", character_database_path="./demo_character_database")

    # Get existing characters
    characters = trainer.list_characters()
    print(f"Found {len(characters)} characters in database")

    if characters:
        # Create a training batch with character consistency data
        batch = EnhancedTrainingBatch(
            images=torch.randn(2, 3, 256, 256),
            timesteps=torch.randint(0, 1000, (2,)),
            noise=torch.randn(2, 3, 256, 256),
            class_labels=torch.randint(0, 10, (2,)),
            character_profiles=characters[:2] if len(characters) >= 2 else [characters[0], characters[0]],
            reference_images=torch.randn(2, 3, 256, 256),
        )

        print("Created training batch with character consistency data")
        print(f"  Batch size: {batch.images.shape[0]}")
        print(f"  Character profiles: {len(batch.character_profiles)}")

        stats = trainer.get_consistency_statistics()
        print(f"  Character count: {stats['character_count']}")

    return trainer


def demonstrate_character_management(db):
    """Demonstrate character management operations."""
    print("\nCharacter Management Demonstration")
    print("=" * 50)

    characters = db.list_characters()
    print(f"Total characters: {len(characters)}")

    for char in characters:
        print(f"  {char.name} (ID: {char.character_id})")
        print(f"    Face Shape: {char.physical_features.face_shape}")
        print(f"    Eye Color: {char.physical_features.eye_color}")
        print(f"    Style: {char.style_preferences.clothing_style}")
        print(f"    References: {len(char.reference_images)}")

    return characters


def demonstrate_cli_usage():
    """Demonstrate CLI usage examples."""
    print("\nCLI Usage Examples")
    print("=" * 50)

    print("Character Management Commands:")
    print("  # Create a character")
    print("  python scripts/cli.py character create 'Elena Rodriguez' --references ref1.jpg ref2.jpg ref3.jpg")
    print("  python scripts/cli.py character create 'Elena Rodriguez' --face-shape oval --eye-color hazel --hair-color brown")
    print()
    print("  # List characters")
    print("  python scripts/cli.py character list")
    print("  python scripts/cli.py character list --filter-name Elena --min-consistency 0.8")
    print()
    print("  # Validate character consistency")
    print("  python scripts/cli.py character validate char_12345678 generated_image.png")
    print()
    print("  # Generate with character consistency")
    print(
        "  python scripts/cli.py generate 'Elena Rodriguez walking in a park' --checkpoint model.pt --character 'Elena Rodriguez'"
    )
    print("  python scripts/cli.py generate 'portrait of Elena' --checkpoint model.pt --character char_12345678")
    print()
    print("  # Character statistics")
    print("  python scripts/cli.py character stats --output character_stats.json")


def save_demonstration_results(db, characters, validation_scores):
    """Save demonstration results to files."""
    results = {
        "characters": [char.to_dict() for char in characters],
        "validation_scores": validation_scores,
        "database_info": {
            "total_characters": len(characters),
            "database_path": str(db.database_path),
        },
    }

    with open("character_consistency_demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open("character_consistency_demo_report.md", "w", encoding="utf-8") as f:
        f.write("# Character Consistency System Demonstration Report\n\n")
        f.write("## Summary\n")
        f.write(f"- Total characters created: {len(characters)}\n")
        f.write(f"- Database location: {db.database_path}\n")


def main():
    """Run the complete character consistency demonstration."""
    try:
        db, elena_profile, _marcus_profile = demonstrate_character_creation()
        validation_scores = demonstrate_consistency_validation(db, elena_profile)
        demonstrate_training_integration()
        all_characters = demonstrate_character_management(db)
        demonstrate_cli_usage()
        save_demonstration_results(db, all_characters, validation_scores)
        print("\nCharacter Consistency Demonstration Complete.")
        return True
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
