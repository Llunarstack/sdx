#!/usr/bin/env python3
"""
Character consistency example: build character profiles, score an image against one, and wire the
profiles into an ``EnhancedTrainer`` batch.
"""

import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dit_variants import EnhancedDiT  # noqa: E402
from training.enhanced_trainer import EnhancedTrainer, EnhancedTrainingBatch  # noqa: E402
from utils.consistency import (  # noqa: E402
    CharacterDatabase,
    PhysicalFeatures,
    StylePreferences,
)

_DATABASE_DIR = "./demo_character_database"


def _section(title: str) -> None:
    print(f"\n{title}\n" + "=" * 50)


def create_sample_reference_images(character_name: str, output_dir: str = "./sample_references") -> list[str]:
    """Three labelled placeholder 'faces' (light red / green / blue) to use as reference images."""
    os.makedirs(output_dir, exist_ok=True)
    reference_paths = []
    for i, color in enumerate([(255, 200, 200), (200, 255, 200), (200, 200, 255)]):
        img = Image.new("RGB", (256, 256), color)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"{character_name}\nRef {i + 1}", fill=(0, 0, 0))
        draw.ellipse([80, 80, 100, 100], fill=(0, 0, 0))  # eyes
        draw.ellipse([156, 80, 176, 100], fill=(0, 0, 0))
        draw.ellipse([120, 120, 136, 140], fill=(100, 100, 100))  # nose
        draw.arc([100, 160, 156, 180], 0, 180, fill=(0, 0, 0), width=3)  # mouth
        img_path = os.path.join(output_dir, f"{character_name.lower().replace(' ', '_')}_ref_{i + 1}.png")
        img.save(img_path)
        reference_paths.append(img_path)
        print(f"Created reference image: {img_path}")
    return reference_paths


def demonstrate_character_creation():
    _section("Character Creation Demonstration")
    db = CharacterDatabase(_DATABASE_DIR)

    elena_profile = db.create_character(
        name="Elena Rodriguez",
        reference_images=create_sample_reference_images("Elena Rodriguez"),
        physical_features=PhysicalFeatures(
            face_shape="oval",
            eye_color="hazel",
            eye_shape="almond",
            hair_color="dark_brown",
            hair_style="long_wavy",
            height="average",
            build="athletic",
            distinctive_marks=["small_scar_left_eyebrow"],
        ),
        style_preferences=StylePreferences(
            clothing_style="casual_modern",
            color_palette=["navy", "white", "gold"],
            accessories=["silver_watch", "small_hoop_earrings"],
        ),
    )
    print(f"Created character: {elena_profile.name}")
    print(f"  Character ID: {elena_profile.character_id}")
    print(f"  Reference Images: {len(elena_profile.reference_images)}")
    print(f"  Has Face Embedding: {'Yes' if elena_profile.face_embedding is not None else 'No'}")
    print(f"  Has Body Embedding: {'Yes' if elena_profile.body_embedding is not None else 'No'}")

    marcus_profile = db.create_character(
        name="Marcus Chen",
        reference_images=create_sample_reference_images("Marcus Chen"),
        physical_features=PhysicalFeatures(
            face_shape="square",
            eye_color="brown",
            eye_shape="round",
            hair_color="black",
            hair_style="short",
            height="tall",
            build="slim",
            facial_hair="goatee",
        ),
        style_preferences=StylePreferences(
            clothing_style="formal",
            color_palette=["black", "gray", "blue"],
            accessories=["glasses", "leather_watch"],
        ),
    )
    print(f"Created character: {marcus_profile.name}")
    print(f"  Character ID: {marcus_profile.character_id}")
    return db, elena_profile, marcus_profile


def demonstrate_consistency_validation(db, character_profile):
    _section(f"Consistency Validation for {character_profile.name}")
    test_image = torch.randn(3, 256, 256)  # stands in for a generated image
    scores = db.validate_consistency(test_image, character_profile.character_id)
    print("Consistency Results:")
    print(f"  Face Similarity: {scores['face_similarity']:.3f}")
    print(f"  Body Similarity: {scores['body_similarity']:.3f}")
    print(f"  Color Consistency: {scores['color_consistency']:.3f}")
    print(f"  Overall Score: {scores['overall_consistency']:.3f}")
    print(f"  Consistency Level: {scores['consistency_level']}")
    return scores


def demonstrate_training_integration():
    _section("Training Integration Demonstration")
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
    trainer = EnhancedTrainer(model, device="cpu", character_database_path=_DATABASE_DIR)
    characters = trainer.list_characters()
    print(f"Found {len(characters)} characters in database")
    if characters:
        batch = EnhancedTrainingBatch(
            images=torch.randn(2, 3, 256, 256),
            timesteps=torch.randint(0, 1000, (2,)),
            noise=torch.randn(2, 3, 256, 256),
            class_labels=torch.randint(0, 10, (2,)),
            character_profiles=(characters * 2)[:2],
            reference_images=torch.randn(2, 3, 256, 256),
        )
        print("Created training batch with character consistency data")
        print(f"  Batch size: {batch.images.shape[0]}")
        print(f"  Character profiles: {len(batch.character_profiles)}")
        print(f"  Character count: {trainer.get_consistency_statistics()['character_count']}")
    return trainer


def demonstrate_character_management(db):
    _section("Character Management Demonstration")
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
    _section("CLI Usage Examples")
    print("  # Lock a character's prompt additions / references into a session file")
    print("  python -m scripts.tools character_session --path elena.json init --name 'Elena Rodriguez'")
    print("  python -m scripts.tools character_session --path elena.json set reference_images ref1.png")
    print()
    print("  # Generate with the session applied")
    print("  python sample.py --ckpt model.pt --prompt 'Elena walking in a park' --character-session elena.json")


def save_demonstration_results(db, characters, validation_scores):
    reports_dir = Path(__file__).resolve().parents[1] / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "characters": [char.to_dict() for char in characters],
        "validation_scores": validation_scores,
        "database_info": {"total_characters": len(characters), "database_path": str(db.database_path)},
    }
    (reports_dir / "character_consistency_demo_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (reports_dir / "character_consistency_demo_report.md").write_text(
        "# Character Consistency System Demonstration Report\n\n"
        "## Summary\n"
        f"- Total characters created: {len(characters)}\n"
        f"- Database location: {db.database_path}\n",
        encoding="utf-8",
    )


def main() -> None:
    db, elena_profile, _ = demonstrate_character_creation()
    validation_scores = demonstrate_consistency_validation(db, elena_profile)
    demonstrate_training_integration()
    all_characters = demonstrate_character_management(db)
    demonstrate_cli_usage()
    save_demonstration_results(db, all_characters, validation_scores)
    print("\nCharacter Consistency Demonstration Complete.")


if __name__ == "__main__":
    main()
