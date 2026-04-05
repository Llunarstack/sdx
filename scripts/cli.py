#!/usr/bin/env python3
"""
SDX Command Line Interface - Comprehensive tool for training, inference, and dataset management.
Now integrated with the master system for unified functionality.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Repo root: scripts/cli.py -> parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from config.train_config import TrainConfig
from utils.analysis.data_analysis import DatasetAnalyzer
from utils.checkpoint.checkpoint_manager import CheckpointManager, analyze_checkpoint_differences, merge_checkpoints
from utils.generation.advanced_inference import PromptOptimizer
from utils.generation.master_integration import create_sdx_master, quick_generate
from utils.modeling.model_viz import analyze_model_architecture, print_model_summary
from utils.training.config_validator import estimate_memory_usage, suggest_optimizations, validate_train_config
from utils.training.error_handling import validate_checkpoint


def cmd_analyze_dataset(args):
    """Analyze dataset quality and statistics."""
    print("🔍 Analyzing dataset...")

    analyzer = DatasetAnalyzer(data_path=args.data_path, manifest_path=args.manifest)

    if args.output:
        report = analyzer.generate_report(args.output)
        print(f"Report saved to {args.output}")
    else:
        report = analyzer.generate_report()
        print(report)


def cmd_validate_config(args):
    """Validate training configuration."""
    print("✅ Validating configuration...")

    # Load config
    if args.config.endswith(".json"):
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        cfg = TrainConfig(**config_dict)
    else:
        # Assume it's a Python file with TrainConfig
        import importlib.util

        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        cfg = config_module.cfg

    # Validate
    issues = validate_train_config(cfg)

    if not issues:
        print("✅ Configuration is valid!")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")

    # Memory estimation
    if args.estimate_memory:
        memory_est = estimate_memory_usage(cfg)
        print("\n💾 Memory Estimation:")
        print(f"  Model Memory: {memory_est['model_memory_gb']:.1f} GB")
        print(f"  Batch Memory: {memory_est['batch_memory_gb']:.1f} GB")
        print(f"  Total Estimated: {memory_est['total_estimated_gb']:.1f} GB")
        print(f"  Recommended VRAM: {memory_est['recommended_vram_gb']:.1f} GB")

    # Optimization suggestions
    if args.suggest_optimizations:
        suggestions = suggest_optimizations(cfg)
        if suggestions:
            print("\n💡 Optimization Suggestions:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")


def cmd_manage_checkpoints(args):
    """Manage model checkpoints."""
    manager = CheckpointManager(args.checkpoint_dir)

    if args.action == "list":
        checkpoints = manager.list_checkpoints(sort_by=args.sort_by)
        print(f"📁 Checkpoints in {args.checkpoint_dir}:")
        print(f"{'Name':<30} {'Step':<8} {'Loss':<10} {'Size (MB)':<10} {'Best':<6}")
        print("-" * 70)

        for cp in checkpoints:
            is_best = "✓" if cp.get("is_best", False) else ""
            print(f"{cp['name']:<30} {cp['step']:<8} {cp['loss']:<10.4f} {cp['file_size_mb']:<10.1f} {is_best:<6}")

    elif args.action == "cleanup":
        manager.cleanup_old_checkpoints(keep_best=args.keep_best, keep_recent=args.keep_recent)

    elif args.action == "compare":
        if len(args.checkpoints) != 2:
            print("Error: Exactly 2 checkpoints required for comparison")
            return

        comparison = manager.compare_checkpoints(args.checkpoints[0], args.checkpoints[1])
        print("📊 Checkpoint Comparison:")
        print(f"  Checkpoint 1: {comparison['checkpoint1']}")
        print(f"  Checkpoint 2: {comparison['checkpoint2']}")
        print(f"  Step Difference: {comparison['step_diff']}")
        print(f"  Loss Difference: {comparison['loss_diff']:.6f}")
        print(f"  Size Difference: {comparison['size_diff_mb']:.1f} MB")

    elif args.action == "analyze":
        if len(args.checkpoints) != 2:
            print("Error: Exactly 2 checkpoints required for analysis")
            return

        analysis = analyze_checkpoint_differences(args.checkpoints[0], args.checkpoints[1])

        print("🔬 Detailed Checkpoint Analysis:")
        print(f"  Total Parameters: {analysis['statistics']['total_parameters']:,}")
        print(f"  Average Difference: {analysis['statistics']['average_difference']:.8f}")
        print(f"  Max Difference: {analysis['statistics']['max_difference']:.8f}")
        print(f"  Max Diff Parameter: {analysis['statistics']['max_difference_parameter']}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"Detailed analysis saved to {args.output}")


def cmd_merge_checkpoints(args):
    """Merge multiple checkpoints."""
    print(f"🔄 Merging {len(args.checkpoints)} checkpoints...")

    weights = None
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]

    merged_path = merge_checkpoints(
        checkpoint_paths=args.checkpoints, output_path=args.output, weights=weights, merge_method=args.method
    )

    print(f"✅ Merged checkpoint saved to {merged_path}")


def cmd_optimize_prompt(args):
    """Optimize prompts for better generation."""
    optimizer = PromptOptimizer()

    if args.prompt:
        # Single prompt
        optimized = optimizer.optimize_prompt(
            args.prompt, style=args.style, add_quality=not args.no_quality, boost_subject=not args.no_boost
        )

        print(f"Original:  {args.prompt}")
        print(f"Optimized: {optimized}")

        if args.negative:
            neg_optimized = optimizer.optimize_negative_prompt(args.negative)
            print(f"Negative:  {neg_optimized}")

        # Suggestions
        suggestions = optimizer.suggest_improvements(args.prompt)
        if suggestions:
            print("\n💡 Suggestions:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")

    elif args.file:
        # Batch process file
        with open(args.file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

        output_file = args.output or args.file.replace(".txt", "_optimized.txt")

        with open(output_file, "w") as f:
            for prompt in prompts:
                optimized = optimizer.optimize_prompt(
                    prompt, style=args.style, add_quality=not args.no_quality, boost_subject=not args.no_boost
                )
                f.write(optimized + "\n")

        print(f"✅ Optimized {len(prompts)} prompts saved to {output_file}")


def cmd_analyze_model(args):
    """Analyze model architecture."""
    import torch
    from models import DiT_models_text

    from config import get_dit_build_kwargs

    print(f"🔍 Analyzing model: {args.model}")

    # Load model
    if args.checkpoint:
        # Load from checkpoint
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config")
        if cfg is None:
            print("Error: Checkpoint must contain config")
            return

        model_name = getattr(cfg, "model_name", args.model)
        model_fn = DiT_models_text.get(model_name)
        if model_fn is None:
            print(f"Error: Unknown model {model_name}")
            return

        model = model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
        state = ckpt.get("ema") or ckpt.get("model")
        model.load_state_dict(state, strict=True)
    else:
        # Create new model
        model_fn = DiT_models_text.get(args.model)
        if model_fn is None:
            print(f"Error: Unknown model {args.model}")
            return

        # Use default config
        from config.train_config import TrainConfig

        cfg = TrainConfig(model_name=args.model)
        model = model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))

    # Analyze
    analysis = analyze_model_architecture(model)
    print_model_summary(model)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {args.output}")


def cmd_validate_checkpoint(args):
    """Validate checkpoint integrity."""
    print(f"🔍 Validating checkpoint: {args.checkpoint}")

    if validate_checkpoint(args.checkpoint):
        print("✅ Checkpoint is valid")

        # Load and show info
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        print(f"  Step: {ckpt.get('step', 'Unknown')}")
        print(f"  Loss: {ckpt.get('loss', 'Unknown')}")
        print(f"  Timestamp: {ckpt.get('timestamp', 'Unknown')}")

        if "config" in ckpt:
            cfg = ckpt["config"]
            print(f"  Model: {getattr(cfg, 'model_name', 'Unknown')}")
            print(f"  Image Size: {getattr(cfg, 'image_size', 'Unknown')}")

        # Check for EMA
        has_ema = "ema" in ckpt
        print(f"  Has EMA: {'Yes' if has_ema else 'No'}")

    else:
        print("❌ Checkpoint is invalid or corrupted")


def cmd_generate_image(args):
    """Generate image using multimodal system with character consistency."""
    print("🎨 Generating image...")

    try:
        # Parse LoRA configurations
        lora_configs = []
        if args.loras:
            for lora_spec in args.loras:
                if ":" in lora_spec:
                    name, strength = lora_spec.split(":", 1)
                    try:
                        strength = float(strength)
                    except ValueError:
                        strength = 1.0
                else:
                    name, strength = lora_spec, 1.0

                lora_configs.append({"name": name, "strength": strength})

        # Apply style harmonization if requested
        final_prompt = args.prompt
        final_lora_configs = lora_configs

        if args.harmonize_styles and (
            lora_configs
            or any(
                style_word in args.prompt.lower() for style_word in ["anime", "realistic", "cartoon", "3d", "painting"]
            )
        ):
            from utils.consistency.style_harmonization import create_style_harmonization_system

            print("🎨 Applying style harmonization...")
            harmonizer = create_style_harmonization_system()

            harmonization_result = harmonizer.harmonize_styles(
                prompt=args.prompt,
                lora_configs=lora_configs,
                user_preferences={
                    "harmonization_mode": args.harmonization_mode,
                    "allow_prompt_modification": True,
                    "max_strength_reduction": 0.4,
                },
            )

            final_prompt = harmonization_result["harmonized_prompt"]
            final_lora_configs = harmonization_result["adjusted_loras"]

            if harmonization_result["changes_made"]:
                print("🔧 Style harmonization applied:")
                for change in harmonization_result["changes_made"]:
                    print(f"   - {change}")
                print(f"   Final prompt: {final_prompt}")
            else:
                print("✅ No style conflicts detected - using original prompt")

        # Create generation request
        request_kwargs = {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "negative_prompt": args.negative or "",
            "use_precision_control": args.precision_control,
            "use_anatomy_correction": args.anatomy_correction,
            "has_text": args.has_text,
            "enhance_output": not args.no_enhance,
            "quality_level": args.quality,
            "lora_configs": final_lora_configs,
        }

        # Character consistency support
        if args.character:
            from utils.consistency.character_consistency import CharacterDatabase

            # Try to load character by ID or name
            db = CharacterDatabase()
            character = None

            # First try as character ID
            character = db.get_character(args.character)

            # If not found, try to find by name
            if not character:
                characters = db.list_characters({"name": args.character})
                if characters:
                    character = characters[0]

            if character:
                print(f"🎭 Using character: {character.name} (ID: {character.character_id})")
                request_kwargs["character_profile"] = character
                request_kwargs["character_name"] = character.name
                request_kwargs["character_id"] = character.character_id
            else:
                print(f"⚠️  Character '{args.character}' not found. Proceeding without character consistency.")
                request_kwargs["character_name"] = args.character

        if args.style:
            request_kwargs["style_name"] = args.style
        if args.scene:
            request_kwargs["scene_id"] = args.scene

        # Generate image
        result = quick_generate(final_prompt, args.checkpoint, **request_kwargs)

        # Save image
        output_path = args.output or f"generated_{result.generation_params.get('seed', 'random')}.png"
        result.image.save(output_path)

        print(f"✅ Image saved to {output_path}")
        print(f"Quality score: {result.quality_score:.2f}/100")

        # Character consistency validation if character was used
        if args.character and "character_profile" in request_kwargs:
            try:
                import torchvision.transforms as transforms
                from PIL import Image

                # Load generated image and validate consistency
                image = Image.open(output_path).convert("RGB")
                transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
                image_tensor = transform(image)

                character_profile = request_kwargs["character_profile"]
                db = CharacterDatabase()
                consistency_scores = db.validate_consistency(image_tensor, character_profile.character_id)

                print("🎭 Character Consistency Results:")
                print(f"   Overall Score: {consistency_scores['overall_consistency']:.3f}")
                print(f"   Level: {consistency_scores['consistency_level']}")

                if consistency_scores["overall_consistency"] < 0.7:
                    print("⚠️  Low character consistency detected. Consider regenerating with different parameters.")

            except Exception as e:
                print(f"⚠️  Could not validate character consistency: {e}")

        if result.issues_detected:
            print("⚠️  Issues detected:")
            for issue in result.issues_detected:
                print(f"  - {issue}")

        if result.optimization_applied:
            print("🔧 Optimizations applied:")
            for opt in result.optimization_applied:
                print(f"  - {opt}")

        # Save metadata if requested
        if args.save_metadata:
            metadata_path = output_path.replace(".png", "_metadata.json")
            metadata = {
                "prompt_used": result.prompt_used,
                "negative_prompt_used": result.negative_prompt_used,
                "generation_params": result.generation_params,
                "quality_analysis": result.quality_analysis,
                "processing_steps": result.processing_steps,
            }

            # Add character consistency data if available
            if args.character and "character_profile" in request_kwargs:
                metadata["character_consistency"] = {
                    "character_id": request_kwargs["character_id"],
                    "character_name": request_kwargs["character_name"],
                    "consistency_scores": consistency_scores if "consistency_scores" in locals() else None,
                }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"📄 Metadata saved to {metadata_path}")

    except Exception as e:
        print(f"❌ Generation failed: {e}")


def cmd_character_management(args):
    """Handle character management commands."""
    if not args.char_action:
        print("Error: No character action specified. Use --help for available actions.")
        return

    # This function delegates to specific character action handlers
    # The actual work is done by the individual command functions


def cmd_create_character_profile(args):
    """Create a new character profile with consistency features."""
    from utils.consistency.character_consistency import CharacterDatabase, PhysicalFeatures, StylePreferences

    print(f"🎭 Creating character profile: {args.name}")

    # Initialize character database
    db = CharacterDatabase()

    # Create physical features
    physical_features = PhysicalFeatures(
        face_shape=args.face_shape,
        eye_color=args.eye_color,
        hair_color=args.hair_color,
        hair_style=args.hair_style,
        height=args.height,
        build=args.build,
    )

    # Create style preferences
    style_preferences = StylePreferences(clothing_style=args.clothing_style, color_palette=args.color_palette)

    try:
        # Create character profile
        profile = db.create_character(
            name=args.name,
            reference_images=args.references,
            physical_features=physical_features,
            style_preferences=style_preferences,
        )

        print("✅ Character profile created successfully!")
        print(f"   Character ID: {profile.character_id}")
        print(f"   Name: {profile.name}")
        print(f"   Reference Images: {len(profile.reference_images)}")
        print(f"   Face Embedding: {'✓' if profile.face_embedding is not None else '✗'}")
        print(f"   Body Embedding: {'✓' if profile.body_embedding is not None else '✗'}")

        # Save to output file if specified
        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)
            print(f"   Profile saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error creating character profile: {e}")


def cmd_list_characters(args):
    """List all character profiles."""
    from utils.consistency.character_consistency import CharacterDatabase

    print("🎭 Character Profiles:")

    # Initialize database
    db = CharacterDatabase()

    # Apply filters
    filters = {}
    if args.filter_name:
        filters["name"] = args.filter_name
    if args.min_consistency:
        filters["min_consistency_score"] = args.min_consistency

    # Get characters
    characters = db.list_characters(filters)

    if not characters:
        print("   No characters found.")
        return

    # Display characters
    print(f"{'ID':<12} {'Name':<20} {'Refs':<5} {'Consistency':<12} {'Created':<12}")
    print("-" * 70)

    for char in characters:
        consistency_score = f"{char.consistency_score:.2f}" if char.consistency_score > 0 else "N/A"
        created_date = char.created_date[:10] if char.created_date else "Unknown"

        print(
            f"{char.character_id:<12} {char.name:<20} {len(char.reference_images):<5} "
            f"{consistency_score:<12} {created_date:<12}"
        )

    # Save to output file if specified
    if args.output:
        import json

        character_data = [char.to_dict() for char in characters]
        with open(args.output, "w") as f:
            json.dump(character_data, f, indent=2)
        print(f"\n📄 Character list saved to: {args.output}")


def cmd_update_character(args):
    """Update an existing character profile."""
    from utils.consistency.character_consistency import CharacterDatabase

    print(f"🎭 Updating character: {args.character_id}")

    # Initialize database
    db = CharacterDatabase()

    # Build updates dictionary
    updates = {}
    if args.name:
        updates["name"] = args.name
    if args.face_shape:
        updates["physical_features.face_shape"] = args.face_shape
    if args.eye_color:
        updates["physical_features.eye_color"] = args.eye_color
    if args.hair_color:
        updates["physical_features.hair_color"] = args.hair_color
    if args.clothing_style:
        updates["style_preferences.clothing_style"] = args.clothing_style
    if args.color_palette:
        updates["style_preferences.color_palette"] = args.color_palette

    # Handle reference image updates
    if args.add_references or args.remove_references:
        character = db.get_character(args.character_id)
        if not character:
            print(f"❌ Character {args.character_id} not found")
            return

        current_refs = character.reference_images.copy()

        if args.add_references:
            current_refs.extend(args.add_references)

        if args.remove_references:
            current_refs = [ref for ref in current_refs if ref not in args.remove_references]

        updates["reference_images"] = current_refs

    try:
        # Update character
        updated_profile = db.update_character(args.character_id, updates)

        print("✅ Character updated successfully!")
        print(f"   Character ID: {updated_profile.character_id}")
        print(f"   Name: {updated_profile.name}")
        print(f"   Last Updated: {updated_profile.last_updated}")

    except Exception as e:
        print(f"❌ Error updating character: {e}")


def cmd_delete_character(args):
    """Delete a character profile."""
    from utils.consistency.character_consistency import CharacterDatabase

    if not args.confirm:
        print(f"⚠️  Are you sure you want to delete character {args.character_id}?")
        print("   Use --confirm to proceed with deletion.")
        return

    print(f"🗑️  Deleting character: {args.character_id}")

    # Initialize database
    db = CharacterDatabase()

    try:
        success = db.delete_character(args.character_id)

        if success:
            print(f"✅ Character {args.character_id} deleted successfully!")
        else:
            print(f"❌ Character {args.character_id} not found")

    except Exception as e:
        print(f"❌ Error deleting character: {e}")


def cmd_validate_character_consistency(args):
    """Validate character consistency in an image."""
    import torchvision.transforms as transforms
    from PIL import Image
    from utils.consistency.character_consistency import CharacterDatabase

    print("🔍 Validating character consistency:")
    print(f"   Character ID: {args.character_id}")
    print(f"   Image: {args.image}")

    # Initialize database
    db = CharacterDatabase()

    try:
        # Load and preprocess image
        image = Image.open(args.image).convert("RGB")
        transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        image_tensor = transform(image)

        # Validate consistency
        scores = db.validate_consistency(image_tensor, args.character_id)

        print("\n📊 Consistency Results:")
        print(f"   Face Similarity: {scores['face_similarity']:.3f}")
        print(f"   Body Similarity: {scores['body_similarity']:.3f}")
        print(f"   Color Consistency: {scores['color_consistency']:.3f}")
        print(f"   Overall Score: {scores['overall_consistency']:.3f}")
        print(f"   Consistency Level: {scores['consistency_level']}")

        # Interpretation
        if scores["overall_consistency"] >= 0.9:
            print("✅ Excellent character consistency!")
        elif scores["overall_consistency"] >= 0.8:
            print("✅ Good character consistency")
        elif scores["overall_consistency"] >= 0.7:
            print("⚠️  Fair character consistency - some improvements needed")
        else:
            print("❌ Poor character consistency - significant issues detected")

        # Save results if specified
        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(scores, f, indent=2)
            print(f"\n📄 Validation results saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error validating character consistency: {e}")


def cmd_character_statistics(args):
    """Show character database statistics."""
    from utils.consistency.character_consistency import CharacterDatabase

    print("📊 Character Database Statistics:")

    # Initialize database
    db = CharacterDatabase()

    characters = db.list_characters()

    if not characters:
        print("   No characters in database.")
        return

    # Calculate statistics
    total_characters = len(characters)
    total_references = sum(len(char.reference_images) for char in characters)
    avg_references = total_references / total_characters if total_characters > 0 else 0

    characters_with_embeddings = sum(
        1 for char in characters if char.face_embedding is not None or char.body_embedding is not None
    )

    consistency_scores = [char.consistency_score for char in characters if char.consistency_score > 0]
    avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0

    # Display statistics
    print(f"   Total Characters: {total_characters}")
    print(f"   Total Reference Images: {total_references}")
    print(f"   Average References per Character: {avg_references:.1f}")
    print(f"   Characters with Embeddings: {characters_with_embeddings}")
    print(f"   Average Consistency Score: {avg_consistency:.3f}")

    # Character breakdown by features
    face_shapes = {}
    eye_colors = {}
    hair_colors = {}

    for char in characters:
        if char.physical_features:
            face_shapes[char.physical_features.face_shape] = face_shapes.get(char.physical_features.face_shape, 0) + 1
            eye_colors[char.physical_features.eye_color] = eye_colors.get(char.physical_features.eye_color, 0) + 1
            hair_colors[char.physical_features.hair_color] = hair_colors.get(char.physical_features.hair_color, 0) + 1

    if face_shapes:
        print("\n   Face Shape Distribution:")
        for shape, count in sorted(face_shapes.items()):
            print(f"     {shape}: {count}")

    if eye_colors:
        print("\n   Eye Color Distribution:")
        for color, count in sorted(eye_colors.items()):
            print(f"     {color}: {count}")

    # Save statistics if specified
    if args.output:
        import json

        stats = {
            "total_characters": total_characters,
            "total_references": total_references,
            "average_references": avg_references,
            "characters_with_embeddings": characters_with_embeddings,
            "average_consistency": avg_consistency,
            "face_shapes": face_shapes,
            "eye_colors": eye_colors,
            "hair_colors": hair_colors,
        }

        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n📄 Statistics saved to: {args.output}")


def cmd_style_management(args):
    """Handle style management commands."""
    if not args.style_action:
        print("Error: No style action specified. Use --help for available actions.")
        return

    # This function delegates to specific style action handlers


def cmd_analyze_styles(args):
    """Analyze style conflicts in prompt and LoRAs."""
    from utils.consistency.style_harmonization import create_style_harmonization_system

    print("🎨 Analyzing style conflicts:")
    print(f"   Prompt: {args.prompt}")

    # Parse LoRA configurations
    lora_configs = []
    if args.loras:
        for lora_spec in args.loras:
            if ":" in lora_spec:
                name, strength = lora_spec.split(":", 1)
                try:
                    strength = float(strength)
                except ValueError:
                    strength = 1.0
            else:
                name, strength = lora_spec, 1.0

            lora_configs.append({"name": name, "strength": strength})

    # Parse embedding configurations
    embedding_configs = []
    if args.embeddings:
        for emb_spec in args.embeddings:
            if ":" in emb_spec:
                name, strength = emb_spec.split(":", 1)
                try:
                    strength = float(strength)
                except ValueError:
                    strength = 1.0
            else:
                name, strength = emb_spec, 1.0

            embedding_configs.append({"name": name, "strength": strength})

    try:
        # Analyze styles
        harmonizer = create_style_harmonization_system()
        result = harmonizer.harmonize_styles(
            prompt=args.prompt,
            lora_configs=lora_configs,
            embeddings=embedding_configs,
            user_preferences={"harmonization_mode": "analyze_only"},
        )

        print("\n📊 Style Analysis Results:")
        print(f"   Dominant Style: {result['style_analysis']['dominant_style']}")
        print(f"   Conflict Level: {result['style_analysis']['conflict_level']}")
        print(f"   Number of Conflicts: {result['style_analysis']['num_conflicts']}")
        print(f"   Harmonization Needed: {'Yes' if result['style_analysis']['harmonization_applied'] else 'No'}")

        if result["detected_styles"]:
            print("\n🎭 Detected Styles:")
            for style in result["detected_styles"]:
                print(f"     {style['name']} ({style['source']}): {style['type']} (strength: {style['strength']:.2f})")

        if result["conflicts"]:
            print("\n⚠️  Style Conflicts:")
            for conflict in result["conflicts"]:
                print(
                    f"     {conflict['style1']} ↔ {conflict['style2']}: {conflict['severity']} "
                    f"(score: {conflict['conflict_score']:.2f})"
                )

        # Provide recommendations
        if result["style_analysis"]["conflict_level"] != "none":
            print("\n💡 Recommendations:")
            print("   - Consider using style harmonization to resolve conflicts")
            print(f"   - Focus on the dominant {result['style_analysis']['dominant_style']} style")
            if result["style_analysis"]["conflict_level"] in ["severe", "incompatible"]:
                print("   - Remove or reduce conflicting style elements")
                print("   - Use bridging terms to blend compatible styles")
        else:
            print("\n✅ No conflicts detected - styles work well together!")

        # Save results if specified
        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n📄 Analysis results saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error analyzing styles: {e}")


def cmd_harmonize_styles(args):
    """Harmonize conflicting styles."""
    from utils.consistency.style_harmonization import create_style_harmonization_system

    print("🎨 Harmonizing styles:")
    print(f"   Original Prompt: {args.prompt}")

    # Parse configurations (same as analyze)
    lora_configs = []
    if args.loras:
        for lora_spec in args.loras:
            if ":" in lora_spec:
                name, strength = lora_spec.split(":", 1)
                try:
                    strength = float(strength)
                except ValueError:
                    strength = 1.0
            else:
                name, strength = lora_spec, 1.0

            lora_configs.append({"name": name, "strength": strength})

    embedding_configs = []
    if args.embeddings:
        for emb_spec in args.embeddings:
            if ":" in emb_spec:
                name, strength = emb_spec.split(":", 1)
                try:
                    strength = float(strength)
                except ValueError:
                    strength = 1.0
            else:
                name, strength = emb_spec, 1.0

            embedding_configs.append({"name": name, "strength": strength})

    try:
        # Set up user preferences
        user_preferences = {
            "harmonization_mode": args.mode,
            "max_strength_reduction": args.max_reduction,
            "allow_prompt_modification": args.allow_prompt_changes,
            "allow_negative_prompts": True,
        }

        # Harmonize styles
        harmonizer = create_style_harmonization_system()
        result = harmonizer.harmonize_styles(
            prompt=args.prompt,
            lora_configs=lora_configs,
            embeddings=embedding_configs,
            user_preferences=user_preferences,
        )

        print("\n✨ Harmonization Results:")
        print(f"   Harmonized Prompt: {result['harmonized_prompt']}")

        if result["changes_made"]:
            print("\n🔧 Changes Made:")
            for change in result["changes_made"]:
                print(f"     - {change}")
        else:
            print("\n✅ No changes needed - styles already harmonious!")

        # Show adjusted LoRAs
        if result["adjusted_loras"] != lora_configs:
            print("\n🎛️  Adjusted LoRAs:")
            for lora in result["adjusted_loras"]:
                original_lora = next((item for item in lora_configs if item["name"] == lora["name"]), None)
                if original_lora and original_lora["strength"] != lora["strength"]:
                    print(f"     {lora['name']}: {original_lora['strength']:.2f} → {lora['strength']:.2f}")
                else:
                    print(f"     {lora['name']}: {lora['strength']:.2f}")

        # Show style analysis
        print("\n📊 Style Analysis:")
        print(f"   Conflict Level: {result['style_analysis']['conflict_level']}")
        print(f"   Dominant Style: {result['style_analysis']['dominant_style']}")

        # Save results if specified
        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n📄 Harmonization results saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error harmonizing styles: {e}")


def cmd_check_style_compatibility(args):
    """Check compatibility between different style types."""
    from utils.consistency.style_harmonization import StyleCompatibilityMatrix, StyleType

    print("🔍 Checking style compatibility:")

    # Parse style types
    style_types = []
    for style_name in args.styles:
        try:
            style_type = StyleType(style_name.lower())
            style_types.append(style_type)
        except ValueError:
            print(f"⚠️  Unknown style type: {style_name}")
            print(f"   Available types: {[s.value for s in StyleType]}")
            continue

    if len(style_types) < 2:
        print("❌ Need at least 2 valid style types for compatibility check")
        return

    try:
        compatibility_matrix = StyleCompatibilityMatrix()

        print("\n📊 Compatibility Matrix:")
        print(f"{'Style 1':<15} {'Style 2':<15} {'Compatibility':<12} {'Status'}")
        print("-" * 60)

        compatibility_results = []

        for i, style1 in enumerate(style_types):
            for style2 in style_types[i + 1 :]:
                compatibility = compatibility_matrix.get_compatibility(style1, style2)

                if compatibility >= 0.8:
                    status = "Excellent"
                elif compatibility >= 0.6:
                    status = "Good"
                elif compatibility >= 0.4:
                    status = "Fair"
                elif compatibility >= 0.2:
                    status = "Poor"
                else:
                    status = "Incompatible"

                print(f"{style1.value:<15} {style2.value:<15} {compatibility:<12.2f} {status}")

                compatibility_results.append(
                    {"style1": style1.value, "style2": style2.value, "compatibility": compatibility, "status": status}
                )

        # Provide recommendations
        print("\n💡 Recommendations:")
        excellent_pairs = [r for r in compatibility_results if r["compatibility"] >= 0.8]
        poor_pairs = [r for r in compatibility_results if r["compatibility"] < 0.3]

        if excellent_pairs:
            print("   ✅ Highly compatible combinations:")
            for pair in excellent_pairs:
                print(f"     - {pair['style1']} + {pair['style2']}")

        if poor_pairs:
            print("   ⚠️  Avoid these combinations:")
            for pair in poor_pairs:
                print(f"     - {pair['style1']} + {pair['style2']} (use harmonization)")

        # Save results if specified
        if args.output:
            import json

            results = {
                "compatibility_matrix": compatibility_results,
                "summary": {
                    "total_pairs": len(compatibility_results),
                    "excellent_pairs": len(excellent_pairs),
                    "poor_pairs": len(poor_pairs),
                },
            }
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Compatibility results saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error checking compatibility: {e}")


def cmd_create_character(args):
    """Create character profile for consistency."""
    print(f"👤 Creating character: {args.name}")

    try:
        master = create_sdx_master()
        profile = master.create_character(args.name, args.description, args.reference_prompt)

        print(f"✅ Character '{args.name}' created successfully")
        print(f"Physical features: {profile.get('physical_features', {})}")
        print(f"Style tags: {profile.get('style_tags', [])}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(profile, f, indent=2)
            print(f"📄 Profile saved to {args.output}")

    except Exception as e:
        print(f"❌ Character creation failed: {e}")


def cmd_create_style(args):
    """Create style profile for consistency."""
    print(f"🎨 Creating style: {args.name}")

    try:
        master = create_sdx_master()
        profile = master.create_style(args.name, args.description, args.reference_prompt)

        print(f"✅ Style '{args.name}' created successfully")
        print(f"Color palette: {profile.get('color_palette', [])}")
        print(f"Artistic elements: {profile.get('artistic_elements', [])}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(profile, f, indent=2)
            print(f"📄 Profile saved to {args.output}")

    except Exception as e:
        print(f"❌ Style creation failed: {e}")


def cmd_validate_setup(args):
    """Validate complete SDX setup."""
    print("🔍 Validating SDX setup...")

    try:
        master = create_sdx_master(args.config if hasattr(args, "config") else None)

        if hasattr(args, "checkpoint") and args.checkpoint:
            master.load_model(checkpoint_path=args.checkpoint)

        validation = master.validate_setup()

        print("\n📊 Setup Validation Results:")
        print(f"Config loaded: {'✅' if validation['config_loaded'] else '❌'}")
        print(f"Model loaded: {'✅' if validation['model_loaded'] else '❌'}")
        print(f"Diffusion loaded: {'✅' if validation['diffusion_loaded'] else '❌'}")
        print(f"Advanced systems: {'✅' if validation['advanced_systems'] else '❌'}")
        print(f"Multimodal ready: {'✅' if validation['multimodal_ready'] else '❌'}")

        print(f"\nReady for training: {'✅' if validation['ready_for_training'] else '❌'}")
        print(f"Ready for inference: {'✅' if validation['ready_for_inference'] else '❌'}")

        if validation["issues"]:
            print("\n⚠️  Issues found:")
            for issue in validation["issues"]:
                print(f"  - {issue}")

        if validation["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in validation["recommendations"]:
                print(f"  - {rec}")

    except Exception as e:
        print(f"❌ Validation failed: {e}")


def cmd_get_statistics(args):
    """Get comprehensive system statistics."""
    print("📊 Gathering statistics...")

    try:
        master = create_sdx_master()

        if hasattr(args, "checkpoint") and args.checkpoint:
            master.load_model(checkpoint_path=args.checkpoint)

        stats = master.get_statistics()

        print("\n🖥️  System Information:")
        system_info = stats.get("system_info", {})
        print(f"PyTorch version: {system_info.get('torch_version', 'Unknown')}")
        print(f"CUDA available: {system_info.get('cuda_available', False)}")
        print(f"GPU count: {system_info.get('gpu_count', 0)}")

        if stats.get("model_info"):
            print("\n🤖 Model Information:")
            model_info = stats["model_info"]
            print(f"Total parameters: {model_info.get('total_parameters', 0):,}")
            print(f"Model size: {model_info.get('model_size_mb', 0):.1f} MB")

        if stats.get("generation_stats"):
            print("\n🎨 Generation Statistics:")
            gen_stats = stats["generation_stats"]
            print(f"Total generations: {gen_stats.get('total_generations', 0)}")
            print(f"Successful generations: {gen_stats.get('successful_generations', 0)}")
            print(f"Average quality score: {gen_stats.get('average_quality_score', 0):.2f}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"📄 Statistics saved to {args.output}")

    except Exception as e:
        print(f"❌ Statistics gathering failed: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SDX - Comprehensive CLI for diffusion model training and inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Dataset analysis
    dataset_parser = subparsers.add_parser("analyze-dataset", help="Analyze dataset quality")
    dataset_parser.add_argument("--data-path", help="Path to dataset directory")
    dataset_parser.add_argument("--manifest", help="Path to JSONL manifest file")
    dataset_parser.add_argument("--output", help="Output file for report")
    dataset_parser.set_defaults(func=cmd_analyze_dataset)

    # Config validation
    config_parser = subparsers.add_parser("validate-config", help="Validate training configuration")
    config_parser.add_argument("config", help="Path to config file")
    config_parser.add_argument("--estimate-memory", action="store_true", help="Estimate memory usage")
    config_parser.add_argument("--suggest-optimizations", action="store_true", help="Suggest optimizations")
    config_parser.set_defaults(func=cmd_validate_config)

    # Checkpoint management
    ckpt_parser = subparsers.add_parser("checkpoints", help="Manage checkpoints")
    ckpt_parser.add_argument("action", choices=["list", "cleanup", "compare", "analyze"])
    ckpt_parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory")
    ckpt_parser.add_argument("--sort-by", choices=["step", "loss", "timestamp"], default="step")
    ckpt_parser.add_argument("--keep-best", type=int, default=3, help="Number of best checkpoints to keep")
    ckpt_parser.add_argument("--keep-recent", type=int, default=5, help="Number of recent checkpoints to keep")
    ckpt_parser.add_argument("--checkpoints", nargs="+", help="Checkpoint names for comparison/analysis")
    ckpt_parser.add_argument("--output", help="Output file for analysis results")
    ckpt_parser.set_defaults(func=cmd_manage_checkpoints)

    # Checkpoint merging
    merge_parser = subparsers.add_parser("merge-checkpoints", help="Merge multiple checkpoints")
    merge_parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths to merge")
    merge_parser.add_argument("--output", required=True, help="Output path for merged checkpoint")
    merge_parser.add_argument("--weights", help="Comma-separated weights for merging")
    merge_parser.add_argument(
        "--method", choices=["weighted_average", "max", "min"], default="weighted_average", help="Merge method"
    )
    merge_parser.set_defaults(func=cmd_merge_checkpoints)

    # Prompt optimization
    prompt_parser = subparsers.add_parser("optimize-prompt", help="Optimize prompts")
    prompt_parser.add_argument("--prompt", help="Single prompt to optimize")
    prompt_parser.add_argument("--file", help="File with prompts to optimize")
    prompt_parser.add_argument("--output", help="Output file for optimized prompts")
    prompt_parser.add_argument("--style", choices=["photorealistic", "anime", "artistic", "3d"])
    prompt_parser.add_argument("--negative", help="Negative prompt to optimize")
    prompt_parser.add_argument("--no-quality", action="store_true", help="Don't add quality tags")
    prompt_parser.add_argument("--no-boost", action="store_true", help="Don't boost subject tags")
    prompt_parser.set_defaults(func=cmd_optimize_prompt)

    # Model analysis
    model_parser = subparsers.add_parser("analyze-model", help="Analyze model architecture")
    model_parser.add_argument("--model", default="DiT-XL/2-Text", help="Model name")
    model_parser.add_argument("--checkpoint", help="Load model from checkpoint")
    model_parser.add_argument("--output", help="Output file for analysis")
    model_parser.set_defaults(func=cmd_analyze_model)

    # Generate image
    generate_parser = subparsers.add_parser("generate", help="Generate image with advanced features")
    generate_parser.add_argument("prompt", help="Text prompt for generation")
    generate_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    generate_parser.add_argument("--output", help="Output image path")
    generate_parser.add_argument("--width", type=int, default=512, help="Image width")
    generate_parser.add_argument("--height", type=int, default=512, help="Image height")
    generate_parser.add_argument("--steps", type=int, default=20, help="Number of steps")
    generate_parser.add_argument("--cfg-scale", type=float, default=7.5, help="CFG scale")
    generate_parser.add_argument("--seed", type=int, help="Random seed")
    generate_parser.add_argument("--negative", help="Negative prompt")
    generate_parser.add_argument("--character", help="Character name for consistency")
    generate_parser.add_argument("--style", help="Style name for consistency")
    generate_parser.add_argument("--scene", help="Scene ID for consistency")
    generate_parser.add_argument("--loras", nargs="+", help="LoRA names and strengths (format: name:strength)")
    generate_parser.add_argument("--harmonize-styles", action="store_true", help="Apply style harmonization")
    generate_parser.add_argument(
        "--harmonization-mode",
        choices=["balanced", "preserve_dominant", "blend_all"],
        default="balanced",
        help="Style harmonization mode",
    )
    generate_parser.add_argument("--precision-control", action="store_true", help="Use precision control")
    generate_parser.add_argument("--anatomy-correction", action="store_true", help="Use anatomy correction")
    generate_parser.add_argument("--has-text", action="store_true", help="Image contains text")
    generate_parser.add_argument("--no-enhance", action="store_true", help="Skip image enhancement")
    generate_parser.add_argument("--quality", choices=["low", "medium", "high"], default="high", help="Quality level")
    generate_parser.add_argument("--save-metadata", action="store_true", help="Save generation metadata")
    generate_parser.set_defaults(func=cmd_generate_image)

    # Character consistency management
    char_mgmt_parser = subparsers.add_parser("character", help="Manage character profiles")
    char_mgmt_subparsers = char_mgmt_parser.add_subparsers(dest="char_action", help="Character actions")

    # Create character
    char_create_parser = char_mgmt_subparsers.add_parser("create", help="Create character profile")
    char_create_parser.add_argument("name", help="Character name")
    char_create_parser.add_argument("--references", nargs="+", required=True, help="Reference image paths")
    char_create_parser.add_argument(
        "--face-shape", choices=["oval", "round", "square", "heart", "diamond"], default="oval"
    )
    char_create_parser.add_argument("--eye-color", choices=["brown", "blue", "green", "hazel", "gray"], default="brown")
    char_create_parser.add_argument("--hair-color", default="brown")
    char_create_parser.add_argument("--hair-style", default="medium")
    char_create_parser.add_argument("--height", choices=["short", "average", "tall"], default="average")
    char_create_parser.add_argument("--build", choices=["slim", "average", "athletic", "heavy"], default="average")
    char_create_parser.add_argument("--clothing-style", default="casual")
    char_create_parser.add_argument("--color-palette", nargs="+", default=["blue", "white", "black"])
    char_create_parser.add_argument("--output", help="Output file for character profile")
    char_create_parser.set_defaults(func=cmd_create_character_profile)

    # List characters
    char_list_parser = char_mgmt_subparsers.add_parser("list", help="List character profiles")
    char_list_parser.add_argument("--filter-name", help="Filter by character name")
    char_list_parser.add_argument("--min-consistency", type=float, help="Minimum consistency score")
    char_list_parser.add_argument("--output", help="Output file for character list")
    char_list_parser.set_defaults(func=cmd_list_characters)

    # Update character
    char_update_parser = char_mgmt_subparsers.add_parser("update", help="Update character profile")
    char_update_parser.add_argument("character_id", help="Character ID to update")
    char_update_parser.add_argument("--name", help="New character name")
    char_update_parser.add_argument("--add-references", nargs="+", help="Add reference images")
    char_update_parser.add_argument("--remove-references", nargs="+", help="Remove reference images")
    char_update_parser.add_argument("--face-shape", choices=["oval", "round", "square", "heart", "diamond"])
    char_update_parser.add_argument("--eye-color", choices=["brown", "blue", "green", "hazel", "gray"])
    char_update_parser.add_argument("--hair-color")
    char_update_parser.add_argument("--clothing-style")
    char_update_parser.add_argument("--color-palette", nargs="+")
    char_update_parser.set_defaults(func=cmd_update_character)

    # Delete character
    char_delete_parser = char_mgmt_subparsers.add_parser("delete", help="Delete character profile")
    char_delete_parser.add_argument("character_id", help="Character ID to delete")
    char_delete_parser.add_argument("--confirm", action="store_true", help="Confirm deletion")
    char_delete_parser.set_defaults(func=cmd_delete_character)

    # Validate character consistency
    char_validate_parser = char_mgmt_subparsers.add_parser("validate", help="Validate character consistency")
    char_validate_parser.add_argument("character_id", help="Character ID")
    char_validate_parser.add_argument("image", help="Image path to validate")
    char_validate_parser.add_argument("--output", help="Output file for validation results")
    char_validate_parser.set_defaults(func=cmd_validate_character_consistency)

    # Character statistics
    char_stats_parser = char_mgmt_subparsers.add_parser("stats", help="Character database statistics")
    char_stats_parser.add_argument("--output", help="Output file for statistics")
    char_stats_parser.set_defaults(func=cmd_character_statistics)

    char_mgmt_parser.set_defaults(func=cmd_character_management)

    # Style harmonization management
    style_parser = subparsers.add_parser("style", help="Manage style harmonization")
    style_subparsers = style_parser.add_subparsers(dest="style_action", help="Style actions")

    # Analyze styles
    style_analyze_parser = style_subparsers.add_parser("analyze", help="Analyze style conflicts")
    style_analyze_parser.add_argument("prompt", help="Text prompt to analyze")
    style_analyze_parser.add_argument("--loras", nargs="+", help="LoRA names and strengths (format: name:strength)")
    style_analyze_parser.add_argument("--embeddings", nargs="+", help="Embedding names and strengths")
    style_analyze_parser.add_argument("--output", help="Output file for analysis results")
    style_analyze_parser.set_defaults(func=cmd_analyze_styles)

    # Harmonize styles
    style_harmonize_parser = style_subparsers.add_parser("harmonize", help="Harmonize conflicting styles")
    style_harmonize_parser.add_argument("prompt", help="Text prompt to harmonize")
    style_harmonize_parser.add_argument("--loras", nargs="+", help="LoRA names and strengths")
    style_harmonize_parser.add_argument("--embeddings", nargs="+", help="Embedding names and strengths")
    style_harmonize_parser.add_argument(
        "--mode", choices=["balanced", "preserve_dominant", "blend_all"], default="balanced", help="Harmonization mode"
    )
    style_harmonize_parser.add_argument(
        "--max-reduction", type=float, default=0.5, help="Maximum strength reduction allowed"
    )
    style_harmonize_parser.add_argument(
        "--allow-prompt-changes", action="store_true", help="Allow modifications to the prompt"
    )
    style_harmonize_parser.add_argument("--output", help="Output file for harmonized result")
    style_harmonize_parser.set_defaults(func=cmd_harmonize_styles)

    # Style compatibility check
    style_compat_parser = style_subparsers.add_parser("compatibility", help="Check style compatibility")
    style_compat_parser.add_argument("styles", nargs="+", help="Style types to check compatibility")
    style_compat_parser.add_argument("--output", help="Output file for compatibility matrix")
    style_compat_parser.set_defaults(func=cmd_check_style_compatibility)

    style_parser.set_defaults(func=cmd_style_management)

    # Create style
    style_parser = subparsers.add_parser("create-style", help="Create style profile")
    style_parser.add_argument("name", help="Style name")
    style_parser.add_argument("description", help="Style description")
    style_parser.add_argument("--reference-prompt", help="Reference prompt for style")
    style_parser.add_argument("--output", help="Output file for style profile")
    style_parser.set_defaults(func=cmd_create_style)

    # Validate setup
    validate_parser = subparsers.add_parser("validate-setup", help="Validate SDX setup")
    validate_parser.add_argument("--config", help="Config file path")
    validate_parser.add_argument("--checkpoint", help="Checkpoint to validate")
    validate_parser.set_defaults(func=cmd_validate_setup)

    # Get statistics
    stats_parser = subparsers.add_parser("statistics", help="Get system statistics")
    stats_parser.add_argument("--checkpoint", help="Model checkpoint for model stats")
    stats_parser.add_argument("--output", help="Output file for statistics")
    stats_parser.set_defaults(func=cmd_get_statistics)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    from utils.training.error_handling import setup_logging

    setup_logging(level=20)  # INFO level

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(args, "debug") and args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
