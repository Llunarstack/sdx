"""One-shot rename: innovations subfolders + files to cleaner names."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / "innovations"

FOLDER_RENAMES = {
    "ultra_quality": "quality",
    "semantic_understanding": "semantics",
    "fine_control": "control",
    "speed_optimization": "speed",
    "advanced_features": "capabilities",
}

FILE_RENAMES: dict[str, dict[str, str]] = {
    "quality": {
        "bridge.py": "hooks.py",
        "photorealism_engine.py": "engine.py",
        "subpixel_refinement.py": "subpixel.py",
        "metallic_material.py": "metallic.py",
        "skin_texture.py": "skin.py",
        "cloth_fabric.py": "cloth.py",
        "liquid_physics.py": "liquid.py",
        "global_illumination.py": "global_light.py",
    },
    "semantics": {
        "bridge.py": "hooks.py",
        "semantic_parser.py": "engine.py",
        "semantic_decomposer.py": "decomposer.py",
        "nuance_capture.py": "nuance.py",
        "ambiguity_resolver.py": "ambiguity.py",
        "style_parser.py": "style.py",
    },
    "control": {
        "bridge.py": "hooks.py",
        "precision_control.py": "engine.py",
        "spatial_layout.py": "spatial.py",
        "color_palette.py": "color.py",
        "detail_intensity.py": "detail.py",
        "visual_effects.py": "effects.py",
    },
    "speed": {
        "bridge.py": "hooks.py",
        "realtime_generation.py": "engine.py",
        "token_pruning.py": "token_prune.py",
        "adaptive_quality.py": "adaptive.py",
        "caching.py": "cache.py",
        "layer_skipping.py": "layer_skip.py",
        "lora_acceleration.py": "lora_accel.py",
        "tiled_generation.py": "tiling.py",
        "batched_inference.py": "batching.py",
    },
    "consistency": {
        "bridge.py": "hooks.py",
        "consistency_engine.py": "engine.py",
        "consistent_seeding.py": "seeding.py",
        "character_consistency.py": "character.py",
        "style_consistency.py": "style.py",
        "variation_control.py": "variation.py",
        "semantic_consistency.py": "semantic.py",
        "temporal_consistency.py": "temporal.py",
        "color_consistency.py": "color.py",
    },
    "multimodal": {
        "bridge.py": "hooks.py",
        "multimodal_generation.py": "engine.py",
        "image_to_image.py": "img2img.py",
        "sketch_to_image.py": "sketch2img.py",
        "text_3d_fusion.py": "text_3d.py",
        "audio_to_image.py": "audio2img.py",
        "depth_guided.py": "depth.py",
    },
    "capabilities": {
        "bridge.py": "hooks.py",
        "novel_capabilities.py": "engine.py",
        "infinite_outpainting.py": "outpainting.py",
        "realtime_inpainting.py": "inpainting.py",
        "magic_eraser.py": "eraser.py",
        "object_remixing.py": "remix.py",
        "prompt_weighting.py": "weights.py",
        "dynamic_quality.py": "dynamic.py",
    },
}

# Text replacements in source (order matters — longer first)
TEXT_REPLACEMENTS = [
    ("advanced_innovations.ultra_quality", "advanced_innovations.quality"),
    ("advanced_innovations.semantic_understanding", "advanced_innovations.semantics"),
    ("advanced_innovations.fine_control", "advanced_innovations.control"),
    ("advanced_innovations.speed_optimization", "advanced_innovations.speed"),
    ("advanced_innovations.advanced_features", "advanced_innovations.capabilities"),
    (".ultra_quality", ".quality"),
    (".semantic_understanding", ".semantics"),
    (".fine_control", ".control"),
    (".speed_optimization", ".speed"),
    (".advanced_features", ".capabilities"),
    ('"ultra_quality"', '"quality"'),
    ('"semantic_understanding"', '"semantics"'),
    ('"fine_control"', '"control"'),
    ('"speed_optimization"', '"speed"'),
    ('"advanced_features"', '"capabilities"'),
    ("photorealism_engine", "engine"),
    ("semantic_parser", "engine"),
    ("precision_control", "engine"),
    ("realtime_generation", "engine"),
    ("consistency_engine", "engine"),
    ("multimodal_generation", "engine"),
    ("novel_capabilities", "engine"),
    ("subpixel_refinement", "subpixel"),
    ("metallic_material", "metallic"),
    ("skin_texture", "skin"),
    ("cloth_fabric", "cloth"),
    ("liquid_physics", "liquid"),
    ("global_illumination", "global_light"),
    ("semantic_decomposer", "decomposer"),
    ("nuance_capture", "nuance"),
    ("ambiguity_resolver", "ambiguity"),
    ("style_parser", "style"),
    ("spatial_layout", "spatial"),
    ("color_palette", "color"),
    ("detail_intensity", "detail"),
    ("visual_effects", "effects"),
    ("token_pruning", "token_prune"),
    ("adaptive_quality", "adaptive"),
    ("layer_skipping", "layer_skip"),
    ("lora_acceleration", "lora_accel"),
    ("tiled_generation", "tiling"),
    ("batched_inference", "batching"),
    ("consistent_seeding", "seeding"),
    ("character_consistency", "character"),
    ("style_consistency", "style"),
    ("variation_control", "variation"),
    ("semantic_consistency", "semantic"),
    ("temporal_consistency", "temporal"),
    ("color_consistency", "color"),
    ("image_to_image", "img2img"),
    ("sketch_to_image", "sketch2img"),
    ("text_3d_fusion", "text_3d"),
    ("audio_to_image", "audio2img"),
    ("depth_guided", "depth"),
    ("infinite_outpainting", "outpainting"),
    ("realtime_inpainting", "inpainting"),
    ("magic_eraser", "eraser"),
    ("object_remixing", "remix"),
    ("prompt_weighting", "weights"),
    ("dynamic_quality", "dynamic"),
    (".bridge", ".hooks"),
]


def rename_folders() -> None:
    for old, new in FOLDER_RENAMES.items():
        src = AI / old
        dst = AI / new
        if src.is_dir() and not dst.exists():
            src.rename(dst)
            print(f"folder: {old} -> {new}")


def rename_files() -> None:
    for pkg, mapping in FILE_RENAMES.items():
        pkg_dir = AI / pkg
        if not pkg_dir.is_dir():
            # maybe still old folder name
            for old, new in FOLDER_RENAMES.items():
                if new == pkg and (AI / old).is_dir():
                    pkg_dir = AI / old
                    break
        if not pkg_dir.is_dir():
            print(f"skip missing package {pkg}")
            continue
        for old_name, new_name in mapping.items():
            src = pkg_dir / old_name
            dst = pkg_dir / new_name
            if src.is_file() and not dst.exists():
                src.rename(dst)
                print(f"  {pkg}/{old_name} -> {new_name}")


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    orig = text
    for old, new in TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_sources() -> None:
    targets = list((ROOT / "innovations").rglob("*.py"))
    targets += list((ROOT / "tests").glob("test_innovations.py"))
    n = 0
    for p in targets:
        if patch_file(p):
            n += 1
            print(f"patched {p.relative_to(ROOT)}")
    print(f"patched {n} files")


def main() -> None:
    rename_folders()
    rename_files()
    patch_sources()


if __name__ == "__main__":
    main()
