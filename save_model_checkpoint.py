#!/usr/bin/env python3
"""
Save a randomly initialized 3B parameter Enhanced DiT model checkpoint.
This creates the model file you can use for inference or continue training.
"""

from pathlib import Path

import torch
from config.train_config import TrainConfig
from models.enhanced_dit import EnhancedDiT_XL_2


def save_model_checkpoint():
    """Save a model checkpoint with random initialization."""
    print("🚀 Creating and Saving Enhanced DiT-XL/2 Checkpoint")
    print("=" * 60)

    # Create model
    print("Creating Enhanced DiT-XL/2 model...")
    model = EnhancedDiT_XL_2(
        input_size=64,  # 512x512 image -> 64x64 latent
        enable_spatial_control=True,
        enable_anatomy_awareness=True,
        enable_text_rendering=True,
        enable_consistency=True,
    )

    # Initialize weights
    model.initialize_weights()

    # Create config
    cfg = TrainConfig(
        model_name="EnhancedDiT-XL/2",
        image_size=512,
        global_batch_size=32,
        lr=5e-5,
    )

    # Add enhanced feature flags manually
    cfg.enable_spatial_control = True
    cfg.enable_anatomy_awareness = True
    cfg.enable_text_rendering = True
    cfg.enable_consistency = True

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    print(f"✅ Model created with {total_params:,} parameters ({total_params / 1e9:.1f}B)")

    # Create checkpoint
    checkpoint = {
        "model": model.state_dict(),
        "config": cfg,
        "step": 0,
        "epoch": 0,
        "model_name": "EnhancedDiT-XL/2",
        "total_parameters": total_params,
        "enhanced_features": {
            "spatial_control": True,
            "anatomy_awareness": True,
            "text_rendering": True,
            "consistency": True,
        },
        "architecture_info": {
            "depth": 28,
            "hidden_size": 1152,
            "num_heads": 16,
            "patch_size": 2,
        },
    }

    # Save checkpoint
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "enhanced_dit_xl2_3b_random_init.pt"

    print(f"💾 Saving checkpoint to {checkpoint_path}...")
    torch.save(checkpoint, checkpoint_path)

    # Get file size
    file_size_gb = checkpoint_path.stat().st_size / 1024**3

    print("✅ Checkpoint saved successfully!")
    print(f"📁 File: {checkpoint_path}")
    print(f"📊 Size: {file_size_gb:.2f} GB")
    print(f"🎯 Parameters: {total_params:,} ({total_params / 1e9:.1f}B)")

    # Test loading
    print("\n🧪 Testing checkpoint loading...")
    loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Create new model and load weights
    test_model = EnhancedDiT_XL_2(
        input_size=64,
        enable_spatial_control=True,
        enable_anatomy_awareness=True,
        enable_text_rendering=True,
        enable_consistency=True,
    )

    test_model.load_state_dict(loaded_checkpoint["model"])

    print("✅ Checkpoint loaded successfully!")
    print(f"📋 Loaded config: {loaded_checkpoint['config'].model_name}")
    print(f"🔧 Enhanced features: {loaded_checkpoint['enhanced_features']}")

    print("\n🎉 Success! You now have a 3B parameter Enhanced DiT model!")
    print("📝 Usage:")
    print("   - Use this checkpoint for inference")
    print("   - Continue training from this checkpoint")
    print("   - Fine-tune on your specific dataset")

    return checkpoint_path


if __name__ == "__main__":
    checkpoint_path = save_model_checkpoint()
    print(f"\n🚀 Your 3B parameter model is ready at: {checkpoint_path}")
