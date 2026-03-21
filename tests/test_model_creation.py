#!/usr/bin/env python3
"""
Test script to create and verify the 3B parameter Enhanced DiT model.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from models.enhanced_dit import EnhancedDiT_XL_2  # noqa: E402


def test_model_creation():
    """Test creating the 3B parameter model."""
    print("Testing Enhanced DiT-XL/2 Model Creation")
    print("=" * 50)

    # Create the model
    print("Creating Enhanced DiT-XL/2 model...")
    model = EnhancedDiT_XL_2(
        input_size=64,  # 512x512 image -> 64x64 latent
        enable_spatial_control=True,
        enable_anatomy_awareness=True,
        enable_text_rendering=True,
        enable_consistency=True,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model created successfully!")
    print("Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**3:.2f} GB (FP32)")
    print(f"  Model size: {total_params * 2 / 1024**3:.2f} GB (FP16)")

    # Test forward pass
    print("\nTesting forward pass...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Create test inputs
    batch_size = 1
    x = torch.randn(batch_size, 4, 64, 64, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)

    # Enhanced feature inputs
    spatial_layout = torch.randn(batch_size, 10, 4, device=device)
    anatomy_mask = torch.randn(batch_size, 64 * 64, device=device)
    text_tokens = torch.randint(0, 1000, (batch_size, 77), device=device)
    text_positions = torch.randn(batch_size, 77, 2, device=device)
    typography_style = torch.randint(0, 20, (batch_size,), device=device)
    character_id = torch.randint(0, 100, (batch_size,), device=device)
    style_id = torch.randint(0, 50, (batch_size,), device=device)

    with torch.no_grad():
        _output = model(
            x,
            t,
            y,
            spatial_layout=spatial_layout,
            anatomy_mask=anatomy_mask,
            text_tokens=text_tokens,
            text_positions=text_positions,
            typography_style=typography_style,
            character_id=character_id,
            style_id=style_id,
        )

    print("Forward pass OK.")
    assert _output is not None


if __name__ == "__main__":
    _model = test_model_creation()
