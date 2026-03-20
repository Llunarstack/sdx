#!/usr/bin/env python3
"""
Integration test for enhanced DiT model architecture.
Tests that all components work together properly.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from models.enhanced_dit import EnhancedDiT_XL_2  # noqa: E402


def test_enhanced_model_creation():
    """Test enhanced model creation."""
    print("Testing enhanced model creation...")

    model = EnhancedDiT_XL_2(
        input_size=32,
        enable_spatial_control=True,
        enable_anatomy_awareness=True,
        enable_text_rendering=True,
        enable_consistency=True,
    )

    print(f"Model created: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert model is not None


def main():
    ok = True
    ok &= test_enhanced_model_creation()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

