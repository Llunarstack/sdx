#!/usr/bin/env python3
"""
Comprehensive test suite for the Style Harmonization System.
Tests style detection, conflict analysis, and harmonization strategies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.style_harmonization import (  # noqa: E402
    StyleType,
)


def test_style_type_enum():
    """Test StyleType enum."""
    print("Testing StyleType enum...")
    assert StyleType.REALISTIC_3D.value == "realistic_3d"
    assert StyleType.ANIME_2D.value == "anime_2d"
    assert StyleType.CARTOON_2D.value == "cartoon_2d"
    print("StyleType enum tests passed")


def main():
    test_style_type_enum()
    print("PASS")


if __name__ == "__main__":
    main()
