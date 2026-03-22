#!/usr/bin/env python3
"""
Integration test script for SDX advanced features.
Tests all systems and their connections.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path (this file lives in tests/integration/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import all required modules
from utils.advanced_prompting import create_advanced_prompting_system  # noqa: E402
from utils.anatomy_correction import create_anatomy_correction_system  # noqa: E402
from utils.consistency_system import create_consistency_system  # noqa: E402
from utils.image_editing import create_editing_pipeline  # noqa: E402
from utils.master_integration import create_sdx_master  # noqa: E402
from utils.precision_control import create_precision_control_system  # noqa: E402
from utils.text_rendering import create_text_rendering_pipeline  # noqa: E402


def test_imports():
    """Test all imports work correctly."""
    print("Testing imports...")

    # Core utilities
    from utils.advanced_inference import PromptOptimizer  # noqa: F401
    from utils.advanced_prompting import create_advanced_prompting_system  # noqa: F401
    from utils.anatomy_correction import create_anatomy_correction_system  # noqa: F401
    from utils.checkpoint_manager import CheckpointManager  # noqa: F401
    from utils.config_validator import validate_train_config  # noqa: F401
    from utils.consistency_system import create_consistency_system  # noqa: F401
    from utils.data_analysis import DatasetAnalyzer  # noqa: F401
    from utils.error_handling import setup_logging  # noqa: F401
    from utils.image_editing import create_editing_pipeline  # noqa: F401

    # Master integration
    from utils.master_integration import create_sdx_master  # noqa: F401
    from utils.metrics import MetricsTracker  # noqa: F401
    from utils.model_viz import analyze_model_architecture  # noqa: F401

    # Advanced systems
    from utils.precision_control import create_precision_control_system  # noqa: F401
    from utils.text_rendering import create_text_rendering_pipeline  # noqa: F401

    print("All imports successful")


def test_system_initialization():
    """Test system initialization."""
    print("\nTesting system initialization...")

    # Test individual systems
    _precision_system = create_precision_control_system()
    _anatomy_system = create_anatomy_correction_system()
    _consistency_system = create_consistency_system()
    _prompting_system = create_advanced_prompting_system()
    _text_system = create_text_rendering_pipeline()
    _editing_system = create_editing_pipeline()

    print("Individual systems initialized")
    assert _precision_system is not None
    assert _anatomy_system is not None
    assert _consistency_system is not None
    assert _prompting_system is not None
    assert _text_system is not None
    assert _editing_system is not None

    # Test master system
    _master = create_sdx_master()
    print("Master system initialized")
    assert _master is not None


def main():
    print("SDX Integration Test")
    print("=" * 50)

    try:
        test_imports()
        test_system_initialization()
        print("\nPASS")
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        print("\nFAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
