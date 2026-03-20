#!/usr/bin/env python3
"""
Enhanced SDX Setup Script - Automatically configure and validate the enhanced SDX environment.
"""
import sys
import subprocess
import importlib.util
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error output: {result.stderr}")
        return False
    return result.returncode == 0


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            return True
        else:
            print("⚠️  CUDA not available - CPU training will be very slow")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Install main requirements
    if not run_command("pip install -r requirements.txt"):
        print("❌ Failed to install requirements")
        return False
    
    # Install optional dependencies for enhanced features
    optional_deps = [
        "scipy",  # For image analysis
        "tqdm",   # For progress bars
        "wandb",  # For experiment tracking
        "tensorboard",  # For logging
        "safetensors",  # For safe model loading
        "omegaconf",   # For configuration management
    ]
    
    for dep in optional_deps:
        print(f"Installing {dep}...")
        run_command(f"pip install {dep}", check=False)
    
    print("✅ Dependencies installed")
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "experiments",
        "checkpoints", 
        "logs",
        "datasets",
        "outputs",
        ".kiro/steering",
        ".kiro/settings"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {dir_name}")
    
    print("✅ Directories created")


def create_example_config():
    """Create example configuration files."""
    print("\n⚙️  Creating example configurations...")
    
    # Example training config
    example_config = """# Example training configuration
from config.train_config import TrainConfig

cfg = TrainConfig(
    # Data
    data_path="./datasets/your_dataset",
    image_size=512,
    global_batch_size=32,
    
    # Model
    model_name="DiT-XL/2-Text",
    
    # Training
    passes=3,
    lr=1e-4,
    use_bf16=True,
    grad_checkpointing=True,
    
    # Validation
    val_split=0.05,
    save_best=True,
    
    # Enhanced features
    refinement_prob=0.25,
    use_xformers=True,
    use_compile=True,
)
"""
    
    with open("example_config.py", "w") as f:
        f.write(example_config)
    
    # Example environment file
    env_example = """# Copy this to .env and fill in your values
HF_TOKEN=your_huggingface_token_here
CUDA_VISIBLE_DEVICES=0
WANDB_API_KEY=your_wandb_key_here
"""
    
    if not Path(".env").exists():
        with open(".env.example", "w") as f:
            f.write(env_example)
    
    print("✅ Example configurations created")


def validate_installation():
    """Validate the installation."""
    print("\n🔍 Validating installation...")
    
    try:
        # Test required modules without unused import side effects.
        required_modules = [
            "torch",
            "transformers",
            "diffusers",
            "utils.error_handling",
            "utils.config_validator",
            "utils.data_analysis",
        ]
        for module_name in required_modules:
            if importlib.util.find_spec(module_name) is None:
                raise ImportError(f"Missing module: {module_name}")
        
        print("✅ Core imports successful")
        
        # Test CLI
        result = subprocess.run([sys.executable, "cli.py", "--help"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CLI tool working")
        else:
            print("⚠️  CLI tool may have issues")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def run_quick_test():
    """Run a quick test of the enhanced features."""
    print("\n🧪 Running quick tests...")
    
    try:
        # Test configuration validation
        from config.train_config import TrainConfig
        from utils.config_validator import validate_train_config
        
        cfg = TrainConfig(
            data_path="./datasets/test",
            model_name="DiT-XL/2-Text",
            global_batch_size=16,
            passes=1
        )
        
        issues = validate_train_config(cfg)
        print(f"   Config validation: {len(issues)} issues found (expected)")
        
        # Test prompt optimization
        from utils.advanced_inference import PromptOptimizer
        
        optimizer = PromptOptimizer()
        optimized = optimizer.optimize_prompt("a girl in a garden", style="anime")
        print(f"   Prompt optimization: '{optimized[:50]}...'")
        
        # Test dataset analyzer (without actual data)
        from utils.data_analysis import DatasetAnalyzer
        _analyzer = DatasetAnalyzer()
        print("   Dataset analyzer: initialized successfully")
        
        print("✅ Quick tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 Enhanced SDX setup complete!")
    print("\n📋 Next steps:")
    print("1. Prepare your dataset in a folder with images + .txt caption files")
    print("2. Analyze your dataset:")
    print("   python cli.py analyze-dataset --data-path ./your_dataset")
    print("3. Create and validate your config:")
    print("   cp example_config.py my_config.py")
    print("   python cli.py validate-config my_config.py --estimate-memory")
    print("4. Start training:")
    print("   python train.py --config my_config.py")
    print("5. Generate images:")
    print("   python sample.py --ckpt ./checkpoints/best.pt --prompt 'your prompt'")
    
    print("\n🔧 Enhanced features available:")
    print("- Dataset quality analysis")
    print("- Configuration validation")
    print("- Advanced checkpoint management")
    print("- Prompt optimization")
    print("- Model architecture analysis")
    print("- Training metrics tracking")
    print("- Image quality enhancement")
    
    print("\n📚 Documentation:")
    print("- Enhanced features: docs/ENHANCED_FEATURES.md")
    print("- CLI help: python cli.py --help")
    print("- Original docs: docs/README.md")


def main():
    """Main setup function."""
    print("🚀 Enhanced SDX Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check CUDA after PyTorch installation
    check_cuda()
    
    # Create directories
    create_directories()
    
    # Create example configs
    create_example_config()
    
    # Validate installation
    if not validate_installation():
        print("⚠️  Installation validation failed - some features may not work")
    
    # Run quick tests
    if not run_quick_test():
        print("⚠️  Quick tests failed - check your installation")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()