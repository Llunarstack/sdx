"""
Master Integration System - Central hub connecting all SDX components and advanced features.
Provides unified interface for all functionality with proper initialization and error handling.
"""
from pathlib import Path
from typing import Dict, Optional, Any
import torch

# Core SDX imports
try:
    from config.train_config import TrainConfig
    from models import DiT_models_text
    from diffusion import create_diffusion
except ImportError as e:
    print(f"Warning: Core SDX imports failed: {e}")
    print("Make sure you're running from the SDX root directory")

# Advanced feature imports
from .error_handling import setup_logging, validate_checkpoint, get_model_info
from .config_validator import validate_train_config, estimate_memory_usage, suggest_optimizations
from .metrics import MetricsTracker, log_system_info
from .data_analysis import DatasetAnalyzer
from .checkpoint_manager import CheckpointManager
from .multimodal_generation import create_multimodal_system, GenerationRequest, GenerationResult

# Advanced systems
from .precision_control import create_precision_control_system
from .anatomy_correction import create_anatomy_correction_system
from .consistency_system import create_consistency_system
from .advanced_prompting import create_advanced_prompting_system
from .text_rendering import create_text_rendering_pipeline
from .image_editing import create_editing_pipeline


class SDXMaster:
    """Master class integrating all SDX functionality."""
    
    def __init__(self, config_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.logger = setup_logging()
        
        # Core components
        self.model = None
        self.diffusion = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.config = None
        
        # Advanced systems
        self.precision_system = None
        self.anatomy_system = None
        self.consistency_system = None
        self.prompting_system = None
        self.text_system = None
        self.editing_system = None
        self.multimodal_system = None
        
        # Utilities
        self.metrics_tracker = None
        self.checkpoint_manager = None
        self.dataset_analyzer = None
        
        # Initialize systems
        self._initialize_advanced_systems()
        
        # Load config if provided
        if config_path:
            self.load_config(config_path)
        
        self.logger.info("SDX Master initialized successfully")
    
    def _initialize_advanced_systems(self):
        """Initialize all advanced systems."""
        try:
            self.logger.info("Initializing advanced systems...")
            
            # Core advanced systems
            self.precision_system = create_precision_control_system()
            self.anatomy_system = create_anatomy_correction_system()
            self.consistency_system = create_consistency_system()
            self.prompting_system = create_advanced_prompting_system()
            self.text_system = create_text_rendering_pipeline()
            self.editing_system = create_editing_pipeline()
            
            # Utilities
            self.dataset_analyzer = DatasetAnalyzer()
            
            self.logger.info("Advanced systems initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize advanced systems: {e}")
            raise
    
    def load_config(self, config_path: str):
        """Load training configuration."""
        try:
            if config_path.endswith('.json'):
                import json
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                self.config = TrainConfig(**config_dict)
            else:
                # Python config file
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                self.config = config_module.cfg
            
            self.logger.info(f"Configuration loaded from {config_path}")
            
            # Validate configuration
            issues = validate_train_config(self.config)
            if issues:
                self.logger.warning(f"Configuration issues found: {len(issues)}")
                for issue in issues:
                    self.logger.warning(f"  {issue}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise
    
    def load_model(self, checkpoint_path: Optional[str] = None, model_name: Optional[str] = None):
        """Load model for inference."""
        try:
            if checkpoint_path:
                self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
                
                # Validate checkpoint
                if not validate_checkpoint(checkpoint_path):
                    raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
                
                # Load checkpoint
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                self.config = ckpt.get("config")
                
                if self.config is None:
                    raise ValueError("Checkpoint must contain config")
                
                model_name = getattr(self.config, "model_name", "DiT-XL/2-Text")
                
            elif model_name:
                self.logger.info(f"Creating new model: {model_name}")
                if self.config is None:
                    self.config = TrainConfig(model_name=model_name)
            else:
                raise ValueError("Either checkpoint_path or model_name must be provided")
            
            # Get model function
            model_fn = DiT_models_text.get(model_name)
            if model_fn is None:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Create model
            from config import get_dit_build_kwargs
            build_kwargs = get_dit_build_kwargs(self.config, class_dropout_prob=0.0)
            self.model = model_fn(**build_kwargs).to(self.device)
            
            # Load weights if from checkpoint
            if checkpoint_path:
                state = ckpt.get("ema") or ckpt.get("model")
                self.model.load_state_dict(state, strict=True)
            
            self.model.eval()
            
            # Log model info
            model_info = get_model_info(self.model)
            self.logger.info(f"Model loaded: {model_info}")
            
            # Initialize multimodal system with loaded model
            self.multimodal_system = create_multimodal_system(
                model=self.model,
                diffusion=self.diffusion,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                vae=self.vae,
                device=self.device
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def load_diffusion_components(self):
        """Load T5 encoder, VAE, and diffusion components."""
        try:
            self.logger.info("Loading diffusion components...")
            
            if self.config is None:
                raise ValueError("Config must be loaded before diffusion components")
            
            # This would load T5 and VAE - placeholder for now
            self.logger.info("T5 and VAE loading would happen here")
            
            # Create diffusion
            self.diffusion = create_diffusion(
                timestep_respacing=self.config.timestep_respacing,
                noise_schedule=self.config.beta_schedule,
                model_mean_type="epsilon",
                model_var_type="learned_range",
                loss_type="mse"
            )
            
            self.logger.info("Diffusion components loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to load diffusion components: {e}")
            raise
    
    def setup_training(self, results_dir: str = "./experiments"):
        """Setup training environment."""
        try:
            if self.config is None:
                raise ValueError("Config must be loaded before training setup")
            
            # Create results directory
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize checkpoint manager
            self.checkpoint_manager = CheckpointManager(str(results_path / "checkpoints"))
            
            # Initialize metrics tracker
            self.metrics_tracker = MetricsTracker(str(results_path))
            
            # Log system info
            system_info = log_system_info()
            self.logger.info(f"System info: {system_info}")
            
            # Memory estimation
            memory_est = estimate_memory_usage(self.config)
            self.logger.info(f"Estimated memory usage: {memory_est['total_estimated_gb']:.1f}GB")
            
            # Optimization suggestions
            suggestions = suggest_optimizations(self.config)
            if suggestions:
                self.logger.info("Optimization suggestions:")
                for suggestion in suggestions:
                    self.logger.info(f"  - {suggestion}")
            
            self.logger.info("Training environment setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup training: {e}")
            raise
    
    def analyze_dataset(self, data_path: str = None, manifest_path: str = None) -> Dict[str, Any]:
        """Analyze dataset quality and statistics."""
        try:
            if data_path is None and self.config:
                data_path = self.config.data_path
            
            if manifest_path is None and self.config:
                manifest_path = self.config.manifest_jsonl
            
            self.dataset_analyzer = DatasetAnalyzer(data_path, manifest_path)
            quality_check = self.dataset_analyzer.check_data_quality()
            
            self.logger.info("Dataset analysis complete")
            return quality_check
            
        except Exception as e:
            self.logger.error(f"Dataset analysis failed: {e}")
            raise
    
    def generate_image(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate image with full advanced features."""
        try:
            if self.multimodal_system is None:
                raise ValueError("Model must be loaded before generation")
            
            # Create generation request
            request = GenerationRequest(prompt=prompt, **kwargs)
            
            # Generate using multimodal system
            import asyncio
            result = asyncio.run(self.multimodal_system['generator'].generate(request))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise
    
    def optimize_prompt(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Optimize prompt using advanced prompting system."""
        try:
            optimizer = self.prompting_system['optimizer']
            return optimizer.optimize_prompt(prompt, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Prompt optimization failed: {e}")
            raise
    
    def create_character(self, name: str, description: str, reference_prompt: str = None) -> Dict[str, Any]:
        """Create character profile for consistency."""
        try:
            if self.multimodal_system:
                return self.multimodal_system['generator'].create_character(name, description, reference_prompt)
            else:
                consistency_manager = self.consistency_system['consistency_manager']
                profile = consistency_manager.create_character_profile(name, description, reference_prompt)
                return profile.__dict__
                
        except Exception as e:
            self.logger.error(f"Character creation failed: {e}")
            raise
    
    def create_style(self, name: str, description: str, reference_prompt: str = None) -> Dict[str, Any]:
        """Create style profile for consistency."""
        try:
            if self.multimodal_system:
                return self.multimodal_system['generator'].create_style(name, description, reference_prompt)
            else:
                consistency_manager = self.consistency_system['consistency_manager']
                profile = consistency_manager.create_style_profile(name, description, reference_prompt)
                return profile.__dict__
                
        except Exception as e:
            self.logger.error(f"Style creation failed: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "system_info": log_system_info(),
            "generation_stats": {},
            "training_stats": {},
            "model_info": {}
        }
        
        try:
            if self.multimodal_system:
                stats["generation_stats"] = self.multimodal_system['generator'].get_statistics()
            
            if self.metrics_tracker:
                stats["training_stats"] = self.metrics_tracker.get_summary()
            
            if self.model:
                stats["model_info"] = get_model_info(self.model)
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
        
        return stats
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate complete setup."""
        validation = {
            "config_loaded": self.config is not None,
            "model_loaded": self.model is not None,
            "diffusion_loaded": self.diffusion is not None,
            "advanced_systems": True,  # Always true if we got this far
            "multimodal_ready": self.multimodal_system is not None,
            "issues": [],
            "recommendations": []
        }
        
        # Check for issues
        if not validation["config_loaded"]:
            validation["issues"].append("No configuration loaded")
            validation["recommendations"].append("Load config with load_config()")
        
        if not validation["model_loaded"]:
            validation["issues"].append("No model loaded")
            validation["recommendations"].append("Load model with load_model()")
        
        if not validation["diffusion_loaded"]:
            validation["issues"].append("Diffusion components not loaded")
            validation["recommendations"].append("Load diffusion with load_diffusion_components()")
        
        # GPU check
        if not torch.cuda.is_available():
            validation["issues"].append("CUDA not available")
            validation["recommendations"].append("Install CUDA for GPU acceleration")
        
        validation["ready_for_training"] = (
            validation["config_loaded"] and 
            validation["model_loaded"] and 
            validation["diffusion_loaded"]
        )
        
        validation["ready_for_inference"] = (
            validation["model_loaded"] and 
            validation["multimodal_ready"]
        )
        
        return validation


def create_sdx_master(config_path: Optional[str] = None, device: str = "cuda") -> SDXMaster:
    """Create and initialize SDX Master system."""
    return SDXMaster(config_path, device)


# Convenience functions for quick access
def quick_generate(prompt: str, checkpoint_path: str, **kwargs) -> GenerationResult:
    """Quick image generation with minimal setup."""
    master = create_sdx_master()
    master.load_model(checkpoint_path=checkpoint_path)
    return master.generate_image(prompt, **kwargs)


def quick_optimize_prompt(prompt: str, **kwargs) -> Dict[str, Any]:
    """Quick prompt optimization."""
    master = create_sdx_master()
    return master.optimize_prompt(prompt, **kwargs)


def quick_analyze_dataset(data_path: str) -> Dict[str, Any]:
    """Quick dataset analysis."""
    master = create_sdx_master()
    return master.analyze_dataset(data_path=data_path)


# CLI integration helper
def get_master_for_cli() -> SDXMaster:
    """Get master instance for CLI usage."""
    return create_sdx_master()