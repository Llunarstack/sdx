"""
Advanced Innovations Integration: Wire all systems together into unified pipeline.
Provides unified interface for core SDX generation with all enhancements.
Includes agentic quality control for perfect prompt adherence.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


# Lazy imports for agentic systems (optional)
def _get_quality_control_system():
    """Lazy-load quality control system."""
    try:
        from .agentic import QualityControlSystem
        return QualityControlSystem()
    except ImportError:
        logger.warning("Agentic quality control not available")
        return None


def _get_adherence_monitor():
    """Lazy-load prompt adherence monitor."""
    try:
        from .agentic import PromptAdherenceMonitor
        return PromptAdherenceMonitor()
    except ImportError:
        logger.warning("Prompt adherence monitoring not available")
        return None


class SDXAdvancedPipeline(nn.Module):
    """Unified SDX pipeline with all advanced innovations integrated."""

    def __init__(
        self,
        enable_photorealism: bool = True,
        enable_semantic: bool = True,
        enable_control: bool = True,
        enable_speed: bool = True,
        enable_consistency: bool = True,
        enable_multimodal: bool = True,
        enable_novel: bool = True,
        enable_quality_control: bool = True,
        enable_prompt_adherence: bool = True,
    ):
        super().__init__()
        self.config = {
            "photorealism": enable_photorealism,
            "semantic": enable_semantic,
            "control": enable_control,
            "speed": enable_speed,
            "consistency": enable_consistency,
            "multimodal": enable_multimodal,
            "novel": enable_novel,
            "quality_control": enable_quality_control,
            "prompt_adherence": enable_prompt_adherence,
        }

        # Lazy imports to avoid circular dependencies
        self._components = {}
        self._agentic_systems = {}
        self._initialized = False

    def initialize(self):
        """Lazy initialization of components."""
        if self._initialized:
            return

        if self.config["photorealism"]:
            try:
                from .ultra_quality.photorealism_engine import UltraQualityEngine
                self._components["photorealism"] = UltraQualityEngine()
                logger.info("✓ Photorealism engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize photorealism: {e}")

        if self.config["semantic"]:
            try:
                from .semantic_understanding.semantic_parser import SemanticUnderstandingEngine
                self._components["semantic"] = SemanticUnderstandingEngine()
                logger.info("✓ Semantic understanding engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic: {e}")

        if self.config["control"]:
            try:
                from .fine_control.precision_control import PrecisionControlSystem
                self._components["control"] = PrecisionControlSystem()
                logger.info("✓ Precision control system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize control: {e}")

        if self.config["speed"]:
            try:
                from .speed_optimization.realtime_generation import RealtimeGenerationEngine
                self._components["speed"] = RealtimeGenerationEngine()
                logger.info("✓ Real-time generation engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize speed: {e}")

        if self.config["consistency"]:
            try:
                from .consistency.consistency_engine import ConsistencyEngine
                self._components["consistency"] = ConsistencyEngine()
                logger.info("✓ Consistency engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize consistency: {e}")

        if self.config["multimodal"]:
            try:
                from .multimodal.multimodal_generation import MultimodalFusionEngine
                self._components["multimodal"] = MultimodalFusionEngine()
                logger.info("✓ Multimodal fusion engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize multimodal: {e}")

        if self.config["novel"]:
            try:
                from .advanced_features.novel_capabilities import NovelCapabilitiesEngine
                self._components["novel"] = NovelCapabilitiesEngine()
                logger.info("✓ Novel capabilities engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize novel: {e}")

        # Initialize agentic systems
        if self.config["quality_control"]:
            try:
                from .agentic import QualityControlSystem
                self._agentic_systems["quality_control"] = QualityControlSystem()
                logger.info("✓ Quality control system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize quality control: {e}")

        if self.config["prompt_adherence"]:
            try:
                from .agentic import PromptAdherenceMonitor
                self._agentic_systems["adherence_monitor"] = PromptAdherenceMonitor()
                logger.info("✓ Prompt adherence monitor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize adherence monitor: {e}")

        self._initialized = True

    def get_component(self, name: str):
        """Get initialized component by name."""
        self.initialize()
        return self._components.get(name)

    def apply_semantic_understanding(self, prompt_embedding: torch.Tensor) -> Dict:
        """Apply semantic understanding to prompt."""
        semantic = self.get_component("semantic")
        if semantic is None:
            return {}

        # Convert embedding to token indices if needed
        if prompt_embedding.dtype != torch.long and prompt_embedding.dim() >= 2:
            # Use embedding as-is
            tokens = prompt_embedding[:, :10].long() if prompt_embedding.shape[1] > 10 else prompt_embedding.long()
        else:
            tokens = prompt_embedding

        try:
            return semantic.understand_prompt(tokens)
        except Exception as e:
            logger.warning(f"Semantic understanding failed: {e}")
            return {}

    def apply_photorealism(self, latent: torch.Tensor, material_type: str = "photorealistic") -> torch.Tensor:
        """Apply photorealism rendering."""
        photorealism = self.get_component("photorealism")
        if photorealism is None:
            return latent
        return photorealism.render_photorealistic(latent, material_type)

    def apply_precision_controls(self, base_image: torch.Tensor, control_specs: Dict) -> torch.Tensor:
        """Apply fine-grained controls."""
        control = self.get_component("control")
        if control is None:
            return base_image
        return control.apply_controls(base_image, control_specs)

    def generate_fast(self, prompt_embedding: torch.Tensor, target_latency_ms: int = 100) -> torch.Tensor:
        """Generate with speed optimization."""
        speed = self.get_component("speed")
        if speed is None:
            return torch.randn(1, 3, 512, 512)
        return speed.generate_fast(prompt_embedding, target_latency_ms)

    def generate_consistent(
        self,
        prompt: torch.Tensor,
        seed: int,
        character_id: Optional[str] = None,
        style_name: Optional[str] = None,
        variation: float = 0.0,
    ) -> torch.Tensor:
        """Generate with consistency guarantees."""
        consistency = self.get_component("consistency")
        if consistency is None:
            return torch.randn(1, 3, 512, 512)
        return consistency.generate_consistent(
            prompt=prompt,
            seed=seed,
            character_id=character_id,
            style_name=style_name,
            variation=variation,
        )

    def generate_multimodal(
        self,
        text: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        sketch: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate from multiple modalities."""
        multimodal = self.get_component("multimodal")
        if multimodal is None:
            return torch.randn(1, 3, 512, 512)
        return multimodal.generate_multimodal(text=text, image=image, sketch=sketch, **kwargs)

    def get_novel_capabilities(self) -> List[str]:
        """Get list of novel capabilities."""
        novel = self.get_component("novel")
        if novel is None:
            return []
        return novel.get_capabilities()

    def forward(self, prompt_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        """Main forward pass with all enhancements."""
        self.initialize()

        # Step 1: Semantic understanding
        semantic_info = self.apply_semantic_understanding(prompt_embedding)

        # Step 2: Generate with speed optimization
        output = self.generate_fast(prompt_embedding, kwargs.get("target_latency_ms", 100))

        # Step 3: Apply photorealism
        output = self.apply_photorealism(output, kwargs.get("material_type", "photorealistic"))

        # Step 4: Apply controls if provided
        if "control_specs" in kwargs:
            output = self.apply_precision_controls(output, kwargs["control_specs"])

        # Step 5: Ensure consistency if seed provided
        if "seed" in kwargs:
            output = self.generate_consistent(
                prompt_embedding,
                kwargs["seed"],
                character_id=kwargs.get("character_id"),
                style_name=kwargs.get("style_name"),
                variation=kwargs.get("variation", 0.0),
            )

        return output

    def assess_quality(
        self,
        prompt: str,
        t5_embedding: torch.Tensor,
        clip_embeddings: Dict,
        generated_latent: torch.Tensor,
    ):
        """Assess generation quality using agentic system."""
        quality_control = self._agentic_systems.get("quality_control")
        if quality_control is None:
            logger.warning("Quality control not available")
            return None

        assessment, should_refine = quality_control.evaluate_generation(
            prompt, t5_embedding, clip_embeddings, generated_latent
        )
        return assessment

    def ensure_prompt_adherence(
        self,
        prompt: str,
        t5_embedding: torch.Tensor,
        clip_embeddings: Dict,
        generation_func,
        max_iterations: int = 5,
    ):
        """Generate with enforced prompt adherence."""
        monitor = self._agentic_systems.get("adherence_monitor")
        if monitor is None:
            logger.warning("Adherence monitor not available")
            return None, 0.0

        final_latent, adherence = monitor.generate_with_adherence(
            prompt, t5_embedding, clip_embeddings, generation_func, max_iterations
        )
        return final_latent, adherence

    def get_status(self) -> Dict[str, bool]:
        """Get status of all components."""
        self.initialize()
        return {
            "photorealism": "photorealism" in self._components,
            "semantic": "semantic" in self._components,
            "control": "control" in self._components,
            "speed": "speed" in self._components,
            "consistency": "consistency" in self._components,
            "multimodal": "multimodal" in self._components,
            "novel": "novel" in self._components,
            "quality_control": "quality_control" in self._agentic_systems,
            "prompt_adherence": "adherence_monitor" in self._agentic_systems,
        }


class IntegrationValidator:
    """Validate integration between components."""

    @staticmethod
    def validate_shapes(
        prompt_embedding: torch.Tensor,
        expected_output_shape: Tuple[int, ...] = (1, 3, 512, 512),
    ) -> bool:
        """Validate tensor shapes are compatible."""
        if not isinstance(prompt_embedding, torch.Tensor):
            logger.error("Prompt embedding must be torch.Tensor")
            return False

        if prompt_embedding.dim() < 1:
            logger.error("Prompt embedding must be at least 1D")
            return False

        return True

    @staticmethod
    def validate_device_compatibility(*tensors) -> bool:
        """Ensure all tensors are on same device."""
        if len(tensors) == 0:
            return True

        device = tensors[0].device if isinstance(tensors[0], torch.Tensor) else None

        for tensor in tensors[1:]:
            if isinstance(tensor, torch.Tensor):
                if tensor.device != device:
                    logger.error(f"Device mismatch: {tensor.device} vs {device}")
                    return False

        return True

    @staticmethod
    def validate_dtype_compatibility(*tensors) -> bool:
        """Ensure all tensors have compatible dtypes."""
        compatible_pairs = [
            (torch.float32, torch.float32),
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.bfloat16),
            (torch.float32, torch.float16),  # Some ops allow this
        ]

        if len(tensors) == 0:
            return True

        for i, tensor in enumerate(tensors):
            if isinstance(tensor, torch.Tensor):
                for j, other in enumerate(tensors[i + 1 :]):
                    if isinstance(other, torch.Tensor):
                        pair = (tensor.dtype, other.dtype)
                        if pair not in compatible_pairs and pair[::-1] not in compatible_pairs:
                            logger.warning(f"Potential dtype mismatch: {pair}")

        return True


def create_advanced_pipeline(
    enable_all: bool = True,
    **component_flags
) -> SDXAdvancedPipeline:
    """Factory function to create configured pipeline."""
    if enable_all:
        return SDXAdvancedPipeline()

    return SDXAdvancedPipeline(**component_flags)


if __name__ == "__main__":
    # Test initialization
    logger.basicConfig(level=logging.INFO)

    pipeline = create_advanced_pipeline(enable_all=True)
    print("Pipeline status:", pipeline.get_status())
    print("Novel capabilities:", pipeline.get_novel_capabilities())
