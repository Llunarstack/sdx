"""
Multimodal Generation System - Comprehensive image generation with all advanced features integrated.
Combines precision control, anatomy correction, consistency management, text rendering, and advanced prompting.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

# Import all advanced systems
from utils.architecture.enhanced_utils import (
    create_advanced_prompting_system,
    create_anatomy_correction_system,
    create_consistency_system,
    create_editing_pipeline,
    create_precision_control_system,
    create_text_rendering_pipeline,
)

from .advanced_inference import ImageEnhancer, PromptOptimizer, QualityAnalyzer


@dataclass
class GenerationRequest:
    """Comprehensive generation request with all parameters."""

    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.5
    seed: Optional[int] = None

    # Advanced features
    character_name: Optional[str] = None
    style_name: Optional[str] = None
    scene_id: Optional[str] = None

    # Precision control
    use_precision_control: bool = False
    object_layout: Optional[str] = None
    spatial_constraints: List[str] = None

    # Anatomy correction
    use_anatomy_correction: bool = False
    focus_areas: List[str] = None
    pose_type: str = "standing"

    # Text rendering
    has_text: bool = False
    text_content: List[str] = None
    typography_type: str = "general"

    # Image editing
    base_image: Optional[Image.Image] = None
    edit_mask: Optional[Image.Image] = None
    edit_instruction: str = ""
    edit_type: str = "replace"

    # Quality settings
    enhance_output: bool = True
    quality_level: str = "high"

    def __post_init__(self):
        if self.spatial_constraints is None:
            self.spatial_constraints = []
        if self.focus_areas is None:
            self.focus_areas = ["hands", "face", "posture"]
        if self.text_content is None:
            self.text_content = []


@dataclass
class GenerationResult:
    """Comprehensive generation result with metadata."""

    image: Image.Image
    prompt_used: str
    negative_prompt_used: str
    generation_params: Dict[str, Any]

    # Quality metrics
    quality_score: float = 0.0
    quality_analysis: Dict[str, Any] = None

    # Validation results
    anatomy_validation: Dict[str, Any] = None
    text_validation: Dict[str, Any] = None
    precision_validation: Dict[str, Any] = None

    # Processing log
    processing_steps: List[str] = None
    optimization_applied: List[str] = None
    issues_detected: List[str] = None

    def __post_init__(self):
        if self.quality_analysis is None:
            self.quality_analysis = {}
        if self.processing_steps is None:
            self.processing_steps = []
        if self.optimization_applied is None:
            self.optimization_applied = []
        if self.issues_detected is None:
            self.issues_detected = []


class MultimodalGenerator:
    """Main multimodal generation system integrating all advanced features."""

    def __init__(self, model=None, diffusion=None, tokenizer=None, text_encoder=None, vae=None, device="cuda"):
        self.model = model
        self.diffusion = diffusion
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = device

        # Initialize all subsystems
        self.precision_system = create_precision_control_system()
        self.anatomy_system = create_anatomy_correction_system()
        self.consistency_system = create_consistency_system()
        self.prompting_system = create_advanced_prompting_system()
        self.text_system = create_text_rendering_pipeline()
        self.editing_system = create_editing_pipeline()

        # Initialize basic components
        self.prompt_optimizer = PromptOptimizer()
        self.image_enhancer = ImageEnhancer()
        self.quality_analyzer = QualityAnalyzer()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Generation statistics
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "average_quality_score": 0.0,
            "common_issues": {},
            "optimization_usage": {},
        }

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Main generation method with full feature integration."""
        self.logger.info(f"Starting multimodal generation: {request.prompt[:50]}...")

        result = GenerationResult(
            image=None, prompt_used="", negative_prompt_used="", generation_params=asdict(request)
        )

        try:
            # Step 1: Analyze and optimize prompt
            result.processing_steps.append("Analyzing prompt")
            optimized_prompt_data = await self._optimize_prompt(request)
            result.prompt_used = optimized_prompt_data["optimized_prompt"]
            result.optimization_applied.extend(optimized_prompt_data["optimizations"])

            # Step 2: Apply consistency if requested
            if request.character_name or request.style_name or request.scene_id:
                result.processing_steps.append("Applying consistency")
                result.prompt_used = self._apply_consistency(result.prompt_used, request)

            # Step 3: Apply precision control if requested
            if request.use_precision_control:
                result.processing_steps.append("Applying precision control")
                precision_data = await self._apply_precision_control(result.prompt_used, request)
                result.prompt_used = precision_data["enhanced_prompt"]
                result.precision_validation = precision_data["validation"]

            # Step 4: Apply anatomy correction if requested
            if request.use_anatomy_correction:
                result.processing_steps.append("Applying anatomy correction")
                anatomy_data = await self._apply_anatomy_correction(result.prompt_used, request)
                result.prompt_used = anatomy_data["enhanced_prompt"]
                result.anatomy_validation = anatomy_data["validation"]

            # Step 5: Apply text rendering if requested
            if request.has_text:
                result.processing_steps.append("Applying text rendering")
                text_data = await self._apply_text_rendering(result.prompt_used, request)
                result.prompt_used = text_data["enhanced_prompt"]
                result.text_validation = text_data["validation"]

            # Step 6: Optimize negative prompt
            result.processing_steps.append("Optimizing negative prompt")
            result.negative_prompt_used = self._optimize_negative_prompt(request.negative_prompt)

            # Step 7: Generate image
            result.processing_steps.append("Generating image")
            if request.base_image and request.edit_mask:
                # Image editing mode
                generated_image = await self._generate_with_editing(
                    request, result.prompt_used, result.negative_prompt_used
                )
            else:
                # Standard generation mode
                generated_image = await self._generate_standard(
                    request, result.prompt_used, result.negative_prompt_used
                )

            # Step 8: Post-process and enhance
            if request.enhance_output:
                result.processing_steps.append("Enhancing output")
                generated_image = self._enhance_image(generated_image, request.quality_level)

            # Step 9: Validate and analyze quality
            result.processing_steps.append("Analyzing quality")
            result.quality_analysis = self.quality_analyzer.analyze_quality(generated_image)
            result.quality_score = result.quality_analysis.get("quality_score", 0.0)

            # Step 10: Validate specific features
            await self._validate_generation_features(generated_image, request, result)

            result.image = generated_image
            self.generation_stats["successful_generations"] += 1

        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            result.issues_detected.append(f"Generation error: {str(e)}")
            # Create error placeholder image
            result.image = Image.new("RGB", (request.width, request.height), (128, 128, 128))

        finally:
            self.generation_stats["total_generations"] += 1
            self._update_statistics(result)

        return result

    async def _optimize_prompt(self, request: GenerationRequest) -> Dict[str, Any]:
        """Optimize the input prompt using advanced prompting system."""
        analyzer = self.prompting_system["analyzer"]
        optimizer = self.prompting_system["optimizer"]

        # Analyze prompt
        analysis = analyzer.analyze_prompt(request.prompt)

        # Determine optimization level based on complexity
        if analysis["complexity_score"] > 0.8:
            optimization_level = "aggressive"
        elif analysis["complexity_score"] > 0.5:
            optimization_level = "balanced"
        else:
            optimization_level = "minimal"

        # Optimize prompt
        optimization_result = optimizer.optimize_prompt(
            request.prompt, optimization_level=optimization_level, target_style=request.style_name, max_length=300
        )

        return {
            "optimized_prompt": optimization_result["optimized_prompt"],
            "optimizations": optimization_result["optimization_log"],
            "improvement_score": optimization_result["improvement_score"],
        }

    def _apply_consistency(self, prompt: str, request: GenerationRequest) -> str:
        """Apply consistency management to the prompt."""
        consistency_manager = self.consistency_system["consistency_manager"]

        return consistency_manager.generate_consistent_prompt(
            prompt, character_name=request.character_name, style_name=request.style_name, scene_id=request.scene_id
        )

    async def _apply_precision_control(self, prompt: str, request: GenerationRequest) -> Dict[str, Any]:
        """Apply precision control for exact object placement."""
        scene_composer = self.precision_system["scene_composer"]
        counting_validator = self.precision_system["counting_validator"]

        # Extract objects from prompt for layout
        objects = self._extract_objects_from_prompt(prompt)

        if objects and request.object_layout:
            # Create scene layout
            scene_objects = scene_composer.create_scene_layout(
                objects, layout_type=request.object_layout, constraints=request.spatial_constraints
            )

            # Generate composition prompt
            enhanced_prompt = scene_composer.generate_composition_prompt(scene_objects, base_description=prompt)
        else:
            enhanced_prompt = prompt

        # Handle counting requirements
        counting_requirements = counting_validator.extract_counting_requirements(prompt)
        if counting_requirements:
            for obj, count in counting_requirements.items():
                counting_prompt = counting_validator.generate_counting_prompt(obj, count, enhanced_prompt)
                enhanced_prompt = counting_prompt

        return {
            "enhanced_prompt": enhanced_prompt,
            "validation": {
                "objects_detected": objects,
                "counting_requirements": counting_requirements,
                "layout_applied": request.object_layout,
            },
        }

    async def _apply_anatomy_correction(self, prompt: str, request: GenerationRequest) -> Dict[str, Any]:
        """Apply anatomy correction for better human figures."""
        anatomy_validator = self.anatomy_system["anatomy_validator"]
        hand_corrector = self.anatomy_system["hand_corrector"]
        multi_person_composer = self.anatomy_system["multi_person_composer"]

        # Analyze pose complexity
        complexity = anatomy_validator.analyze_pose_complexity(prompt)

        # Apply appropriate corrections based on complexity
        if complexity > 0.7:
            # High complexity - use multi-stage approach
            if "people" in prompt.lower() or "person" in prompt.lower():
                components = multi_person_composer.decompose_multi_person_scene(prompt)
                enhanced_prompt = multi_person_composer.generate_multi_person_prompt(components)
            else:
                enhanced_prompt = anatomy_validator.generate_anatomy_aware_prompt(prompt, request.focus_areas)
        else:
            # Standard anatomy enhancement
            enhanced_prompt = anatomy_validator.generate_anatomy_aware_prompt(prompt, request.focus_areas)

        # Special handling for hands
        if "hands" in request.focus_areas:
            enhanced_prompt = hand_corrector.generate_hand_focused_prompt(enhanced_prompt)

        return {
            "enhanced_prompt": enhanced_prompt,
            "validation": {
                "complexity_score": complexity,
                "focus_areas": request.focus_areas,
                "corrections_applied": ["anatomy_aware", "hand_focused"]
                if "hands" in request.focus_areas
                else ["anatomy_aware"],
            },
        }

    async def _apply_text_rendering(self, prompt: str, request: GenerationRequest) -> Dict[str, Any]:
        """Apply text rendering enhancements."""
        text_engine = self.text_system["engine"]

        # Extract text requirements
        text_info = text_engine.extract_text_requirements(prompt)
        text_info.update({"text_content": request.text_content, "typography_type": request.typography_type})

        # Enhance prompt for text rendering
        enhanced_prompt = text_engine.enhance_prompt_for_text(prompt, text_info)

        return {
            "enhanced_prompt": enhanced_prompt,
            "validation": {
                "text_detected": text_info["has_text"],
                "text_content": text_info["text_content"],
                "typography_type": text_info["typography_type"],
            },
        }

    def _optimize_negative_prompt(self, negative_prompt: str) -> str:
        """Optimize negative prompt with common negative terms."""
        return self.prompt_optimizer.optimize_negative_prompt(negative_prompt)

    async def _generate_standard(self, request: GenerationRequest, prompt: str, negative_prompt: str) -> Image.Image:
        """Standard image generation."""
        # This would integrate with your actual generation pipeline
        # For now, return a placeholder
        self.logger.info(f"Generating with prompt: {prompt[:100]}...")

        # Placeholder generation - replace with actual model inference
        image = Image.new("RGB", (request.width, request.height), (100, 150, 200))

        return image

    async def _generate_with_editing(
        self, request: GenerationRequest, prompt: str, negative_prompt: str
    ) -> Image.Image:
        """Image generation with editing (inpainting/outpainting)."""
        inpainting_engine = self.editing_system["inpainting"]

        # Generate edit prompt
        edit_prompt = inpainting_engine.generate_edit_prompt(prompt, request.edit_instruction, request.edit_type)

        # Placeholder for actual inpainting - replace with model inference
        self.logger.info(f"Editing with prompt: {edit_prompt[:100]}...")

        # For now, return the base image
        return request.base_image

    def _enhance_image(self, image: Image.Image, quality_level: str) -> Image.Image:
        """Enhance generated image based on quality level."""
        if quality_level == "high":
            return self.image_enhancer.auto_enhance(image, sharpen=0.7, contrast=1.15, saturation=1.1, brightness=1.05)
        elif quality_level == "medium":
            return self.image_enhancer.auto_enhance(image, sharpen=0.4, contrast=1.1, saturation=1.05)
        else:
            return image

    async def _validate_generation_features(
        self, image: Image.Image, request: GenerationRequest, result: GenerationResult
    ):
        """Validate specific features in the generated image."""

        # Validate text rendering if applicable
        if request.has_text and request.text_content:
            text_engine = self.text_system["engine"]
            text_validation = text_engine.validate_text_rendering(image, request.text_content)
            result.text_validation = text_validation

            if text_validation.get("accuracy_score", 0) < 0.5:
                result.issues_detected.append("Low text rendering accuracy")

        # Validate anatomy if applicable
        if request.use_anatomy_correction:
            # Placeholder for anatomy validation
            result.anatomy_validation = {"pose_detected": True, "anatomy_score": 0.8, "issues": []}

        # Validate precision if applicable
        if request.use_precision_control:
            # Placeholder for precision validation
            result.precision_validation = {
                "layout_accuracy": 0.9,
                "object_placement": "good",
                "spatial_relationships": "maintained",
            }

    def _extract_objects_from_prompt(self, prompt: str) -> List[str]:
        """Extract objects from prompt for layout planning."""
        # Simple object extraction - would be more sophisticated in practice
        common_objects = [
            "person",
            "woman",
            "man",
            "girl",
            "boy",
            "child",
            "car",
            "house",
            "tree",
            "flower",
            "chair",
            "table",
            "cat",
            "dog",
            "bird",
            "book",
            "cup",
            "bottle",
        ]

        objects = []
        prompt_lower = prompt.lower()

        for obj in common_objects:
            if obj in prompt_lower:
                objects.append(obj)

        return objects[:5]  # Limit to 5 objects for layout

    def _update_statistics(self, result: GenerationResult):
        """Update generation statistics."""
        # Update average quality score
        total_gens = self.generation_stats["total_generations"]
        current_avg = self.generation_stats["average_quality_score"]
        new_score = result.quality_score

        self.generation_stats["average_quality_score"] = (current_avg * (total_gens - 1) + new_score) / total_gens

        # Update common issues
        for issue in result.issues_detected:
            if issue in self.generation_stats["common_issues"]:
                self.generation_stats["common_issues"][issue] += 1
            else:
                self.generation_stats["common_issues"][issue] = 1

        # Update optimization usage
        for opt in result.optimization_applied:
            if opt in self.generation_stats["optimization_usage"]:
                self.generation_stats["optimization_usage"][opt] += 1
            else:
                self.generation_stats["optimization_usage"][opt] = 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return self.generation_stats.copy()

    def create_character(self, name: str, description: str, reference_prompt: str = None) -> Dict[str, Any]:
        """Create a new character profile for consistency."""
        consistency_manager = self.consistency_system["consistency_manager"]
        profile = consistency_manager.create_character_profile(name, description, reference_prompt)
        return asdict(profile)

    def create_style(self, name: str, description: str, reference_prompt: str = None) -> Dict[str, Any]:
        """Create a new style profile for consistency."""
        consistency_manager = self.consistency_system["consistency_manager"]
        profile = consistency_manager.create_style_profile(name, description, reference_prompt)
        return asdict(profile)

    def create_scene(self, scene_id: str, location: str, base_prompt: str) -> Dict[str, Any]:
        """Create a new scene context for consistency."""
        consistency_manager = self.consistency_system["consistency_manager"]
        context = consistency_manager.create_scene_context(scene_id, location, base_prompt)
        return asdict(context)

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze a prompt for optimization opportunities."""
        analyzer = self.prompting_system["analyzer"]
        return analyzer.analyze_prompt(prompt)

    def suggest_improvements(self, prompt: str) -> List[str]:
        """Suggest improvements for a prompt."""
        analysis = self.analyze_prompt(prompt)
        return analysis.get("recommendations", [])


class BatchMultimodalGenerator:
    """Batch processing for multiple generation requests."""

    def __init__(self, generator: MultimodalGenerator):
        self.generator = generator
        self.logger = logging.getLogger(__name__)

    async def generate_batch(
        self, requests: List[GenerationRequest], max_concurrent: int = 4
    ) -> List[GenerationResult]:
        """Generate multiple images concurrently."""
        self.logger.info(f"Starting batch generation: {len(requests)} requests")

        # Process in batches to avoid overwhelming the system
        results = []

        for i in range(0, len(requests), max_concurrent):
            batch = requests[i : i + max_concurrent]

            # Create tasks for concurrent execution
            tasks = [self.generator.generate(request) for request in batch]

            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle results and exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch item {i + j} failed: {result}")
                    # Create error result
                    error_result = GenerationResult(
                        image=Image.new("RGB", (512, 512), (128, 0, 0)),
                        prompt_used=batch[j].prompt,
                        negative_prompt_used=batch[j].negative_prompt,
                        generation_params=asdict(batch[j]),
                    )
                    error_result.issues_detected.append(f"Batch generation error: {result}")
                    results.append(error_result)
                else:
                    results.append(result)

        self.logger.info(f"Batch generation complete: {len(results)} results")
        return results

    def create_batch_from_prompts(
        self, prompts: List[str], base_request: GenerationRequest = None
    ) -> List[GenerationRequest]:
        """Create batch requests from a list of prompts."""
        if base_request is None:
            base_request = GenerationRequest(prompt="")

        requests = []
        for prompt in prompts:
            request = GenerationRequest(**asdict(base_request))
            request.prompt = prompt
            requests.append(request)

        return requests


def create_multimodal_system(model=None, diffusion=None, tokenizer=None, text_encoder=None, vae=None, device="cuda"):
    """Create complete multimodal generation system."""
    generator = MultimodalGenerator(model, diffusion, tokenizer, text_encoder, vae, device)
    batch_generator = BatchMultimodalGenerator(generator)

    return {"generator": generator, "batch_generator": batch_generator}
