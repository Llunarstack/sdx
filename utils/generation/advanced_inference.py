"""
Advanced inference utilities: batch processing, prompt optimization, and quality enhancement.
Integrated with precision control, anatomy correction, consistency management, and advanced prompting.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

_log = logging.getLogger(__name__)


class PromptOptimizer:
    """Optimize prompts for better generation quality."""

    def __init__(self):
        self.quality_tags = [
            "masterpiece",
            "best quality",
            "high quality",
            "highres",
            "8k",
            "ultra detailed",
            "absurdres",
            "detailed",
            "sharp focus",
            "professional",
            "perfect composition",
        ]

        self.negative_defaults = [
            "worst quality",
            "low quality",
            "blurry",
            "bad anatomy",
            "bad hands",
            "missing fingers",
            "extra fingers",
            "cropped",
            "jpeg artifacts",
            "signature",
            "watermark",
            "username",
        ]

        self.style_modifiers = {
            "photorealistic": ["photorealistic", "realistic", "photo", "hyperrealistic"],
            "anime": ["anime", "manga", "cel shading", "anime style"],
            "artistic": ["digital art", "concept art", "illustration", "painting"],
            "3d": ["3d render", "octane render", "cinema 4d", "blender"],
        }

    def optimize_prompt(
        self, prompt: str, style: Optional[str] = None, add_quality: bool = True, boost_subject: bool = True
    ) -> str:
        """Optimize a prompt for better generation."""
        # Clean and normalize prompt
        prompt = prompt.strip()

        # Split into tags if comma-separated
        if "," in prompt:
            tags = [tag.strip() for tag in prompt.split(",")]
        else:
            tags = [prompt]

        optimized_tags = []

        # Add quality tags at the beginning if requested
        if add_quality and not any(qt in prompt.lower() for qt in self.quality_tags[:3]):
            optimized_tags.extend(["masterpiece", "best quality", "high quality"])

        # Process existing tags
        for tag in tags:
            if tag:
                # Boost important subject tags
                if boost_subject and self._is_subject_tag(tag):
                    if not tag.startswith("(") and not tag.startswith("(("):
                        tag = f"({tag})"

                optimized_tags.append(tag)

        # Add style modifiers if specified
        if style and style in self.style_modifiers:
            style_tags = self.style_modifiers[style]
            for style_tag in style_tags[:2]:  # Add top 2 style tags
                if style_tag not in prompt.lower():
                    optimized_tags.append(style_tag)

        return ", ".join(optimized_tags)

    def optimize_negative_prompt(self, negative_prompt: str = "") -> str:
        """Optimize negative prompt with common negative terms."""
        negative_tags = []

        # Add existing negative prompt
        if negative_prompt:
            if "," in negative_prompt:
                negative_tags.extend([tag.strip() for tag in negative_prompt.split(",")])
            else:
                negative_tags.append(negative_prompt.strip())

        # Add default negative terms if not present
        for default_neg in self.negative_defaults:
            if default_neg not in negative_prompt.lower():
                negative_tags.append(default_neg)

        return ", ".join(negative_tags)

    def _is_subject_tag(self, tag: str) -> bool:
        """Check if a tag describes the main subject."""
        subject_indicators = [
            "girl",
            "boy",
            "woman",
            "man",
            "person",
            "character",
            "anime",
            "portrait",
            "face",
            "solo",
            "1girl",
            "1boy",
            "2girls",
            "2boys",
        ]

        tag_lower = tag.lower().strip("()[]")
        return any(indicator in tag_lower for indicator in subject_indicators)

    def suggest_improvements(self, prompt: str) -> List[str]:
        """Suggest improvements for a prompt."""
        suggestions = []

        # Check for quality tags
        if not any(qt in prompt.lower() for qt in self.quality_tags[:5]):
            suggestions.append("Add quality tags like 'masterpiece, best quality'")

        # Check for emphasis
        if not re.search(r"[()[\]]", prompt):
            suggestions.append("Use emphasis syntax: (important) or [less important]")

        # Check for style specification
        if not any(style in prompt.lower() for styles in self.style_modifiers.values() for style in styles):
            suggestions.append("Consider specifying art style (anime, photorealistic, etc.)")

        # Check prompt length
        if len(prompt) < 30:
            suggestions.append("Prompt is quite short - consider adding more details")
        elif len(prompt) > 300:
            suggestions.append("Prompt is very long - consider focusing on key elements")

        return suggestions


class BatchInference:
    """Handle batch inference with progress tracking and error handling."""

    def __init__(
        self,
        model,
        diffusion,
        tokenizer,
        text_encoder,
        vae,
        device,
        *,
        generate_fn: Optional[Callable[..., Image.Image]] = None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = device
        self._generate_fn = generate_fn

    def generate_batch(
        self, prompts: List[str], negative_prompts: Optional[List[str]] = None, **generation_kwargs
    ) -> List[Image.Image]:
        """Generate images for a batch of prompts."""
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)

        if len(negative_prompts) != len(prompts):
            raise ValueError("Number of negative prompts must match number of prompts")

        images = []

        for i, (prompt, neg_prompt) in enumerate(zip(prompts, negative_prompts)):
            try:
                _log.info("Generating image %s/%s: %s...", i + 1, len(prompts), prompt[:50])
                image = self._generate_single(prompt, neg_prompt, **generation_kwargs)
                images.append(image)

            except Exception as e:
                _log.error("Error generating image %s: %s", i + 1, e)
                error_image = Image.new("RGB", (512, 512), color="red")
                images.append(error_image)

        return images

    def _generate_single(self, prompt: str, negative_prompt: str = "", **kwargs) -> Image.Image:
        if self._generate_fn is not None:
            # Caller supplies a callable that closes over model/diffusion/VAE/etc., or accepts **kwargs.
            return self._generate_fn(prompt, negative_prompt, **kwargs)
        from utils.generation.simple_latent_generate import sample_one_image_pil

        return sample_one_image_pil(
            model=self.model,
            diffusion=self.diffusion,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            device=self.device,
            prompt=prompt,
            negative_prompt=negative_prompt,
            **kwargs,
        )

    def process_prompt_file(self, prompt_file: str, output_dir: str, **generation_kwargs):
        """Process prompts from a file and save generated images."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load prompts
        prompts = []
        negative_prompts = []

        with open(prompt_file, "r") as f:
            if prompt_file.endswith(".json"):
                data = json.load(f)
                for item in data:
                    prompts.append(item.get("prompt", ""))
                    negative_prompts.append(item.get("negative_prompt", ""))
            else:
                # Text file, one prompt per line
                for line in f:
                    line = line.strip()
                    if line:
                        if "|" in line:  # Format: prompt | negative_prompt
                            parts = line.split("|", 1)
                            prompts.append(parts[0].strip())
                            negative_prompts.append(parts[1].strip())
                        else:
                            prompts.append(line)
                            negative_prompts.append("")

        # Generate images
        images = self.generate_batch(prompts, negative_prompts, **generation_kwargs)

        # Save images
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            # Create safe filename from prompt
            safe_name = re.sub(r"[^\w\s-]", "", prompt[:50]).strip()
            safe_name = re.sub(r"[-\s]+", "-", safe_name)

            image_path = output_path / f"{i:04d}_{safe_name}.png"
            image.save(image_path)

            # Save metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompts[i] if i < len(negative_prompts) else "",
                "generation_kwargs": generation_kwargs,
                "image_path": str(image_path),
            }

            metadata_path = output_path / f"{i:04d}_{safe_name}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        _log.info("Generated %s images in %s", len(images), output_dir)


class ImageEnhancer:
    """Post-process generated images for quality enhancement."""

    @staticmethod
    def sharpen_image(image: Image.Image, strength: float = 1.0) -> Image.Image:
        """Sharpen image using unsharp mask."""
        if strength <= 0:
            return image

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply unsharp mask
        blurred = image.filter(ImageFilter.GaussianBlur(radius=1.0))

        # Create sharpened image
        sharpened = Image.blend(image, blurred, -strength)

        return sharpened

    @staticmethod
    def enhance_contrast(image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Enhance image contrast."""
        if factor == 1.0:
            return image

        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def enhance_colors(image: Image.Image, saturation: float = 1.1, brightness: float = 1.0) -> Image.Image:
        """Enhance image colors and brightness."""
        result = image

        if saturation != 1.0:
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(saturation)

        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(brightness)

        return result

    @staticmethod
    def reduce_noise(image: Image.Image, strength: int = 1) -> Image.Image:
        """Reduce image noise using median filter."""
        if strength <= 0:
            return image

        # Apply median filter for noise reduction
        return image.filter(ImageFilter.MedianFilter(size=strength * 2 + 1))

    @staticmethod
    def auto_enhance(
        image: Image.Image,
        sharpen: float = 0.5,
        contrast: float = 1.1,
        saturation: float = 1.05,
        brightness: float = 1.0,
    ) -> Image.Image:
        """Apply automatic enhancement pipeline."""
        result = image

        # Apply enhancements in order
        result = ImageEnhancer.enhance_contrast(result, contrast)
        result = ImageEnhancer.enhance_colors(result, saturation, brightness)
        result = ImageEnhancer.sharpen_image(result, sharpen)

        return result


class QualityAnalyzer:
    """Analyze generated image quality and provide feedback."""

    @staticmethod
    def calculate_sharpness(image: Image.Image) -> float:
        """Calculate image sharpness using Laplacian variance."""
        # Convert to grayscale
        gray = image.convert("L")

        # Convert to numpy array
        img_array = np.array(gray)

        # Calculate Laplacian
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        # Apply convolution
        from scipy import ndimage

        laplacian_img = ndimage.convolve(img_array, laplacian)

        # Calculate variance (higher = sharper)
        return float(np.var(laplacian_img))

    @staticmethod
    def calculate_contrast(image: Image.Image) -> float:
        """Calculate image contrast using standard deviation."""
        # Convert to grayscale
        gray = image.convert("L")

        # Convert to numpy array
        img_array = np.array(gray)

        # Calculate standard deviation (higher = more contrast)
        return float(np.std(img_array))

    @staticmethod
    def detect_artifacts(image: Image.Image) -> Dict[str, float]:
        """Detect common generation artifacts."""
        artifacts = {"blur_score": 0.0, "noise_score": 0.0, "compression_score": 0.0}

        # Blur detection (inverse of sharpness)
        sharpness = QualityAnalyzer.calculate_sharpness(image)
        artifacts["blur_score"] = max(0, 1.0 - sharpness / 1000.0)  # Normalize

        # Noise detection (high frequency content)
        gray = np.array(image.convert("L"))
        median = np.array(image.convert("L").filter(ImageFilter.MedianFilter(size=3)))
        noise_estimate = np.std(gray - median)
        artifacts["noise_score"] = min(1.0, noise_estimate / 50.0)  # Normalize

        # Compression artifacts (simplified detection)
        # Look for blocking artifacts by analyzing 8x8 blocks
        h, w = gray.shape
        block_variance = []

        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = gray[i : i + 8, j : j + 8]
                block_variance.append(np.var(block))

        if block_variance:
            compression_indicator = np.std(block_variance) / (np.mean(block_variance) + 1e-8)
            artifacts["compression_score"] = min(1.0, compression_indicator / 10.0)

        return artifacts

    @staticmethod
    def analyze_quality(image: Image.Image) -> Dict[str, Any]:
        """Comprehensive quality analysis."""
        analysis = {
            "sharpness": QualityAnalyzer.calculate_sharpness(image),
            "contrast": QualityAnalyzer.calculate_contrast(image),
            "artifacts": QualityAnalyzer.detect_artifacts(image),
            "resolution": image.size,
            "aspect_ratio": image.size[0] / image.size[1],
        }

        # Overall quality score (0-100)
        sharpness_score = min(100, analysis["sharpness"] / 10.0)
        contrast_score = min(100, analysis["contrast"] / 2.0)
        artifact_penalty = sum(analysis["artifacts"].values()) * 20

        analysis["quality_score"] = max(0, (sharpness_score + contrast_score) / 2 - artifact_penalty)

        # Quality rating
        if analysis["quality_score"] >= 80:
            analysis["quality_rating"] = "Excellent"
        elif analysis["quality_score"] >= 60:
            analysis["quality_rating"] = "Good"
        elif analysis["quality_score"] >= 40:
            analysis["quality_rating"] = "Fair"
        else:
            analysis["quality_rating"] = "Poor"

        return analysis


def create_image_grid(
    images: List[Image.Image], cols: int = 4, padding: int = 10, background_color: str = "white"
) -> Image.Image:
    """Create a grid of images."""
    if not images:
        raise ValueError("No images provided")

    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols

    # Get image size (assume all images are the same size)
    img_width, img_height = images[0].size

    # Calculate grid size
    grid_width = cols * img_width + (cols + 1) * padding
    grid_height = rows * img_height + (rows + 1) * padding

    # Create grid image
    grid = Image.new("RGB", (grid_width, grid_height), background_color)

    # Place images in grid
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        x = col * (img_width + padding) + padding
        y = row * (img_height + padding) + padding

        grid.paste(img, (x, y))

    return grid
