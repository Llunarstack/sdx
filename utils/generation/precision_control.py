"""
Precision Control System - Address the core limitation of AI image models lacking true spatial reasoning.
Implements scene composition, object placement validation, and multi-stage generation for exact control.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class SceneObject:
    """Represents an object in a scene with precise positioning."""

    name: str
    position: Tuple[float, float]  # Normalized coordinates (0-1)
    size: Tuple[float, float]  # Normalized width, height
    rotation: float = 0.0
    z_order: int = 0  # Depth ordering
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class SceneComposer:
    """Compose scenes with precise object placement and relationships."""

    def __init__(self):
        self.spatial_relationships = {
            "left_of": lambda obj1, obj2: obj1.position[0] < obj2.position[0] - 0.1,
            "right_of": lambda obj1, obj2: obj1.position[0] > obj2.position[0] + 0.1,
            "above": lambda obj1, obj2: obj1.position[1] < obj2.position[1] - 0.1,
            "below": lambda obj1, obj2: obj1.position[1] > obj2.position[1] + 0.1,
            "near": lambda obj1, obj2: self._distance(obj1, obj2) < 0.2,
            "far_from": lambda obj1, obj2: self._distance(obj1, obj2) > 0.3,
            "touching": lambda obj1, obj2: self._distance(obj1, obj2) < 0.05,
            "overlapping": lambda obj1, obj2: self._overlaps(obj1, obj2),
        }

        self.layout_templates = {
            "grid": self._create_grid_layout,
            "circle": self._create_circle_layout,
            "line": self._create_line_layout,
            "pyramid": self._create_pyramid_layout,
            "random": self._create_random_layout,
        }

    def _distance(self, obj1: SceneObject, obj2: SceneObject) -> float:
        """Calculate normalized distance between two objects."""
        dx = obj1.position[0] - obj2.position[0]
        dy = obj1.position[1] - obj2.position[1]
        return math.sqrt(dx * dx + dy * dy)

    def _overlaps(self, obj1: SceneObject, obj2: SceneObject) -> bool:
        """Check if two objects overlap."""
        x1, y1 = obj1.position
        w1, h1 = obj1.size
        x2, y2 = obj2.position
        w2, h2 = obj2.size

        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def create_scene_layout(
        self, objects: List[str], layout_type: str = "balanced", constraints: List[str] = None
    ) -> List[SceneObject]:
        """Create a scene layout with precise object positioning."""
        scene_objects = []

        if layout_type in self.layout_templates:
            positions = self.layout_templates[layout_type](len(objects))
        else:
            positions = self._create_balanced_layout(len(objects))

        for i, (obj_name, pos) in enumerate(zip(objects, positions)):
            scene_obj = SceneObject(
                name=obj_name,
                position=pos,
                size=(0.1, 0.1),  # Default size
                z_order=i,
            )
            scene_objects.append(scene_obj)

        # Apply constraints
        if constraints:
            scene_objects = self._apply_constraints(scene_objects, constraints)

        return scene_objects

    def _create_grid_layout(self, num_objects: int) -> List[Tuple[float, float]]:
        """Create grid layout positions."""
        cols = math.ceil(math.sqrt(num_objects))
        rows = math.ceil(num_objects / cols)

        positions = []
        for i in range(num_objects):
            row = i // cols
            col = i % cols

            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            positions.append((x, y))

        return positions

    def _create_circle_layout(self, num_objects: int) -> List[Tuple[float, float]]:
        """Create circular layout positions."""
        positions = []
        center_x, center_y = 0.5, 0.5
        radius = 0.3

        for i in range(num_objects):
            angle = 2 * math.pi * i / num_objects
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))

        return positions

    def _create_line_layout(self, num_objects: int) -> List[Tuple[float, float]]:
        """Create horizontal line layout."""
        positions = []
        for i in range(num_objects):
            x = (i + 0.5) / num_objects
            y = 0.5
            positions.append((x, y))

        return positions

    def _create_pyramid_layout(self, num_objects: int) -> List[Tuple[float, float]]:
        """Create pyramid layout positions."""
        positions = []
        rows = math.ceil((-1 + math.sqrt(1 + 8 * num_objects)) / 2)

        obj_idx = 0
        for row in range(rows):
            objects_in_row = min(row + 1, num_objects - obj_idx)
            for col in range(objects_in_row):
                x = (col + 0.5) / objects_in_row
                y = (row + 0.5) / rows
                positions.append((x, y))
                obj_idx += 1
                if obj_idx >= num_objects:
                    break
            if obj_idx >= num_objects:
                break

        return positions

    def _create_balanced_layout(self, num_objects: int) -> List[Tuple[float, float]]:
        """Create balanced layout avoiding overlaps."""
        positions = []

        # Use golden ratio spiral for natural distribution
        golden_angle = math.pi * (3 - math.sqrt(5))

        for i in range(num_objects):
            theta = i * golden_angle
            r = math.sqrt(i) / math.sqrt(num_objects) * 0.4

            x = 0.5 + r * math.cos(theta)
            y = 0.5 + r * math.sin(theta)

            # Clamp to valid range
            x = max(0.1, min(0.9, x))
            y = max(0.1, min(0.9, y))

            positions.append((x, y))

        return positions

    def _create_random_layout(self, num_objects: int) -> List[Tuple[float, float]]:
        """Create random non-overlapping layout."""
        positions = []
        attempts = 0
        max_attempts = 1000

        while len(positions) < num_objects and attempts < max_attempts:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)

            # Check for overlaps with existing positions
            valid = True
            for existing_x, existing_y in positions:
                if abs(x - existing_x) < 0.15 and abs(y - existing_y) < 0.15:
                    valid = False
                    break

            if valid:
                positions.append((x, y))

            attempts += 1

        # Fill remaining with grid if random failed
        while len(positions) < num_objects:
            remaining = num_objects - len(positions)
            grid_positions = self._create_grid_layout(remaining)
            positions.extend(grid_positions)

        return positions[:num_objects]

    def _apply_constraints(self, objects: List[SceneObject], constraints: List[str]) -> List[SceneObject]:
        """Apply spatial constraints to scene objects."""
        # Parse and apply constraints like "A left_of B", "C above D"
        for constraint in constraints:
            parts = constraint.lower().split()
            if len(parts) >= 3:
                obj1_name = parts[0]
                relationship = parts[1]
                obj2_name = parts[2]

                obj1 = next((o for o in objects if o.name.lower() == obj1_name), None)
                obj2 = next((o for o in objects if o.name.lower() == obj2_name), None)

                if obj1 and obj2 and relationship in self.spatial_relationships:
                    self._enforce_relationship(obj1, obj2, relationship)

        return objects

    def _enforce_relationship(self, obj1: SceneObject, obj2: SceneObject, relationship: str):
        """Enforce a spatial relationship between two objects."""
        if relationship == "left_of":
            obj1.position = (obj2.position[0] - 0.2, obj1.position[1])
        elif relationship == "right_of":
            obj1.position = (obj2.position[0] + 0.2, obj1.position[1])
        elif relationship == "above":
            obj1.position = (obj1.position[0], obj2.position[1] - 0.2)
        elif relationship == "below":
            obj1.position = (obj1.position[0], obj2.position[1] + 0.2)
        elif relationship == "near":
            # Move closer but not overlapping
            dx = obj2.position[0] - obj1.position[0]
            dy = obj2.position[1] - obj1.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0.15:
                factor = 0.15 / dist
                obj1.position = (obj2.position[0] - dx * factor, obj2.position[1] - dy * factor)

    def generate_composition_prompt(self, scene_objects: List[SceneObject], base_description: str = "") -> str:
        """Generate detailed prompt for precise scene composition."""
        # Sort by z-order for proper layering description
        sorted_objects = sorted(scene_objects, key=lambda x: x.z_order)

        prompt_parts = []

        if base_description:
            prompt_parts.append(base_description)

        # Describe overall composition
        if len(scene_objects) > 1:
            prompt_parts.append("carefully composed scene")

        # Describe each object with position
        position_descriptions = []
        for obj in sorted_objects:
            x, y = obj.position

            # Convert normalized position to descriptive terms
            h_pos = "left" if x < 0.33 else "right" if x > 0.67 else "center"
            v_pos = "top" if y < 0.33 else "bottom" if y > 0.67 else "middle"

            if h_pos == "center" and v_pos == "middle":
                pos_desc = "in the center"
            elif h_pos == "center":
                pos_desc = f"in the {v_pos}"
            elif v_pos == "middle":
                pos_desc = f"on the {h_pos}"
            else:
                pos_desc = f"in the {v_pos} {h_pos}"

            position_descriptions.append(f"{obj.name} {pos_desc}")

        if position_descriptions:
            prompt_parts.append(", ".join(position_descriptions))

        # Add precision enhancers
        prompt_parts.extend(
            [
                "precise positioning",
                "exact placement",
                "clear spatial relationships",
                "well-composed",
                "balanced layout",
                "professional composition",
            ]
        )

        return ", ".join(prompt_parts)

    def create_layout_guide(
        self, scene_objects: List[SceneObject], image_size: Tuple[int, int] = (512, 512)
    ) -> Image.Image:
        """Create a visual layout guide for the scene."""
        guide = Image.new("RGB", image_size, (240, 240, 240))
        draw = ImageDraw.Draw(guide)

        width, height = image_size

        # Draw grid
        for i in range(0, width, width // 10):
            draw.line([(i, 0), (i, height)], fill=(200, 200, 200), width=1)
        for i in range(0, height, height // 10):
            draw.line([(0, i), (width, i)], fill=(200, 200, 200), width=1)

        # Draw objects
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128)]

        for i, obj in enumerate(scene_objects):
            color = colors[i % len(colors)]

            # Convert normalized position to pixel coordinates
            x = int(obj.position[0] * width)
            y = int(obj.position[1] * height)
            w = int(obj.size[0] * width)
            h = int(obj.size[1] * height)

            # Draw bounding box
            draw.rectangle([x - w // 2, y - h // 2, x + w // 2, y + h // 2], outline=color, width=2)

            # Draw label
            draw.text((x, y - h // 2 - 20), obj.name, fill=color)

        return guide


class CountingValidator:
    """Validate and enforce counting constraints in generated images."""

    def __init__(self):
        self.count_keywords = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
        }

    def extract_counting_requirements(self, prompt: str) -> Dict[str, int]:
        """Extract counting requirements from prompt."""
        requirements = {}

        # Look for patterns like "5 apples", "three cats", etc.
        import re

        # Pattern: number + object
        patterns = [
            r"(\w+)\s+(\w+)s?\b",  # "five cats" or "5 dogs"
            r"(\d+)\s+(\w+)s?\b",  # "5 cats"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, prompt.lower())
            for count_str, object_name in matches:
                if count_str in self.count_keywords:
                    count = self.count_keywords[count_str]
                    requirements[object_name] = count

        return requirements

    def generate_counting_prompt(self, object_name: str, count: int, base_prompt: str = "") -> str:
        """Generate prompt optimized for accurate counting."""
        # Use multiple reinforcement strategies
        count_word = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }.get(count, str(count))

        counting_phrases = [
            f"exactly {count} {object_name}",
            f"{count_word} {object_name}",
            f"precisely {count} {object_name}",
            f"total of {count} {object_name}",
        ]

        # Choose the most appropriate phrase
        main_phrase = counting_phrases[0]

        prompt_parts = []
        if base_prompt:
            prompt_parts.append(base_prompt)

        prompt_parts.extend(
            [
                main_phrase,
                f"count them: {count}",
                "clearly visible",
                "distinct objects",
                "well separated",
                "easy to count",
                "accurate count",
            ]
        )

        return ", ".join(prompt_parts)

    def validate_count_in_image(self, image: Image.Image, object_name: str, expected_count: int) -> Dict[str, Any]:
        """Validate object count in generated image (placeholder for object detection)."""
        # This would integrate with object detection models
        # For now, return a placeholder validation
        return {
            "object": object_name,
            "expected_count": expected_count,
            "detected_count": expected_count,  # Placeholder
            "accuracy": 1.0,
            "confidence": 0.8,
            "issues": [],
        }


class MultiStageGenerator:
    """Multi-stage generation for complex scenes requiring precision."""

    def __init__(self):
        self.generation_stages = {
            "layout": "Generate basic layout and composition",
            "objects": "Add main objects with precise positioning",
            "details": "Add fine details and refinements",
            "validation": "Validate and correct any issues",
        }

    def plan_generation_stages(self, prompt: str, complexity_score: float = None) -> List[Dict[str, Any]]:
        """Plan multi-stage generation based on prompt complexity."""
        stages = []

        # Analyze prompt complexity
        if complexity_score is None:
            complexity_score = self._assess_complexity(prompt)

        if complexity_score > 0.7:  # High complexity
            stages = [
                {
                    "stage": "background",
                    "prompt": self._extract_background_elements(prompt),
                    "focus": "scene setting and environment",
                },
                {
                    "stage": "main_objects",
                    "prompt": self._extract_main_objects(prompt),
                    "focus": "primary subjects and objects",
                },
                {
                    "stage": "relationships",
                    "prompt": self._extract_relationships(prompt),
                    "focus": "object interactions and positioning",
                },
                {"stage": "details", "prompt": self._extract_details(prompt), "focus": "fine details and refinements"},
            ]
        elif complexity_score > 0.4:  # Medium complexity
            stages = [
                {
                    "stage": "composition",
                    "prompt": self._simplify_for_composition(prompt),
                    "focus": "overall composition and layout",
                },
                {"stage": "refinement", "prompt": prompt, "focus": "add details and polish"},
            ]
        else:  # Low complexity
            stages = [{"stage": "single", "prompt": prompt, "focus": "complete generation in one stage"}]

        return stages

    def _assess_complexity(self, prompt: str) -> float:
        """Assess prompt complexity score (0-1)."""
        complexity_factors = {
            "object_count": len(prompt.split()) * 0.01,
            "relationships": prompt.lower().count(" and ") * 0.1,
            "spatial_terms": sum(
                1 for term in ["left", "right", "above", "below", "behind", "front"] if term in prompt.lower()
            )
            * 0.1,
            "counting": sum(1 for digit in "0123456789" if digit in prompt) * 0.05,
            "precision_terms": sum(
                1 for term in ["exactly", "precisely", "specific", "particular"] if term in prompt.lower()
            )
            * 0.15,
        }

        total_complexity = sum(complexity_factors.values())
        return min(total_complexity, 1.0)

    def _extract_background_elements(self, prompt: str) -> str:
        """Extract background/environment elements from prompt."""
        background_keywords = ["background", "setting", "environment", "scene", "location"]

        # Simple extraction - would be more sophisticated in practice
        words = prompt.split()
        background_words = []

        for i, word in enumerate(words):
            if any(bg_word in word.lower() for bg_word in background_keywords):
                # Include surrounding context
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                background_words.extend(words[start:end])

        if background_words:
            return " ".join(background_words) + ", simple composition, clear background"
        else:
            return "simple background, clean environment, minimal distractions"

    def _extract_main_objects(self, prompt: str) -> str:
        """Extract main objects from prompt."""
        # Remove background terms and focus on subjects
        main_terms = []
        skip_terms = ["background", "behind", "environment", "setting"]

        words = prompt.split()
        for word in words:
            if not any(skip in word.lower() for skip in skip_terms):
                main_terms.append(word)

        return " ".join(main_terms[:10])  # Limit to first 10 relevant terms

    def _extract_relationships(self, prompt: str) -> str:
        """Extract spatial relationships from prompt."""
        relationship_terms = [
            "left of",
            "right of",
            "above",
            "below",
            "behind",
            "in front of",
            "next to",
            "near",
            "far from",
            "between",
        ]

        relationship_parts = []
        for term in relationship_terms:
            if term in prompt.lower():
                # Extract context around relationship term
                idx = prompt.lower().find(term)
                start = max(0, idx - 20)
                end = min(len(prompt), idx + len(term) + 20)
                relationship_parts.append(prompt[start:end])

        if relationship_parts:
            return ", ".join(relationship_parts) + ", precise positioning, clear relationships"
        else:
            return prompt + ", well-positioned objects, clear spatial arrangement"

    def _extract_details(self, prompt: str) -> str:
        """Extract detail-focused elements from prompt."""
        detail_keywords = ["detailed", "intricate", "fine", "texture", "pattern", "ornate"]

        if any(keyword in prompt.lower() for keyword in detail_keywords):
            return prompt + ", high detail, fine textures, intricate patterns"
        else:
            return prompt + ", refined details, polished appearance, high quality"

    def _simplify_for_composition(self, prompt: str) -> str:
        """Simplify prompt for initial composition stage."""
        # Remove detail-heavy terms for cleaner initial composition
        detail_terms = ["detailed", "intricate", "ornate", "complex", "elaborate"]

        simplified = prompt
        for term in detail_terms:
            simplified = simplified.replace(term, "")

        return simplified + ", clean composition, clear layout, simple forms"


def create_precision_control_system():
    """Create complete precision control system."""
    return {
        "scene_composer": SceneComposer(),
        "counting_validator": CountingValidator(),
        "multi_stage_generator": MultiStageGenerator(),
    }
