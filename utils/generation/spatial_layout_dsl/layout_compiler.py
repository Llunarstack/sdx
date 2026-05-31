"""Domain-specific language for spatial layout specification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F


class LayoutPosition(Enum):
    """Predefined positions in layout."""

    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


class LayoutAlignment(Enum):
    """Alignment options."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class LayoutRegion:
    """Describes a region in the layout."""

    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    priority: int
    prompt: str
    constraints: dict


@dataclass
class CompiledLayout:
    """Compiled layout with attention masks."""

    regions: list[LayoutRegion]
    attention_mask: torch.Tensor | None
    region_masks: dict[str, torch.Tensor]
    prompt_map: dict[str, str]


class LayoutDSLCompiler:
    """Compiles spatial layout specifications into control masks."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.regions = []

    def parse_layout_string(self, layout_str: str) -> CompiledLayout:
        """Parse layout DSL string into compiled layout.

        Example DSL:
            portrait_left {
                x: 0-0.4, y: 0-1
                priority: 10
                prompt: "portrait of a woman, detailed face"
            }
            background_right {
                x: 0.4-1, y: 0-1
                priority: 5
                prompt: "landscape background, mountains"
            }
        """
        regions = []
        current_region = None
        current_name = None

        for line in layout_str.strip().split("\n"):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "{" in line:
                current_name = line.split("{")[0].strip()
                current_region = {"name": current_name}
            elif "}" in line:
                if current_region:
                    regions.append(self._compile_region(current_region))
                current_region = None
            elif ":" in line and current_region:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().rstrip(",")

                if key == "x":
                    x_range = value.split("-")
                    current_region["x_min"] = float(x_range[0])
                    current_region["x_max"] = float(x_range[1])
                elif key == "y":
                    y_range = value.split("-")
                    current_region["y_min"] = float(y_range[0])
                    current_region["y_max"] = float(y_range[1])
                elif key == "priority":
                    current_region["priority"] = int(value)
                elif key == "prompt":
                    current_region["prompt"] = value.strip('"\'')
                else:
                    if "constraints" not in current_region:
                        current_region["constraints"] = {}
                    current_region["constraints"][key] = value

        return self._build_layout(regions)

    def _compile_region(self, region_dict: dict) -> LayoutRegion:
        """Compile a region dict into LayoutRegion."""
        return LayoutRegion(
            name=region_dict.get("name", "unnamed"),
            x_min=region_dict.get("x_min", 0.0),
            y_min=region_dict.get("y_min", 0.0),
            x_max=region_dict.get("x_max", 1.0),
            y_max=region_dict.get("y_max", 1.0),
            priority=region_dict.get("priority", 5),
            prompt=region_dict.get("prompt", ""),
            constraints=region_dict.get("constraints", {}),
        )

    def _build_layout(self, regions: list[LayoutRegion]) -> CompiledLayout:
        """Build compiled layout with masks."""
        sorted_regions = sorted(regions, key=lambda r: r.priority, reverse=True)

        region_masks = {}
        prompt_map = {}

        for region in sorted_regions:
            mask = self._create_region_mask(region)
            region_masks[region.name] = mask
            prompt_map[region.name] = region.prompt

        combined_mask = self._combine_region_masks(sorted_regions, region_masks)

        return CompiledLayout(
            regions=sorted_regions,
            attention_mask=combined_mask,
            region_masks=region_masks,
            prompt_map=prompt_map,
        )

    def _create_region_mask(self, region: LayoutRegion) -> torch.Tensor:
        """Create binary mask for region."""
        mask = torch.zeros((self.height, self.width), dtype=torch.float32)

        y_min = int(region.y_min * self.height)
        y_max = int(region.y_max * self.height)
        x_min = int(region.x_min * self.width)
        x_max = int(region.x_max * self.width)

        mask[y_min:y_max, x_min:x_max] = 1.0

        return mask

    def _combine_region_masks(self, regions: list[LayoutRegion], masks: dict) -> torch.Tensor:
        """Combine multiple region masks with priorities."""
        combined = torch.zeros((self.height, self.width), dtype=torch.float32)

        for region in regions:
            mask = masks[region.name]
            weight = region.priority / 10.0
            combined = combined + weight * mask

        combined = combined / (combined.max() + 1e-8)
        return combined

    def add_region(
        self,
        name: str,
        position: LayoutPosition | tuple[float, float, float, float],
        prompt: str,
        priority: int = 5,
        **constraints,
    ) -> LayoutRegion:
        """Add a region to layout programmatically.

        Args:
            name: Region name
            position: LayoutPosition enum or (x_min, y_min, x_max, y_max)
            prompt: Text prompt for region
            priority: Priority level (0-10)
            **constraints: Additional constraints (size, aspect_ratio, etc.)
        """
        if isinstance(position, LayoutPosition):
            bounds = self._get_position_bounds(position)
        else:
            bounds = position

        region = LayoutRegion(
            name=name,
            x_min=bounds[0],
            y_min=bounds[1],
            x_max=bounds[2],
            y_max=bounds[3],
            priority=priority,
            prompt=prompt,
            constraints=constraints,
        )

        self.regions.append(region)
        return region

    def _get_position_bounds(self, position: LayoutPosition) -> tuple[float, float, float, float]:
        """Get bounds for predefined position."""
        bounds_map = {
            LayoutPosition.CENTER: (0.25, 0.25, 0.75, 0.75),
            LayoutPosition.TOP_LEFT: (0.0, 0.0, 0.4, 0.4),
            LayoutPosition.TOP_CENTER: (0.2, 0.0, 0.8, 0.4),
            LayoutPosition.TOP_RIGHT: (0.6, 0.0, 1.0, 0.4),
            LayoutPosition.CENTER_LEFT: (0.0, 0.3, 0.4, 0.7),
            LayoutPosition.CENTER_RIGHT: (0.6, 0.3, 1.0, 0.7),
            LayoutPosition.BOTTOM_LEFT: (0.0, 0.6, 0.4, 1.0),
            LayoutPosition.BOTTOM_CENTER: (0.2, 0.6, 0.8, 1.0),
            LayoutPosition.BOTTOM_RIGHT: (0.6, 0.6, 1.0, 1.0),
        }
        return bounds_map[position]

    def compile(self) -> CompiledLayout:
        """Compile current regions into layout."""
        sorted_regions = sorted(self.regions, key=lambda r: r.priority, reverse=True)

        region_masks = {}
        prompt_map = {}

        for region in sorted_regions:
            mask = self._create_region_mask(region)
            region_masks[region.name] = mask
            prompt_map[region.name] = region.prompt

        combined_mask = self._combine_region_masks(sorted_regions, region_masks)

        return CompiledLayout(
            regions=sorted_regions,
            attention_mask=combined_mask,
            region_masks=region_masks,
            prompt_map=prompt_map,
        )

    def generate_unified_prompt(self, compiled: CompiledLayout) -> str:
        """Generate unified prompt from layout regions."""
        prompts_by_priority = [(r.priority, r.prompt) for r in compiled.regions]
        prompts_by_priority.sort(reverse=True)

        unified = "; ".join(p[1] for p in prompts_by_priority if p[1])
        return unified

    def visualize_layout(self, compiled: CompiledLayout) -> str:
        """Create ASCII visualization of layout."""
        grid_w, grid_h = 40, 20
        grid = [["." for _ in range(grid_w)] for _ in range(grid_h)]

        for name, mask in compiled.region_masks.items():
            char = name[0].upper() if name else "?"

            for y in range(grid_h):
                for x in range(grid_w):
                    sample_y = int((y / grid_h) * self.height)
                    sample_x = int((x / grid_w) * self.width)

                    if sample_y < mask.shape[0] and sample_x < mask.shape[1]:
                        if mask[sample_y, sample_x] > 0.5:
                            grid[y][x] = char

        return "\n".join("".join(row) for row in grid)
