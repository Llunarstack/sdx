"""
Camera, POV, Perspective & Viewing Angle System.

AI models consistently fail at:
  1. Extreme angles: worm's-eye, bird's-eye, Dutch tilt
  2. Lens characteristics: fisheye distortion, anamorphic flares, tilt-shift
  3. POV: first-person, over-the-shoulder, through-the-keyhole
  4. Depth of field: shallow bokeh vs deep focus
  5. Perspective distortion: forced perspective, foreshortening
  6. Cinematic framing: rule of thirds, leading lines, negative space

Architecture:
  - CameraSpecParser: extracts camera/lens/angle specs from text
  - CameraEmbedder: encodes camera specs as conditioning vectors
  - PerspectiveDistortionModule: applies perspective-aware spatial biases
    to image tokens so the model "knows" which patches are foreground/background
  - LensDistortionEncoder: encodes lens-specific distortion patterns
  - DepthOfFieldModule: conditions on DoF to guide blur/sharpness distribution
  - CameraConditioner: top-level module

Key insight: camera specs are a separate conditioning axis from content.
We encode them independently and inject them at every transformer block
via AdaLN-style modulation, not just as text tokens.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Camera spec data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraSpec:
    """Parsed camera/lens/angle specification."""
    # Viewing angle
    angle: str = "eye_level"           # eye_level, birds_eye, worms_eye, dutch, overhead, low_angle, high_angle
    pov: str = "third_person"          # first_person, third_person, over_shoulder, through_object, drone
    # Lens
    focal_length_mm: float = 50.0      # 14, 24, 35, 50, 85, 135, 200+
    aperture: float = 2.8              # f/1.2 to f/22
    lens_type: str = "standard"        # standard, wide, telephoto, fisheye, tilt_shift, anamorphic, macro
    # Depth of field
    dof: str = "medium"                # shallow, medium, deep
    focus_distance: float = 0.5        # 0=very close, 1=infinity
    # Composition
    composition: str = "centered"      # centered, rule_of_thirds, golden_ratio, leading_lines, negative_space
    # Distortion
    distortion: float = 0.0            # -1=barrel (fisheye), 0=none, +1=pincushion
    # Cinematic
    aspect_ratio: str = "16:9"         # 1:1, 4:3, 16:9, 2.39:1 (anamorphic)
    film_format: str = "digital"       # digital, 35mm, medium_format, large_format, super8


# ---------------------------------------------------------------------------
# Camera Spec Parser
# ---------------------------------------------------------------------------

class CameraSpecParser:
    """
    Extracts camera/lens/angle specifications from text prompts.

    Handles natural language like:
      "shot on 85mm f/1.4, shallow depth of field, rule of thirds"
      "worm's eye view, extreme low angle, fisheye lens"
      "first person POV, over the shoulder shot, Dutch angle"
      "bird's eye view, drone shot, wide angle 24mm"
    """

    ANGLE_MAP = {
        r"bird.?s.?eye|aerial|top.?down|overhead|from above": "birds_eye",
        r"worm.?s.?eye|extreme low angle|ground level|from below": "worms_eye",
        r"dutch angle|canted|tilted frame|oblique": "dutch",
        r"low angle|looking up|upward angle": "low_angle",
        r"high angle|looking down|downward angle": "high_angle",
        r"eye.?level|straight on|neutral angle": "eye_level",
        r"overhead|directly above|top view": "overhead",
    }

    POV_MAP = {
        r"first.?person|fps|pov|through the eyes|subjective": "first_person",
        r"over.?the.?shoulder|ots shot|behind": "over_shoulder",
        r"through.?(?:a\s+)?(?:keyhole|window|door|glass|mask|scope|binoculars)": "through_object",
        r"drone|aerial|fly.?over": "drone",
        r"third.?person|observer|external": "third_person",
    }

    LENS_MAP = {
        r"fisheye|fish.?eye|ultra.?wide|180.?degree": "fisheye",
        r"tilt.?shift|miniature effect|selective focus": "tilt_shift",
        r"anamorphic|cinemascope|widescreen|2\.39": "anamorphic",
        r"macro|close.?up|extreme close.?up|ecу": "macro",
        r"telephoto|long lens|compressed|200mm|300mm|400mm": "telephoto",
        r"wide.?angle|wide lens|14mm|16mm|20mm|24mm|28mm": "wide",
        r"portrait lens|85mm|105mm|135mm": "portrait",
    }

    DOF_MAP = {
        r"shallow.?dof|shallow depth|bokeh|blurry background|f\/1|f\/1\.[24]|f\/2": "shallow",
        r"deep.?dof|deep focus|everything sharp|f\/8|f\/11|f\/16|f\/22": "deep",
        r"medium.?dof|moderate focus|f\/4|f\/5\.6": "medium",
    }

    COMPOSITION_MAP = {
        r"rule of thirds|thirds": "rule_of_thirds",
        r"golden ratio|golden spiral|phi": "golden_ratio",
        r"leading lines|diagonal lines|converging": "leading_lines",
        r"negative space|empty space|minimalist composition": "negative_space",
        r"symmetr|centered|centered composition": "centered",
        r"frame within frame|framing": "frame_in_frame",
    }

    def _match_map(self, text: str, mapping: Dict[str, str], default: str) -> str:
        for pattern, value in mapping.items():
            if re.search(pattern, text, re.IGNORECASE):
                return value
        return default

    def _extract_focal_length(self, text: str) -> float:
        m = re.search(r'(\d+)\s*mm', text, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return 50.0

    def _extract_aperture(self, text: str) -> float:
        m = re.search(r'f\s*/\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return 2.8

    def parse(self, text: str) -> CameraSpec:
        return CameraSpec(
            angle=self._match_map(text, self.ANGLE_MAP, "eye_level"),
            pov=self._match_map(text, self.POV_MAP, "third_person"),
            focal_length_mm=self._extract_focal_length(text),
            aperture=self._extract_aperture(text),
            lens_type=self._match_map(text, self.LENS_MAP, "standard"),
            dof=self._match_map(text, self.DOF_MAP, "medium"),
            composition=self._match_map(text, self.COMPOSITION_MAP, "centered"),
        )


# ---------------------------------------------------------------------------
# Camera Embedder
# ---------------------------------------------------------------------------

class CameraEmbedder(nn.Module):
    """
    Encodes a CameraSpec into a conditioning vector injected at every block.

    Args:
        hidden_size: Output conditioning dimension.
    """

    ANGLES = ["eye_level", "birds_eye", "worms_eye", "dutch", "low_angle", "high_angle", "overhead"]
    POVS   = ["first_person", "third_person", "over_shoulder", "through_object", "drone"]
    LENSES = ["standard", "wide", "telephoto", "fisheye", "tilt_shift", "anamorphic", "macro", "portrait"]
    DOFS   = ["shallow", "medium", "deep"]
    COMPS  = ["centered", "rule_of_thirds", "golden_ratio", "leading_lines", "negative_space", "frame_in_frame"]

    def __init__(self, hidden_size: int):
        super().__init__()
        self.angle_embed = nn.Embedding(len(self.ANGLES), hidden_size // 4)
        self.pov_embed   = nn.Embedding(len(self.POVS),   hidden_size // 4)
        self.lens_embed  = nn.Embedding(len(self.LENSES), hidden_size // 4)
        self.dof_embed   = nn.Embedding(len(self.DOFS),   hidden_size // 4)
        self.comp_embed  = nn.Embedding(len(self.COMPS),  hidden_size // 4)

        # Continuous params: focal_length, aperture, distortion, focus_distance
        self.continuous_proj = nn.Linear(4, hidden_size // 4)

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size // 4 * 6, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

        self._angle_to_id = {a: i for i, a in enumerate(self.ANGLES)}
        self._pov_to_id   = {p: i for i, p in enumerate(self.POVS)}
        self._lens_to_id  = {lens_name: i for i, lens_name in enumerate(self.LENSES)}
        self._dof_to_id   = {d: i for i, d in enumerate(self.DOFS)}
        self._comp_to_id  = {c: i for i, c in enumerate(self.COMPS)}

    def forward(self, spec: CameraSpec, device: torch.device) -> torch.Tensor:
        """Returns (1, hidden_size) camera conditioning vector."""
        def _id(mapping, key, default=0):
            return torch.tensor(mapping.get(key, default), device=device)

        a = self.angle_embed(_id(self._angle_to_id, spec.angle))
        p = self.pov_embed(_id(self._pov_to_id, spec.pov))
        lens_emb = self.lens_embed(_id(self._lens_to_id, spec.lens_type))
        d = self.dof_embed(_id(self._dof_to_id, spec.dof))
        c = self.comp_embed(_id(self._comp_to_id, spec.composition))

        # Normalise continuous params to [0, 1]
        fl_norm = (math.log(max(spec.focal_length_mm, 1)) - math.log(14)) / (math.log(800) - math.log(14))
        ap_norm = (math.log(max(spec.aperture, 0.7)) - math.log(0.7)) / (math.log(22) - math.log(0.7))
        cont = torch.tensor(
            [fl_norm, ap_norm, (spec.distortion + 1) / 2, spec.focus_distance],
            device=device, dtype=torch.float32
        )
        cont_emb = self.continuous_proj(cont)

        combined = torch.cat([a, p, lens_emb, d, c, cont_emb], dim=-1).unsqueeze(0)  # (1, D*6/4)
        return self.fusion(combined)  # (1, hidden_size)


# ---------------------------------------------------------------------------
# Perspective Distortion Module
# ---------------------------------------------------------------------------

class PerspectiveDistortionModule(nn.Module):
    """
    Applies perspective-aware spatial biases to image tokens.

    For a bird's-eye view, patches near the top of the image should
    represent "far away" objects (smaller, more compressed). For a
    worm's-eye view, the opposite. This module injects these spatial
    priors directly into the patch token sequence.

    Args:
        hidden_size: Token dimension.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Perspective field generator: maps (angle_emb, patch_position) -> spatial bias
        self.field_gen = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size // 2),  # +2 for (row, col) normalised
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        nn.init.zeros_(self.field_gen[-1].weight)
        nn.init.zeros_(self.field_gen[-1].bias)

        # Vanishing point predictor: where does the perspective converge?
        self.vp_predictor = nn.Sequential(
            nn.Linear(hidden_size, 2),  # (vp_x, vp_y) normalised [0,1]
            nn.Sigmoid(),
        )

        # Foreshortening scale: how much to compress depth axis
        self.foreshorten_scale = nn.Parameter(torch.ones(1))

    def _get_patch_positions(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Returns (N, 2) normalised (row, col) positions."""
        rows = torch.linspace(0, 1, h, device=device)
        cols = torch.linspace(0, 1, w, device=device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
        return torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)  # (N, 2)

    def forward(
        self,
        x: torch.Tensor,
        camera_emb: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) image tokens.
            camera_emb: (B, D) camera conditioning vector.
            h_patches, w_patches: Spatial grid dimensions.
        Returns:
            (B, N, D) perspective-conditioned tokens.
        """
        B, N, D = x.shape
        positions = self._get_patch_positions(h_patches, w_patches, x.device)  # (N, 2)

        # Expand camera_emb to per-patch: (B, N, D)
        cam_expanded = camera_emb.unsqueeze(1).expand(B, N, D)

        # Concatenate position with camera embedding
        pos_expanded = positions.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        field_input = torch.cat([cam_expanded, pos_expanded], dim=-1)  # (B, N, D+2)

        # Generate perspective field
        perspective_field = self.field_gen(field_input)  # (B, N, D)

        # Predict vanishing point and apply foreshortening
        vp = self.vp_predictor(camera_emb)  # (B, 2) — vanishing point
        vp_expanded = vp.unsqueeze(1)  # (B, 1, 2)

        # Distance from vanishing point: patches near VP are "far away"
        dist_to_vp = (pos_expanded - vp_expanded).pow(2).sum(-1, keepdim=True).sqrt()  # (B, N, 1)
        foreshorten = torch.sigmoid(self.foreshorten_scale) * dist_to_vp

        # Apply: tokens near VP get compressed (foreshortened)
        x = x + 0.15 * perspective_field * (1.0 + foreshorten)
        return x


# ---------------------------------------------------------------------------
# Lens Distortion Encoder
# ---------------------------------------------------------------------------

class LensDistortionEncoder(nn.Module):
    """
    Encodes lens-specific spatial distortion patterns into patch tokens.

    Fisheye: barrel distortion (edges stretched outward)
    Tilt-shift: selective focus plane (only a horizontal band is sharp)
    Anamorphic: horizontal stretch + oval bokeh
    Telephoto: compressed depth, flat perspective

    Args:
        hidden_size: Token dimension.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Distortion field: maps (lens_type_emb, position) -> distortion vector
        self.distortion_field = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        nn.init.zeros_(self.distortion_field[-1].weight)
        nn.init.zeros_(self.distortion_field[-1].bias)

        # Sharpness map: which patches are in focus?
        self.sharpness_map = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        camera_emb: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) image tokens.
            camera_emb: (B, D) camera conditioning.
        Returns:
            (B, N, D) lens-distorted tokens.
        """
        B, N, D = x.shape
        rows = torch.linspace(0, 1, h_patches, device=x.device)
        cols = torch.linspace(0, 1, w_patches, device=x.device)
        gr, gc = torch.meshgrid(rows, cols, indexing='ij')
        positions = torch.stack([gr.flatten(), gc.flatten()], dim=-1)  # (N, 2)

        cam_exp = camera_emb.unsqueeze(1).expand(B, N, D)
        pos_exp = positions.unsqueeze(0).expand(B, -1, -1)
        inp = torch.cat([cam_exp, pos_exp], dim=-1)

        distortion = self.distortion_field(inp)  # (B, N, D)
        sharpness = self.sharpness_map(inp)       # (B, N, 1)

        # Apply: distortion modulates token content, sharpness gates it
        return x + 0.1 * distortion * sharpness


# ---------------------------------------------------------------------------
# Depth of Field Module
# ---------------------------------------------------------------------------

class DepthOfFieldModule(nn.Module):
    """
    Conditions image tokens on depth-of-field to guide blur/sharpness distribution.

    Shallow DoF: only patches near the focus plane are "sharp" (high detail).
    Deep DoF: all patches are sharp.

    This injects a per-patch sharpness signal that the model uses to decide
    how much detail to render in each region.

    Args:
        hidden_size: Token dimension.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Focus plane predictor: given camera emb, where is the focus plane?
        self.focus_plane = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),  # normalised depth [0, 1]
        )
        # DoF width predictor: how wide is the in-focus zone?
        self.dof_width = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        # Sharpness injection
        self.sharp_proj = nn.Linear(1, hidden_size, bias=False)
        nn.init.zeros_(self.sharp_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        camera_emb: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
        h_patches: int = 16,
        w_patches: int = 16,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) image tokens.
            camera_emb: (B, D) camera conditioning.
            depth_map: (B, 1, H, W) optional depth map. If None, uses linear depth.
        Returns:
            (B, N, D) DoF-conditioned tokens.
        """
        B, N, D = x.shape

        focus_depth = self.focus_plane(camera_emb)  # (B, 1)
        dof_w = self.dof_width(camera_emb) * 0.5 + 0.05  # (B, 1) in [0.05, 0.55]

        if depth_map is not None:
            # Use provided depth map
            d = F.avg_pool2d(depth_map, kernel_size=depth_map.shape[-1] // h_patches)
            d = d.flatten(2).transpose(1, 2)  # (B, N, 1)
        else:
            # Linear depth: top of image = far, bottom = near
            rows = torch.linspace(0, 1, h_patches, device=x.device)
            cols = torch.linspace(0, 1, w_patches, device=x.device)
            gr, _ = torch.meshgrid(rows, cols, indexing='ij')
            d = gr.flatten().unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)  # (B, N, 1)

        # Sharpness: Gaussian around focus plane
        focus_depth_exp = focus_depth.unsqueeze(1)  # (B, 1, 1)
        dof_w_exp = dof_w.unsqueeze(1)              # (B, 1, 1)
        sharpness = torch.exp(-0.5 * ((d - focus_depth_exp) / (dof_w_exp + 1e-6)) ** 2)  # (B, N, 1)

        # Inject sharpness signal into tokens
        sharp_signal = self.sharp_proj(sharpness)  # (B, N, D)
        return x + 0.1 * sharp_signal


# ---------------------------------------------------------------------------
# Camera Conditioner (top-level)
# ---------------------------------------------------------------------------

class CameraConditioner(nn.Module):
    """
    Top-level camera conditioning module.

    Parses camera specs from text, encodes them, and applies:
    - Perspective distortion to patch tokens
    - Lens distortion patterns
    - Depth of field sharpness distribution
    - Camera conditioning vector added to timestep embedding

    Usage:
        conditioner = CameraConditioner(hidden_size=1152)
        spec = conditioner.parse("shot on 85mm f/1.4, shallow dof, rule of thirds")
        cam_emb = conditioner.embed(spec, device)
        x = conditioner(x, cam_emb, h_patches=16, w_patches=16)
        # Add cam_emb to your timestep conditioning: c = t_emb + cam_emb
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.parser = CameraSpecParser()
        self.embedder = CameraEmbedder(hidden_size)
        self.perspective = PerspectiveDistortionModule(hidden_size)
        self.lens = LensDistortionEncoder(hidden_size)
        self.dof = DepthOfFieldModule(hidden_size)

    def parse(self, text: str) -> CameraSpec:
        return self.parser.parse(text)

    def embed(self, spec: CameraSpec, device: torch.device) -> torch.Tensor:
        """Returns (1, D) camera conditioning vector."""
        return self.embedder(spec, device)

    def forward(
        self,
        x: torch.Tensor,
        camera_emb: torch.Tensor,
        h_patches: int,
        w_patches: int,
        depth_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply all camera conditioning to image tokens.

        Args:
            x: (B, N, D) image tokens.
            camera_emb: (B, D) camera conditioning vector.
            h_patches, w_patches: Spatial grid dimensions.
            depth_map: (B, 1, H, W) optional depth map.
        Returns:
            (B, N, D) camera-conditioned tokens.
        """
        x = self.perspective(x, camera_emb, h_patches, w_patches)
        x = self.lens(x, camera_emb, h_patches, w_patches)
        x = self.dof(x, camera_emb, depth_map, h_patches, w_patches)
        return x


__all__ = [
    "CameraSpec",
    "CameraSpecParser",
    "CameraEmbedder",
    "PerspectiveDistortionModule",
    "LensDistortionEncoder",
    "DepthOfFieldModule",
    "CameraConditioner",
]
