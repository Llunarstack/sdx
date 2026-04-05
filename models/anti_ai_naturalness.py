"""
Anti-AI Naturalness System — style-aware imperfection injection for every art style.

Covers every major art style category with its own authentic "human-made" signature:

  TRADITIONAL MEDIA
    film_35mm       — grain, light leaks, halation, color shift, vignette
    film_medium_fmt — finer grain, richer tonal range, slight color cast
    polaroid        — overexposed corners, color drift, white border artifacts
    oil_painting    — impasto texture, brush direction variation, glazing layers
    watercolor      — wet-on-wet bleed, granulation, paper texture, backruns
    gouache         — flat matte with subtle brush marks, slight opacity variation
    acrylic         — fast-dry ridges, palette knife marks, canvas weave
    pencil_sketch   — hatching direction, pressure variation, smudge zones
    charcoal        — soft edge bleed, smear marks, paper tooth
    ink_pen         — line weight variation, nib pressure, ink pooling at corners
    pastel          — soft blending, chalk dust scatter, paper grain
    printmaking     — registration offset, ink spread, plate texture

  DIGITAL / SCREEN MEDIA
    digital_paint   — brush opacity variation, layer blend seams, color pick drift
    pixel_art       — intentional dithering, limited palette banding, sub-pixel AA
    vector_art      — anchor point micro-wobble, gradient banding, path overlap
    concept_art     — loose underdrawing visible, color temp shift per pass
    matte_painting  — photobash seam softening, perspective warp at edges

  ANIME / MANGA / ILLUSTRATION
    anime           — line wobble, cel shading inconsistency, speed line artifacts
    manga           — screen tone patterns, ink bleed, panel border bleed
    chibi           — proportion exaggeration consistency, clean line variation
    webtoon         — flat color with subtle gradient, panel gutter artifacts
    light_novel_illus — soft focus edges, pastel color bleed, highlight flare

  CARTOON / ANIMATION
    cartoon_western — wobbly outlines, flat color brush variation, registration shift
    cartoon_retro   — NTSC color bleed, scan line artifacts, cel dust
    cartoon_modern  — clean vector with subtle anchor wobble, gradient mesh seams
    stop_motion     — frame-to-frame texture jitter, clay fingerprint marks
    claymation      — fingerprint smudges, clay seam lines, uneven surface

  3D RENDER STYLES
    3d_blender      — render noise, firefly specks, normal map seams, HDRI reflection
    3d_unreal       — temporal AA ghosting, lumen GI bleed, material LOD pop
    3d_cinema4d     — subsurface scattering halo, caustic noise, depth of field ring
    3d_octane       — path trace noise, bloom halation, spectral dispersion
    3d_vray         — irradiance cache splotch, glossy reflection noise
    3d_stylized     — toon shader edge inconsistency, rim light bleed
    3d_claymation   — clay texture variation, fingerprint normal map, seam lines
    voxel           — aliased voxel edge, ambient occlusion corner darkening

  GAME / INTERACTIVE
    game_realistic  — TAA smear, screen-space reflection artifact, shadow acne
    game_stylized   — outline width variation, cel band inconsistency
    pixel_game      — sprite dithering, palette swap artifact, scanline flicker
    low_poly        — flat shading facet variation, edge highlight inconsistency

  PHOTOGRAPHY STYLES
    photograph      — lens distortion, chromatic aberration, sensor noise
    portrait_photo  — skin retouching artifact, dodge/burn halo, frequency sep seam
    street_photo    — motion blur, lens flare, dust spot
    macro_photo     — diffraction softening, focus breathing, depth stack seam
    infrared_photo  — channel swap artifact, halation, grain cluster
    lomography      — heavy vignette, color cross-process, light leak

  MIXED / HYBRID
    mixed_media     — collage edge artifact, texture overlay seam, scale mismatch
    ai_enhanced     — upscale artifact, detail hallucination, color posterize
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Minimal implementations (API parity for GenerationPipeline / models.__init__)
# ---------------------------------------------------------------------------


@dataclass
class MediumProfile:
    """Detected or user-specified medium for naturalness routing."""

    name: str = "unknown"
    family: str = "generic"


def detect_medium(prompt: str) -> MediumProfile:
    pl = (prompt or "").lower()
    if any(k in pl for k in ("anime", "manga", "cel shading", "visual novel")):
        return MediumProfile(name="anime", family="2d")
    if any(k in pl for k in ("photo", "photorealistic", "dslr", "raw photo")):
        return MediumProfile(name="photograph", family="photo")
    if any(k in pl for k in ("oil", "watercolor", "sketch", "pencil")):
        return MediumProfile(name="traditional", family="traditional")
    return MediumProfile()


def detect_all_mediums(prompt: str) -> List[MediumProfile]:
    m = detect_medium(prompt)
    return [m] if m.name != "unknown" else []


class AntiAINaturalnessController(nn.Module):
    """Placeholder: passes image tokens through (extend with real imperfection injection)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        medium: Optional[MediumProfile],
        h_patches: int,
        w_patches: int,
        strength: float = 0.5,
    ) -> torch.Tensor:
        _ = medium, h_patches, w_patches, strength
        return x


class _StubSubmodule(nn.Module):
    """Unused in pipeline today; exported for API symmetry."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x


MediumArtifactEncoder = _StubSubmodule
TextureImperfectionModule = _StubSubmodule
AsymmetryModule = _StubSubmodule
LineWobbleModule = _StubSubmodule
RenderNoiseModule = _StubSubmodule
ColorImperfectionModule = _StubSubmodule
StyleFamilyRouter = _StubSubmodule
