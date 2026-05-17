"""
Artistic post-processing: compositional director, value structure, intentional asymmetry,
lost-and-found edges, and subsurface scattering simulation.

These address the specific AI image generation failures described in the AGI roadmap:
- Center-weighted bias / "horror vacui" (fills every space)
- Flat value structure (no light/dark discipline)
- Uncanny valley symmetry (faces too perfect)
- Plastic/synthetic look (no subsurface scattering simulation)
- Hard uniform edges (no "lost and found" variation)

All functions operate on (H, W, 3) uint8 numpy arrays and return the same.
All are optional post-processing passes — strength=0 is always a no-op.

Usage in sample.py:
    from utils.quality.artistic_post_process import apply_artistic_pipeline
    img_np = apply_artistic_pipeline(img_np, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ArtisticPostConfig:
    """All artistic post-processing knobs in one place."""

    # Compositional director
    composition_mode: str = "none"          # "none"|"rule_of_thirds"|"golden_ratio"|"dynamic_symmetry"
    composition_strength: float = 0.0       # 0-1: how strongly to nudge toward the composition guide

    # Value structure (light/dark discipline)
    value_structure: bool = False           # enforce value hierarchy
    value_shadow_lift: float = 0.0          # 0-0.15: lift shadows slightly (reduces crush)
    value_highlight_roll: float = 0.0       # 0-0.15: roll off highlights (reduces blow-out)
    value_midtone_contrast: float = 0.0     # 0-0.3: boost midtone separation

    # Intentional asymmetry (anti-uncanny-valley)
    asymmetry_strength: float = 0.0         # 0-1: how much to break perfect symmetry
    asymmetry_seed: Optional[int] = None

    # Lost-and-found edges (vary edge sharpness)
    lost_found_strength: float = 0.0        # 0-1: how much to soften some edges
    lost_found_seed: Optional[int] = None

    # Subsurface scattering simulation (skin/wax/translucent materials)
    sss_strength: float = 0.0               # 0-1: SSS simulation strength
    sss_radius: float = 3.0                 # blur radius for SSS approximation
    sss_color: Tuple[float, float, float] = (1.0, 0.6, 0.4)  # warm skin scatter color

    # Vignette (compositional framing)
    vignette_strength: float = 0.0          # 0-1
    vignette_softness: float = 0.5          # 0-1: how soft the falloff is

    # Chromatic aberration (lens character)
    chromatic_aberration: float = 0.0       # 0-1: pixel shift amount

    # Micro-detail recovery (sharpens fine texture without halos)
    micro_detail: float = 0.0              # 0-1


# ---------------------------------------------------------------------------
# Value structure
# ---------------------------------------------------------------------------

def _rgb_to_lum(img: np.ndarray) -> np.ndarray:
    """(H,W,3) float -> (H,W) luminance."""
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def apply_value_structure(
    image: np.ndarray,
    *,
    shadow_lift: float = 0.0,
    highlight_roll: float = 0.0,
    midtone_contrast: float = 0.0,
) -> np.ndarray:
    """
    Enforce value discipline:
    - shadow_lift: raise the black point slightly (0.0-0.15) — prevents crushed shadows
    - highlight_roll: compress highlights (0.0-0.15) — prevents blown-out whites
    - midtone_contrast: boost separation in the midtone range (0.0-0.3)

    All operations are luminance-only to preserve hue/saturation.
    """
    if shadow_lift <= 0 and highlight_roll <= 0 and midtone_contrast <= 0:
        return image

    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    lum = _rgb_to_lum(img) / 255.0  # [0,1]

    new_lum = lum.copy()

    # Shadow lift: remap [0, lift_point] -> [lift_floor, lift_point]
    if shadow_lift > 0:
        sl = float(min(0.15, shadow_lift))
        shadow_mask = np.clip(1.0 - lum / max(sl * 3, 0.1), 0.0, 1.0)
        new_lum = new_lum + sl * shadow_mask

    # Highlight roll-off: compress [1-roll_point, 1] -> [1-roll_point, 1-roll_floor]
    if highlight_roll > 0:
        hr = float(min(0.15, highlight_roll))
        hi_mask = np.clip((lum - (1.0 - hr * 3)) / max(hr * 3, 0.01), 0.0, 1.0)
        new_lum = new_lum - hr * hi_mask

    # Midtone contrast: S-curve in [0.2, 0.8] range
    if midtone_contrast > 0:
        mc = float(min(0.3, midtone_contrast))
        mid_mask = np.clip(1.0 - np.abs(lum - 0.5) / 0.3, 0.0, 1.0)
        # Push midtones away from 0.5
        push = np.sign(lum - 0.5) * mc * mid_mask
        new_lum = new_lum + push

    new_lum = np.clip(new_lum, 0.0, 1.0)

    # Rescale RGB channels to match new luminance
    old_lum = np.clip(_rgb_to_lum(img) / 255.0, 1e-4, 1.0)
    scale = (new_lum / old_lum)[..., np.newaxis]
    out = np.clip(img * scale, 0.0, 255.0)

    return out.astype(np.uint8) if is_uint else out


# ---------------------------------------------------------------------------
# Compositional director
# ---------------------------------------------------------------------------

def _rule_of_thirds_map(h: int, w: int) -> np.ndarray:
    """
    Returns (H, W) weight map that is highest at rule-of-thirds intersections.
    Used to nudge the image's visual weight toward these points.
    """
    yy = np.linspace(0, 1, h)
    xx = np.linspace(0, 1, w)
    grid_x, grid_y = np.meshgrid(xx, yy)

    # Four intersection points
    thirds = [1/3, 2/3]
    weight = np.zeros((h, w), dtype=np.float32)
    sigma = 0.12  # spread of each hotspot
    for ty in thirds:
        for tx in thirds:
            d2 = (grid_y - ty)**2 + (grid_x - tx)**2
            weight += np.exp(-d2 / (2 * sigma**2))

    return weight / (weight.max() + 1e-8)


def _golden_ratio_map(h: int, w: int) -> np.ndarray:
    """
    Returns (H, W) weight map based on golden ratio spiral approximation.
    Hotspots at golden ratio intersections (phi = 1.618).
    """
    phi = 1.618033988749895
    yy = np.linspace(0, 1, h)
    xx = np.linspace(0, 1, w)
    grid_x, grid_y = np.meshgrid(xx, yy)

    # Golden ratio intersection points
    gx = [1.0 / phi, 1.0 - 1.0 / phi]
    gy = [1.0 / phi, 1.0 - 1.0 / phi]
    weight = np.zeros((h, w), dtype=np.float32)
    sigma = 0.10
    for ty in gy:
        for tx in gx:
            d2 = (grid_y - ty)**2 + (grid_x - tx)**2
            weight += np.exp(-d2 / (2 * sigma**2))

    return weight / (weight.max() + 1e-8)


def apply_compositional_director(
    image: np.ndarray,
    mode: str = "rule_of_thirds",
    strength: float = 0.15,
) -> np.ndarray:
    """
    Nudge the image's visual weight toward compositional guide points.

    This works by slightly boosting luminance/contrast at the guide intersections
    and very subtly reducing it at the center — counteracting the AI's center bias.

    mode: "rule_of_thirds" | "golden_ratio" | "dynamic_symmetry"
    strength: 0-1 (0.1-0.25 is subtle but effective)
    """
    if strength <= 0 or mode == "none":
        return image

    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    h, w = img.shape[:2]
    s = float(min(1.0, max(0.0, strength)))

    if mode == "golden_ratio":
        guide_map = _golden_ratio_map(h, w)
    elif mode == "dynamic_symmetry":
        # Diagonal emphasis: boost along the main diagonals
        yy = np.linspace(0, 1, h)
        xx = np.linspace(0, 1, w)
        grid_x, grid_y = np.meshgrid(xx, yy)
        d1 = np.exp(-((grid_y - grid_x)**2) / (2 * 0.08**2))
        d2 = np.exp(-((grid_y - (1 - grid_x))**2) / (2 * 0.08**2))
        guide_map = np.clip(d1 + d2, 0, 1).astype(np.float32)
        guide_map /= guide_map.max() + 1e-8
    else:  # rule_of_thirds (default)
        guide_map = _rule_of_thirds_map(h, w)

    # Center suppression map: reduce center weight slightly
    yy = np.linspace(-1, 1, h)
    xx = np.linspace(-1, 1, w)
    gx, gy = np.meshgrid(xx, yy)
    center_weight = np.exp(-(gx**2 + gy**2) / (2 * 0.4**2))
    center_suppress = 1.0 - 0.08 * s * center_weight

    # Apply: boost guide points, suppress center
    boost = 1.0 + s * 0.12 * guide_map[..., np.newaxis]
    suppress = center_suppress[..., np.newaxis]
    out = np.clip(img * boost * suppress, 0.0, 255.0)

    return out.astype(np.uint8) if is_uint else out


# ---------------------------------------------------------------------------
# Intentional asymmetry (anti-uncanny-valley)
# ---------------------------------------------------------------------------

def apply_intentional_asymmetry(
    image: np.ndarray,
    strength: float = 0.3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Introduce subtle, natural asymmetry to break the AI's tendency toward
    perfect bilateral symmetry (which reads as uncanny/artificial).

    Method:
    - Apply a very slight, smooth spatial warp using a low-frequency noise field
    - The warp is asymmetric: left/right halves get slightly different treatment
    - Magnitude is kept small enough to be subliminal (not visible as distortion)

    strength: 0-1 (0.1-0.4 is the sweet spot; above 0.6 becomes visible)
    """
    if strength <= 0:
        return image

    try:
        from PIL import Image as _PIL
        from scipy.ndimage import map_coordinates
    except ImportError:
        return image

    rng = np.random.default_rng(seed)
    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    h, w = img.shape[:2]
    s = float(min(1.0, max(0.0, strength)))

    # Generate smooth low-frequency displacement field
    # Use a small noise grid then upsample — ensures smooth, organic warping
    grid_h, grid_w = max(4, h // 16), max(4, w // 16)
    noise_y = rng.normal(0, 1, (grid_h, grid_w)).astype(np.float32)
    noise_x = rng.normal(0, 1, (grid_h, grid_w)).astype(np.float32)

    # Upsample to full resolution
    noise_y_full = np.array(
        _PIL.fromarray(((noise_y - noise_y.min()) / (np.ptp(noise_y) + 1e-8) * 255).astype(np.uint8))
        .resize((w, h), _PIL.BILINEAR), dtype=np.float32
    ) / 255.0 * 2.0 - 1.0  # [-1, 1]
    noise_x_full = np.array(
        _PIL.fromarray(((noise_x - noise_x.min()) / (np.ptp(noise_x) + 1e-8) * 255).astype(np.uint8))
        .resize((w, h), _PIL.BILINEAR), dtype=np.float32
    ) / 255.0 * 2.0 - 1.0

    # Scale displacement: max ~2-4 pixels at strength=1
    max_disp = s * 3.0
    dy = noise_y_full * max_disp
    dx = noise_x_full * max_disp

    # Make it asymmetric: apply stronger warp to one half
    # This is the key — symmetric warp would cancel out and look natural but not break symmetry
    half_w = w // 2
    asymmetry_mask = np.ones((h, w), dtype=np.float32)
    asymmetry_mask[:, :half_w] *= 0.4   # left half: weaker warp
    asymmetry_mask[:, half_w:] *= 1.0   # right half: stronger warp
    dy *= asymmetry_mask
    dx *= asymmetry_mask

    # Build coordinate maps
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float32)
    map_rows = np.clip(rows + dy, 0, h - 1)
    map_cols = np.clip(cols + dx, 0, w - 1)

    # Apply warp to each channel
    out = np.zeros_like(img)
    for c in range(3):
        out[..., c] = map_coordinates(
            img[..., c], [map_rows, map_cols], order=1, mode='reflect'
        )

    out = np.clip(out, 0.0, 255.0)
    return out.astype(np.uint8) if is_uint else out


# ---------------------------------------------------------------------------
# Lost-and-found edges
# ---------------------------------------------------------------------------

def apply_lost_and_found_edges(
    image: np.ndarray,
    strength: float = 0.3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Vary edge sharpness across the image to mimic human artistic mark-making.

    In traditional art, edges are not uniformly sharp — some are "lost" (soft,
    blending into surroundings) and some are "found" (crisp, defining form).
    AI images tend to have uniformly sharp or uniformly soft edges, which reads
    as mechanical.

    Method:
    - Detect edges using a simple gradient magnitude
    - Generate a smooth random mask that selects which edges to soften
    - Apply local blur only to the "lost" edge regions
    - Keep "found" edges sharp

    strength: 0-1 (0.2-0.5 is natural; above 0.7 becomes painterly)
    """
    if strength <= 0:
        return image

    try:
        from PIL import Image as _PIL
        from scipy.ndimage import gaussian_filter, sobel
    except ImportError:
        return image

    rng = np.random.default_rng(seed)
    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    h, w = img.shape[:2]
    s = float(min(1.0, max(0.0, strength)))

    # Detect edges
    gray = _rgb_to_lum(img)
    edge_x = sobel(gray, axis=1)
    edge_y = sobel(gray, axis=0)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    edge_mag = edge_mag / (edge_mag.max() + 1e-8)  # [0, 1]

    # Generate smooth "lost" mask — which edges to soften
    # Use low-frequency noise so the pattern is organic, not random speckle
    grid_h, grid_w = max(4, h // 8), max(4, w // 8)
    noise = rng.uniform(0, 1, (grid_h, grid_w)).astype(np.float32)
    lost_mask = np.array(
        _PIL.fromarray((noise * 255).astype(np.uint8))
        .resize((w, h), _PIL.BILINEAR), dtype=np.float32
    ) / 255.0

    # Only apply to actual edge regions (not flat areas)
    lost_mask = lost_mask * edge_mag

    # Threshold: top s*50% of edge pixels get softened
    threshold = np.percentile(lost_mask, 100 - s * 50)
    lost_binary = (lost_mask > threshold).astype(np.float32)

    # Smooth the binary mask to avoid hard transitions
    lost_smooth = gaussian_filter(lost_binary, sigma=2.0)
    lost_smooth = np.clip(lost_smooth, 0.0, 1.0)[..., np.newaxis]

    # Create softened version
    blur_radius = 0.8 + s * 1.5
    img_soft = gaussian_filter(img, sigma=(blur_radius, blur_radius, 0))

    # Blend: lost edges get softened, found edges stay sharp
    out = img * (1.0 - lost_smooth) + img_soft * lost_smooth
    out = np.clip(out, 0.0, 255.0)

    return out.astype(np.uint8) if is_uint else out


# ---------------------------------------------------------------------------
# Subsurface scattering simulation
# ---------------------------------------------------------------------------

def apply_sss_simulation(
    image: np.ndarray,
    strength: float = 0.3,
    radius: float = 3.0,
    scatter_color: Tuple[float, float, float] = (1.0, 0.6, 0.4),
) -> np.ndarray:
    """
    Simulate subsurface scattering for skin/wax/translucent materials.

    Real SSS: light enters a semi-translucent surface, scatters internally,
    and exits at a different point with a warm color shift (for skin: red/orange).

    Approximation:
    - Detect bright regions (likely lit skin/translucent areas)
    - Apply a colored blur to simulate scattered light bleeding
    - Blend back with the original at the specified strength
    - The result: edges of lit areas get a warm color bleed, shadows stay cool

    strength: 0-1 (0.15-0.4 for subtle skin; 0.5-0.8 for wax/candle)
    radius: blur radius in pixels (2-6 typical)
    scatter_color: RGB multiplier for the scattered light (warm for skin)
    """
    if strength <= 0:
        return image

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return image

    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    s = float(min(1.0, max(0.0, strength)))
    r = float(max(0.5, radius))

    # Detect bright/lit regions (likely skin or translucent material)
    lum = _rgb_to_lum(img) / 255.0
    lit_mask = np.clip((lum - 0.3) / 0.5, 0.0, 1.0)  # bright areas

    # Create scatter layer: blurred version with warm color tint
    sc = np.array(scatter_color, dtype=np.float32)
    sc = sc / (sc.max() + 1e-8)  # normalize

    scatter = gaussian_filter(img, sigma=(r, r, 0))
    # Apply warm color tint to scatter
    scatter_tinted = scatter * sc[np.newaxis, np.newaxis, :]

    # Only apply SSS in lit regions (shadows don't scatter outward)
    mask = lit_mask[..., np.newaxis]
    out = img + s * mask * (scatter_tinted - img) * 0.4

    out = np.clip(out, 0.0, 255.0)
    return out.astype(np.uint8) if is_uint else out


# ---------------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------------

def apply_vignette(
    image: np.ndarray,
    strength: float = 0.3,
    softness: float = 0.5,
) -> np.ndarray:
    """
    Radial vignette: darken edges to draw the eye toward the center/subject.
    strength: 0-1 (0.15-0.4 is natural; above 0.6 is dramatic)
    softness: 0-1 (0=hard edge, 1=very gradual)
    """
    if strength <= 0:
        return image

    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    h, w = img.shape[:2]
    s = float(min(1.0, max(0.0, strength)))
    soft = float(min(1.0, max(0.05, softness)))

    yy = np.linspace(-1, 1, h)
    xx = np.linspace(-1, 1, w)
    gx, gy = np.meshgrid(xx, yy)
    r = np.sqrt(gx**2 + gy**2)

    # Smooth falloff controlled by softness
    sigma = 0.3 + soft * 0.7
    vig = 1.0 - s * np.clip(r / (sigma * 1.5), 0.0, 1.0)**2

    out = np.clip(img * vig[..., np.newaxis], 0.0, 255.0)
    return out.astype(np.uint8) if is_uint else out


# ---------------------------------------------------------------------------
# Chromatic aberration (lens character)
# ---------------------------------------------------------------------------

def apply_chromatic_aberration(
    image: np.ndarray,
    strength: float = 0.2,
) -> np.ndarray:
    """
    Subtle chromatic aberration: shift R and B channels slightly outward from center.
    Adds lens character and breaks the "too perfect" digital look.
    strength: 0-1 (0.1-0.3 is subtle; above 0.5 is obvious)
    """
    if strength <= 0:
        return image

    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    h, w = img.shape[:2]
    s = float(min(1.0, max(0.0, strength)))

    # Pixel shift amount (max ~3px at strength=1)
    shift = int(round(s * 3))
    if shift == 0:
        return image

    out = img.copy()

    # Shift red channel outward (toward top-left)
    out[shift:, shift:, 0] = img[:h-shift, :w-shift, 0]
    # Shift blue channel outward (toward bottom-right)
    out[:h-shift, :w-shift, 2] = img[shift:, shift:, 2]
    # Green stays centered

    return out.astype(np.uint8) if is_uint else out


# ---------------------------------------------------------------------------
# Micro-detail recovery
# ---------------------------------------------------------------------------

def apply_micro_detail(
    image: np.ndarray,
    strength: float = 0.3,
) -> np.ndarray:
    """
    Recover fine texture detail using a high-frequency emphasis on luminance only.
    Unlike standard sharpening, this operates at a very fine scale (radius=0.5)
    and only on luminance — no RGB halos.

    strength: 0-1 (0.2-0.5 is natural; above 0.7 is aggressive)
    """
    if strength <= 0:
        return image

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return image

    is_uint = image.dtype == np.uint8
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 255.0)
    s = float(min(1.0, max(0.0, strength)))

    lum = _rgb_to_lum(img)
    blur = gaussian_filter(lum, sigma=0.5, mode='reflect')
    hf = lum - blur  # high-frequency component
    new_lum = np.clip(lum + s * hf * 0.8, 1e-3, 255.0)

    scale = (new_lum / (lum + 1e-3))[..., np.newaxis]
    out = np.clip(img * scale, 0.0, 255.0)

    return out.astype(np.uint8) if is_uint else out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def apply_artistic_pipeline(
    image: np.ndarray,
    cfg: ArtisticPostConfig,
) -> np.ndarray:
    """
    Apply the full artistic post-processing pipeline in the correct order.

    Order matters:
    1. Value structure (sets the tonal foundation)
    2. Compositional director (nudges visual weight)
    3. SSS simulation (adds material depth)
    4. Lost-and-found edges (varies edge quality)
    5. Intentional asymmetry (breaks perfect symmetry)
    6. Micro-detail (recovers fine texture)
    7. Chromatic aberration (adds lens character)
    8. Vignette (frames the composition)
    """
    out = image

    # 1. Value structure
    if cfg.value_structure or cfg.value_shadow_lift > 0 or cfg.value_highlight_roll > 0 or cfg.value_midtone_contrast > 0:
        out = apply_value_structure(
            out,
            shadow_lift=cfg.value_shadow_lift,
            highlight_roll=cfg.value_highlight_roll,
            midtone_contrast=cfg.value_midtone_contrast,
        )

    # 2. Compositional director
    if cfg.composition_mode != "none" and cfg.composition_strength > 0:
        out = apply_compositional_director(out, cfg.composition_mode, cfg.composition_strength)

    # 3. SSS simulation
    if cfg.sss_strength > 0:
        out = apply_sss_simulation(
            out,
            strength=cfg.sss_strength,
            radius=cfg.sss_radius,
            scatter_color=cfg.sss_color,
        )

    # 4. Lost-and-found edges
    if cfg.lost_found_strength > 0:
        out = apply_lost_and_found_edges(out, cfg.lost_found_strength, cfg.lost_found_seed)

    # 5. Intentional asymmetry
    if cfg.asymmetry_strength > 0:
        out = apply_intentional_asymmetry(out, cfg.asymmetry_strength, cfg.asymmetry_seed)

    # 6. Micro-detail
    if cfg.micro_detail > 0:
        out = apply_micro_detail(out, cfg.micro_detail)

    # 7. Chromatic aberration
    if cfg.chromatic_aberration > 0:
        out = apply_chromatic_aberration(out, cfg.chromatic_aberration)

    # 8. Vignette
    if cfg.vignette_strength > 0:
        out = apply_vignette(out, cfg.vignette_strength, cfg.vignette_softness)

    return out


__all__ = [
    "ArtisticPostConfig",
    "apply_artistic_pipeline",
    "apply_value_structure",
    "apply_compositional_director",
    "apply_intentional_asymmetry",
    "apply_lost_and_found_edges",
    "apply_sss_simulation",
    "apply_vignette",
    "apply_chromatic_aberration",
    "apply_micro_detail",
]
