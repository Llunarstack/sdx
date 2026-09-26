"""Texture / photoreal inventions 38–48."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "LensExifPlan",
    "plan_lens_exif",
    "sensor_noise_scale",
    "demosaic_realism_addon",
    "sss_hint_addon",
    "asymmetry_lottery_addon",
    "jpeg_spice_addon",
    "filmic_highlight_addon",
    "grime_corner_addon",
    "anti_beauty_filter_addon",
    "brdf_material_addon",
    "histogram_match_hint",
]


@dataclass
class LensExifPlan:
    focal_mm: float = 50.0
    film_stock: str = "portra400"
    aperture: float = 2.8
    positive: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FILM = {
    "portra400": "Kodak Portra 400 color science, soft skin rolloff",
    "tri_x": "Kodak Tri-X push grain, high contrast B&W",
    "velvia": "Fuji Velvia saturated landscape film",
    "cinestill800t": "CineStill 800T tungsten halation",
}


def plan_lens_exif(*, focal_mm: float = 50.0, film_stock: str = "portra400", aperture: float = 2.8) -> LensExifPlan:
    film = _FILM.get(film_stock, film_stock)
    pos = f"{focal_mm:.0f}mm lens look, f/{aperture:.1f}, optical vignetting subtle, {film}, EXIF-plausible photography"
    return LensExifPlan(focal_mm=focal_mm, film_stock=film_stock, aperture=aperture, positive=pos)


def sensor_noise_scale(progress: float, iso: float = 400.0) -> float:
    """Late-step luminance noise strength from ISO (#40)."""
    p = float(progress)
    if p < 0.7:
        return 0.0
    iso_f = max(100.0, float(iso)) / 400.0
    return float(0.01 * iso_f * (p - 0.7) / 0.3)


def demosaic_realism_addon() -> tuple[str, str]:
    return (
        "subtle Bayer demosaic texture, real camera capture feel",
        "perfect synthetic CG smoothness, unreal engine plastic",
    )


def sss_hint_addon() -> tuple[str, str]:
    return (
        "subsurface scattering in ears and nostrils, soft skin translucency",
        "opaque wax skin, no subsurface scattering",
    )


def asymmetry_lottery_addon(seed: int = 0) -> tuple[str, str]:
    r = random.Random(int(seed))
    side = r.choice(["left", "right"])
    bit = r.choice(["eyebrow higher", "catchlight stronger", "ear slightly visible", "smirk"])
    return (f"natural asymmetry: {side} {bit}", "bilateral perfect symmetry, mannequin face")


def jpeg_spice_addon(quality: int = 92) -> tuple[str, str]:
    return (
        f"mild JPEG {quality} compression character, uploaded-photo realism",
        "raw unprocessed scientific float precision look",
    )


def filmic_highlight_addon() -> tuple[str, str]:
    return (
        "filmic highlight rolloff, soft clipped whites, no RGB burn",
        "harsh digital highlight clipping, neon burned whites",
    )


def grime_corner_addon() -> tuple[str, str]:
    return (
        "subtle corner grime, lived-in micro-occlusion dirt",
        "sterile showroom cleanliness everywhere",
    )


def anti_beauty_filter_addon() -> tuple[str, str]:
    return (
        "unfiltered skin, visible pores and peach fuzz",
        "Instagram beauty filter, Facetune smooth, poreless plastic",
    )


def brdf_material_addon(material: str = "skin") -> tuple[str, str]:
    table = {
        "skin": ("dielectric skin BRDF, soft specular", "metallic skin"),
        "metal": ("conductor BRDF, sharp environment reflections", "diffuse chalk metal"),
        "glass": ("specular transmission, refraction cues", "opaque frosted wrong glass"),
        "fabric": ("anisotropic fabric sheen, weave microspecular", "plastic painted cloth"),
    }
    return table.get(material, table["skin"])


def histogram_match_hint() -> str:
    return "match local contrast histogram to reference corpus stills when refs exist"
