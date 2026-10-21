"""Prompt packs for physics-heavy scenes: fluids, transparency, rigid/soft contact, lighting.

Diffusion has no simulator, so these lean on clear material boundaries, lighting cues and negatives that
punish the usual failures (merged liquids, wrong refraction, double shadows).
"""

from __future__ import annotations

__all__ = [
    "PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN",
    "PHYSICS_COMMON_NEGATIVE_ADDON",
]


PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN = {
    "fluids": [
        "clear liquid surface, meniscus, coherent reflections on water, believable specular highlights",
        "fluid simulation quality, natural splash shape, volume conservation cues, wet surface contact line",
        "underwater caustics subtle, refractive distortion through water, air-water interface sharp",
        "pouring liquid, laminar flow then breakup, droplets with highlight and shadow",
    ],
    "transparency": [
        "transparent glass, correct refraction, visible background distortion through glass, thin-edge highlights",
        "frosted glass, diffuse transmission, soft silhouette, believable thickness",
        "clear acrylic, fresnel reflections, double reflection on curved surface, clean edges",
        "translucent material, subsurface scatter subtle, light transmission, depth-consistent opacity",
    ],
    "physics_materials": [
        "physically plausible lighting, consistent shadow direction, contact shadows, ambient occlusion",
        "material-consistent reflections, roughness matches surface, specular on wet vs dry regions",
        "rigid body contact, stable stacking, weight believable, no interpenetration",
        "single light narrative, coherent bounce light, no contradictory shadow sources",
    ],
    "soft_bodies": [
        "fabric drapes under gravity, folds follow tension, hem obeys weight, cloth collision with body",
        "soft deformation, compression where supported, natural wrinkles, no clipping through skin",
        "hair strands grouped, gravity on long hair, flyaways subtle, lighting on hair volume",
        "elastic material, stretch localized, no melting merge with background",
    ],
}


# Appended to the negative when the prompt mentions fluids / glass / physics.
PHYSICS_COMMON_NEGATIVE_ADDON = (
    "impossible physics, merged materials, contradictory lighting, floating objects, "
    "wrong refraction, flat liquid, plastic water, pasted transparency, interpenetration, "
    "muddy fluids, incoherent splash, duplicate reflections, shadow direction mismatch"
)
