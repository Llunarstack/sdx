"""
Prompt packs for **physics-heavy** scenes: fluids, transparency, rigid contact, lighting.

Diffusion models have no simulator; quality comes from **clear material boundaries**, **lighting cues**,
and **negatives that punish common failures** (merged liquids, wrong refraction, double shadows).
Use in training captions and at inference (prepend positives, append negatives).
"""

from __future__ import annotations

__all__ = [
    "PHYSICS_MATERIAL_DOMAIN_NAMES",
    "PHYSICS_MATERIAL_RECOMMENDED_PROMPTS_BY_DOMAIN",
    "PHYSICS_MATERIAL_RECOMMENDED_NEGATIVE_BY_DOMAIN",
    "PHYSICS_COMMON_NEGATIVE_ADDON",
    "FLUIDS_PROMPT_TIPS",
    "TRANSPARENCY_REFRACTION_TIPS",
    "RIGID_CONTACT_SHADOW_TIPS",
    "PARTICULATE_VOLUMETRIC_TIPS",
    "SOFT_BODY_CLOTH_TIPS",
    "PHYSICS_TRAINING_CAPTION_TIPS",
]

# Domains merged into ``RECOMMENDED_PROMPTS_BY_DOMAIN`` / ``RECOMMENDED_NEGATIVE_BY_DOMAIN`` in prompt_domains.
PHYSICS_MATERIAL_DOMAIN_NAMES = ("fluids", "transparency", "physics_materials", "soft_bodies")

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

PHYSICS_MATERIAL_RECOMMENDED_NEGATIVE_BY_DOMAIN = {
    "fluids": (
        "merged liquids, muddy water, solid-looking water, wrong viscosity, floating droplets with no surface, "
        "missing meniscus, flat water without reflection, contradictory splash direction, ice that looks like plastic"
    ),
    "transparency": (
        "wrong refraction, background not bent through glass, opaque glass, double image ghost, "
        "inconsistent transparency, see-through where metal should be, halo cutout, pasted glass effect, "
        "missing fresnel, edges too dark or too glowing, z-fighting transparency"
    ),
    "physics_materials": (
        "floating objects, no contact shadow, contradictory shadows, multiple suns, "
        "interpenetration, merged rigid objects, impossible balance, magnet-like sticking, "
        "scale inconsistency, perspective break, warped floor grid"
    ),
    "soft_bodies": (
        "cloth clipping through body, stiff cape, melted fabric, hair fused with face, "
        "tentacle merge, rubbery skin fold, anatomy melting into clothes, gravity-defying drape"
    ),
}

# Short pack to **append** to your usual negative when the prompt mentions fluids / glass / physics.
PHYSICS_COMMON_NEGATIVE_ADDON = (
    "impossible physics, merged materials, contradictory lighting, floating objects, "
    "wrong refraction, flat liquid, plastic water, pasted transparency, interpenetration, "
    "muddy fluids, incoherent splash, duplicate reflections, shadow direction mismatch"
)

FLUIDS_PROMPT_TIPS = [
    "Name the **container** and **fill level**: 'water glass half full', 'wine in stem glass' reduces flat-puddle outputs.",
    "Describe **one** dominant fluid; 'oil and water' needs 'separate layers, sharp interface' or the model may blend them.",
    "Add **light direction** matching the scene: 'rim light on droplets', 'window reflection on water surface'.",
    "For splashes: 'single coherent splash crown', 'droplets with shadow on surface' beats vague 'splash'.",
    "Underwater: 'caustic light patterns on floor', 'snell distortion at interface' cues refraction without jargon overload.",
]

TRANSPARENCY_REFRACTION_TIPS = [
    "Specify **thickness**: 'thin drinking glass' vs 'thick crystal vase' changes refraction strength expectations.",
    "Mention **what is seen through** the glass: 'city lights refracted', 'hand behind frosted glass silhouette'.",
    "Use 'fresnel highlight on edges' or 'bright edge, darker center' for curved glass readability.",
    "For stacked panes: 'double reflection', 'offset ghost image subtle' can reduce single-pane paste look.",
    "Frosted: 'silhouette readable, detail blurred' separates it from opaque grey blobs.",
]

RIGID_CONTACT_SHADOW_TIPS = [
    "State **contact**: 'book resting on table', 'feet on ground' to anchor shadows.",
    "One **key light** plus soft fill avoids contradictory cast shadows.",
    "Add 'contact shadow under object', 'occlusion at junction' for small-scale believability.",
    "Stacking: 'center of mass over base', 'stable pile' reduces impossible towers.",
]

PARTICULATE_VOLUMETRIC_TIPS = [
    "Smoke/fog: 'volumetric god rays', 'soft falloff with depth' beats flat grey overlay.",
    "Dust: 'particles in backlight', 'specks with bokeh' for depth; avoid uniform noise.",
    "Rain: 'streak direction matches wind', 'wet pavement reflections' ties weather to ground.",
    "Fire: 'turbulent plume', 'cooler soot core' can separate flame from solid objects.",
]

SOFT_BODY_CLOTH_TIPS = [
    "Cloth: name **weight** — 'heavy wool drape' vs 'light silk flow'.",
    "Collisions: 'skirt clears chair edge', 'sleeve wraps wrist' reduces clipping language.",
    "Hair: 'ponytail pulled by gravity', 'bangs cast shadow on forehead' links volume to light.",
]

PHYSICS_TRAINING_CAPTION_TIPS = [
    "Pair rare physics words with **visible outcomes**: not just 'water' but 'water with window reflection distorted'.",
    "Include **negatives in data sparingly**; prefer positive descriptions of correct behavior (caption_utils can still add quality tags).",
    "Multi-domain scenes: order **material then lighting** — 'glass of water on wood table, afternoon side light'.",
    "For transparency datasets, vary **background complexity** so the model learns refraction, not cutout masks.",
]
