"""
Prompt snippets, negative add-ons, and aspect hints for book/comic/manga workflows.

Grounded in common sequential-art practice (ink, screentone, panels, lettering).
Used by ``generate_book.py`` (``--lexicon-style``, ``--aspect-preset``) and by tools.

This module does **not** call external APIs; it only returns strings and (w, h) tuples.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Negative prompts — reduce typical gen-AI failures on comic/manga pages
# ---------------------------------------------------------------------------

LETTERING_NEGATIVE_ADDON = (
    "gibberish text, misspelled words, wrong letters, watermark, artist signature on art, "
    "merged speech bubbles, unreadable tiny font, text outside bubble, subtitles style"
)

ANATOMY_PANEL_NEGATIVE_ADDON = (
    "extra fingers, fused fingers, mangled hands, duplicate faces, asymmetrical eyes, "
    "broken panel borders, random grid overlay"
)

INK_STYLE_NEGATIVE_ADDON = (
    "plastic skin, airbrushed, overly smooth shading, CGI render, 3d model look, "
    "chromatic aberration, heavy jpeg artifacts"
)

ARTIST_LETTERING_STRICT_NEGATIVE = (
    "crossing balloon tails, uncertain speaker attribution, balloon overlap on faces, "
    "tiny unreadable dialogue, caption boxes out of reading order"
)

# Stricter panel / lettering failures (use with --book-accuracy production)
PRODUCTION_TIER_NEGATIVE_ADDON = (
    "cropped dialogue, speech balloon pointing wrong character, overlapping illegible text, "
    "spine misaligned title on cover, barcode on interior page, low dpi moire, muddy screentone"
)

def combined_comic_negative(*, include_lettering: bool = True, include_anatomy: bool = True) -> str:
    parts = [INK_STYLE_NEGATIVE_ADDON]
    if include_lettering:
        parts.append(LETTERING_NEGATIVE_ADDON)
    if include_anatomy:
        parts.append(ANATOMY_PANEL_NEGATIVE_ADDON)
    return ", ".join(parts)

# ---------------------------------------------------------------------------
# Style snippets (append to user prompt or book prefix)
# ---------------------------------------------------------------------------

STYLE_SNIPPETS: Dict[str, str] = {
    "none": "",
    "shonen": "dynamic action lines, speed lines, bold ink weight, expressive eyes, impact frames",
    "shoujo": "delicate linework, soft screentone gradients, sparkles, emotional close-ups, flower motifs",
    "seinen": "realistic proportions, heavy blacks, detailed backgrounds, mature atmosphere, fine hatching",
    "slice_of_life": "everyday setting, calm composition, natural lighting, cozy atmosphere, clear silhouettes",
    "chibi": "super deformed, large head small body, cute proportions, simplified features, comedy timing",
    "webtoon": "vertical composition, mobile reading format, wide establishing shots, scrolling-friendly layout",
    "manhwa_color": "full color, soft gradients, clean lineart, korean webtoon shading, glossy highlights",
    "graphic_novel": "cinematic lighting, painterly ink, cross-hatching, dramatic shadows, graphic novel composition",
    "editorial": "clear hierarchy, readable at small size, professional print margins implied, balanced negative space",
    "light_novel": "light novel cover illustration, ornate title treatment, character pin-up, publisher-ready layout",
    "yonkoma": "four panel strip, gag beat timing, simple backgrounds, punchline panel emphasis",
    # Broader modern creation styles
    "anime_2d": "2d anime look, clean lineart, controlled cel shading, expressive face language, stable style identity",
    "anime_3d": "3d anime render, toon-shaded forms, clean anime face proportions, stylized but coherent lighting",
    "cartoon_2d": "2d cartoon style, simplified readable forms, graphic shape language, appealing squash-and-stretch energy",
    "cartoon_3d": "3d cartoon style, stylized proportions, readable silhouettes, family-film visual clarity",
    "web_comic": "web comic panel readability, clean dialogue staging, high mobile legibility, consistent episode style",
    "digital_art": "digital illustration workflow, intentional brush economy, clean layer discipline, focal rendering hierarchy",
    "digital_3d": "digital 3d artwork, coherent materials, controlled stylization level, lighting consistency across scene assets",
    "drawing_ink": "drawing-first look, confident line quality, readable construction, purposeful hatch rhythm",
    "painting_oil": "oil painting treatment, visible brush character, painterly edge variety, intentional value grouping",
    "painting_watercolor": "watercolor treatment, transparent layering, paper interaction cues, controlled pigment blooms",
    "realistic_photo": "realistic photographic rendering, plausible optics and depth, natural skin/material texture, grounded light behavior",
    "realistic_painting": "realist painting treatment, anatomy-accurate forms, nuanced value transitions, observational detail discipline",
    "fantasy_concept": "fantasy concept-art style, worldbuilding motifs, cinematic atmosphere, disciplined focal hierarchy",
    "sci_fi_concept": "sci-fi concept-art style, coherent design language, believable tech forms, controlled lighting logic",
    "cyberpunk": "cyberpunk art direction, neon-accented night palette, atmospheric perspective, layered urban depth",
    "steampunk": "steampunk visual language, brass/leather material motifs, mechanical ornamentation, period-tech coherence",
    "noir_comic": "noir comic style, high-contrast chiaroscuro, moody rain-lit environments, strong shadow storytelling",
    "art_nouveau": "art nouveau style cues, flowing ornamental line rhythm, elegant decorative framing, poster-readability",
    "ukiyo_e": "ukiyo-e inspired treatment, flat color planes, woodblock-like contour discipline, decorative composition balance",
    "pixel_retro": "retro pixel style, limited palette design, sprite-readability discipline, clean cluster placement",
    "voxel_isometric": "isometric voxel style, clean block structure, readable depth layering, stylized world consistency",
    "clay_stop_motion": "clay/stop-motion inspired style, tactile handcrafted forms, miniature-set lighting cues, material irregularity",
    "render_octane": "high-fidelity octane-like render style, controlled bloom, cinematic ray-traced reflections, polished material separation",
    "render_eevee": "fast stylized eevee-like render style, clear shape readability, balanced realtime lighting response",
    "editorial_fashion_photo": "editorial fashion photo language, controlled styling hierarchy, premium print-ready framing and lighting",
    "black_white_film": "black-and-white film language, silver-grain tonal rolloff, dramatic monochrome value storytelling",
    # Fine-art movement inspired
    "baroque": "baroque-inspired visual language, dramatic chiaroscuro, theatrical composition, rich material depth",
    "rococo": "rococo-inspired ornamental elegance, pastel luxury palette, refined decorative rhythm, graceful silhouettes",
    "impressionist": "impressionist-inspired brush economy, atmospheric color vibration, light-first form simplification",
    "expressionist": "expressionist-inspired emotional distortion, bold contour energy, dramatic color intensity for mood",
    "cubist": "cubist-inspired geometric decomposition, multi-plane form language, structured abstract readability",
    "surrealist": "surrealist-inspired dream logic, symbolic juxtaposition, uncanny yet compositionally coherent staging",
    "art_deco": "art-deco inspired geometry, premium symmetry accents, elegant poster hierarchy, metallic motif discipline",
    "minimalist": "minimalist style treatment, reduced element count, strong negative space, high intentional composition clarity",
    "maximalist": "maximalist style treatment, dense ornamental layering, rich texture complexity, disciplined focal hierarchy",
    # Anime/manga specializations
    "anime_shonen_battle": "shonen battle anime style, kinetic action framing, impact effects, strong hero-villain silhouette readability",
    "anime_shojo_romance": "shojo romance anime style, emotive close-up language, soft decorative accents, elegant line rhythm",
    "anime_seinen_gritty": "seinen gritty anime style, mature tonal control, grounded anatomy, cinematic panel tension",
    "anime_isekai_fantasy": "isekai fantasy anime style, luminous magical motifs, costume readability, adventurous worldbuilding atmosphere",
    "anime_mecha": "mecha anime style, machine silhouette clarity, hard-surface readability, dynamic cockpit/action staging",
    "anime_idol": "idol-anime visual style, polished stage-light color scripting, expressive performance poses, clean fashion detail",
    # Web/comic and print niches
    "newspaper_comic": "newspaper comic-strip style, concise panel storytelling, ink-first readability, punchline rhythm clarity",
    "manga_horror": "horror manga style, psychological contrast staging, unsettling shadow design, tension-focused composition pacing",
    "retro_pulp_cover": "retro pulp-cover style, bold title-area composition, vintage paint/print texture cues, dramatic color blocks",
    "poster_graphic": "graphic poster style, clear typographic-safe composition zones, high-impact silhouette hierarchy",
    # 3d/render specializations
    "archviz_real": "archviz realism style, clean perspective control, believable material response, premium interior/exterior lighting",
    "product_cg": "product-CG style, precision form edges, specular discipline, commercial hero-shot clarity",
    "clay_render_bw": "monochrome clay-render style, form readability first, neutral lighting for silhouette and volume evaluation",
    "toon_render_hybrid": "toon-render hybrid style, stylized contour logic with controlled realistic light response",
    # Photography language
    "film_35mm": "35mm film-photo language, organic grain character, natural highlight rolloff, lens personality cues",
    "polaroid_vintage": "polaroid-vintage photo style, instant-film tonal character, nostalgic color cast control, analog texture feel",
    "street_noir_photo": "street noir photo language, wet-night reflections, practical-light contrast drama, candid urban framing",
    "wildlife_naturalist": "wildlife naturalist photo style, habitat authenticity, natural detail falloff, respectful observational framing",
}

ART_MEDIUM_VARIANTS: Dict[str, Dict[str, str]] = {
    "none": {},
    "digital_art": {
        "none": "",
        "painting": "digital painting workflow, textured brush packs with controlled edge variety, layered value design",
        "cel_shaded": "clean cel-shaded digital workflow, hard-light transitions, graphic shadow shape control",
        "semi_real": "semi-real digital rendering, believable materials with stylized simplification, controlled detail density",
        "vector": "vector-illustration workflow, crisp shape intersections, consistent stroke and fill grammar",
        "pixel": "pixel-art workflow, strict pixel grid discipline, deliberate cluster placement and palette economy",
        "photobash": "photobash-assisted digital art, coherent blend passes, perspective and lighting unification",
        "lineart_clean": "clean digital lineart medium, deliberate contour economy, line-weight hierarchy, anti-noise stroke discipline",
        "comic_halftone": "digital comic-halftone medium, controlled dot-screen texture, print-safe value separation",
        "matte_concept": "matte concept workflow, large-shape value planning, atmosphere depth stacking, art-directable composition blocks",
        "anime_render": "anime-oriented digital render workflow, stable face grammar, controlled cel highlights, readable silhouette first",
        "concept_sheet": "concept-sheet digital workflow, orthographic readability, clean callout zones, design-iteration clarity",
        "visual_dev": "visual-development medium, exploratory shape language, color-key discipline, iterative scene storytelling",
        "ui_icon": "ui-icon illustration medium, crisp silhouette simplification, scalable readability, clean vector-informed polish",
        "storybook_paint": "storybook digital-paint medium, warm narrative color design, readable character staging, print-safe texture discipline",
    },
    "drawing_art": {
        "none": "",
        "pencil": "pencil drawing medium, graphite pressure variation, clean construction lines, controlled shading planes",
        "graphite": "graphite drawing finish, smooth-to-grain transition control, realistic tonal buildup",
        "charcoal": "charcoal drawing look, expressive dark massing, smudge control, directional gesture marks",
        "ink": "ink drawing medium, confident line economy, varied line weight, intentional black spotting",
        "pen_ink": "pen-and-ink medium, hatching discipline, contour clarity, crosshatch value control",
        "marker": "marker drawing medium, layered marker gradients, clear shape blocks, edge cleanliness",
        "pastel": "pastel drawing medium, powdery pigment texture, soft transitions with controlled accents",
        "chalk": "chalk drawing medium, matte chalk texture, gestural edges, visible substrate interaction",
        "crayon": "crayon drawing medium, waxy stroke character, playful texture, color layering marks",
        "conte": "conte drawing medium, warm earthy marks, expressive pressure variation, controlled smudge transitions",
        "technical_pen": "technical-pen drawing medium, precise line fidelity, clean construction discipline, measured hatch spacing",
        "blueprint": "blueprint drafting style, technical line hierarchy, structural annotation feel, clear geometric readability",
        "gesture": "gesture-drawing medium, rapid pose energy, movement-first line economy, expressive anatomical flow",
        "life_study": "life-study drawing medium, observational proportion discipline, form-turn shading clarity, subtle line restraint",
        "crosshatch": "crosshatch-focused drawing medium, directional hatch logic, tonal layering control, contour-readability discipline",
        "comic_pencil": "comic-pencil medium, construction-to-finish workflow, dynamic figure posing, clear storytelling line rhythm",
    },
    "digital_3d_art": {
        "none": "",
        "stylized": "stylized 3d art medium, shape-language exaggeration, readable forms, coherent toon-to-pbr balance",
        "pbr_realistic": "realistic 3d pbr medium, physically plausible roughness/metalness behavior, grounded lighting response",
        "hard_surface": "hard-surface 3d medium, clean bevel logic, panel breakup discipline, functional mechanical detailing",
        "character_sculpt": "character sculpt render medium, anatomy-aware forms, clean primary-secondary-tertiary shape hierarchy",
        "clay_render": "clay render medium, neutral material preview lighting, silhouette and form readability first",
        "toon_3d": "toon 3d medium, anime/cartoon contour readability, clean ramp shading, controlled specular stylization",
        "low_poly": "low-poly 3d medium, deliberate faceted geometry language, efficient texture readability",
        "voxel": "voxel 3d medium, block-structure coherence, grid discipline, stylized volumetric readability",
        "octane_render": "octane-style render medium, cinematic ray-traced highlights, clean material separation, filmic contrast control",
        "eevee_render": "eevee-style render medium, stylized realtime lighting, clear silhouette readability, efficient shading clarity",
        "isometric_3d": "isometric 3d medium, perspective-consistent blockouts, readable elevation layering, gameplay-friendly composition",
        "kitbash_env": "3d kitbash environment medium, coherent module scale, lighting-unified assembly, believable scene construction",
        "archviz": "architectural-visualization 3d medium, premium interior/exterior lighting, material plausibility, camera composition discipline",
        "product_render": "product-render 3d medium, commercial studio-light setup, clean reflections, precise edge readability",
        "vfx_concept": "vfx-concept 3d medium, cinematic scene blocking, atmosphere layering, film-shot continuity cues",
        "mecha_3d": "mecha-focused 3d medium, hard-surface hierarchy, panel-seam discipline, readable mechanical articulation",
    },
    "painting_art": {
        "none": "",
        "oil": "oil painting medium, impasto accents, rich color temperature shifts, painterly edge transitions",
        "watercolor": "watercolor painting medium, transparent wash layering, pigment pooling control, paper tooth interaction",
        "gouache": "gouache painting medium, matte opaque planes, decisive shape simplification, controlled edge hierarchy",
        "acrylic": "acrylic painting medium, opaque layer stacking, crisp-over-soft edge rhythm, color block confidence",
        "tempera": "tempera painting medium, matte historical finish, fine hatch-texture layering, restrained palette unity",
        "fresco": "fresco painting medium, wall-plaster texture cues, muted mineral palette, broad-form value masses",
        "ink_wash": "ink-wash painting medium, tonal brush dilution control, atmospheric gradation, calligraphic flow",
        "impasto": "impasto-heavy painting medium, thick paint body, tactile stroke relief, directional mark structure",
        "plein_air": "plein-air painting medium, natural outdoor light observation, fast decisive brush economy, atmospheric color shifts",
        "digital_oil": "digital-oil painting medium, buttery stroke blending with intentional edge breakup, painterly form modeling",
        "watercolor_ink": "watercolor-plus-ink medium, transparent wash structure with crisp ink accents, controlled value anchoring",
        "illustration_poster": "illustration-poster paint medium, strong shape simplification, bold graphic value grouping, print-readability priority",
        "surreal_paint": "surreal painting medium, dreamlike motif transitions, symbolic compositional logic, controlled uncanny mood",
        "noir_paint": "noir painting medium, dramatic shadow geometry, moody palette control, cinematic tension framing",
        "mural": "mural-style painting medium, broad readable forms, public-space composition scale, durable color-block design",
    },
    "realistic_art": {
        "none": "",
        "photoreal": "photoreal medium treatment, optical realism, physically plausible textures, natural dynamic range",
        "studio_portrait": "studio portrait realism medium, key-fill-rim discipline, skin microtexture fidelity, lens-aware depth",
        "cinematic_still": "cinematic still realism medium, motivated practical lighting, filmic composition grammar, tonal rolloff discipline",
        "documentary": "documentary realism medium, candid behavioral authenticity, environmental storytelling realism",
        "architecture": "architectural realism medium, vertical perspective discipline, material reflectance plausibility, clean spatial depth",
        "product": "product realism medium, precise form edges, controlled specular highlights, catalog-quality clarity",
        "wildlife_macro": "wildlife/macro realism medium, natural detail falloff, authentic microtexture, plausible focal depth",
        "realism_painting": "realist painting medium, observational proportion accuracy, subtle form modeling, measured brush economy",
        "street_photo": "street-photography realism medium, candid timing authenticity, grounded environmental perspective, natural light behavior",
        "fashion_editorial": "fashion-editorial realism medium, controlled studio/location lighting, styling clarity, premium composition polish",
        "black_white_film": "black-and-white film realism medium, monochrome tonal discipline, silver-grain texture cueing, cinematic contrast control",
        "drone_landscape": "drone-landscape realism medium, aerial depth layering, believable atmospheric perspective, large-scale scene coherence",
        "sports_photo": "sports-photography realism medium, action-timing clarity, subject isolation under motion, lens-appropriate compression",
        "wedding_photo": "wedding-photography realism medium, flattering skin tone rendering, candid emotional timing, premium event-light handling",
        "food_photo": "food-photography realism medium, appetizing material detail, controlled specular highlights, editorial table composition",
        "macro_product": "macro product realism medium, microscopic texture clarity, shallow-depth discipline, high-end commercial polish",
    },
    "anime_cartoon_webcomic": {
        "none": "",
        "anime_2d": "2d anime medium, clean linework, controlled cel shading, stable facial grammar",
        "anime_3d": "3d anime medium, toon-surface readability, anime-proportion consistency, stylized camera language",
        "manga_bw": "manga black-and-white medium, screentone discipline, dramatic black placement, panel rhythm clarity",
        "manhwa_color": "manhwa color medium, clean digital color rendering, webtoon readability, polished character finish",
        "cartoon_2d": "2d cartoon medium, bold graphic simplification, readable expressions, playful posing language",
        "cartoon_3d": "3d cartoon medium, appealing stylized volumes, clear silhouette language, family animation polish",
        "western_comic": "western comic medium, panel storytelling cadence, dynamic figure staging, inking readability",
        "web_comic": "web comic medium, mobile-first readability, dialogue clarity, consistent episodic visual identity",
        "webtoon_vertical": "vertical webtoon medium, top-to-bottom beat spacing, scroll-friendly framing, clean dialogue grouping",
        "chibi": "chibi stylization medium, super-deformed proportions, clear cute silhouette language, comedic timing clarity",
        "isekai_anime": "isekai anime medium, luminous fantasy atmosphere, clean anime rendering grammar, readable costume motifs",
        "mecha_anime": "mecha anime medium, mechanical silhouette readability, panel-friendly hard-surface clarity, dynamic action staging",
        "western_animation": "western animation medium, expressive pose language, clear shape appeal, production-ready color scripting",
        "gag_strip": "gag-strip comic medium, setup-payoff rhythm clarity, readable expression beats, concise panel storytelling",
        "anime_movie": "anime-movie style medium, cinematic background painting depth, high-finish keyframe polish, emotive camera language",
        "anime_tv_clean": "anime-tv clean medium, production-friendly line clarity, efficient cel-value control, stable character-on-model consistency",
        "manhua": "manhua style medium, polished color rendering, dramatic composition emphasis, serialized readability",
        "superhero_modern": "modern superhero comic medium, dynamic anatomy exaggeration, impact staging, color-scripted action readability",
        "indie_webtoon": "indie webtoon medium, distinctive personal line grammar, mobile-first readability, episodic beat consistency",
    },
    "mixed_media_art": {
        "none": "",
        "collage": "mixed-media collage medium, layered paper/photo fragments, coherent value grouping, intentional cut-edge rhythm",
        "paper_cut": "paper-cut medium, crisp layered silhouette logic, tactile handcrafted depth, clean color-block readability",
        "risograph": "risograph print medium, limited-ink palette discipline, registration-character charm, poster-ready graphic clarity",
        "screenprint": "screenprint medium, bold separations, ink-layer stacking logic, texture-aware flat-shape design",
        "ink_paint_mix": "ink-plus-paint mixed medium, controlled line-then-mass workflow, material interplay with clear focal priority",
    },
}

ART_MEDIUM_PACK_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "digital_painting_pro": {"family": "digital_art", "variant": "painting"},
    "drawing_ink_pro": {"family": "drawing_art", "variant": "ink"},
    "stylized_3d_game": {"family": "digital_3d_art", "variant": "stylized"},
    "pbr_3d_realism": {"family": "digital_3d_art", "variant": "pbr_realistic"},
    "oil_painting_classic": {"family": "painting_art", "variant": "oil"},
    "watercolor_storybook": {"family": "painting_art", "variant": "watercolor"},
    "photo_real_cinematic": {"family": "realistic_art", "variant": "cinematic_still"},
    "anime_2d_pro": {"family": "anime_cartoon_webcomic", "variant": "anime_2d"},
    "anime_3d_pro": {"family": "anime_cartoon_webcomic", "variant": "anime_3d"},
    "cartoon_2d_pro": {"family": "anime_cartoon_webcomic", "variant": "cartoon_2d"},
    "webcomic_mobile": {"family": "anime_cartoon_webcomic", "variant": "web_comic"},
    "fantasy_concept_keyart": {"family": "digital_art", "variant": "matte_concept"},
    "cyberpunk_noir_panel": {"family": "anime_cartoon_webcomic", "variant": "western_comic"},
    "mecha_anime_action": {"family": "anime_cartoon_webcomic", "variant": "mecha_anime"},
    "editorial_fashion_real": {"family": "realistic_art", "variant": "fashion_editorial"},
    "film_noir_bw_real": {"family": "realistic_art", "variant": "black_white_film"},
    "octane_3d_cinematic": {"family": "digital_3d_art", "variant": "octane_render"},
    "isometric_voxel_world": {"family": "digital_3d_art", "variant": "voxel"},
    "mixed_media_collage": {"family": "mixed_media_art", "variant": "collage"},
    "risograph_poster": {"family": "mixed_media_art", "variant": "risograph"},
    "concept_sheet_design": {"family": "digital_art", "variant": "concept_sheet"},
    "visual_dev_story": {"family": "digital_art", "variant": "visual_dev"},
    "comic_pencil_storyboard": {"family": "drawing_art", "variant": "comic_pencil"},
    "crosshatch_ink_master": {"family": "drawing_art", "variant": "crosshatch"},
    "archviz_cinematic": {"family": "digital_3d_art", "variant": "archviz"},
    "product_cg_studio": {"family": "digital_3d_art", "variant": "product_render"},
    "mecha_3d_detail": {"family": "digital_3d_art", "variant": "mecha_3d"},
    "surreal_paint_studio": {"family": "painting_art", "variant": "surreal_paint"},
    "mural_graphic_large": {"family": "painting_art", "variant": "mural"},
    "sports_photo_pro": {"family": "realistic_art", "variant": "sports_photo"},
    "wedding_photo_editorial": {"family": "realistic_art", "variant": "wedding_photo"},
    "food_photo_editorial": {"family": "realistic_art", "variant": "food_photo"},
    "anime_movie_keyart": {"family": "anime_cartoon_webcomic", "variant": "anime_movie"},
    "superhero_action_modern": {"family": "anime_cartoon_webcomic", "variant": "superhero_modern"},
    "indie_webtoon_episode": {"family": "anime_cartoon_webcomic", "variant": "indie_webtoon"},
}

# Artist-oriented production bundles inspired by common comic/manga workflows:
# - clear reading order and eye path
# - controlled shot language (establishing -> medium -> close-up)
# - value hierarchy and texture discipline (screentone/halftone)
ARTIST_CRAFT_PROFILES: Dict[str, str] = {
    "none": "",
    "manga_pro": (
        "clear right-to-left panel flow, dominant-to-secondary focal hierarchy, "
        "establishing shot then medium then close-up rhythm, disciplined screentone values, "
        "clean silhouettes with strategic black fills"
    ),
    "western_comic_pro": (
        "clear left-to-right panel flow, readable gutter transitions, "
        "cinematic shot progression (establishing medium close-up), "
        "strong figure-ground separation, balloon-safe composition"
    ),
    "webtoon_pro": (
        "vertical scroll rhythm, long-to-short beat spacing, mobile-first readability, "
        "staging with strong top-to-bottom eye path, clear dialogue grouping per beat"
    ),
    "children_book": (
        "large readable shapes, simple value grouping, warm storytelling poses, "
        "clean focal hierarchy with generous negative space for text"
    ),
    "cinematic_storyboard": (
        "shot continuity discipline, decisive camera axis, clear staging per beat, "
        "high readability thumbnails, intent-first framing"
    ),
}

SHOT_LANGUAGE_HINTS: Dict[str, str] = {
    "none": "",
    "mixed": "balanced mix of establishing, medium, and close-up shots with clear continuity",
    "cinematic": "film-language shot grammar, motivated camera changes, over-shoulder dialogue coverage",
    "manga_dynamic": "dynamic manga framing, diagonal action staging, impact close-ups with speed emphasis",
    "dialogue_coverage": "dialogue-first shot coverage, over-shoulder and reaction close-ups, readable speaker turns",
}

PACING_PLAN_HINTS: Dict[str, str] = {
    "none": "",
    "decompressed": "decompressed pacing, more panels per action beat, breathing room in gutters",
    "balanced": "balanced pacing with alternating wide setup and tight emotional beats",
    "compressed": "compressed pacing, fewer panels with decisive story beats and efficient transitions",
}

LETTERING_CRAFT_HINTS: Dict[str, str] = {
    "none": "",
    "standard": (
        "speech balloons placed in reading order, tails clearly pointing to speaker, "
        "text kept inside balloons with consistent margin"
    ),
    "strict": (
        "strict lettering discipline, top-to-bottom reading path, non-intersecting balloon tails, "
        "speaker-first balloon placement, avoid covering faces/hands"
    ),
}

VALUE_PLAN_HINTS: Dict[str, str] = {
    "none": "",
    "bw_hierarchy": "black-white value hierarchy, three-step value grouping, focal point highest contrast",
    "color_script": "cohesive color script across page beats, controlled palette shifts, value-first readability",
}

SCREENTONE_PLAN_HINTS: Dict[str, str] = {
    "none": "",
    "clean": "clean screentone application, moire-safe dot scale, controlled gradients on form turns",
    "dramatic": "dramatic screentone contrast, heavy blacks plus selective tone gradients for depth",
}

ORIGINAL_CHARACTER_ARCHETYPES: Dict[str, str] = {
    "none": "",
    "shonen_lead": "energetic protagonist silhouette, readable hero shape language, expressive action-ready posture",
    "cool_rival": "sharp rival silhouette, restrained expression set, angular design accents",
    "mentor": "grounded mentor presence, mature posture language, iconic costume readability",
    "antihero": "edgy antihero contrast, asymmetrical design motifs, controlled intensity in expression",
    "magical_girl": "clean iconic outfit language, transformation-ready silhouette, emotive face readability",
    "noir_detective": "detective silhouette cues, coat/hat shape identity, moody expression control",
    "space_pilot": "functional sci-fi costume logic, helmet/gear identity anchors, practical movement silhouette",
}

ARTIST_PACK_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "manga_cinematic": {
        "craft_profile": "manga_pro",
        "shot_language": "manga_dynamic",
        "pacing_plan": "balanced",
        "lettering_craft": "strict",
        "value_plan": "bw_hierarchy",
        "screentone_plan": "dramatic",
    },
    "comic_dialogue": {
        "craft_profile": "western_comic_pro",
        "shot_language": "dialogue_coverage",
        "pacing_plan": "decompressed",
        "lettering_craft": "strict",
        "value_plan": "color_script",
        "screentone_plan": "clean",
    },
    "webtoon_scroll": {
        "craft_profile": "webtoon_pro",
        "shot_language": "mixed",
        "pacing_plan": "decompressed",
        "lettering_craft": "standard",
        "value_plan": "color_script",
        "screentone_plan": "clean",
    },
    "storyboard_fast": {
        "craft_profile": "cinematic_storyboard",
        "shot_language": "cinematic",
        "pacing_plan": "compressed",
        "lettering_craft": "none",
        "value_plan": "bw_hierarchy",
        "screentone_plan": "none",
    },
}

OC_PACK_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "heroine_scifi": {
        "archetype": "space_pilot",
        "visual_traits": "short asymmetric hair, sharp brow shape, utility earpiece",
        "wardrobe": "flight jacket, utility belt, reinforced boots",
        "silhouette": "broad upper torso shape with tapered legs",
        "color_motifs": "teal accents on dark neutral base",
        "expression_sheet": "confident smirk, focused glare, determined shout",
    },
    "rival_dark": {
        "archetype": "cool_rival",
        "visual_traits": "narrow eyes, angular fringe, distinct facial mark",
        "wardrobe": "high-collar coat with geometric trim",
        "silhouette": "tall narrow silhouette with sharp shoulder points",
        "color_motifs": "black, crimson, steel gray",
        "expression_sheet": "cold stare, restrained smirk, contempt glance",
    },
    "mentor_classic": {
        "archetype": "mentor",
        "visual_traits": "older face planes, pronounced brow, calm gaze",
        "wardrobe": "layered robe or coat with iconic accessory",
        "silhouette": "stable triangular silhouette",
        "color_motifs": "earth tones with one signature accent color",
        "expression_sheet": "calm smile, stern warning, reflective concern",
    },
}

BOOK_STYLE_PACK_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "manga_nsfw_action": {
        "artist_pack": "manga_cinematic",
        "oc_pack": "none",
        "safety_mode": "nsfw",
        "nsfw_pack": "explicit_detail",
        "nsfw_civitai_pack": "action",
        "civitai_trigger_bank": "medium",
    },
    "webtoon_nsfw_romance": {
        "artist_pack": "webtoon_scroll",
        "oc_pack": "none",
        "safety_mode": "nsfw",
        "nsfw_pack": "romantic",
        "nsfw_civitai_pack": "style",
        "civitai_trigger_bank": "light",
    },
    "comic_dialogue_safe": {
        "artist_pack": "comic_dialogue",
        "oc_pack": "none",
        "safety_mode": "sfw",
        "nsfw_pack": "none",
        "nsfw_civitai_pack": "none",
        "civitai_trigger_bank": "none",
    },
    "oc_launch_safe": {
        "artist_pack": "manga_cinematic",
        "oc_pack": "heroine_scifi",
        "safety_mode": "sfw",
        "nsfw_pack": "none",
        "nsfw_civitai_pack": "none",
        "civitai_trigger_bank": "none",
    },
}

HUMANIZE_PACK_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "lite": {
        "humanize_profile": "lite",
        "imperfection_level": "lite",
        "materiality_mode": "paper",
        "asymmetry_level": "lite",
        "negative_level": "lite",
    },
    "balanced": {
        "humanize_profile": "balanced",
        "imperfection_level": "balanced",
        "materiality_mode": "print",
        "asymmetry_level": "balanced",
        "negative_level": "balanced",
    },
    "strong": {
        "humanize_profile": "strong",
        "imperfection_level": "strong",
        "materiality_mode": "ink_paper",
        "asymmetry_level": "strong",
        "negative_level": "strong",
    },
    "painterly": {
        "humanize_profile": "painterly",
        "imperfection_level": "balanced",
        "materiality_mode": "canvas",
        "asymmetry_level": "balanced",
        "negative_level": "balanced",
    },
    "filmic": {
        "humanize_profile": "filmic",
        "imperfection_level": "lite",
        "materiality_mode": "film",
        "asymmetry_level": "lite",
        "negative_level": "lite",
    },
}

HUMANIZE_PROFILE_HINTS: Dict[str, str] = {
    "none": "",
    "lite": "subtle hand-drawn irregularities, natural edge variance, avoid sterile symmetry",
    "balanced": "human-made mark-making cadence, varied stroke pressure, intentional imperfection rhythm",
    "strong": "visible handcrafted quirks, non-repeating micro-variation, organic contour wobble",
    "painterly": "human brush economy, purposeful brush breaks, varied paint lay-in and edge softness",
    "filmic": "human-captured photographic feel, mild lens personality, natural scene imperfections",
}

HUMANIZE_IMPERFECTION_HINTS: Dict[str, str] = {
    "none": "",
    "lite": "slight line wobble and tiny spacing variation where appropriate",
    "balanced": "controlled imperfection in line weight, texture breakup, and shape repetition",
    "strong": "pronounced hand-made variance in line rhythm, spacing, and micro-texture",
}

HUMANIZE_MATERIALITY_HINTS: Dict[str, str] = {
    "none": "",
    "paper": "subtle paper tooth interaction, natural ink absorption feel",
    "ink_paper": "ink-on-paper behavior, dry-brush streaks, halftone print texture discipline",
    "canvas": "canvas tooth response, painterly pigment buildup, non-uniform brush drag",
    "print": "print-like halftone behavior, slight registration character, realistic reproduction feel",
    "film": "organic film grain feel, photographic texture depth, non-digital tonal rolloff",
}

HUMANIZE_ASYMMETRY_HINTS: Dict[str, str] = {
    "none": "",
    "lite": "natural facial asymmetry and non-mirrored detail placement",
    "balanced": "human asymmetry in features, posture, and repeated costume details",
    "strong": "clearly non-mirrored human asymmetry across face, pose, and accessories",
}

HUMANIZE_NEGATIVE_HINTS: Dict[str, str] = {
    "none": "",
    "lite": "plastic skin, over-smoothed gradients, sterile perfect symmetry",
    "balanced": "ai soup texture, uniform procedural lines, copy-paste detail repetition, uncanny perfection",
    "strong": "waxy skin, rubbery limbs, overclean vectorized edges, synthetic texture tiling, dead-eyed symmetry",
}

# Reading-order hints for Western vs manga (prompt-only; model follows data)
READING_ORDER_HINT = {
    "manga": "right-to-left reading order cues, manga page layout",
    "comic": "left-to-right comic layout, western gutters",
    "novel_cover": "",
    "storyboard": "sequential storyboard frames, numbered panels implied",
}

# Vertical Japanese lettering (tategaki) — use when your dataset includes JP
TATEGAKI_HINT = (
    "vertical japanese text in speech bubble, tategaki, correct stroke order impression, "
    "legible jp characters"
)

SFX_ONOMATOPOEIA_HINT = (
    "impact sfx typography, hand-drawn sound effects, integrated with art not overlay subtitle"
)

# Optional polish for print / cover work (use with models trained on book art)
PRINT_FINISH_HINT = (
    "print-ready line weight, crisp halftone, no banding, clean margins, professional reproduction"
)

COVER_SPOTLIGHT_HINT = (
    "strong focal point on hero figure, title area reserved, balanced negative space for typography"
)

# Panel / grid hints (sequential art — model follows training; these are soft cues)
PANEL_LAYOUT_HINTS: Dict[str, str] = {
    "none": "",
    "single": "single full-bleed panel, one clear focal composition",
    "two_panel_horizontal": "two horizontal panels stacked, clear gutter between tiers",
    "two_panel_vertical": "two vertical panels side by side, western comic gutters",
    "three_panel_strip": "three panel horizontal strip, equal rhythm, readable flow",
    "four_koma": "four panel vertical strip, yonkoma beat timing, punchline bottom panel",
    "splash": "large splash panel with inset smaller panel, dynamic hierarchy",
    "grid_2x2": "four equal panels in 2x2 grid, consistent line weight across cells",
}

# ---------------------------------------------------------------------------
# Aspect presets (width x height) — suggestions; 0 means “model native” in generate_book
# ---------------------------------------------------------------------------

# Webtoon / scroll: tall canvas; many models trained near 512–768 short side
ASPECT_PRESETS: Dict[str, Tuple[int, int]] = {
    "none": (0, 0),
    "square": (512, 512),
    "print_manga": (768, 1024),  # portrait page-ish
    "webtoon_tall": (512, 1536),  # vertical strip
    "widescreen_panel": (1024, 512),
    "cover_hd": (1024, 1024),
    "double_page_spread": (1536, 1024),
    "print_us_comic": (900, 1400),
}


def style_snippet(name: str) -> str:
    return STYLE_SNIPPETS.get((name or "none").lower().strip(), "")


def reading_order_for_book_type(book_type: str) -> str:
    return READING_ORDER_HINT.get((book_type or "manga").lower().strip(), "")


def merge_prompt_fragments(*parts: str, joiner: str = ", ") -> str:
    """Join non-empty stripped fragments."""
    out = [p.strip() for p in parts if p and str(p).strip()]
    return joiner.join(out)


def enhance_book_prefix(
    base_prefix: str,
    *,
    lexicon_style: str = "none",
    book_type: str = "manga",
    include_tategaki_hint: bool = False,
    include_sfx_hint: bool = False,
    include_print_finish: bool = False,
    include_cover_spotlight: bool = False,
) -> str:
    """
    Merge the existing book-type prefix with optional lexicon style + reading-order hints.
    """
    bits = [base_prefix.strip()]
    sn = style_snippet(lexicon_style)
    if sn:
        bits.append(sn)
    ro = reading_order_for_book_type(book_type)
    if ro:
        bits.append(ro)
    if include_tategaki_hint:
        bits.append(TATEGAKI_HINT)
    if include_sfx_hint:
        bits.append(SFX_ONOMATOPOEIA_HINT)
    if include_print_finish:
        bits.append(PRINT_FINISH_HINT)
    if include_cover_spotlight:
        bits.append(COVER_SPOTLIGHT_HINT)
    return merge_prompt_fragments(*bits)


def suggest_negative_addon(
    *,
    use_lexicon_negative: bool = True,
    user_negative: str = "",
    production_tier: bool = False,
    artist_lettering_strict: bool = False,
) -> str:
    """Combine user negative with lexicon anti-artifact clauses (dedupe loosely by concat)."""
    u = (user_negative or "").strip()
    if not use_lexicon_negative:
        return u
    extra = combined_comic_negative()
    if production_tier:
        extra = f"{extra}, {PRODUCTION_TIER_NEGATIVE_ADDON}"
    if artist_lettering_strict:
        extra = f"{extra}, {ARTIST_LETTERING_STRICT_NEGATIVE}"
    if not u:
        return extra
    return f"{u}, {extra}"


def aspect_dimensions(preset_name: str) -> Tuple[int, int]:
    key = (preset_name or "none").lower().strip()
    return ASPECT_PRESETS.get(key, (0, 0))


def panel_layout_hint(name: str) -> str:
    return PANEL_LAYOUT_HINTS.get((name or "none").lower().strip(), "")


def artist_craft_bundle(
    *,
    craft_profile: str = "none",
    shot_language: str = "none",
    pacing_plan: str = "none",
    lettering_craft: str = "none",
    value_plan: str = "none",
    screentone_plan: str = "none",
) -> str:
    """
    Merge practical artist-facing craft hints for sequential art quality.
    """
    bits = [
        ARTIST_CRAFT_PROFILES.get((craft_profile or "none").lower().strip(), ""),
        SHOT_LANGUAGE_HINTS.get((shot_language or "none").lower().strip(), ""),
        PACING_PLAN_HINTS.get((pacing_plan or "none").lower().strip(), ""),
        LETTERING_CRAFT_HINTS.get((lettering_craft or "none").lower().strip(), ""),
        VALUE_PLAN_HINTS.get((value_plan or "none").lower().strip(), ""),
        SCREENTONE_PLAN_HINTS.get((screentone_plan or "none").lower().strip(), ""),
    ]
    return merge_prompt_fragments(*bits)


def _lookup_medium_hint(family: str, variant: str) -> str:
    fam = (family or "none").lower().strip()
    var = (variant or "none").lower().strip()
    if fam in ART_MEDIUM_VARIANTS and var in ART_MEDIUM_VARIANTS[fam]:
        return ART_MEDIUM_VARIANTS[fam][var]
    if var != "none":
        for fam_map in ART_MEDIUM_VARIANTS.values():
            if var in fam_map:
                return fam_map[var]
    return ""


def resolve_art_medium_controls(
    *,
    art_medium_pack: str = "none",
    art_medium_family: str = "none",
    art_medium_variant: str = "none",
    art_medium_extra: str = "",
) -> Dict[str, str]:
    """
    Resolve broad art-medium controls from one pack + explicit overrides.
    """
    pack = ART_MEDIUM_PACK_PRESETS.get((art_medium_pack or "none").lower().strip(), {})
    out = {
        "family": str(pack.get("family", "none")),
        "variant": str(pack.get("variant", "none")),
        "extra": "",
    }
    if (art_medium_family or "none").lower().strip() != "none":
        out["family"] = str(art_medium_family).strip()
    if (art_medium_variant or "none").lower().strip() != "none":
        out["variant"] = str(art_medium_variant).strip()
    if str(art_medium_extra).strip():
        out["extra"] = str(art_medium_extra).strip()
    return out


def art_medium_bundle(
    *,
    family: str = "none",
    variant: str = "none",
    extra: str = "",
) -> str:
    """
    Build prompt fragments for broad medium families + concrete variants.
    """
    bits: List[str] = []
    hint = _lookup_medium_hint(family, variant)
    if hint:
        bits.append(hint)
    if str(extra).strip():
        bits.append(str(extra).strip())
    return merge_prompt_fragments(*bits)


def original_character_bundle(
    *,
    name: str = "",
    archetype: str = "none",
    visual_traits: str = "",
    wardrobe: str = "",
    silhouette: str = "",
    color_motifs: str = "",
    expression_sheet: str = "",
) -> str:
    """
    Build an artist-facing original-character (OC) consistency block.
    """
    bits = []
    if str(name).strip():
        bits.append(f"original character {str(name).strip()}, consistent identity across panels")
    arch = ORIGINAL_CHARACTER_ARCHETYPES.get((archetype or "none").lower().strip(), "")
    if arch:
        bits.append(arch)
    if str(visual_traits).strip():
        bits.append(f"signature traits: {str(visual_traits).strip()}")
    if str(wardrobe).strip():
        bits.append(f"consistent wardrobe: {str(wardrobe).strip()}")
    if str(silhouette).strip():
        bits.append(f"silhouette lock: {str(silhouette).strip()}")
    if str(color_motifs).strip():
        bits.append(f"color motif: {str(color_motifs).strip()}")
    if str(expression_sheet).strip():
        bits.append(f"expression sheet anchors: {str(expression_sheet).strip()}")
    bits.append("same face structure and hairstyle in every panel")
    return merge_prompt_fragments(*bits)


def humanize_prompt_bundle(
    *,
    humanize_profile: str = "none",
    imperfection_level: str = "none",
    materiality_mode: str = "none",
    asymmetry_level: str = "none",
) -> str:
    """Build positive prompt cues for a more human-made result."""
    bits = [
        HUMANIZE_PROFILE_HINTS.get((humanize_profile or "none").lower().strip(), ""),
        HUMANIZE_IMPERFECTION_HINTS.get((imperfection_level or "none").lower().strip(), ""),
        HUMANIZE_MATERIALITY_HINTS.get((materiality_mode or "none").lower().strip(), ""),
        HUMANIZE_ASYMMETRY_HINTS.get((asymmetry_level or "none").lower().strip(), ""),
    ]
    return merge_prompt_fragments(*bits)


def humanize_negative_addon(negative_level: str = "none") -> str:
    """Negative prompt addon to suppress common synthetic artifacts."""
    return HUMANIZE_NEGATIVE_HINTS.get((negative_level or "none").lower().strip(), "")


def resolve_artist_controls(
    *,
    artist_pack: str = "none",
    craft_profile: str = "none",
    shot_language: str = "none",
    pacing_plan: str = "none",
    lettering_craft: str = "none",
    value_plan: str = "none",
    screentone_plan: str = "none",
) -> Dict[str, str]:
    """
    Resolve artist controls from a preset pack + explicit CLI overrides.

    Explicit non-``none`` values always win over pack defaults.
    """
    pack = ARTIST_PACK_PRESETS.get((artist_pack or "none").lower().strip(), {})
    out = {
        "craft_profile": pack.get("craft_profile", "none"),
        "shot_language": pack.get("shot_language", "none"),
        "pacing_plan": pack.get("pacing_plan", "none"),
        "lettering_craft": pack.get("lettering_craft", "none"),
        "value_plan": pack.get("value_plan", "none"),
        "screentone_plan": pack.get("screentone_plan", "none"),
    }
    if (craft_profile or "none").lower().strip() != "none":
        out["craft_profile"] = craft_profile
    if (shot_language or "none").lower().strip() != "none":
        out["shot_language"] = shot_language
    if (pacing_plan or "none").lower().strip() != "none":
        out["pacing_plan"] = pacing_plan
    if (lettering_craft or "none").lower().strip() != "none":
        out["lettering_craft"] = lettering_craft
    if (value_plan or "none").lower().strip() != "none":
        out["value_plan"] = value_plan
    if (screentone_plan or "none").lower().strip() != "none":
        out["screentone_plan"] = screentone_plan
    return out


def resolve_oc_controls(
    *,
    oc_pack: str = "none",
    name: str = "",
    archetype: str = "none",
    visual_traits: str = "",
    wardrobe: str = "",
    silhouette: str = "",
    color_motifs: str = "",
    expression_sheet: str = "",
) -> Dict[str, str]:
    """
    Resolve OC controls from a preset pack + explicit CLI overrides.
    """
    pack = OC_PACK_PRESETS.get((oc_pack or "none").lower().strip(), {})
    out = {
        "name": "",
        "archetype": pack.get("archetype", "none"),
        "visual_traits": pack.get("visual_traits", ""),
        "wardrobe": pack.get("wardrobe", ""),
        "silhouette": pack.get("silhouette", ""),
        "color_motifs": pack.get("color_motifs", ""),
        "expression_sheet": pack.get("expression_sheet", ""),
    }
    if str(name).strip():
        out["name"] = str(name).strip()
    if (archetype or "none").lower().strip() != "none":
        out["archetype"] = archetype
    if str(visual_traits).strip():
        out["visual_traits"] = str(visual_traits).strip()
    if str(wardrobe).strip():
        out["wardrobe"] = str(wardrobe).strip()
    if str(silhouette).strip():
        out["silhouette"] = str(silhouette).strip()
    if str(color_motifs).strip():
        out["color_motifs"] = str(color_motifs).strip()
    if str(expression_sheet).strip():
        out["expression_sheet"] = str(expression_sheet).strip()
    return out


def resolve_book_style_controls(
    *,
    book_style_pack: str = "none",
    artist_pack: str = "none",
    oc_pack: str = "none",
    safety_mode: str = "",
    nsfw_pack: str = "",
    nsfw_civitai_pack: str = "",
    civitai_trigger_bank: str = "",
) -> Dict[str, str]:
    """
    Resolve higher-level style controls from one pack + explicit overrides.

    Explicit values win over pack defaults. For `artist_pack` and `oc_pack`,
    `"none"` is treated as unset unless no pack default exists.
    """
    pack = BOOK_STYLE_PACK_PRESETS.get((book_style_pack or "none").lower().strip(), {})
    out = {
        "artist_pack": pack.get("artist_pack", "none"),
        "oc_pack": pack.get("oc_pack", "none"),
        "safety_mode": pack.get("safety_mode", ""),
        "nsfw_pack": pack.get("nsfw_pack", ""),
        "nsfw_civitai_pack": pack.get("nsfw_civitai_pack", ""),
        "civitai_trigger_bank": pack.get("civitai_trigger_bank", ""),
    }
    if (artist_pack or "none").lower().strip() != "none":
        out["artist_pack"] = artist_pack
    if (oc_pack or "none").lower().strip() != "none":
        out["oc_pack"] = oc_pack
    if str(safety_mode).strip():
        out["safety_mode"] = str(safety_mode).strip()
    if str(nsfw_pack).strip():
        out["nsfw_pack"] = str(nsfw_pack).strip()
    if str(nsfw_civitai_pack).strip():
        out["nsfw_civitai_pack"] = str(nsfw_civitai_pack).strip()
    if str(civitai_trigger_bank).strip():
        out["civitai_trigger_bank"] = str(civitai_trigger_bank).strip()
    return out


def resolve_humanize_controls(
    *,
    humanize_pack: str = "none",
    humanize_profile: str = "none",
    imperfection_level: str = "none",
    materiality_mode: str = "none",
    asymmetry_level: str = "none",
    negative_level: str = "none",
) -> Dict[str, str]:
    """
    Resolve humanization controls from one pack + explicit overrides.
    """
    pack = HUMANIZE_PACK_PRESETS.get((humanize_pack or "none").lower().strip(), {})
    out = {
        "humanize_profile": pack.get("humanize_profile", "none"),
        "imperfection_level": pack.get("imperfection_level", "none"),
        "materiality_mode": pack.get("materiality_mode", "none"),
        "asymmetry_level": pack.get("asymmetry_level", "none"),
        "negative_level": pack.get("negative_level", "none"),
    }
    if (humanize_profile or "none").lower().strip() != "none":
        out["humanize_profile"] = humanize_profile
    if (imperfection_level or "none").lower().strip() != "none":
        out["imperfection_level"] = imperfection_level
    if (materiality_mode or "none").lower().strip() != "none":
        out["materiality_mode"] = materiality_mode
    if (asymmetry_level or "none").lower().strip() != "none":
        out["asymmetry_level"] = asymmetry_level
    if (negative_level or "none").lower().strip() != "none":
        out["negative_level"] = negative_level
    return out


def infer_auto_humanize_controls(
    *,
    book_type: str = "manga",
    lexicon_style: str = "none",
    safety_mode: str = "",
) -> Dict[str, str]:
    """
    Infer practical default humanization settings from high-level intent.
    """
    bt = (book_type or "manga").lower().strip()
    ls = (lexicon_style or "none").lower().strip()
    sm = (safety_mode or "").lower().strip()

    if sm == "nsfw":
        return resolve_humanize_controls(humanize_pack="balanced")
    if bt == "storyboard":
        return resolve_humanize_controls(humanize_pack="lite")
    if bt == "novel_cover":
        return resolve_humanize_controls(humanize_pack="painterly")
    if ls in {"graphic_novel", "seinen", "editorial"}:
        return resolve_humanize_controls(humanize_pack="painterly")
    if ls in {"webtoon", "manhwa_color"}:
        return resolve_humanize_controls(humanize_pack="filmic")
    return resolve_humanize_controls(humanize_pack="balanced")
