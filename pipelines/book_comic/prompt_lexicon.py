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
    # 3d anime game art styles
    "anime_game_toon_pbr": "anime-game toon-PBR hybrid, readable stylized faces, clean anisotropic highlights, controlled soft GI bounce",
    "genshin_like_3d_anime": "genshin-like 3d anime look, painterly texture polish, clean stylized materials, bright adventure color scripting",
    "honkai_starrail_3d_anime": "honkai star rail inspired 3d anime style, premium key-art lighting, refined costume material separation, cinematic pose polish",
    "zenless_zone_urban_anime": "urban stylized 3d anime aesthetic, punchy graphic contrast, modern fashion silhouette language, energetic action framing",
    "persona_ui_stylized": "persona-inspired stylized rendering, graphic UI-aware composition zones, bold color blocks, high-contrast mood language",
    "arcane_painterly_3d": "arcane-inspired painterly 3d treatment, hand-painted surface nuance, dramatic cinematic grade, expressive facial readability",
    "guiltygear_hybrid": "guilty gear inspired anime-3d hybrid, aggressive shape readability, frame-by-frame pose impact, stylized line-like shadow logic",
    "anime_cg_cutscene": "anime cg cutscene style, cinematic camera continuity, polished facial rig readability, coherent post-process bloom restraint",
    # Popular digital / 3d / drawing / painting additions
    "digital_fantasy_splash": "digital fantasy splash-art style, dramatic hero framing, layered atmosphere depth, premium rim-light accents",
    "digital_semi_real_portrait": "semi-real digital portrait style, controlled skin material transitions, painterly edge variety, studio-aware lighting",
    "digital_mobile_game_iconic": "mobile-game illustration style, iconic silhouette readability, bright controlled palette, marketable character polish",
    "pbr_cinematic_keyart": "cinematic PBR key-art style, disciplined roughness-metal response, volumetric light depth, premium hero composition",
    "stylized_3d_overwatch_like": "stylized 3d hero-shooter style, clean shape language, readable material grouping, gameplay-first clarity",
    "unreal_realtime_cinematic": "unreal-engine real-time cinematic style, physically grounded light transport, temporal-stable shading, filmic post workflow",
    "lineart_character_sheet": "character-sheet lineart style, construction clarity, clean contour hierarchy, design-callout readability",
    "ink_crosshatch_noir": "ink crosshatch noir style, high-contrast hatch rhythm, decisive shadow masses, dramatic graphic storytelling",
    "oil_portrait_classical": "classical oil portrait style, warm-cool flesh modeling, layered glaze depth, controlled highlight placement",
    "gouache_poster_graphic": "gouache poster style, matte opaque color blocks, bold shape simplification, print-strong composition hierarchy",
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
        "splash_art": "digital splash-art medium, cinematic focal framing, controlled FX layering, premium finish rendering hierarchy",
        "portrait_render": "digital portrait-render medium, skin/value discipline, subtle material transitions, controlled edge hierarchy",
        "mobile_game_illustration": "mobile-game digital illustration medium, iconic silhouette-first design, bright market-friendly palette control",
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
        "lineart_sheet": "lineart character-sheet medium, orthographic design readability, contour consistency, callout-ready clean drafting",
        "ink_noir": "ink noir medium, dense shadow-mass design, high-contrast storytelling, deliberate crosshatch cadence",
        "storyboard_rough": "storyboard rough medium, shot-beat clarity, fast perspective shorthand, camera-motion readability",
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
        "unreal_cinematic": "unreal-cinematic 3d medium, real-time filmic lighting discipline, temporally stable shading, production shot continuity",
        "anime_game_3d": "anime-game 3d medium, toon-PBR hybrid coherence, stylized facial stability, premium character key-art finish",
        "hero_stylized_game": "hero-stylized game 3d medium, gameplay readability-first silhouettes, simplified material grammar, expressive pose polish",
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
        "oil_portrait": "oil portrait medium, flesh-tone temperature orchestration, glazing depth, controlled edge transitions",
        "gouache_poster": "gouache poster medium, matte opacity discipline, graphic massing, high-impact print readability",
        "watercolor_botanical": "watercolor botanical medium, transparent layered detail, delicate edge handling, natural paper bloom behavior",
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
    "digital_splash_master": {"family": "digital_art", "variant": "splash_art"},
    "digital_portrait_master": {"family": "digital_art", "variant": "portrait_render"},
    "mobile_game_illustration_pro": {"family": "digital_art", "variant": "mobile_game_illustration"},
    "unreal_cinematic_3d": {"family": "digital_3d_art", "variant": "unreal_cinematic"},
    "anime_game_3d_pro": {"family": "digital_3d_art", "variant": "anime_game_3d"},
    "hero_stylized_3d_pro": {"family": "digital_3d_art", "variant": "hero_stylized_game"},
    "lineart_sheet_pro": {"family": "drawing_art", "variant": "lineart_sheet"},
    "ink_noir_pro": {"family": "drawing_art", "variant": "ink_noir"},
    "storyboard_rough_pro": {"family": "drawing_art", "variant": "storyboard_rough"},
    "oil_portrait_master": {"family": "painting_art", "variant": "oil_portrait"},
    "gouache_poster_master": {"family": "painting_art", "variant": "gouache_poster"},
    "watercolor_botanical_master": {"family": "painting_art", "variant": "watercolor_botanical"},
}

POPULAR_STYLE_PRESETS: List[Dict[str, str]] = [
    # Digital (1-15)
    {
        "id": "digital_fantasy_splash",
        "category": "digital",
        "lexicon_style": "digital_fantasy_splash",
        "art_medium_pack": "digital_splash_master",
        "color_render_pack": "painting_value_master",
        "tags": "digital fantasy splash key art epic character",
        "description": "High-impact digital fantasy splash composition.",
    },
    {
        "id": "digital_semi_real_portrait",
        "category": "digital",
        "lexicon_style": "digital_semi_real_portrait",
        "art_medium_pack": "digital_portrait_master",
        "color_render_pack": "painting_value_master",
        "tags": "digital portrait semi-real character face beauty",
        "description": "Semi-real portrait rendering for polished character art.",
    },
    {
        "id": "digital_mobile_game_iconic",
        "category": "digital",
        "lexicon_style": "digital_mobile_game_iconic",
        "art_medium_pack": "mobile_game_illustration_pro",
        "color_render_pack": "anime_cel_master",
        "tags": "mobile game icon illustration bright marketable",
        "description": "Iconic mobile-game style with readable silhouette language.",
    },
    {
        "id": "digital_manhwa_webtoon",
        "category": "digital",
        "lexicon_style": "manhwa_color",
        "art_medium_pack": "webcomic_mobile",
        "color_render_pack": "anime_cel_master",
        "tags": "manhwa webtoon scrolling romance clean color",
        "description": "Modern manhwa/webtoon polish for episodic pages.",
    },
    {
        "id": "digital_visual_dev_story",
        "category": "digital",
        "lexicon_style": "digital_art",
        "art_medium_pack": "visual_dev_story",
        "color_render_pack": "painting_value_master",
        "tags": "visual development story color key concept",
        "description": "Visual development framing for narrative scenes.",
    },
    {
        "id": "digital_concept_sheet",
        "category": "digital",
        "lexicon_style": "digital_art",
        "art_medium_pack": "concept_sheet_design",
        "color_render_pack": "anime_cel_master",
        "tags": "concept sheet turnarounds orthographic design",
        "description": "Production-friendly concept-sheet design preset.",
    },
    {
        "id": "digital_anime_keyart",
        "category": "digital",
        "lexicon_style": "anime_movie_keyart",
        "art_medium_pack": "anime_movie_keyart",
        "color_render_pack": "anime_cel_master",
        "tags": "anime keyart movie poster polished",
        "description": "High-finish anime key art with cinematic framing.",
    },
    {
        "id": "digital_superhero_modern",
        "category": "digital",
        "lexicon_style": "superhero_action_modern",
        "art_medium_pack": "superhero_action_modern",
        "color_render_pack": "comic_noir_master",
        "tags": "superhero comic modern action",
        "description": "Contemporary superhero action with graphic impact.",
    },
    {
        "id": "digital_cyberpunk_noir",
        "category": "digital",
        "lexicon_style": "cyberpunk",
        "art_medium_pack": "cyberpunk_noir_panel",
        "color_render_pack": "hybrid_2d3d_master",
        "tags": "cyberpunk neon noir city rain",
        "description": "Noir cyberpunk atmosphere with focused contrast.",
    },
    {
        "id": "digital_poster_graphic",
        "category": "digital",
        "lexicon_style": "poster_graphic",
        "art_medium_pack": "gouache_poster_master",
        "color_render_pack": "comic_noir_master",
        "tags": "poster graphic typography-safe composition",
        "description": "Graphic-poster look with strong visual hierarchy.",
    },
    {
        "id": "digital_retro_pulp",
        "category": "digital",
        "lexicon_style": "retro_pulp_cover",
        "art_medium_pack": "gouache_poster_master",
        "color_render_pack": "painting_value_master",
        "tags": "retro pulp cover vintage",
        "description": "Vintage pulp-cover direction for bold narrative art.",
    },
    {
        "id": "digital_arcane_painterly",
        "category": "digital",
        "lexicon_style": "arcane_painterly_3d",
        "artist_pack": "anime3d_arcane_painterly",
        "color_render_pack": "hybrid_2d3d_master",
        "tags": "arcane painterly 3d cinematic",
        "description": "Painterly-composited dramatic rendering style.",
    },
    {
        "id": "digital_shinkai_cinematic",
        "category": "digital",
        "lexicon_style": "anime_2d",
        "artist_pack": "anime_shinkai_cinematic",
        "color_render_pack": "anime_cel_master",
        "tags": "shinkai anime sky lighting cinematic",
        "description": "Luminous anime-cinematic environmental storytelling.",
    },
    {
        "id": "digital_ghibli_story",
        "category": "digital",
        "lexicon_style": "anime_2d",
        "artist_pack": "anime_ghibli_story",
        "color_render_pack": "painting_value_master",
        "tags": "ghibli storybook warm whimsical",
        "description": "Whimsical storybook direction with warm readability.",
    },
    {
        "id": "digital_trigger_action",
        "category": "digital",
        "lexicon_style": "anime_shonen_battle",
        "artist_pack": "anime_trigger_action",
        "color_render_pack": "anime_cel_master",
        "tags": "trigger dynamic action anime",
        "description": "High-energy dynamic anime action framing.",
    },
    # 3D (16-30)
    {
        "id": "3d_pbr_cinematic_keyart",
        "category": "3d",
        "lexicon_style": "pbr_cinematic_keyart",
        "art_medium_pack": "unreal_cinematic_3d",
        "color_render_pack": "pbr_3d_master",
        "tags": "3d pbr cinematic key art unreal",
        "description": "Cinematic PBR hero shot with filmic depth.",
    },
    {
        "id": "3d_unreal_realtime",
        "category": "3d",
        "lexicon_style": "unreal_realtime_cinematic",
        "art_medium_pack": "unreal_cinematic_3d",
        "color_render_pack": "pbr_3d_master",
        "tags": "unreal realtime cinematic render",
        "description": "Realtime cinematic scene language for 3D renders.",
    },
    {
        "id": "3d_stylized_hero_game",
        "category": "3d",
        "lexicon_style": "stylized_3d_overwatch_like",
        "art_medium_pack": "hero_stylized_3d_pro",
        "artist_pack": "game_blizzard_heroic",
        "color_render_pack": "toon_3d_master",
        "tags": "stylized 3d hero shooter game",
        "description": "Stylized hero game aesthetic with gameplay readability.",
    },
    {
        "id": "3d_anime_game_general",
        "category": "3d",
        "lexicon_style": "anime_game_toon_pbr",
        "art_medium_pack": "anime_game_3d_pro",
        "color_render_pack": "hybrid_2d3d_master",
        "tags": "anime game 3d toon pbr hybrid",
        "description": "General anime-game 3D hybrid rendering preset.",
    },
    {
        "id": "3d_genshin_like",
        "category": "3d",
        "lexicon_style": "genshin_like_3d_anime",
        "artist_pack": "anime3d_genshin_keyart",
        "color_render_pack": "hybrid_2d3d_master",
        "tags": "genshin anime game 3d",
        "description": "Bright adventure anime-game rendering style.",
    },
    {
        "id": "3d_honkai_like",
        "category": "3d",
        "lexicon_style": "honkai_starrail_3d_anime",
        "artist_pack": "anime3d_hsr_cinematic",
        "color_render_pack": "hybrid_2d3d_master",
        "tags": "honkai star rail anime 3d",
        "description": "Premium anime-game cinematic key-art direction.",
    },
    {
        "id": "3d_zenless_like",
        "category": "3d",
        "lexicon_style": "zenless_zone_urban_anime",
        "artist_pack": "anime3d_zzz_urban",
        "color_render_pack": "hybrid_2d3d_master",
        "tags": "zenless urban anime 3d",
        "description": "Urban stylized anime-3D action composition.",
    },
    {
        "id": "3d_persona_ui_style",
        "category": "3d",
        "lexicon_style": "persona_ui_stylized",
        "art_medium_pack": "hero_stylized_3d_pro",
        "color_render_pack": "toon_3d_master",
        "tags": "persona ui stylized graphic",
        "description": "Graphic UI-driven stylized render language.",
    },
    {
        "id": "3d_guiltygear_hybrid",
        "category": "3d",
        "lexicon_style": "guiltygear_hybrid",
        "art_medium_pack": "anime_game_3d_pro",
        "color_render_pack": "toon_3d_master",
        "tags": "guilty gear hybrid toon",
        "description": "Hard-hitting anime/3D hybrid action styling.",
    },
    {
        "id": "3d_anime_cutscene",
        "category": "3d",
        "lexicon_style": "anime_cg_cutscene",
        "art_medium_pack": "anime_game_3d_pro",
        "color_render_pack": "hybrid_2d3d_master",
        "tags": "anime cg cutscene cinematic",
        "description": "Anime CG cutscene composition and camera continuity.",
    },
    {
        "id": "3d_archviz_real",
        "category": "3d",
        "lexicon_style": "archviz_real",
        "art_medium_pack": "archviz_cinematic",
        "color_render_pack": "pbr_3d_master",
        "tags": "archviz interior exterior realistic 3d",
        "description": "Architectural visualization realism direction.",
    },
    {
        "id": "3d_product_cg",
        "category": "3d",
        "lexicon_style": "product_cg",
        "art_medium_pack": "product_cg_studio",
        "color_render_pack": "pbr_3d_master",
        "tags": "product cgi studio render",
        "description": "Commercial product-CG hero shot style.",
    },
    {
        "id": "3d_octane_cinematic",
        "category": "3d",
        "lexicon_style": "render_octane",
        "art_medium_pack": "octane_3d_cinematic",
        "color_render_pack": "pbr_3d_master",
        "tags": "octane cinematic 3d render",
        "description": "Cinematic high-fidelity render profile.",
    },
    {
        "id": "3d_eevee_stylized",
        "category": "3d",
        "lexicon_style": "render_eevee",
        "art_medium_pack": "hero_stylized_3d_pro",
        "color_render_pack": "toon_3d_master",
        "tags": "eevee stylized realtime 3d",
        "description": "Stylized real-time render tuned for clarity.",
    },
    {
        "id": "3d_isometric_voxel",
        "category": "3d",
        "lexicon_style": "voxel_isometric",
        "art_medium_pack": "isometric_voxel_world",
        "color_render_pack": "toon_3d_master",
        "tags": "voxel isometric world building",
        "description": "Readable isometric voxel world style.",
    },
    # Drawing (31-40)
    {
        "id": "drawing_lineart_sheet",
        "category": "drawing",
        "lexicon_style": "lineart_character_sheet",
        "art_medium_pack": "lineart_sheet_pro",
        "color_render_pack": "anime_cel_master",
        "tags": "lineart character sheet design",
        "description": "Production-oriented lineart character-sheet workflow.",
    },
    {
        "id": "drawing_ink_noir",
        "category": "drawing",
        "lexicon_style": "ink_crosshatch_noir",
        "art_medium_pack": "ink_noir_pro",
        "color_render_pack": "comic_noir_master",
        "tags": "ink crosshatch noir high contrast",
        "description": "Noir crosshatch inking with strong value structure.",
    },
    {
        "id": "drawing_storyboard_rough",
        "category": "drawing",
        "lexicon_style": "manga_horror",
        "art_medium_pack": "storyboard_rough_pro",
        "color_render_pack": "comic_noir_master",
        "tags": "storyboard rough shot planning",
        "description": "Fast storyboard roughs for camera-flow ideation.",
    },
    {
        "id": "drawing_crosshatch_master",
        "category": "drawing",
        "lexicon_style": "drawing_ink",
        "art_medium_pack": "crosshatch_ink_master",
        "color_render_pack": "comic_noir_master",
        "tags": "crosshatch ink linework",
        "description": "Crosshatch-heavy ink discipline preset.",
    },
    {
        "id": "drawing_comic_pencil",
        "category": "drawing",
        "lexicon_style": "newspaper_comic",
        "art_medium_pack": "comic_pencil_storyboard",
        "color_render_pack": "anime_cel_master",
        "tags": "comic pencil storyboard",
        "description": "Comic-pencil pre-ink workflow look.",
    },
    {
        "id": "drawing_manga_bw",
        "category": "drawing",
        "lexicon_style": "seinen",
        "art_medium_pack": "drawing_ink_pro",
        "color_render_pack": "comic_noir_master",
        "tags": "manga black and white screentone",
        "description": "Manga-oriented monochrome inking baseline.",
    },
    {
        "id": "drawing_blueprint_technical",
        "category": "drawing",
        "lexicon_style": "sci_fi_concept",
        "art_medium_family": "drawing_art",
        "art_medium_variant": "blueprint",
        "color_render_pack": "anime_cel_master",
        "tags": "blueprint technical drawing",
        "description": "Technical blueprint drafting language.",
    },
    {
        "id": "drawing_life_study",
        "category": "drawing",
        "lexicon_style": "drawing_ink",
        "art_medium_family": "drawing_art",
        "art_medium_variant": "life_study",
        "color_render_pack": "anime_cel_master",
        "tags": "life study anatomy graphite",
        "description": "Observational life-study sketch discipline.",
    },
    {
        "id": "drawing_gesture_dynamic",
        "category": "drawing",
        "lexicon_style": "anime_shonen_battle",
        "art_medium_family": "drawing_art",
        "art_medium_variant": "gesture",
        "color_render_pack": "anime_cel_master",
        "tags": "gesture dynamic movement",
        "description": "Gesture-driven dynamic action sketches.",
    },
    {
        "id": "drawing_pen_ink_clean",
        "category": "drawing",
        "lexicon_style": "drawing_ink",
        "art_medium_family": "drawing_art",
        "art_medium_variant": "pen_ink",
        "color_render_pack": "comic_noir_master",
        "tags": "pen and ink clean line",
        "description": "Classic pen-and-ink contour clarity style.",
    },
    # Painting (41-50)
    {
        "id": "painting_oil_portrait",
        "category": "painting",
        "lexicon_style": "oil_portrait_classical",
        "art_medium_pack": "oil_portrait_master",
        "color_render_pack": "painting_value_master",
        "tags": "oil portrait classical skin tones",
        "description": "Classical oil portraiture with rich value modeling.",
    },
    {
        "id": "painting_gouache_poster",
        "category": "painting",
        "lexicon_style": "gouache_poster_graphic",
        "art_medium_pack": "gouache_poster_master",
        "color_render_pack": "painting_value_master",
        "tags": "gouache poster graphic flat",
        "description": "Graphic gouache poster style with strong silhouette.",
    },
    {
        "id": "painting_watercolor_botanical",
        "category": "painting",
        "lexicon_style": "painting_watercolor",
        "art_medium_pack": "watercolor_botanical_master",
        "color_render_pack": "painting_value_master",
        "tags": "watercolor botanical detailed nature",
        "description": "Botanical watercolor detail and transparent layering.",
    },
    {
        "id": "painting_oil_classic",
        "category": "painting",
        "lexicon_style": "painting_oil",
        "art_medium_pack": "oil_painting_classic",
        "color_render_pack": "painting_value_master",
        "tags": "oil painting classical brushwork",
        "description": "Classic oil treatment for painterly fidelity.",
    },
    {
        "id": "painting_watercolor_storybook",
        "category": "painting",
        "lexicon_style": "painting_watercolor",
        "art_medium_pack": "watercolor_storybook",
        "color_render_pack": "painting_value_master",
        "tags": "watercolor storybook children illustration",
        "description": "Storybook watercolor with warm narrative readability.",
    },
    {
        "id": "painting_surreal",
        "category": "painting",
        "lexicon_style": "surrealist",
        "art_medium_pack": "surreal_paint_studio",
        "color_render_pack": "painting_value_master",
        "tags": "surreal painting dreamlike",
        "description": "Surreal painterly composition language.",
    },
    {
        "id": "painting_mural_graphic",
        "category": "painting",
        "lexicon_style": "maximalist",
        "art_medium_pack": "mural_graphic_large",
        "color_render_pack": "painting_value_master",
        "tags": "mural graphic large scale",
        "description": "Large-scale mural graphic paint direction.",
    },
    {
        "id": "painting_noir",
        "category": "painting",
        "lexicon_style": "noir_comic",
        "art_medium_family": "painting_art",
        "art_medium_variant": "noir_paint",
        "color_render_pack": "comic_noir_master",
        "tags": "noir paint chiaroscuro",
        "description": "Noir paint mood with high-contrast value staging.",
    },
    {
        "id": "painting_plein_air",
        "category": "painting",
        "lexicon_style": "impressionist",
        "art_medium_family": "painting_art",
        "art_medium_variant": "plein_air",
        "color_render_pack": "painting_value_master",
        "tags": "plein air landscape light",
        "description": "Outdoor natural-light plein-air paint workflow.",
    },
    {
        "id": "painting_watercolor_ink_mix",
        "category": "painting",
        "lexicon_style": "art_nouveau",
        "art_medium_family": "painting_art",
        "art_medium_variant": "watercolor_ink",
        "color_render_pack": "painting_value_master",
        "tags": "watercolor ink mixed media",
        "description": "Watercolor-plus-ink mixed painting treatment.",
    },
]

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

ARTIST_STYLE_PROFILES: Dict[str, str] = {
    "none": "",
    # Anime / manga creators
    "shinkai_cinematic_anime": (
        "Makoto Shinkai inspired anime language, luminous atmospheric skies, cinematic backlight glow, "
        "careful mood-driven color scripting"
    ),
    "miyazaki_ghibli_storybook": (
        "Hayao Miyazaki and Studio Ghibli inspired storytelling, hand-crafted environment warmth, "
        "organic shape appeal, whimsical but grounded scene readability"
    ),
    "trigger_dynamic_action": (
        "Studio Trigger inspired kinetic framing, exaggerated action silhouettes, bold shape language, "
        "high-energy color blocking"
    ),
    "clamp_fashion_linework": (
        "CLAMP inspired elegant character design, fashion-forward silhouette styling, "
        "decorative line rhythm, refined dramatic staging"
    ),
    "toriyama_clean_adventure": (
        "Akira Toriyama inspired clean contour economy, playful adventure readability, "
        "clear shape hierarchy, expressive character posing"
    ),
    "otomo_urban_mecha": (
        "Katsuhiro Otomo inspired urban-mecha detail discipline, grounded perspective, "
        "industrial design coherence, dramatic city-scale staging"
    ),
    "nomura_character_design": (
        "Tetsuya Nomura inspired character-fashion identity, iconic silhouette hooks, "
        "ornate accessory readability, high-finish key-art polish"
    ),
    "shinkawa_brush_stealth": (
        "Yoji Shinkawa inspired expressive ink-brush energy, tactical silhouette clarity, "
        "dynamic monochrome-plus-accent contrast language"
    ),
    # Game art direction / studios
    "riot_splash_fantasy": (
        "Riot splash-art inspired heroic focal hierarchy, premium VFX color accents, "
        "champion-readable silhouette design, high-impact key-art composition"
    ),
    "valorant_clean_shapes": (
        "VALORANT inspired clean tactical readability, simplified material language, "
        "crisp shape grammar, competitive legibility-first art direction"
    ),
    "blizzard_cinematic_heroic": (
        "Blizzard inspired heroic character emphasis, bold stylized realism, "
        "cinematic lighting beats, polished material separation"
    ),
    "fromsoftware_dark_fantasy": (
        "FromSoftware inspired somber world mood, ruin-scale composition, "
        "mythic silhouette staging, restrained desaturated dramatic palette"
    ),
    "zelda_painterly_adventure": (
        "Zelda-inspired adventure tone, painterly environment readability, "
        "storybook color harmony, exploration-first composition language"
    ),
    # Cartoon / animation studios
    "pixar_shape_script": (
        "Pixar-inspired shape appeal and story clarity, warm cinematic color script, "
        "emotion-first staging, family-film readability polish"
    ),
    "disney_animation_staging": (
        "Disney animation-inspired character appeal, clear pose-to-silhouette communication, "
        "musical-theatrical staging rhythm, expressive lighting continuity"
    ),
    "cartoon_network_graphic": (
        "Cartoon Network inspired graphic simplification, strong color blocks, "
        "comedic timing readability, bold stylized expression language"
    ),
    "nickelodeon_expressive": (
        "Nickelodeon-inspired exaggerated expression beats, playful shape distortion, "
        "readable gag-focused staging, high-energy cartoon rhythm"
    ),
    # Western comic / illustration artists
    "mignola_noir_graphic": (
        "Mike Mignola inspired heavy shadow design, stark graphic value structure, "
        "moody noir atmosphere, simplified but powerful silhouette storytelling"
    ),
    "alex_ross_painterly_realism": (
        "Alex Ross inspired painterly comic realism, sculpted heroic forms, "
        "controlled dramatic light modeling, iconic poster-like composition"
    ),
    "frazetta_epic_fantasy": (
        "Frank Frazetta inspired muscular fantasy dynamism, primal composition energy, "
        "earthy heroic palette, bold brush-led focal storytelling"
    ),
    "moebius_line_worldbuilding": (
        "Moebius inspired elegant line economy, imaginative worldbuilding motifs, "
        "surreal-but-readable design language, airy color atmosphere"
    ),
    # 3D anime game / booru-adjacent artist profile packs
    "niziu_anime_3d_colorist": (
        "Niziu style inspired anime-3d colorist language, clean hue separation, "
        "vibrant but controlled saturation, glossy anime-material polish"
    ),
    "reiq_anime_3d_sculpt": (
        "Reiq style inspired anime-3d sculpt readability, stable facial planes, "
        "clear primary-secondary-tertiary forms, elegant stylized topology cues"
    ),
    "swd3e2_game_anime_hybrid": (
        "swd3e2 style inspired game-anime hybrid rendering, strong silhouette legibility, "
        "controlled stylized GI accents, action-centric camera composition"
    ),
    "alpha_3d_anime_keyart": (
        "alpha style inspired 3d anime key-art language, premium lighting polish, "
        "hero-centered composition, clean material break-up and costume hierarchy"
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

LINEWORK_TECHNIQUE_HINTS: Dict[str, str] = {
    "none": "",
    "clean_contour": "clean contour discipline, consistent edge control, intentional line economy",
    "expressive_weight": "expressive line-weight modulation, pressure-driven stroke hierarchy, focal contour emphasis",
    "crosshatch_precision": "precision crosshatch line rhythm, directional stroke coherence, value-by-line density control",
    "sketch_loose": "loose exploratory sketch language, construction-aware gesture lines, controlled roughness",
    "calligraphic_ink": "calligraphic ink stroke cadence, brush-pressure flow, elegant taper variation",
}

RENDERING_TECHNIQUE_HINTS: Dict[str, str] = {
    "none": "",
    "cel_anime": "anime cel-render technique, hard-value bands, selective highlight accents, contour-first readability",
    "painterly_2d": "painterly 2d rendering, edge-varied form turns, brush economy, focal-detail restraint",
    "toon_3d": "toon 3d render technique, ramp-consistent lighting, stylized specular control, silhouette-first forms",
    "pbr_3d": "pbr 3d render technique, physically plausible BRDF response, roughness-metal consistency, GI-aware values",
    "hybrid_2d3d": "hybrid 2d-3d render compositing, unified shading language, coherent stylization bridge",
}

SHADING_TECHNIQUE_PLAN_HINTS: Dict[str, str] = {
    "none": "",
    "chiaroscuro": "chiaroscuro shadow design, dramatic light-dark grouping, narrative contrast control",
    "ambient_occlusion": "ambient-occlusion-aware contact shadowing, grounded object anchoring, subtle crevice depth",
    "rim_bounce": "rim-plus-bounce lighting technique, silhouette separation, controlled reflected-light color cues",
    "subsurface_skin": "subsurface-aware skin shading, translucency restraint, natural facial plane transitions",
    "volumetric_depth": "volumetric depth shading, atmospheric light shafts, depth-staged value falloff",
}

MATERIAL_TECHNIQUE_HINTS: Dict[str, str] = {
    "none": "",
    "fabric_folds": "fabric rendering technique, fold-tension logic, weave-aware specular softness",
    "metal_surface": "metal rendering technique, anisotropic highlight direction, clean micro-scratch cue control",
    "skin_microdetail": "skin micro-detail technique, pore-aware smoothing restraint, natural sheen breakup",
    "paint_texture": "paint texture technique, visible brush lay-in, pigment edge variation, tactile surface rhythm",
    "paper_grain": "paper-grain technique, substrate interaction cues, dry/wet media response realism",
}

COMPOSITION_TECHNIQUE_HINTS: Dict[str, str] = {
    "none": "",
    "rule_of_thirds": "rule-of-thirds composition technique, stable focal placement, balanced negative space",
    "leading_lines": "leading-lines composition technique, guided eye flow, perspective-driven narrative direction",
    "depth_layers": "foreground-midground-background layering technique, depth readability, atmospheric separation",
    "silhouette_focus": "silhouette-first composition technique, clear shape communication, clutter suppression",
    "negative_space": "negative-space composition technique, breathing-room control, high-intent visual hierarchy",
}

ARTIST_TECHNIQUE_PACK_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "digital_2d_master": {
        "linework_technique": "clean_contour",
        "rendering_technique": "painterly_2d",
        "shading_technique_plan": "rim_bounce",
        "material_technique": "paint_texture",
        "composition_technique": "rule_of_thirds",
    },
    "anime_2d_master": {
        "linework_technique": "expressive_weight",
        "rendering_technique": "cel_anime",
        "shading_technique_plan": "rim_bounce",
        "material_technique": "none",
        "composition_technique": "silhouette_focus",
    },
    "drawing_ink_master": {
        "linework_technique": "crosshatch_precision",
        "rendering_technique": "none",
        "shading_technique_plan": "chiaroscuro",
        "material_technique": "paper_grain",
        "composition_technique": "leading_lines",
    },
    "painting_master": {
        "linework_technique": "none",
        "rendering_technique": "painterly_2d",
        "shading_technique_plan": "volumetric_depth",
        "material_technique": "paint_texture",
        "composition_technique": "depth_layers",
    },
    "toon_3d_master": {
        "linework_technique": "clean_contour",
        "rendering_technique": "toon_3d",
        "shading_technique_plan": "rim_bounce",
        "material_technique": "fabric_folds",
        "composition_technique": "silhouette_focus",
    },
    "pbr_3d_master": {
        "linework_technique": "none",
        "rendering_technique": "pbr_3d",
        "shading_technique_plan": "ambient_occlusion",
        "material_technique": "metal_surface",
        "composition_technique": "depth_layers",
    },
}

COLOR_THEORY_HINTS: Dict[str, str] = {
    "none": "",
    "balanced": "60-30-10 color balance, focal color hierarchy, controlled saturation rhythm",
    "complementary": "complementary color contrast with controlled intensity and value-safe separation",
    "split_complementary": "split-complementary palette strategy, high contrast with stable harmony",
    "analogous": "analogous palette flow with subtle temperature transitions and cohesive harmony",
    "triadic": "triadic palette balance with clear dominant-secondary-accent role assignment",
    "tetradic": "tetradic palette discipline with strict value grouping to prevent color chaos",
    "monochrome": "monochrome palette discipline with rich value range and subtle hue drift",
    "warm_cool": "intentional warm/cool temperature orchestration between key, fill, and shadow families",
    "gamut_print": "print-safe gamut-aware color decisions, avoid out-of-gamut clipping and muddy conversion",
}

GRADIENT_BLEND_HINTS: Dict[str, str] = {
    "none": "",
    "clean": "clean gradient transitions, anti-banding ramps, controlled blend-mode stacking",
    "painterly": "painterly gradient transitions, edge-aware blending, visible brush-transition character",
    "atmospheric": "atmospheric depth gradients, aerial perspective color shift by distance and brightness",
    "toon_steps": "discrete toon-step gradients with deliberate threshold placement and readable shape planes",
    "volumetric": "volumetric gradient flow through fog/haze shafts with physically plausible falloff",
}

SHADING_TECHNIQUE_HINTS: Dict[str, str] = {
    "none": "",
    "cel": "cel-shading logic, hard shadow boundaries, intentional shadow-shape design",
    "soft_painterly": "soft painterly form-turn shading, edge hierarchy, controlled reflected-light accents",
    "crosshatch": "crosshatch-driven value construction with directional hatch coherence",
    "chiaroscuro": "chiaroscuro lighting structure, dramatic light-dark masses and focal contrast control",
    "pbr": "physically based shading response, coherent roughness/metalness behavior, plausible specular rolloff",
    "subsurface": "subsurface scattering where appropriate, skin/wax translucency with restrained bloom",
}

RENDER_PIPELINE_HINTS: Dict[str, str] = {
    "none": "",
    "illustration_2d": "2d illustration render discipline, value-first readability, clean silhouette hierarchy",
    "anime_2d": "anime 2d render discipline, stable line/cel grammar, controlled highlight placement",
    "toon_3d": "toon 3d render pipeline, contour readability, stylized ramp-consistent lighting",
    "pbr_3d": "pbr 3d render pipeline, global-illumination bounce color, filmic tone mapping rolloff",
    "cinematic_photo": "cinematic photo-grade pipeline, scene-referred exposure logic, restrained color grading",
    "hybrid_2d3d": "hybrid 2d/3d render integration, unified lighting language, style-consistent material treatment",
}

COLOR_RENDER_PACK_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "anime_cel_master": {
        "color_theory_mode": "triadic",
        "gradient_blend_mode": "toon_steps",
        "shading_technique": "cel",
        "render_pipeline": "anime_2d",
    },
    "painting_value_master": {
        "color_theory_mode": "warm_cool",
        "gradient_blend_mode": "painterly",
        "shading_technique": "soft_painterly",
        "render_pipeline": "illustration_2d",
    },
    "comic_noir_master": {
        "color_theory_mode": "monochrome",
        "gradient_blend_mode": "clean",
        "shading_technique": "chiaroscuro",
        "render_pipeline": "illustration_2d",
    },
    "pbr_3d_master": {
        "color_theory_mode": "split_complementary",
        "gradient_blend_mode": "volumetric",
        "shading_technique": "pbr",
        "render_pipeline": "pbr_3d",
    },
    "toon_3d_master": {
        "color_theory_mode": "analogous",
        "gradient_blend_mode": "toon_steps",
        "shading_technique": "cel",
        "render_pipeline": "toon_3d",
    },
    "photo_grade_master": {
        "color_theory_mode": "balanced",
        "gradient_blend_mode": "atmospheric",
        "shading_technique": "subsurface",
        "render_pipeline": "cinematic_photo",
    },
    "hybrid_2d3d_master": {
        "color_theory_mode": "complementary",
        "gradient_blend_mode": "clean",
        "shading_technique": "soft_painterly",
        "render_pipeline": "hybrid_2d3d",
    },
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
    # Anime / manga focused
    "anime_shinkai_cinematic": {
        "craft_profile": "manga_pro",
        "shot_language": "cinematic",
        "pacing_plan": "balanced",
        "lettering_craft": "standard",
        "value_plan": "color_script",
        "screentone_plan": "clean",
        "artist_style_profile": "shinkai_cinematic_anime",
    },
    "anime_ghibli_story": {
        "craft_profile": "children_book",
        "shot_language": "mixed",
        "pacing_plan": "decompressed",
        "lettering_craft": "standard",
        "value_plan": "color_script",
        "screentone_plan": "none",
        "artist_style_profile": "miyazaki_ghibli_storybook",
    },
    "anime_trigger_action": {
        "craft_profile": "manga_pro",
        "shot_language": "manga_dynamic",
        "pacing_plan": "compressed",
        "lettering_craft": "standard",
        "value_plan": "bw_hierarchy",
        "screentone_plan": "dramatic",
        "artist_style_profile": "trigger_dynamic_action",
    },
    "anime_nomura_character": {
        "craft_profile": "manga_pro",
        "shot_language": "cinematic",
        "pacing_plan": "balanced",
        "lettering_craft": "none",
        "value_plan": "color_script",
        "screentone_plan": "clean",
        "artist_style_profile": "nomura_character_design",
    },
    # Game art focused
    "game_riot_splash": {
        "craft_profile": "western_comic_pro",
        "shot_language": "cinematic",
        "pacing_plan": "compressed",
        "lettering_craft": "none",
        "value_plan": "color_script",
        "screentone_plan": "none",
        "artist_style_profile": "riot_splash_fantasy",
    },
    "game_valorant_clean": {
        "craft_profile": "western_comic_pro",
        "shot_language": "dialogue_coverage",
        "pacing_plan": "balanced",
        "lettering_craft": "none",
        "value_plan": "color_script",
        "screentone_plan": "clean",
        "artist_style_profile": "valorant_clean_shapes",
    },
    "game_blizzard_heroic": {
        "craft_profile": "western_comic_pro",
        "shot_language": "cinematic",
        "pacing_plan": "balanced",
        "lettering_craft": "none",
        "value_plan": "color_script",
        "screentone_plan": "none",
        "artist_style_profile": "blizzard_cinematic_heroic",
    },
    "game_fromsoftware_dark": {
        "craft_profile": "cinematic_storyboard",
        "shot_language": "cinematic",
        "pacing_plan": "decompressed",
        "lettering_craft": "none",
        "value_plan": "bw_hierarchy",
        "screentone_plan": "dramatic",
        "artist_style_profile": "fromsoftware_dark_fantasy",
    },
    # Cartoon / animation focused
    "cartoon_pixar_story": {
        "craft_profile": "children_book",
        "shot_language": "cinematic",
        "pacing_plan": "balanced",
        "lettering_craft": "standard",
        "value_plan": "color_script",
        "screentone_plan": "none",
        "artist_style_profile": "pixar_shape_script",
    },
    "cartoon_disney_staging": {
        "craft_profile": "children_book",
        "shot_language": "mixed",
        "pacing_plan": "balanced",
        "lettering_craft": "standard",
        "value_plan": "color_script",
        "screentone_plan": "none",
        "artist_style_profile": "disney_animation_staging",
    },
    "cartoon_network_graphic": {
        "craft_profile": "western_comic_pro",
        "shot_language": "mixed",
        "pacing_plan": "compressed",
        "lettering_craft": "standard",
        "value_plan": "color_script",
        "screentone_plan": "clean",
        "artist_style_profile": "cartoon_network_graphic",
    },
    # Artist-legacy packs
    "mignola_noir": {
        "craft_profile": "western_comic_pro",
        "shot_language": "cinematic",
        "pacing_plan": "compressed",
        "lettering_craft": "none",
        "value_plan": "bw_hierarchy",
        "screentone_plan": "dramatic",
        "artist_style_profile": "mignola_noir_graphic",
    },
    "alex_ross_painterly": {
        "craft_profile": "western_comic_pro",
        "shot_language": "cinematic",
        "pacing_plan": "balanced",
        "lettering_craft": "none",
        "value_plan": "color_script",
        "screentone_plan": "clean",
        "artist_style_profile": "alex_ross_painterly_realism",
    },
    "frazetta_epic": {
        "craft_profile": "manga_pro",
        "shot_language": "manga_dynamic",
        "pacing_plan": "compressed",
        "lettering_craft": "none",
        "value_plan": "bw_hierarchy",
        "screentone_plan": "dramatic",
        "artist_style_profile": "frazetta_epic_fantasy",
    },
    "moebius_worldbuilding": {
        "craft_profile": "cinematic_storyboard",
        "shot_language": "mixed",
        "pacing_plan": "decompressed",
        "lettering_craft": "none",
        "value_plan": "color_script",
        "screentone_plan": "clean",
        "artist_style_profile": "moebius_line_worldbuilding",
    },
    # 3D anime game / booru-leaning packs
    "anime3d_genshin_keyart": {
        "craft_profile": "children_book",
        "shot_language": "cinematic",
        "pacing_plan": "balanced",
        "lettering_craft": "none",
        "value_plan": "color_script",
        "screentone_plan": "none",
        "artist_style_profile": "niziu_anime_3d_colorist",
    },
    "anime3d_hsr_cinematic": {
        "craft_profile": "cinematic_storyboard",
        "shot_language": "cinematic",
        "pacing_plan": "balanced",
        "lettering_craft": "none",
        "value_plan": "color_script",
        "screentone_plan": "none",
        "artist_style_profile": "alpha_3d_anime_keyart",
    },
    "anime3d_zzz_urban": {
        "craft_profile": "western_comic_pro",
        "shot_language": "manga_dynamic",
        "pacing_plan": "compressed",
        "lettering_craft": "none",
        "value_plan": "color_script",
        "screentone_plan": "clean",
        "artist_style_profile": "swd3e2_game_anime_hybrid",
    },
    "anime3d_arcane_painterly": {
        "craft_profile": "cinematic_storyboard",
        "shot_language": "cinematic",
        "pacing_plan": "decompressed",
        "lettering_craft": "none",
        "value_plan": "color_script",
        "screentone_plan": "none",
        "artist_style_profile": "reiq_anime_3d_sculpt",
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
    artist_style_profile: str = "none",
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
        ARTIST_STYLE_PROFILES.get((artist_style_profile or "none").lower().strip(), ""),
        SHOT_LANGUAGE_HINTS.get((shot_language or "none").lower().strip(), ""),
        PACING_PLAN_HINTS.get((pacing_plan or "none").lower().strip(), ""),
        LETTERING_CRAFT_HINTS.get((lettering_craft or "none").lower().strip(), ""),
        VALUE_PLAN_HINTS.get((value_plan or "none").lower().strip(), ""),
        SCREENTONE_PLAN_HINTS.get((screentone_plan or "none").lower().strip(), ""),
    ]
    return merge_prompt_fragments(*bits)


def resolve_color_render_controls(
    *,
    color_render_pack: str = "none",
    color_theory_mode: str = "none",
    gradient_blend_mode: str = "none",
    shading_technique: str = "none",
    render_pipeline: str = "none",
    color_render_extra: str = "",
) -> Dict[str, str]:
    """
    Resolve color/render controls from one preset pack + explicit overrides.
    """
    pack = COLOR_RENDER_PACK_PRESETS.get((color_render_pack or "none").lower().strip(), {})
    out = {
        "color_theory_mode": str(pack.get("color_theory_mode", "none")),
        "gradient_blend_mode": str(pack.get("gradient_blend_mode", "none")),
        "shading_technique": str(pack.get("shading_technique", "none")),
        "render_pipeline": str(pack.get("render_pipeline", "none")),
        "color_render_extra": "",
    }
    if (color_theory_mode or "none").lower().strip() != "none":
        out["color_theory_mode"] = str(color_theory_mode).strip()
    if (gradient_blend_mode or "none").lower().strip() != "none":
        out["gradient_blend_mode"] = str(gradient_blend_mode).strip()
    if (shading_technique or "none").lower().strip() != "none":
        out["shading_technique"] = str(shading_technique).strip()
    if (render_pipeline or "none").lower().strip() != "none":
        out["render_pipeline"] = str(render_pipeline).strip()
    if str(color_render_extra).strip():
        out["color_render_extra"] = str(color_render_extra).strip()
    return out


def color_render_bundle(
    *,
    color_theory_mode: str = "none",
    gradient_blend_mode: str = "none",
    shading_technique: str = "none",
    render_pipeline: str = "none",
    color_render_extra: str = "",
) -> str:
    """
    Merge artist-facing color theory + shading + rendering cues.
    """
    bits = [
        COLOR_THEORY_HINTS.get((color_theory_mode or "none").lower().strip(), ""),
        GRADIENT_BLEND_HINTS.get((gradient_blend_mode or "none").lower().strip(), ""),
        SHADING_TECHNIQUE_HINTS.get((shading_technique or "none").lower().strip(), ""),
        RENDER_PIPELINE_HINTS.get((render_pipeline or "none").lower().strip(), ""),
        str(color_render_extra or "").strip(),
    ]
    return merge_prompt_fragments(*bits)


def resolve_artist_technique_controls(
    *,
    artist_technique_pack: str = "none",
    linework_technique: str = "none",
    rendering_technique: str = "none",
    shading_technique_plan: str = "none",
    material_technique: str = "none",
    composition_technique: str = "none",
    artist_technique_extra: str = "",
) -> Dict[str, str]:
    """
    Resolve artist-technique controls from preset pack + explicit overrides.
    """
    pack = ARTIST_TECHNIQUE_PACK_PRESETS.get((artist_technique_pack or "none").lower().strip(), {})
    out = {
        "linework_technique": str(pack.get("linework_technique", "none")),
        "rendering_technique": str(pack.get("rendering_technique", "none")),
        "shading_technique_plan": str(pack.get("shading_technique_plan", "none")),
        "material_technique": str(pack.get("material_technique", "none")),
        "composition_technique": str(pack.get("composition_technique", "none")),
        "artist_technique_extra": "",
    }
    if (linework_technique or "none").lower().strip() != "none":
        out["linework_technique"] = str(linework_technique).strip()
    if (rendering_technique or "none").lower().strip() != "none":
        out["rendering_technique"] = str(rendering_technique).strip()
    if (shading_technique_plan or "none").lower().strip() != "none":
        out["shading_technique_plan"] = str(shading_technique_plan).strip()
    if (material_technique or "none").lower().strip() != "none":
        out["material_technique"] = str(material_technique).strip()
    if (composition_technique or "none").lower().strip() != "none":
        out["composition_technique"] = str(composition_technique).strip()
    if str(artist_technique_extra).strip():
        out["artist_technique_extra"] = str(artist_technique_extra).strip()
    return out


def artist_technique_bundle(
    *,
    linework_technique: str = "none",
    rendering_technique: str = "none",
    shading_technique_plan: str = "none",
    material_technique: str = "none",
    composition_technique: str = "none",
    artist_technique_extra: str = "",
) -> str:
    """
    Build advanced artist-technique prompt cues for 2D/3D digital and traditional art.
    """
    bits = [
        LINEWORK_TECHNIQUE_HINTS.get((linework_technique or "none").lower().strip(), ""),
        RENDERING_TECHNIQUE_HINTS.get((rendering_technique or "none").lower().strip(), ""),
        SHADING_TECHNIQUE_PLAN_HINTS.get((shading_technique_plan or "none").lower().strip(), ""),
        MATERIAL_TECHNIQUE_HINTS.get((material_technique or "none").lower().strip(), ""),
        COMPOSITION_TECHNIQUE_HINTS.get((composition_technique or "none").lower().strip(), ""),
        str(artist_technique_extra or "").strip(),
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
    artist_style_profile: str = "none",
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
        "artist_style_profile": pack.get("artist_style_profile", "none"),
        "shot_language": pack.get("shot_language", "none"),
        "pacing_plan": pack.get("pacing_plan", "none"),
        "lettering_craft": pack.get("lettering_craft", "none"),
        "value_plan": pack.get("value_plan", "none"),
        "screentone_plan": pack.get("screentone_plan", "none"),
    }
    if (craft_profile or "none").lower().strip() != "none":
        out["craft_profile"] = craft_profile
    if (artist_style_profile or "none").lower().strip() != "none":
        out["artist_style_profile"] = artist_style_profile
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


def _tokenize_style_query(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if (ch.isalnum() or ch == "_") else " " for ch in str(text or ""))
    return [t for t in cleaned.split() if t]


def suggest_popular_style_presets(
    query: str,
    *,
    category: str = "all",
    limit: int = 10,
) -> List[Dict[str, str]]:
    """
    Rank curated popular style presets by free-text query overlap.

    Returns rows with keys:
      id, category, score, description, and optional CLI fields
      (lexicon_style, art_medium_pack, art_medium_family, art_medium_variant,
       artist_pack, color_render_pack).
    """
    q_tokens = set(_tokenize_style_query(query))
    cat = str(category or "all").strip().lower()
    k = max(1, int(limit))

    scored: List[Dict[str, str]] = []
    for row in POPULAR_STYLE_PRESETS:
        row_cat = str(row.get("category", "")).lower()
        if cat not in ("", "all") and row_cat != cat:
            continue
        text = " ".join(
            [
                str(row.get("id", "")),
                str(row.get("category", "")),
                str(row.get("tags", "")),
                str(row.get("description", "")),
                str(row.get("lexicon_style", "")),
                str(row.get("art_medium_pack", "")),
                str(row.get("art_medium_variant", "")),
                str(row.get("artist_pack", "")),
                str(row.get("color_render_pack", "")),
            ]
        )
        row_tokens = set(_tokenize_style_query(text))
        overlap = len(q_tokens & row_tokens)
        if q_tokens:
            # Bias toward denser overlap; keep all rows when query empty.
            score = overlap / max(1.0, float(len(q_tokens)))
        else:
            score = 0.0
        if q_tokens and overlap == 0:
            continue
        out = dict(row)
        out["score"] = f"{score:.3f}"
        scored.append(out)

    if not scored and cat not in ("", "all"):
        return suggest_popular_style_presets(query, category="all", limit=limit)

    scored.sort(
        key=lambda r: (
            float(r.get("score", "0")),
            r.get("category", ""),
            r.get("id", ""),
        ),
        reverse=True,
    )
    return scored[:k]
