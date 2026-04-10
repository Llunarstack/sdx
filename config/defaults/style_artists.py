# Style and artist tags from tag-based image boards (Danbooru, Gelbooru, etc.).
# Use for strong style conditioning: extract artist/style from captions and use as --style at inference.
#
# Buckets: **digital**, **traditional drawn/painted** (real or simulated media), **3D rendered** — each
# with SFW / NSFW where relevant. Use rating filters in your pipeline.
#
# Quality helpers: bucket + facet fragments (``style_tag_quality_fragments``), CLIP/style-embed
# summaries (``compact_style_summary_for_clip``, ``style_embedding_auxiliary_text``), and
# caption merge helpers. Tag fragments are also merged inside ``style_guidance.style_guidance_fragments``.

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ARTIST_STYLE_PATTERNS",
    "ARTIST_STYLE_TAGS",
    "DIGITAL_ART_STYLE_TAGS_SFW",
    "DIGITAL_ART_STYLE_TAGS_NSFW",
    "RENDERED_3D_STYLE_TAGS_SFW",
    "RENDERED_3D_STYLE_TAGS_NSFW",
    "TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_SFW",
    "TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_NSFW",
    "STYLE_PHRASE_PREFIXES",
    "STYLE_TAG_BUCKET_QUALITY_HINTS",
    "STYLE_TAG_FACET_RULES",
    "describe_style_tag_enrichment",
    "detect_style_tag_buckets",
    "matched_style_facet_ids",
    "matching_style_tags_in_prompt",
    "style_tag_quality_fragments",
    "append_style_tag_quality_to_prompts",
    "style_positive_addon",
    "style_negative_addon",
    "compact_style_summary_for_clip",
    "style_embedding_auxiliary_text",
    "prompt_has_style_artists_signal",
    "extract_style_from_text",
]

# Regex and phrases that indicate an artist or style in the caption/prompt.
# Order matters: more specific first. First match is used for style extraction.
ARTIST_STYLE_PATTERNS = [
    # "by artist_name", "art by X", "drawn by X", "illustration by X"
    (re.compile(r"\b(?:art\s+)?by\s+([^,]+?)(?:,|$)", re.IGNORECASE), 1),
    (re.compile(r"\b(?:drawn|painted|illustrated)\s+by\s+([^,]+?)(?:,|$)", re.IGNORECASE), 1),
    (re.compile(r"\bin\s+the\s+style\s+of\s+([^,]+?)(?:,|$)", re.IGNORECASE), 1),
    (re.compile(r"\bstyle\s+of\s+([^,]+?)(?:,|$)", re.IGNORECASE), 1),
    (re.compile(r"\b(?:like|similar to)\s+([^,]+?)(?:\s+style)?(?:,|$)", re.IGNORECASE), 1),
    # Danbooru/Gelbooru: "artist:name" or "artist name"
    (re.compile(r"\bartist:(\S+)", re.IGNORECASE), 1),
    (re.compile(r"\bstyle:(\S+)", re.IGNORECASE), 1),
]

# Prefixes that often start a style phrase (for comma-separated tags).
STYLE_PHRASE_PREFIXES = (
    "by ",
    "art by ",
    "drawn by ",
    "style of ",
    "in the style of ",
    "artist:",
    "style:",
    "art style ",
    "painting style ",
    "illustration style ",
)

# --- 2D digital (non-photoreal): SFW-focused styles ---
DIGITAL_ART_STYLE_TAGS_SFW: List[str] = [
    "vector_art",
    "flat_design",
    "corporate_memphis",
    "line_art",
    "clean_lineart",
    "thick_lineart",
    "minimalist_illustration",
    "webtoon",
    "manhwa",
    "manhua",
    "western_comic_ink",
    "comic_book_halftone",
    "pixel_art",
    "8bit",
    "16bit",
    "32bit",
    "retro_pixel",
    "isometric_pixel_art",
    "sprite_art",
    "chibi_digital",
    "kawaii_flat",
    "pastel_digital",
    "watercolor_digital",
    "gouache_digital",
    "pastel_drawing",
    "colored_pencil_digital",
    "charcoal_digital",
    "ink_wash_digital",
    "risograph",
    "screenprint_style",
    "linocut_digital",
    "childrens_book_illustration",
    "educational_illustration",
    "infographic_style",
    "ui_mockup",
    "icon_set_glossy",
    "matte_painting",
    "photobash",
    "speedpaint",
    "digital_sketch",
    "pencil_sketch_digital",
    "blueprint_style",
    "technical_diagram",
    "fashion_flat_sketch",
    "storyboard_style",
    "visual_development",
    "key_art_flat",
    "vaporwave_aesthetic",
    "synthwave_art",
    "outrun_style",
    "art_deco_digital",
    "nouveau_digital",
    "ukiyo-e_inspired_digital",
    "clip_studio_paint",
    "procreate_(software)",
    "photoshop_(medium)",
    "krita_(medium)",
    "paint_tool_sai",
    "ms_paint_(medium)",
    "scratchboard_digital",
    "paper_texture_overlay",
    "ascii_art",
    "ansi_art",
    "dithering",
    "ordered_dither",
    "stippling_digital",
    "crosshatching_digital",
    "halftone_dot_screen",
    "ben-day_dots",
    "pop_art_digital",
    "collage_digital",
    "scrapbook_style",
    "zine_style",
    "mixed_media_digital",
    "grainy_print_texture",
    "duotone_illustration",
    "tritone_illustration",
    "neon_line_art",
    "neon_sign_style",
    "graffiti_digital",
    "street_art_style",
    "mural_style_flat",
    "tattoo_flash_sheet",
    "sticker_art",
    "emoji_style",
    "telegram_sticker_style",
    "discord_sticker_style",
    "line_sticker_style",
    "papercraft_style",
    "origami_inspired_flat",
    "kirigami_style",
    "lowbrow_art",
    "pop_surrealism",
    "steampunk_illustration",
    "dieselpunk_illustration",
    "solarpunk_illustration",
    "cyberpunk_illustration",
    "atompunk_illustration",
    "afrofuturism_illustration",
    "biopunk_illustration",
    "tarot_card_art",
    "playing_card_art",
    "oracle_deck_style",
    "holographic_foil_effect",
    "iridescent_gradient",
    "metallic_foil_print_look",
    "embroidery_style_digital",
    "cross_stitch_style_digital",
    "quilt_pattern_style",
    "woodcut_digital",
    "engraving_digital",
    "etching_digital",
    "copperplate_style_digital",
    "sumi-e_digital",
    "chinese_ink_painting_digital",
    "korean_minhwa_style",
    "persian_miniature_inspired",
    "celtic_knot_pattern_flat",
    "isometric_room_cutaway",
    "diorama_style_illustration",
    "exploded_view_diagram",
    "patent_drawing_style",
    "cutaway_technical_illustration",
    "courtroom_sketch_style",
    "urban_sketching_digital",
    "plein_air_digital",
    "live_model_gesture_sketch",
    "impasto_digital",
    "thick_paint_strokes_digital",
    "sgraffito_texture_digital",
    "tempera_style_digital",
    "fresco_style_digital",
    "stained_glass_style_flat",
    "mosaic_tile_style",
    "quilling_paper_style_digital",
    "brutalist_graphic_design",
    "swiss_design_style",
    "bauhaus_inspired_flat",
    "memphis_design_revival",
    "y2k_chrome_graphic",
    "frutiger_aero",
    "skeuomorphic_icon_style",
    "glassmorphism_ui_art",
    "neumorphism_ui_art",
    "wireframe_aesthetic",
    "blueprint_aesthetic",
    "scanline_overlay",
    "crt_screen_effect_art",
    "vhs_cover_art",
    "dvd_cover_illustration",
    "movie_poster_flat_layout",
    "album_cover_illustration",
    "book_cover_illustration",
    "editorial_spot_illustration",
    "newspaper_cartoon",
    "political_cartoon_style",
    "gag_comic_strip",
    "yonkoma_style",
    "4koma",
    "silent_comic_style",
    "manga_tone_screen",
    "screentone_digital",
    "speed_lines_manga",
    "focus_lines_manga",
    "chibi_chibi_super_deformed",
    "super_flat_(movement)",
    "superflat_style",
    "kawaii_pastel_goth",
    "pastel_goth_illustration",
    "cottagecore_illustration",
    "goblincore_illustration",
    "dark_academia_illustration",
    "light_academia_illustration",
    "cozy_gaming_aesthetic_art",
    # 2025–2026 illustration / social trends (human texture, lo-fi, hybrid 3D+flat)
    "lo_fi_illustration",
    "lo_fi_maximalism_graphic",
    "lo_fi_reverie_aesthetic",
    "messy_human_illustration",
    "intentional_imperfection_illustration",
    "expressive_mark_making_digital",
    "wonky_line_art",
    "doodle_aesthetic_digital",
    "naive_art_illustration",
    "playful_childlike_illustration",
    "hand_lettering_illustration_digital",
    "visible_brush_strokes_emphasis",
    "tactile_texture_digital_illustration",
    "grain_forward_editorial_illustration",
    "anti_gloss_matte_texture_digital",
    "human_made_authenticity_signal_art",
    "neo_traditional_ink_watercolor_hybrid_digital",
    "quiet_graphic_minimalism",
    "bold_minimal_shapes_illustration",
    "editorial_negative_space_layout",
    "eco_brand_illustration",
    "sustainability_editorial_flat",
    "wellness_brand_soft_illustration",
    "food_illustration_hand_drawn_trend",
    "gen_z_editorial_illustration",
    "indie_merch_graphic_illustration",
    "bandcamp_aesthetic_art",
    "zine_fair_table_aesthetic",
    "riso_zine_layout_digital",
    "film_grain_editorial_illustration",
    "halation_soft_glow_illustration",
    "light_leak_photo_overlay_illustration",
    "bitmap_typography_aesthetic",
    "pixel_font_ui_illustration",
    "y2k_lofi_graphic_revival",
    "crt_bezel_frame_illustration",
    "timecode_stamp_overlay_art",
    "motion_blur_illustration_effect",
    "downscaled_compression_aesthetic_art",
    "flat_3d_hybrid_illustration",
    "spline_3d_flat_combo_look",
    "isometric_3d_plus_flat_ui_trend",
    "bevel_icon_3d_flat_hybrid",
    "chrome_type_3d_flat_combo",
    "gradient_mesh_revival_digital",
    "airbrush_y2k_revival_digital",
    "thermal_palette_illustration",
    "anaglyph_fake_3d_effect_flat",
    "dreamcore_illustration",
    "weirdcore_illustration_flat",
    "liminal_space_flat_illustration",
    "slime_gloss_texture_illustration",
    "coquette_aesthetic_illustration",
    "balletcore_illustration",
    "clean_girl_aesthetic_illustration",
    "mob_wife_aesthetic_illustration",
    "cottagecore_dark_academia_hybrid_illustration",
    "procreate_dreams_style",
    "adobe_fresco_look_digital",
    "medibang_paint_style",
    "ibis_paint_x_style",
    "autodesk_sketchbook_style_digital",
    "affinity_designer_vector_look",
    "figma_vector_export_illustration",
    "digital_gouache_matte_procreate_look",
    "rebelle_watercolor_engine_look",
    "open_peeps_style_flat",
    "noto_emoji_style_flat",
    "fluent_emoji_3d_flat_style",
    "discord_banner_profile_art_layout",
    "twitch_panel_banner_illustration",
    "link_in_bio_hero_illustration",
    "carousel_slide_illustration_layout",
    "spotify_canvas_style_illustration",
    "substack_header_illustration",
    "newsletter_header_illustration_2020s",
]

# --- 2D digital: mature / adult-genre style anchors (datasets, not content description) ---
DIGITAL_ART_STYLE_TAGS_NSFW: List[str] = [
    "hentai",
    "eroge_cg",
    "eroge_style",
    "doujin_cover_style",
    "doujinshi_art",
    "r-18",
    "rating:explicit",
    "nsfw_art",
    "adult_comic_ink",
    "mature_digital_painting",
    "explicit_art_style",
    "bara_(genre)",
    "yaoi_(genre)",
    "yuri_(genre)",
    "ecchi",
    "pinup_digital",
    "pinup_style",
    "playboy_illustration_style",
    "vargas_style",
    "glamour_illustration",
    "boudoir_illustration",
    "nsfw_vector",
    "adult_webtoon_style",
    "hentai_game_ui_style",
    "r18_comic_cover",
    "explicit_doujin_layout",
    "mature_pixiv_style",
    "fanbox_r18_art_style",
    "patreon_nsfw_pinup_tier",
    "subscribe_star_adult_illustration",
    "newgrounds_mature_style",
    "gelbooru_tag_style",
    "sankaku_style_tags",
    "rule34_xyz_style",
    "e621_furry_nsfw_style",
    "futanari_(genre)",
    "tentacle_genre_style",
    "monster_girl_genre_style",
    "kemono_nsfw_digital",
    "anthro_explicit_style",
    "latex_fashion_illustration",
    "leather_fashion_mature_illustration",
    "lingerie_catalog_illustration",
    "boudoir_photography_style_illustration",
    "shibari_ink_style",
    "rope_bondage_illustration_style",
    "femboy_art_style",
    "otoko_no_ko_style",
    "mature_yaoi_style",
    "mature_yuri_style",
    "bara_manga_style",
    "josei_mature_romance_cover",
    "smut_novel_cover_style",
    "ahegao_(style)",
    "heart-shaped_pupils_style",
    "steam_cloud_censor_style",
    "light_censor_mosaic_style",
    "bar_censor_style",
    "after_sex_illustration_style",
    "implied_afterglow_style",
    "nsfw_vector_flat",
    "adult_vector_pinup",
    "erotic_tarot_parody_style",
    "playboy_cartoon_style",
    "hustler_cartoon_style",
    "vintage_pinup_nose_art_style",
    "burlesque_poster_style",
    "nsfw_furry_badge_style",
    "adult_badge_button_art",
    "nsfw_emote_style",
    "chibi_nsfw_(style)",
    "oppai_focus_composition_style",
    "thigh_focus_illustration_style",
    "foot_focus_illustration_style",
    "macro_micro_(genre)_style",
    "vore_(genre)_cover_style",
    "guro_(genre)_manga_style",
    "ero_guro_style",
    "corruption_(genre)_visual_style",
    "mind_break_(genre)_cover_style",
    "netorare_cover_layout",
    "netori_cover_layout",
    "harem_comedy_manga_style_mature",
    "isekai_r18_light_novel_cover",
    "otome_game_r18_cg_style",
    "galge_cg_style_mature",
    "bishoujo_game_cg_style",
    "eroge_splash_screen_style",
    "vn_choice_screen_overlay_style",
    "dating_sim_ui_style_mature",
    # Platform / creator promo illustration (mature-audience anchors)
    "fansly_banner_art_style",
    "onlyfans_promo_banner_illustration",
    "chaturbate_overlay_panel_art_style",
    "loyalfans_creator_header_illustration",
    "justforfans_promo_graphic_style",
]

# --- 3D rendered / CG: SFW (games, film, product, archviz) ---
RENDERED_3D_STYLE_TAGS_SFW: List[str] = [
    "unreal_engine",
    "unreal_engine_5",
    "unity_3d",
    "godot_style",
    "arnold_render",
    "redshift_render",
    "vray",
    "corona_render",
    "keyshot",
    "cinema_4d",
    "maya_3d",
    "3ds_max",
    "houdini_render",
    "zbrush_render",
    "substance_painter",
    "marvelous_designer",
    "pbr_textures",
    "ray_traced",
    "path_traced",
    "global_illumination",
    "ambient_occlusion_3d",
    "low_poly_3d",
    "voxel_art",
    "voxel_3d",
    "isometric_3d",
    "architectural_visualization",
    "archviz",
    "product_visualization",
    "automotive_cgi",
    "furniture_render",
    "medical_3d_visualization",
    "scientific_visualization_3d",
    "disney_3d",
    "pixar_style",
    "dreamworks_cgi",
    "illumination_style",
    "blue_sky_studios_style",
    "fortnite_style",
    "roblox_style",
    "minecraft_style",
    "animal_crossing_3d",
    "splatoon_style",
    "zelda_botw_style",
    "stylized_3d_character",
    "hand_painted_3d",
    "toon_shader_3d",
    "npr_render",
    "clay_render",
    "white_clay_render",
    "turntable_render",
    "studio_hdri_lighting",
    "blender_cycles",
    "blender_eevee",
    "luxcore_render",
    "mitsuba_renderer",
    "indigo_renderer",
    "maxwell_render",
    "fstorm_render",
    "octane_x",
    "lumion_render",
    "twinmotion_style",
    "enscape_render",
    "d5_render",
    "lumion_walkthrough_still",
    "sketchup_style_render",
    "rhino_render",
    "solidworks_visualize",
    "fusion360_render",
    "cad_render_clean",
    "npr_toon_shader",
    "gradient_ramp_shader_3d",
    "rim_light_shader_3d",
    "matcap_render",
    "sketchfab_viewer_style",
    "marmoset_toolbag_render",
    "substance_designer_preview",
    "quixel_megascans_look",
    "photogrammetry_mesh_render",
    "gaussian_splatting_style",
    "nerf_style_view",
    "volumetric_fog_3d",
    "god_rays_3d",
    "caustics_water_3d",
    "subsurface_scattering_skin_3d",
    "anisotropic_metal_3d",
    "clearcoat_car_paint_3d",
    "fabric_microfiber_shader_3d",
    "fur_groom_render",
    "hair_groom_render",
    "feather_groom_render",
    "crowd_simulation_3d_still",
    "fluid_simulation_3d_still",
    "cloth_simulation_3d_still",
    "rigid_body_pile_3d",
    "destruction_fracture_3d",
    "vfx_explosion_3d",
    "pyro_sim_3d",
    "space_nebula_3d_environment",
    "exterior_archviz_twilight",
    "interior_archviz_daylight",
    "product_hero_shot_white_bg",
    "ecommerce_360_spin_style",
    "packshot_render",
    "cosmetic_product_splash_3d",
    "liquids_splash_cgi",
    "food_photography_cgi",
    "jewelry_macro_cgi",
    "watch_macro_cgi",
    "sneaker_product_cgi",
    "electronics_exploded_3d",
    "smartphone_mockup_3d",
    "laptop_mockup_3d",
    "vehicle_configurator_still",
    "aircraft_livery_preview_3d",
    "ship_hull_render_3d",
    "hard_surface_sci-fi_prop_3d",
    "greeble_kitbash_3d",
    "kitbash3d_style",
    "megascans_scene",
    "quixel_bridge_scene",
    "unity_hdrp",
    "unity_urp",
    "unreal_nanite_scene",
    "unreal_lumen_scene",
    "metahuman_render",
    "digital_double_vfx",
    "previs_style_gray_shade",
    "layout_pose_render_gray",
    "zbrush_bpr",
    "keyshot_hdri_studio",
    "overwatch_style_3d",
    "valorant_style_3d",
    "league_of_legends_cinematic_3d",
    "dota2_cinematic_style",
    "warcraft_cinematic_style",
    "starcraft_cinematic_style",
    "final_fantasy_cg_style",
    "kingdom_hearts_3d_style",
    "persona_summon_cutin_style_3d",
    "metal_gear_style_3d",
    "resident_evil_re_engine",
    "death_stranding_decima",
    "horizon_forbidden_west_style",
    "ghost_of_tsushima_style_3d",
    "spider-man_ps5_style",
    "batman_arkham_style_3d",
    "assassins_creed_style_3d",
    "farcry_style_3d",
    "watch_dogs_style_3d",
    "cyberpunk2077_redengine",
    "elden_ring_style_3d",
    "dark_souls_style_3d",
    "bloodborne_style_3d",
    "sekiro_style_3d",
    "monster_hunter_style_3d",
    "pokemon_3d_model_render",
    "zelda_totk_style_3d",
    "mario_odyssey_style_3d",
    "splatoon_3_style_3d",
    "kirby_star_allies_style_3d",
    "brawl_stars_style_3d",
    "clash_of_clans_style_3d",
    "clash_royale_style_3d",
    "brawlhalla_style_3d",
    "multiversus_style_3d",
    "fall_guys_style_3d",
    "among_us_style_3d",
    "minecraft_render_with_shader",
    "roblox_studio_render",
    "vrchat_world_asset_style",
    "neosvr_asset_style",
    "rec_room_style_3d",
    "vr_sculpt_gesture_style",
    "tilt_brush_style_export",
    "medium_vr_sculpt_style",
    "nomad_sculpt_export",
    "forger_app_render",
    "ipad_procreate_3d_paint",
    "cozy_gamer_room_isometric_3d",
    "low_poly_isometric_city_3d",
    "voxel_city_3d",
    "lego_style_3d",
    "brick_build_3d",
    "playmobil_style_3d",
    "funko_pop_style_3d",
    "nendoroid_style_3d",
    "figma_style_3d",
    "shfiguarts_style_3d",
    "hot_toys_style_3d",
    "collectible_statue_render",
    "garage_kit_paint_render",
    "gunpla_custom_render",
    "mecha_model_kit_render",
    "diorama_photo_3d_scene",
    "train_layout_miniature_3d",
    "warhammer_diorama_style_3d",
    "historical_miniature_paint_3d",
    # 2025–2026 3D + social / product viz trends
    "spline_3d_editor_export",
    "bevel_and_bold_3d_icon_trend",
    "soft_minimal_3d_product_studio",
    "bubbly_3d_social_icon_style",
    "claymation_stop_motion_3d_look",
    "aardman_clay_style_3d",
    "laika_stop_motion_cgi_look",
    "felt_texture_3d_character",
    "yarn_fiber_3d_shader_style",
    "stylized_fabric_sim_3d_still",
    "squishmallow_soft_toy_3d_look",
    "popmart_blind_box_figure_3d",
    "sonny_angel_style_3d_collectible",
    "smiski_style_glow_3d",
    "roblox_ugc_catalog_style_3d",
    "fortnite_cosmetic_render_style",
    "valorant_splash_art_3d_hybrid",
    "overwatch_2_splash_3d_style",
    "genshin_character_splash_3d_style",
    "honkai_splash_art_3d_style",
    "wuthering_waves_style_3d",
    "zenless_zone_zero_splash_3d",
    "punishing_gray_raven_style_3d",
    "arknights_promo_3d_style",
    "blue_archive_cafe_chibi_3d_style",
    "nikke_goddess_of_victory_style_3d",
    "stellar_blade_style_3d",
    "black_myth_wukong_unreal_style",
    "splatoon_inkling_3d_render_style",
    "animal_crossing_villager_3d_style",
    "pokemon_scarlet_violet_style_3d",
    "palworld_creature_render_style",
    "helldivers_propaganda_3d_poster",
    "baldurs_gate_3_character_portrait_3d",
    "hades_game_paint_over_3d",
    "cult_of_the_lamb_flat_3d_hybrid",
    "hollow_knight_silksong_teaser_3d_style",
    "indie_horror_ps1_low_poly_3d",
    "liminal_pool_3d_aesthetic",
    "backrooms_found_footage_3d_still",
]

# --- 3D rendered: adult / mature game & CG style anchors ---
RENDERED_3D_STYLE_TAGS_NSFW: List[str] = [
    "nsfw_3d",
    "adult_game_cg",
    "eroge_3d",
    "mature_3d_character",
    "nsfw_blender",
    "adult_sfm",
    "nsfw_source_filmmaker",
    "adult_vrchat_avatar_style",
    "adult_unity_render",
    "explicit_3d_still",
    "adult_visual_novel_cg",
    "r18_game_asset_style",
    "daz3d_iray_render",
    "daz_studio_default_lighting",
    "poser_firefly_render",
    "poser_superfly_render",
    "xnalara_xps_pose_render",
    "koikatsu_scene_render",
    "koikatsu_party_style",
    "honey_select_2_style",
    "ai_shoujo_style",
    "com3d2_style",
    "custom_maid_3d_style",
    "cm3d2_style",
    "illusion_soft_engine_style",
    "renpy_3d_asset_style",
    "live2d_3d_hybrid_scene",
    "vroid_studio_nsfw_avatar",
    "vrchat_erotic_avatar_render",
    "chilloutvr_nsfw_avatar",
    "second_life_mature_zone_render",
    "imvu_adult_room_render",
    "sims_4_wickedwhims_style_render",
    "skyrim_adult_mod_armor_render",
    "fallout_adult_armor_mod_render",
    "blade_and_soul_preset_nsfw",
    "black_desert_character_creator_nsfw",
    "ffxiv_gpose_nsfw_scene",
    "gpose_mature_lighting",
    "mmd_r18_dance_render",
    "mmd_ray_mmd_shader_nsfw",
    "ray_mmd_style",
    "mmd_nch_shader_style",
    "adult_garrys_mod_scenebuild",
    "sfm_gmod_nsfw_pose",
    "xnalara_nsfw_scene",
    "blender_grease_pencil_nsfw",
    "unreal_metahuman_nsfw_scene",
    "iclone_character_creator_nsfw",
    "character_creator_3_pipeline_nsfw",
    "reallusion_iclonestyle_render",
    "vam_(software)_style",
    "virt_a_mate_style_render",
    "adult_vr_sculpt_export",
    "oculus_medium_nsfw_export",
    "adult_nomans_sky_character_mod",
    "cyberpunk_nude_mod_render",
    "resident_evil_mod_nsfw_costume",
    "tekken_character_custom_nsfw",
    "dead_or_alive_costume_nsfw_render",
    "senran_kagura_style_3d_mature",
    "gal_gun_style_3d_mature",
    "omega_labyrinth_style_3d",
    "monster_girl_quest_3d_fan_render",
    "corruption_of_champions_style_3d",
    "trials_in_tainted_space_style_3d",
    "adult_vn_background_3d_asset",
    "eroge_map_3d_tileset_style",
    "dating_sim_bedroom_3d_asset",
    "onahole_product_cgi_parody",
    "adult_toy_packshot_cgi",
    "silicone_material_shader_nsfw_prop",
    "latex_shader_nsfw_outfit_3d",
    "wet_skin_shader_explicit_3d",
    "sweat_droplet_shader_mature_3d",
    "sheer_fabric_shader_mature_3d",
    "see_through_plastic_shader_3d",
    "nsfw_fluid_sim_splash_3d",
    "censorship_bar_3d_composite",
    "pixelation_mosaic_3d_post",
    "adult_patreon_reward_render_4k",
    "fanbox_god_rays_nsfw_scene",
    "subscribe_star_oil_slick_skin_3d",
]

# --- Traditional hand-drawn / painted (physical media or faithful digital mimic) ---
TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_SFW: List[str] = [
    # Graphite & charcoal
    "pencil_sketch",
    "graphite_drawing",
    "mechanical_pencil_sketch",
    "wooden_pencil_drawing",
    "charcoal_drawing",
    "vine_charcoal",
    "compressed_charcoal",
    "willow_charcoal",
    "white_charcoal_on_toned_paper",
    "carbon_pencil_drawing",
    "graphite_wash",
    # Pens & ink
    "pen_and_ink",
    "dip_pen_drawing",
    "fountain_pen_sketch",
    "fineliner_drawing",
    "micron_pen_art",
    "ballpoint_pen_sketch",
    "brush_pen_ink",
    "bamboo_brush_ink",
    "india_ink_drawing",
    "technical_pen_illustration",
    "crosshatching_ink",
    "parallel_hatching",
    "stippling_ink_traditional",
    "scribble_sketch",
    "blind_contour_drawing",
    "continuous_line_drawing",
    "gesture_drawing_charcoal",
    "figure_drawing_quick_pose",
    "life_drawing_clothed_model",
    # Chalk, pastel, crayon
    "conté_crayon",
    "sanguine_chalk",
    "sepia_chalk",
    "red_chalk_drawing",
    "silverpoint_drawing",
    "soft_pastel_painting",
    "oil_pastel_drawing",
    "pan_pastel",
    "hard_pastel_sketch",
    "chalk_pastel_portrait",
    "sidewalk_chalk_style",
    "wax_crayon_drawing",
    "oil_crayon",
    "colored_pencil_drawing",
    "prismacolor_pencil",
    "polychromos_pencil",
    "watercolor_pencil_drawing",
    "marker_rendering_traditional",
    "copic_marker_sketch",
    "alcohol_marker_illustration",
    "chartpak_marker",
    # Water-based paint
    "watercolor_painting_traditional",
    "wet_on_wet_watercolor",
    "dry_brush_watercolor",
    "watercolor_washes",
    "gouache_painting_traditional",
    "designer_gouache",
    "poster_color_paint",
    "acrylic_gouache",
    "casein_painting",
    "egg_tempera",
    "distemper_paint",
    # Oil & acrylic (canvas / panel)
    "oil_painting_traditional",
    "alla_prima_oil",
    "impasto_oil_paint",
    "palette_knife_oil",
    "glazing_oil_layers",
    "scumbling_oil",
    "underpainting_visible",
    "grisaille_underpainting",
    "verdaccio_underpainting",
    "acrylic_painting_canvas",
    "heavy_body_acrylic",
    "fluid_acrylic_pour",
    "acrylic_pouring_cells",
    "encaustic_painting",
    "fresco_painting",
    "fresco_secco",
    "tempera_panel_painting",
    # East Asian brush
    "sumi-e_traditional",
    "shuimo_hua",
    "gongbi_painting",
    "xieyi_painting",
    "chinese_brush_painting",
    "japanese_ink_wash",
    "korean_ink_painting",
    "nihonga_painting",
    "mineral_pigment_painting",
    "lacquer_painting_traditional",
    # Miniature & manuscript
    "persian_miniature_painting",
    "mughal_miniature_painting",
    "indian_miniature_traditional",
    "byzantine_icon_egg_tempera",
    "medieval_manuscript_illumination",
    "book_of_kells_style",
    "celtic_manuscript_style",
    # Western art-historical handles (prompt vocabulary)
    "renaissance_drawing_study",
    "sfumato_chalk_study",
    "baroque_chiaroscuro_oil",
    "romanticism_oil_sketch",
    "impressionist_oil_painting",
    "plein_air_oil",
    "post_impressionist_oil",
    "expressionist_oil_brushwork",
    "fauvism_paint_style",
    "cubist_oil_fragmentation",
    "surrealist_oil_painting",
    "abstract_expressionist_canvas",
    "action_painting_drips",
    "pointillism_painting",
    "divisionism_dots_paint",
    "tonalist_painting",
    "hudson_river_school_oil",
    "pre_raphaelite_oil",
    "art_nouveau_painted_panel",
    "arts_and_crafts_illustration_paint",
    "trompe_loeil_painting",
    "hyperrealism_oil",
    "photorealistic_oil_painting",
    "hyperrealism_pencil_drawing",
    "photorealistic_graphite",
    # Printmaking (hand-pulled look)
    "woodcut_print",
    "linocut_print",
    "etching_print",
    "aquatint_print",
    "drypoint_print",
    "mezzotint_print",
    "lithograph_stone_print",
    "screenprint_silkscreen",
    "monotype_print",
    "collagraph_print",
    # Decorative & folk painted
    "tole_painting",
    "rosemaling",
    "bauernmalerei",
    "folk_art_acrylic_wood",
    "pysanka_pattern_paint",
    "truck_art_painted_style",
    "sign_painting_enamel",
    "gold_leaf_gilded_icon",
    "verre_eglomise_reverse_glass",
    # Applied / design hand art
    "botanical_watercolor_illustration",
    "scientific_gouache_plate",
    "nature_journal_watercolor",
    "bird_field_guide_illustration",
    "architectural_watercolor_render",
    "fashion_illustration_gouache",
    "fashion_croquis_watercolor",
    "shoe_design_marker_sketch",
    "automotive_design_gouache",
    "industrial_design_marker_comp_traditional",
    "storyboard_pencil_traditional",
    "layout_animation_pencil_paper",
    "traditional_animation_keyframe_pencil",
    "cel_animation_paint_traditional",
    "gouache_animation_background_hand_painted",
    "comic_brush_ink_traditional",
    "manga_manuscript_dip_pen",
    "g_pen_traditional_manga",
    "kab_pen_traditional_manga",
    "screen_tone_glued_traditional",
    # Misc traditional
    "scratchboard_traditional",
    "airbrush_illustration_traditional",
    "spray_paint_canvas_fine_art",
    "coffee_stain_ink_wash",
    "tea_stain_aged_paper",
    "mixed_media_collage_paint",
    "oil_stick_on_paper",
    "oil_bar_drawing",
    "wax_resist_batik_paint",
    "marbling_paper_ebru",
    # Trending physical / hybrid mediums (2020s social + atelier revival)
    "posca_marker_painting",
    "acrylic_paint_marker_on_canvas",
    "molotow_marker_mural_style",
    "krink_drip_marker_street_art",
    "cold_wax_oil_painting",
    "oil_and_cold_wax_medium",
    "encaustic_monotype_hybrid",
    "cyanotype_print_painting",
    "gelli_print_monoprint_texture",
    "eco_print_leaf_stain_art",
    "natural_pigment_foraged_ink",
    "rice_paper_xuan_ink_wash",
    "mulberry_paper_texture_paint",
    "washi_collage_paint_mixed_media",
    "toned_gray_strathmore_sketch",
    "stillman_birn_sketchbook_look",
    "moleskine_sketchbook_aesthetic",
    "hahnemuhle_watercolor_paper_texture",
    "arches_cold_press_watercolor",
    "plein_air_gouache_block_in",
    "urban_sketchers_gouache_on_location",
    "alla_prima_gouache_field_study",
    "gouache_in_procreate_printed_look",
    "acrylic_gouache_matte_study",
    "casein_underpainting_visible",
    "velatura_glaze_oil_technique",
    "sfumato_soft_edge_oil_study",
    "alla_prima_portrait_oil_direct",
    "knife_only_palette_knife_oil",
    "brush_only_no_blending_oil",
    "hyperrealism_colored_pencil_prismacolor",
    "panpastel_portrait_drawing",
    "neocolor_ii_watersoluble_crayon",
    "carandache_luminance_pencil",
    "derwent_lightfast_pencil",
    "tombow_dual_brush_lettering",
    "brush_lettering_sign_paint_hybrid",
    "chalkboard_menu_lettering_traditional",
    "wine_ink_and_iron_gall_ink",
    "walnut_ink_drawing",
    "bister_ink_wash",
    "acrylic_ink_pouring_traditional",
    "alcohol_ink_flow_painting",
    "resin_art_geode_pour",
    "fluid_art_cells_acrylic",
    "pouring_medium_cells_paint",
]

# Mature figure / academic studio vocabulary (adult models; art-school context).
TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_NSFW: List[str] = [
    "academic_nude_figure_drawing",
    "life_drawing_nude_charcoal",
    "life_drawing_nude_pencil",
    "studio_oil_nude_study",
    "pastel_nude_study",
    "watercolor_nude_study",
    "renaissance_nude_study_style",
    "classical_marble_study_sketch",
    "boudoir_watercolor_figure",
    "red_chalk_academic_nude",
    "gesture_nude_short_pose",
    "croquis_nude_one_minute",
    "mature_figure_painting_oil",
    "explicit_figure_drawing_line_art",
    "adult_only_life_class_style",
]

# Core artist / studio / legacy tags (kept first for stable ordering).
_STYLE_TAGS_CORE: List[str] = [
    # Anime / illustration (Danbooru-style; often use underscore)
    "makoto_shinkai",
    "ghibli",
    "studio_ghibli",
    "miyazaki",
    "hayao_miyazaki",
    "kyoani",
    "shaft",
    "trigger",
    "ufotable",
    "a-1_pictures",
    "kyoto_animation",
    "wit_studio",
    "bones_(studio)",
    "mappa",
    "clamp_(studio)",
    "ufotable_style",
    "trigger_style",
    "digital_art",
    "anime_screencap",
    "official_art",
    "fan_art",
    # Danbooru / tag-board frequent artist-style tags (2D)
    "sakimichan",
    "wlop",
    "guweiz",
    "redjuice",
    "kantoku",
    "lam_(artist)",
    "ask_(artist)",
    "toi8",
    "huke_(artist)",
    "saitom",
    "krenz_cushart",
    "daito",
    # Booru 3D / game-like style anchors
    "anime_3d",
    "toon_3d",
    "cel_shaded_3d",
    "stylized_3d",
    "game_cg",
    "render",
    "octane_render",
    "eevee_render",
    "blender_(software)",
    "mmd",
    "source_filmmaker",
    "genshin_impact_style",
    "honkai_star_rail_style",
    "zenless_zone_zero_style",
    # Tagboard prompt syntax helpers
    "artist:",
    "style:",
    "masterpiece",
    "best_quality",
    "highres",
    "absurdres",
    "official_style",
    "by_artist",
    # Art styles (general)
    "oil_painting",
    "watercolor",
    "digital_painting",
    "concept_art",
    "cel_shading",
    "soft_lighting",
    "dramatic_lighting",
    "cinematic",
    "fantasy_art",
    "character_design",
    "environment_art",
    "pixiv",
    "artstation",
    "behance",
]


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


ARTIST_STYLE_TAGS: List[str] = _dedupe_preserve_order(
    _STYLE_TAGS_CORE
    + DIGITAL_ART_STYLE_TAGS_SFW
    + DIGITAL_ART_STYLE_TAGS_NSFW
    + RENDERED_3D_STYLE_TAGS_SFW
    + RENDERED_3D_STYLE_TAGS_NSFW
    + TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_SFW
    + TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_NSFW
)

# When prompts use tags from a bucket, append medium-specific positive/negative hints (first-pass fidelity).
STYLE_TAG_BUCKET_QUALITY_HINTS: Dict[str, Tuple[str, str]] = {
    "digital_sfw": (
        "coherent 2d digital finish, edge and fill language matching the chosen technique, unified color handling, deliberate flatness or texture where that style expects it",
        "photoreal skin or shading bleeding onto graphic flat regions, muddy gradients, banding in clean fills, accidental 3d lighting on purely 2d subjects, inconsistent line weight",
    ),
    "digital_nsfw": (
        "genre-consistent mature 2d digital language, stable character read, rendering finish aligned with the declared anchor tags",
        "style collapse from contradictory tag intent, noisy linework where clean ink or cel clarity was implied, incoherent shading relative to the chosen genre anchor",
    ),
    "traditional_sfw": (
        "believable traditional medium behavior (paper tooth, wet or dry edges, pigment body, brush or pencil economy), lighting that respects the drawn or painted surface",
        "plastic airbrush smoothness contradicting declared media, perfectly even digital gradients where hand variance is expected, conflicting wet and dry paint logic in one passage",
    ),
    "traditional_nsfw": (
        "academic figure clarity for life-study vocabulary, proportional coherence, medium-faithful modeling of form",
        "broken proportions for studio-reference intent, mushy structure without skeletal read, mixed clothed and nude figure-language cues that fight the brief",
    ),
    "rendered_3d_sfw": (
        "renderer-coherent materials and lighting, stable perspective and contact shadows, one clear shader language (PBR or toon) without half measures",
        "2d cel shading on photoreal props mixed arbitrarily, contradictory shadow rigs, waxy subsurface on hard surfaces, shader or exposure fighting within one frame",
    ),
    "rendered_3d_nsfw": (
        "consistent game or VN cg finish for mature anchors, stable mesh and silhouette read, coherent post grade and material rules",
        "uncanny mixed-resolution kitbash, incoherent SSS or fabric shaders, visible material seams, clashing real-time and offline cues",
    ),
}

_STYLE_TAG_BUCKET_ORDER: Tuple[Tuple[str, Sequence[str]], ...] = (
    ("digital_sfw", DIGITAL_ART_STYLE_TAGS_SFW),
    ("digital_nsfw", DIGITAL_ART_STYLE_TAGS_NSFW),
    ("traditional_sfw", TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_SFW),
    ("traditional_nsfw", TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_NSFW),
    ("rendered_3d_sfw", RENDERED_3D_STYLE_TAGS_SFW),
    ("rendered_3d_nsfw", RENDERED_3D_STYLE_TAGS_NSFW),
)

# Extra nudge whenever any bucket matched: cross-cutting coherence (one fragment each).
STYLE_TAG_GLOBAL_BUCKET_POSITIVE = (
    "single dominant focal hierarchy, consistent style grammar from foreground to background, no contradictory medium cues on the same subject"
)
STYLE_TAG_GLOBAL_BUCKET_NEGATIVE = (
    "dueling focal points of equal weight, mid-image style pivot, accidental watermark or UI chrome, compression blocks read as texture"
)

# (facet_id, trigger_tags, positive_hint, negative_hint) — tags are chosen from bucket lists where possible.
STYLE_TAG_FACET_RULES: Tuple[Tuple[str, Tuple[str, ...], str, str], ...] = (
    (
        "pixel_dither_raster",
        (
            "pixel_art",
            "8bit",
            "16bit",
            "32bit",
            "retro_pixel",
            "isometric_pixel_art",
            "sprite_art",
            "ordered_dither",
            "dithering",
            "ansi_art",
        ),
        "crisp pixel grid discipline, hard tile edges, stable palette quantization, one consistent pixel scale across the subject",
        "bilinear softness inside tiles, mixed resolutions in one sprite or scene, gradients that break quantize discipline",
    ),
    (
        "vector_flat_graphic",
        (
            "vector_art",
            "flat_design",
            "corporate_memphis",
            "minimalist_illustration",
            "swiss_design_style",
            "bauhaus_inspired_flat",
            "duotone_illustration",
            "tritone_illustration",
        ),
        "clean vector topology, even flat fills, deliberate corner radius and stroke parity, generous negative space",
        "raster grit inside supposedly flat fields, uneven parallel stroke weight, muddy anti-alias halos on color steps",
    ),
    (
        "comic_ink_tone",
        (
            "line_art",
            "clean_lineart",
            "thick_lineart",
            "western_comic_ink",
            "comic_book_halftone",
            "screentone_digital",
            "manga_tone_screen",
            "webtoon",
            "manhwa",
            "manhua",
            "speed_lines_manga",
        ),
        "ink hierarchy serving story read, stable halftone or tone mesh scale, blacks and holds that stay intentional",
        "gray mush replacing decisive blacks, dot or tone scale drifting across panels, painterly smear breaking comic clarity",
    ),
    (
        "watercolor_gouache",
        (
            "watercolor_painting_traditional",
            "watercolor_digital",
            "wet_on_wet_watercolor",
            "gouache_painting_traditional",
            "gouache_digital",
            "watercolor_washes",
            "botanical_watercolor_illustration",
            "nature_journal_watercolor",
            "fashion_illustration_gouache",
        ),
        "transparent lift and granulation for watercolor, matte opaque passes for gouache, reserved whites or paper color behaving consistently",
        "opaque chalky sludge in a pure watercolor brief, incoherent wet-on-dry timing, lost paper discipline with no breathing highlights",
    ),
    (
        "oil_acrylic_impasto",
        (
            "oil_painting_traditional",
            "oil_painting",
            "impasto_oil_paint",
            "alla_prima_oil",
            "palette_knife_oil",
            "impasto_digital",
            "thick_paint_strokes_digital",
            "acrylic_painting_canvas",
            "fluid_acrylic_pour",
        ),
        "sculptural brush or knife economy, thick-to-thin edge logic, temperature and value transitions that respect form",
        "plastic rainbow smear without form logic, uniform thickness ignoring planes, arbitrary swirl unrelated to light",
    ),
    (
        "print_riso_screen",
        (
            "risograph",
            "riso_zine_layout_digital",
            "screenprint_style",
            "halftone_dot_screen",
            "ben-day_dots",
            "grainy_print_texture",
            "linocut_digital",
            "woodcut_digital",
        ),
        "print-native dot or line frequency, stable registration between separations, paper tooth supporting ink laydown",
        "uncorrelated RGB noise posing as halftone, chaotic moire from conflicting meshes, accidental photoreal shading in print-flat zones",
    ),
    (
        "pbr_raytrace_3d",
        (
            "pbr_textures",
            "ray_traced",
            "path_traced",
            "global_illumination",
            "subsurface_scattering_skin_3d",
            "unreal_engine_5",
            "unreal_engine",
            "arnold_render",
            "redshift_render",
            "corona_render",
            "vray",
            "blender_cycles",
        ),
        "physically grounded specular and roughness reads, contact shadows and AO that agree with the key light, stable exposure",
        "mirror-metal on velvet confusion, roughness that ignores material class, stacked incompatible shadow directions",
    ),
    (
        "toon_npr_3d",
        (
            "toon_shader_3d",
            "npr_render",
            "hand_painted_3d",
            "cel_shaded_3d",
            "stylized_3d_character",
            "gradient_ramp_shader_3d",
            "npr_toon_shader",
            "toon_3d",
            "anime_3d",
        ),
        "art-directed ramp or stepped shading, silhouette-first readability, stylized normals that stay coherent under the rig light",
        "photoreal microdetail on toon volumes, fighting spec lobes, lighting that ignores the toon normal map thesis",
    ),
    (
        "voxel_block_3d",
        (
            "voxel_art",
            "voxel_3d",
            "minecraft_style",
            "voxel_city_3d",
            "roblox_style",
        ),
        "locked voxel lattice, consistent block scale, orthographic or perspective grammar without wobble",
        "off-grid floating cubes, mixed block sizes in one solid, subdivision smoothness betraying voxel intent",
    ),
    (
        "archviz_product_studio_3d",
        (
            "architectural_visualization",
            "archviz",
            "interior_archviz_daylight",
            "exterior_archviz_twilight",
            "product_hero_shot_white_bg",
            "packshot_render",
            "studio_hdri_lighting",
        ),
        "camera-verticals discipline, clean product or room read, believable scale cues and material zoning",
        "warped verticals, floating furniture without contact, plastic showroom sheen on every surface",
    ),
)

_FACET_BY_ID: Dict[str, Tuple[str, str]] = {r[0]: (r[2], r[3]) for r in STYLE_TAG_FACET_RULES}


def _merge_csv_hints(*chunks: str) -> str:
    seen = set()
    out: List[str] = []
    for chunk in chunks:
        if not chunk or not str(chunk).strip():
            continue
        for part in str(chunk).split(","):
            p = part.strip()
            if not p:
                continue
            k = p.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
    return ", ".join(out)


def _prompt_contains_tag(text_lower: str, tag: str) -> bool:
    if tag in text_lower:
        return True
    alt = tag.replace("_", " ")
    return alt in text_lower if alt != tag else False


def detect_style_tag_buckets(prompt: str) -> Tuple[str, ...]:
    """
    Return bucket ids (stable order) when the prompt contains any tag from that bucket's list.
    Uses substring match with underscore or space forms (same idea as ``extract_style_from_text``).
    """
    if not (prompt and prompt.strip()):
        return ()
    tl = prompt.lower()
    found: List[str] = []
    for bid, tag_list in _STYLE_TAG_BUCKET_ORDER:
        for tag in tag_list:
            if _prompt_contains_tag(tl, tag):
                found.append(bid)
                break
    return tuple(found)


def matching_style_tags_in_prompt(prompt: str, *, max_matches: int = 64) -> List[str]:
    """
    All known bucket-list tags present in the prompt (bucket scan order, deduped).
    Useful for logging, UI, or downstream LoRA routing.
    """
    if not (prompt and prompt.strip()):
        return []
    tl = prompt.lower()
    out: List[str] = []
    seen: set[str] = set()
    for _, tag_list in _STYLE_TAG_BUCKET_ORDER:
        for tag in tag_list:
            if tag in seen:
                continue
            if _prompt_contains_tag(tl, tag):
                seen.add(tag)
                out.append(tag)
                if len(out) >= max_matches:
                    return out
    return out


def matched_style_facet_ids(prompt: str) -> Tuple[str, ...]:
    """Facet ids whose trigger tags appear in the prompt (declaration order)."""
    if not (prompt and prompt.strip()):
        return ()
    tl = prompt.lower()
    found: List[str] = []
    for facet_id, tags, _, _ in STYLE_TAG_FACET_RULES:
        if any(_prompt_contains_tag(tl, t) for t in tags):
            found.append(facet_id)
    return tuple(found)


def describe_style_tag_enrichment(prompt: str) -> Dict[str, Any]:
    """Structured summary of bucket hits, facets, matched tags, and merged fragments."""
    pos, neg = style_tag_quality_fragments(prompt)
    return {
        "buckets": list(detect_style_tag_buckets(prompt)),
        "facets": list(matched_style_facet_ids(prompt)),
        "matched_tags": matching_style_tags_in_prompt(prompt),
        "positive_fragment": pos,
        "negative_fragment": neg,
    }


def style_tag_quality_fragments(prompt: str) -> Tuple[str, str]:
    """
    Positive and negative CSV fragments: global coherence, per-bucket hints, and facet hints.
    Safe to merge into caption / negative_caption alongside ``style_guidance_fragments``.
    """
    buckets = detect_style_tag_buckets(prompt)
    facets = matched_style_facet_ids(prompt)
    if not buckets and not facets:
        return "", ""
    pos_parts: List[str] = []
    neg_parts: List[str] = []
    if buckets:
        pos_parts.append(STYLE_TAG_GLOBAL_BUCKET_POSITIVE)
        neg_parts.append(STYLE_TAG_GLOBAL_BUCKET_NEGATIVE)
        for b in buckets:
            pos_parts.append(STYLE_TAG_BUCKET_QUALITY_HINTS[b][0])
            neg_parts.append(STYLE_TAG_BUCKET_QUALITY_HINTS[b][1])
    elif facets:
        # e.g. ``anime_3d`` lives in core tags used as facet triggers but outside bucket lists.
        pos_parts.append(STYLE_TAG_GLOBAL_BUCKET_POSITIVE)
        neg_parts.append(STYLE_TAG_GLOBAL_BUCKET_NEGATIVE)
    for fid in facets:
        fp, fn = _FACET_BY_ID[fid]
        pos_parts.append(fp)
        neg_parts.append(fn)
    return _merge_csv_hints(*pos_parts), _merge_csv_hints(*neg_parts)


def style_positive_addon(prompt: str) -> str:
    """Positive CSV fragment only (bucket + facet + global when applicable)."""
    p, _ = style_tag_quality_fragments(prompt)
    return p


def style_negative_addon(prompt: str) -> str:
    """Negative CSV fragment only."""
    _, n = style_tag_quality_fragments(prompt)
    return n


def append_style_tag_quality_to_prompts(positive: str, negative: str) -> Tuple[str, str]:
    """
    Merge style-tag quality hints into a caption pair (scripts, custom pipelines).
    Uses the same fragments as ``style_guidance_fragments``'s embedded tag pass.
    """
    tp, tn = style_tag_quality_fragments(positive or "")
    out_pos = f"{positive}, {tp}".strip().strip(",") if tp else (positive or "")
    out_neg = _merge_csv_hints(negative or "", tn) if tn else (negative or "")
    return out_pos, out_neg


def prompt_has_style_artists_signal(prompt: str) -> bool:
    """True if bucket lists or facet triggers fire (core-only facet triggers included)."""
    return bool(detect_style_tag_buckets(prompt) or matched_style_facet_ids(prompt))


def compact_style_summary_for_clip(prompt: str, *, max_chars: int = 220) -> str:
    """
    Dense, tokenizer-friendly summary: bucket ids, facet ids, and matched tags.
    Use as a short CLIP prefix or log line (truncate-safe).
    """
    if not (prompt and prompt.strip()):
        return ""
    chunks: List[str] = []
    b = detect_style_tag_buckets(prompt)
    if b:
        chunks.append("buckets:" + "+".join(b))
    f = matched_style_facet_ids(prompt)
    if f:
        chunks.append("facets:" + "+".join(f[:8]))
    tags = matching_style_tags_in_prompt(prompt, max_matches=12)
    if tags:
        chunks.append("tags:" + ",".join(tags))
    if not chunks:
        return ""
    s = " ".join(chunks)
    if len(s) <= max_chars:
        return s
    return s[: max(1, max_chars - 1)] + "…"


def style_embedding_auxiliary_text(prompt: str, *, max_chars: int = 320) -> str:
    """
    Text to concatenate with ``extract_style_from_text`` for a dedicated style channel
    (e.g. T5 style embed): extracted style phrase plus compact bucket/facet/tag summary.
    """
    if not (prompt and prompt.strip()):
        return ""
    parts: List[str] = []
    ext = extract_style_from_text(prompt)
    if ext:
        parts.append(ext.replace("_", " ").strip())
    summary = compact_style_summary_for_clip(prompt, max_chars=max(48, max_chars // 2))
    if summary:
        parts.append(summary)
    if not parts:
        return ""
    s = " — ".join(parts)
    if len(s) <= max_chars:
        return s
    return s[: max(1, max_chars - 1)] + "…"


def extract_style_from_text(text: str, known_tags: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract a style or artist string from caption/prompt for style conditioning.
    Returns the first match from ARTIST_STYLE_PATTERNS, or a known tag if present.
    """
    if not (text and text.strip()):
        return None
    text_lower = text.lower()
    tags = known_tags or ARTIST_STYLE_TAGS

    # 1) Try regex patterns
    for pattern, group in ARTIST_STYLE_PATTERNS:
        m = pattern.search(text)  # use original text so captured style keeps case
        if m:
            style = m.group(group).strip()
            if len(style) > 1 and len(style) < 120:
                return style

    # 2) Check for known artist/style tags (e.g. "artist:miyazaki" or "oil painting")
    for tag in tags:
        # Tag might use underscore; caption might use space
        tag_alt = tag.replace("_", " ")
        if tag in text_lower or tag_alt in text_lower:
            return tag.replace("_", " ").strip()

    return None
