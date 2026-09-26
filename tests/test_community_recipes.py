"""PixAI / Civitai community recipe classification and structural borrow."""

from __future__ import annotations

from utils.caption.prompt_research import research_text_prompt
from utils.prompt.community_recipes import (
    all_seed_recipes,
    apply_community_research,
    classify_lane,
    is_blocked_prompt,
    pixai_seed_recipes,
)


def test_classify_pixai_slots():
    assert classify_lane(
        "pixai_mio wearing a flowing white summer dress, turning around as the breeze "
        "catches her skirt and makes it billow outward. She stands in a sunlit sunflower field."
    ) in ("simple", "intermediate")
    mid = classify_lane(
        "1girl, black hair, blue eyes, looking at viewer, slight smile, standing, "
        "outdoors, ginkgo leaves, autumn, anime style, masterpiece, best quality"
    )
    assert mid == "intermediate"
    hard = classify_lane("Character 1: sailor catgirl. BREAK Character 2: smaller girl, facing away")
    assert hard == "hard"


def test_block_minor_terms():
    assert is_blocked_prompt("loli, 1girl, undress")
    assert not is_blocked_prompt("1girl, solo, standing, school uniform")


def test_simple_does_not_eat_identity():
    recipes = pixai_seed_recipes()
    result = apply_community_research("a red fox in snow", recipes=recipes)
    assert result.lane == "simple"
    low = result.prompt.lower()
    assert "pixai_mio" not in low
    assert "<lora:" not in low


def test_intermediate_borrows_structure_from_six_slot():
    recipes = pixai_seed_recipes()
    result = apply_community_research(
        "1girl, black hair, blue eyes, standing, outdoors",
        recipes=recipes,
    )
    assert result.lane in ("simple", "intermediate")
    # Six-slot neighbor should be allowed to donate looking at viewer / lighting-quality
    # without replacing the user's subject.
    assert result.prompt.lower().startswith("1girl")
    assert "black hair" in result.prompt.lower()


def test_hard_adds_two_person_guard():
    recipes = pixai_seed_recipes()
    result = apply_community_research(
        "1girl catgirl and 1boy, cowgirl, BREAK two people on a bed",
        recipes=recipes,
    )
    # 1girl+1boy+BREAK is hard; guard tokens are structural.
    assert result.lane == "hard"
    assert "two distinct people" in result.prompt.lower() or "no identity merge" in result.prompt.lower()


def test_research_text_prompt_uses_pixai_seeds():
    res = research_text_prompt("1girl, solo, standing, sunflower field")
    assert res.diffusion_prompt
    assert any("lane:" in s or "pixai" in s for s in res.sources)


def test_pov_adds_camera_not_identity():
    recipes = all_seed_recipes()
    result = apply_community_research(
        "adult catgirl in a suit, mask off, pov from the male viewpoint",
        recipes=recipes,
    )
    low = result.prompt.lower()
    assert "pov" in low or "first-person" in low
    assert "pixai_mio" not in low
    assert "kinds:pov" in ",".join(result.sources) or "pov" in result.kinds


def test_edit_keeps_identity_tokens():
    recipes = all_seed_recipes()
    result = apply_community_research(
        "make her sit down",
        recipes=recipes,
        is_edit=True,
    )
    low = result.prompt.lower()
    assert "keep identity" in low
    assert "keep composition" in low


def test_sfw_does_not_borrow_nsfw_tokens():
    recipes = all_seed_recipes()
    result = apply_community_research("1girl, standing, park, fully clothed", recipes=recipes)
    low = result.prompt.lower()
    assert "uncensored" not in low
    assert "rating_explicit" not in low


def test_nsfw_can_borrow_uncensored():
    recipes = all_seed_recipes()
    result = apply_community_research("1girl, nsfw, looking at viewer", recipes=recipes)
    # Uncensored is optional; must not inject fully clothed over nsfw.
    assert "fully clothed" not in result.prompt.lower()


def test_explicit_act_is_nsfw_kind_not_clothed():
    recipes = all_seed_recipes()
    result = apply_community_research("1girl, 1boy, handjob, penis", recipes=recipes)
    assert "nsfw" in result.kinds
    assert "fully clothed" not in result.prompt.lower()
    assert "rating_safe" not in result.prompt.lower()
    assert "nsfw" not in result.negative.lower()
    assert "nude" not in result.negative.lower()
