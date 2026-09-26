"""Tests for agentic stills improvements (interview, drafts, critique, taste)."""

from __future__ import annotations

from pathlib import Path

from utils.agentic.post_gen_critique import build_post_gen_critique, critique_answers_to_fix_plan
from utils.agentic.progressive_interview import (
    answer_interview_item,
    next_interview_item,
    skip_interview_item,
    start_progressive_interview,
)
from utils.generation.agentic_stills import plan_agentic_stills
from utils.generation.draft_thumbs import plan_draft_thumbnails, promote_draft_to_final
from utils.generation.print_intent import default_ref_policy, resolve_print_intent
from utils.generation.user_taste import (
    UserTaste,
    apply_taste_quiz_answers,
    apply_taste_to_prompts,
    save_user_taste,
)


def test_progressive_interview_one_at_a_time():
    state = start_progressive_interview("1girl, catgirl")
    first = next_interview_item(state)
    assert first is not None
    state = answer_interview_item(state, first.id, text="sitting")
    second = next_interview_item(state)
    assert second is not None
    assert second.id != first.id
    state = skip_interview_item(state, second.id)
    assert second.id in state.skipped


def test_draft_thumbs_promote():
    plan = plan_draft_thumbnails("a catgirl", num_drafts=4, base_seed=10)
    assert "--pick-best" in plan.draft_argv
    assert plan.draft_argv[plan.draft_argv.index("--num") + 1] == "4"
    final = promote_draft_to_final(plan, chosen_index=2, user_picked=True)
    assert "--seed" in final
    assert final[final.index("--seed") + 1] == "12"


def test_critique_to_masks(tmp_path: Path):
    img = tmp_path / "out.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    plan = build_post_gen_critique(str(img), "1girl")
    plan = critique_answers_to_fix_plan(
        plan,
        {"face_ok": "no — fix face", "hands_ok": "yes", "outfit_ok": "yes", "bg_ok": "yes"},
        work_dir=tmp_path / "crit",
        width=256,
        height=256,
    )
    assert "face" in plan.fix_regions
    assert plan.mask_paths.get("face")
    assert Path(plan.mask_paths["face"]).is_file()
    assert "--mask" in plan.refine_argv


def test_taste_quiz_and_apply(tmp_path: Path):
    taste = UserTaste()
    taste = apply_taste_quiz_answers(
        taste,
        {
            "style_axis": "anime",
            "lighting_bias": "neon",
            "hates": "plastic hands, watermark",
            "favorite_artists": "eipril",
            "aspect_intent": "phone",
        },
    )
    assert taste.style_axis == "anime"
    assert any("plastic" in n.lower() or "hand" in n.lower() for n in taste.negative_bank)
    pos, neg = apply_taste_to_prompts("1girl, catgirl", "", taste)
    assert "anime" in pos.lower() or "lineart" in pos.lower()
    assert "eipril" in pos.lower()
    assert "watermark" in neg.lower() or "plastic" in neg.lower()
    path = save_user_taste(taste, tmp_path / "taste.json")
    assert path.is_file()


def test_print_intent_and_ref_policy():
    intent = resolve_print_intent("phone")
    assert intent is not None
    assert intent.height > intent.width
    pol = default_ref_policy(nsfw=True, web_consent=True, uploads={"style": ["a.png"]})
    assert pol.mode_for("style") == "upload"
    assert pol.mode_for("subject") == "search"


def test_plan_agentic_stills_offline(tmp_path: Path):
    plan = plan_agentic_stills(
        "1girl, catgirl, eipril style",
        work_dir=str(tmp_path / "as"),
        interview=False,
        drafts=True,
        assume_missing=True,
        allow_web=False,
        print_intent="square",
    )
    assert not plan.needs_user
    assert plan.sample_draft_argv
    assert "--pick-best" in plan.sample_draft_argv
    assert plan.sample_final_argv
    assert Path(plan.work_dir, "plan.json").is_file()
