"""Tests for utils.agentic: agentic stack, perfect-gen loop, visual brain, superior stack.
Also holds what used to be: test_agentic_stack, test_agentic_stills, test_perfect_gen, test_visual_brain, test_superior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from config.presets import AgenticStackDefaults, FlywheelPlan, SuperiorStackDefaults
from models.attention import hybrid_attention_fraction, linear_attention
from utils.agentic.agent import (
    AgentContext,
    AgentTool,
    GenerationStepKind,
    ImageGenerationAgent,
    MemoryBox,
    SceneBrief,
    SceneElement,
    ToolRegistry,
    TrajectoryExperience,
    TrajectoryRecord,
    VisualBrain,
    VisualBrainConfig,
    append_experience_memory,
    apply_memory_to_prompt,
    build_default_plan,
    default_stop_conditions,
    distill_trajectory_experience,
    load_experience_memory,
    load_memory_box,
    plan_from_prompt,
    plan_visual_brain,
    prompt_coverage_score,
    reflect_on_result,
    save_memory_box,
    synthesize_scene_brief,
)
from utils.agentic.interview import (
    answer_interview_item,
    apply_clarify_answers,
    assume_pose_place_defaults,
    build_clarify_questions,
    build_post_gen_critique,
    critique_answers_to_fix_plan,
    next_interview_item,
    skip_interview_item,
    start_progressive_interview,
)
from utils.agentic.quality_loop import (
    AutoImproveConfig,
    CompositeRanker,
    MinePairsConfig,
    QualityGateRunner,
    SelfCorrectConfig,
    SelfCorrectPolicy,
    SuperiorInferenceConfig,
    SuperiorPromptStack,
    TfidfFactIndex,
    ViTMineConfig,
    average_state_dicts,
    benchmark_sample_args_for_negatives,
    blended_reward,
    build_auto_improve_argv,
    build_dpo_train_argv,
    build_markdown_report,
    build_superior_sample_argv,
    build_tfidf_index_from_jsonl,
    expand_prompt_heuristic,
    mine_hard_negatives,
    mine_pairs_from_benchmark,
    mine_vit_preference_pairs,
    run_flywheel,
    tags_from_benchmark_row,
)
from utils.generation.editing import parse_part_requests
from utils.generation.guidance import GuidanceProbe, qsilk_micrograin_stabilize
from utils.generation.planners import (
    PerfectGenConfig,
    UserTaste,
    apply_taste_quiz_answers,
    apply_taste_to_prompts,
    plan_agentic_stills,
    plan_draft_thumbnails,
    plan_perfect_gen,
    promote_draft_to_final,
    resolve_print_intent,
    save_user_taste,
)
from utils.modeling.architecture import (
    BlockCacheConfig,
    BlockDiTCache,
    CFGRejectionTracker,
    FeatureCacheConfig,
    FeatureCachePolicy,
    TaylorBlockCache,
    TaylorCacheConfig,
    apply_dynamic_width,
    pick_best_candidate_index,
    spatial_token_importance,
    taylor_forecast_tensor,
    timestep_dynamic_width,
)
from utils.prompt.scene import decompose_prompt_facets
from utils.training.preference import (
    BranchGRPOConfig,
    GRPOGuardConfig,
    TurningPointGRPOConfig,
    branch_rollout_flow_samples,
    detect_turning_point_indices,
    enumerate_branch_paths,
    estimate_x0_from_xt_flow,
    group_relative_advantages,
    grpo_guard_weighted_loss,
    grpo_weighted_loss,
    incremental_rewards_from_trajectory,
    iso_temporal_group_advantages,
    ratio_norm_advantages,
    rectify_policy_gradient,
    reference_kl_penalty,
    safeguard_dpo_margins,
    safeguarded_dpo_preference_loss,
    sde_discretization_lambda,
    step_advantages_from_gains,
    timestep_dpo_weight,
    tp_grpo_step_weights,
)

# ==================================================================================================
# test_agentic_stack: Agentic stack: planner, reflector, experience, agent dry-run.
# ==================================================================================================


def test_plan_from_prompt_full_utpc() -> None:
    plan = plan_from_prompt("a cat", use_rag=True, expand=True, verify=True, self_correct=True)
    kinds = [s.kind for s in plan.steps]
    assert GenerationStepKind.knowledge_retrieval in kinds
    assert GenerationStepKind.refine_prompt in kinds
    assert GenerationStepKind.diffusion_sample in kinds
    assert GenerationStepKind.verify_visual in kinds
    assert GenerationStepKind.critique_revise in kinds
    assert len(plan.steps) == 5


def test_plan_from_prompt_minimal() -> None:
    plan = plan_from_prompt("x", use_rag=False, expand=False, verify=False, self_correct=False)
    assert len(plan.steps) == 1
    assert plan.steps[0].kind == GenerationStepKind.diffusion_sample


def test_build_default_plan_respects_defaults() -> None:
    d = AgenticStackDefaults(expand_prompt=False, self_correct=False)
    plan = build_default_plan("dog", d, use_rag=False)
    kinds = [s.kind for s in plan.steps]
    assert GenerationStepKind.refine_prompt not in kinds
    assert GenerationStepKind.critique_revise not in kinds


def test_reflect_accepts_good_metrics() -> None:
    out = reflect_on_result({"composite": 0.8, "clip": 0.3, "sharpness": 200.0}, iteration=0)
    assert out.accepted
    assert out.prompt_patch == ""


def test_reflect_patches_low_metrics() -> None:
    out = reflect_on_result({"composite": 0.4, "clip": 0.1, "sharpness": 50.0}, iteration=1)
    assert not out.accepted
    assert "sharp" in out.prompt_patch or "subject" in out.prompt_patch
    assert "blur" in out.negative_patch.lower() or "quality" in out.negative_patch.lower()


def test_distill_trajectory_experience() -> None:
    trajectories = [
        TrajectoryRecord(
            trajectory_id="a",
            prompt_final="cat, studio lighting, tack sharp",
            negative_prompt="blur",
            tool_sequence=["rag_retrieve", "expand_prompt", "generate", "verify"],
            out_path="a.png",
            composite=0.75,
            clip_score=0.3,
            metrics={"composite": 0.75, "clip": 0.3},
            accepted=True,
        ),
        TrajectoryRecord(
            trajectory_id="b",
            prompt_final="cat",
            negative_prompt="",
            tool_sequence=["generate", "verify"],
            out_path="b.png",
            composite=0.4,
            clip_score=0.1,
            metrics={"composite": 0.4, "clip": 0.1},
            accepted=False,
        ),
    ]
    exp = distill_trajectory_experience("cat", trajectories)
    assert exp is not None
    assert exp.best_id == "a"
    assert exp.worst_id == "b"
    assert "studio" in exp.prompt_delta.lower() or "sharp" in exp.prompt_delta.lower()
    assert "rag_retrieve" in exp.tool_delta


def test_experience_memory_roundtrip(tmp_path: Path) -> None:
    mem = tmp_path / "experience_memory.jsonl"
    exp = TrajectoryExperience(
        goal_prompt="dog",
        best_id="x",
        worst_id="y",
        prompt_delta="cinematic lighting",
        negative_delta="blur",
    )
    append_experience_memory(mem, exp)
    loaded = load_experience_memory(mem)
    assert len(loaded) == 1
    assert loaded[0].prompt_delta == "cinematic lighting"
    merged = apply_memory_to_prompt("a dog in park", loaded)
    assert "cinematic lighting" in merged


def test_tool_registry_dry_run(tmp_path: Path) -> None:
    ctx = AgentContext(
        ckpt="fake.pt",
        prompt="sunset",
        work_dir=str(tmp_path),
        out="out.png",
        dry_run=True,
    )
    reg = ToolRegistry(ctx, repo_root=tmp_path, expand_prompt=True)
    reg.execute(AgentTool.expand_prompt)
    gen = reg.execute(AgentTool.generate)
    assert gen.ok
    verify = reg.execute(AgentTool.verify)
    assert verify.ok
    assert verify.data["composite"] > 0.0


def test_agent_run_dry_run(tmp_path: Path) -> None:
    ctx = AgentContext(
        ckpt="fake.pt",
        prompt="mountain lake",
        work_dir=str(tmp_path / "run"),
        dry_run=True,
    )
    d = AgenticStackDefaults(max_reflect_loops=1, min_composite_accept=0.8)
    res = ImageGenerationAgent(d, repo_root=tmp_path).run(ctx)
    assert res.accepted
    trace_path = tmp_path / "run" / "agent_trace.json"
    assert trace_path.is_file()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["goal_prompt"] == "mountain lake"
    assert len(trace["trajectories"]) == 1


def test_agent_evolve_dry_run(tmp_path: Path) -> None:
    ctx = AgentContext(
        ckpt="fake.pt",
        prompt="red car",
        work_dir=str(tmp_path / "evolve"),
        dry_run=True,
    )
    res = ImageGenerationAgent(AgenticStackDefaults(), repo_root=tmp_path).run_evolve(ctx, variants=2)
    assert len(res.trace.trajectories) == 2
    assert (tmp_path / "evolve" / "evolve_trace.json").is_file()


def test_default_stop_conditions() -> None:
    stops = default_stop_conditions(AgenticStackDefaults(max_reflect_loops=5))
    assert stops.max_outer_loops == 5


def test_role_pipeline_dry_run(tmp_path: Path) -> None:
    from utils.agentic.agent import RolePipeline

    ctx = AgentContext(
        ckpt="fake.pt",
        prompt="forest path",
        work_dir=str(tmp_path),
        out="roles.png",
        dry_run=True,
    )
    pipe = RolePipeline.from_context(ctx, repo_root=tmp_path)
    res = pipe.run()
    assert res.ok
    assert len(res.stages) == 3
    assert res.stages[0].role == "reasoner"
    assert res.stages[1].role == "designer"
    assert res.stages[2].role == "verifier"
    assert res.metrics["composite"] > 0.0


# ==================================================================================================
# test_agentic_stills: Tests for agentic stills improvements (interview, drafts, critique, taste).
# ==================================================================================================


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


def test_print_intent():
    intent = resolve_print_intent("phone")
    assert intent is not None
    assert intent.height > intent.width


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


# ==================================================================================================
# test_perfect_gen: Tests for perfect-gen facet decompose / clarify / memory / plan glue.
# ==================================================================================================


def test_facet_decompose_subject_attire_style():
    f = decompose_prompt_facets("1girl, Furry, catgirl, black lingerie, eipril style")
    assert any("catgirl" in s.lower() or "furry" in s.lower() for s in f.subject)
    assert any("lingerie" in a.lower() for a in f.attire)
    assert f.style or f.artist_hint
    assert "subject" in f.search_queries
    assert "attire" in f.search_queries
    assert "style" in f.search_queries
    assert "pose" in f.missing
    assert "place" in f.missing


def test_clarify_assume_defaults():
    f = decompose_prompt_facets("1girl, catgirl, black lingerie")
    q = build_clarify_questions(f)
    assert q.needs_user
    assert any("Pose" in x for x in q.questions)
    assumed = assume_pose_place_defaults(f)
    assert "pose" in assumed.assumed
    assert "place" in assumed.assumed
    assert "standing" in assumed.enriched_prompt.lower()


def test_apply_clarify_answers():
    out = apply_clarify_answers("1girl, catgirl", {"pose": "sitting", "place": "bedroom"})
    assert "sitting" in out
    assert "bedroom" in out


def test_memory_box_roundtrip(tmp_path: Path):
    box = MemoryBox(session_id="abc", prompt="hi")
    box.set_refs("style", ["a.png", "b.png"])
    path = tmp_path / "mem.json"
    save_memory_box(box, path)
    loaded = load_memory_box(path)
    assert loaded is not None
    assert loaded.session_id == "abc"
    assert loaded.refs["style"] == ["a.png", "b.png"]


def test_plan_perfect_gen_offline(tmp_path: Path):
    plan = plan_perfect_gen(
        "1girl, catgirl, black lingerie, eipril style",
        PerfectGenConfig(
            work_dir=str(tmp_path / "pg"),
            session_id="test",
            allow_web=False,
            assume_missing=True,
            run_rag=False,
        ),
    )
    assert not plan.needs_user
    assert "standing" in plan.enriched_prompt.lower() or plan.clarify.get("assumed")
    assert Path(plan.memory_path).is_file()
    assert "--prompt" in plan.sample_argv
    mem = json.loads(Path(plan.memory_path).read_text(encoding="utf-8"))
    assert mem["facets"]["attire"]


def test_plan_perfect_gen_ask_blocks(tmp_path: Path):
    plan = plan_perfect_gen(
        "1girl, catgirl",
        PerfectGenConfig(
            work_dir=str(tmp_path / "ask"),
            allow_web=False,
            assume_missing=False,
            run_rag=False,
        ),
    )
    assert plan.needs_user
    assert plan.questions


def test_gen_interview_asks_for_photos():
    from utils.agentic.interview import build_gen_interview, format_interview_for_user

    iv = build_gen_interview("1girl, catgirl, black lingerie, eipril style")
    assert iv.needs_user
    kinds = {i.kind for i in iv.items}
    assert "photo" in kinds
    assert "choice" in kinds
    photo_ids = {i.id for i in iv.items if i.kind == "photo"}
    assert "face_photo" in photo_ids
    assert "style_photos" in photo_ids or "attire_photo" in photo_ids
    text = format_interview_for_user(iv)
    assert "PHOTO" in text
    assert "Pose" in text or "pose" in text.lower()


def test_plan_interview_blocks_until_answered(tmp_path: Path):
    plan = plan_perfect_gen(
        "1girl, catgirl, eipril style",
        PerfectGenConfig(
            work_dir=str(tmp_path / "iv"),
            allow_web=False,
            interview=True,
            assume_missing=False,
            run_rag=False,
        ),
    )
    assert plan.needs_user
    assert plan.photo_requests
    assert plan.interview_text


def test_plan_interview_with_answers(tmp_path: Path):
    plan = plan_perfect_gen(
        "1girl, catgirl",
        PerfectGenConfig(
            work_dir=str(tmp_path / "iv2"),
            allow_web=False,
            interview=True,
            assume_missing=True,
            run_rag=False,
            interview_answers={
                "pose": "sitting",
                "place": "bedroom",
                "attire": "black dress",
            },
        ),
    )
    assert not plan.needs_user
    assert "sitting" in plan.enriched_prompt.lower() or plan.clarify


# ==================================================================================================
# test_visual_brain: Visual Brain: scene brief, planner, dry-run orchestrator.
# ==================================================================================================


def test_parse_part_requests_from_prompt() -> None:
    reqs = parse_part_requests("use the hat from image 1 and background from image 2")
    assert len(reqs) >= 1
    assert any(r.part == "hat" for r in reqs)


def test_synthesize_scene_brief_keeps_user_prompt() -> None:
    brief = synthesize_scene_brief(
        "a red sports car at sunset",
        understandings=[],
        expected_text="FAST",
    )
    assert "red sports car" in brief.user_prompt
    assert brief.expected_text == "FAST"
    facts = brief.to_facts()
    assert any("Primary user request" in f for f in facts)


def test_prompt_coverage_score() -> None:
    brief = SceneBrief(user_prompt="cat")
    s = prompt_coverage_score(brief, {"composite": 0.7, "clip": 0.3, "ocr_match": 0.9})
    assert 0.0 < s <= 1.0


def test_plan_visual_brain_has_understand_steps() -> None:
    plan = plan_visual_brain("sign reading HELLO", has_references=True, web_search=True, expected_text="HELLO")
    kinds = [s.kind for s in plan.steps]
    assert GenerationStepKind.semantic_segment in kinds
    assert GenerationStepKind.verify_text_ocr in kinds
    tools = [s.kwargs.get("tool") for s in plan.steps]
    assert "web_search" in tools
    assert "understand_refs" in tools


def test_visual_brain_dry_run(tmp_path: Path) -> None:
    brain = VisualBrain(config=VisualBrainConfig(web_search=False))
    res = brain.run(
        ckpt="fake.pt",
        prompt="a cat in a hat",
        work_dir=str(tmp_path / "run"),
        dry_run=True,
    )
    assert res.accepted
    assert res.brief.user_prompt == "a cat in a hat"


def test_scene_brief_save_load(tmp_path: Path) -> None:
    brief = SceneBrief(
        user_prompt="test",
        elements=[SceneElement(name="subject", description="cat")],
    )
    p = tmp_path / "brief.json"
    brief.save(p)
    loaded = SceneBrief.load(p)
    assert loaded.user_prompt == "test"
    assert len(loaded.elements) == 1


# ==================================================================================================
# test_superior: Superior stack: DPO, soup, quality gates.
# ==================================================================================================


# ==================================================================================================
# test_superior_extended: Superior stack: DPO, soup, quality gates.
# ==================================================================================================


def test_mine_pairs_from_benchmark(tmp_path: Path) -> None:
    bench = tmp_path / "bench.json"
    w = tmp_path / "w.png"
    loser = tmp_path / "l.png"
    w.write_bytes(b"x")
    loser.write_bytes(b"y")
    bench.write_text(
        json.dumps(
            [
                {"case": "c1", "prompt": "a cat", "output": str(w), "composite": 0.9},
                {"case": "c1", "prompt": "a cat", "output": str(loser), "composite": 0.5},
            ]
        ),
        encoding="utf-8",
    )
    cfg = MinePairsConfig(benchmark_json=str(bench), out_jsonl=str(tmp_path / "p.jsonl"), min_margin=0.1)
    pairs = mine_pairs_from_benchmark(cfg)
    assert len(pairs) == 1
    assert pairs[0]["win_image_path"].endswith("w.png")


def test_build_dpo_argv() -> None:
    from utils.agentic.quality_loop import DPOStageConfig

    argv = build_dpo_train_argv(DPOStageConfig(ckpt="m.pt", preference_jsonl="p.jsonl"))
    assert "--ckpt" in argv and "m.pt" in argv


def test_average_state_dicts() -> None:
    a = {"w": torch.tensor([1.0, 2.0])}
    b = {"w": torch.tensor([3.0, 4.0])}
    avg = average_state_dicts([a, b])
    assert torch.allclose(avg["w"], torch.tensor([2.0, 3.0]))


def test_quality_gate_runner() -> None:
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    img[10:20, 10:20, 0] = 255
    runner = QualityGateRunner()
    res = runner.evaluate(img, prompt="test")
    assert "sharpness" in res.scores


# ==================================================================================================
# test_superior_stack: Tests for utils/superior stack.
# ==================================================================================================


def test_tfidf_retrieval_orders_relevant_facts() -> None:
    idx = TfidfFactIndex(
        facts=[
            "neon signs reflect on wet pavement in cyberpunk alleys",
            "pastoral meadow with sheep at sunrise",
            "database indexing with B-trees",
        ]
    )
    hits = idx.query("cyberpunk alley neon rain", top_k=2)
    assert "cyberpunk" in hits[0].lower()


def test_superior_prompt_stack_no_rag() -> None:
    out = SuperiorPromptStack(append_domain_tips=False).enrich("a red fox")
    assert out == "a red fox"


def test_build_superior_sample_argv() -> None:
    cfg = SuperiorInferenceConfig(num_candidates=4, local_rag_jsonl="facts.jsonl")
    argv = build_superior_sample_argv(ckpt="m.pt", prompt="cat", out="o.png", config=cfg)
    assert "--pick-best" in argv
    assert "superior_composite" in argv
    assert "--local-rag-jsonl" in argv


def test_composite_ranker_picks_sharper_image() -> None:
    blur = np.full((32, 32, 3), 128, dtype=np.uint8)
    sharp = blur.copy()
    sharp[8:24, 8:24] = 255
    sharp[8:24, 8:24:1] = 0
    ranker = CompositeRanker({"edge": 1.0, "exposure": 0.5})
    idx, scores = ranker.pick_best_index([blur, sharp], prompt="test", device="cpu")
    assert idx == 1 and scores[1] > scores[0]


def test_self_correct_policy_threshold() -> None:
    pol = SelfCorrectPolicy(SelfCorrectConfig(align_threshold=0.4))
    assert pol.needs_correction(0.2)
    assert not pol.needs_correction(0.9)


def test_build_index_missing_file() -> None:
    idx = build_tfidf_index_from_jsonl("/nonexistent/path.jsonl")
    assert idx.n_docs == 0


# ==================================================================================================
# test_superior_waves: Superior stack wave tests (torch-free waves 3-4, merged).
# ==================================================================================================

# ---- Wave 3: superior stack tests. ----


def test_blended_reward() -> None:
    assert blended_reward(0.8, 0.6, vit_weight=0.5) == 0.7


def test_mine_vit_pairs_without_vit_scores() -> None:
    rows = [
        {"case": "c", "prompt": "cat", "output": "a.png", "composite": 0.9},
        {"case": "c", "prompt": "cat", "output": "b.png", "composite": 0.4},
    ]

    def _fake_score(path, prompt, **kw):
        return (0.5, 0.9 if "a.png" in str(path) else 0.3)

    import utils.agentic.quality_loop as vm

    orig = vm.score_image_vit
    vm.score_image_vit = _fake_score
    try:
        cfg = ViTMineConfig(vit_ckpt="fake.pt", min_margin=0.1)
        pairs = mine_vit_preference_pairs(rows, cfg)
        assert len(pairs) >= 1
        assert pairs[0]["source"] == "vit_mining"
    finally:
        vm.score_image_vit = orig


def test_expand_prompt_heuristic() -> None:
    out = expand_prompt_heuristic("a red fox in snow")
    assert "detail" in out.lower() or "lighting" in out.lower()


def test_build_auto_improve_argv() -> None:
    cfg = AutoImproveConfig(
        base_ckpt="base.pt",
        vit_ckpt="vit/best.pt",
        local_rag_jsonl="facts.jsonl",
        model_soup=True,
    )
    argv = build_auto_improve_argv(cfg)
    assert "--vit-ckpt" in argv
    assert "--local-rag-jsonl" in argv
    assert "--model-soup" in argv
    assert "superior_composite" in argv


def test_benchmark_vit_mine_path(tmp_path: Path) -> None:
    from scripts.tools.bench import _write_preference_jsonl

    rows = [
        {"case": "c", "prompt": "x", "output": "w.png", "composite": 0.9},
        {"case": "c", "prompt": "x", "output": "l.png", "composite": 0.3},
    ]
    import utils.agentic.quality_loop as vm

    orig = vm.mine_vit_preference_pairs
    vm.mine_vit_preference_pairs = lambda r, c: [
        {"win_image_path": "w.png", "lose_image_path": "l.png", "caption": "x"}
    ]
    try:
        n = _write_preference_jsonl(
            tmp_path / "p.jsonl",
            rows,
            min_margin=0.05,
            max_pairs_per_case=1,
            vit_ckpt="v.pt",
            vit_mine=True,
        )
        assert n == 1
    finally:
        vm.mine_vit_preference_pairs = orig


# ---- Wave 4: superior stack tests. ----


def test_tags_from_benchmark_row_blur() -> None:
    tags = tags_from_benchmark_row({"composite": 0.4, "edge_sharpness": 50.0})
    assert "low_composite" in tags
    assert "blur" in tags


def test_mine_hard_negatives() -> None:
    rows = [
        {"composite": 0.3, "ocr_match": 0.2, "expected_text": "HELLO"},
        {"composite": 0.25, "edge_sharpness": 30.0},
    ]
    bundle = mine_hard_negatives(rows)
    suffix = bundle.negative_suffix.lower()
    assert "blurry" in suffix or "text" in suffix


def test_eval_report_markdown(tmp_path: Path) -> None:
    (tmp_path / "leaderboard.json").write_text(
        json.dumps([{"model": "base", "mean_composite": 0.8, "std_composite": 0.1, "robust_score": 0.75, "cases": 5}]),
        encoding="utf-8",
    )
    (tmp_path / "results.json").write_text(
        json.dumps([{"case": "c1", "composite": 0.4, "edge_sharpness": 40.0}]),
        encoding="utf-8",
    )
    md = build_markdown_report(tmp_path)
    assert "Leaderboard" in md
    assert "base" in md


def test_flywheel_dry_run() -> None:
    plan = FlywheelPlan(
        base_ckpt="fake.pt",
        work_dir="tmp_flywheel_test",
        skip_curate=True,
        defaults=SuperiorStackDefaults(auto_loop_iterations=1),
    )
    summary = run_flywheel(plan, dry_run=True)
    assert summary.get("status") == "ok"


def test_superior_preset_exists() -> None:
    from config.presets import PRESETS

    assert "superior" in PRESETS
    assert PRESETS["superior"].cfg_scale == 7.0


# ==================================================================================================
# test_superior_waves_torch: Superior stack wave tests (torch-dependent waves 5-12, merged).
# ==================================================================================================

# ---- Wave 5: research-backed DPO, feature cache, online reward. ----


def test_timestep_dpo_weight_high_noise() -> None:
    t = torch.tensor([0, 500, 999], dtype=torch.long)
    w = timestep_dpo_weight(t, 1000, mode="high_noise", power=0.5)
    assert w[2] > w[0]


def test_safeguard_shrinks_loser_margin() -> None:
    win = torch.tensor([2.0, 1.0])
    lose = torch.tensor([0.5, 0.4])
    lw, ll = safeguard_dpo_margins(win, lose, strength=0.85)
    assert (ll > lose).all()
    assert (lw == win).all()


def test_safeguarded_dpo_with_weights() -> None:
    win = torch.tensor([1.0, 0.8])
    lose = torch.tensor([0.2, 0.1])
    rw = torch.tensor([1.0, 0.9])
    rl = torch.tensor([0.8, 0.7])
    tw = torch.tensor([1.0, 2.0])
    loss = safeguarded_dpo_preference_loss(win, lose, rw, rl, beta=10.0, safeguard_strength=0.85, timestep_weights=tw)
    assert torch.isfinite(loss)


def test_feature_cache_reuse() -> None:
    pol = FeatureCachePolicy(FeatureCacheConfig(delta_threshold=0.5, max_reuse_streak=2))
    x = torch.zeros(1, 4, 4, 4)
    pred = torch.ones(1, 4, 4, 4)
    pol.note_fresh(x, pred)
    x2 = x + 0.01
    assert pol.should_skip_forward(x2)
    cached = pol.cached_prediction()
    assert torch.allclose(cached, pred)


def test_online_reward_scores() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[20:40, 20:40] = 200
    s = CompositeRanker({"ai_look": 0.5, "art": 0.5}).score_one(rgb, prompt="", device="cpu")
    assert 0.0 <= s <= 1.0


def test_inference_argv_fdg() -> None:
    cfg = SuperiorInferenceConfig(fdg_low=0.5, fdg_high=1.5)
    argv = build_superior_sample_argv(ckpt="m.pt", prompt="cat", out="o.png", config=cfg)
    assert argv[argv.index("--fdg-high") + 1] == "1.5"


# ---- Wave 6: block cache, flow-GRPO, consistency/LADD distill, hard-negative flywheel. ----


def test_block_cache_skip_and_apply() -> None:
    cache = BlockDiTCache(BlockCacheConfig(rel_l1_threshold=0.5, recompute_every=99))
    fp1 = torch.tensor([1.0, 2.0, 3.0])
    fp2 = torch.tensor([1.01, 2.01, 3.01])
    cache.begin_forward(fp1)
    x = torch.zeros(1, 4, 2, 2)
    cache.note_block(0, x, x + 1.0)
    cache.begin_forward(fp2)
    assert cache.should_skip_block(0)
    out = cache.apply_residual(x, 0)
    assert torch.allclose(out, x + 1.0)


def test_grpo_advantages_zero_mean() -> None:
    r = torch.tensor([0.2, 0.5, 0.9, 0.3])
    adv = group_relative_advantages(r)
    assert abs(float(adv.mean())) < 0.01


def test_grpo_weighted_loss() -> None:
    loss = torch.tensor([1.0, 2.0, 3.0])
    adv = torch.tensor([1.0, 0.0, -1.0])
    w = grpo_weighted_loss(loss, adv)
    assert torch.isfinite(w)


def test_reference_kl_penalty() -> None:
    a = torch.randn(1, 4, 4, 4)
    b = a + 0.1
    p = reference_kl_penalty(a, b, coef=0.1)
    assert float(p) > 0.0


def test_hard_negative_benchmark_args() -> None:
    bundle = mine_hard_negatives([{"composite": 0.3, "edge_sharpness": 20.0}])
    args = benchmark_sample_args_for_negatives(bundle)
    assert "--negative-prompt" in args
    assert args[1]


# ---- Wave 7: TaylorSeer, DenseGRPO, LCM sampling hooks. ----


def test_taylor_forecast_linear() -> None:
    h0 = torch.ones(2, 2)
    h1 = h0 + 0.5
    pred = taylor_forecast_tensor([h0, h1], steps_since_anchor=2, interval=4, max_order=1)
    assert pred.shape == h0.shape
    assert float(pred.mean()) > float(h1.mean())


def test_taylor_block_cache_apply() -> None:
    cache = TaylorBlockCache(TaylorCacheConfig(rel_l1_threshold=0.5, recompute_every=8))
    fp = torch.tensor([1.0, 2.0])
    cache.begin_forward(fp)
    x = torch.zeros(1, 4, 2, 2)
    cache.note_block(0, x, x + 1.0)
    cache.begin_forward(fp + 0.001)
    cache.begin_forward(fp + 0.002)
    out = cache.apply_residual(x, 0)
    assert out.shape == x.shape


def test_step_advantages_from_gains() -> None:
    adv = step_advantages_from_gains([0.1, 0.5, 0.2, 0.3])
    assert adv.numel() == 4


def test_flow_x0_estimate() -> None:
    x_t = torch.tensor([[[[1.0]]]])
    v = torch.tensor([[[[0.5]]]])
    t = torch.tensor([0.5])
    x0 = estimate_x0_from_xt_flow(x_t, v, t)
    assert float(x0.item()) == 0.75


# ---- Wave 8: Flash-GRPO, BranchGRPO. ----


def test_sde_lambda_peaks_mid() -> None:
    mid = float(sde_discretization_lambda(0.5))
    edge = float(sde_discretization_lambda(0.02))
    assert mid > edge


def test_iso_temporal_advantages() -> None:
    rewards = torch.tensor([0.9, 0.8, 0.2, 0.1])
    t_idx = torch.tensor([10, 10, 20, 20])
    adv = iso_temporal_group_advantages(rewards, t_idx, num_timesteps=50)
    assert adv[0] > adv[1]
    assert adv[2] > adv[3]


def test_rectify_policy_gradient() -> None:
    loss = torch.tensor(1.0)
    rect = rectify_policy_gradient(loss, 0.5)
    assert float(rect) > float(loss)


def test_branch_paths() -> None:
    paths = enumerate_branch_paths(20, branch_factor=2, split_fractions=(0.35, 0.65))
    assert len(paths) == 4


def test_branch_config_defaults() -> None:
    cfg = BranchGRPOConfig()
    assert cfg.branch_factor >= 2


# ---- Wave 9: QSilk micrograin. ----


def test_qsilk_stabilize_bounded() -> None:
    x = torch.randn(1, 4, 16, 16) * 5.0
    y = qsilk_micrograin_stabilize(x, detail_amount=0.1)
    assert y.abs().max() <= x.abs().max() + 0.5


# ---- Wave 10: TP-GRPO, DyDiT dynamic width, linear attention. ----


def test_incremental_rewards() -> None:
    inc = incremental_rewards_from_trajectory([0.2, 0.5, 0.9])
    assert inc == [0.2, 0.3, 0.4]


def test_turning_point_detection() -> None:
    inc = [0.1, -0.2, -0.1, 0.3]
    tps = detect_turning_point_indices(inc)
    assert 0 in tps


def test_tp_grpo_weights_boost_turning_point() -> None:
    traj = [0.1, 0.3, 0.55, 0.5, 0.9]
    w = tp_grpo_step_weights(traj, 0.9, config=TurningPointGRPOConfig(long_term_weight=0.5))
    assert len(w) == len(traj)
    assert max(w) > min(w)


def test_timestep_dynamic_width_monotone() -> None:
    early = timestep_dynamic_width(0.0, early=0.8, late=1.0)
    late = timestep_dynamic_width(1.0, early=0.8, late=1.0)
    assert early < late


def test_apply_dynamic_width() -> None:
    out = torch.ones(1, 4, 2, 2)
    scaled = apply_dynamic_width(out, 0.0, early=0.5)
    assert float(scaled.mean()) == 0.5


def test_spatial_importance_shape() -> None:
    x = torch.randn(2, 4, 8, 8)
    imp = spatial_token_importance(x)
    assert imp.shape == (2, 1, 8, 8)


def test_linear_attention_runs() -> None:
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    out = linear_attention(q, k, v)
    assert out.shape == v.shape


def test_hybrid_attention_blend() -> None:
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    out = hybrid_attention_fraction(q, k, v, linear_frac=0.5)
    assert out.shape == v.shape


# ---- Wave 11: GRPO-Guard, CFG-Rejection, branch rollout. ----


def test_grpo_guard_weights() -> None:
    adv = torch.tensor([1.0, -1.0, 0.2])
    normed = ratio_norm_advantages(adv)
    assert normed.numel() == 3


def test_grpo_guard_loss() -> None:
    loss = torch.tensor([0.5, 0.8])
    adv = torch.tensor([1.0, -0.5])
    t = torch.tensor([0.3, 0.7])
    total = grpo_guard_weighted_loss(loss, adv, t, config=GRPOGuardConfig())
    assert float(total.item()) > 0.0


def test_cfg_rejection_tracker() -> None:
    tr = CFGRejectionTracker(tau_steps=2)
    c = torch.randn(1, 4, 4, 4)
    u = torch.zeros_like(c)
    tr.note(c, u)
    tr.note(c * 2, u)
    assert tr.accumulated_early_score() > 0.0


def test_pick_best_candidate() -> None:
    assert pick_best_candidate_index([0.9, 0.2, 0.5]) == 1


def test_branch_rollout_paths() -> None:
    calls: list[int] = []

    def _roll() -> torch.Tensor:
        calls.append(1)
        return torch.zeros(1, 2, 2, 2)

    outs = branch_rollout_flow_samples(_roll, num_paths=4, steps=8, base_seed=0)
    assert len(outs) == 4
    assert len(calls) == 4
    assert len(enumerate_branch_paths(8)) >= 1


# ---- Wave 12: DBCache fingerprint, guidance probe. ----


def test_guidance_probe_rerank() -> None:
    probe = GuidanceProbe(tau_steps=2)
    c = torch.randn(2, 4, 4, 4)
    u = torch.zeros_like(c)
    probe.note(c, u, step=0)
    probe.note(c * 2, u, step=1)
    order = probe.rerank_indices()
    assert len(order) == 2


def test_block_cache_cfg_split_fingerprint() -> None:
    cache = BlockDiTCache()
    t = torch.randn(4, 256)
    x = torch.randn(4, 16, 64)
    fp_full = cache.fingerprint_from_tensors(t, x, cfg_split=False)
    fp_split = cache.fingerprint_from_tensors(t, x, cfg_split=True)
    assert fp_split.shape == fp_full.shape
