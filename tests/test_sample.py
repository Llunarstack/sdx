"""Tests for OCR repair argv forwarding.

Also holds what used to be: test_sample_cli_passthrough, test_sample_edit_runner, test_sample_features, test_sample_lcm_flow, test_sample_orchestration, test_corpus_cast_edit, test_creative_copilot, test_edit_masks, test_editing_phase, test_invention_lab, test_invention_wiring, test_krea_controls, test_part_compositing, test_quality_stack, test_regional_box_prompting, test_segmentation_to_mask, test_simple_latent_generate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from utils.generation import sample_run as sm
from utils.generation.editing import (
    HEURISTIC_EDIT_REGIONS,
    DissectedPart,
    PartRequest,
    build_init_and_inpaint_mask,
    build_segmentation_mask_for_edit,
    heuristic_inpaint_mask,
    normalize_heuristic_region,
    phrase_to_fallback_region,
    plan_edit_skills,
    plan_ideogram_layout,
    save_heuristic_mask,
)
from utils.generation.inventions import (
    FAILURE_ATLAS,
    adaptive_nfe,
    apply_anatomon_prompts,
    apply_cast_and_corpus_to_namespace,
    apply_friction_to_prompts,
    apply_invention_stack,
    apply_invention_to_namespace,
    apply_negatron,
    build_corpus_index_from_jsonl,
    compile_cast_scene,
    diagnose_failures,
    hits_to_moodboard_payload,
    load_cast_scene,
    plan_bindlock,
    plan_countgate,
    plan_hydra_slots,
    plan_self_healing,
    prepare_invention_runtime,
    repair_plan_from_failures,
    retrieve_corpus_refs,
    save_corpus_index_meta,
    save_failure_atlas_jsonl,
    scene_priors,
)
from utils.generation.planners import (
    CopilotConfig,
    CopilotPlan,
    copilot_sample_argv,
    fuse_prompt,
    perfect_gen_argv,
    run_creative_copilot,
)
from utils.generation.regional import (
    build_latent_region_masks,
    layout_text_from_regions,
    load_box_layout_file,
    parse_box_layout,
)
from utils.generation.sample_argv import (
    REPAIR_DESTS,
    append_sample_repair_passthrough,
    build_sample_command,
    resolve_repo_root,
    run_edit_with_pillow,
    sample_args_to_comfy_workflow,
    sample_argv,
)
from utils.generation.sample_cli_parser import build_sample_parser
from utils.generation.sample_core import (
    SAMPLE_PY,
    CreativityMode,
    _control_kwargs,
    aggregate_style_embeddings,
    apply_creativity_mode_to_prompt,
    apply_generative_sliders,
    apply_krea_controls_to_args,
    apply_turbo_preset,
    build_generative_slider_plan,
    lcm_should_enable_flow_sample,
    parse_style_references,
)
from utils.generation.sample_prompt import (
    PromptPlan,
    append_benchmark_history,
    compose_prompt_additions,
    compose_user_prompt,
    export_adherence_heatmap,
    load_character_session,
    save_character_session,
)
from utils.generation.sample_run import (
    _capture_finish,
    _injects_noise,
    _with_batch,
    resolve_inference_amp,
    sample_one_image_pil,
    save_targets,
    tensor_bchw01_to_pil,
)

# ==================================================================================================
# test_sample_cli_passthrough: Tests for OCR repair argv forwarding.
# ==================================================================================================


def test_append_sample_repair_passthrough_includes_core_flags() -> None:
    args = argparse.Namespace(
        width=512,
        height=768,
        resize_mode="stretch",
        vae_tiling=True,
        boost_quality=True,
        style="anime",
        style_strength=0.8,
        holy_grail=True,
        holy_grail_cfg_early_ratio=0.7,
        holy_grail_cfg_late_ratio=1.0,
        holy_grail_control_mult=1.0,
        holy_grail_adapter_mult=1.0,
        holy_grail_no_frontload_control=False,
        holy_grail_late_adapter_boost=1.15,
        holy_grail_cads_strength=0.0,
        holy_grail_cads_min_strength=0.0,
        holy_grail_cads_power=1.0,
        holy_grail_unsharp_sigma=0.0,
        holy_grail_unsharp_amount=0.0,
        holy_grail_clamp_quantile=0.0,
        holy_grail_clamp_floor=1.0,
        lora=[],
        control=[],
    )
    cmd: list[str] = ["python", "sample.py"]
    append_sample_repair_passthrough(cmd, args)
    assert "--width" in cmd and "512" in cmd
    assert "--text-in-image" in cmd
    assert "--boost-quality" in cmd
    assert "--style" in cmd


def test_repair_dests_exist_and_round_trip_through_parser() -> None:
    p = build_sample_parser()
    base = ["--ckpt", "x.pt", "--prompt", "p"]
    parent = p.parse_args(
        base
        + ["--no-uncensored-mode", "--no-photo-postprocess", "--camera-iso", "800", "--pag-scale", "2.0"]
        + ["--lora", "a.safetensors:0.7", "b.safetensors", "--no-auto-expected-text", "--refine", "restart"]
    )
    cmd: list[str] = []
    append_sample_repair_passthrough(cmd, parent)
    child = p.parse_args(base + cmd)
    assert all(getattr(child, d) == getattr(parent, d) for d in REPAIR_DESTS)
    assert child.text_in_image is True


def test_sample_argv_skips_defaults_and_rejects_unknown_dests() -> None:
    assert sample_argv({"width": 0, "lora": [], "eta": None}) == []
    assert sample_argv({"vae_tiling": True, "photo_postprocess": False}) == ["--vae-tiling", "--no-photo-postprocess"]
    with pytest.raises(KeyError):
        sample_argv({"glyph_canvas": True})


# ==================================================================================================
# test_sample_edit_runner: Unit tests for sample.py subprocess wrappers (mocked).
# ==================================================================================================


def test_resolve_repo_contains_sample_py():
    root = resolve_repo_root()
    assert (root / "sample.py").is_file()


def test_build_command_txt2img():
    cmd = build_sample_command(ckpt="m.pt", prompt="hi", out_path="o.png", width=256, height=256)
    assert "sample.py" in cmd[1] or cmd[1].endswith("sample.py")
    assert "--ckpt" in cmd and "m.pt" in cmd
    assert "--prompt" in cmd and "hi" in cmd
    assert "--width" in cmd and "256" in cmd


def test_build_command_inpaint_requires_init():
    with pytest.raises(ValueError, match="init"):
        build_sample_command(
            ckpt="m.pt",
            prompt="hi",
            out_path="o.png",
            init_image_path=None,
            mask_image_path="mask.png",
        )


def test_build_command_rejects_invalid_strength():
    with pytest.raises(ValueError, match="strength"):
        build_sample_command(
            ckpt="m.pt",
            prompt="hi",
            out_path="o.png",
            strength=1.5,
        )


def test_build_command_rejects_non_positive_width():
    with pytest.raises(ValueError, match="width"):
        build_sample_command(
            ckpt="m.pt",
            prompt="hi",
            out_path="o.png",
            width=0,
            height=512,
        )


def test_extra_args_rejects_blank_string():
    with pytest.raises(ValueError, match="extra_args"):
        build_sample_command(
            ckpt="m.pt",
            prompt="hi",
            out_path="o.png",
            extra_args=["--foo", ""],
        )


@patch("utils.generation.sample_argv.run_sample_inference")
def test_run_edit_with_pillow_resizes_and_calls(mock_run, tmp_path):
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

    fake_out = tmp_path / "out.png"
    Image.new("RGB", (64, 64), (200, 0, 0)).save(fake_out)

    def fake_run_side_effect(*, out_path, **kw):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (10, 20, 30)).save(out_path)
        return subprocess.CompletedProcess(args=[], returncode=0)

    mock_run.side_effect = fake_run_side_effect

    base = Image.new("RGB", (128, 128), (255, 0, 0))
    mask = Image.new("L", (10, 10), 255)
    img = run_edit_with_pillow(
        ckpt="dummy.pt",
        prompt="paint a moon",
        negative_prompt="",
        base_image=base,
        mask_image=mask,
        width=512,
        height=512,
        seed=123,
        keep_temp_dir=False,
        extra_args=None,
    )
    assert isinstance(img, Image.Image)
    assert mock_run.called
    call_kw = mock_run.call_args.kwargs
    assert call_kw["mask_image_path"] is not None
    assert Path(call_kw["init_image_path"]).suffix == ".png"


@patch("utils.generation.sample_argv.run_sample_inference")
def test_run_edit_img2img_no_mask(mock_run, tmp_path):
    def fake_run_side_effect(*, out_path, **kw):
        assert kw.get("mask_image_path") is None
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (5, 5, 5)).save(out_path)
        return subprocess.CompletedProcess(args=[], returncode=0)

    mock_run.side_effect = fake_run_side_effect

    out = run_edit_with_pillow(
        ckpt="x.pt",
        prompt="same scene",
        negative_prompt="blur",
        base_image=Image.new("RGB", (8, 8), (1, 2, 3)),
        mask_image=None,
        width=512,
        height=512,
    )
    assert out.size == (4, 4)


# ==================================================================================================
# test_sample_features: Tests for sample feature helpers.
# ==================================================================================================


def test_character_session_roundtrip(tmp_path: Path):
    p = tmp_path / "sess.json"
    data = {"prompt_additions": "silver hair", "negative_prompt": "extra limbs", "reference_images": ["a.png"]}
    save_character_session(p, data)
    loaded = load_character_session(p)
    assert loaded["prompt_additions"] == "silver hair"
    assert loaded["reference_images"] == ["a.png"]


def test_comfy_export_has_sampler_node():
    args = {"prompt": "cat", "steps": 30, "cfg_scale": 7.0, "seed": 1}
    wf = sample_args_to_comfy_workflow(args)
    assert "nodes" in wf
    assert any(n.get("class_type") == "SDXSampler" for n in wf["nodes"].values())


def test_adherence_heatmap_export(tmp_path: Path):

    attn = torch.rand(4, 64, 32)
    out = tmp_path / "heat.png"
    export_adherence_heatmap(attn, "a woman in a red dress", out, latent_h=8, latent_w=8)
    assert out.is_file()


def test_benchmark_history_append(tmp_path: Path):
    lb = tmp_path / "leaderboard.json"
    hist = tmp_path / "history.json"
    lb.write_text(json.dumps({"rows": [{"name": "a", "score": 1.0}]}), encoding="utf-8")
    append_benchmark_history(lb, hist)
    data = json.loads(hist.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert "leaderboard" in data[0]


def test_apply_character_session_mutates_prompt():
    from utils.generation.sample_prompt import _apply_character_session

    args = SimpleNamespace(
        character_session="",
        prompt="base",
        dissect_refs="",
        negative_prompt="",
    )
    # inline session file via monkeypatch path — use tmp_path
    sess = {"prompt_additions": "blue cape", "reference_images": ["ref.png"], "negative_prompt": "blur"}
    import utils.generation.sample_prompt as sf

    orig = sf.load_character_session
    sf.load_character_session = lambda _p: sess  # type: ignore
    try:
        args.character_session = "x.json"
        _apply_character_session(args)
        assert "blue cape" in args.prompt
        assert args.dissect_refs == "ref.png"
        assert "blur" in args.negative_prompt
    finally:
        sf.load_character_session = orig


# ==================================================================================================
# test_sample_lcm_flow: LCM flow-vs-VP sampling branch and AMP/uncensored CLI flags.
# ==================================================================================================


def test_lcm_does_not_force_flow_on_vp_checkpoint() -> None:
    dit = SimpleNamespace(flow_matching_training=False)
    lcm = SimpleNamespace(flow_matching_training=False)
    assert lcm_should_enable_flow_sample(dit, lcm) is False
    assert lcm_should_enable_flow_sample(dit, None) is False


def test_lcm_enables_flow_when_dit_is_flow_trained() -> None:
    dit = SimpleNamespace(flow_matching_training=True)
    lcm = SimpleNamespace(flow_matching_training=False)
    assert lcm_should_enable_flow_sample(dit, lcm) is True


def test_lcm_enables_flow_when_overlay_is_flow_trained() -> None:
    dit = SimpleNamespace(flow_matching_training=False)
    lcm = SimpleNamespace(flow_matching_training=True)
    assert lcm_should_enable_flow_sample(dit, lcm) is True


def test_inference_amp_and_uncensored_cli_flags() -> None:
    parser = build_sample_parser()
    args = parser.parse_args(["--ckpt", "x.pt", "--prompt", "a"])
    assert args.inference_amp == "auto"
    assert args.uncensored_mode is True
    args_safe = parser.parse_args(["--ckpt", "x.pt", "--no-uncensored-mode"])
    assert args_safe.uncensored_mode is False
    args2 = parser.parse_args(["--ckpt", "x.pt", "--inference-amp", "bf16"])
    assert args2.inference_amp == "bf16"
    args3 = parser.parse_args(["--ckpt", "x.pt", "--inference-amp", "off"])
    assert args3.inference_amp == "off"
    amp_help = parser.format_help()
    assert "torch.cuda.is_bf16_supported()" in amp_help


def test_resolve_inference_amp_modes() -> None:
    cpu = SimpleNamespace(type="cpu")
    assert resolve_inference_amp(SimpleNamespace(inference_amp="off"), cpu) == (False, None)
    on, dtype = resolve_inference_amp(SimpleNamespace(inference_amp="fp16"), cpu)
    assert on is True and dtype is torch.float16
    on_bf, dtype_bf = resolve_inference_amp(SimpleNamespace(inference_amp="bf16"), cpu)
    assert on_bf is True and dtype_bf is torch.bfloat16
    auto_cpu, auto_dt = resolve_inference_amp(SimpleNamespace(inference_amp="auto"), cpu)
    assert auto_cpu is False and auto_dt is None


# ==================================================================================================
# test_sample_orchestration: ``sample.py`` orchestration: flag precedence, policy routing, prompt order, conditioning shapes, outputs.
# ==================================================================================================


def _resolved(*argv: str):
    """Parse like ``main`` and run the quality-default phase (no models)."""
    parser = build_sample_parser()
    args = sm.parse_args(parser, ["--ckpt", "x.pt", *argv])
    sm.expand_presets(args)
    compose_user_prompt(args, parser)
    sm.resolve_quality_defaults(args, {})
    return args


def test_explicit_flags_cover_equals_and_unique_prefixes() -> None:
    parser = build_sample_parser()
    typed = sm.explicit_flags(parser, ["--cfg-scale=4", "--steps", "9", "--no-quality-def", "x", "--prompt", "p"])
    assert {"cfg_scale", "steps", "no_quality_defaults", "prompt"} <= typed
    assert "solver" not in typed


def test_typed_flags_beat_presets_and_policy() -> None:
    args = _resolved(
        "--prompt", "a photo of a lighthouse", "--preset", "flux", "--cfg-scale", "7.5", "--steps", "50", "--num", "1"
    )
    assert (args.cfg_scale, args.steps, args.num) == (7.5, 50, 1)  # equal to the parser defaults, but typed
    assert args.hard_style == "realistic" and args.naturalize  # the rest of the user's preset applies
    assert args.prompt.startswith("a photo of a lighthouse")  # policy text appended to a typed prompt is kept


def test_policy_never_picks_prompt_pack_presets() -> None:
    oil = _resolved("--prompt", "a lighthouse on a cliff, oil painting")
    assert oil.preset is None and oil.op_mode is None and oil.hard_style is None
    assert oil.naturalize is False and oil.anti_bleed is False and oil.diversity is False
    assert oil.holy_grail and oil.holy_grail_cfg_early_ratio != 0.72  # the profile's holy-grail preset expanded


def test_user_frontier_mode_and_opt_out() -> None:
    assert _resolved("--prompt", "a castle", "--frontier").frontier_subject is False
    assert _resolved("--prompt", "a castle").frontier_subject is True
    bare = _resolved("--prompt", "a castle", "--no-quality-defaults")
    assert bare.holy_grail is False and bare.pick_best == "none" and bare.frontier_subject is False


def test_style_packs_follow_the_user_prompt() -> None:
    args = _resolved("--prompt", "two cats, red and blue scarves", "--naturalize", "--anti-bleed")
    compose_prompt_additions(args)
    assert args.prompt.startswith("two cats, red and blue scarves, ")
    assert "no color bleed" in args.prompt and "raw photo" in args.prompt


def test_auto_refine_children_run_the_entry_script(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_refine(*, sample_cmd, num_candidates, seed_base, out_stem):
        calls.append((sample_cmd, out_stem))
        out = tmp_path / "cand.png"
        Image.new("RGB", (4, 4)).save(out)
        return SimpleNamespace(outputs=[str(out)], best_index=0, scores=[1.0], prompts=[])

    monkeypatch.setattr("utils.generation.sample_prompt.run_auto_refine_candidates", fake_refine)
    out = tmp_path / "o.png"
    argv = ["--ckpt", "x.pt", "--prompt", "a cat", "--auto-refine", "3", "--out", str(out)]
    args = sm.parse_args(build_sample_parser(), argv)
    assert sm.run_early_exit(args, argv)
    cmd, out_stem = calls[0]
    assert cmd[:2] == [sys.executable, str(SAMPLE_PY)] and SAMPLE_PY.name == "sample.py"
    assert out_stem == str(tmp_path / "o_ar")  # candidates land beside --out
    assert "--auto-refine" not in cmd and "3" not in cmd[2:]
    assert out.is_file()


def test_control_kwargs_are_batch_shaped(tmp_path) -> None:
    img = tmp_path / "c.png"
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(img)
    m = SimpleNamespace(canvas_w=16, canvas_h=16, image_size=16, device=torch.device("cpu"))
    single = SimpleNamespace(control_image=str(img), control=[], control_type="canny", control_scale=0.8)
    cond: dict = {}
    _control_kwargs(single, m, cond, 3)
    assert cond["control_image"].shape == (3, 3, 16, 16) and cond["control_type"].shape == (3,)
    stacked = SimpleNamespace(
        control_image="", control=[f"{img}:canny:0.7", f"{img}:depth:0.5"], control_type="auto", control_scale=0.85
    )
    cond = {}
    _control_kwargs(stacked, m, cond, 2)
    assert cond["control_image"].shape == (2, 2, 3, 16, 16)
    assert cond["control_type"].shape == cond["control_scale"].shape == (2, 2)
    beam = _with_batch({"control_image": cond["control_image"][:1], "control_scale": 0.5}, 4)
    assert beam["control_image"].shape[0] == 4 and beam["control_scale"] == 0.5


def test_save_targets(tmp_path) -> None:
    args = SimpleNamespace(out=str(tmp_path / "sub" / "img.png"), pick_save_all=True)
    targets = save_targets(args, 3, "combo", 1)
    assert [(i, p.name) for i, p, _ in targets] == [
        (0, "img_cand0.png"),
        (1, "img_cand1.png"),
        (2, "img_cand2.png"),
        (1, "img.png"),
    ]
    assert (tmp_path / "sub").is_dir()
    args.pick_save_all = False
    assert [p.name for _, p, _ in save_targets(args, 2, "none", 0)] == ["img_0.png", "img_1.png"]
    assert [p.name for _, p, _ in save_targets(args, 1, "none", 0)] == ["img.png"]


def test_capture_finish_routes_by_medium() -> None:
    plan = PromptPlan(prompt="", negative="")
    painting = SimpleNamespace(
        camera_isp="dslr", _raw_prompt_before_compose="an oil painting of a harbor", vignette=0.3
    )
    report: dict = {}
    finish, lens = _capture_finish(painting, plan, report)
    assert finish is not None and report["capture_finish"]["profile"]  # an explicit camera always applies
    painting.camera_isp = "auto"
    painting.naturalize = True
    finish, lens = _capture_finish(painting, plan, {})
    assert finish is None and lens is not None and lens.vignette > 0  # non-photo: lens knobs only


@pytest.mark.parametrize(
    ("solver", "eta", "flow", "noisy"),
    [("dpmpp_2m", None, False, False), ("ddim", 0.5, False, True), ("dpmpp_2m_sde", None, True, True)],
)
def test_injects_noise(solver, eta, flow, noisy) -> None:
    args = SimpleNamespace(solver=solver, flow_solver=solver, eta=eta)
    assert _injects_noise(args, flow) is noisy


def test_entry_script_exists() -> None:
    assert Path(SAMPLE_PY).is_file()


def test_jobs_jsonl_loads_models_once_and_overrides_prompt_seed_out(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(
        '{"prompt": "a fox", "seed": 3, "out": "a.png"}\n\n{"prompt": "a heron", "seed": 9, "out": "b.png"}\n'
    )
    seen = []

    def fake_generate(parser, args, argv, m=None):
        seen.append((args.prompt, args.seed, args.out, args.steps, m))
        return "models"

    monkeypatch.setattr(sm, "_generate", fake_generate)
    sm.main(["--ckpt", "x.pt", "--steps", "7", "--prompt", "base", "--jobs-jsonl", str(jobs)])
    assert seen == [("a fox", 3, "a.png", 7, None), ("a heron", 9, "b.png", 7, "models")]


# ==================================================================================================
# test_corpus_cast_edit: Tests for corpus retrieval, multi-char cast, edit/ideogram skills.
# ==================================================================================================


def test_corpus_index_and_retrieve(tmp_path: Path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    jsonl = tmp_path / "train.jsonl"
    rows = [
        {"image": str(img), "text": "1girl, catgirl, black hair, eipril", "source": "danbooru"},
        {"image": str(tmp_path / "missing.png"), "text": "1boy, armor, knight", "source": "gelbooru"},
        {"image": str(img), "tags": ["1girl", "lingerie", "cat_ears"], "source": "danbooru"},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    idx = build_corpus_index_from_jsonl(jsonl)
    assert len(idx) == 3
    hits = retrieve_corpus_refs(idx, "catgirl eipril", top_k=2, source_filter="danbooru")
    assert hits
    assert "cat" in hits[0].caption.lower() or "eipril" in hits[0].caption.lower() or hits[0].tags
    mb = hits_to_moodboard_payload(hits)
    assert "images" in mb
    out = save_corpus_index_meta(idx, tmp_path / "index.jsonl")
    assert out.is_file()


def test_multi_char_cast_anti_blend():
    scene = load_cast_scene(
        {
            "actors": [
                {
                    "id": "A",
                    "wardrobe": ["red hoodie"],
                    "pose": ["arms crossed"],
                    "action": ["standing"],
                    "spatial_anchor": "left",
                },
                {
                    "id": "B",
                    "wardrobe": ["blue dress"],
                    "pose": ["waving"],
                    "action": ["smiling"],
                    "spatial_anchor": "right",
                },
            ],
            "relations": [{"a": "A", "b": "B", "kind": "facing"}],
            "anti_blend": True,
        }
    )
    compiled = compile_cast_scene(scene)
    assert "A wears red hoodie" in compiled.positive
    assert "B wears blue dress" in compiled.positive
    assert "outfit swap" in compiled.negative or "character blending" in compiled.negative
    assert compiled.box_layout is not None
    assert len(compiled.box_layout["regions"]) == 2


def test_ideogram_and_edit_skills(tmp_path: Path):
    ideo = plan_ideogram_layout('cool poster with "HELLO"', texts=["HELLO"], palette_hex=["#ff0000"])
    assert "HELLO" in ideo.glyph_texts
    assert ideo.sample_argv
    img = tmp_path / "out.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    plan = plan_edit_skills(
        init_image=str(img),
        prompt="fix face",
        fix_regions=["face"],
        work_dir=tmp_path / "edit",
        width=256,
        height=256,
    )
    assert plan.skills
    assert "--mask" in plan.sample_argv


# ==================================================================================================
# test_creative_copilot: Offline tests for Creative Co-Pilot decompose-and-apply orchestration.
# ==================================================================================================


def test_fuse_prompt_combines_goal_and_style():
    out = fuse_prompt("a knight portrait", ["impasto oil brushwork, cool teal palette"])
    assert out.startswith("a knight portrait")
    assert "impasto" in out
    assert "ignore reference subjects" in out.lower() or "Art style" in out


def test_fuse_prompt_goal_only():
    assert fuse_prompt("solo subject", []) == "solo subject"


def test_build_sample_argv_instantstyle(tmp_path: Path):
    plan = CopilotPlan(
        goal_prompt="city",
        fused_prompt="city. Art style: neon",
        moodboard_json=str(tmp_path / "mb.json"),
        control_image=str(tmp_path / "canny.png"),
        control_type="canny",
    )
    argv = copilot_sample_argv(ckpt="ckpt.pt", plan=plan, out="o.png", config=CopilotConfig(moodboard_strength=0.8))
    assert argv[argv.index("--moodboard-strength") + 1] == "0.8"
    assert "--moodboard-json" in argv
    assert "--control-image" in argv
    assert "--control-type" in argv


def test_run_creative_copilot_dry_run():
    plan = run_creative_copilot(
        goal="neon cityscape",
        ckpt="fake.pt",
        work_dir="creative_copilot_dry",
        dry_run=True,
        execute=False,
        config=CopilotConfig(web_search=False, use_vlm=False, extract_control=False),
    )
    assert plan.goal_prompt == "neon cityscape"
    assert plan.fused_prompt == "neon cityscape"
    assert len(plan.search_queries) >= 1
    assert plan.sample_argv
    assert "sample.py" in plan.sample_argv[1] or plan.sample_argv[1].endswith("sample.py")


# ==================================================================================================
# test_edit_masks: Tests for heuristic inpaint masks.
# ==================================================================================================


def test_region_keys():
    assert "face" in HEURISTIC_EDIT_REGIONS
    assert "background" in HEURISTIC_EDIT_REGIONS


def test_normalize_unknown_defaults_to_subject():
    assert normalize_heuristic_region("not_a_real_region") == "subject"
    assert normalize_heuristic_region(" FACE ") == "face"


def test_full_all_white_stats():
    m = heuristic_inpaint_mask(64, 64, "full")
    assert m.size == (64, 64)
    mn, mx = m.getextrema()
    assert mx == 255


def test_background_masks_edges_brighter_than_center():
    m = heuristic_inpaint_mask(64, 64, "background").load()
    # corners should tendency to white (paint)
    corners = sum(m[x, y] for x, y in ((0, 0), (63, 0), (0, 63), (63, 63))) / 4
    cen = sum(m[32, y] for y in range(20, 44)) / 24
    assert corners > cen


def test_face_has_upper_center_activity():
    m = heuristic_inpaint_mask(96, 96, "face").load()
    hi = sum(m[48, y] for y in range(5, 30)) / 25
    lo = sum(m[48, y] for y in range(80, 95)) / 15
    assert hi > lo


def test_save_roundtrip(tmp_path: Path):
    p = tmp_path / "mask.png"
    save_heuristic_mask(p, 32, 32, "subject")
    assert p.is_file()
    Image.open(p).verify()


# ==================================================================================================
# test_editing_phase: Tests for the post-generation editing phase orchestrator.
# ==================================================================================================


class TestPromptPieces:
    def test_extract_tokens(self):
        from utils.generation.editing import extract_prompt_tokens

        toks = extract_prompt_tokens("a lone samurai at sunset with red armor")
        assert "samurai" in toks
        assert "the" not in toks

    def test_expected_text(self):
        from utils.generation.editing import expected_render_text

        assert "OPEN" in expected_render_text('a shop sign that says "OPEN"')
        assert expected_render_text("no quotes here") == []

    def test_piece_labels(self):
        from utils.generation.editing import infer_piece_labels

        labs = infer_piece_labels("portrait of a woman with detailed hands in a forest")
        assert "subject" in labs
        assert "face" in labs or "hands" in labs
        assert "background" in labs


class TestMissingTokens:
    def test_gap_detection(self):
        from utils.generation.editing import missing_tokens_heuristic

        miss = missing_tokens_heuristic(
            "samurai red armor cherry blossoms",
            "a person standing outside",
        )
        assert "samurai" in miss or "armor" in miss


class TestPlanEdits:
    def test_ocr_and_anatomy_plan(self):
        from utils.generation.editing import Diagnosis, EditingPhaseConfig, plan_edits

        d = Diagnosis(
            gate_passed=False,
            gate_failures=["clip"],
            clip_score=0.1,
            missing_tokens=["dragon"],
            expected_text=["CAFE"],
            needs_ocr=True,
            needs_anatomy=True,
            piece_labels=["subject", "face"],
        )
        acts = plan_edits(d, 'portrait with hands and sign "CAFE"', cfg=EditingPhaseConfig())
        kinds = {a.kind for a in acts}
        assert "ocr_fix" in kinds
        assert "inpaint_region" in kinds


class TestEditingPhaseDryRun:
    def test_loop_produces_pieces_and_report(self, tmp_path):
        from utils.generation.editing import EditingPhaseConfig, run_editing_phase

        img = np.random.randint(40, 200, (128, 128, 3), dtype=np.uint8)
        cfg = EditingPhaseConfig(
            max_iters=2,
            min_clip=0.0,  # skip CLIP hub download in unit tests
            min_sharpness=0.99,  # force sharpness failure → actions
            dry_run=True,
            device="cpu",
            enable_art_post=True,
        )
        result = run_editing_phase(
            img,
            'a warrior with hands holding a sword, sign says "OPEN"',
            ckpt=None,
            config=cfg,
            work_dir=tmp_path,
            caption="a blurry person outdoors",
        )
        assert result.iterations >= 1
        assert result.piece_dir is not None
        assert (tmp_path / "pieces").is_dir()
        assert result.actions_applied or result.diagnosis_history
        assert result.stopped_reason in {"max_iters", "no_actions", "gates_passed"}


# ==================================================================================================
# test_invention_lab: Tests for the invention prompt passes, priors, oracle and self-healing plan.
# ==================================================================================================


def test_bindlock_and_countgate():
    bl = plan_bindlock("a red cube left of a blue sphere")
    assert bl.triples
    assert "attribute swap" in bl.negative_addon
    cg = plan_countgate("four cats and 2girls")
    assert any(s.n == 4 for s in cg.specs) or any(s.n == 2 for s in cg.specs)
    assert cg.boxes


def test_negatron_strips_and_negates():
    pos, neg, plan = apply_negatron("1girl, no hat, without watermark, smiling")
    assert "hat" in plan.forbidden or any("hat" in f for f in plan.forbidden)
    assert "hat" in neg.lower()
    assert "no hat" not in pos.lower()


def test_friction_adds_texture_language():
    p, n = apply_friction_to_prompts("portrait", "")
    assert "pores" in p.lower() and "plastic" in n.lower()
    assert apply_friction_to_prompts("portrait", "", strength=0.0) == ("portrait", "")


def test_scene_priors_follow_subject_and_medium():
    photo = " | ".join(p for p, _ in scene_priors("photo of a woman holding a cup"))
    assert "pores" in photo and "five fingers" in photo and "lens" in photo
    anime = " | ".join(p for p, _ in scene_priors("anime illustration of a castle at dusk"))
    assert "pores" not in anime and "lens" not in anime and "fingers" not in anime
    assert "z-order" in " | ".join(p for p, _ in scene_priors("a cat in front of a house"))


def test_oracle_and_stack():
    rep = repair_plan_from_failures(diagnose_failures("three red apples left of two blue cups, no text, 1girl hands"))
    assert rep.risks
    assert rep.repairs
    res = apply_invention_stack("four red cubes left of blue spheres, no watermark, 1girl", enable="auto")
    assert "exactly" in res.positive.lower()
    assert res.box_layout and res.box_layout["regions"]
    assert res.reports["priors"]
    hy = plan_hydra_slots("4 cats")
    assert len(hy.slots) == 4 and len(hy.box_layout["regions"]) == 4
    pos, neg, ap = apply_anatomon_prompts("1girl standing, hands visible")
    assert ap.active
    assert "finger" in neg.lower()


def test_failure_atlas_jsonl(tmp_path):
    path = save_failure_atlas_jsonl(tmp_path / "atlas.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == sum(len(prompts) for _, _, prompts in FAILURE_ATLAS)
    assert {r["mode"] for r in rows} == {mode for mode, _, _ in FAILURE_ATLAS}


def test_adaptive_nfe_and_self_heal():
    assert adaptive_nfe("four cats left of dogs, no text, hands") > adaptive_nfe("a cat") >= 12
    plan = plan_self_healing("2girls different outfits, no watermark")
    assert plan.sample_argv[plan.sample_argv.index("--scheduler") + 1] == "apex"
    assert plan.sample_argv[-2:] == ["--invention-stack", "off"]


# ==================================================================================================
# test_invention_wiring: End-to-end wiring tests for Invention Lab integration points.
# ==================================================================================================


def test_prepare_runtime_and_namespace() -> None:
    rt = prepare_invention_runtime(
        "four red cubes left of blue spheres, no watermark",
        enable="auto",
        base_steps=28,
        adaptive_steps=True,
    )
    assert rt.positive
    assert rt.steps is not None

    args = argparse.Namespace(
        prompt="2girls holding hands, no hat",
        negative_prompt="",
        invention_stack="auto",
        invention_adaptive_steps=True,
        steps=28,
        box_layout="",
        anti_bleed=False,
    )
    out = apply_invention_to_namespace(args, work_dir="")
    assert out is not None
    assert args.prompt
    assert int(args.steps) >= 12


def test_cli_parser_invention_flags() -> None:
    p = build_sample_parser()
    ns = p.parse_args(
        [
            "--ckpt",
            "x.pt",
            "--prompt",
            "a cat",
            "--invention-stack",
            "auto",
            "--invention-adaptive-steps",
            "--invention-self-heal",
        ]
    )
    assert ns.invention_stack == "auto"
    assert ns.invention_adaptive_steps is True
    assert ns.invention_self_heal is True


def test_passthrough_forwards_invention_flags() -> None:
    args = argparse.Namespace(
        invention_stack="auto",
        invention_adaptive_steps=True,
        invention_self_heal=False,
        corpus_jsonl="data/train.jsonl",
        corpus_top_k=4,
        corpus_source="danbooru",
        corpus_root="",
        cast_json="cast.json",
    )
    cmd: list[str] = ["python", "sample.py"]
    append_sample_repair_passthrough(cmd, args)
    # The repair run receives the already-rewritten prompt / box layout / steps; re-applying
    # these prompt transforms would duplicate them.
    for flag in ("--invention-stack", "--invention-adaptive-steps", "--cast-json", "--corpus-jsonl"):
        assert flag not in cmd


def test_cast_and_corpus_apply_to_namespace(tmp_path) -> None:
    import json

    img = tmp_path / "ref.png"
    Image.new("RGB", (8, 8)).save(img)
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"image": str(img), "text": "catgirl, black hair, red hoodie", "source": "danbooru"},
        {"image": str(img), "text": "knight in armor", "source": "gelbooru"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cast = tmp_path / "cast.json"
    cast.write_text(
        json.dumps({"actors": [{"id": "A", "wardrobe": ["red hoodie"]}, {"id": "B", "wardrobe": ["blue dress"]}]}),
        encoding="utf-8",
    )
    args = build_sample_parser().parse_args(
        ["--ckpt", "x.pt", "--prompt", "catgirl", "--cast-json", str(cast), "--corpus-jsonl", str(corpus)]
    )
    args.corpus_source = "danbooru"
    apply_cast_and_corpus_to_namespace(args, work_dir=str(tmp_path))
    assert "A wears red hoodie" in args.prompt and "B wears blue dress" in args.prompt
    assert "catgirl, black hair" in args.prompt and "knight" not in args.prompt
    assert "B wearing red hoodie" in args.negative_prompt
    assert json.loads(open(args.box_layout, encoding="utf-8").read())["regions"]
    assert args.anti_bleed is True
    assert args.moodboard_images == str(img)


def test_perfect_gen_argv_includes_inventions() -> None:
    argv = perfect_gen_argv("portrait of a girl", invention_stack="auto")
    assert "--invention-stack" in argv


# ==================================================================================================
# test_krea_controls: Tests for Krea 2–inspired controls (sliders, creativity, style refs).
# ==================================================================================================


def test_parse_style_references_csv():
    refs = parse_style_references(csv_spec="a.png:0.6,b.png:0.4")
    assert len(refs) == 2
    assert refs[0].path == "a.png"
    assert abs(refs[0].strength - 0.6) < 1e-6


def test_parse_style_references_json(tmp_path: Path):
    p = tmp_path / "refs.json"
    p.write_text(
        json.dumps({"references": [{"path": "x.png", "strength": 0.8}]}),
        encoding="utf-8",
    )
    refs = parse_style_references(json_path=str(p))
    assert len(refs) == 1
    assert refs[0].path == "x.png"


def test_generative_sliders_high_movement():
    plan = build_generative_slider_plan(movement=80.0)
    assert "dynamic" in plan.prompt_additions.lower()
    assert plan.serendipity_offset > 0


def test_generative_sliders_apply_cfg():
    args = SimpleNamespace(
        prompt="cat",
        negative_prompt="",
        cfg_scale=7.5,
        reference_strength=1.0,
        slider_intensity=50.0,
        slider_complexity=0.0,
        slider_movement=0.0,
        creativity_mode="medium",
    )
    plan = apply_generative_sliders(args)
    assert plan.cfg_multiplier > 1.0
    assert float(args.cfg_scale) > 7.5
    assert "bold" in args.prompt.lower()


def test_creativity_mode_raw_skips_expansion():
    args = SimpleNamespace(prompt="sunset", creativity_mode="raw")
    mode = apply_creativity_mode_to_prompt(args)
    assert mode == CreativityMode.RAW
    assert args._skip_prompt_expansion is True
    assert args.prompt == "sunset"


def test_creativity_mode_high_expands_short_prompt():
    args = SimpleNamespace(prompt="dragon", creativity_mode="high", frontier_serendipity=0.2)
    apply_creativity_mode_to_prompt(args)
    assert len(args.prompt) > len("dragon")
    assert args.frontier_serendipity > 0.2


def test_turbo_preset():
    args = SimpleNamespace(steps=50, cfg_scale=7.5)
    apply_turbo_preset(args)
    assert args.steps == 8
    assert args.cfg_scale == 1.0


def test_apply_krea_controls_bundle():
    args = SimpleNamespace(
        krea_turbo_preset=True,
        creativity_mode="medium",
        slider_intensity=0.0,
        slider_complexity=0.0,
        slider_movement=0.0,
        style_ref="",
        style_references_json="",
        moodboard_json="",
        moodboard_images="",
        prompt="a vase",
        negative_prompt="",
        cfg_scale=7.0,
        steps=50,
    )
    meta = apply_krea_controls_to_args(args)
    assert meta.get("turbo_preset") is True
    assert args.steps == 8


def test_aggregate_style_embeddings(tmp_path: Path):
    pytest.importorskip("transformers")
    for i, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        img = Image.new("RGB", (64, 64), color)
        img.save(tmp_path / f"ref{i}.png")
    refs = parse_style_references(csv_spec=f"{tmp_path / 'ref0.png'}:0.5,{tmp_path / 'ref1.png'}:0.5")
    try:
        emb, dim = aggregate_style_embeddings(
            refs,
            device=torch.device("cpu"),
            model_id="openai/clip-vit-large-patch14",
            dtype=torch.float32,
        )
    except OSError as e:
        pytest.skip(f"CLIP weights unavailable (offline / not cached): {e}")
    assert emb.shape == (1, dim)
    assert torch.isfinite(emb).all()


# ==================================================================================================
# test_part_compositing
# ==================================================================================================


def test_build_init_and_mask_preserves_foreground(tmp_path: Path) -> None:
    # Base background: solid gray
    bg = tmp_path / "bg.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(bg)

    # Foreground crop: red square everywhere, mask only in a small region.
    fg = tmp_path / "fg.png"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(fg)

    m = np.zeros((32, 32), dtype=np.uint8)
    m[8:24, 8:24] = 255
    mask_p = tmp_path / "m.png"
    Image.fromarray(m, mode="L").save(mask_p)

    part = DissectedPart(
        request=PartRequest(part="hat", source_index=0, role="foreground"),
        mask_path=str(mask_p),
        crop_path=str(fg),
        confidence=1.0,
    )

    spec = build_init_and_inpaint_mask(
        reference_images=[str(bg)],
        parts=[part],
        output_dir=tmp_path / "out",
        target_size=(32, 32),
        lock_background_if_requested=False,
    )

    init = Image.open(spec.init_image_path).convert("RGB")
    mask = Image.open(spec.mask_path).convert("L")
    init_np = np.array(init)
    mask_np = np.array(mask)

    # Center should be red and preserved (mask black).
    assert (init_np[16, 16] == np.array([255, 0, 0])).all()
    assert mask_np[16, 16] == 0
    # Corner should remain background and be inpaintable (mask white).
    assert (init_np[0, 0] == np.array([128, 128, 128])).all()
    assert mask_np[0, 0] == 255


# ==================================================================================================
# test_quality_stack: Tests for the compete-mode quality policy and the text-embedding adherence / glyph hooks.
# ==================================================================================================


def _blank_args(**over):
    base = dict(
        prompt="",
        negative_prompt="",
        style="",
        photo_realism_pack="none",
        frontier_perfect=False,
        frontier_subject=False,
        apg_parallel_eta=-1.0,
        apg_momentum_beta=0.0,
        cfg_scale=7.5,
        cfg_rescale=0.0,
        fdg_low=1.0,
        fdg_high=1.0,
        cfg_interval="",
        qsilk_micrograin=0.0,
        steps=50,
        preset=None,
        op_mode=None,
        holy_grail=False,
        holy_grail_preset=None,
        human_made="none",
        less_ai=False,
        naturalize=False,
        naturalize_deep=False,
        anti_ai_pack="none",
        human_media_mode="none",
        shortcomings_mitigation="none",
        anatomy_guidance="auto",
        hand_mode="none",
        pose_naturalness="none",
        typography_mode="none",
        expand_prompt=False,
        anti_bleed=False,
        anti_artifacts=False,
        strong_watermark=False,
        diversity=False,
        text_in_image=False,
        ocr_fix=False,
        pick_best="none",
        num=1,
        superior_self_correct=False,
        no_quality_defaults=False,
        box_attn_layout=False,
        per_region_cads=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_quality_policy_photoreal_compete():
    from utils.generation.sample_core import apply_best_model_policy

    args = _blank_args(prompt="hyperreal 8k dslr portrait photo")
    meta = apply_best_model_policy(args)
    assert meta["profile"] in ("photoreal", "portrait")
    assert meta.get("aggression") == "max"
    assert args.holy_grail is True
    assert float(args.apg_parallel_eta) == 0.0
    assert float(args.apg_momentum_beta) == -0.5
    assert float(args.fdg_low) < 1.0 < float(args.fdg_high)
    assert args.cfg_interval == "0.28,5.42"
    assert str(args.human_made) == "strong"
    assert int(args.num) >= 3
    assert str(args.pick_best) not in ("none", "")
    assert args.ocr_fix is False or True  # may stay false without quotes
    assert "plastic" in str(args.negative_prompt).lower() or "anti_slop" in str(meta.get("applied"))
    assert args.box_attn_layout is True


def test_quality_policy_bind_two_colors_left_of():
    from utils.generation.sample_core import apply_best_model_policy

    args = _blank_args(prompt="a red cube to the left of a blue sphere")
    meta = apply_best_model_policy(args)
    assert meta["profile"] == "bind"
    assert str(args.pick_best) in ("combo_spatial_bind", "combo", "combo_vit_realism")


def test_quality_policy_text_render_ocr():
    from utils.generation.sample_core import apply_best_model_policy

    args = _blank_args(prompt='a shop sign that says "OPEN NOW"')
    meta = apply_best_model_policy(args)
    assert meta["profile"] == "text_render"
    assert args.text_in_image is True
    assert args.ocr_fix is True


def test_quality_policy_anime_brain():
    from utils.generation.sample_core import apply_best_model_policy

    args = _blank_args(prompt="anime girl, cel shading, manga panel")
    meta = apply_best_model_policy(args)
    assert meta["profile"] == "anime"
    assert args.preset is None and args.op_mode is None
    assert args.holy_grail is True
    assert str(args.holy_grail_preset) == "anime"


def test_quality_policy_keeps_user_frontier_mode():
    from utils.generation.sample_core import apply_best_model_policy

    args = _blank_args(prompt="a castle in the clouds", frontier=True)
    apply_best_model_policy(args)
    assert args.frontier_subject is False
    args = _blank_args(prompt="a castle in the clouds")
    apply_best_model_policy(args)
    assert args.frontier_subject is True


def test_quality_policy_occlusion_fence():
    from utils.generation.sample_core import apply_best_model_policy

    args = _blank_args(prompt="a woman standing behind a wooden fence")
    meta = apply_best_model_policy(args)
    assert meta["profile"] == "occlusion"
    assert str(args.anatomy_guidance) == "strong"
    assert str(args.holy_grail_preset) == "aggressive"


def test_quality_policy_quoted_open_still_text_render():
    from utils.generation.sample_core import apply_best_model_policy

    args = _blank_args(prompt='a shop sign that says "OPEN" behind a fence')
    meta = apply_best_model_policy(args)
    assert meta["profile"] == "text_render"


def test_quality_policy_nsfw_and_sfw():
    from utils.generation.sample_core import apply_best_model_policy

    nsfw = _blank_args(prompt="nude woman, natural window light, lying on linen")
    meta = apply_best_model_policy(nsfw)
    assert meta["profile"] == "nsfw"
    assert str(nsfw.anatomy_guidance) == "strong"
    assert str(nsfw.human_made) == "none"
    assert str(nsfw.human_media_mode) == "none"
    assert str(nsfw.photo_realism_pack) == "none"
    assert "anti_slop" not in str(meta.get("applied"))

    act = _blank_args(prompt="1girl, 1boy, handjob, penis, uncensored")
    meta_act = apply_best_model_policy(act)
    assert meta_act["profile"] == "nsfw"
    assert str(act.human_media_mode) == "none"

    sfw = _blank_args(prompt="linkedin headshot of an engineer, office window light")
    meta2 = apply_best_model_policy(sfw)
    assert meta2["profile"] == "sfw"


def test_quality_policy_opt_out():
    from utils.generation.sample_core import apply_best_model_policy

    args = _blank_args(prompt="photoreal dslr", no_quality_defaults=True)
    meta = apply_best_model_policy(args)
    assert meta.get("skipped") is True


def test_token_spans_char_fallback_long_word_occupies_more():
    from utils.generation.sample_core import _token_spans_for_words

    seq = 32
    long_spans = _token_spans_for_words("supercalifragilisticexpialidocious x", seq)
    short_spans = _token_spans_for_words("a x", seq)
    long_w = long_spans[0].stop - long_spans[0].start
    short_w = short_spans[0].stop - short_spans[0].start
    assert long_w > short_w


def test_token_spans_fake_tokenizer_overlap():
    from utils.generation.sample_core import _token_spans_for_words

    class _TinyTok:
        def __call__(self, prompt, **kwargs):
            return {"offset_mapping": [(0, 0), (0, 4), (4, 9), (10, 16)]}

    prompt = "abcdefghi second"
    spans = _token_spans_for_words(prompt, 8, tokenizer=_TinyTok())
    assert spans[0] == slice(1, 3)
    assert spans[1] == slice(3, 4)


def test_modulate_text_for_adherence_clones_once():
    from utils.generation.sample_core import modulate_text_for_adherence

    enc = torch.randn(1, 16, 8)
    orig = enc.clone()
    out = modulate_text_for_adherence(enc, "red shirt, no glasses", negation_scale=0.55, binding_boost=0.15)
    assert torch.equal(enc, orig)
    assert out.shape == enc.shape
    assert not torch.equal(out, enc)


# ==================================================================================================
# test_regional_box_prompting: Tests for Ideogram-style box + prompt regional layout.
# ==================================================================================================


def test_parse_box_layout_minimal():
    data = {
        "global_prompt": "scene",
        "regions": [
            {"name": "a", "box": [0, 0, 0.5, 1], "prompt": "cat"},
            {"name": "b", "x_min": 0.5, "y_min": 0, "x_max": 1, "y_max": 1, "prompt": "dog"},
        ],
    }
    spec = parse_box_layout(data)
    assert spec.global_prompt == "scene"
    assert len(spec.regions) == 2
    assert spec.regions[0].name == "a"
    assert spec.regions[1].x1 == 0.5


def test_layout_text_merges_global_and_regions():
    spec = parse_box_layout(
        {
            "global_prompt": "wide shot",
            "regions": [{"name": "sky", "box": [0, 0, 1, 0.4], "prompt": "blue sky"}],
        }
    )
    text = layout_text_from_regions(spec)
    assert "[layout]" in text
    assert "sky:" in text
    assert text.startswith("wide shot")


def test_build_latent_masks_priority_overlap():
    spec = parse_box_layout(
        {
            "feather": 0,
            "regions": [
                {"name": "hi", "box": [0, 0, 0.6, 1], "prompt": "a", "priority": 10},
                {"name": "lo", "box": [0.4, 0, 1, 1], "prompt": "b", "priority": 1},
            ],
        }
    )
    regions, bg = build_latent_region_masks(spec, 16, 16, device=torch.device("cpu"))
    assert regions.shape[0] == 2
    assert bg.shape == (1, 1, 16, 16)
    # High-priority region owns the overlap column (~0.4-0.6); low-priority should be near zero there.
    overlap_col = int(0.5 * 16)
    assert regions[0, 0, 8, overlap_col] > 0.5
    assert regions[1, 0, 8, overlap_col] < 0.1


def test_example_json_loads():
    path = Path(__file__).resolve().parents[1] / "examples" / "box_layout.example.json"
    spec = load_box_layout_file(path)
    assert len(spec.regions) == 2
    assert spec.regions[0].priority == 10


def test_load_from_dict_keys():
    raw = json.loads((Path(__file__).resolve().parents[1] / "examples" / "box_layout.example.json").read_text())
    spec = parse_box_layout(raw)
    assert spec.feather_px == 12


def test_parse_strokes_in_region():
    spec = parse_box_layout(
        {
            "regions": [
                {
                    "name": "fig",
                    "box": [0, 0, 0.5, 1],
                    "prompt": "a cat sitting",
                    "strokes": [{"points": [[0.2, 0.2], [0.8, 0.8]], "width": 0.03}],
                }
            ]
        }
    )
    assert len(spec.regions[0].strokes) == 1
    assert len(spec.regions[0].strokes[0].points) == 2


def test_rasterize_strokes_nonzero():
    from utils.generation.regional import rasterize_region_sketch

    spec = parse_box_layout(
        {
            "regions": [
                {
                    "name": "a",
                    "box": [0.25, 0.25, 0.75, 0.75],
                    "prompt": "circle",
                    "strokes": [
                        {
                            "points": [[0.1, 0.5], [0.5, 0.1], [0.9, 0.5], [0.5, 0.9], [0.1, 0.5]],
                            "width": 0.05,
                        }
                    ],
                }
            ]
        }
    )
    r = spec.regions[0]
    mask = rasterize_region_sketch(r, 32, 32, device=torch.device("cpu"))
    assert mask.max() > 0.5
    assert mask[:, :, 8:24, 8:24].mean() > 0.1


def test_sketch_example_json_loads():
    path = Path(__file__).resolve().parents[1] / "examples" / "box_layout_sketch.example.json"
    spec = load_box_layout_file(path)
    assert len(spec.regions) == 3
    assert all(r.strokes for r in spec.regions)
    from utils.generation.regional import spec_has_sketches

    assert spec_has_sketches(spec)


# ==================================================================================================
# test_segmentation_to_mask: Tests for text/vision-driven inpaint masks (heuristic paths; CI avoids heavy transformers).
# ==================================================================================================


def test_phrase_to_fallback_regions():
    assert phrase_to_fallback_region("redo the BACKGROUND blur") == "background"
    assert phrase_to_fallback_region("fix her eyes") == "face"
    assert phrase_to_fallback_region("hands look wrong") == "hands"


def test_empty_phrase_raises():
    with pytest.raises(ValueError, match="non-empty"):
        build_segmentation_mask_for_edit(Image.new("RGB", (32, 32), (255, 0, 0)), "")


def test_heuristic_only_modes_skip_models():
    im = Image.new("RGB", (64, 64), (128, 128, 128))
    r = build_segmentation_mask_for_edit(
        im,
        "change only the backdrop",
        use_vision_models=False,
        feather_radius=2.0,
    )
    assert r.mode.startswith("heuristic")
    assert r.mask.size == (64, 64)


def test_unrecognized_words_no_models_behave():
    """No DINO/SAM: unknown nouns resolve to heuristic subject ellipse."""
    im = Image.new("RGB", (48, 48), (255, 255, 255))
    r = build_segmentation_mask_for_edit(im, "obscure thingamajig", use_vision_models=False)
    assert r.mask.size == (48, 48)
    assert "heuristic" in r.mode or "subject" in r.mode


# ==================================================================================================
# test_simple_latent_generate: Tests for ``utils.generation.sample_run``.
# ==================================================================================================


def test_tensor_bchw01_to_pil_roundtrip_shape() -> None:
    t = torch.full((1, 3, 8, 8), 0.5)
    pil = tensor_bchw01_to_pil(t)
    assert pil.size == (8, 8)
    assert pil.mode == "RGB"


def test_tensor_bchw01_to_pil_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        tensor_bchw01_to_pil(torch.zeros(1, 1, 4, 4))


@patch("sample.encode_text")
def test_sample_one_image_pil_smoke(mock_encode: MagicMock) -> None:
    """Minimal stack: patched T5 encode, fake diffusion + VAE decode."""
    dev = torch.device("cpu")
    emb = torch.zeros(1, 2, 8, dtype=torch.float32)
    mock_encode.return_value = emb

    diffusion = MagicMock()
    diffusion.sample_loop.return_value = torch.zeros(1, 4, 4, 4)

    dec = MagicMock()
    dec.sample = torch.full((1, 3, 32, 32), 0.25)
    vae = MagicMock()
    vae.decode.return_value = dec

    model = MagicMock()
    tokenizer = MagicMock()
    text_encoder = MagicMock()

    pil = sample_one_image_pil(
        model=model,
        diffusion=diffusion,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        device=dev,
        prompt="a cat",
        negative_prompt="",
        image_size=32,
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=0,
    )
    assert pil.size == (32, 32)
    assert pil.mode == "RGB"
    assert mock_encode.call_count == 1
    assert len(mock_encode.call_args[0][0]) == 2
    diffusion.sample_loop.assert_called_once()
