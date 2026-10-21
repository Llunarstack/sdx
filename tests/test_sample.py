"""Tests for OCR repair argv forwarding.

Also holds what used to be: test_sample_cli_passthrough, test_sample_edit_runner, test_sample_features, test_sample_lcm_flow, test_sample_orchestration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from utils.generation import sample_main as sm
from utils.generation.comfy_export import sample_args_to_comfy_workflow
from utils.generation.sample_cli_parser import build_sample_parser
from utils.generation.sample_cli_passthrough import REPAIR_DESTS, append_sample_repair_passthrough, sample_argv
from utils.generation.sample_conditioning import _control_kwargs
from utils.generation.sample_edit_runner import (
    build_sample_command,
    resolve_repo_root,
    run_edit_with_pillow,
)
from utils.generation.sample_features import (
    append_benchmark_history,
    export_adherence_heatmap,
    load_character_session,
    save_character_session,
)
from utils.generation.sample_helpers import SAMPLE_PY
from utils.generation.sample_output import _capture_finish, save_targets
from utils.generation.sample_prompt import PromptPlan, compose_prompt_additions, compose_user_prompt
from utils.generation.sample_sampling import _injects_noise, _with_batch, resolve_inference_amp
from utils.generation.sample_setup import lcm_should_enable_flow_sample

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


@patch("utils.generation.sample_edit_runner.run_sample_inference")
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


@patch("utils.generation.sample_edit_runner.run_sample_inference")
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
    from utils.generation.sample_features import _apply_character_session

    args = SimpleNamespace(
        character_session="",
        prompt="base",
        dissect_refs="",
        negative_prompt="",
    )
    # inline session file via monkeypatch path — use tmp_path
    sess = {"prompt_additions": "blue cape", "reference_images": ["ref.png"], "negative_prompt": "blur"}
    import utils.generation.sample_features as sf

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

    monkeypatch.setattr("utils.generation.sample_features.run_auto_refine_candidates", fake_refine)
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
