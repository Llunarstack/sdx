"""Offline tests for Creative Co-Pilot decompose-and-apply orchestration."""

from __future__ import annotations

from pathlib import Path

from utils.generation.creative_copilot import (
    CopilotConfig,
    CopilotPlan,
    build_sample_argv,
    fuse_prompt,
    run_creative_copilot,
)


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
    cfg = CopilotConfig(reference_style_mode="instantstyle", reference_strength=0.7)
    argv = build_sample_argv(ckpt="ckpt.pt", plan=plan, out="o.png", config=cfg)
    assert "--reference-style-mode" in argv
    assert "instantstyle" in argv
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
