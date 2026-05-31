"""Visual Brain: scene brief, planner, dry-run orchestrator."""

from __future__ import annotations

from pathlib import Path

from research.agi_image.planning.generation_plan import GenerationStepKind
from utils.agentic.planner import plan_visual_brain
from utils.brain.scene_brief import SceneBrief, SceneElement, prompt_coverage_score, synthesize_scene_brief
from utils.brain.visual_brain import VisualBrain, VisualBrainConfig
from utils.generation.image_dissection import parse_part_requests


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
