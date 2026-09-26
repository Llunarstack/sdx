"""Tests for perfect-gen facet decompose / clarify / memory / plan glue."""

from __future__ import annotations

import json
from pathlib import Path

from utils.agentic.clarify_pose_place import (
    apply_clarify_answers,
    assume_pose_place_defaults,
    build_clarify_questions,
)
from utils.brain.memory_box import MemoryBox, load_memory_box, save_memory_box
from utils.generation.perfect_gen import PerfectGenConfig, plan_perfect_gen
from utils.prompt.facet_decompose import decompose_prompt_facets


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
    from utils.agentic.gen_interview import build_gen_interview, format_interview_for_user

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
