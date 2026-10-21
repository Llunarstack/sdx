"""Video quality presets (video_quality / videomax) and per-medium motion grammars."""

from __future__ import annotations

import pytest
from pipelines.video.motion_grammar import grammar_for_engine, grammar_for_prompt
from pipelines.video.segment_processor import (
    apply_video_quality,
    build_keyframe_sample_extras,
    build_sample_cmd_for_keyframe,
    parse_process_options,
    plan_videomax,
)
from pipelines.video.types import KeyframeEditJob


def test_prompt_cues_pick_the_medium() -> None:
    assert grammar_for_prompt("sakuga anime fight, cel shaded").name == "anime_2d"
    assert grammar_for_prompt("16-bit pixel art hero run").name == "pixel_art"
    assert grammar_for_prompt("lego brickfilm chase").name == "stop_motion"
    assert grammar_for_prompt("ecommerce product turntable packshot").name == "product"
    assert grammar_for_prompt("dreamworks style family film").name == "pixar_3d"
    assert grammar_for_prompt("a quiet street at dusk").name == "realistic"
    assert grammar_for_engine("claymation").name == "stop_motion"


def test_videomax_is_style_aware() -> None:
    anime = apply_video_quality(parse_process_options({"videomax": True}), prompt="ghibli anime walk cycle")
    real = apply_video_quality(parse_process_options({"video_quality": "max"}), prompt="photoreal documentary")
    assert anime.videomax and real.videomax
    assert anime.motion_grammar == "anime_2d" and real.motion_grammar == "realistic"
    assert anime.deflicker_strength < real.deflicker_strength
    assert anime.min_physics == 0.0 < real.min_physics
    assert real.sample.num >= 3 and real.sample.holy_grail
    assert real.max_retries >= 3


def test_forced_grammar_and_presets() -> None:
    forced = apply_video_quality(parse_process_options({"videomax": True, "motion_grammar": "cartoon"}), prompt="x")
    assert forced.motion_grammar == "cartoon"
    pixel = apply_video_quality(parse_process_options({"motion_grammar": "pixel_art"}))
    assert pixel.temporal_smooth == 0 and not pixel.videomax
    strong = apply_video_quality(parse_process_options({"video_quality": "strong"}))
    assert strong.sample.pick_best == "combo_vit" and strong.depth_interpolate
    assert apply_video_quality(parse_process_options({})) == parse_process_options({})
    with pytest.raises(ValueError):
        apply_video_quality(parse_process_options({"video_quality": "ultra"}))


def test_plan_videomax_overrides_are_real_options() -> None:
    fields = set(parse_process_options({}).__dataclass_fields__)
    assert set(plan_videomax("a red fox in snow").overrides) <= fields


def test_keyframe_argv_uses_existing_sample_flags() -> None:
    opts = apply_video_quality(parse_process_options({"video_quality": "strong"}))
    extras = build_keyframe_sample_extras(opts.sample)
    assert extras[:4] == ["--num", "3", "--pick-best", "combo_vit"]
    assert "--holy-grail" in extras and "--less-ai" in extras
    job = KeyframeEditJob(
        segment_index=0,
        frame_index=0,
        source_frame_path="in.png",
        output_path="out.png",
        prompt="hall",
        sample_extra_args=["--num", "5", "--seed", "1"],
    )
    cmd = build_sample_cmd_for_keyframe(job, ckpt="c.pt", width=1280, height=720, sample_options=opts.sample)
    assert cmd[cmd.index("--width") + 1] == "1280" and "--image-size" not in cmd
    assert cmd.count("--num") == 1 and cmd.count("--seed") == 1
    assert cmd[cmd.index("--num") + 1] == "3"
