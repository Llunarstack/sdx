"""Director timeline, CGI, v2v/i2i, uploads/refs."""

from __future__ import annotations

from pipelines.video.director_timeline import (
    events_active_at,
    media_mode_overrides,
    parse_cli_refs,
    parse_director_timeline,
    prompt_for_keyframe,
)
from pipelines.video.motion_grammar import grammar_for_engine
from pipelines.video.process_options import ProcessOptions, parse_process_options
from pipelines.video.scene_graph import parse_scene_dict, validate_scene_graph
from pipelines.video.style_router import route_style
from pipelines.video.types import VideoMode


def test_cgi_style_and_grammar() -> None:
    route = route_style("octane lookdev cgi hero render")
    assert route.style == "cgi"
    assert grammar_for_engine("cgi").name == "cgi"
    assert float(route.videomax_profile.get("min_permanence", 0)) >= 0.60


def test_parse_timed_events_concurrent() -> None:
    tl = parse_director_timeline(
        {
            "events": [
                {"at_sec": 2.0, "until_sec": 3.0, "prompt": "explosion", "effect": "fire"},
                {"at_sec": 2.0, "until_sec": 2.5, "camera": "whip pan", "concurrent": True},
                {"at_sec": 5.0, "prompt": "rain", "shot_id": "close"},
            ]
        }
    )
    assert len(tl.events) == 3
    active = events_active_at(tl.events, 2.1, shot_id="")
    assert len(active) >= 2
    p, n, s = prompt_for_keyframe("hero walks", t_sec=2.1, events=tl.events, shot_id="")
    assert "explosion" in p or "fire" in p
    assert "whip" in p.lower() or "pan" in p.lower()


def test_cli_refs_roles() -> None:
    refs = parse_cli_refs(["hero.png:identity", "look.png:style:0.9", "plate.mp4:motion"])
    assert refs[0]["role"] == "identity"
    assert refs[1]["role"] == "style"
    assert abs(float(refs[1]["strength"]) - 0.9) < 1e-6
    assert refs[2]["role"] == "motion"


def test_media_mode_overrides() -> None:
    assert media_mode_overrides("v2v")["mode"] == "v2v"
    assert media_mode_overrides("i2i")["edit_strength"] >= 0.6
    assert media_mode_overrides("cgi")["motion_grammar"] == "cgi"


def test_scene_v2v_cgi_events() -> None:
    data = {
        "mode": "v2v",
        "scene": {"prompt": "cgi alley", "duration_sec": 4, "style": "cgi"},
        "motion_clip": "does_not_need_to_exist_for_parse.mp4",
        "events": [{"at_sec": 1.0, "prompt": "neon flicker"}],
        "references": [{"path": "face.png", "role": "identity"}],
        "shots": [{"id": "a", "prompt": "walk", "duration_sec": 4}],
        "edit": {"motion_grammar": "cgi"},
    }
    g = parse_scene_dict(data)
    assert g.mode == VideoMode.V2V
    assert len(g.events) >= 1
    assert len(g.references) >= 1
    # missing file is ok at validate for motion path existence? validate only checks field present
    issues = validate_scene_graph(g)
    assert not any("v2v mode requires" in x for x in issues)


def test_process_options_director_events() -> None:
    opts = parse_process_options(
        {"director_events": [{"at_sec": 0.5, "prompt": "boom"}], "multimodal_refs": [{"path": "a.png", "role": "style"}]}
    )
    assert opts.director_events
    assert opts.multimodal_refs
