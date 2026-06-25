"""Control modes and image+text binding compile tests."""

from __future__ import annotations

from pathlib import Path

from pipelines.video.controls import ControlMode, InputBinding, compile_shot_control_plan
from pipelines.video.scene_graph import compile_scene_graph, parse_scene_dict


def test_image_text_binding_in_prompt():
    cp = compile_shot_control_plan(
        shot_id="s0",
        shot_index=0,
        base_prompt="forest at dusk",
        base_negative="blur",
        bindings=[
            InputBinding(
                entity_id="hero",
                image="hero.png",
                image_role="identity and armor",
                text_role="walking forward",
                control=ControlMode.IDENTITY,
                reference_strength=0.8,
            )
        ],
    )
    assert "[image:hero provides" in cp.positive_prompt
    assert "[text:walking forward]" in cp.positive_prompt


def test_control_modes_multi_ref():
    cp = compile_shot_control_plan(
        shot_id="s0",
        shot_index=0,
        base_prompt="scene",
        base_negative="",
        bindings=[
            InputBinding(
                entity_id="a",
                image="a.png",
                control=ControlMode.STYLE,
                reference_strength=0.5,
            ),
            InputBinding(
                entity_id="b",
                image="b.png",
                control=ControlMode.STYLE,
                reference_strength=0.7,
            ),
        ],
    )
    assert "--style-ref" in cp.sample_extra_args


def test_i2v_scene_inputs_compile():
    data = {
        "mode": "i2v",
        "scene": {"prompt": "hero walks", "duration_sec": 3},
        "inputs": [
            {
                "id": "hero_ref",
                "image": "hero.png",
                "provides": "identity",
                "text_changes": "walking",
                "control": "identity",
            }
        ],
        "characters": {"hero": {"bind_input": "hero_ref", "description": "knight"}},
        "shots": [{"id": "a", "prompt": "wide walk", "duration_sec": 3, "characters": ["hero"]}],
    }
    compiled = compile_scene_graph(parse_scene_dict(data))
    assert len(compiled.control_plans) == 1


def test_auto_rig_writes_layout(tmp_path: Path):
    from pipelines.video.auto_rig import auto_rig_character, write_rig_box_layout

    img = tmp_path / "char.png"
    from PIL import Image

    Image.new("RGB", (128, 256), (80, 80, 120)).save(img)
    rig = auto_rig_character("hero", img, text_by_part={"head": "preserve face", "legs": "walking"})
    layout = write_rig_box_layout(rig, tmp_path / "rig.json")
    assert layout.is_file()
    assert "regions" in layout.read_text(encoding="utf-8")
