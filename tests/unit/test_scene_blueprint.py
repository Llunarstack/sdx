import json

from utils.prompt.scene_blueprint import load_scene_blueprint


def test_scene_blueprint_compiles_core_sections(tmp_path):
    blueprint = {
        "scene": ["city rooftop at night"],
        "composition": ["two-subject composition", "clear depth layering"],
        "camera": ["three-quarter view", "medium shot"],
        "actors": [
            {
                "id": "a",
                "spatial_anchor": "left side",
                "role": "lead",
                "identity": ["short silver hair"],
                "pose": ["standing pose"],
                "avoid": ["merged limbs"],
            },
            {
                "id": "b",
                "role": "support",
                "identity": ["black jacket"],
            },
        ],
        "relations": [{"a": "a", "kind": "next to", "b": "b", "detail": "left side"}],
        "avoid": ["watermark"],
        "anti_artifacts": True,
    }
    p = tmp_path / "scene.json"
    p.write_text(json.dumps(blueprint), encoding="utf-8")

    pos, neg = load_scene_blueprint(str(p), strength=1.2)
    assert "city rooftop at night" in pos
    assert "a position: left side" in pos
    assert "a role: lead" in pos
    assert "a next to b, left side" in pos
    assert "watermark" in neg
    assert "inconsistent perspective" in neg


def test_scene_blueprint_invalid_shape_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    try:
        load_scene_blueprint(str(p))
        assert False, "Expected ValueError"
    except ValueError:
        assert True
