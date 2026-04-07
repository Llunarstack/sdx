from utils.prompt.content_controls import apply_content_controls, infer_content_controls_from_prompt


def test_artist_composition_lite_adds_tags():
    p, n = apply_content_controls(
        "1girl",
        "bad",
        artist_composition="lite",
    )
    assert "rule of thirds" in p.lower()
    assert "muddy focal point" in n.lower()


def test_infer_artist_composition_from_keywords():
    inf = infer_content_controls_from_prompt("city street, two point perspective, vanishing point")
    assert inf.get("artist_composition") == "perspective"
    inf2 = infer_content_controls_from_prompt("golden ratio portrait, classical composition")
    assert inf2.get("artist_composition") == "classical"
