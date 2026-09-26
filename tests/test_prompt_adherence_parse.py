"""PromptParser extracts clothing, objects, people, and spatial relation tokens."""

from models.prompt_adherence import PromptParser


def test_parse_binds_red_dress_and_blue_suit():
    parsed = PromptParser().parse("a woman in a red dress and a man in a blue suit")
    subjects = {t.subject for t in parsed.triples}
    assert "dress" in subjects
    assert "suit" in subjects
    dress = next(t for t in parsed.triples if t.subject == "dress")
    suit = next(t for t in parsed.triples if t.subject == "suit")
    assert "red" in dress.attributes
    assert "blue" in suit.attributes


def test_parse_spatial_left_of_and_colored_objects():
    parsed = PromptParser().parse("a red cube to the left of a blue sphere")
    subjects = {t.subject for t in parsed.triples}
    assert "cube" in subjects
    assert "sphere" in subjects
    assert any("left" in t.relations for t in parsed.triples)
    cube = next(t for t in parsed.triples if t.subject == "cube")
    assert "red" in cube.attributes
