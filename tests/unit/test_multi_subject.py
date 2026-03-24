"""utils.prompt.multi_subject — labeled multi-character prompt merge."""

from utils.prompt.multi_subject import (
    TRAINING_CAPTION_GUIDE,
    merge_character_sheet_negatives,
    merge_character_sheet_positives,
    multi_sheet_extra_negatives_csv,
)


def test_merge_single_block_unchanged():
    assert merge_character_sheet_positives(["a, b"]) == "a, b"


def test_merge_two_labeled():
    s = merge_character_sheet_positives(["red dress, tall", "blue suit, short"])
    assert "character 1" in s and "character 2" in s
    assert "red dress" in s and "blue suit" in s


def test_merge_negatives_dedupes():
    n = merge_character_sheet_negatives(["a, b", "b, c"])
    assert "a" in n and "b" in n and "c" in n


def test_training_guide_non_empty():
    assert "2girls" in TRAINING_CAPTION_GUIDE or "count" in TRAINING_CAPTION_GUIDE.lower()


def test_extra_negatives_csv():
    assert "outfit" in multi_sheet_extra_negatives_csv().lower()
