from utils.prompt.neg_filter import filter_negative_by_positive, positive_token_set


def test_positive_token_set_basic():
    s = positive_token_set("a, b cat")
    assert "a" in s
    assert "b" in s
    assert "cat" in s


def test_filter_removes_overlapping_tokens():
    pos = "red dress, woman"
    neg = "blurry, red dress, bad hands"
    out = filter_negative_by_positive(pos, neg)
    assert "blurry" in out
    assert "bad" in out.lower() or "hands" in out.lower()
    assert "red dress" not in out


def test_filter_empty_positive_returns_negative():
    assert filter_negative_by_positive("", "a, b") == "a, b"
