from scripts.scrape.post_cap import post_cap_reached, posts_remaining


def test_unlimited_cap():
    assert not post_cap_reached(0, 0)
    assert not post_cap_reached(999999, 0)


def test_finite_cap():
    assert not post_cap_reached(4, 5)
    assert post_cap_reached(5, 5)
    assert posts_remaining(10, 3) == 7
    assert posts_remaining(0, 3) == 0
