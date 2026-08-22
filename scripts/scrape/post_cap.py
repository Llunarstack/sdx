"""Post-count limits for booru crawls (0 = unlimited)."""

from __future__ import annotations


def post_cap_reached(yielded: int, max_posts: int) -> bool:
    return max_posts > 0 and yielded >= max_posts


def posts_remaining(max_posts: int, yielded: int) -> int:
    """Batch size hint for paginated APIs; 0 means no cap."""
    if max_posts <= 0:
        return 0
    return max(0, max_posts - yielded)
