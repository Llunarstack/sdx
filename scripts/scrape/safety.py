"""Mandatory content-safety gate for scraped booru posts.

The blocklist targets tags used to label sexual content involving minors (real
or drawn). Scraping rule34/danbooru/e621 without this filter pulls illegal
content, so this gate is applied to *every* post before download and cannot be
turned off from the CLI.

Defense in depth:
  1. ``blocked_query_tags()`` negates these tags in the API query where the
     site's tag budget allows (first-line filter).
  2. ``is_allowed()`` re-checks the returned post's own tag list (authoritative;
     catches anything the query missed, e.g. danbooru's 2-tag anon limit).
"""

from __future__ import annotations

from collections.abc import Iterable

# Exact-match tokens (boorus tokenize tags precisely, so exact match avoids
# substring false positives like "cuba"/"incubator" while staying unbypassable
# by the tag itself). Lowercased; underscores and spaces are normalized.
_BLOCKED_TAGS: frozenset[str] = frozenset(
    {
        # drawn minor content
        "loli",
        "lolicon",
        "loli_dominant",
        "shota",
        "shotacon",
        "shota_dominant",
        "cub",
        "cubquest",
        "toddlercon",
        "todlercon",
        "toddler",
        # explicit underage labels used across boorus (e621 uses "young")
        "young",
        "young_human",
        "underage",
        "child",
        "children",
        "baby",
        "infant",
        "preteen",
        "pre-teen",
        "kindergartner",
        "elementary_school",
        # aged-down / age regression sexualization
        "age_regression",
        "aged_down",
    }
)

# Substring roots with no legitimate tag use. Kept tiny on purpose.
_BLOCKED_SUBSTRINGS: tuple[str, ...] = ("lolicon", "shotacon", "toddlercon")


def normalize_tag(tag: str) -> str:
    return tag.strip().lower().replace(" ", "_")


def blocked_query_tags() -> list[str]:
    """Negated-tag fragments to inject into an API query (best-effort)."""
    return [f"-{t}" for t in sorted(_BLOCKED_TAGS)]


def blocking_tags(tags: Iterable[str]) -> list[str]:
    """Return the offending tags in ``tags`` (empty list = post is allowed)."""
    hits: list[str] = []
    for raw in tags:
        t = normalize_tag(raw)
        if not t:
            continue
        if t in _BLOCKED_TAGS or any(sub in t for sub in _BLOCKED_SUBSTRINGS):
            hits.append(t)
    return hits


def is_allowed(tags: Iterable[str]) -> bool:
    """True iff none of the post's tags hit the blocklist."""
    return not blocking_tags(tags)
