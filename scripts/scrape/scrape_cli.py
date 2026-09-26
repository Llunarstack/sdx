"""Scrape a booru site into an SDX training manifest.

    python -m scripts.scrape.scrape_cli --site danbooru --tags "landscape scenery" \\
        --max-posts 500 --out datasets/danbooru_scenery

    python -m scripts.scrape.scrape_cli --site e621 --tags "forest" \\
        --ratings s --max-posts 1000 --out /workspace/data/e621_forest

Output layout::

    <out>/images/<md5>.<ext>
    <out>/manifest.jsonl   # {"image_path", "caption", "rating", ...} rows

Feed it straight into training::

    python train.py --manifest-jsonl <out>/manifest.jsonl --data-path <out> ...

The mandatory CSAM-safety gate (scripts/scrape/safety.py) runs on every post and
cannot be disabled. Credentials come from the secrets file (default
``D:\\Development\\secret.txt``; override with --secrets or $SDX_SECRETS_FILE).
"""

from __future__ import annotations

import argparse
import sys

from .booru_client import BooruClient
from .secrets_config import get_credentials
from .sites import ADAPTERS, RATE_LIMITS, build_adapter

# Rating letters differ per site. danbooru: g(eneral) s(ensitive) q e.
# e621/rule34: s(afe) q e. So "safe" must accept both g and s to be portable.
_RATING_ALIASES: dict[str, set[str]] = {
    "s": {"s", "g"},
    "safe": {"s", "g"},
    "g": {"g"},
    "general": {"g"},
    "sensitive": {"s"},
    "q": {"q"},
    "questionable": {"q"},
    "e": {"e"},
    "explicit": {"e"},
}


def _parse_ratings(spec: str) -> set[str] | None:
    spec = (spec or "").strip().lower()
    if not spec or spec == "all":
        return None
    out: set[str] = set()
    for token in spec.replace(",", " ").split():
        if token in _RATING_ALIASES:
            out |= _RATING_ALIASES[token]
    return out or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scrape a booru site into an SDX JSONL manifest.")
    p.add_argument("--site", required=True, choices=sorted(ADAPTERS), help="Booru site to scrape.")
    p.add_argument("--tags", default="", help="Space-separated tag query (site syntax).")
    p.add_argument("--out", required=True, help="Output directory (images/ + manifest.jsonl).")
    p.add_argument("--max-posts", type=int, default=200, help="Max posts to download this run.")
    p.add_argument(
        "--ratings",
        default="all",
        help="Comma/space list of ratings to keep: s,q,e (or 'all'). Blocklist always enforced.",
    )
    p.add_argument(
        "--secrets",
        default=None,
        help="Path to secrets file (default $SDX_SECRETS_FILE or D:\\Development\\secret.txt).",
    )
    p.add_argument(
        "--rate", type=float, default=None, help="API requests/sec ceiling (default: per-site polite value)."
    )
    p.add_argument("--workers", type=int, default=8, help="Parallel image-download threads (default 8).")
    p.add_argument(
        "--split-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split animated GIFs and videos (mp4/webm/…) into JPEG training frames (default: on).",
    )
    p.add_argument(
        "--frame-fps",
        type=float,
        default=1.0,
        help="Target fps when sampling video frames (default 1.0 = one frame per second).",
    )
    p.add_argument(
        "--max-frames-per-post",
        type=int,
        default=120,
        help="Cap frames extracted per GIF/video post (default 120).",
    )
    p.add_argument(
        "--user-agent",
        default=None,
        help="HTTP User-Agent. e621/danbooru require a descriptive one; a default with your login is built if omitted.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Fetch + filter, but do not download images or write manifest."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        creds = get_credentials(args.site, args.secrets)
    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    adapter = build_adapter(args.site, creds)
    if adapter.auth is None and not getattr(adapter, "creds", creds).api_key:
        print(
            f"WARNING: no API credentials resolved for {args.site}; requests will be anonymous (rate-limited).",
            file=sys.stderr,
        )

    contact = creds.username or creds.email or "anonymous"
    user_agent = args.user_agent or f"sdx-dataset-scraper/1.0 (by {contact})"
    rate = args.rate if args.rate is not None else RATE_LIMITS.get(args.site, 2.0)

    client = BooruClient(
        adapter,
        args.out,
        user_agent=user_agent,
        rate_per_sec=rate,
        ratings=_parse_ratings(args.ratings),
        max_workers=int(args.workers),
        split_frames=bool(args.split_frames),
        frame_fps=float(args.frame_fps),
        max_frames_per_post=int(args.max_frames_per_post),
    )

    print(
        f"Scraping {args.site} tags={args.tags!r} -> {args.out} (max {args.max_posts}, {rate}/s, dry_run={args.dry_run})"
    )
    stats = client.run(args.tags, max_posts=args.max_posts, dry_run=args.dry_run)

    print("\n=== done ===")
    print(
        f"fetched={stats.fetched} downloaded={stats.downloaded} "
        f"posts_split={stats.posts_split} frames_extracted={stats.frames_extracted} "
        f"skipped_existing={stats.skipped_existing} skipped_rating={stats.skipped_rating} "
        f"skipped_no_url={stats.skipped_no_url} errors={stats.errors}"
    )
    print(f"BLOCKED (unsafe content): {stats.skipped_unsafe}")
    if stats.unsafe_tags:
        top = sorted(stats.unsafe_tags.items(), key=lambda kv: -kv[1])[:10]
        print("  blocked tags:", ", ".join(f"{t}={n}" for t, n in top))
    if not args.dry_run:
        print(f"manifest: {client.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
