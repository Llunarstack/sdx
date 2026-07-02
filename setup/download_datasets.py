#!/usr/bin/env python3
"""Download the booru datasets (danbooru, e621, rule34.xxx, rule34.xyz) into SDX manifests.

Full-site crawl by default (no tag filter, all ratings), which is what "download
everything" means. Runs the three sites concurrently, each with the threaded,
resumable, cursor-paginating engine in ``scripts/scrape``.

    python setup/download_datasets.py --out /workspace/data              # everything
    python setup/download_datasets.py --sites danbooru e621 --tags "landscape"
    python setup/download_datasets.py --max-posts 100000 --workers 16

Ratings default to ``all`` (SFW + NSFW). The mandatory CSAM-safety gate in
scripts/scrape/safety.py runs on every post regardless and cannot be disabled.
Each site writes to ``<out>/<site>/{images,manifest.jsonl}`` and resumes on rerun.

Note on scale: a true full crawl is millions of images / tens of TB and takes
days — point --out at your network volume and just let it resume across runs.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scrape.booru_client import BooruClient  # noqa: E402
from scripts.scrape.scrape_cli import _parse_ratings  # noqa: E402
from scripts.scrape.secrets_config import get_credentials  # noqa: E402
from scripts.scrape.sites import ADAPTERS, RATE_LIMITS, build_adapter  # noqa: E402
from utils.prompt.artist_registry import build_from_manifests  # noqa: E402

ALL_SITES = sorted(ADAPTERS)  # danbooru, e621, rule34xxx, rule34xyz


def _scrape_site(site: str, args) -> tuple[str, object]:
    creds = get_credentials(site, args.secrets)
    adapter = build_adapter(site, creds)
    contact = creds.username or creds.email or "anonymous"
    user_agent = f"sdx-dataset-scraper/1.0 (by {contact})"
    rate = args.rate if args.rate is not None else RATE_LIMITS.get(site, 2.0)
    out_dir = Path(args.out) / site

    client = BooruClient(
        adapter,
        out_dir,
        user_agent=user_agent,
        rate_per_sec=rate,
        ratings=_parse_ratings(args.ratings),
        max_workers=int(args.workers),
        split_frames=bool(getattr(args, "split_frames", True)),
        frame_fps=float(getattr(args, "frame_fps", 2.0)),
        max_frames_per_post=int(getattr(args, "max_frames_per_post", 0)),
        delete_raw_after_split=not bool(getattr(args, "keep_raw_media", False)),
    )
    print(
        f"[{site}] start -> {out_dir} (ratings={args.ratings}, {rate}/s api, "
        f"{args.workers} dl threads, split_frames={getattr(args, 'split_frames', True)})",
        flush=True,
    )
    stats = client.run(args.tags, max_posts=args.max_posts, dry_run=args.dry_run)
    print(
        f"[{site}] done: downloaded={stats.downloaded} fetched={stats.fetched} "
        f"posts_split={stats.posts_split} frames={stats.frames_extracted} "
        f"blocked_unsafe={stats.skipped_unsafe} errors={stats.errors}",
        flush=True,
    )
    return site, stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Full booru dataset downloader (danbooru, e621, rule34.xxx, rule34.xyz).")
    p.add_argument("--out", required=True, help="Base output dir (per-site subfolders created).")
    p.add_argument("--sites", nargs="*", default=ALL_SITES, choices=ALL_SITES, help="Sites to crawl (default: all).")
    p.add_argument("--tags", default="", help="Tag query (default: empty = whole site).")
    p.add_argument("--ratings", default="all", help="s/q/e or 'all' (default all = SFW+NSFW). Blocklist always on.")
    p.add_argument("--max-posts", type=int, default=0, help="Per-site cap (0 = unlimited full crawl).")
    p.add_argument("--workers", type=int, default=16, help="Image-download threads per site.")
    p.add_argument(
        "--split-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split GIF/video posts into JPEG frames (default: on).",
    )
    p.add_argument("--frame-fps", type=float, default=2.0, help="Video frame sample rate (0 = native fps).")
    p.add_argument("--max-frames-per-post", type=int, default=0, help="Max frames per GIF/video (0 = all frames).")
    p.add_argument(
        "--keep-raw-media",
        action="store_true",
        help="Keep original gif/mp4/webm after frame split (doubles disk use).",
    )
    p.add_argument("--rate", type=float, default=None, help="API req/s per site (default: per-site polite value).")
    p.add_argument("--secrets", default=None, help="Secrets file (default $SDX_SECRETS_FILE or D:\\Development\\secret.txt).")
    p.add_argument("--dry-run", action="store_true", help="Fetch + filter only; download nothing.")
    args = p.parse_args(argv)
    if args.secrets is None:
        from scripts.scrape.secrets_config import get_secrets_path

        args.secrets = str(get_secrets_path())

    sites = list(dict.fromkeys(args.sites))
    print(f"Crawling {sites} -> {args.out}\n")

    results = {}
    # One thread per site: pagination is sequential within a site but sites run
    # in parallel, and each site fans out its own download thread pool.
    with ThreadPoolExecutor(max_workers=len(sites)) as pool:
        futures = {pool.submit(_scrape_site, s, args): s for s in sites}
        for fut in as_completed(futures):
            site = futures[fut]
            try:
                _, stats = fut.result()
                results[site] = stats
            except Exception as e:  # one site failing shouldn't kill the others
                print(f"[{site}] ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    print("\n=== summary ===")
    grand = 0
    failed_sites: list[str] = []
    for site in sites:
        st = results.get(site)
        if st is None:
            print(f"  {site}: failed")
            failed_sites.append(site)
            continue
        grand += st.downloaded
        print(
            f"  {site}: {st.downloaded} images, {st.posts_split} split, "
            f"{st.frames_extracted} frames, {st.skipped_unsafe} blocked, {st.errors} errors"
        )
    print(f"total downloaded: {grand}")
    if failed_sites:
        print(f"FAILED sites: {', '.join(failed_sites)}", file=sys.stderr)
        return 1
    if not args.dry_run:
        manifests = [Path(args.out) / s / "manifest.jsonl" for s in sites if (Path(args.out) / s / "manifest.jsonl").is_file()]
        if manifests:
            reg = build_from_manifests(manifests)
            index_out = Path(args.out) / "artist_index.json"
            reg.save(index_out)
            print(f"artist index: {len(reg):,} names -> {index_out}")
    print(f"\nTrain on the combined set with, e.g.:")
    for site in sites:
        print(f"  python train.py --manifest-jsonl {Path(args.out) / site / 'manifest.jsonl'} --data-path {Path(args.out) / site} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
