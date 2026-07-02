"""Shared scraping engine: polite rate limiting, resumable image download,
SDX JSONL manifest writing. Site specifics live in :mod:`scripts.scrape.sites`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import requests

from .safety import blocking_tags


@dataclass
class Post:
    """Normalized post from any booru site."""

    site: str
    id: str
    md5: str
    file_url: str
    ext: str
    tags: list[str]
    rating: str = ""
    width: int = 0
    height: int = 0

    @property
    def caption(self) -> str:
        # Tag-style caption (comma-separated), matching the SDX dataloader's
        # danbooru convention. Underscores -> spaces reads better for T5.
        return ", ".join(t.replace("_", " ") for t in self.tags)


@dataclass
class RateLimiter:
    """Simple minimum-interval throttle (requests/sec ceiling)."""

    min_interval_s: float
    _last: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        gap = self.min_interval_s - (now - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


@dataclass
class ScrapeStats:
    fetched: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    skipped_unsafe: int = 0
    skipped_rating: int = 0
    skipped_no_url: int = 0
    errors: int = 0
    unsafe_tags: dict[str, int] = field(default_factory=dict)


class BooruClient:
    """Drives one site adapter: page -> filter -> download -> manifest."""

    def __init__(
        self,
        adapter,
        out_dir: str | Path,
        *,
        user_agent: str,
        rate_per_sec: float,
        ratings: Optional[set[str]] = None,
        timeout_s: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        self.adapter = adapter
        self.out_dir = Path(out_dir)
        self.images_dir = self.out_dir / "images"
        self.manifest_path = self.out_dir / "manifest.jsonl"
        self.ratings = ratings  # None = allow all ratings (blocklist still enforced)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.limiter = RateLimiter(1.0 / max(rate_per_sec, 0.01))
        self.stats = ScrapeStats()

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        self._seen_md5: set[str] = set()

    # -- helpers ----------------------------------------------------------
    def _load_resume_state(self) -> None:
        """Populate seen-md5 set from an existing manifest so reruns resume."""
        if not self.manifest_path.is_file():
            return
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self._seen_md5.add(json.loads(line)["md5"])
            except (json.JSONDecodeError, KeyError):
                continue

    def _request_json(self, url: str, params: dict, *, auth=None):
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                r = self.session.get(url, params=params, auth=auth, timeout=self.timeout_s)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                last_exc = e
                time.sleep(1.5 * (attempt + 1))
        if last_exc:
            raise last_exc
        return None

    def _download_image(self, post: Post, dest: Path) -> bool:
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                with self.session.get(post.file_url, timeout=self.timeout_s, stream=True) as r:
                    r.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            if chunk:
                                f.write(chunk)
                    tmp.replace(dest)
                return True
            except requests.RequestException:
                time.sleep(1.5 * (attempt + 1))
        return False

    def _passes_rating(self, post: Post) -> bool:
        if self.ratings is None:
            return True
        return (post.rating or "").lower()[:1] in self.ratings

    # -- main loop --------------------------------------------------------
    def run(self, tags: str, *, max_posts: int, dry_run: bool = False) -> ScrapeStats:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._load_resume_state()

        manifest = None if dry_run else open(self.manifest_path, "a", encoding="utf-8")
        try:
            for post in self._iter_posts(tags, max_posts=max_posts):
                self.stats.fetched += 1

                unsafe = blocking_tags(post.tags)
                if unsafe:
                    self.stats.skipped_unsafe += 1
                    for t in unsafe:
                        self.stats.unsafe_tags[t] = self.stats.unsafe_tags.get(t, 0) + 1
                    continue
                if not post.file_url or not post.md5:
                    self.stats.skipped_no_url += 1
                    continue
                if not self._passes_rating(post):
                    self.stats.skipped_rating += 1
                    continue
                if post.md5 in self._seen_md5:
                    self.stats.skipped_existing += 1
                    continue

                fname = f"{post.md5}.{post.ext.lstrip('.') or 'jpg'}"
                dest = self.images_dir / fname
                if dry_run:
                    self.stats.downloaded += 1
                    self._seen_md5.add(post.md5)
                    continue

                if not dest.is_file():
                    if not self._download_image(post, dest):
                        self.stats.errors += 1
                        continue

                row = {
                    "image_path": str(Path("images") / fname),
                    "caption": post.caption,
                    "rating": post.rating,
                    "md5": post.md5,
                    "source": f"{post.site}:{post.id}",
                    "width": post.width,
                    "height": post.height,
                }
                if manifest is not None:
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    manifest.flush()
                self._seen_md5.add(post.md5)
                self.stats.downloaded += 1
        finally:
            if manifest is not None:
                manifest.close()
        return self.stats

    def _iter_posts(self, tags: str, *, max_posts: int) -> Iterator[Post]:
        yielded = 0
        page = self.adapter.first_page
        while yielded < max_posts:
            data = self._request_json(
                self.adapter.posts_url,
                self.adapter.build_params(tags, page),
                auth=self.adapter.auth,
            )
            posts = list(self.adapter.parse_posts(data))
            if not posts:
                break
            for post in posts:
                if yielded >= max_posts:
                    break
                yield post
                yielded += 1
            page += 1
