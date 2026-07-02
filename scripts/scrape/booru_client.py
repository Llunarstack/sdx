"""Shared scraping engine: polite rate limiting, resumable image download,
SDX JSONL manifest writing. Site specifics live in :mod:`scripts.scrape.sites`.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .frame_split import extract_training_frames, is_splittable_ext, needs_frame_split
from .post_cap import post_cap_reached
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
    artist_tags: list[str] = field(default_factory=list)
    character_tags: list[str] = field(default_factory=list)
    copyright_tags: list[str] = field(default_factory=list)  # series / franchise

    @property
    def caption(self) -> str:
        # Tag-style caption (comma-separated), matching the SDX dataloader's
        # danbooru convention. Underscores -> spaces reads better for T5.
        # Artist tags are already in ``tags``; ``artist_tags`` is metadata for the index.
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
    posts_split: int = 0
    frames_extracted: int = 0
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
        timeout_s: float = 60.0,
        max_retries: int = 6,
        max_workers: int = 8,
        dl_chunk_bytes: int = 1 << 18,
        split_frames: bool = True,
        frame_fps: float = 1.0,
        max_frames_per_post: int = 120,
        delete_raw_after_split: bool = True,
    ) -> None:
        self.adapter = adapter
        self.out_dir = Path(out_dir)
        self.images_dir = self.out_dir / "images"
        self.manifest_path = self.out_dir / "manifest.jsonl"
        self.ratings = ratings  # None = allow all ratings (blocklist still enforced)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_workers = max(1, int(max_workers))
        self.dl_chunk_bytes = max(1 << 14, int(dl_chunk_bytes))
        self.split_frames = bool(split_frames)
        self.frame_fps = max(0.1, float(frame_fps))
        self.max_frames_per_post = max(1, int(max_frames_per_post))
        self.delete_raw_after_split = bool(delete_raw_after_split)
        # API pagination only — image downloads use the thread pool without this throttle.
        self.api_limiter = RateLimiter(1.0 / max(rate_per_sec, 0.01))
        self.stats = ScrapeStats()

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            raise_on_status=False,
        )
        pool = max_workers + 4
        adapter_http = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
        self.session.mount("https://", adapter_http)
        self.session.mount("http://", adapter_http)

        self._seen_md5: set[str] = set()
        self._lock = threading.Lock()
        self._manifest_fh = None

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
                row = json.loads(line)
                if parent := row.get("parent_md5"):
                    self._seen_md5.add(str(parent))
                self._seen_md5.add(row["md5"])
            except (json.JSONDecodeError, KeyError):
                continue

    def _request_json(self, url: str, params: dict, *, auth=None):
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self.api_limiter.wait()
            try:
                r = self.session.get(url, params=params, auth=auth, timeout=self.timeout_s)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(min(60.0, 2.0 * (attempt + 1)))
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                last_exc = e
                time.sleep(min(45.0, 1.5 * (attempt + 1)))
        if last_exc:
            raise last_exc
        return None

    def _download_image(self, post: Post, dest: Path) -> bool:
        for attempt in range(self.max_retries):
            try:
                with self.session.get(post.file_url, timeout=self.timeout_s, stream=True) as r:
                    if r.status_code == 429 or r.status_code >= 500:
                        time.sleep(min(60.0, 2.0 * (attempt + 1)))
                        continue
                    r.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=self.dl_chunk_bytes):
                            if chunk:
                                f.write(chunk)
                    tmp.replace(dest)
                return True
            except requests.RequestException:
                time.sleep(min(45.0, 1.5 * (attempt + 1)))
        return False

    def _passes_rating(self, post: Post) -> bool:
        if self.ratings is None:
            return True
        return (post.rating or "").lower()[:1] in self.ratings

    def _accept(self, post: Post) -> bool:
        """Apply safety/rating/dedupe gates. Updates stats. True = download it."""
        unsafe = blocking_tags(post.tags)
        if unsafe:
            with self._lock:
                self.stats.skipped_unsafe += 1
                for t in unsafe:
                    self.stats.unsafe_tags[t] = self.stats.unsafe_tags.get(t, 0) + 1
            return False
        if not post.file_url or not post.md5:
            with self._lock:
                self.stats.skipped_no_url += 1
            return False
        if not self._passes_rating(post):
            with self._lock:
                self.stats.skipped_rating += 1
            return False
        with self._lock:
            if post.md5 in self._seen_md5:
                self.stats.skipped_existing += 1
                return False
            self._seen_md5.add(post.md5)  # claim it now so workers don't race
        return True

    def _write_manifest_row(self, post: Post, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            if self._manifest_fh is not None:
                self._manifest_fh.write(line)
                self._manifest_fh.flush()
            self.stats.downloaded += 1

    def _fetch_one(self, post: Post) -> None:
        """Worker: download, optionally split GIF/video frames, append manifest rows."""
        ext = post.ext.lstrip(".") or "jpg"
        fname = f"{post.md5}.{ext}"
        dest = self.images_dir / fname

        if not dest.is_file():
            if not self._download_image(post, dest):
                with self._lock:
                    self.stats.errors += 1
                    self._seen_md5.discard(post.md5)
                return

        split = self.split_frames and is_splittable_ext(ext)
        frames = []
        if split:
            try:
                frames = extract_training_frames(
                    dest,
                    self.images_dir,
                    post.md5,
                    ext,
                    fps=self.frame_fps,
                    max_frames=self.max_frames_per_post,
                )
            except Exception:
                frames = []

        if frames:
            with self._lock:
                self.stats.posts_split += 1
                self.stats.frames_extracted += len(frames)
            for fr in frames:
                self._write_manifest_row(
                    post,
                    {
                        "image_path": fr.rel_path,
                        "caption": post.caption,
                        "rating": post.rating,
                        "md5": fr.md5,
                        "parent_md5": post.md5,
                        "frame_index": fr.frame_index,
                        "source": f"{post.site}:{post.id}",
                        "width": fr.width,
                        "height": fr.height,
                        **({"artist_tags": list(post.artist_tags)} if post.artist_tags else {}),
                        **({"character_tags": list(post.character_tags)} if post.character_tags else {}),
                        **({"copyright_tags": list(post.copyright_tags)} if post.copyright_tags else {}),
                    },
                )
            if self.delete_raw_after_split and needs_frame_split(dest, ext):
                try:
                    dest.unlink(missing_ok=True)
                except OSError:
                    pass
            return

        if split and needs_frame_split(dest, ext):
            with self._lock:
                self.stats.errors += 1
                self._seen_md5.discard(post.md5)
            return

        # Still image (or single-frame gif).
        row = {
            "image_path": str(Path("images") / fname),
            "caption": post.caption,
            "rating": post.rating,
            "md5": post.md5,
            "source": f"{post.site}:{post.id}",
            "width": post.width,
            "height": post.height,
        }
        if post.artist_tags:
            row["artist_tags"] = list(post.artist_tags)
        if post.character_tags:
            row["character_tags"] = list(post.character_tags)
        if post.copyright_tags:
            row["copyright_tags"] = list(post.copyright_tags)
        self._write_manifest_row(post, row)

    # -- main loop --------------------------------------------------------
    def run(self, tags: str, *, max_posts: int, dry_run: bool = False) -> ScrapeStats:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._load_resume_state()

        self._manifest_fh = None if dry_run else open(self.manifest_path, "a", encoding="utf-8")
        # Pagination (API) stays sequential and rate-limited; image downloads run
        # in a thread pool so network I/O overlaps instead of serializing.
        pool = None if dry_run else ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            for post in self._iter_posts(tags, max_posts=max_posts):
                with self._lock:
                    self.stats.fetched += 1
                if not self._accept(post):
                    continue
                if dry_run or pool is None:
                    with self._lock:
                        self.stats.downloaded += 1
                    continue
                pool.submit(self._fetch_one, post)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)  # drain in-flight downloads
            if self._manifest_fh is not None:
                self._manifest_fh.close()
                self._manifest_fh = None
        return self.stats

    def _iter_posts(self, tags: str, *, max_posts: int) -> Iterator[Post]:
        """Yield posts, using id-cursor pagination when the site supports it."""
        if hasattr(self.adapter, "iter_posts"):
            yield from self.adapter.iter_posts(
                self.session,
                tags,
                max_posts=max_posts,
                limiter=self.api_limiter,
                timeout_s=self.timeout_s,
            )
            return
        yielded = 0
        page = self.adapter.first_page
        before_id: Optional[int] = None
        use_cursor = bool(getattr(self.adapter, "cursor_supported", False))
        while not post_cap_reached(yielded, max_posts):
            params = self.adapter.build_params(tags, page, before_id) if use_cursor else self.adapter.build_params(tags, page)
            data = self._request_json(self.adapter.posts_url, params, auth=self.adapter.auth)
            posts = list(self.adapter.parse_posts(data))
            if not posts:
                break
            min_id = None
            for post in posts:
                if post_cap_reached(yielded, max_posts):
                    break
                yield post
                yielded += 1
                pid = int(post.id) if str(post.id).isdigit() else None
                if pid is not None:
                    min_id = pid if min_id is None else min(min_id, pid)
            if use_cursor:
                if min_id is None or min_id <= 1:
                    break  # reached the start of the site
                before_id = min_id
            else:
                page += 1
