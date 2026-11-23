"""Booru dataset scraper (danbooru, e621, rule34.xxx, rule34.xyz) -> SDX JSONL manifests.

    python -m scripts.scrape --site danbooru --tags "landscape scenery" --max-posts 500 --out datasets/danbooru_scenery
    python -m scripts.scrape --site e621 --tags "forest" --ratings s --max-posts 1000 --out /workspace/data/e621_forest

Output: ``<out>/images/<md5>.<ext>`` + ``<out>/manifest.jsonl`` (``image_path``, ``caption``, ``rating``, ...);
train with ``python train.py --manifest-jsonl <out>/manifest.jsonl --data-path <out>``.

Every post passes the mandatory content-safety gate below (``blocking_tags`` / ``is_allowed``) before download.
It drops CSAM-adjacent content, cannot be disabled, and runs at fetch time so illegal content never lands on disk.
Credentials come from the secrets file (default ``D:\\Development\\secret.txt``; override with --secrets or
$SDX_SECRETS_FILE).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.safety.minor_guard import normalize, reasons  # noqa: E402

_log = logging.getLogger(__name__)


# --- mandatory content-safety gate --------------------------------------------
# The blocklist targets tags used to label sexual content involving minors (real or drawn). Defense in depth:
#   1. ``blocked_query_tags()`` negates these tags in the API query where the site's tag budget allows.
#   2. ``blocking_tags()`` re-checks each returned post's own tags (authoritative; catches anything the query
#      missed, e.g. danbooru's 2-tag anon limit). Each tag is matched exactly against the blocklist and also
#      screened by the training rule of utils.safety.minor_guard (catches ``oppai_loli``, ``l0li``, ``teen``...).

# Exact-match tokens (boorus tokenize tags precisely, so exact match avoids substring false positives like
# "cuba"/"incubator" while staying unbypassable by the tag itself). Lowercased; underscores/spaces normalized.
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
    return normalize(tag).replace(" ", "_")


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
        if t in _BLOCKED_TAGS or any(sub in t for sub in _BLOCKED_SUBSTRINGS) or reasons(t, training=True):
            hits.append(t)
    return hits


def is_allowed(tags: Iterable[str]) -> bool:
    """True iff none of the post's tags hit the blocklist."""
    return not blocking_tags(tags)


# --- credentials ----------------------------------------------------------------
# The secrets file is informal (``key: value`` lines under a site header). Credentials are only read into
# memory; nothing is written back or logged. Keep the secrets file OUTSIDE the repo.

DEFAULT_SECRETS_PATH = r"D:\Development\secret.txt"

_SITE_ALIASES = {
    "danbooru": "danbooru",
    "e621": "e621",
    "rule34xxx": "rule34xxx",
    "rule34.xxx": "rule34xxx",
    "rule34.xyz": "rule34xyz",
    "saucenao": "saucenao",
    "tineye": "tineye",
}
_USER_KEYS = ("user", "username", "login")
_PASS_KEYS = ("pas", "pass", "password", "p")
_API_KEYS = ("api", "api_key", "apikey", "api?", "key")
_EMAIL_KEYS = ("e", "em", "email")


@dataclass
class SiteCredentials:
    site: str
    username: str | None = None
    password: str | None = None
    api_key: str | None = None
    user_id: str | None = None
    email: str | None = None


def get_secrets_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve secrets file — ``D:\\Development\\secret.txt`` unless overridden."""
    return Path(path or os.environ.get("SDX_SECRETS_FILE") or DEFAULT_SECRETS_PATH)


def parse_secrets_file(path: str | os.PathLike[str] | None = None) -> dict[str, SiteCredentials]:
    """Return ``{canonical_site: SiteCredentials}`` parsed from the secrets file."""
    p = get_secrets_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Secrets file not found: {p}. Pass --secrets PATH or set SDX_SECRETS_FILE.")

    out: dict[str, SiteCredentials] = {}
    current: SiteCredentials | None = None
    for raw_line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = line.rstrip(":").strip().lower()
        if header in _SITE_ALIASES and (":" not in line or line.endswith(":") or " " not in line.rstrip(":")):
            site = _SITE_ALIASES[header]
            current = out.setdefault(site, SiteCredentials(site=site))
            continue
        if ":" not in line or current is None:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if not value:
            continue
        if current.site == "rule34xxx" and "api_key=" in value:
            api = re.search(r"api_key=([0-9a-fA-F]+)", value)
            uid = re.search(r"user_id=(\d+)", value)
            current.api_key = current.api_key or (api.group(1) if api else None)
            current.user_id = current.user_id or (uid.group(1) if uid else None)
            continue
        if key in _USER_KEYS:
            current.username = current.username or value
        elif key in _PASS_KEYS:
            current.password = current.password or value
        elif key in _API_KEYS:
            current.api_key = current.api_key or value
        elif key in _EMAIL_KEYS:
            current.email = current.email or value
        elif key == "ip" and current.site == "danbooru" and current.api_key is None:
            current.api_key = value  # danbooru api keys are sometimes mislabeled in the file
    return out


def get_credentials(site: str, path: str | os.PathLike[str] | None = None) -> SiteCredentials:
    creds = parse_secrets_file(path)
    canonical = _SITE_ALIASES.get(site.lower(), site.lower())
    if canonical not in creds:
        raise KeyError(f"No credentials for site {site!r} in secrets file. Found: {sorted(creds)}")
    return creds[canonical]


# --- post caps --------------------------------------------------------------------


def post_cap_reached(yielded: int, max_posts: int) -> bool:
    return max_posts > 0 and yielded >= max_posts


def posts_remaining(max_posts: int, yielded: int) -> int:
    """Batch size hint for paginated APIs; 0 means no cap."""
    if max_posts <= 0:
        return 0
    return max(0, max_posts - yielded)


# --- media validation --------------------------------------------------------------

# JPEG/PNG/WebP stills + frame-split JPEG outputs only.
TRAINABLE_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

# Never keep these for DiT training (archives, flash, partial downloads, markup).
BLOCKED_DOWNLOAD_EXTS = frozenset(
    {
        ".zip",
        ".swf",
        ".rar",
        ".7z",
        ".exe",
        ".bat",
        ".cmd",
        ".scr",
        ".msi",
        ".dll",
        ".html",
        ".htm",
        ".php",
        ".part",
        ".tmp",
        ".crdownload",
    }
)

# Raw animation sources — frame-split to JPEG, then delete the source file.
SPLITTABLE_SOURCE_EXTS = frozenset({".gif", ".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"})
VIDEO_EXTS = frozenset({"webm", "mp4", "mov", "mkv", "avi", "m4v"})
GIF_EXT = "gif"


def _bare_ext(ext: str) -> str:
    return (ext or "").lower().strip().lstrip(".")


def _head(path: Path, n: int) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def is_blocked_download_ext(ext: str) -> bool:
    e = _bare_ext(ext)
    return bool(e) and f".{e}" in BLOCKED_DOWNLOAD_EXTS


def sniff_media_kind(path: Path) -> str:
    """Return a short kind label: image, video, zip, html, swf, empty, missing, unknown."""
    try:
        head = _head(path, 32)
    except OSError:
        return "missing"
    if not head:
        return "empty"
    if head.startswith((b"<!DOCTYPE", b"<html", b"<?xml", b"<HTML")):
        return "html"
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return "zip"
    if head[:3] in (b"CWS", b"FWS"):
        return "swf"
    if head.startswith((b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"BM")):
        return "image"
    if head.startswith(b"RIFF") and b"WEBP" in head[:16]:
        return "image"
    if head[4:8] == b"ftyp" or head.startswith(b"\x1a\x45\xdf\xa3"):
        return "video"
    return "unknown"


def validate_trainable_image(path: Path, *, max_pixels: int = 178_956_970) -> bool:
    """True when path is a readable RGB-capable raster within PIL size limits."""
    if not path.is_file() or sniff_media_kind(path) != "image":
        return False
    try:
        if path.stat().st_size < 64:
            return False
    except OSError:
        return False
    old_limit = getattr(Image, "MAX_IMAGE_PIXELS", None)
    try:
        Image.MAX_IMAGE_PIXELS = max_pixels
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB")
        return True
    except Exception:
        return False
    finally:
        if old_limit is not None:
            Image.MAX_IMAGE_PIXELS = old_limit


def save_still_as_jpeg(src: Path, dest: Path, *, quality: int = 92) -> bool:
    try:
        with Image.open(src) as im:
            rgb = im.convert("RGB")
        dest.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dest, "JPEG", quality=quality, optimize=True)
        return True
    except Exception:
        return False


# --- GIF / video frame splitting -----------------------------------------------------
# SDX training expects static RGB frames: animated posts become ``images/<parent_md5>_f000001.jpg`` with one
# manifest row per frame.


@dataclass
class ExtractedFrame:
    rel_path: str  # e.g. images/abc_f000001.jpg
    abs_path: Path
    md5: str
    width: int
    height: int
    frame_index: int


def _looks_like_video_file(path: Path, *, min_bytes: int = 512) -> bool:
    """Reject empty downloads, HTML error pages, and other obvious non-video blobs."""
    try:
        if path.stat().st_size < min_bytes:
            return False
        head = _head(path, 16)
    except OSError:
        return False
    return not head.startswith((b"<!DOCTYPE", b"<html", b"<?xml", b"{"))


def _ffprobe_readable(path: Path) -> bool:
    if shutil.which("ffprobe") is None:
        return True
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0"]
            + ["-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and "video" in (proc.stdout or "").lower()


def is_splittable_ext(ext: str) -> bool:
    e = _bare_ext(ext)
    return e in VIDEO_EXTS or e == GIF_EXT


def needs_frame_split(path: Path, ext: str) -> bool:
    """True when the downloaded file should be exploded into training frames."""
    ext = _bare_ext(ext)
    if ext in VIDEO_EXTS:
        return True
    if ext == GIF_EXT:
        try:
            with Image.open(path) as im:
                return int(getattr(im, "n_frames", 1) or 1) > 1
        except Exception:
            return False
    return False


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def _frame(dest: Path, index: int, size: tuple[int, int] | None = None) -> ExtractedFrame:
    w, h = size or _image_size(dest)
    return ExtractedFrame(
        rel_path=str(Path("images") / dest.name),
        abs_path=dest,
        md5=_file_md5(dest),
        width=w,
        height=h,
        frame_index=index,
    )


def _extract_gif_frames(
    src: Path, out_dir: Path, parent_md5: str, *, max_frames: int, jpeg_quality: int
) -> list[ExtractedFrame]:
    frames: list[ExtractedFrame] = []
    with Image.open(src) as im:
        total = int(getattr(im, "n_frames", 1) or 1)
        if max_frames <= 0 or total <= max_frames:
            pick = list(range(total))
        else:
            step = total / max_frames
            pick = sorted({min(total - 1, int(i * step)) for i in range(max_frames)})
        for out_i, src_i in enumerate(pick, start=1):
            dest = out_dir / f"{parent_md5}_f{out_i:06d}.jpg"
            size = None
            if not dest.is_file():
                im.seek(src_i)
                rgb = im.convert("RGB")
                dest.parent.mkdir(parents=True, exist_ok=True)
                rgb.save(dest, "JPEG", quality=jpeg_quality, optimize=True)
                size = rgb.size
            frames.append(_frame(dest, out_i, size))
    return frames


def _extract_video_frames_ffmpeg(
    src: Path, out_dir: Path, parent_md5: str, *, fps: float, max_frames: int
) -> list[ExtractedFrame]:
    tmp_dir = out_dir / f".split_{parent_md5}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if fps > 0:
        cmd.extend(["-vf", f"fps={fps}"])
    if max_frames > 0:
        cmd.extend(["-frames:v", str(int(max_frames))])
    cmd.extend(["-q:v", "2", str(tmp_dir / "frame_%06d.jpg")])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed on {src.name}: {err[:500]}")
    paths = sorted(tmp_dir.glob("frame_*.jpg"))
    if not paths:
        raise RuntimeError(f"ffmpeg produced no frames for {src.name}")
    frames: list[ExtractedFrame] = []
    for i, tmp_path in enumerate(paths, start=1):
        dest = out_dir / f"{parent_md5}_f{i:06d}.jpg"
        if not dest.is_file():
            shutil.move(str(tmp_path), str(dest))
        else:
            tmp_path.unlink(missing_ok=True)
        frames.append(_frame(dest, i))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return frames


def _extract_video_frames_cv2(
    src: Path, out_dir: Path, parent_md5: str, *, fps: float, max_frames: int
) -> list[ExtractedFrame]:
    import cv2

    # Force the FFMPEG backend; default backend order can mis-detect files as image sequences.
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video with cv2+ffmpeg backend: {src}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    step = max(1, int(round(src_fps / fps))) if fps > 0 and src_fps > 0 else 1
    frames: list[ExtractedFrame] = []
    i = saved = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            saved += 1
            dest = out_dir / f"{parent_md5}_f{saved:06d}.jpg"
            if not dest.is_file():
                cv2.imwrite(str(dest), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            h, w = bgr.shape[:2]
            frames.append(_frame(dest, saved, (w, h)))
            if max_frames > 0 and saved >= max_frames:
                break
        i += 1
    cap.release()
    return frames


def existing_frames(out_dir: Path, parent_md5: str) -> list[ExtractedFrame]:
    """Return already-extracted frames for a parent post (resume)."""
    frames: list[ExtractedFrame] = []
    for p in sorted(out_dir.glob(f"{parent_md5}_f*.jpg")):
        try:
            size = _image_size(p)
        except Exception:
            size = (0, 0)
        try:
            idx = int(p.stem.rsplit("_f", 1)[-1])
        except (IndexError, ValueError):
            idx = 0
        frames.append(_frame(p, idx, size))
    return frames


def extract_training_frames(
    src: Path,
    images_dir: Path,
    parent_md5: str,
    ext: str,
    *,
    fps: float = 1.0,
    max_frames: int = 120,
    jpeg_quality: int = 92,
) -> list[ExtractedFrame]:
    """Split ``src`` into JPEG frames (``[]`` for stills or unreadable media)."""
    ext = _bare_ext(ext)
    existing = existing_frames(images_dir, parent_md5)
    if existing:
        return existing
    if not needs_frame_split(src, ext):
        return []
    if ext == GIF_EXT:
        return _extract_gif_frames(src, images_dir, parent_md5, max_frames=max_frames, jpeg_quality=jpeg_quality)
    if not _looks_like_video_file(src):
        _log.warning("skip frame split (not a video file): %s", src.name)
        return []
    if not _ffprobe_readable(src):
        _log.warning("skip frame split (ffprobe found no video stream): %s", src.name)
        return []
    if shutil.which("ffmpeg") is not None:
        try:
            return _extract_video_frames_ffmpeg(src, images_dir, parent_md5, fps=fps, max_frames=max_frames)
        except (RuntimeError, OSError) as exc:
            _log.warning("ffmpeg frame split failed for %s: %s", src.name, exc)
    try:
        return _extract_video_frames_cv2(src, images_dir, parent_md5, fps=fps, max_frames=max_frames)
    except Exception as exc:
        _log.warning("opencv frame split failed for %s: %s", src.name, exc)
        return []


# --- scraping engine ------------------------------------------------------------------


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
        # Comma-separated danbooru-style caption; underscores -> spaces reads better for T5.
        return ", ".join(t.replace("_", " ") for t in self.tags)

    def meta_tags(self) -> dict[str, list[str]]:
        return {
            k: list(v)
            for k, v in (
                ("artist_tags", self.artist_tags),
                ("character_tags", self.character_tags),
                ("copyright_tags", self.copyright_tags),
            )
            if v
        }


@dataclass
class RateLimiter:
    """Minimum-interval throttle (requests/sec ceiling)."""

    min_interval_s: float
    _last: float = 0.0

    def wait(self) -> None:
        gap = self.min_interval_s - (time.monotonic() - self._last)
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
    skipped_blocked_ext: int = 0
    skipped_invalid_media: int = 0
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
        ratings: set[str] | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 6,
        max_workers: int = 8,
        dl_chunk_bytes: int | None = None,
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
        chunk = dl_chunk_bytes if dl_chunk_bytes is not None else int(os.environ.get("SDX_DL_CHUNK_BYTES", 1 << 20))
        self.dl_chunk_bytes = max(1 << 14, int(chunk))
        self.split_frames = bool(split_frames)
        self.frame_fps = max(0.0, float(frame_fps))  # 0 = native fps
        self.max_frames_per_post = max(0, int(max_frames_per_post))  # 0 = all frames
        self.delete_raw_after_split = bool(delete_raw_after_split)
        # API pagination only — image downloads use the thread pool without this throttle.
        self.api_limiter = RateLimiter(1.0 / max(rate_per_sec, 0.01))
        self.stats = ScrapeStats()
        self.session = self._build_session(self.max_workers)
        self.session.headers.update({"User-Agent": user_agent})
        self._tls = threading.local()
        self._seen_md5: set[str] = set()
        self._lock = threading.Lock()
        self._manifest_fh = None
        self._manifest_pending = 0
        self._manifest_flush_every = max(1, int(os.environ.get("SDX_MANIFEST_FLUSH_EVERY", "64")))

    def _build_session(self, pool_size: int) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.max_retries,
            connect=self.max_retries,
            read=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            raise_on_status=False,
        )
        pool = max(pool_size + 8, 32)
        adapter_http = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
        session.mount("https://", adapter_http)
        session.mount("http://", adapter_http)
        return session

    def _dl_session(self) -> requests.Session:
        """Per-thread session — requests.Session is not safe for concurrent GETs."""
        s = getattr(self._tls, "session", None)
        if s is None:
            s = self._build_session(self.max_workers)
            s.headers.update(self.session.headers)
            self._tls.session = s
        return s

    def _load_resume_state(self) -> None:
        """Populate seen-md5 set from an existing manifest so reruns resume."""
        if not self.manifest_path.is_file():
            return
        with self.manifest_path.open(encoding="utf-8") as fh:
            for line in fh:
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
        last_exc: Exception | None = None
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
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(self.max_retries):
            try:
                with self._dl_session().get(post.file_url, timeout=self.timeout_s, stream=True) as r:
                    if r.status_code == 429 or r.status_code >= 500:
                        time.sleep(min(60.0, 2.0 * (attempt + 1)))
                        continue
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=self.dl_chunk_bytes):
                            if chunk:
                                f.write(chunk)
                    tmp.replace(dest)
                return True
            except requests.RequestException:
                time.sleep(min(45.0, 1.5 * (attempt + 1)))
            finally:
                if not dest.is_file():
                    _unlink(tmp)
        return False

    def _accept(self, post: Post) -> bool:
        """Apply safety/rating/dedupe gates. Updates stats. True = download it."""
        # An untagged post cannot be screened, so it fails closed.
        unsafe = blocking_tags(post.tags) if post.tags else ["<untagged>"]
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
        if self.ratings is not None and (post.rating or "").lower()[:1] not in self.ratings:
            with self._lock:
                self.stats.skipped_rating += 1
            return False
        with self._lock:
            if post.md5 in self._seen_md5:
                self.stats.skipped_existing += 1
                return False
            self._seen_md5.add(post.md5)  # claim it now so workers don't race
        return True

    def _reject(self, post: Post, stat: str, dest: Path | None = None) -> None:
        if dest is not None:
            _unlink(dest)
            _unlink(dest.with_suffix(dest.suffix + ".part"))
        with self._lock:
            setattr(self.stats, stat, getattr(self.stats, stat) + 1)
            self._seen_md5.discard(post.md5)

    def _write_manifest_row(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            if self._manifest_fh is not None:
                self._manifest_fh.write(line)
                self._manifest_pending += 1
                if self._manifest_pending >= self._manifest_flush_every:
                    self._manifest_fh.flush()
                    self._manifest_pending = 0
            self.stats.downloaded += 1

    def _fetch_one(self, post: Post) -> None:
        """Worker: download, optionally split GIF/video frames, append manifest rows."""
        ext = post.ext.lstrip(".") or "jpg"
        if is_blocked_download_ext(ext):
            self._reject(post, "skipped_blocked_ext")
            return
        fname = f"{post.md5}.{ext}"
        dest = self.images_dir / fname
        if not dest.is_file() and not self._download_image(post, dest):
            self._reject(post, "errors")
            return
        if sniff_media_kind(dest) in {"html", "zip", "swf", "empty", "missing"}:
            self._reject(post, "skipped_invalid_media", dest)
            return

        source = f"{post.site}:{post.id}"
        split = self.split_frames and is_splittable_ext(ext)
        frames: list[ExtractedFrame] = []
        if split:
            try:
                frames = extract_training_frames(
                    dest, self.images_dir, post.md5, ext, fps=self.frame_fps, max_frames=self.max_frames_per_post
                )
            except Exception:
                frames = []
        if frames:
            with self._lock:
                self.stats.posts_split += 1
                self.stats.frames_extracted += len(frames)
            for fr in frames:
                self._write_manifest_row(
                    {
                        "image_path": fr.rel_path,
                        "caption": post.caption,
                        "rating": post.rating,
                        "md5": fr.md5,
                        "parent_md5": post.md5,
                        "frame_index": fr.frame_index,
                        "source": source,
                        "width": fr.width,
                        "height": fr.height,
                        **post.meta_tags(),
                    }
                )
            if self.delete_raw_after_split:
                _unlink(dest)
                _unlink(dest.with_suffix(dest.suffix + ".part"))
            return
        if split and needs_frame_split(dest, ext):
            self._reject(post, "errors", dest)
            return

        # Still image — normalize GIF to JPEG; reject unreadable rasters.
        out_fname = fname
        jpg = self.images_dir / f"{post.md5}.jpg"
        if ext.lower() == "gif" and save_still_as_jpeg(dest, jpg):
            _unlink(dest)
            out_fname = jpg.name
        elif not validate_trainable_image(dest):
            self._reject(post, "skipped_invalid_media", dest)
            return
        self._write_manifest_row(
            {
                "image_path": str(Path("images") / out_fname),
                "caption": post.caption,
                "rating": post.rating,
                "md5": post.md5,
                "source": source,
                "width": post.width,
                "height": post.height,
                **post.meta_tags(),
            }
        )

    def run(self, tags: str, *, max_posts: int, dry_run: bool = False) -> ScrapeStats:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        for part in self.images_dir.glob("*.part"):
            _unlink(part)
        self._load_resume_state()
        self._manifest_fh = None if dry_run else open(self.manifest_path, "a", encoding="utf-8")
        # Pagination stays sequential and rate-limited; downloads overlap in a thread pool.
        pool = None if dry_run else ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            for post in self._iter_posts(tags, max_posts=max_posts):
                with self._lock:
                    self.stats.fetched += 1
                if not self._accept(post):
                    continue
                if pool is None:
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
                self._manifest_pending = 0
        return self.stats

    def _iter_posts(self, tags: str, *, max_posts: int) -> Iterator[Post]:
        """Yield posts, using id-cursor pagination when the site supports it."""
        if hasattr(self.adapter, "iter_posts"):
            yield from self.adapter.iter_posts(
                self.session, tags, max_posts=max_posts, limiter=self.api_limiter, timeout_s=self.timeout_s
            )
            return
        yielded = 0
        page = self.adapter.first_page
        before_id: int | None = None
        use_cursor = bool(getattr(self.adapter, "cursor_supported", False))
        while not post_cap_reached(yielded, max_posts):
            params = (
                self.adapter.build_params(tags, page, before_id)
                if use_cursor
                else self.adapter.build_params(tags, page)
            )
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
                if str(post.id).isdigit():
                    pid = int(post.id)
                    min_id = pid if min_id is None else min(min_id, pid)
            if use_cursor:
                if min_id is None or min_id <= 1:
                    break  # reached the start of the site
                before_id = min_id
            else:
                page += 1


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


# --- site adapters ----------------------------------------------------------------------
# All official JSON endpoints:
#   danbooru   https://danbooru.donmai.us/posts.json        (login + api_key)
#   e621       https://e621.net/posts.json                  (HTTP basic: login + api_key)
#   rule34xxx  https://api.rule34.xxx/index.php?...&json=1  (api_key + user_id)
#   rule34xyz  https://rule34.xyz/api/v2/post/search/root   (JWT from email/password)


def _ext_from_url(url: str) -> str:
    if "." in url.rsplit("/", 1)[-1]:
        return url.rsplit(".", 1)[-1].split("?")[0].lower()
    return ""


def _split_tag_field(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [t for t in str(value).split() if t.strip()]


class DanbooruAdapter:
    site = "danbooru"
    posts_url = "https://danbooru.donmai.us/posts.json"
    first_page = 1
    # danbooru limits anonymous/basic accounts to 2 search tags, so the blocklist cannot be injected here;
    # the per-post gate in BooruClient is authoritative.
    page_limit = 200
    cursor_supported = True  # numeric paging caps at ~page 1000; page=b<id> crawls the whole site

    def __init__(self, creds: SiteCredentials) -> None:
        self.creds = creds
        self.auth: tuple[str, str] | None = None

    def build_params(self, tags: str, page: int, before_id: int | None = None) -> dict:
        params = {"tags": tags, "limit": self.page_limit, "page": f"b{before_id}" if before_id else page}
        if self.creds.username and self.creds.api_key:
            params["login"] = self.creds.username
            params["api_key"] = self.creds.api_key
        return params

    def parse_posts(self, data) -> Iterator[Post]:
        if not isinstance(data, list):
            return
        for d in data:
            file_url = d.get("file_url") or d.get("large_file_url") or ""
            yield Post(
                site=self.site,
                id=str(d.get("id", "")),
                md5=d.get("md5", ""),
                file_url=file_url,
                ext=d.get("file_ext", "") or _ext_from_url(file_url),
                tags=(d.get("tag_string", "") or "").split(),
                rating=d.get("rating", ""),
                width=int(d.get("image_width", 0) or 0),
                height=int(d.get("image_height", 0) or 0),
                artist_tags=_split_tag_field(d.get("tag_string_artist")),
                character_tags=_split_tag_field(d.get("tag_string_character")),
                copyright_tags=_split_tag_field(d.get("tag_string_copyright")),
            )


class E621Adapter:
    site = "e621"
    posts_url = "https://e621.net/posts.json"
    first_page = 1
    page_limit = 320  # e621 hard max
    cursor_supported = True  # page=b<id> crawls past the deep-page cap

    def __init__(self, creds: SiteCredentials) -> None:
        self.creds = creds
        # e621 accepts credentials via HTTP basic auth.
        self.auth: tuple[str, str] | None = (
            (creds.username, creds.api_key) if creds.username and creds.api_key else None
        )

    def build_params(self, tags: str, page: int, before_id: int | None = None) -> dict:
        # e621 allows up to 40 tags, so the blocklist is negated in-query too.
        full_tags = " ".join([tags, *blocked_query_tags()]).strip()
        return {"tags": full_tags, "limit": self.page_limit, "page": f"b{before_id}" if before_id else page}

    def parse_posts(self, data) -> Iterator[Post]:
        posts = data.get("posts") if isinstance(data, dict) else None
        if not isinstance(posts, list):
            return
        for d in posts:
            f = d.get("file") or {}
            tag_groups = d.get("tags") or {}
            all_tags: list[str] = []
            for group in tag_groups.values():
                if isinstance(group, list):
                    all_tags.extend(group)
            yield Post(
                site=self.site,
                id=str(d.get("id", "")),
                md5=f.get("md5", ""),
                file_url=f.get("url") or "",
                ext=f.get("ext", ""),
                tags=all_tags,
                rating=d.get("rating", ""),
                width=int(f.get("width", 0) or 0),
                height=int(f.get("height", 0) or 0),
                artist_tags=list(tag_groups.get("artist") or []),
                character_tags=list(tag_groups.get("character") or []),
                copyright_tags=list(tag_groups.get("copyright") or []),
            )


class Rule34xxxAdapter:
    """Gelbooru ``page=dapi&s=post&q=index`` API at api.rule34.xxx."""

    site = "rule34xxx"
    posts_url = "https://api.rule34.xxx/index.php"
    cdn_base = "https://api-cdn.rule34.xxx"
    first_page = 0
    page_limit = 1000
    cursor_supported = False

    def __init__(self, creds: SiteCredentials) -> None:
        self.creds = creds
        self.auth = None

    def build_params(self, tags: str, page: int, before_id: int | None = None) -> dict:
        params = {
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": "1",
            "tags": tags,
            "pid": page,
            "limit": self.page_limit,
        }
        if self.creds.api_key and self.creds.user_id:
            params["api_key"] = self.creds.api_key
            params["user_id"] = self.creds.user_id
        return params

    def parse_posts(self, data) -> Iterator[Post]:
        """Gelbooru JSON: a bare array or a ``{"post": [...]}`` wrapper."""
        posts = data if isinstance(data, list) else data.get("post") if isinstance(data, dict) else None
        if isinstance(posts, dict):
            posts = [posts]
        for d in posts or []:
            if not isinstance(d, dict):
                continue
            file_url = d.get("file_url") or ""
            if not file_url and d.get("directory") is not None and d.get("image"):
                file_url = f"{self.cdn_base}/images/{d['directory']}/{d['image']}"
            yield Post(
                site=self.site,
                id=str(d.get("id", "")),
                md5=d.get("hash", "") or d.get("md5", ""),
                file_url=file_url,
                ext=_ext_from_url(file_url),
                tags=(d.get("tags", "") or "").split(),
                rating=d.get("rating", ""),
                width=int(d.get("width", 0) or 0),
                height=int(d.get("height", 0) or 0),
            )


# rule34.xyz v2 JSON API (replaced the legacy Gelbooru ``page=dapi`` in 2025).
R34XYZ_ROOT = "https://rule34.xyz"
R34XYZ_CDN = "https://rule34xyz.b-cdn.net"
R34XYZ_SEARCH_URL = f"{R34XYZ_ROOT}/api/v2/post/search/root"
R34XYZ_SIGNIN_URL = f"{R34XYZ_ROOT}/api/v2/auth/signin"
# Prefer video for animated posts, then still image preview.
_FORMAT_PRIORITY = ("100", "101", "102", "10", "11", "12", "13", "14", "31", "32", "112")
_FORMAT_EXT = {
    **{k: ("pic.jpg", "jpg") for k in ("10", "11", "12", "13", "14", "31", "32")},
    "100": ("mov.mp4", "mp4"),
    "101": ("mov720.mp4", "mp4"),
    "102": ("mov480.mp4", "mp4"),
    "112": ("mov.mp4", "mp4"),
}
_TAG_TYPES = {2: "copyright", 4: "character", 8: "artist"}


def _md5_for_post(post_id: int | str) -> str:
    return hashlib.md5(f"rule34xyz:{post_id}".encode()).hexdigest()


def _r34xyz_login(session: requests.Session, creds: SiteCredentials, *, timeout_s: float = 45.0) -> None:
    """Attach ``Authorization: Bearer …`` when email/password or a token is available."""
    if creds.api_key and str(creds.api_key).lower().startswith("bearer "):
        session.headers["Authorization"] = creds.api_key
        return
    if creds.api_key and "." in creds.api_key:
        session.headers["Authorization"] = f"Bearer {creds.api_key}"
        return
    email = (creds.email or creds.username or "").strip()
    password = (creds.password or "").strip()
    if not email or not password:
        return  # anonymous search still works for many tags
    r = session.post(R34XYZ_SIGNIN_URL, json={"email": email, "password": password}, timeout=timeout_s)
    r.raise_for_status()
    jwt = (r.json() or {}).get("jwt")
    if not jwt:
        raise RuntimeError(f"{creds.site}: sign-in succeeded but no jwt returned")
    session.headers["Authorization"] = f"Bearer {jwt}"


def file_url_from_post(post: dict) -> tuple[str, str]:
    files = post.get("files") or {}
    fmt = next((k for k in _FORMAT_PRIORITY if k in files), None) or next(iter(files), None)
    if fmt is None:
        return "", ""
    filename, ext = _FORMAT_EXT.get(fmt, (f"file.{fmt}", "bin"))
    post_id = int(post["id"])
    root = R34XYZ_CDN if (files.get(fmt) or [0])[0] else R34XYZ_ROOT
    return f"{root}/posts/{post_id // 1000}/{post_id}/{post_id}.{filename}", ext


def tags_from_post(post: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    all_tags: list[str] = []
    buckets: dict[str, list[str]] = {"artist": [], "character": [], "copyright": []}
    for tag in post.get("tags") or []:
        if not isinstance(tag, dict):
            continue
        value = str(tag.get("value", "")).strip()
        if not value:
            continue
        all_tags.append(value)
        bucket = _TAG_TYPES.get(int(tag.get("type", 0) or 0))
        if bucket:
            buckets[bucket].append(value)
    return all_tags, buckets["artist"], buckets["character"], buckets["copyright"]


def post_from_api(post: dict, *, site: str = "rule34xyz") -> Post | None:
    file_url, ext = file_url_from_post(post)
    if not file_url:
        return None
    post_id = str(post.get("id", ""))
    tags, artists, characters, copyrights = tags_from_post(post)
    return Post(
        site=site,
        id=post_id,
        md5=_md5_for_post(post_id),
        file_url=file_url,
        ext=ext,
        tags=tags,
        rating="",
        width=int(post.get("width", 0) or 0),
        height=int(post.get("height", 0) or 0),
        artist_tags=artists,
        character_tags=characters,
        copyright_tags=copyrights,
    )


class Rule34xyzAdapter:
    """rule34.xyz adapter using ``/api/v2/post/search/root`` (POST JSON)."""

    site = "rule34xyz"
    posts_url = R34XYZ_SEARCH_URL
    first_page = 0
    page_limit = 60
    cursor_supported = False
    auth = None

    def __init__(self, creds: SiteCredentials) -> None:
        self.creds = creds

    def iter_posts(
        self, session: requests.Session, tags: str, *, max_posts: int, limiter: RateLimiter, timeout_s: float
    ) -> Iterator[Post]:
        _r34xyz_login(session, self.creds, timeout_s=timeout_s)
        include_tags = tags.split() if tags else []
        skip = 0
        cursor: str | None = None
        yielded = 0
        per_page = self.page_limit
        while not post_cap_reached(yielded, max_posts):
            limiter.wait()
            take = per_page if max_posts <= 0 else min(per_page, posts_remaining(max_posts, yielded) or per_page)
            body: dict = {"Skip": skip, "take": take, "CountTotal": False, "IncludeLinks": True, "OrderBy": 0}
            if include_tags:
                body["includeTags"] = include_tags
            if cursor:
                body["cursor"] = cursor
            r = session.post(R34XYZ_SEARCH_URL, json=body, timeout=timeout_s)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} from rule34.xyz search", response=r)
            r.raise_for_status()
            data = r.json() or {}
            items = data.get("items") or []
            if not items:
                break
            for item in items:
                if post_cap_reached(yielded, max_posts):
                    return
                post_data = item
                if not post_data.get("tags"):
                    limiter.wait()
                    pr = session.get(f"{R34XYZ_ROOT}/api/v2/post/{item['id']}", timeout=timeout_s)
                    pr.raise_for_status()
                    post_data = pr.json()
                post = post_from_api(post_data)
                if post is None:
                    continue
                yield post
                yielded += 1
            if len(items) < per_page:
                break
            skip += len(items)
            cursor = data.get("cursor")


ADAPTERS = {
    "danbooru": DanbooruAdapter,
    "e621": E621Adapter,
    "rule34xxx": Rule34xxxAdapter,
    "rule34xyz": Rule34xyzAdapter,
}

# Default scrape set for RunPod (all booru sites in runpod/secrets.example.txt).
DEFAULT_SCRAPE_SITES = ("danbooru", "rule34xxx", "e621", "rule34xyz")

# API pagination rate limits (req/s); image CDN downloads use a separate pool and do NOT consume this budget.
#   danbooru: 10 reads/s global cap — stay under; rule34xxx: Gelbooru-family ~10/s with creds;
#   e621: hard 2/s — never exceed 1.5; rule34xyz: JWT v2 API — keep conservative.
RATE_LIMITS = {"danbooru": 9.5, "e621": 1.5, "rule34xxx": 9.5, "rule34xyz": 2.0}


def build_adapter(site: str, creds: SiteCredentials):
    site = site.lower()
    if site not in ADAPTERS:
        raise ValueError(f"Unsupported site {site!r}. Choices: {sorted(ADAPTERS)}")
    return ADAPTERS[site](creds)


# --- CLI ------------------------------------------------------------------------------

# Rating letters differ per site. danbooru: g(eneral) s(ensitive) q e; e621/rule34: s(afe) q e.
# So "safe" must accept both g and s to be portable.
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


def parse_ratings(spec: str) -> set[str] | None:
    spec = (spec or "").strip().lower()
    if not spec or spec == "all":
        return None
    out: set[str] = set()
    for token in spec.replace(",", " ").split():
        out |= _RATING_ALIASES.get(token, set())
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
        help="Target fps when sampling video frames (default 1.0 = one frame per second; 0 = native).",
    )
    p.add_argument(
        "--max-frames-per-post", type=int, default=120, help="Cap frames per GIF/video post (default 120, 0 = all)."
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
    if adapter.auth is None and not creds.api_key:
        print(
            f"WARNING: no API credentials resolved for {args.site}; requests will be anonymous (rate-limited).",
            file=sys.stderr,
        )
    user_agent = args.user_agent or f"sdx-dataset-scraper/1.0 (by {creds.username or creds.email or 'anonymous'})"
    rate = args.rate if args.rate is not None else RATE_LIMITS.get(args.site, 2.0)
    client = BooruClient(
        adapter,
        args.out,
        user_agent=user_agent,
        rate_per_sec=rate,
        ratings=parse_ratings(args.ratings),
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
