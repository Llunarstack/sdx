"""Validate and classify downloaded booru media before manifest rows are written."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

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

_IMAGE_MAGIC = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"RIFF", ".webp"),  # WEBP after RIFF....WEBP
    (b"BM", ".bmp"),
)


def normalize_ext(ext: str) -> str:
    e = (ext or "").lower().strip()
    if not e:
        return ""
    return e if e.startswith(".") else f".{e}"


def is_blocked_download_ext(ext: str) -> bool:
    return normalize_ext(ext) in BLOCKED_DOWNLOAD_EXTS


def is_trainable_image_ext(ext: str) -> bool:
    return normalize_ext(ext) in TRAINABLE_IMAGE_EXTS


def sniff_media_kind(path: Path) -> str:
    """Return a short kind label: image, video, zip, html, unknown."""
    try:
        head = path.read_bytes()[:32]
    except OSError:
        return "missing"
    if not head:
        return "empty"
    if head.startswith((b"<!DOCTYPE", b"<html", b"<?xml", b"<HTML")):
        return "html"
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return "zip"
    if head[:3] == b"CWS" or head[:3] == b"FWS":
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
    if not path.is_file():
        return False
    if sniff_media_kind(path) not in {"image"}:
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
