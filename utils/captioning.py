"""Image understanding for captions and references: WD tagger, OCR / VLM captions / control maps, SauceNAO + TinEye reverse search with booru tags, web reference-image search, multi-source profiling and diffusion-prompt research."""

from __future__ import annotations

import csv
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from utils.prompt.scene import PromptFacets, decompose_prompt_facets

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None  # type: ignore[assignment,misc]

_IDENTITY_CATEGORIES = frozenset({"artist", "copyright", "character", "meta", "rating"})
# selected_tags.csv stores Danbooru category ids; 9 is the tagger's rating head.
_CATEGORY_NAMES = {"0": "general", "1": "artist", "3": "copyright", "4": "character", "5": "meta", "9": "rating"}
_DEFAULT_THRESHOLD = 0.35
_DEFAULT_MAX_TAGS = 48
_MODEL_INPUT_SIZE = 448


def default_tagger_dir() -> Path:
    root = os.environ.get("SDX_PRETRAINED", "").strip()
    if root:
        return Path(root) / "WD-EVA02-Large-Tagger"
    return Path(__file__).resolve().parents[2] / "pretrained" / "WD-EVA02-Large-Tagger"


@dataclass(frozen=True)
class WDTag:
    name: str
    category: str
    score: float


def merge_wd_tags_into_caption(
    caption: str,
    wd_tags: Sequence[str],
    *,
    identity_tags: Iterable[str] | None = None,
    max_add: int = 32,
) -> str:
    """Append WD tags not already present; never override booru identity tags."""
    cap = (caption or "").strip()
    seen: set[str] = {t.strip().lower().replace("_", " ") for t in cap.split(",") if t.strip()}
    for t in identity_tags or ():
        t = str(t).strip().replace("_", " ")
        if t:
            seen.add(t.lower())
    added: list[str] = []
    for raw in wd_tags:
        tag = str(raw).strip().replace("_", " ")
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        added.append(tag)
        if len(added) >= max_add:
            break
    if not added:
        return cap
    return (cap + ", " + ", ".join(added)).strip(", ")


def merge_wd_tags_into_row(row: dict, wd_tag_names: Sequence[str]) -> dict:
    """Patch a manifest row with merged caption + ``wd_tags`` metadata."""
    merged = dict(row)
    identity: list[str] = []
    for key in ("artist_tags", "character_tags", "copyright_tags"):
        for t in row.get(key) or []:
            identity.append(str(t))
    cap = str(row.get("caption") or row.get("tags") or "").strip()
    merged["caption"] = merge_wd_tags_into_caption(cap, wd_tag_names, identity_tags=identity)
    merged["wd_tags"] = list(wd_tag_names)
    sources = [str(s) for s in (row.get("tag_sources") or [])]
    if "wd_tagger" not in sources:
        sources.append("wd_tagger")
    merged["tag_sources"] = sources
    if row.get("caption") and not merged.get("booru_caption"):
        merged["booru_caption"] = row["caption"]
    return merged


class WDTagger:
    """ONNX WD tagger (SmilingWolf/wd-eva02-large-tagger-v3)."""

    def __init__(
        self,
        model_dir: str | os.PathLike[str] | None = None,
        *,
        threshold: float = _DEFAULT_THRESHOLD,
        max_tags: int = _DEFAULT_MAX_TAGS,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else default_tagger_dir()
        self.threshold = float(threshold)
        self.max_tags = int(max_tags)
        self._session = None
        self._tag_names: list[str] = []
        self._tag_categories: list[str] = []

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        if np is None or Image is None:
            raise ModuleNotFoundError("numpy and Pillow are required for WD tagger inference.")
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ModuleNotFoundError(
                "onnxruntime is required for WD tagger. Install: pip install onnxruntime-gpu"
            ) from e

        onnx_path = self.model_dir / "model.onnx"
        csv_path = self.model_dir / "selected_tags.csv"
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"WD tagger ONNX not found: {onnx_path}\n"
                "Run: python setup/download_pretrained.py --only WD-EVA02-Large-Tagger"
            )
        if not csv_path.is_file():
            raise FileNotFoundError(f"WD tagger tag list not found: {csv_path}")

        with csv_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = (row.get("name") or row.get("tag") or "").strip()
                cat = (row.get("category") or "general").strip().lower()
                cat = _CATEGORY_NAMES.get(cat, cat)
                if name:
                    self._tag_names.append(name)
                    self._tag_categories.append(cat)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._input_size = inp.shape[1] if isinstance(inp.shape[1], int) else _MODEL_INPUT_SIZE

    @property
    def tag_names(self) -> list[str]:
        self._ensure_loaded()
        return list(self._tag_names)

    @staticmethod
    def _preprocess(image: Image.Image, size: int) -> np.ndarray:
        """SmilingWolf's tagger input: alpha over white, white-padded square, bicubic, BGR NHWC in [0, 255]."""
        image = image.convert("RGBA")
        w, h = image.size
        if w == 0 or h == 0:
            raise ValueError("empty image")
        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        canvas.alpha_composite(image)
        side = max(w, h)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(canvas.convert("RGB"), ((side - w) // 2, (side - h) // 2))
        if side != size:
            square = square.resize((size, size), Image.Resampling.BICUBIC)
        arr = np.asarray(square, dtype=np.float32)[:, :, ::-1]
        return np.ascontiguousarray(arr[None])

    def scores(self, image_path: str | os.PathLike[str]) -> list[WDTag]:
        """Every tag with its probability, highest first (no threshold or category filter)."""
        self._ensure_loaded()
        assert np is not None and Image is not None
        with Image.open(image_path) as img:
            batch = self._preprocess(img, self._input_size)
        probs = self._session.run(None, {self._input_name: batch})[0][0]  # the graph ends in a sigmoid
        tags = [
            WDTag(name=n, category=c, score=float(p)) for n, c, p in zip(self._tag_names, self._tag_categories, probs, strict=True)
        ]
        tags.sort(key=lambda t: t.score, reverse=True)
        return tags

    def select(self, tags: Sequence[WDTag]) -> list[WDTag]:
        """Caption tags: above ``threshold``, no identity/rating tags, at most ``max_tags``."""
        kept = [t for t in tags if t.score >= self.threshold and t.category not in _IDENTITY_CATEGORIES]
        return kept[: self.max_tags]


@dataclass(slots=True)
class ImageUnderstanding:
    path: str
    caption: str = ""
    ocr_text: str = ""
    tags: list[str] = field(default_factory=list)
    control_maps: dict[str, str] = field(default_factory=dict)
    width: int = 0
    height: int = 0
    source: str = "local"
    metadata: dict[str, Any] = field(default_factory=dict)


def ocr_image(path: str, *, device: str = "cuda") -> str:
    """OCR via TrOCR / GOT-OCR when weights exist, else pytesseract."""
    try:
        from utils.modeling.hf_loaders import ocr_got, ocr_trocr, ocr_trocr_handwritten

        for fn in (ocr_trocr, ocr_trocr_handwritten, ocr_got):
            text = fn(path, device=device)
            if text:
                return text
    except Exception:
        pass
    try:
        import pytesseract

        img = Image.open(path).convert("RGB")
        return (pytesseract.image_to_string(img) or "").strip()
    except Exception:
        return ""


def caption_image_vlm(path: str, *, user_prompt: str = "", device: str = "cuda") -> str:
    """Caption via HF chain (moondream → BLIP-2 → Florence-2 → …) with heuristic fallback."""
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        from utils.modeling.hf_loaders import caption_image_chain

        cap, _backend = caption_image_chain(str(p), user_prompt=user_prompt, device=device)
        if cap:
            return cap
    except Exception:
        pass
    try:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        arr = np.array(img.resize((64, 64)))
        mean = arr.mean(axis=(0, 1))
        return f"reference image {w}x{h}, average color RGB {mean.astype(int).tolist()}"
    except Exception:
        return f"reference image at {p.name}"


def extract_control_maps(
    path: str,
    output_dir: Path,
    *,
    types: Sequence[str] = ("canny", "softedge", "hed", "depth", "normals"),
    device: str = "cuda",
) -> dict[str, str]:
    """Build control maps via ``hf_control`` (PIL proxies + HF depth/normals)."""
    try:
        from utils.modeling.hf_loaders import extract_control_maps_batch

        return extract_control_maps_batch(path, output_dir, types=types, device=device)
    except Exception:
        pass
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    maps: dict[str, str] = {}
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return maps
    if "canny" in types:
        gray = img.convert("L").filter(ImageFilter.FIND_EDGES)
        dest = out_dir / f"{stem}_canny.png"
        gray.save(dest)
        maps["canny"] = str(dest)
    return maps


def understand_image(
    path: str,
    *,
    user_prompt: str = "",
    work_dir: Path,
    source: str = "local",
    run_ocr: bool = True,
    run_vlm: bool = True,
    run_control: bool = True,
    device: str = "cuda",
) -> ImageUnderstanding:
    """Full understanding pass for one reference image."""
    p = Path(path)
    u = ImageUnderstanding(path=str(p.resolve()), source=source)
    if not p.is_file():
        return u
    try:
        img = Image.open(p)
        u.width, u.height = img.size
    except Exception:
        pass
    ctrl_dir = Path(work_dir) / "control_maps"
    if run_ocr:
        u.ocr_text = ocr_image(str(p), device=device)
    if run_vlm:
        u.caption = caption_image_vlm(str(p), user_prompt=user_prompt, device=device)
    if run_control:
        u.control_maps = extract_control_maps(str(p), ctrl_dir, device=device)
    if user_prompt.strip():
        try:
            from utils.modeling.hf_loaders import detect_objects

            tokens = [t.strip() for t in user_prompt.replace(",", " ").split() if len(t.strip()) > 3][:8]
            if tokens:
                hits = detect_objects(str(p), tokens, device=device)
                if hits:
                    u.metadata["detected_objects"] = hits
                    u.tags.append("object_detect")
        except Exception:
            pass
    tags: list[str] = []
    if u.ocr_text:
        tags.append("has_text")
    if u.caption:
        tags.append("vlm_caption")
    if u.control_maps:
        tags.extend(f"control_{k}" for k in u.control_maps)
    u.tags = tags
    return u


def understand_images(
    paths: Sequence[str],
    *,
    user_prompt: str = "",
    work_dir: Path,
    sources: Sequence[str] | None = None,
    **kwargs: Any,
) -> list[ImageUnderstanding]:
    """Understand many references in order."""
    srcs = list(sources or ["local"] * len(paths))
    out: list[ImageUnderstanding] = []
    for i, path in enumerate(paths):
        src = srcs[i] if i < len(srcs) else "local"
        out.append(
            understand_image(
                path,
                user_prompt=user_prompt,
                work_dir=work_dir,
                source=src,
                **kwargs,
            )
        )
    return out


_BOORU_URL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"danbooru\.donmai\.us/posts/(\d+)", re.I), "danbooru"),
    (re.compile(r"safebooru\.donmai\.us/posts/(\d+)", re.I), "danbooru"),
    (re.compile(r"e621\.net/posts/(\d+)", re.I), "e621"),
    (
        re.compile(
            r"(?:pixiv\.net/(?:en/)?artworks/(\d+)|pixiv\.net/member_illust\.php\?[^#]*illust_id=(\d+))",
            re.I,
        ),
        "pixiv",
    ),
]

_RATE_LOCK = threading.Lock()
_LAST_REQ: dict[str, float] = {"saucenao": 0.0, "tineye": 0.0}
_MIN_INTERVAL = {"saucenao": 3.5, "tineye": 2.0}


@dataclass
class ReverseHit:
    similarity: float
    site: str
    site_id: str
    engine: str = "saucenao"
    author: str = ""
    material: str = ""  # copyright / series hint
    characters: list[str] = field(default_factory=list)
    external_url: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class BooruPostTags:
    post_id: str
    character_tags: list[str]
    copyright_tags: list[str]
    artist_tags: list[str]
    general_tags: list[str]
    tag_string: str = ""


@lru_cache(maxsize=8)
def get_api_key(service: str, secrets_path: str | None = None) -> str:
    """Optional SauceNAO/TinEye key from ``<SERVICE>_API_KEY`` or secret.txt; web upload works without one."""
    svc = service.strip().lower()
    from_env = (os.environ.get(f"{svc.upper()}_API_KEY") or "").strip()
    if from_env:
        return from_env
    try:
        from scripts.scrape import get_credentials

        return (get_credentials(svc, secrets_path).api_key or "").strip()
    except Exception:
        return ""


def fetch_booru_post(site: str, post_id: str, *, timeout_s: float = 30.0) -> BooruPostTags | None:
    """Tags of a Danbooru or e621 post (authenticated with secret.txt credentials when present)."""
    pid = str(post_id).strip()
    if site not in ("danbooru", "e621") or not pid.isdigit():
        return None
    import requests

    try:
        from scripts.scrape import get_credentials, get_secrets_path

        creds = get_credentials(site, get_secrets_path())
        login = (creds.username, creds.api_key) if creds.username and creds.api_key else None
    except Exception:
        login = None
    headers = {"User-Agent": "sdx-image-profiler/1.0"}
    try:
        if site == "danbooru":
            params = {"login": login[0], "api_key": login[1]} if login else {}
            url = f"https://danbooru.donmai.us/posts/{pid}.json"
            r = requests.get(url, params=params, timeout=timeout_s, headers=headers)
        else:
            r = requests.get(f"https://e621.net/posts/{pid}.json", auth=login, timeout=timeout_s, headers=headers)
        r.raise_for_status()
        d = r.json()
    except Exception:
        return None
    if site == "danbooru":
        groups = {
            k: str(d.get(f"tag_string_{k}") or "").split() for k in ("character", "copyright", "artist", "general")
        }
        tag_string = str(d.get("tag_string") or "")
    else:
        raw = (d.get("post") or {}).get("tags") or {}
        groups = {k: [str(x) for x in v if str(x).strip()] for k, v in raw.items() if isinstance(v, list)}
        tag_string = " ".join(t for v in groups.values() for t in v)
    return BooruPostTags(
        post_id=pid,
        character_tags=groups.get("character", []),
        copyright_tags=groups.get("copyright", []),
        artist_tags=groups.get("artist", []),
        general_tags=groups.get("general", []),
        tag_string=tag_string,
    )


def parse_source_url(url: str) -> tuple[str, str]:
    """Extract ``(site, post_id)`` from a booru/pixiv URL."""
    u = (url or "").strip()
    if not u:
        return "", ""
    for pat, site in _BOORU_URL_PATTERNS:
        m = pat.search(u)
        if not m:
            continue
        post_id = next((g for g in m.groups() if g), "")
        if post_id:
            return site, post_id
    return "", ""


def _rate_limit(engine: str) -> None:
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _MIN_INTERVAL.get(engine, 2.0) - (now - _LAST_REQ.get(engine, 0.0))
        if wait > 0:
            time.sleep(wait)
        _LAST_REQ[engine] = time.monotonic()


def _iter_urls_from_tineye_match(match: dict) -> Iterable[str]:
    if match.get("image_url"):
        yield str(match["image_url"])
    for backlink in match.get("backlinks") or []:
        if backlink.get("url"):
            yield str(backlink["url"])
        if backlink.get("backlink"):
            yield str(backlink["backlink"])


def _hits_from_saucenao_json(payload: dict) -> list[ReverseHit]:
    hits: list[ReverseHit] = []
    for result in payload.get("results") or []:
        header = result.get("header") or {}
        data = result.get("data") or {}
        sim = float(header.get("similarity") or 0.0)
        ext = data.get("ext_urls") or []
        ext_url = str(ext[0]) if ext else ""
        site = ""
        site_id = ""
        author = str(data.get("member_name") or data.get("author_name") or "")
        material = str(data.get("material") or data.get("source") or "")
        chars: list[str] = []
        if "danbooru_id" in data:
            site = "danbooru"
            site_id = str(data["danbooru_id"])
        elif "pixiv_id" in data:
            site = "pixiv"
            site_id = str(data["pixiv_id"])
        elif ext_url:
            site, site_id = parse_source_url(ext_url)
            if not site and "e621" in ext_url:
                site = "e621"
            elif not site and "danbooru" in ext_url:
                site = "danbooru"
        char = data.get("char") or data.get("characters")
        if isinstance(char, str) and char.strip():
            chars = [c.strip() for c in char.split(",") if c.strip()]
        hits.append(
            ReverseHit(
                similarity=sim,
                site=site,
                site_id=site_id,
                engine="saucenao",
                author=author,
                material=material,
                characters=chars,
                external_url=ext_url,
                raw=data,
            )
        )
    hits.sort(key=lambda h: -h.similarity)
    return hits


def _hits_from_saucenao_html(html: str) -> list[ReverseHit]:
    """Parse SauceNAO HTML results page when JSON is unavailable."""
    hits: list[ReverseHit] = []
    blocks = re.split(r'class="resultadosub"', html, flags=re.I)
    for block in blocks[1:6]:
        sim_m = re.search(r"(\d+(?:\.\d+)?)\s*%", block)
        sim = float(sim_m.group(1)) if sim_m else 0.0
        url_m = re.search(r'href="(https?://[^"]+)"', block)
        ext_url = url_m.group(1) if url_m else ""
        site, site_id = parse_source_url(ext_url)
        if not site:
            if "danbooru" in block.lower():
                dm = re.search(r"danbooru\.donmai\.us/posts/(\d+)", block, re.I)
                if dm:
                    site, site_id = "danbooru", dm.group(1)
            elif "e621" in block.lower():
                em = re.search(r"e621\.net/posts/(\d+)", block, re.I)
                if em:
                    site, site_id = "e621", em.group(1)
        material = ""
        mat_m = re.search(r"Material:\s*([^<\n]+)", block, re.I)
        if mat_m:
            material = mat_m.group(1).strip()
        chars: list[str] = []
        char_m = re.search(r"Characters?:\s*([^<\n]+)", block, re.I)
        if char_m:
            chars = [c.strip() for c in char_m.group(1).split(",") if c.strip()]
        if sim > 0 or site:
            hits.append(
                ReverseHit(
                    similarity=sim,
                    site=site,
                    site_id=site_id,
                    engine="saucenao",
                    material=material,
                    characters=chars,
                    external_url=ext_url,
                    raw={"html_block": block[:500]},
                )
            )
    if not hits:
        for sim_s, url in re.findall(
            r"(\d+(?:\.\d+)?)\s*%[\s\S]{0,400}?href=\"(https?://[^\"]+)\"",
            html,
            flags=re.I,
        )[:5]:
            site, site_id = parse_source_url(url)
            hits.append(
                ReverseHit(
                    similarity=float(sim_s),
                    site=site,
                    site_id=site_id,
                    engine="saucenao",
                    external_url=url,
                    raw={},
                )
            )
    hits.sort(key=lambda h: -h.similarity)
    return hits


def _multipart_post(url: str, field_name: str, filename: str, data: bytes, *, timeout_s: float) -> bytes:
    boundary = "----sdxboundary"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Mozilla/5.0 (compatible; SDX-ImageProfiler/1.0)",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def saucenao_search_web(image_path: str | Path, *, timeout_s: float = 45.0) -> list[ReverseHit]:
    """Upload image to SauceNAO web form — same as the browser, no API key."""
    p = Path(image_path)
    if not p.is_file():
        return []
    _rate_limit("saucenao")
    raw = p.read_bytes()
    # JSON when possible (browser POST + output_type=2).
    for url in (
        "https://saucenao.com/search.php?output_type=2&db=999",
        "https://saucenao.com/search.php?output_type=2&dbmask=1023",
        "https://saucenao.com/search.php",
    ):
        try:
            payload_bytes = _multipart_post(url, "file", p.name, raw, timeout_s=timeout_s)
            text = payload_bytes.decode("utf-8", errors="ignore").strip()
            if text.startswith("{"):
                hits = _hits_from_saucenao_json(json.loads(text))
                if hits:
                    return hits
            if "<html" in text.lower():
                hits = _hits_from_saucenao_html(text)
                if hits:
                    return hits
        except Exception:
            continue
    return []


def saucenao_search_file(
    image_path: str | Path,
    *,
    api_key: str | None = None,
    timeout_s: float = 30.0,
    prefer_web: bool = True,
) -> list[ReverseHit]:
    """Query SauceNAO — web upload by default; API key optional for higher limits."""
    key = (api_key or get_api_key("saucenao")).strip()
    if prefer_web or not key:
        hits = saucenao_search_web(image_path, timeout_s=timeout_s)
        if hits:
            return hits
    if not key:
        return saucenao_search_web(image_path, timeout_s=timeout_s)

    p = Path(image_path)
    if not p.is_file():
        return []
    _rate_limit("saucenao")
    raw = p.read_bytes()
    url = f"https://saucenao.com/search.php?api_key={urllib.parse.quote(key)}&output_type=2&dbmask=1023"
    try:
        payload_bytes = _multipart_post(url, "file", p.name, raw, timeout_s=timeout_s)
        return _hits_from_saucenao_json(json.loads(payload_bytes.decode("utf-8", errors="ignore")))
    except Exception:
        return saucenao_search_web(image_path, timeout_s=timeout_s)


def _hits_from_tineye_json(payload: dict) -> list[ReverseHit]:
    hits: list[ReverseHit] = []
    matches = (payload.get("results") or {}).get("matches")
    if matches is None and isinstance(payload.get("matches"), list):
        matches = payload["matches"]
    for match in matches or []:
        score = float(match.get("score") or 0.0)
        domain = str(match.get("domain") or "")
        image_url = str(match.get("image_url") or "")
        site = ""
        site_id = ""
        ext_url = image_url
        for candidate in _iter_urls_from_tineye_match(match):
            parsed_site, parsed_id = parse_source_url(candidate)
            if parsed_site:
                site, site_id = parsed_site, parsed_id
                ext_url = candidate
                break
        material = domain
        if not site and domain:
            if "danbooru" in domain:
                site = "danbooru"
            elif "e621" in domain:
                site = "e621"
            elif "pixiv" in domain:
                site = "pixiv"
        hits.append(
            ReverseHit(
                similarity=score,
                site=site,
                site_id=site_id,
                engine="tineye",
                material=material,
                external_url=ext_url,
                raw=match,
            )
        )
    hits.sort(key=lambda h: -h.similarity)
    return hits


def tineye_search_web(image_path: str | Path, *, timeout_s: float = 60.0, limit: int = 10) -> list[ReverseHit]:
    """Upload image to TinEye website (same as browser) — no API key."""
    p = Path(image_path)
    if not p.is_file():
        return []
    try:
        import requests
    except ImportError:
        return []

    _rate_limit("tineye")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SDX-ImageProfiler/1.0)"})
    try:
        session.get("https://tineye.com/", timeout=timeout_s)
        raw = p.read_bytes()
        # TinEye SPA uses this internal JSON endpoint after upload.
        for field in ("image", "image_upload"):
            resp = session.post(
                "https://tineye.com/api/v1/result_json/",
                params={"sort": "score", "order": "desc", "limit": limit},
                files={field: (p.name, raw, "application/octet-stream")},
                timeout=timeout_s,
            )
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                    hits = _hits_from_tineye_json(payload)
                    if hits:
                        return hits
                except Exception:
                    pass
        # Legacy upload form returns HTML with links.
        resp = session.post(
            "https://tineye.com/search/upload",
            files={"image": (p.name, raw, "application/octet-stream")},
            timeout=timeout_s,
        )
        if resp.status_code == 200 and resp.text:
            hits: list[ReverseHit] = []
            for url in re.findall(r'href="(https?://[^"]+)"', resp.text)[:20]:
                site, site_id = parse_source_url(url)
                if site:
                    hits.append(
                        ReverseHit(
                            similarity=70.0,
                            site=site,
                            site_id=site_id,
                            engine="tineye",
                            external_url=url,
                            raw={},
                        )
                    )
            if hits:
                return hits
    except Exception:
        pass
    return []


def tineye_search_file(
    image_path: str | Path,
    *,
    api_key: str | None = None,
    timeout_s: float = 60.0,
    limit: int = 10,
    prefer_web: bool = True,
) -> list[ReverseHit]:
    """Query TinEye — web upload by default; paid API key optional."""
    key = (api_key or get_api_key("tineye")).strip()
    if prefer_web or not key:
        hits = tineye_search_web(image_path, timeout_s=timeout_s, limit=limit)
        if hits:
            return hits
    if not key:
        return tineye_search_web(image_path, timeout_s=timeout_s, limit=limit)

    p = Path(image_path)
    if not p.is_file():
        return []

    try:
        import requests
    except ImportError:
        return tineye_search_web(image_path, timeout_s=timeout_s, limit=limit)

    _rate_limit("tineye")
    url = "https://api.tineye.com/rest/search/"
    headers = {"x-api-key": key, "User-Agent": "SDX-ImageProfiler/1.0"}
    files = {"image_upload": (p.name, p.read_bytes())}
    data = {"offset": 0, "limit": str(limit), "sort": "score", "order": "desc"}
    try:
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        if int(payload.get("code") or 0) == 200:
            return _hits_from_tineye_json(payload)
    except Exception:
        pass
    return tineye_search_web(image_path, timeout_s=timeout_s, limit=limit)


def reverse_search_file(
    image_path: str | Path,
    *,
    use_saucenao: bool = True,
    use_tineye: bool = True,
    saucenao_api_key: str | None = None,
    tineye_api_key: str | None = None,
) -> list[ReverseHit]:
    """Run SauceNAO and/or TinEye via web upload; merge hits by similarity."""
    hits: list[ReverseHit] = []
    if use_saucenao:
        hits.extend(saucenao_search_file(image_path, api_key=saucenao_api_key))
    if use_tineye:
        hits.extend(tineye_search_file(image_path, api_key=tineye_api_key))
    hits.sort(key=lambda h: (-h.similarity, h.engine))
    return hits


def hit_meets_threshold(hit: ReverseHit, min_sim: float) -> bool:
    """TinEye booru URL matches can use a lower bar — we verify via booru API."""
    if hit.engine == "tineye" and hit.site in ("danbooru", "e621") and hit.site_id:
        return hit.similarity >= min(min_sim, 50.0)
    return hit.similarity >= min_sim


@dataclass(slots=True)
class ImageSearchHit:
    url: str
    title: str = ""
    source: str = "duckduckgo"
    local_path: str = ""


@dataclass(slots=True)
class ImageSearchResult:
    query: str
    hits: list[ImageSearchHit] = field(default_factory=list)
    notes: str = ""


_UA = "Mozilla/5.0 (compatible; SDX-VisualBrain/1.0)"


def _http_get(url: str, *, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _ddg_vqd(query: str) -> str | None:
    q = urllib.parse.quote_plus(query)
    html = _http_get(f"https://duckduckgo.com/?q={q}&iax=images&ia=images").decode("utf-8", errors="ignore")
    for pat in (r"vqd=([\d-]+)", r'"vqd"\s*:\s*"([^"]+)"', r"vqd='([^']+)'"):
        m = re.search(pat, html)
        if m:
            return m.group(1)
    return None


def search_images_duckduckgo(query: str, *, max_results: int = 5) -> ImageSearchResult:
    """Return image URLs from DuckDuckGo image search."""
    out = ImageSearchResult(query=str(query).strip())
    if not out.query:
        out.notes = "empty query"
        return out
    try:
        vqd = _ddg_vqd(out.query)
        if not vqd:
            out.notes = "no vqd token"
            return out
        q = urllib.parse.quote_plus(out.query)
        v = urllib.parse.quote_plus(vqd)
        js_url = f"https://duckduckgo.com/i.js?q={q}&vqd={v}&o=json"
        raw = _http_get(js_url).decode("utf-8", errors="ignore")
        data = json.loads(raw)
        for row in (data.get("results") or [])[: max(1, int(max_results))]:
            url = str(row.get("image") or row.get("thumbnail") or "").strip()
            if url:
                out.hits.append(
                    ImageSearchHit(
                        url=url,
                        title=str(row.get("title") or "")[:200],
                        source="duckduckgo",
                    )
                )
        if not out.hits:
            out.notes = "no results"
    except Exception as exc:
        out.notes = f"ddg error: {exc}"
    return out


def search_images_wikimedia(query: str, *, max_results: int = 5) -> ImageSearchResult:
    """Fallback: Wikimedia Commons file search (CC images)."""
    out = ImageSearchResult(query=str(query).strip())
    if not out.query:
        return out
    try:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {out.query}",
                "gsrlimit": str(max(1, int(max_results))),
                "prop": "imageinfo",
                "iiprop": "url",
            }
        )
        raw = _http_get(f"https://commons.wikimedia.org/w/api.php?{params}")
        data = json.loads(raw.decode("utf-8"))
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            infos = page.get("imageinfo") or []
            if infos and infos[0].get("url"):
                out.hits.append(
                    ImageSearchHit(
                        url=str(infos[0]["url"]),
                        title=str(page.get("title") or "")[:200],
                        source="wikimedia",
                    )
                )
    except Exception as exc:
        out.notes = f"wikimedia error: {exc}"
    return out


def search_reference_images(
    query: str,
    *,
    max_results: int = 5,
    allow_web: bool = True,
) -> ImageSearchResult:
    """Search online; try DDG then Wikimedia."""
    if not allow_web or not str(query or "").strip():
        return ImageSearchResult(query=str(query or ""), notes="web disabled or empty")
    res = search_images_duckduckgo(query, max_results=max_results)
    if res.hits:
        return res
    wiki = search_images_wikimedia(query, max_results=max_results)
    if wiki.hits:
        return wiki
    if not res.notes:
        res.notes = wiki.notes or "no hits"
    return res


def download_search_hits(
    hits: list[ImageSearchHit],
    output_dir: Path,
    *,
    max_download: int = 5,
) -> list[ImageSearchHit]:
    """Download remote URLs to *output_dir*; set ``local_path`` on success."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    updated: list[ImageSearchHit] = []
    for i, hit in enumerate(hits[: max(1, int(max_download))]):
        if not hit.url:
            continue
        ext = ".jpg"
        if ".png" in hit.url.lower():
            ext = ".png"
        dest = out_dir / f"search_{i:02d}{ext}"
        try:
            data = _http_get(hit.url, timeout=25.0)
            if len(data) < 512:
                continue
            dest.write_bytes(data)
            updated.append(
                ImageSearchHit(
                    url=hit.url,
                    title=hit.title,
                    source=hit.source,
                    local_path=str(dest),
                )
            )
        except Exception:
            continue
    return updated


@dataclass
class FacetRefPack:
    facets: dict[str, Any]
    queries: dict[str, str] = field(default_factory=dict)
    paths: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    nsfw: bool = False

    def moodboard_paths(self, *, prefer: tuple[str, ...] = ("style", "subject", "attire")) -> list[str]:
        return list(dict.fromkeys(p for key in prefer for p in self.paths.get(key) or []))


def gather_facet_references(
    prompt: str | PromptFacets,
    *,
    work_dir: str | Path,
    max_per_facet: int = 4,
    allow_web: bool = True,
    facets_filter: tuple[str, ...] = ("subject", "attire", "style"),
) -> FacetRefPack:
    """Search and download refs into work_dir/refs/{facet}/."""
    if isinstance(prompt, PromptFacets):
        facets = prompt
    else:
        facets = decompose_prompt_facets(prompt)

    work = Path(work_dir)
    pack = FacetRefPack(facets=facets.to_dict(), nsfw=facets.nsfw)
    pack.queries = dict(facets.search_queries)

    for facet in facets_filter:
        q = facets.search_queries.get(facet) or ""
        if not q.strip():
            continue
        # NSFW prompts: broader query wording (provider safe-search is not always controllable).
        query = q
        if facets.nsfw and facet in ("subject", "attire"):
            query = f"{q} (reference art)"
        res = search_reference_images(query, max_results=max_per_facet, allow_web=allow_web)
        if not res.hits:
            pack.notes.append(f"{facet}: no hits ({res.notes or 'empty'})")
            continue
        dest = work / "refs" / facet
        try:
            hits = download_search_hits(res.hits, dest, max_download=max_per_facet)
        except Exception as exc:
            pack.notes.append(f"{facet}: download failed: {exc}")
            continue
        pack.paths[facet] = [str(h.local_path) for h in hits if getattr(h, "local_path", None)]
        pack.notes.append(f"{facet}: {len(pack.paths[facet])} files from '{query}'")
    return pack


_VLM_SAFE_EXTRA = frozenset(
    {
        "solo",
        "1girl",
        "1boy",
        "2girls",
        "2boys",
        "outdoors",
        "indoors",
        "sky",
        "night",
        "day",
        "rain",
        "snow",
        "sunset",
        "forest",
        "city",
        "beach",
        "school uniform",
        "long hair",
        "short hair",
        "looking at viewer",
        "smile",
        "standing",
        "sitting",
        "lying",
    }
)


@dataclass
class ImageProfile:
    caption: str
    scene_summary: str = ""
    character_tags: list[str] = field(default_factory=list)
    copyright_tags: list[str] = field(default_factory=list)
    artist_tags: list[str] = field(default_factory=list)
    extra_tags: list[str] = field(default_factory=list)
    is_original_character: bool = False
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)

    def to_manifest_patch(self) -> dict:
        out = {
            "caption": self.caption,
            "scene_summary": self.scene_summary,
            "tag_confidence": round(self.confidence, 4),
            "tag_sources": list(self.sources),
        }
        if self.character_tags:
            out["character_tags"] = self.character_tags
        if self.copyright_tags:
            out["copyright_tags"] = self.copyright_tags
        if self.artist_tags:
            out["artist_tags"] = self.artist_tags
        if self.is_original_character:
            out["is_original_character"] = True
        return out


def _norm_tag(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower().replace("_", " "))


def _merge_tags(*groups: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for g in groups:
        for t in g:
            n = _norm_tag(t)
            if not n or n in seen:
                continue
            seen.add(n)
            out.append(n)
    return out


def _build_caption(
    *,
    characters: Sequence[str],
    copyrights: Sequence[str],
    artists: Sequence[str],
    general: Sequence[str],
) -> str:
    parts = _merge_tags(characters, copyrights, artists, general)
    return ", ".join(parts)


def _summary_from_tags(
    *,
    characters: Sequence[str],
    copyrights: Sequence[str],
    artists: Sequence[str],
    caption: str,
) -> str:
    bits: list[str] = []
    if characters:
        bits.append(f"Characters: {', '.join(_merge_tags(characters))}.")
    if copyrights:
        bits.append(f"Series: {', '.join(_merge_tags(copyrights))}.")
    if artists:
        bits.append(f"Artist: {', '.join(_merge_tags(artists))}.")
    if caption:
        bits.append(f"Tags: {caption}.")
    return " ".join(bits).strip()


def _parse_row_lists(row: dict, key: str) -> list[str]:
    val = row.get(key)
    if not val:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    return [t.strip() for t in str(val).split(",") if t.strip()]


def _vlm_scene_summary(image_path: Path, *, device: str = "cuda") -> tuple[str, list[str]]:
    prompt = (
        "Describe this image in detail: who is present, what are they doing, "
        "where is the scene, clothing, expression, composition, lighting, and mood. "
        "If the character looks like a known anime/game character, name them and the series. "
        "If it appears to be an original character, say so."
    )
    try:
        summary = caption_image_vlm(str(image_path), user_prompt=prompt, device=device)
    except Exception:
        summary = ""
    if not summary:
        return "", []

    extras: list[str] = []
    low = summary.lower()
    for tag in _VLM_SAFE_EXTRA:
        if tag in low:
            extras.append(tag)
    if re.search(r"\boriginal character\b|\boc\b|\boriginal design\b", low):
        extras.append("original_character")
    return summary, extras


def _apply_reverse_hit(hit, *, characters, copyrights, artists, sources, confidence):
    sources.append(f"{hit.engine}:{hit.similarity:.0f}%")
    confidence = max(confidence, min(0.99, hit.similarity / 100.0))
    if hit.characters:
        characters.extend(hit.characters)
    if hit.material:
        for part in re.split(r"[/|]", hit.material):
            part = part.strip()
            if part and len(part) > 2:
                copyrights.append(part)
    if hit.author and not artists:
        artists.append(hit.author)
    if hit.site in ("danbooru", "e621") and hit.site_id:
        post = fetch_booru_post(hit.site, hit.site_id)
        if post:
            sources.append(f"{hit.site}_api:{hit.site_id}")
            confidence = max(confidence, 0.98)
            characters.extend(post.character_tags)
            copyrights.extend(post.copyright_tags)
            artists.extend(post.artist_tags)
    return confidence


def profile_image(
    image_path: str | Path,
    *,
    booru_row: dict | None = None,
    use_reverse_search: bool = True,
    use_saucenao: bool = True,
    use_tineye: bool = True,
    use_vlm: bool = True,
    reverse_min_sim: float = 75.0,
    device: str = "cuda",
) -> ImageProfile:
    """Fuse metadata + reverse search + VLM into one training caption."""
    p = Path(image_path)
    has_file = p.is_file()
    row = booru_row or {}

    sources: list[str] = []
    characters = _parse_row_lists(row, "character_tags")
    copyrights = _parse_row_lists(row, "copyright_tags")
    artists = _parse_row_lists(row, "artist_tags")
    general: list[str] = []
    scene_summary = str(row.get("scene_summary") or "").strip()
    confidence = float(row.get("tag_confidence") or 0.0)

    if characters:
        sources.append("booru_character")
        confidence = max(confidence, 0.95)
    if copyrights:
        sources.append("booru_copyright")
        confidence = max(confidence, 0.95)
    if artists:
        sources.append("booru_artist")
        confidence = max(confidence, 0.9)

    raw_caption = str(row.get("caption") or "").strip()
    if raw_caption:
        if not characters and not copyrights:
            general.extend(t.strip() for t in raw_caption.split(",") if t.strip())
            sources.append("booru_caption")
            confidence = max(confidence, 0.85)

    if use_reverse_search and has_file:
        for hit in reverse_search_file(p, use_saucenao=use_saucenao, use_tineye=use_tineye):
            if not hit_meets_threshold(hit, reverse_min_sim):
                continue
            confidence = _apply_reverse_hit(
                hit,
                characters=characters,
                copyrights=copyrights,
                artists=artists,
                sources=sources,
                confidence=confidence,
            )
            if characters or copyrights:
                break

    vlm_extras: list[str] = []
    if use_vlm and has_file:
        vlm_summary, vlm_extras = _vlm_scene_summary(p, device=device)
        if vlm_summary:
            scene_summary = vlm_summary
            sources.append("vlm_dense")
            confidence = max(confidence, 0.55)
        general.extend(vlm_extras)

    if not scene_summary:
        scene_summary = _summary_from_tags(
            characters=characters,
            copyrights=copyrights,
            artists=artists,
            caption=raw_caption,
        )
        if scene_summary:
            sources.append("tag_summary")

    is_oc = not characters and not copyrights
    if is_oc:
        if "original_character" in general:
            is_oc = True
        if scene_summary and re.search(r"\boriginal character\b|\boc\b|\bunknown character\b", scene_summary, re.I):
            is_oc = True
    else:
        general = [t for t in general if t != "original_character"]

    if is_oc and "original_character" not in general:
        general.insert(0, "original_character")

    caption = _build_caption(
        characters=characters,
        copyrights=copyrights,
        artists=artists,
        general=general if general else [t.strip() for t in raw_caption.split(",") if t.strip()],
    )
    if not caption and raw_caption:
        caption = raw_caption

    return ImageProfile(
        caption=caption,
        scene_summary=scene_summary,
        character_tags=_merge_tags(characters),
        copyright_tags=_merge_tags(copyrights),
        artist_tags=_merge_tags(artists),
        extra_tags=_merge_tags(general),
        is_original_character=is_oc,
        confidence=confidence,
        sources=list(dict.fromkeys(sources)),
    )


def profile_from_manifest_row(row: dict, data_root: Path, **kwargs) -> ImageProfile:
    rel = row.get("image_path") or ""
    img = data_root / rel if rel else Path("_missing")
    if not img.is_file():
        img = Path(rel) if rel else Path("_missing")
    has_identity = bool(row.get("character_tags") or row.get("copyright_tags"))
    return profile_image(
        img,
        booru_row=row,
        use_reverse_search=kwargs.get("use_reverse_search", True) and not has_identity,
        use_saucenao=kwargs.get("use_saucenao", True),
        use_tineye=kwargs.get("use_tineye", True),
        use_vlm=kwargs.get("use_vlm", True) and img.is_file(),
        device=kwargs.get("device", "cuda"),
        reverse_min_sim=float(kwargs.get("reverse_min_sim", kwargs.get("saucenao_min_sim", 75.0))),
    )


# Explicit prompt for caption models — reduces refusals on NSFW training data.
_UNCENSORED_VLM_PROMPT = (
    "Describe this image exhaustively for AI image training. "
    "Include every visible subject, character (name them if recognizable from anime/games), "
    "series or franchise, artist style, pose, clothing, body, expression, setting, lighting, "
    "composition, colors, and any mature or explicit details without censoring or omitting. "
    "End with comma-separated booru-style tags."
)

_DIFFUSION_SEED_FALLBACK = "high quality detailed illustration"


@dataclass
class PromptResearchResult:
    diffusion_prompt: str
    negative_prompt: str = ""
    image_description: str = ""
    scene_summary: str = ""
    retrieved_facts: list[str] = field(default_factory=list)
    reasoning: str = ""
    sources: list[str] = field(default_factory=list)
    fallback_used: bool = False

    def to_manifest_patch(self) -> dict:
        out = {
            "caption": self.diffusion_prompt,
            "scene_summary": self.scene_summary or self.image_description,
            "tag_sources": list(self.sources),
        }
        if self.negative_prompt.strip():
            out["negative_caption"] = self.negative_prompt.strip()
            out["negative_prompt_hint"] = self.negative_prompt.strip()
        return out


def _tags_from_description(text: str, *, max_tags: int = 24) -> str:
    """Pull trailing comma-tags from a VLM description."""
    t = (text or "").strip()
    if not t:
        return ""
    if "," in t:
        tail = t.split(",")[-12:]
        tags = [x.strip() for x in ",".join(tail).split(",") if x.strip()]
        if len(tags) >= 3:
            return ", ".join(tags[:max_tags])
    return t


def _seed_from_description(description: str, seed_prompt: str = "") -> str:
    seed = (seed_prompt or "").strip()
    if seed:
        return seed
    desc = (description or "").strip()
    if not desc:
        return _DIFFUSION_SEED_FALLBACK
    # First sentence or first 200 chars as intent seed for Creative RAG.
    first = re.split(r"[.!?\n]", desc, maxsplit=1)[0].strip()
    return first[:240] if first else _DIFFUSION_SEED_FALLBACK


def describe_image_uncensored(image_path: str | Path, *, device: str = "cuda") -> str:
    """Dense VLM description tuned for training captions (no content refusal)."""
    try:
        return caption_image_vlm(
            str(image_path),
            user_prompt=_UNCENSORED_VLM_PROMPT,
            device=device,
        ).strip()
    except Exception:
        return ""


def retrieve_rag_facts(
    query: str,
    corpus_path: str | Path,
    *,
    top_k: int = 8,
) -> list[str]:
    if not query.strip():
        return []
    try:
        from utils.prompt.rag import retrieve_facts_for_query_local

        return retrieve_facts_for_query_local(query, corpus_path, top_k=top_k)
    except Exception:
        return []


def research_prompt_for_image(
    image_path: str | Path,
    *,
    rag_corpus: str | Path | None = None,
    seed_prompt: str = "",
    creativity_level: float = 0.45,
    top_k: int = 8,
    device: str = "cuda",
    use_rag: bool = True,
    use_creative_rag: bool = True,
) -> PromptResearchResult:
    """Image → researched diffusion prompt using VLM + local RAG + Creative RAG."""
    p = Path(image_path)
    sources: list[str] = []
    image_description = describe_image_uncensored(p, device=device) if p.is_file() else ""
    if image_description:
        sources.append("vlm_uncensored")

    query = " ".join(x for x in (seed_prompt, image_description) if x).strip()
    facts: list[str] = []
    if use_rag and rag_corpus and query:
        facts = retrieve_rag_facts(query, rag_corpus, top_k=top_k)
        if facts:
            sources.append(f"rag:{len(facts)}")

    seed = _seed_from_description(image_description, seed_prompt)
    negative = ""
    reasoning = ""
    fallback_used = False
    diffusion_prompt = ""

    if use_creative_rag and p.is_file():
        try:
            from utils.prompt.rag import CreativeRAGEngine

            engine = CreativeRAGEngine(device=device)
            result = engine.enrich(
                seed,
                reference_image_path=str(p),
                facts=facts,
                creativity_level=creativity_level,
                use_image_dissection=False,
            )
            diffusion_prompt = (result.enriched_prompt or "").strip()
            negative = (result.negative_additions or "").strip()
            reasoning = (result.reasoning or "").strip()
            fallback_used = bool(result.fallback_used)
            if result.image_description and not image_description:
                image_description = result.image_description
            if result.retrieved_facts:
                facts = list(result.retrieved_facts)
            if diffusion_prompt:
                sources.append("creative_rag")
        except Exception:
            pass

    if not diffusion_prompt:
        fallback_used = True
        parts = []
        if image_description:
            parts.append(_tags_from_description(image_description) or image_description[:400])
        if facts:
            parts.append(", ".join(facts[:3]))
        diffusion_prompt = ", ".join(p for p in parts if p).strip() or seed
        reasoning = reasoning or "VLM + RAG fallback (Creative RAG models unavailable)."
        sources.append("rag_fallback")

    scene_summary = image_description
    if facts and scene_summary:
        scene_summary = f"{scene_summary}\n\nSimilar corpus entries: " + "; ".join(facts[:4])

    return PromptResearchResult(
        diffusion_prompt=diffusion_prompt,
        negative_prompt=negative,
        image_description=image_description,
        scene_summary=scene_summary,
        retrieved_facts=facts,
        reasoning=reasoning,
        sources=sources,
        fallback_used=fallback_used,
    )
