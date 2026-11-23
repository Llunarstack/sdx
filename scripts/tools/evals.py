#!/usr/bin/env python3
"""Competitor-weakness eval harness: fixed prompt suite, graceful scorers, and the runner CLI.

Grades generated images against a versioned prompt suite that probes the failure modes leading
text-to-image models share (see docs/RESEARCH.md). Model-agnostic: score sdx output or a
competitor's the same way.

Every scorer returns ``(score | None, detail)`` with ``score`` in [0, 1] (higher = better) or ``None``
when its backend is missing here; the report states coverage instead of crashing:
  * CLIP adherence  -> transformers CLIP (weights usually cached).
  * Object count    -> native counter (native/rust) when built, else SciPy connected components.
  * In-image text   -> pytesseract/easyocr when installed, else None.

Input maps each suite id to an image: a directory of ``<id>.png``/``.jpg`` files, or a JSONL of
``{"id": ..., "image_path": ...}`` rows (``"prompt"`` may replace ``"id"``).

    python -m scripts.tools evals --dump-suite suite.jsonl
    python -m scripts.tools evals --images ./out_dir --out report.json
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from utils.quality.critics import occlusion_score as occlusion_score  # noqa: E402
from utils.quality.critics import reflection_score as reflection_score  # noqa: E402
from utils.quality.critics import spatial_bind_score as spatial_bind_score  # noqa: E402

# ---------------------------------------------------------------- prompt suite
# Keep the suite stable and append-only; bump SUITE_VERSION when adding prompts.

SUITE_VERSION = "1.2.0"


@dataclass(frozen=True, slots=True)
class SuiteItem:
    id: str
    prompt: str
    category: str
    expected_count: int | None = None
    expected_text: str | None = None
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


CATEGORIES: dict[str, str] = {
    "long_adherence": "Long, multi-clause prompts where late tokens get dropped.",
    "relation": "Spatial/relational composition (A left of B, holding C).",
    "count": "Exact object counts — no major model guarantees these.",
    "text_render": "Legible in-image text of a specific string.",
    "anatomy": "Hands, small/far faces, crowds — degrade at distance.",
    "negation": "Excluding a concept the prompt names ('without X').",
    "style_control": "Adhering to a named, specific style.",
    "contact": "Feet / objects planted on the supporting surface (anti-float).",
    "bind": "Attribute binding: colors/clothes stay on the correct person.",
    "occlusion": "Subject visible through or behind occluders, not melted into them.",
    "reflection": "Physically plausible reflections, not a second hallucinated object.",
}


SUITE: tuple[SuiteItem, ...] = (
    SuiteItem(
        "long_01",
        "A weathered lighthouse keeper in a yellow raincoat stands on wet black "
        "rocks at dusk, holding a brass lantern in his left hand, a border collie "
        "beside him, storm clouds and a distant sailboat on the horizon",
        "long_adherence",
        note="Check every clause: raincoat color, lantern hand, dog, sailboat.",
    ),
    SuiteItem(
        "long_02",
        "A cozy bookstore cafe interior: exposed brick, a spiral staircase on the "
        "right, a orange tabby cat asleep on a stack of books, steam rising from a "
        "latte, warm afternoon light through tall windows on the left",
        "long_adherence",
        note="Staircase-right and windows-left must both hold.",
    ),
    SuiteItem(
        "rel_01",
        "A red cube to the left of a blue sphere, with a green pyramid behind both",
        "relation",
        note="Left/right and behind must be correct simultaneously.",
    ),
    SuiteItem(
        "rel_02",
        "A cat sitting on top of a microwave that is on top of a refrigerator",
        "relation",
        note="Vertical stacking order.",
    ),
    SuiteItem("count_01", "Exactly three red apples on a white plate", "count", expected_count=3),
    SuiteItem("count_02", "Five yellow rubber ducks in a row on a bathtub edge", "count", expected_count=5),
    SuiteItem("count_03", "A single lit candle in a dark room", "count", expected_count=1),
    SuiteItem(
        "text_01",
        'A vintage enamel shop sign that reads "OPEN" in bold serif letters',
        "text_render",
        expected_text="OPEN",
    ),
    SuiteItem(
        "text_02",
        'A latte with the word "HELLO" drawn in the foam',
        "text_render",
        expected_text="HELLO",
    ),
    SuiteItem(
        "anat_01",
        "Close-up of two hands carefully tying a shoelace, five fingers each",
        "anatomy",
        note="Finger count and hand structure.",
    ),
    SuiteItem(
        "anat_02",
        "A crowd of a dozen people crossing a busy street, seen from across the road",
        "anatomy",
        note="Small/far faces and bodies stay coherent.",
    ),
    SuiteItem(
        "neg_01",
        "A dining table set for breakfast, without any coffee or mugs",
        "negation",
        note="Coffee/mugs must be ABSENT.",
    ),
    SuiteItem(
        "neg_02",
        "An empty beach with no people and no boats",
        "negation",
        note="People/boats must be ABSENT.",
    ),
    SuiteItem(
        "style_01",
        "A mountain landscape in flat 2-color risograph print style, coarse grain",
        "style_control",
        note="Risograph look, limited palette.",
    ),
    SuiteItem(
        "style_02",
        "A portrait of a fox in the style of a technical blueprint, white lines on blue",
        "style_control",
        note="Blueprint palette and line quality.",
    ),
    SuiteItem(
        "contact_01",
        "A woman standing on wet pavement at night, shoes planted on the ground, "
        "contact shadow under both feet, no hovering",
        "contact",
        note="Both feet on pavement with contact shadows; no floating.",
    ),
    SuiteItem(
        "contact_02",
        "A coffee mug sitting on a wooden table, base flush with the tabletop, no floating",
        "contact",
        note="Mug base flush with table; no gap underneath.",
    ),
    SuiteItem(
        "bind_01",
        "A woman in a red dress on the left and a man in a blue suit on the right, colors must not swap",
        "bind",
        note="Red dress stays on the woman; blue suit stays on the man.",
    ),
    SuiteItem(
        "bind_02",
        "A brown dog wearing a blue collar sitting beside a white cat with a green bell, wrong pairing forbidden",
        "bind",
        note="Blue collar on dog; green bell on cat — no attribute swap.",
    ),
    SuiteItem(
        "occlude_01",
        "A person standing behind a chain-link fence, face and jacket visible through "
        "the gaps, fence in the foreground",
        "occlusion",
        note="Person readable through fence gaps; not melted into mesh.",
    ),
    SuiteItem(
        "occlude_02",
        "A red apple partly hidden behind a wine glass, both complete objects, glass in front",
        "occlusion",
        note="Apple and glass remain distinct; glass occludes in front.",
    ),
    SuiteItem(
        "reflect_01",
        "A chrome kettle on black granite, a sharp reflection of the kettle in the stone, not a second kettle",
        "reflection",
        note="Reflection matches kettle pose; no duplicate hallucinated object.",
    ),
    SuiteItem(
        "rel_03",
        "A yellow lamp between a red chair and a blue stool",
        "relation",
        note="Lamp in the middle; chair left, stool right; colors stay bound.",
    ),
    SuiteItem(
        "rel_04",
        "A sailor holding a brass lantern in his left hand, standing on wet rocks",
        "relation",
        note="Lantern is gripped, not floating; left hand occupancy.",
    ),
    SuiteItem(
        "text_03",
        'A storefront with two lines of text: top line "OPEN NOW" and bottom line "24 HOURS"',
        "text_render",
        expected_text="OPEN NOW",
        note='Top line "OPEN NOW" is auto-graded; bottom line "24 HOURS" is human review.',
    ),
)


def suite_as_manifest_rows() -> list[dict]:
    """The suite as JSONL-ready rows (id, prompt, category, expectations)."""
    rows = []
    for s in SUITE:
        row = {"id": s.id, "prompt": s.prompt, "category": s.category}
        if s.expected_count is not None:
            row["expected_count"] = s.expected_count
        if s.expected_text is not None:
            row["expected_text"] = s.expected_text
        if s.note:
            row["note"] = s.note
        rows.append(row)
    return rows


# ---------------------------------------------------------------- CLIP adherence

_CLIP: dict = {}


def _clip(model_id: str, device: str):
    key = (model_id, device)
    if key in _CLIP:
        return _CLIP[key]
    import torch
    from transformers import CLIPModel, CLIPProcessor

    proc = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _CLIP[key] = (model, proc, torch)
    return _CLIP[key]


def clip_adherence(
    image_path: str,
    prompt: str,
    *,
    model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> tuple[float | None, dict]:
    """CLIP image-text cosine remapped from [-1, 1] to [0, 1].

    CLIP is largely blind to negation and exact counts, so it measures topical adherence only; the
    harness pairs it with dedicated count/text scorers for those categories.
    """
    try:
        from PIL import Image
        from utils.generation.guidance import _clip_feature_tensor

        model, proc, torch = _clip(model_id, device)
        img = Image.open(image_path).convert("RGB")
        inputs = proc(text=[prompt], images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            imf = _clip_feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            txf = _clip_feature_tensor(
                model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
            )
            imf = imf / (imf.norm(dim=-1, keepdim=True) + 1e-8)
            txf = txf / (txf.norm(dim=-1, keepdim=True) + 1e-8)
            cos = float((imf * txf).sum(-1).item())
        return max(0.0, min(1.0, (cos + 1.0) / 2.0)), {"cosine": cos}
    except Exception as e:  # pragma: no cover - env dependent
        return None, {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- object count


def count_objects(image_path: str, *, min_area_frac: float = 0.005) -> int | None:
    """Count salient foreground blobs (native counter when built, else SciPy); None if unreadable."""
    try:
        from PIL import Image

        arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    except Exception:
        return None
    n = _native_count_blobs(arr, min_area_frac)
    if n is not None:
        return int(n)
    return _count_blobs_numpy(arr, min_area_frac=min_area_frac)


def _ensure_native_on_path() -> None:
    native_py = ROOT / "native" / "_experimental" / "python"
    if native_py.is_dir() and str(native_py) not in sys.path:
        sys.path.insert(0, str(native_py))


def _native_count_blobs(arr: np.ndarray, min_area_frac: float) -> int | None:
    try:
        _ensure_native_on_path()
        from sdx_native.ffi import count_blobs as _native_count  # type: ignore

        return _native_count(arr, min_area_frac=min_area_frac)
    except Exception:
        return None


def _otsu_threshold(gray: np.ndarray) -> float:
    """Otsu threshold in [0,1] from a 256-bin histogram."""
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    w_b = 0.0
    sum_b = 0.0
    best_t, best_var = 0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    # Threshold between bins so a flat-colored class sitting exactly on best_t lands on the intended side.
    return (best_t + 0.5) / 255.0


def _count_blobs_numpy(gray: np.ndarray, *, min_area_frac: float) -> int:
    """Reference connected-components blob count (4-connectivity).

    The background class is whichever of {dark, light} dominates the border ring. Components touching
    the border, or larger than 60% of the frame, are background/leaks and not counted.
    """
    from scipy import ndimage

    total = gray.size
    dark = gray < _otsu_threshold(gray)
    border = np.concatenate([dark[0, :], dark[-1, :], dark[:, 0], dark[:, -1]])
    border_dark_frac = float(border.mean()) if border.size else 0.5
    mask = ~dark if border_dark_frac >= 0.5 else dark

    labels, n = ndimage.label(mask)
    if n == 0:
        return 0
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    keep = (areas >= max(1, int(min_area_frac * total))) & (areas <= int(0.60 * total))
    keep[0] = False
    keep[np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])] = False
    return int(keep.sum())


def count_score(image_path: str, expected_count: int) -> tuple[float | None, dict]:
    """1.0 if detected == expected, decaying as 1 / (1 + |error|)."""
    n = count_objects(image_path)
    if n is None:
        return None, {"detected": None}
    err = abs(n - int(expected_count))
    return 1.0 / (1.0 + err), {"detected": n, "expected": int(expected_count), "abs_error": err}


# ---------------------------------------------------------------- in-image text (OCR)


def _ocr_available() -> str | None:
    for engine in ("pytesseract", "easyocr"):
        try:
            __import__(engine)
            return engine
        except Exception:
            pass
    return None


@lru_cache(maxsize=1)
def _easyocr_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def text_score(image_path: str, expected_text: str) -> tuple[float | None, dict]:
    """Fraction of the expected string recovered by OCR; None if no OCR engine is installed."""
    engine = _ocr_available()
    if engine is None:
        return None, {"engine": None}
    try:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        if engine == "pytesseract":
            import pytesseract

            got = pytesseract.image_to_string(img)
        else:
            got = " ".join(_easyocr_reader().readtext(np.asarray(img), detail=0))
    except Exception as e:  # pragma: no cover - env dependent
        return None, {"engine": engine, "error": str(e)}

    want = _normalize_text(expected_text)
    have = _normalize_text(got)
    if not want:
        return None, {"engine": engine}
    score = 1.0 if want in have else _lcs_ratio(want, have)
    return score, {"engine": engine, "expected": expected_text, "ocr": got.strip()[:80]}


def _lcs_ratio(a: str, b: str) -> float:
    """Longest common substring length / len(a)."""
    if not a:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size / len(a)


def availability(model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> dict:
    """Which scorers can actually grade in this environment."""
    try:
        _clip(model_id, device)
        clip_ok = True
    except Exception:
        clip_ok = False
    try:
        _ensure_native_on_path()
        from sdx_native.ffi import rust_available as _native_available

        native = bool(_native_available())
    except Exception:
        native = False
    return {
        "clip_adherence": clip_ok,
        "count": True,
        "count_native": native,
        "text_ocr": _ocr_available(),
        "occlusion": True,
        "reflection": True,
    }


# ---------------------------------------------------------------- runner

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# category -> (report key, scorer(image_path, item))
_CATEGORY_SCORERS = {
    "count": ("count_score", lambda p, it: count_score(p, it.expected_count)),
    "text_render": ("text_score", lambda p, it: text_score(p, it.expected_text)),
    "occlusion": ("occlusion_score", lambda p, it: occlusion_score(p)),
    "reflection": ("reflection_score", lambda p, it: reflection_score(p)),
    "relation": ("spatial_bind_score", lambda p, it: spatial_bind_score(p, it.prompt)),
    "bind": ("spatial_bind_score", lambda p, it: spatial_bind_score(p, it.prompt)),
}
_SUMMARY_LABELS = {
    "count_score": "count",
    "text_score": "text",
    "occlusion_score": "occ",
    "reflection_score": "refl",
    "spatial_bind_score": "bind",
}


def _resolve_images(images_arg: str) -> dict[str, str]:
    """{suite_id: image_path} from a directory of ``<id>.<ext>`` or a JSONL mapping."""
    from utils.runtime import json_loads

    p = Path(images_arg)
    by_id: dict[str, str] = {}
    if p.is_dir():
        for item in SUITE:
            for ext in _IMG_EXTS:
                cand = p / f"{item.id}{ext}"
                if cand.exists():
                    by_id[item.id] = str(cand)
                    break
        return by_id
    prompt_to_id = {s.prompt: s.id for s in SUITE}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json_loads(line)
            sid = row.get("id")
            if sid is None and "prompt" in row:
                sid = prompt_to_id.get(str(row["prompt"]))
            img = row.get("image_path") or row.get("image") or row.get("path")
            if sid and isinstance(img, str):
                by_id[str(sid)] = img
    return by_id


def _mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return round(statistics.fmean(v), 4) if v else None


def evaluate(images_by_id: dict[str, str], *, clip_model: str, device: str) -> dict:
    suite_by_id = {s.id: s for s in SUITE}
    items = []
    for sid, img_path in images_by_id.items():
        item = suite_by_id.get(sid)
        if item is None:
            continue
        adh, adh_d = clip_adherence(img_path, item.prompt, model_id=clip_model, device=device)
        rec: dict = {
            "id": sid,
            "category": item.category,
            "image_path": img_path,
            "adherence": adh,
            "adherence_detail": adh_d,
        }
        spec = _CATEGORY_SCORERS.get(item.category)
        needs = {"count": item.expected_count, "text_render": item.expected_text}
        if spec is not None and needs.get(item.category, True) is not None:
            key, fn = spec
            rec[key], rec[key.replace("_score", "_detail")] = fn(img_path, item)
        items.append(rec)

    per_cat: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_items = [r for r in items if r["category"] == cat]
        if not cat_items:
            continue
        agg = {"n": len(cat_items), "adherence": _mean([r["adherence"] for r in cat_items])}
        if cat in _CATEGORY_SCORERS:
            key = _CATEGORY_SCORERS[cat][0]
            agg[key] = _mean([r.get(key) for r in cat_items])
        per_cat[cat] = agg

    return {
        "suite_version": SUITE_VERSION,
        "availability": availability(clip_model, device),
        "n_scored": len(items),
        "n_suite": len(SUITE),
        "overall_adherence": _mean([r["adherence"] for r in items]),
        "per_category": per_cat,
        "items": items,
    }


def _print_summary(report: dict) -> None:
    av = report["availability"]
    print(f"\n=== sdx eval harness (suite v{report['suite_version']}) ===")
    print(f"scored {report['n_scored']}/{report['n_suite']} suite items")
    print(f"backends: clip={av['clip_adherence']} count_native={av['count_native']} ocr={av['text_ocr']}")
    oa = report["overall_adherence"]
    print(f"overall adherence: {oa if oa is not None else 'n/a'}")
    print(f"{'category':<16}{'n':>4}{'adherence':>12}{'extra':>16}")
    for cat, agg in report["per_category"].items():
        extra = next((f"{lbl}={agg[k]}" for k, lbl in _SUMMARY_LABELS.items() if k in agg), "")
        adh = agg["adherence"]
        print(f"{cat:<16}{agg['n']:>4}{(adh if adh is not None else 'n/a'):>12}{extra:>16}")


def main(argv: list[str] | None = None) -> int:
    from utils.runtime import json_dumps

    ap = argparse.ArgumentParser(description="Run the sdx competitor-weakness eval harness")
    ap.add_argument("--images", help="Directory of <id>.png, or a JSONL id->image_path mapping")
    ap.add_argument("--out", default="", help="Write full JSON report here")
    ap.add_argument("--dump-suite", default="", help="Write the prompt suite as JSONL and exit")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    if args.dump_suite:
        rows = suite_as_manifest_rows()
        Path(args.dump_suite).write_text("\n".join(json_dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        print(f"wrote {len(rows)} suite prompts -> {args.dump_suite}")
        return 0

    if not args.images:
        ap.error("--images is required (or use --dump-suite)")
    images_by_id = _resolve_images(args.images)
    if not images_by_id:
        print("No images matched suite ids. Name files <id>.png or provide a JSONL mapping.", file=sys.stderr)
        return 2

    report = evaluate(images_by_id, clip_model=args.clip_model, device=args.device)
    if args.out:
        Path(args.out).write_text(json_dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
