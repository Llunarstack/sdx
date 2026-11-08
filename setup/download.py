#!/usr/bin/env python3
"""Download everything SDX needs: pretrained models, booru datasets, HF dataset packs.

    python setup/download.py pretrained --profile train        # pretrained_status.json catalog -> pretrained/
    python setup/download.py datasets --out /workspace/data    # booru scrape (danbooru + rule34.xxx + ...)
    python setup/download.py hf-datasets --dest /workspace/data
    python setup/download.py t5                                # T5-XXL model.safetensors
    python setup/download.py repa                              # REPA vision encoder (DINOv3-L)
    python setup/download.py models --t5 --vae                 # hand-picked T5 / VAE / CLIP / LLM sets
    python setup/download.py llm --best                        # prompt-expansion LLM
    python setup/download.py scaffold --penta                  # config + tokenizer only scaffolds
    python setup/download.py stack --mode fast                 # curated high-end stack

Set ``HF_TOKEN`` (or ``hf auth login`` / secret.txt) for gated models and higher rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STATUS_JSON = REPO_ROOT / "pretrained_status.json"
PACKS_JSON = REPO_ROOT / "setup" / "hf_dataset_packs.json"
PRETRAINED = REPO_ROOT / "pretrained"


# --- shared HF helpers -------------------------------------------------------


def _hf_token() -> str | None:
    try:
        from utils.modeling.model_paths import apply_hf_token_to_env, get_hf_token

        apply_hf_token_to_env()
        return get_hf_token()
    except ImportError:
        pass
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return None


def _enable_fast_transfer() -> bool:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        return True
    except ImportError:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        return False


def _retry_wait(exc: BaseException, attempt: int) -> int:
    msg = str(exc)
    m = re.search(r"Retry after (\d+)", msg, re.IGNORECASE)
    if m:
        return int(m.group(1)) + 5
    if "429" in msg or "rate limit" in msg.lower():
        return min(180, 30 * attempt)
    return min(60, 4 * attempt)


def _have_hub() -> bool:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Install first: pip install huggingface_hub hf_transfer", file=sys.stderr)
        return False
    return True


def _snapshot(repo_id: str, local_dir: str | Path | None, *, retries: int = 1, **kwargs) -> str | None:
    """``snapshot_download`` with retry/backoff; ``None`` kwargs are dropped. Returns the local path or None."""
    from huggingface_hub import snapshot_download

    kw = {k: v for k, v in kwargs.items() if v}
    if local_dir:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        kw["local_dir"] = str(local_dir)
    for attempt in range(1, retries + 1):
        try:
            return snapshot_download(repo_id=repo_id, **kw)
        except Exception as e:
            if attempt == retries:
                if retries == 1:
                    raise
                break
            wait = _retry_wait(e, attempt)
            print(
                f"  attempt {attempt}/{retries} failed ({type(e).__name__}); "
                f"retrying in {wait}s — run: hf auth login (or set token in secret.txt)",
                file=sys.stderr,
            )
            time.sleep(wait)
    return None


def _dir_populated(local_dir: Path, *, min_bytes: int = 1_000_000) -> bool:
    if not local_dir.is_dir():
        return False
    total = 0
    for p in local_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            if total >= min_bytes:
                return True
    return False


# --- pretrained (pretrained_status.json catalog) -----------------------------

_PROFILE_ALIASES = {
    "full": None,
    "all": None,
    "ultimate": None,  # runpod full pipeline — entire pretrained_status.json catalog
    "train": "train",
    "enrich": "enrich",
    "inference": "inference",
    "minimal": "train",
}


def _load_status() -> dict:
    return json.loads(STATUS_JSON.read_text(encoding="utf-8"))


def _profile_names(status: dict, profile: str) -> set[str] | None:
    pl = profile.lower()
    if pl not in _PROFILE_ALIASES:
        raise ValueError(f"Unknown profile {profile!r} (use train|enrich|inference|ultimate|full)")
    key = _PROFILE_ALIASES[pl]
    if key is None:
        return None
    names = status.get("profiles", {}).get(key, [])
    if not names:
        raise ValueError(f"Profile {key!r} has no models in pretrained_status.json")
    return {n.lower() for n in names}


def _download_url_files(local_dir: Path, files: dict[str, str], *, retries: int) -> bool:
    """Download explicit relative-path -> URL mappings (e.g. CodeFormer GitHub releases)."""
    for rel, url in files.items():
        dest = local_dir / rel
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        ok = False
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
                    shutil.copyfileobj(resp, fh)
                tmp.replace(dest)
                ok = True
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                wait = min(60, 4 * attempt)
                print(
                    f"  {rel}: attempt {attempt}/{retries} failed ({type(e).__name__}: {e}); retrying in {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
            finally:
                tmp.unlink(missing_ok=True)
        if not ok:
            return False
    return all((local_dir / rel).is_file() for rel in files)


def _pretrained(args: argparse.Namespace) -> int:
    if not _have_hub():
        return 2
    fast = _enable_fast_transfer()
    token = _hf_token()
    print(f"hf_transfer fast download: {'ON' if fast else 'OFF (pip install hf_transfer for a big speedup)'}")
    if token:
        try:
            from utils.modeling.model_paths import hf_auth_source

            src = {"cli": "huggingface-cli login", "secret": "secret.txt"}.get(hf_auth_source(), "OK")
        except ImportError:
            src = "OK"
        print(f"HF auth: {src} (higher rate limits, gated models OK)")
    else:
        print(
            "HF auth: NOT SET — you will hit 429 rate limits.\n"
            "  Run: huggingface-cli login\n"
            "  Or add to /workspace/secret.txt:\n"
            "    huggingface\n"
            "    token: hf_your_token_here",
            file=sys.stderr,
        )

    workers = args.workers if args.workers > 0 else (4 if token else 2)
    workers = max(1, min(workers, 4 if token else 2))

    status = _load_status()
    registry = status.get("models", [])
    if args.only:
        want = {n.lower() for n in args.only}
        registry = [m for m in registry if m.get("name", "").lower() in want]
    else:
        prof = (args.profile or "full").lower()
        want = _profile_names(status, prof)
        if want is not None:
            registry = [m for m in registry if m.get("name", "").lower() in want]
            print(f"Profile: {prof} ({len(registry)} models)")
        else:
            print(f"Profile: {prof} (full catalog — {len(registry)} models)")

    if not registry:
        print("No models matched.", file=sys.stderr)
        return 2

    dest_base = Path(args.dest)
    total_gb = sum(float(m.get("size_gb", 0) or 0) for m in registry)
    print(f"Downloading {len(registry)} models (~{total_gb:.1f} GB) -> {dest_base}\n")

    ok, skipped, failed, optional_failed = 0, 0, [], []
    n = len(registry)
    for i, m in enumerate(registry, 1):
        name = m.get("name", "?")
        repo_id = m.get("hf_fallback")
        download_files = m.get("download_files") or {}
        revision = m.get("hf_revision") or None
        if not repo_id and not download_files:
            print(f"[{i}/{n}] {name}: no hf_fallback or download_files, skipping", file=sys.stderr)
            continue
        local_dir = dest_base / name
        present = None
        if not args.force:
            if download_files and all((local_dir / rel).is_file() for rel in download_files):
                present = "github releases"
            elif repo_id and _dir_populated(local_dir):
                present = repo_id
        if present:
            print(f"[{i}/{n}] {name}  <-  {present}  (already present, skip)")
            skipped += 1
            ok += 1
            continue
        if download_files:
            print(f"[{i}/{n}] {name}  <-  github releases  (~{m.get('size_gb', '?')} GB)")
            success = _download_url_files(local_dir, download_files, retries=args.retries)
        else:
            rev_note = f" @{revision}" if revision else ""
            print(f"[{i}/{n}] {name}  <-  {repo_id}{rev_note}  (~{m.get('size_gb', '?')} GB)")
            success = _snapshot(
                repo_id,
                local_dir,
                retries=args.retries,
                max_workers=workers,
                revision=revision,
                token=token,
                allow_patterns=m.get("allow_patterns") or None,
            ) is not None and _dir_populated(local_dir)
        if success:
            ok += 1
        else:
            (optional_failed if m.get("optional") else failed).append(name)
            print(f"  FAILED after {args.retries} retries: {name}", file=sys.stderr)
        if repo_id and not download_files and not token and i < n:
            time.sleep(10)

    print(f"\nDone: {ok}/{n} ok ({skipped} skipped as present). Dest: {dest_base}")
    if optional_failed:
        print(f"Optional models failed (non-fatal): {', '.join(optional_failed)}", file=sys.stderr)
    if failed:
        print(f"Required models failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


# --- t5 / repa (ensure single encoders) --------------------------------------


def _t5(_args: argparse.Namespace) -> int:
    base = Path(os.environ.get("SDX_PRETRAINED", PRETRAINED))
    if PRETRAINED.is_symlink():
        base = PRETRAINED.resolve()
    enc = os.environ.get("SDX_TEXT_ENCODER", "").strip()
    if enc:
        dest = Path(enc)
    elif (base / "T5-XXL-safetensors" / "model.safetensors").is_file():
        dest = base / "T5-XXL-safetensors"
    else:
        dest = base / "T5-XXL"
    safetensors = dest / "model.safetensors"
    if safetensors.is_file() and safetensors.stat().st_size > 1_000_000_000:
        print(f"T5 safetensors OK: {safetensors}")
        return 0

    repo = "mcmonkey/google_t5-v1_1-xxl_encoderonly"
    print(f"Downloading T5 encoder safetensors from {repo} -> {dest} (~9 GB)")
    if not _have_hub():
        return 2
    token = _hf_token()
    allow = ["model.safetensors", "config.json", "spiece.model", "tokenizer_config.json", "special_tokens_map.json"]
    if _snapshot(repo, dest, retries=5, token=token, max_workers=2 if token else 1, allow_patterns=allow) is None:
        return 1
    if safetensors.is_file():
        print(f"Downloaded: {safetensors}")
        return 0
    print("Download finished but model.safetensors not found", file=sys.stderr)
    return 1


_REPA_FOLDERS = (
    (("dinov3-vitl16", "dinov3-vit-l"), "DINOv3-ViT-L16"),
    (("dinov3-vitb16",), "DINOv3-ViT-B16"),
    (("dinov3-vits16",), "DINOv3-ViT-S16"),
    (("dinov2-base",), "DINOv2-Base"),
    (("dinov2-large",), "DINOv2-Large"),
)


def _repa(args: argparse.Namespace) -> int:
    model = args.model
    if Path(model).is_dir():
        print(f"REPA encoder already local: {model}")
        return 0
    lower = model.lower()
    name = next((folder for keys, folder in _REPA_FOLDERS if any(k in lower for k in keys)), "REPA-encoder")
    folder = Path(args.dest) / name
    if folder.is_dir() and any(folder.iterdir()):
        print(f"REPA encoder OK: {folder}")
        return 0
    if not _have_hub():
        return 1
    print(f"Downloading REPA encoder {model} -> {folder}")
    _snapshot(model, folder)
    print(f"Done: {folder}")
    return 0


# --- models / llm / stack (hand-picked HF sets) -------------------------------

ALLOW_T5 = [
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer_config.json",
]
ALLOW_VAE = ["config.json", "*.safetensors", "diffusion_pytorch_model.bin"]
ALLOW_CLIP = ["config.json", "preprocessor_config.json", "*.safetensors", "pytorch_model.bin"]
ALLOW_CLIP_TOK = ALLOW_CLIP + [
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
ALLOW_LLM = [
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]
ALLOW_ADVANCED_GENERIC = [
    "config.json",
    "model_index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.ckpt",
    "*.pth",
]
ALLOW_CONFIG_ONLY = [
    "**/*.json",
    "**/*.jinja",
    "**/*.yaml",
    "**/*.yml",
    "**/*.txt",
    "**/*.md",
    "**/.gitattributes",
    "**/tokenizer.model",
    "**/spiece.model",
    "**/vocab.json",
    "**/merges.txt",
]
IGNORE_WEIGHTS = [f"{d}*.{e}" for e in ("safetensors", "bin", "pt", "pth", "ckpt") for d in ("", "**/")]

T5_REPOS = [
    ("google/t5-v1_1-xxl", "T5-XXL"),  # 4096-dim, best understanding (default)
    ("google/t5-v1_1-xl", "T5-XL"),
    ("google/t5-v1_1-large", "T5-Large"),
]
VAE_REPOS = [
    ("REPA-E/e2e-flux-vae", "REPA-E-FLUX-VAE"),  # 16-channel FLUX VAE, end-to-end tuned (train default)
    ("stabilityai/sd-vae-ft-mse", "sd-vae-ft-mse"),
    ("stabilityai/sd-vae-ft-ema", "sd-vae-ft-ema"),
    ("stabilityai/sdxl-vae", "sdxl-vae"),
    ("madebyollin/sdxl-vae-fp16-fix", "sdxl-vae-fp16-fix"),
]
CLIP_REPOS = [("openai/clip-vit-large-patch14", "CLIP-ViT-L-14")]
TRIPLE_TEXT_REPOS = [
    ("google/t5-v1_1-xxl", "T5-XXL"),
    ("openai/clip-vit-large-patch14", "CLIP-ViT-L-14"),
    ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "CLIP-ViT-bigG-14"),
]
PENTA_TEXT_REPOS = TRIPLE_TEXT_REPOS + [
    ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "CLIP-ViT-H-14"),
    ("creative-graphic-design/LongCLIP-L", "LongCLIP-L"),
]
LLM_DEFAULT = "HuggingFaceTB/SmolLM2-360M-Instruct"
LLM_BEST = "Qwen/Qwen3-14B"
ADVANCED_REPOS = [
    ("creative-graphic-design/LongCLIP-L", "LongCLIP-L"),
    ("moondream/moondream3-preview", "moondream3-preview"),
    ("MizzenAI/HPSv3", "HPSv3"),
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", "DINOv3-ViT-L16"),
    ("google/siglip2-so400m-patch16-384", "SigLIP2-SO400M"),
    ("Qwen/Qwen3-8B", "Qwen3-8B"),
    ("prs-eth/marigold-depth-v1-1", "Marigold-Depth-v1-1"),
    ("prs-eth/marigold-normals-v1-1", "Marigold-Normals-v1-1"),
    ("madebyollin/taesd", "TAESD"),
    ("madebyollin/taesdxl", "TAESDXL"),
    ("openai/consistency-decoder", "Consistency-Decoder"),
    ("facebook/convnextv2-large-22k-384", "ConvNeXtV2-Large"),
    ("camenduru/improved-aesthetic-predictor", "LAION-Aesthetic-v2"),
    ("camenduru/AnyDoor", "AnyDoor-Ref"),  # non-standard layout; identity-preserving experiments
]
CODEFORMER_FILES = {
    f"weights/{sub}/{name}": f"https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/{name}"
    for sub, name in (
        ("CodeFormer", "codeformer.pth"),
        ("facelib", "detection_Resnet50_Final.pth"),
        ("facelib", "parsing_parsenet.pth"),
    )
}
_PENTA_ALLOW = {"T5": ALLOW_T5, "LongCLIP": ALLOW_ADVANCED_GENERIC}


def _models(args: argparse.Namespace) -> int:
    if not _have_hub():
        return 1
    root = Path(args.model_dir)
    root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, list[str], list[str] | None, str]] = []

    def add(repos, allow, label: str, ignore=None) -> None:
        for repo_id, folder in repos:
            a = allow(folder) if callable(allow) else allow
            jobs.append((repo_id, folder, a, ignore, label))

    if args.t5 or args.all:
        add(T5_REPOS, ALLOW_T5, "T5 (essential files only)")
    if args.vae or args.all:
        add(VAE_REPOS, ALLOW_VAE, "VAE (essential files only)")
    if args.clip or args.all:
        add(CLIP_REPOS, ALLOW_CLIP, "CLIP (essential files only)")
    if args.triple_text_encoders:
        add(TRIPLE_TEXT_REPOS, lambda f: ALLOW_T5 if f.startswith("T5") else ALLOW_CLIP, "triple text encoder")
    if args.penta_text_encoders:
        add(
            PENTA_TEXT_REPOS,
            lambda f: next((v for k, v in _PENTA_ALLOW.items() if f.startswith(k)), ALLOW_CLIP),
            "penta text encoder",
        )
    if args.llm or args.all:
        add([(LLM_DEFAULT, "SmolLM2-360M-Instruct")], ALLOW_LLM, "LLM (essential files only)")
    if args.llm_best or args.all:
        add([(LLM_BEST, "Qwen3-14B")], ALLOW_LLM, "LLM (essential files only)")
    if args.advanced:
        label = "advanced model (" + ("config-only" if args.config_only else "essential files only") + ")"
        add(
            ADVANCED_REPOS,
            ALLOW_CONFIG_ONLY if args.config_only else ALLOW_ADVANCED_GENERIC,
            label,
            IGNORE_WEIGHTS if args.config_only else None,
        )

    if not jobs:
        print(
            "Choose at least one: --t5, --vae, --clip, --triple-text-encoders, --penta-text-encoders, "
            "--llm, --llm-best, --advanced, or --all",
            file=sys.stderr,
        )
        return 1

    for repo_id, folder, allow, ignore, label in jobs:
        local_dir = root / folder
        print(f"Downloading {label}: {repo_id} -> {local_dir}")
        _snapshot(repo_id, local_dir, max_workers=args.max_workers, allow_patterns=allow, ignore_patterns=ignore)
    n = len(jobs)
    if args.advanced and not args.config_only:
        for rel, url in CODEFORMER_FILES.items():
            target = root / "CodeFormer" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading CodeFormer weight: {url} -> {target}")
            urllib.request.urlretrieve(url, target)
        n += 1
    if args.triple_text_encoders:
        print("  -> Train: python train.py ... --text-encoder-mode triple")
    if args.penta_text_encoders:
        print("  -> Train: python train.py ... --text-encoder-mode penta")
    print(f"Done. {n} model(s) in {root}")
    return 0


def _llm(args: argparse.Namespace) -> int:
    model = LLM_BEST if args.best else args.model
    local_dir = args.local_dir
    if local_dir is None and args.best:
        local_dir = str(PRETRAINED / "Qwen3-14B")
    if not _have_hub():
        return 1
    print(f"Downloading {model} (max_workers={args.max_workers})...")
    path = _snapshot(model, local_dir, max_workers=args.max_workers, cache_dir=args.cache_dir)
    print(f"Done. Model at: {path}")
    return 0


def _scaffold(args: argparse.Namespace) -> int:
    from utils.modeling.model_paths import (
        HF_SCAFFOLD_REGISTRY,
        download_scaffold_batch,
        resolve_entries,
        scaffold_registry,
    )

    if args.list:
        for e in scaffold_registry():
            print(f"{e.name}\t{e.role}\t{e.repo_id}")
        return 0
    if not _have_hub():
        return 1

    if args.all:
        entries = list(HF_SCAFFOLD_REGISTRY)
    elif args.penta or args.triple:
        from utils.modeling.encoders import PENTA_CATALOG, TRIPLE_CATALOG

        entries = resolve_entries(names=list(PENTA_CATALOG if args.penta else TRIPLE_CATALOG), roles=None)
    else:
        entries = resolve_entries(names=args.name or None, roles=args.role or None)
    if not entries:
        print("Choose --all, --role, or --name (use --list to see registry).", file=sys.stderr)
        return 1

    model_root = Path(args.model_dir) if str(args.model_dir).strip() else PRETRAINED
    report = download_scaffold_batch(
        entries,
        model_root=model_root,
        max_workers=int(args.max_workers),
        skip_existing=not args.force,
        continue_on_error=not args.fail_fast,
    )
    failed = list(report.get("failed", []))
    print(f"Done. scaffold_ok={len(report.get('ok', []))} failed={len(failed)} -> {model_root}")
    for msg in failed:
        print(f"  FAILED: {msg}", file=sys.stderr)
    if str(args.out_json).strip():
        out = Path(str(args.out_json).strip())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report -> {out}")
    return 1 if failed and args.fail_fast else 0


STACK_JOBS = [
    ("google/t5-v1_1-xxl", "T5-XXL", ALLOW_T5),
    ("openai/clip-vit-large-patch14", "CLIP-ViT-L-14", ALLOW_CLIP_TOK),
    ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "CLIP-ViT-bigG-14", ALLOW_CLIP_TOK),
    ("Qwen/Qwen3-14B", "Qwen3-14B", ALLOW_LLM),  # prompt expansion / RAG
    ("Qwen/Qwen3-8B", "Qwen3-8B", ALLOW_LLM),  # Qwen T2I text encoder
    ("Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Instruct", ["preprocessor_config.json", *ALLOW_LLM]),
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", "DINOv3-ViT-L16", ALLOW_CLIP),  # REPA encoder
    (
        "google/siglip2-so400m-patch16-384",
        "SigLIP2-SO400M",
        ["preprocessor_config.json", *[a for a in ALLOW_LLM if a != "generation_config.json"]],
    ),
    (
        "MizzenAI/HPSv3",
        "HPSv3",
        [*ALLOW_CLIP, "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    ),
    (
        "moondream/moondream3-preview",
        "moondream3-preview",
        ["preprocessor_config.json", *ALLOW_LLM[:7]],
    ),
    (
        "stabilityai/stable-cascade-prior",
        "StableCascade-Prior",
        [
            "model_index.json",
            "scheduler/*",
            "prior/*",
            "text_encoder/*",
            "tokenizer/*",
            "*.json",
            "*.safetensors",
            "*.bin",
        ],
    ),
    (
        "stabilityai/stable-cascade",
        "StableCascade-Decoder",
        ["model_index.json", "scheduler/*", "decoder/*", "vqgan/*", "*.json", "*.safetensors", "*.bin"],
    ),
]


def _has_model_weights(local_dir: Path) -> bool:
    return any(p.suffix in (".safetensors", ".bin", ".pt", ".ckpt") for p in local_dir.rglob("*") if p.is_file())


def _stack(args: argparse.Namespace) -> int:
    if not _have_hub():
        return 1
    mode = str(args.mode)
    user_workers = int(args.max_workers)
    mw = (
        max(12, user_workers) if mode == "fast" else min(2, user_workers) if mode == "low-mem" else min(6, user_workers)
    )
    use_hf_transfer = bool(args.hf_transfer or mode == "fast")
    skip_existing = bool(args.skip_existing or mode in ("balanced", "low-mem"))
    print(
        f"Downloader config: mode={mode} max_workers={mw} hf_transfer={use_hf_transfer} skip_existing={skip_existing}"
    )
    if use_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    ok = failed = 0
    for repo_id, folder, allow in STACK_JOBS:
        local_dir = Path(args.model_dir) / folder
        if skip_existing and local_dir.is_dir() and _has_model_weights(local_dir):
            print(f"Skipping (already has weights): {repo_id} -> {local_dir}")
            ok += 1
            continue
        print(f"Downloading {repo_id} -> {local_dir}")
        try:
            _snapshot(repo_id, local_dir, allow_patterns=allow, max_workers=mw)
            ok += 1
        except Exception as e:
            if not args.continue_on_error:
                raise
            print(f"Failed: {repo_id} -> {local_dir}: {e}", file=sys.stderr)
            failed += 1
    print(f"Done. downloaded={ok} failed={failed} target_dir={args.model_dir}")
    return 0


# --- datasets (booru scrape) --------------------------------------------------


def _site_rate(site: str, args: argparse.Namespace) -> float:
    from scripts.scrape import RATE_LIMITS

    if args.rate is not None:
        return float(args.rate)
    env = os.environ.get(f"SDX_API_RATE_{site.upper()}")
    return float(env) if env else float(RATE_LIMITS.get(site, 2.0))


def _scrape_site(site: str, args: argparse.Namespace):
    from scripts.scrape import BooruClient, build_adapter, get_credentials, parse_ratings

    creds = get_credentials(site, args.secrets)
    rate = _site_rate(site, args)
    out_dir = Path(args.out) / site
    client = BooruClient(
        build_adapter(site, creds),
        out_dir,
        user_agent=f"sdx-dataset-scraper/1.0 (by {creds.username or creds.email or 'anonymous'})",
        rate_per_sec=rate,
        ratings=parse_ratings(args.ratings),
        max_workers=int(args.workers),
        split_frames=bool(args.split_frames),
        frame_fps=float(args.frame_fps),
        max_frames_per_post=int(args.max_frames_per_post),
        delete_raw_after_split=not args.keep_raw_media,
    )
    print(
        f"[{site}] start -> {out_dir} (ratings={args.ratings}, {rate}/s api, "
        f"{args.workers} dl threads, split_frames={args.split_frames})",
        flush=True,
    )
    stats = client.run(args.tags, max_posts=args.max_posts, dry_run=args.dry_run)
    print(
        f"[{site}] done: downloaded={stats.downloaded} fetched={stats.fetched} "
        f"posts_split={stats.posts_split} frames={stats.frames_extracted} "
        f"blocked_unsafe={stats.skipped_unsafe} blocked_ext={stats.skipped_blocked_ext} "
        f"invalid_media={stats.skipped_invalid_media} errors={stats.errors}",
        flush=True,
    )
    return stats


def _datasets(args: argparse.Namespace) -> int:
    """Booru crawl. The mandatory CSAM-safety gate in scripts/scrape.py runs on every post and cannot be disabled."""
    from scripts.scrape import DEFAULT_SCRAPE_SITES, get_secrets_path
    from utils.prompt.text import build_from_manifests

    if args.secrets is None:
        args.secrets = str(get_secrets_path())
    if args.sites is not None:
        sites = args.sites
    elif os.environ.get("SDX_SCRAPE_SITES", "").strip():
        sites = os.environ["SDX_SCRAPE_SITES"].replace(",", " ").split()
    else:
        sites = list(DEFAULT_SCRAPE_SITES)
    sites = list(dict.fromkeys(sites))
    print(f"Crawling {sites} -> {args.out}\n")

    results = {}
    # Pagination is sequential within a site; sites run in parallel, each with its own download pool.
    with ThreadPoolExecutor(max_workers=len(sites)) as pool:
        futures = {pool.submit(_scrape_site, s, args): s for s in sites}
        for fut in as_completed(futures):
            site = futures[fut]
            try:
                results[site] = fut.result()
            except Exception as e:  # one site failing shouldn't kill the others
                print(f"[{site}] ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    print("\n=== summary ===")
    failed_sites = [s for s in sites if s not in results]
    for site in sites:
        st = results.get(site)
        if st is None:
            print(f"  {site}: failed")
            continue
        print(
            f"  {site}: {st.downloaded} images, {st.posts_split} split, "
            f"{st.frames_extracted} frames, {st.skipped_unsafe} blocked, {st.errors} errors"
        )
    print(f"total downloaded: {sum(st.downloaded for st in results.values())}")
    if failed_sites:
        print(f"FAILED sites: {', '.join(failed_sites)}", file=sys.stderr)
        return 1
    out = Path(args.out)
    if not args.dry_run:
        manifests = [out / s / "manifest.jsonl" for s in sites if (out / s / "manifest.jsonl").is_file()]
        if manifests:
            reg = build_from_manifests(manifests)
            reg.save(out / "artist_index.json")
            print(f"artist index: {len(reg):,} names -> {out / 'artist_index.json'}")
    print("\nTrain on the combined set with, e.g.:")
    for site in sites:
        print(f"  python train.py --manifest-jsonl {out / site / 'manifest.jsonl'} --data-path {out / site} ...")
    return 0


# --- hf-datasets (HF booru-style packs) ---------------------------------------


def _truthy_env(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip() in ("1", "true", "yes")


def _manifest_ok(site_dir: Path) -> bool:
    m = site_dir / "manifest.jsonl"
    return m.is_file() and m.stat().st_size >= 32


def _export_cmd(spec: dict, *, out_dir: Path, max_samples: int, image_format: str, turbo: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.tools",
        "hf_export_to_sdx_manifest",
        "--dataset",
        spec["dataset"],
        "--split",
        str(spec.get("split") or "train"),
        "--image-field",
        str(spec.get("image_field") or "image"),
        "--caption-field",
        str(spec.get("caption_field") or "tag_string"),
        "--out-dir",
        str(out_dir),
        "--streaming",
        "--image-format",
        image_format,
    ]
    for key, flag in (("config", "--config"), ("revision", "--revision"), ("caption_tag_join", "--caption-tag-join")):
        if spec.get(key):
            cmd.extend([flag, str(spec[key])])
    if max_samples > 0:
        cmd.extend(["--max-samples", str(max_samples)])
    if turbo:
        cmd.append("--turbo")
    return cmd


def _hf_datasets(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_NUM_THREADS", "32")
    try:
        _hf_token()
    except Exception:
        pass
    turbo = _truthy_env("SDX_HF_TURBO", "1")
    parallel = args.parallel or int(os.environ.get("SDX_HF_PARALLEL_PACKS", "1") or 1)
    force = args.force or _truthy_env("SDX_HF_FORCE")

    packs = {x["name"]: x for x in json.loads(PACKS_JSON.read_text(encoding="utf-8")).get("packs") or []}
    if args.only:
        want = [n for n in args.only if n in packs]
    elif os.environ.get("SDX_HF_SITES", "").strip():
        want = os.environ["SDX_HF_SITES"].replace(",", " ").split()
    else:
        want = list(packs)
    missing = [n for n in want if n not in packs]
    if missing:
        print(f"Unknown pack names: {missing} (known: {sorted(packs)})", file=sys.stderr)
        return 2
    if not want:
        print("No HF dataset packs selected.", file=sys.stderr)
        return 2

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Hugging Face datasets: {len(want)} packs -> {dest}")
    for name in want:
        print(f"  {name:12} {packs[name].get('site') or name}  <-  {packs[name]['dataset']}")
    print(f"  max_samples per pack: {args.max_samples or 'unlimited'}")
    print(
        f"  turbo: {turbo}  parallel_packs: {parallel}  export_workers: {os.environ.get('SDX_HF_EXPORT_WORKERS', '?')}"
    )
    print(f"  hf_transfer: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '?')}\n")

    todo = []
    for name in want:
        if not force and _manifest_ok(dest / name):
            print(f"[{name}] already exported — skip ({dest / name / 'manifest.jsonl'})")
        else:
            todo.append(name)
    skipped = len(want) - len(todo)

    def run_pack(name: str) -> int:
        spec = packs[name]
        print(f"[{name}] <- {spec['dataset']}  ->  {dest / name}")
        cmd = _export_cmd(
            spec, out_dir=dest / name, max_samples=args.max_samples, image_format=args.image_format, turbo=turbo
        )
        return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode

    ok = skipped
    with ThreadPoolExecutor(max_workers=max(1, min(parallel, len(todo) or 1))) as pool:
        for name, rc in zip(todo, pool.map(run_pack, todo), strict=True):
            if rc == 0:
                ok += 1
            else:
                print(f"  FAILED: {name}", file=sys.stderr)

    print(f"\nDone: {ok}/{len(want)} packs ({skipped} skipped as present).")
    return 0 if ok == len(want) else 1


# --- CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from scripts.scrape import ADAPTERS, DEFAULT_SCRAPE_SITES

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pretrained", help="Download SDX pretrained models from pretrained_status.json")
    p.add_argument("--dest", default=str(PRETRAINED), help="Destination base dir.")
    p.add_argument("--only", nargs="*", default=None, help="Download only these model names.")
    p.add_argument(
        "--profile",
        default=os.environ.get("SDX_MODEL_PROFILE", "full"),
        help="train | enrich | inference | ultimate | full (default: full or SDX_MODEL_PROFILE).",
    )
    p.add_argument(
        "--workers", type=int, default=0, help="Parallel file workers per repo (default 4 with HF_TOKEN else 2)."
    )
    p.add_argument("--retries", type=int, default=5, help="Retries per repo on network error.")
    p.add_argument("--force", action="store_true", help="Re-download even when folder looks complete.")
    p.set_defaults(fn=_pretrained)

    p = sub.add_parser("datasets", help="Booru dataset downloader (default: danbooru + rule34.xxx + e621 + rule34.xyz)")
    p.add_argument("--out", required=True, help="Base output dir (per-site subfolders created).")
    p.add_argument(
        "--sites",
        nargs="*",
        default=None,
        choices=sorted(ADAPTERS),
        help=f"Sites to crawl (default: {list(DEFAULT_SCRAPE_SITES)} or $SDX_SCRAPE_SITES).",
    )
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
        "--keep-raw-media", action="store_true", help="Keep original gif/mp4/webm after frame split (doubles disk use)."
    )
    p.add_argument("--rate", type=float, default=None, help="API req/s per site (default: per-site polite value).")
    p.add_argument(
        "--secrets", default=None, help="Secrets file (default $SDX_SECRETS_FILE or D:\\Development\\secret.txt)."
    )
    p.add_argument("--dry-run", action="store_true", help="Fetch + filter only; download nothing.")
    p.set_defaults(fn=_datasets)

    p = sub.add_parser("hf-datasets", help="HF dataset packs -> SDX per-site folders + manifests")
    p.add_argument("--dest", default=os.environ.get("SDX_DATA", "/workspace/data"), help="Data root.")
    p.add_argument("--only", nargs="*", default=None, help="Subset of pack names.")
    p.add_argument(
        "--max-samples",
        type=int,
        default=int(os.environ.get("SDX_HF_MAX_SAMPLES", os.environ.get("SDX_MAX_SAMPLES", "0"))),
        help="Per-pack row cap (0 = full stream).",
    )
    p.add_argument("--force", action="store_true", help="Re-export even when manifest exists.")
    p.add_argument(
        "--image-format", default=os.environ.get("SDX_HF_IMAGE_FORMAT", "jpg"), choices=("jpg", "png", "webp")
    )
    p.add_argument("--parallel", type=int, default=0, help="Export N packs at once (0=SDX_HF_PARALLEL_PACKS).")
    p.set_defaults(fn=_hf_datasets)

    p = sub.add_parser("t5", help="Ensure T5-XXL has model.safetensors (avoids torch>=2.6 requirement for .bin)")
    p.set_defaults(fn=_t5)

    p = sub.add_parser("repa", help="Download REPA encoder (DINOv3-L by default) if missing")
    p.add_argument(
        "--model",
        default=os.environ.get("SDX_REPA_ENCODER", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
        help="HF id or local path",
    )
    p.add_argument("--dest", default=os.environ.get("SDX_PRETRAINED", "/workspace/pretrained"), help="Pretrained root")
    p.set_defaults(fn=_repa)

    p = sub.add_parser("models", help="Hand-picked T5 / VAE / CLIP / LLM / advanced sets into pretrained/")
    p.add_argument("--model-dir", type=str, default=str(PRETRAINED), help="Base dir (default: project pretrained/)")
    p.add_argument("--t5", action="store_true", help="T5 text encoders: XXL (best), XL, Large")
    p.add_argument("--vae", action="store_true", help="VAEs: REPA-E FLUX, sd-vae-ft-mse/ema, sdxl-vae, fp16-fix")
    p.add_argument("--clip", action="store_true", help="CLIP ViT-L/14 (optional, for triple/penta stacks)")
    p.add_argument("--triple-text-encoders", action="store_true", help="T5-XXL + CLIP-L + CLIP-bigG")
    p.add_argument("--penta-text-encoders", action="store_true", help="T5-XXL + CLIP-L + bigG + H + LongCLIP-L")
    p.add_argument("--llm", action="store_true", help="Default LLM (SmolLM2-360M) for prompt expansion")
    p.add_argument("--llm-best", action="store_true", help="Best LLM (Qwen3-14B) for prompt expansion")
    p.add_argument(
        "--advanced",
        action="store_true",
        help="Advanced optional models (LongCLIP, moondream3, HPSv3, DINOv3, SigLIP2, Qwen3-8B, Marigold, TAESD, "
        "CodeFormer, ConvNeXtV2, consistency decoder, aesthetic predictor, AnyDoor ref).",
    )
    p.add_argument(
        "--config-only",
        action="store_true",
        help="With --advanced: configs/tokenizers only (no weights). Prefer `scaffold` for the full registry.",
    )
    p.add_argument("--all", action="store_true", help="All T5 sizes, all VAEs, CLIP, and both LLMs")
    p.add_argument("--max-workers", type=int, default=4, help="Parallel download workers")
    p.set_defaults(fn=_models)

    p = sub.add_parser("llm", help="Download an LLM for prompt expansion (--best = Qwen3-14B)")
    p.add_argument("--best", action="store_true", help="Download the best-quality model: Qwen3-14B")
    p.add_argument("--model", type=str, default=LLM_DEFAULT, help="HF model ID (default: fast 360M)")
    p.add_argument("--cache-dir", type=str, default=None, help="HF cache dir (default: HF_HOME)")
    p.add_argument("--local-dir", type=str, default=None, help="Save to this dir instead of the HF cache.")
    p.add_argument("--max-workers", type=int, default=8, help="Max parallel download workers")
    p.set_defaults(fn=_llm)

    p = sub.add_parser("scaffold", help="HF config-only scaffolds into pretrained/")
    p.add_argument("--model-dir", type=str, default="", help="Target root (default: pretrained/)")
    p.add_argument("--all", action="store_true", help="Download full scaffold registry")
    p.add_argument("--role", action="append", default=[], help="Filter by role (text_encoder, vlm, reward, ...).")
    p.add_argument("--name", action="append", default=[], help="Download specific catalog names. Repeatable.")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--force", action="store_true", help="Re-download even if folder exists")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first download error")
    p.add_argument("--list", action="store_true", help="List registry and exit")
    p.add_argument("--out-json", type=str, default="", help="Write download report JSON")
    p.add_argument(
        "--penta", action="store_true", help="Penta text encoder scaffolds (T5-XXL, CLIP-L, bigG, H, LongCLIP)"
    )
    p.add_argument("--triple", action="store_true", help="Triple text encoder scaffolds (T5-XXL, CLIP-L, bigG)")
    p.set_defaults(fn=_scaffold)

    p = sub.add_parser("stack", help="Curated high-end SDX model stack")
    p.add_argument("--model-dir", default=str(PRETRAINED))
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument(
        "--mode",
        choices=("fast", "balanced", "low-mem"),
        default="balanced",
        help="fast: max throughput, balanced: safe default, low-mem: lower RAM/network pressure",
    )
    p.add_argument("--hf-transfer", action="store_true", help="Enable HF transfer acceleration if available")
    p.add_argument("--skip-existing", action="store_true", help="Skip repos that already have model weights")
    p.add_argument("--continue-on-error", action="store_true", help="Skip repos that fail and keep going")
    p.set_defaults(fn=_stack)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
