#!/usr/bin/env python3
"""End-to-end integration smoke test before RunPod spend.

Runs all-site scrape (download + GIF/video frame split) → merge → enrich → RAG →
control maps → dataset load → feature checks → optional 1-step train dry-run.

    python scripts/integration_smoke.py
    python scripts/integration_smoke.py --skip-train
    python scripts/integration_smoke.py --data-root data/integration_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_env_defaults() -> None:
    """Apply runpod/env.defaults (Linux paths) without clobbering existing env."""
    import platform
    import re

    path = ROOT / "runpod" / "env.defaults"
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'export\s+(\w+)="\$\{\1:-([^}]*)\}"', line)
            if m:
                key, default = m.group(1), m.group(2)
                os.environ.setdefault(key, default)
    if platform.system().lower() == "windows":
        win = {
            "SDX_SECRETS_FILE": r"D:\Development\secret.txt",
            "SDX_DATA": str(ROOT / "data"),
            "SDX_PRETRAINED": str(ROOT / "pretrained"),
            "SDX_RESULTS": str(ROOT / "results"),
        }
        for key, local in win.items():
            val = os.environ.get(key, "")
            if not val or val.startswith("/"):
                os.environ[key] = local


@dataclass
class StepResult:
    name: str
    status: str  # PASS | FAIL | SKIP
    seconds: float = 0.0
    detail: str = ""


@dataclass
class SmokeReport:
    steps: list[StepResult] = field(default_factory=list)

    def add(self, name: str, status: str, *, seconds: float = 0.0, detail: str = "") -> None:
        self.steps.append(StepResult(name, status, seconds, detail))
        icon = {"PASS": "ok", "FAIL": "FAIL", "SKIP": "skip"}.get(status, status)
        line = f"[{icon}] {name}"
        if seconds:
            line += f" ({seconds:.1f}s)"
        if detail:
            line += f" — {detail}"
        print(line, flush=True)

    def ok(self) -> bool:
        return all(s.status != "FAIL" for s in self.steps)

    def summary(self) -> str:
        passed = sum(1 for s in self.steps if s.status == "PASS")
        failed = sum(1 for s in self.steps if s.status == "FAIL")
        skipped = sum(1 for s in self.steps if s.status == "SKIP")
        return f"{passed} passed, {failed} failed, {skipped} skipped"


def _run_step(report: SmokeReport, name: str, fn: Callable[[], str | None], *, skip: bool = False) -> None:
    if skip:
        report.add(name, "SKIP", detail="flagged skip")
        return
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        report.add(name, "PASS", seconds=time.perf_counter() - t0, detail=detail)
    except Exception as e:
        report.add(name, "FAIL", seconds=time.perf_counter() - t0, detail=f"{e}")
        traceback.print_exc()


def _py(args: list[str], *, cwd: Path = ROOT, timeout: int = 600) -> str:
    cmd = [sys.executable] + args
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-800:]
        raise RuntimeError(f"exit {r.returncode}: {tail}")
    return (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""


ALL_SCRAPE_SITES = ("danbooru", "e621", "rule34xxx", "rule34xyz")


def _read_manifest_rows(path: Path) -> list[dict]:
    from data.manifest_utils import read_manifest_rows

    return read_manifest_rows(path)


def _training_manifest(combined: Path, enriched: Path) -> Path:
    from data.manifest_utils import pick_training_manifest

    return pick_training_manifest(combined, enriched)


def _enrich_timeout_seconds(combined: Path, *, base: int = 120, per_row: float = 0.5, cap: int = 7200) -> int:
    n = len(_read_manifest_rows(combined))
    return min(cap, max(base, int(n * per_row)))


def _scrape_cmd(data_root: Path, secrets: Path, *, sites: list[str], max_posts: int, tags: str = "") -> list[str]:
    cmd = [
        "setup/download_datasets.py",
        "--out",
        str(data_root),
        "--sites",
        *sites,
        "--max-posts",
        str(max_posts),
        "--workers",
        "4",
        "--secrets",
        str(secrets),
        "--split-frames",
    ]
    if tags:
        cmd.extend(["--tags", tags])
    return cmd


def _frame_count_by_site(data_root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for site in ALL_SCRAPE_SITES:
        out[site] = sum(1 for r in _read_manifest_rows(data_root / site / "manifest.jsonl") if r.get("parent_md5"))
    return out


def _scrape_site_tags(
    data_root: Path,
    secrets: Path,
    site: str,
    *,
    max_posts: int,
    tags: str,
    timeout: int = 600,
) -> None:
    _py(_scrape_cmd(data_root, secrets, sites=[site], max_posts=max_posts, tags=tags), timeout=timeout)


def _pytest(patterns: list[str]) -> str:
    patterns = patterns or [
        "tests/test_image_profiler.py",
        "tests/test_reverse_search.py",
        "tests/test_data_pipeline.py",
    ]
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *patterns, "-q", "--tb=line"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError((r.stdout or r.stderr or "")[-600:])
    return "pytest green"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SDX integration smoke test (mini scrape → train).")
    p.add_argument("--data-root", default=str(ROOT / "data" / "integration_smoke"))
    p.add_argument("--secrets", default=os.environ.get("SDX_SECRETS_FILE", r"D:\Development\secret.txt"))
    p.add_argument("--scrape-posts", type=int, default=15, help="Max posts per site (all 4 booru sites).")
    p.add_argument("--frame-scrape-posts", type=int, default=10, help="Max animated posts per site (frame split test).")
    p.add_argument("--frame-scrape-tags", default="animated", help="Tag query for frame-split scrape.")
    p.add_argument("--skip-scrape", action="store_true", help="Reuse existing data under --data-root.")
    p.add_argument("--skip-frame-scrape", action="store_true", help="Skip animated/GIF scrape pass.")
    p.add_argument("--skip-train", action="store_true", help="Skip train.py dry-run (needs GPU + HF weights).")
    p.add_argument("--skip-reverse", action="store_true", help="Skip SauceNAO/TinEye web upload test.")
    p.add_argument("--train-steps", type=int, default=0, help="If >0 use --max-steps instead of --dry-run.")
    args = p.parse_args(argv)

    _load_env_defaults()
    data_root = Path(args.data_root).resolve()
    secrets = Path(args.secrets).resolve()
    combined = data_root / "combined" / "manifest.jsonl"
    enriched = data_root / "enriched" / "manifest.jsonl"
    rag = data_root / "rag_corpus.jsonl"
    control_manifest = data_root / "control" / "manifest.jsonl"
    results_dir = (ROOT / "results" / "integration_smoke").resolve()

    report = SmokeReport()
    print("SDX integration smoke", flush=True)
    print(f"  data_root={data_root}  (scrapes -> <site>/images/, manifests per site)", flush=True)
    print(f"  combined={combined}", flush=True)
    print(f"  enriched={enriched}", flush=True)
    print(f"  secrets={secrets}", flush=True)
    print(f"  train_results={results_dir}", flush=True)

    def secrets_parse() -> str:
        from scripts.scrape.secrets_config import get_secrets_path, parse_secrets_file

        path = secrets if secrets.is_file() else get_secrets_path()
        creds = parse_secrets_file(path)
        return f"sites={sorted(creds.keys())}"

    _run_step(report, "secrets_parse", secrets_parse)

    def unit_tests() -> str:
        return _pytest(
            [
                "tests/test_image_profiler.py",
                "tests/test_reverse_search.py",
                "tests/test_prompt_research.py",
                "tests/test_auto_oc.py",
                "tests/test_prompt_composer_artists.py",
                "tests/test_frame_split.py",
                "tests/test_rule34xyz_v2.py",
                "tests/test_data_pipeline.py::TestText2ImageDatasetJSONL::test_relative_paths_with_data_root",
            ]
        )

    _run_step(report, "unit_tests", unit_tests)

    def scrape_all_sites() -> str:
        data_root.mkdir(parents=True, exist_ok=True)
        if not secrets.is_file():
            raise FileNotFoundError(f"secrets missing: {secrets}")
        _py(
            _scrape_cmd(data_root, secrets, sites=list(ALL_SCRAPE_SITES), max_posts=args.scrape_posts),
            timeout=900,
        )
        parts = []
        for site in ALL_SCRAPE_SITES:
            rows = _read_manifest_rows(data_root / site / "manifest.jsonl")
            if not rows:
                raise RuntimeError(f"{site}: manifest empty or missing")
            parts.append(f"{site}={len(rows)}")
        return ", ".join(parts)

    _run_step(report, "scrape_all_sites", scrape_all_sites, skip=args.skip_scrape)

    def scrape_animated_frames() -> str:
        if not secrets.is_file():
            raise FileNotFoundError(f"secrets missing: {secrets}")
        _py(
            _scrape_cmd(
                data_root,
                secrets,
                sites=list(ALL_SCRAPE_SITES),
                max_posts=args.frame_scrape_posts,
                tags=args.frame_scrape_tags,
            ),
            timeout=900,
        )
        by_site = _frame_count_by_site(data_root)
        missing = [s for s, n in by_site.items() if n == 0]
        for site in missing:
            for fallback_tag in ("gif", "video"):
                if fallback_tag == args.frame_scrape_tags:
                    continue
                _scrape_site_tags(
                    data_root,
                    secrets,
                    site,
                    max_posts=max(args.frame_scrape_posts, 15),
                    tags=fallback_tag,
                    timeout=600,
                )
                if _frame_count_by_site(data_root).get(site, 0) > 0:
                    break
        by_site = _frame_count_by_site(data_root)
        still_missing = [s for s, n in by_site.items() if n == 0]
        if still_missing:
            raise RuntimeError(f"no frame rows for sites={still_missing} after animated/gif/video fallbacks")
        frame_rows = sum(by_site.values())
        split_sites = [s for s, n in by_site.items() if n > 0]
        return f"frames={frame_rows} sites={','.join(split_sites)}"

    _run_step(report, "scrape_animated_frames", scrape_animated_frames, skip=args.skip_scrape or args.skip_frame_scrape)

    def verify_site_network() -> str:
        import shutil

        parts = []
        for site in ALL_SCRAPE_SITES:
            rows = _read_manifest_rows(data_root / site / "manifest.jsonl")
            if not rows:
                raise FileNotFoundError(f"{site}: no manifest")
            img_dir = data_root / site / "images"
            n_files = len(list(img_dir.glob("*"))) if img_dir.is_dir() else 0
            missing = 0
            for r in rows[: min(5, len(rows))]:
                rel = str(r.get("image_path", ""))
                p = data_root / site / rel if rel else Path("_missing")
                if not p.is_file():
                    missing += 1
            if n_files == 0:
                raise RuntimeError(f"{site}: no image files on disk")
            parts.append(f"{site}:{len(rows)}rows/{n_files}files")
        ff = shutil.which("ffmpeg")
        return "; ".join(parts) + (f"; ffmpeg={'yes' if ff else 'no'}")

    _run_step(report, "verify_site_network", verify_site_network, skip=args.skip_scrape)

    def verify_frame_splits() -> str:
        by_site = _frame_count_by_site(data_root)
        for site, n in by_site.items():
            if n == 0:
                raise RuntimeError(f"{site}: no frame-split rows (parent_md5) in manifest")
        parents = {
            site: len(
                {
                    r.get("parent_md5")
                    for r in _read_manifest_rows(data_root / site / "manifest.jsonl")
                    if r.get("parent_md5")
                }
            )
            for site in ALL_SCRAPE_SITES
        }
        detail = ", ".join(f"{s}:{by_site[s]}f/{parents[s]}p" for s in ALL_SCRAPE_SITES)
        return f"total_frames={sum(by_site.values())} ({detail})"

    _run_step(report, "verify_frame_splits", verify_frame_splits, skip=args.skip_scrape)

    def merge_manifests() -> str:
        return _py(
            ["setup/merge_manifests.py", "--data-root", str(data_root), "--out", str(combined)],
            timeout=60,
        )

    _run_step(report, "merge_manifests", merge_manifests, skip=args.skip_scrape and combined.is_file())

    def enrich_captions() -> str:
        if not combined.is_file():
            raise FileNotFoundError(combined)
        if enriched.is_file() and not _read_manifest_rows(enriched):
            enriched.unlink(missing_ok=True)
        timeout = _enrich_timeout_seconds(combined)
        return _py(
            [
                "setup/enrich_manifest_captions.py",
                "--manifest",
                str(combined),
                "--data-root",
                str(data_root),
                "--out",
                str(enriched),
                "--booru-only",
                "--no-vlm",
                "--no-reverse-search",
                "--workers",
                "4",
            ],
            timeout=timeout,
        )

    _run_step(report, "enrich_captions", enrich_captions)

    def build_rag() -> str:
        src = _training_manifest(combined, enriched)
        return _py(
            ["setup/build_rag_corpus.py", "--manifest", str(src), "--out", str(rag)],
            timeout=60,
        )

    _run_step(report, "build_rag_corpus", build_rag)

    def control_maps() -> str:
        if not combined.is_file():
            raise FileNotFoundError(combined)
        src = _training_manifest(combined, enriched)
        return _py(
            [
                "setup/preprocess_control_maps.py",
                "--manifest",
                str(src),
                "--data-root",
                str(data_root),
                "--out",
                str(control_manifest),
                "--control-type",
                "canny",
                "--workers",
                "4",
            ],
            timeout=300,
        )

    _run_step(report, "control_maps", control_maps)

    def dataset_load() -> str:
        from data.t2i_dataset import Text2ImageDataset

        man = _training_manifest(combined, enriched)
        ds = Text2ImageDataset(str(man), image_size=64, data_root=str(data_root))
        if len(ds) == 0:
            raise RuntimeError("dataset empty")
        item = ds[0]
        return f"{len(ds)} samples, tensor={tuple(item['pixel_values'].shape)}"

    _run_step(report, "dataset_load", dataset_load)

    def training_wiring() -> str:
        from data.t2i_dataset import Text2ImageDataset

        train_man = _training_manifest(combined, enriched)
        rows = _read_manifest_rows(train_man)
        if not rows:
            raise RuntimeError("training manifest empty")
        if train_man == enriched and not any(r.get("scene_summary") for r in rows):
            raise RuntimeError("enriched manifest missing scene_summary")
        ds = Text2ImageDataset(str(train_man), image_size=64, data_root=str(data_root))
        if len(ds) == 0:
            raise RuntimeError("training dataset empty")
        if control_manifest.is_file():
            ctrl_rows = _read_manifest_rows(control_manifest)
            if ctrl_rows and not ctrl_rows[0].get("control_image"):
                raise RuntimeError("control manifest missing control_image")
        return f"train={train_man.name} samples={len(ds)}"

    _run_step(report, "training_wiring", training_wiring)

    def control_dataset_load() -> str:
        from data.t2i_dataset import Text2ImageDataset

        if not control_manifest.is_file():
            raise FileNotFoundError(control_manifest)
        ds = Text2ImageDataset(str(control_manifest), image_size=64, data_root=str(data_root))
        if len(ds) == 0:
            raise RuntimeError("control dataset empty")
        item = ds[0]
        has_ctrl = "control_image" in item
        return f"{len(ds)} samples, control={has_ctrl}"

    _run_step(report, "control_dataset_load", control_dataset_load)

    def artist_index() -> str:
        from utils.prompt.artist_registry import build_from_manifests

        man = _training_manifest(combined, enriched)
        reg = build_from_manifests([man])
        return f"{len(reg)} artists"

    _run_step(report, "artist_index", artist_index)

    def prompt_composer() -> str:
        from utils.prompt.prompt_composer import compose_prompt

        out = compose_prompt("+character: 1girl, solo | base scene", artist_index=None)
        assert out.positive
        return out.positive[:80]

    _run_step(report, "prompt_composer", prompt_composer)

    def image_profiler() -> str:
        from utils.caption.image_profiler import profile_from_manifest_row

        man = _training_manifest(combined, enriched)
        rows = _read_manifest_rows(man)
        if not rows:
            raise RuntimeError(f"manifest empty: {man}")
        prof = profile_from_manifest_row(rows[0], data_root, use_vlm=False, use_reverse_search=False)
        return f"conf={prof.confidence:.2f} cap_len={len(prof.caption)}"

    _run_step(report, "image_profiler", image_profiler)

    def reverse_search_web() -> str:
        from data.t2i_dataset import Text2ImageDataset
        from utils.caption.reverse_search import reverse_search_file

        man = _training_manifest(combined, enriched)
        ds = Text2ImageDataset(str(man), image_size=64, data_root=str(data_root))
        img_path = ds.samples[0]["path"]
        resolved = ds._resolve_media_path(img_path)
        if resolved is None or not resolved.is_file():
            raise FileNotFoundError(img_path)
        hits = reverse_search_file(resolved, use_saucenao=True, use_tineye=False)
        return f"hits={len(hits)}" + (f" top={hits[0].similarity:.0f}%" if hits else " (none)")

    _run_step(report, "reverse_search_saucenao", reverse_search_web, skip=args.skip_reverse)

    def rag_retrieval() -> str:
        from utils.prompt.rag_prompt import retrieve_facts_for_query_local

        if not rag.is_file():
            raise FileNotFoundError(rag)
        facts = retrieve_facts_for_query_local("1girl solo", rag, top_k=3)
        return f"facts={len(facts)}"

    _run_step(report, "rag_retrieval", rag_retrieval)

    def dit_forward_smoke() -> str:
        return _py(["-m", "scripts.tools", "quick_test"], timeout=120)

    _run_step(report, "dit_forward_smoke", dit_forward_smoke)

    def smoke_dataset() -> str:
        return _py(
            [
                "-m",
                "scripts.tools",
                "make_smoke_dataset",
                "--out",
                str(data_root / "smoke_tiny"),
                "--count",
                "4",
                "--size",
                "128",
            ]
        )

    _run_step(report, "make_smoke_dataset", smoke_dataset)

    def _t5_ready() -> bool:
        candidates: list[Path] = []
        pretrained = os.environ.get("SDX_PRETRAINED", "")
        if pretrained:
            candidates.append(Path(pretrained))
        candidates.append(ROOT / "pretrained")
        for root in candidates:
            if not root.is_dir():
                continue
            for name in ("T5-XXL", "t5-v1_1-xxl", "google_t5-v1_1-xxl"):
                if (root / name).is_dir():
                    return True
        cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--t5-v1_1-xxl" / "snapshots"
        if cache.is_dir():
            for snap in cache.iterdir():
                if (snap / "pytorch_model.bin").is_file():
                    return True
                if any(snap.glob("*.safetensors")):
                    return True
        return False

    def mini_train_manifest() -> str:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if not _t5_ready():
            raise RuntimeError("T5-XXL not cached — run bash runpod/download.sh --models-only first")
        man = _training_manifest(combined, enriched)
        if not man.is_file():
            raise FileNotFoundError(man)
        results_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "train.py",
            "--manifest-jsonl",
            str(man.resolve()),
            "--data-path",
            str(data_root),
            "--results-dir",
            str(results_dir),
            "--model",
            "DiT-B/2-Text",
            "--image-size",
            "128",
            "--global-batch-size",
            "1",
            "--no-compile",
            "--num-workers",
            "0",
            "--log-every",
            "1",
            "--flow-matching-training",
            "--text-encoder",
            os.environ.get("SDX_TEXT_ENCODER", "google/t5-v1_1-xxl"),
        ]
        if args.train_steps > 0:
            cmd += ["--max-steps", str(args.train_steps)]
        else:
            cmd.append("--dry-run")
        return _py(cmd, timeout=1800)

    train_skip = args.skip_train or not _t5_ready()
    train_skip_reason = (
        "T5-XXL not cached (run bash runpod/download.sh --models-only)"
        if not _t5_ready() and not args.skip_train
        else "flagged skip"
    )

    def _run_skip(name: str, skip: bool, reason: str) -> None:
        if skip:
            report.add(name, "SKIP", detail=reason)

    _run_skip("mini_train_dry_run", train_skip, train_skip_reason)
    if not train_skip:
        _run_step(report, "mini_train_dry_run", mini_train_manifest)

    def mini_train_control() -> str:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if not control_manifest.is_file():
            raise FileNotFoundError(control_manifest)
        out = results_dir / "control"
        out.mkdir(parents=True, exist_ok=True)
        return _py(
            [
                "train.py",
                "--manifest-jsonl",
                str(control_manifest),
                "--data-path",
                str(data_root),
                "--results-dir",
                str(out),
                "--model",
                "DiT-B/2-Text",
                "--image-size",
                "128",
                "--global-batch-size",
                "1",
                "--no-compile",
                "--num-workers",
                "0",
                "--dry-run",
                "--control-cond-dim",
                "1",
                "--control-num-types",
                "9",
                "--flow-matching-training",
            ],
            timeout=1800,
        )

    _run_skip("mini_train_control_dry_run", train_skip, train_skip_reason)
    if not train_skip:
        _run_step(report, "mini_train_control_dry_run", mini_train_control)

    print()
    print("=" * 60)
    print(report.summary())
    report_path = data_root / "integration_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {"ok": report.ok(), "summary": report.summary(), "steps": [s.__dict__ for s in report.steps]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    return 0 if report.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
