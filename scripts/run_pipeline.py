#!/usr/bin/env python3
"""Run the full SDX image-gen pipeline in order (setup → smoke → download → train).

Cross-platform entry point. Thin wrappers: ``runpod/run.sh`` / ``runpod/run.ps1``.

Recommended order (default — everything):

  1. setup          Install deps, create dirs, link pretrained
  2. verify         Import / CUDA / ffmpeg sanity checks
  3. smoke          Integration smoke (all-site scrape + pipeline checks)
  4. pretrained     Download HF weights (~100+ GB, resumable)
  5. datasets       Crawl all booru sites + merge manifest + artist index
  6. preprocess     Enrich captions, RAG corpus, control maps
  7. train          Full / LoRA / control training (``SDX_TRAIN_MODE``)
  8. sample         Optional inference smoke (``--with-sample``)

Examples:

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-setup --skip-smoke
    python scripts/run_pipeline.py --from preprocess
    python scripts/run_pipeline.py --only smoke --smoke-args --skip-train
    python scripts/run_pipeline.py --smoke-only
    SDX_TRAIN_MODE=lora python scripts/run_pipeline.py --from train

Environment (see ``runpod/env.defaults``):

    SDX_SECRETS_FILE, SDX_DATA, SDX_PRETRAINED, SDX_RESULTS,
    SDX_TRAIN_MODE, SDX_GLOBAL_BATCH_SIZE, SDX_MAX_POSTS, …
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEP_ORDER = (
    "setup",
    "verify",
    "pretrained",
    "smoke",
    "datasets",
    "preprocess",
    "train",
    "sample",
)


@dataclass
class StepResult:
    name: str
    status: str  # PASS | FAIL | SKIP
    seconds: float = 0.0
    detail: str = ""


@dataclass
class PipelineReport:
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


def _is_windows() -> bool:
    return platform.system().lower() == "windows"


def _load_env_defaults(root: Path) -> None:
    """Apply ``runpod/env.defaults`` without overwriting existing env vars."""
    path = root / "runpod" / "env.defaults"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'export\s+(\w+)="\$\{\1:-([^}]*)\}"', line)
        if not m:
            continue
        key, default = m.group(1), m.group(2)
        if key not in os.environ:
            os.environ[key] = default

    # Windows-friendly fallbacks when env.defaults uses /workspace paths.
    if _is_windows():
        win_defaults = {
            "SDX_SECRETS_FILE": r"D:\Development\secret.txt",
            "SDX_DATA": str(root / "data"),
            "SDX_PRETRAINED": str(root / "pretrained"),
            "SDX_RESULTS": str(root / "results"),
            "SDX_ROOT": str(root),
            "HF_HOME": str(root / ".cache" / "huggingface"),
        }
        for key, local in win_defaults.items():
            val = os.environ.get(key, "")
            if not val or val.startswith("/"):
                os.environ[key] = local


def _paths() -> dict[str, Path]:
    data = Path(os.environ.get("SDX_DATA", str(ROOT / "data")))
    pretrained = Path(os.environ.get("SDX_PRETRAINED", str(ROOT / "pretrained")))
    results = Path(os.environ.get("SDX_RESULTS", str(ROOT / "results")))
    secrets = Path(os.environ.get("SDX_SECRETS_FILE", r"D:\Development\secret.txt"))
    return {
        "data": data,
        "pretrained": pretrained,
        "results": results,
        "secrets": secrets,
        "combined": data / "combined" / "manifest.jsonl",
        "enriched": data / "enriched" / "manifest.jsonl",
        "rag": Path(os.environ.get("SDX_RAG_CORPUS", str(data / "rag_corpus.jsonl"))),
        "control": Path(os.environ.get("SDX_CONTROL_MANIFEST", str(data / "control" / "manifest.jsonl"))),
        "artist_index": Path(os.environ.get("SDX_ARTIST_INDEX", str(data / "artist_index.json"))),
        "smoke_data": data / "integration_smoke",
    }


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int | None = None,
    shell: bool = False,
) -> str:
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd) if not shell else cmd}", flush=True)
    r = subprocess.run(cmd, cwd=str(cwd), timeout=timeout, shell=shell)
    if r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode}")
    return ""


def _py(args: list[str], *, timeout: int | None = None) -> str:
    return _run_cmd([sys.executable, *args], timeout=timeout)


def _run_step(
    report: PipelineReport,
    name: str,
    fn: Callable[[], str | None],
    *,
    skip: bool = False,
    detail: str = "",
) -> bool:
    if skip:
        report.add(name, "SKIP", detail=detail or "flagged skip")
        return True
    t0 = time.perf_counter()
    try:
        out = fn() or ""
        report.add(name, "PASS", seconds=time.perf_counter() - t0, detail=out)
        return True
    except Exception as e:
        report.add(name, "FAIL", seconds=time.perf_counter() - t0, detail=str(e))
        return False


def _step_setup() -> None:
    if _is_windows():
        ps1 = ROOT / "runpod" / "setup.ps1"
        _run_cmd(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)], timeout=3600)
    else:
        _run_cmd(["bash", str(ROOT / "runpod" / "setup.sh")], timeout=7200)


def _step_verify() -> None:
    if _is_windows():
        _py(["-m", "toolkit.training.env_health"], timeout=120)
        _py(
            [
                "-c",
                "import importlib; mods=['torch','transformers','diffusers','PIL','cv2','requests','rich']; "
                "[importlib.import_module(m) for m in mods]; "
                "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}')",
            ],
            timeout=120,
        )
        import shutil

        for bin_name in ("ffmpeg", "git"):
            if not shutil.which(bin_name):
                print(f"  WARN: {bin_name} not on PATH")
    else:
        _run_cmd(["bash", str(ROOT / "runpod" / "verify_env.sh")], timeout=300)


def _step_smoke(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    smoke_root = paths["smoke_data"] if args.smoke_data_root is None else Path(args.smoke_data_root)
    cmd = [
        "scripts/integration_smoke.py",
        "--data-root",
        str(smoke_root),
        "--secrets",
        str(paths["secrets"]),
    ]
    if args.skip_smoke_train:
        cmd.append("--skip-train")
    if args.skip_smoke_scrape:
        cmd.append("--skip-scrape")
    if args.skip_smoke_reverse:
        cmd.append("--skip-reverse")
    cmd.extend(args.smoke_extra)
    _py(cmd, timeout=args.smoke_timeout)


def _step_pretrained(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    paths["pretrained"].mkdir(parents=True, exist_ok=True)
    cmd = [
        "setup/download_pretrained.py",
        "--dest",
        str(paths["pretrained"]),
        "--workers",
        str(args.pretrained_workers),
    ]
    _py(cmd, timeout=args.pretrained_timeout or None)


def _step_datasets(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    paths["data"].mkdir(parents=True, exist_ok=True)
    if not paths["secrets"].is_file():
        raise FileNotFoundError(f"secrets missing: {paths['secrets']}")
    dl = [
        "setup/download_datasets.py",
        "--out",
        str(paths["data"]),
        "--ratings",
        "all",
        "--workers",
        str(args.scrape_workers),
        "--max-posts",
        str(args.max_posts),
        "--secrets",
        str(paths["secrets"]),
        "--frame-fps",
        str(args.frame_fps),
        "--max-frames-per-post",
        str(args.max_frames_per_post),
    ]
    if args.split_frames:
        dl.append("--split-frames")
    else:
        dl.append("--no-split-frames")
    if args.keep_raw_media:
        dl.append("--keep-raw-media")
    if args.scrape_sites:
        dl.extend(["--sites", *args.scrape_sites])
    _py(dl, timeout=args.dataset_timeout or None)
    _py(
        [
            "setup/merge_manifests.py",
            "--data-root",
            str(paths["data"]),
            "--out",
            str(paths["combined"]),
        ],
        timeout=600,
    )
    _py(
        [
            "setup/build_artist_index.py",
            "--data-root",
            str(paths["data"]),
            "--out",
            str(paths["artist_index"]),
        ],
        timeout=600,
    )


def _step_preprocess(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    from data.manifest_utils import manifest_has_rows

    manifest = paths["combined"]
    if not manifest.is_file():
        raise FileNotFoundError(f"combined manifest missing: {manifest}")

    rag_path = paths["rag"]
    rag_path.parent.mkdir(parents=True, exist_ok=True)
    print("  seed RAG corpus (booru tags)", flush=True)
    _py(
        [
            "setup/build_rag_corpus.py",
            "--manifest",
            str(manifest),
            "--out",
            str(rag_path),
        ],
        timeout=600,
    )

    enrich_out = paths["enriched"]
    enrich_out.parent.mkdir(parents=True, exist_ok=True)
    prompt_research = os.environ.get("SDX_PROMPT_RESEARCH", "1") == "1"
    enrich_workers = 1 if prompt_research else args.enrich_workers
    enrich = [
        "setup/enrich_manifest_captions.py",
        "--manifest",
        str(manifest),
        "--data-root",
        str(paths["data"]),
        "--out",
        str(enrich_out),
        "--workers",
        str(enrich_workers),
    ]
    if prompt_research:
        enrich.extend(
            [
                "--prompt-research",
                "--rag-corpus",
                str(rag_path),
            ]
        )
    else:
        if os.environ.get("SDX_ENRICH_VLM", "0") != "1":
            enrich.append("--no-vlm")
        if os.environ.get("SDX_ENRICH_REVERSE", "0") != "1":
            enrich.append("--no-reverse-search")
        enrich.append("--booru-only")
    try:
        _py(enrich, timeout=args.enrich_timeout)
    except RuntimeError:
        print("  WARN: caption enrichment failed (non-fatal)", flush=True)

    rag_manifest = manifest
    if enrich_out.is_file():
        rows = [ln for ln in enrich_out.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if rows:
            rag_manifest = enrich_out
    print("  final RAG corpus (researched captions)", flush=True)
    _py(
        [
            "setup/build_rag_corpus.py",
            "--manifest",
            str(rag_manifest),
            "--out",
            str(rag_path),
        ],
        timeout=600,
    )
    control_src = manifest
    if manifest_has_rows(enrich_out):
        control_src = enrich_out
    _py(
        [
            "setup/preprocess_control_maps.py",
            "--manifest",
            str(control_src),
            "--data-root",
            str(paths["data"]),
            "--out",
            str(paths["control"]),
            "--control-type",
            os.environ.get("SDX_CONTROL_TYPE", "canny"),
            "--workers",
            str(args.control_workers),
        ],
        timeout=args.control_timeout,
    )


def _step_train(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    mode = os.environ.get("SDX_TRAIN_MODE", "full")
    enriched = paths["enriched"]
    manifest = Path(os.environ.get("SDX_MANIFEST", str(paths["combined"])))
    extra: list[str] = []

    max_steps = os.environ.get("SDX_MAX_STEPS", "")
    if max_steps:
        extra.extend(["--max-steps", max_steps])
    init_ckpt = os.environ.get("SDX_INIT_CKPT", "")
    if init_ckpt:
        extra.extend(["--init-from", init_ckpt])

    if mode in ("full", "lora") and not os.environ.get("SDX_MANIFEST"):
        from data.manifest_utils import pick_training_manifest

        manifest = pick_training_manifest(paths["combined"], enriched)

    if mode == "lora":
        extra.extend(
            [
                "--lora-train",
                "--lora-rank",
                os.environ.get("SDX_LORA_RANK", "32"),
                "--lora-alpha",
                os.environ.get("SDX_LORA_ALPHA", "32"),
            ]
        )
    elif mode == "control":
        manifest = paths["control"]
        extra.extend(
            [
                "--control-cond-dim",
                "1",
                "--control-num-types",
                "9",
                "--control-scale",
                os.environ.get("SDX_CONTROL_SCALE", "0.85"),
            ]
        )
    elif mode == "lora_control":
        manifest = paths["control"]
        extra.extend(
            [
                "--lora-train",
                "--lora-rank",
                os.environ.get("SDX_LORA_RANK", "32"),
                "--lora-alpha",
                os.environ.get("SDX_LORA_ALPHA", "32"),
                "--control-cond-dim",
                "1",
                "--control-num-types",
                "9",
                "--control-scale",
                os.environ.get("SDX_CONTROL_SCALE", "0.85"),
            ]
        )
    elif mode != "full":
        raise ValueError(f"Unknown SDX_TRAIN_MODE={mode!r}")

    if not manifest.is_file():
        raise FileNotFoundError(f"training manifest missing: {manifest}")

    paths["results"].mkdir(parents=True, exist_ok=True)
    cmd = [
        "train.py",
        "--manifest-jsonl",
        str(manifest),
        "--data-path",
        str(paths["data"]),
        "--results-dir",
        str(paths["results"]),
        "--flow-matching-training",
        "--live-dashboard",
        "--train-style-guidance-mode",
        "auto",
        "--region-caption-mode",
        "append",
        "--epochs",
        os.environ.get("SDX_EPOCHS", "20"),
        "--global-batch-size",
        os.environ.get("SDX_GLOBAL_BATCH_SIZE", "4"),
        "--image-size",
        os.environ.get("SDX_IMAGE_SIZE", "512"),
        *extra,
    ]
    _py(cmd, timeout=args.train_timeout or None)


def _step_sample(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    ckpt = Path(os.environ.get("SDX_SAMPLE_CKPT", str(paths["results"] / "best.pt")))
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    prompt = os.environ.get(
        "SDX_PROMPT",
        "@wlop +character: 1girl, silver hair +scene: cherry blossoms",
    )
    cmd = [
        "sample.py",
        "--ckpt",
        str(ckpt),
        "--prompt",
        prompt,
    ]
    rag = paths["rag"]
    if rag.is_file():
        cmd.extend(["--local-rag-jsonl", str(rag), "--local-rag-top-k", os.environ.get("SDX_RAG_TOP_K", "8")])
    box = os.environ.get("SDX_BOX_LAYOUT", "")
    if box and Path(box).is_file():
        cmd.extend(["--box-layout", box, "--box-layout-mode", "regional_cfg"])
    lora = os.environ.get("SDX_LORA", "")
    if lora and Path(lora).is_file():
        cmd.extend(["--lora", f"{lora}:1.0"])
    _py(cmd, timeout=600)


def _resolve_steps(args: argparse.Namespace) -> list[tuple[str, bool]]:
    if args.smoke_only:
        names = ["smoke"]
    elif args.only:
        names = [args.only]
    else:
        names = list(STEP_ORDER)
        if not args.with_sample:
            names = [s for s in names if s != "sample"]
        if args.from_step:
            names = names[names.index(args.from_step) :]
    skip_map = {
        "setup": args.skip_setup,
        "verify": args.skip_verify,
        "smoke": args.skip_smoke,
        "pretrained": args.skip_pretrained,
        "datasets": args.skip_datasets,
        "preprocess": args.skip_preprocess,
        "train": args.skip_train,
        "sample": args.skip_sample or not args.with_sample,
    }
    return [(n, skip_map[n]) for n in names]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run the full SDX pipeline in order (setup → smoke → download → train).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--from", dest="from_step", choices=STEP_ORDER, help="Start at this step.")
    p.add_argument("--only", choices=STEP_ORDER, help="Run a single step only.")
    p.add_argument("--smoke-only", action="store_true", help="Shortcut for --only smoke.")
    p.add_argument("--with-sample", action="store_true", help="Run sample.sh after training.")
    p.add_argument("--dry-run", action="store_true", help="Print planned steps and exit.")

    p.add_argument("--skip-setup", action="store_true")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--skip-pretrained", action="store_true")
    p.add_argument("--skip-datasets", action="store_true")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-sample", action="store_true")

    p.add_argument("--skip-smoke-train", action="store_true", help="Pass --skip-train to integration_smoke.")
    p.add_argument("--skip-smoke-scrape", action="store_true", help="Pass --skip-scrape to integration_smoke.")
    p.add_argument("--skip-smoke-reverse", action="store_true", help="Pass --skip-reverse to integration_smoke.")
    p.add_argument(
        "--smoke-data-root", default=None, help="Integration smoke data dir (default: SDX_DATA/integration_smoke)."
    )
    p.add_argument("--smoke-timeout", type=int, default=3600)
    p.add_argument(
        "--smoke-arg",
        action="append",
        default=[],
        dest="smoke_extra",
        metavar="ARG",
        help="Extra arg forwarded to integration_smoke.py (repeatable).",
    )

    p.add_argument("--scrape-workers", type=int, default=int(os.environ.get("SDX_SCRAPE_WORKERS", "20")))
    p.add_argument("--max-posts", type=int, default=int(os.environ.get("SDX_MAX_POSTS", "0")))
    p.add_argument("--frame-fps", type=float, default=float(os.environ.get("SDX_FRAME_FPS", "2")))
    p.add_argument("--max-frames-per-post", type=int, default=int(os.environ.get("SDX_MAX_FRAMES_PER_POST", "0")))
    p.add_argument(
        "--split-frames", action=argparse.BooleanOptionalAction, default=os.environ.get("SDX_SPLIT_FRAMES", "1") == "1"
    )
    p.add_argument("--keep-raw-media", action="store_true", default=os.environ.get("SDX_KEEP_RAW_MEDIA", "0") == "1")
    p.add_argument("--scrape-sites", nargs="*", default=None, choices=["danbooru", "e621", "rule34xxx", "rule34xyz"])
    p.add_argument("--dataset-timeout", type=int, default=0, help="0 = no timeout (full crawl).")
    p.add_argument("--pretrained-workers", type=int, default=int(os.environ.get("SDX_DL_WORKERS", "16")))
    p.add_argument("--pretrained-timeout", type=int, default=0, help="0 = no timeout.")
    p.add_argument("--enrich-workers", type=int, default=int(os.environ.get("SDX_ENRICH_WORKERS", "8")))
    p.add_argument("--enrich-timeout", type=int, default=7200)
    p.add_argument("--control-workers", type=int, default=int(os.environ.get("SDX_CONTROL_WORKERS", "12")))
    p.add_argument("--control-timeout", type=int, default=7200)
    p.add_argument("--train-timeout", type=int, default=0, help="0 = no timeout.")

    args = p.parse_args(argv)

    _load_env_defaults(ROOT)
    os.environ.setdefault("SDX_ROOT", str(ROOT))
    paths = _paths()

    planned = _resolve_steps(args)
    print("SDX pipeline", flush=True)
    print(f"  root={ROOT}", flush=True)
    print(f"  data={paths['data']}", flush=True)
    print(f"  secrets={paths['secrets']}", flush=True)
    print(f"  steps={[n for n, _ in planned]}", flush=True)

    if args.dry_run:
        for name, skip in planned:
            print(f"  {'SKIP' if skip else 'RUN'} {name}")
        return 0

    report = PipelineReport()
    handlers: dict[str, Callable[[], str | None]] = {
        "setup": lambda: (_step_setup(), "")[1],
        "verify": lambda: (_step_verify(), "")[1],
        "smoke": lambda: (_step_smoke(args, paths), str(paths["smoke_data"]))[1],
        "pretrained": lambda: (_step_pretrained(args, paths), str(paths["pretrained"]))[1],
        "datasets": lambda: (_step_datasets(args, paths), str(paths["combined"]))[1],
        "preprocess": lambda: (_step_preprocess(args, paths), str(paths["enriched"]))[1],
        "train": lambda: (_step_train(args, paths), os.environ.get("SDX_TRAIN_MODE", "full"))[1],
        "sample": lambda: (_step_sample(args, paths), "sample done")[1],
    }

    for name, skip in planned:
        if not _run_step(report, name, handlers[name], skip=skip):
            break

    print()
    print("=" * 60)
    print(report.summary())
    report_path = paths["data"] / "pipeline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {"ok": report.ok(), "summary": report.summary(), "steps": [asdict(s) for s in report.steps]}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    if report.ok():
        print()
        print("Done. Next:")
        print(f"  python sample.py --ckpt {paths['results'] / 'best.pt'} --prompt '@wlop 1girl'")
        if paths["rag"].is_file():
            print(f"    --local-rag-jsonl {paths['rag']}")
    return 0 if report.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
