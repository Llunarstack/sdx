"""
Optional ``native/`` helpers: discover built CLIs and shared libraries, run JSONL tools,
FNV fingerprints, ctypes access to ``libsdx_latent``, and pure-Python fallbacks.

Nothing here is required for ``train.py`` / ``sample.py``; failures degrade gracefully.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from sdx_native.latent_geometry import latent_numel as py_latent_numel
from sdx_native.latent_geometry import latent_spatial_size as py_latent_spatial_size
from sdx_native.latent_geometry import num_patch_tokens as py_num_patch_tokens
from sdx_native.latent_geometry import patch_grid_dim as py_patch_grid_dim


def _find_repo_root() -> Path:
    """Nearest ancestor holding ``pyproject.toml``; a fixed ``parents[N]`` would break if this package moves."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[3]


REPO_ROOT = _find_repo_root()

# Every ctypes library a wrapper in this package can load (``native/cpp`` targets and Rust cdylibs).
SHARED_LIBRARIES: tuple[str, ...] = (
    "sdx_latent",
    "sdx_inference_timesteps",
    "sdx_beta_schedules",
    "sdx_image_metrics",
    "sdx_mask_ops",
    "sdx_cuda_image_metrics",
    "sdx_cuda_gaussian_blur",
    "sdx_cuda_ml",
    "sdx_diffusion_math",
    "sdx_prompt_ops",
    "sdx_quality_scorers",
    "sdx_image_ops",
)


def _exe(rel_dir: str, name: str) -> Path | None:
    base = REPO_ROOT / "native" / rel_dir
    for n in (f"{name}.exe", name):
        p = base / n
        if p.is_file():
            return p
    return None


def rust_jsonl_tools_exe() -> Path | None:
    """``sdx-jsonl-tools`` release binary if ``cargo build --release`` was run."""
    return _exe("rust/sdx-jsonl-tools/target/release", "sdx-jsonl-tools")


def rust_noise_schedule_exe() -> Path | None:
    """``sdx-noise-schedule`` release binary if ``cargo build --release`` was run."""
    return _exe("rust/sdx-noise-schedule/target/release", "sdx-noise-schedule")


def zig_linecrc_exe() -> Path | None:
    """Zig ``sdx-linecrc`` if ``zig build`` was run."""
    return _exe("_experimental/zig/sdx-linecrc/zig-out/bin", "sdx-linecrc")


def zig_pathstat_exe() -> Path | None:
    """Zig ``sdx-pathstat`` — file sizes for a path list (one per line)."""
    return _exe("_experimental/zig/sdx-pathstat/zig-out/bin", "sdx-pathstat")


def go_sdx_manifest_exe() -> Path | None:
    """Go merge helper if built next to source."""
    return _exe("go/sdx-manifest", "sdx-manifest")


def mojo_cli_path() -> str:
    """Modular ``mojo`` or ``magic`` CLI if on PATH."""
    return (shutil.which("mojo") or shutil.which("magic") or "").strip()


def shared_library_path(name: str) -> Path | None:
    """First built artifact for ``name``: ``native/cpp/build`` target, else Rust crate ``name`` with ``-``."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    rust = REPO_ROOT / "native" / "rust" / name.replace("_", "-") / "target"
    files = (f"{name}.dll", f"lib{name}.so", f"lib{name}.dylib")
    candidates = [
        cpp / "Release" / files[0],
        cpp / "Debug" / files[0],
        cpp / files[1],
        cpp / files[2],
        *(rust / profile / f for profile in ("release", "debug") for f in files),
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


@functools.cache
def load_shared_library(name: str) -> ctypes.CDLL | None:
    """``ctypes.CDLL`` for :func:`shared_library_path`, or ``None`` when not built / not loadable."""
    p = shared_library_path(name)
    if p is None:
        return None
    try:
        return ctypes.CDLL(str(p))
    except OSError:
        return None


def _run(exe: Path | None, missing: str, args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    if exe is None:
        raise FileNotFoundError(missing)
    return subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=timeout)


_JSONL_TOOLS_MISSING = "Rust sdx-jsonl-tools not built (cargo build --release in native/rust/sdx-jsonl-tools)"


def run_rust_noise_schedule(args: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    """Run ``sdx-noise-schedule`` (e.g. ``[\"linear\", \"--steps\", \"1000\"]``)."""
    missing = "Rust sdx-noise-schedule not built (cargo build --release in native/rust/sdx-noise-schedule)"
    return _run(rust_noise_schedule_exe(), missing, args, timeout)


def run_rust_jsonl_stats(manifest: Path, *, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    return _run(rust_jsonl_tools_exe(), _JSONL_TOOLS_MISSING, ["stats", str(manifest)], timeout)


def run_rust_image_paths(
    manifest: Path,
    *,
    all_rows: bool = False,
    sort: bool = False,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    """Run `sdx-jsonl-tools image-paths` — stdout is one path per line."""
    args = ["image-paths", str(manifest)]
    if all_rows:
        args.append("--all-rows")
    if sort:
        args.append("--sort")
    return _run(rust_jsonl_tools_exe(), _JSONL_TOOLS_MISSING, args, timeout)


def run_rust_dup_image_paths(
    manifest: Path, *, top: int = 20, timeout: float = 600
) -> subprocess.CompletedProcess[str]:
    """Run `sdx-jsonl-tools dup-image-paths` — duplicate path report."""
    args = ["dup-image-paths", str(manifest), "--top", str(top)]
    return _run(rust_jsonl_tools_exe(), _JSONL_TOOLS_MISSING, args, timeout)


def run_rust_jsonl_validate(
    manifest: Path,
    *,
    min_caption_len: int = 0,
    max_caption_len: int = 0,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    args = [
        "validate",
        str(manifest),
        "--min-caption-len",
        str(min_caption_len),
        "--max-caption-len",
        str(max_caption_len),
    ]
    return _run(rust_jsonl_tools_exe(), _JSONL_TOOLS_MISSING, args, timeout)


def file_md5_hex(path: Path, *, prefer_native_md5: bool = True) -> str:
    """
    MD5 hex digest of file bytes (streaming). Prefers ``sdx-jsonl-tools file-md5`` when built
    (large-image dedup without reading the file into Python); otherwise ``hashlib`` in 1 MiB chunks.
    Returns an empty string on failure.
    """
    exe = rust_jsonl_tools_exe() if prefer_native_md5 else None
    if exe is not None and path.is_file():
        try:
            r = subprocess.run([str(exe), "file-md5", str(path)], capture_output=True, text=True, timeout=3600)
            hx = (r.stdout or "").strip().lower()
            if r.returncode == 0 and len(hx) == 32 and all(c in "0123456789abcdef" for c in hx):
                return hx
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        h = hashlib.md5()
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


# FNV-1a 64 as in ``native/_experimental/zig/sdx-linecrc`` file mode. The offset basis is the standard
# 14695981039346656037 with its last two digits missing; it must stay identical to the Zig tool's so both
# produce the same fingerprints.
_FNV_OFFSET = 146959810393466560
_FNV_PRIME = 1099511628211


def fnv1a64_file(path: Path, chunk: int = 65536) -> tuple[int, int, int]:
    """
    Same fingerprint as ``sdx-linecrc --file`` (streaming over file bytes).

    Returns ``(hash_u64, line_count, byte_count)``.
    """
    h = _FNV_OFFSET
    line_count = 0
    byte_count = 0
    with path.open("rb") as f:
        while buf := f.read(chunk):
            byte_count += len(buf)
            line_count += buf.count(b"\n")
            for b in buf:
                h ^= b
                h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h, line_count, byte_count


def manifest_fingerprint_line(path: Path) -> str:
    """Human-readable fingerprint; uses Zig if built, else Python FNV (same **file** mode)."""
    zig = zig_linecrc_exe()
    if zig:
        try:
            r = subprocess.run([str(zig), "--file", str(path)], capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
    h, lines, nbytes = fnv1a64_file(path)
    return f"fnv1a64={h:x} lines={lines} bytes={nbytes}"


class LatentLib:
    """ctypes wrapper for ``libsdx_latent`` with Python fallbacks."""

    def __init__(self) -> None:
        dll = load_shared_library("sdx_latent")
        if dll is not None:
            for fn, nargs in (
                ("sdx_latent_spatial_size", 2),
                ("sdx_patch_grid_dim", 2),
                ("sdx_num_patch_tokens", 3),
                ("sdx_latent_numel", 3),
            ):
                # Older builds lack sdx_latent_numel.
                if hasattr(dll, fn):
                    getattr(dll, fn).argtypes = (ctypes.c_int,) * nargs
                    getattr(dll, fn).restype = ctypes.c_int
        self._dll: Any = dll

    @property
    def available(self) -> bool:
        return self._dll is not None

    def num_patch_tokens(self, image_hw: int, vae_scale: int, patch_size: int) -> int:
        if self._dll is not None:
            return int(self._dll.sdx_num_patch_tokens(int(image_hw), int(vae_scale), int(patch_size)))
        return py_num_patch_tokens(image_hw, vae_scale, patch_size)

    def latent_spatial_size(self, image_hw: int, vae_scale: int) -> int:
        if self._dll is not None:
            return int(self._dll.sdx_latent_spatial_size(int(image_hw), int(vae_scale)))
        return py_latent_spatial_size(image_hw, vae_scale)

    def patch_grid_dim(self, latent_hw: int, patch_size: int) -> int:
        if self._dll is not None:
            return int(self._dll.sdx_patch_grid_dim(int(latent_hw), int(patch_size)))
        return py_patch_grid_dim(latent_hw, patch_size)

    def latent_numel(self, channels: int, latent_h: int, latent_w: int) -> int:
        if self._dll is not None and hasattr(self._dll, "sdx_latent_numel"):
            return int(self._dll.sdx_latent_numel(int(channels), int(latent_h), int(latent_w)))
        return py_latent_numel(channels, latent_h, latent_w)


@functools.cache
def get_latent_lib() -> LatentLib:
    return LatentLib()


def merge_jsonl_files(
    inputs: list[Path],
    output: Path,
    *,
    dedupe_key: str = "image_path",
    prefer_go: bool = True,
) -> None:
    """Merge JSONL files; first row wins per ``dedupe_key``. Prefers Go binary if present."""
    go = go_sdx_manifest_exe() if prefer_go else None
    if go is not None:
        cmd = [str(go), "merge", "-o", str(output), "--dedupe-key", dedupe_key, *(str(p) for p in inputs)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout or "go merge failed")
        return

    seen: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fout:
        for inp in inputs:
            with inp.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    key = next((str(obj[k]) for k in (dedupe_key, "path", "image", "image_path") if k in obj), None)
                    if key is None or key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def native_stack_status() -> dict[str, Any]:
    """Summary for diagnostics (e.g. ``quick_test --show-native``)."""
    import importlib.util

    status: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "rust_sdx_jsonl_tools": str(rust_jsonl_tools_exe() or ""),
        "rust_sdx_noise_schedule": str(rust_noise_schedule_exe() or ""),
        "zig_sdx_linecrc": str(zig_linecrc_exe() or ""),
        "zig_sdx_pathstat": str(zig_pathstat_exe() or ""),
        "go_sdx_manifest": str(go_sdx_manifest_exe() or ""),
        "mojo_or_magic_cli": mojo_cli_path(),
        "xxhash_installed": importlib.util.find_spec("xxhash") is not None,
    }
    for name in SHARED_LIBRARIES:
        status[f"lib{name}"] = str(shared_library_path(name) or "")
    return status
