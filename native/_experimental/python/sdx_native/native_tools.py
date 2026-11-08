"""
Optional ``native/`` helpers: discover built CLIs and shared libraries, run the Rust JSONL / noise-schedule
tools, FNV fingerprints, latent geometry (C++ ``sdx_cpu`` or pure Python), JSONL merge, and status.

Nothing here is required for ``train.py`` / ``sample.py``; every native path has a pure-Python fallback.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def _find_repo_root() -> Path:
    """Nearest ancestor holding ``pyproject.toml``; a fixed ``parents[N]`` would break if this package moves."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return p.parents[3]


REPO_ROOT = _find_repo_root()
NATIVE = REPO_ROOT / "native"
_RUST_TARGET = NATIVE / "rust" / "target"

# ctypes libraries loaded by ``sdx_native.ffi``: ``native/cpp`` targets and the ``native/rust`` cdylib.
SHARED_LIBRARIES: tuple[str, ...] = ("sdx_cpu", "sdx_cuda", "sdx_rs")


def _exe(base: Path, name: str) -> Path | None:
    for n in (f"{name}.exe", name):
        if (base / n).is_file():
            return base / n
    return None


def rust_jsonl_tools_exe() -> Path | None:
    """``sdx-jsonl-tools`` release binary if ``cargo build --release`` was run in ``native/rust``."""
    return _exe(_RUST_TARGET / "release", "sdx-jsonl-tools")


def rust_noise_schedule_exe() -> Path | None:
    """``sdx-noise-schedule`` release binary if ``cargo build --release`` was run in ``native/rust``."""
    return _exe(_RUST_TARGET / "release", "sdx-noise-schedule")


def zig_linecrc_exe() -> Path | None:
    """Zig ``sdx-linecrc`` if ``zig build`` was run."""
    return _exe(NATIVE / "_experimental/zig/sdx-linecrc/zig-out/bin", "sdx-linecrc")


def zig_pathstat_exe() -> Path | None:
    """Zig ``sdx-pathstat`` (file sizes for a path list, one per line)."""
    return _exe(NATIVE / "_experimental/zig/sdx-pathstat/zig-out/bin", "sdx-pathstat")


def go_sdx_manifest_exe() -> Path | None:
    """Go merge / explore helper if built next to its source."""
    return _exe(NATIVE / "go/sdx-manifest", "sdx-manifest")


def shared_library_path(name: str) -> Path | None:
    """First non-empty build of ``name`` under ``native/cpp/build`` (any generator) or ``native/rust/target``."""
    cpp = NATIVE / "cpp" / "build"
    dirs = (cpp, cpp / "Release", cpp / "Debug", _RUST_TARGET / "release", _RUST_TARGET / "debug")
    files = (f"{name}.dll", f"lib{name}.dll", f"lib{name}.so", f"lib{name}.dylib")
    for d in dirs:
        for f in files:
            p = d / f
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


_RUST_MISSING = "Rust {} not built (cargo build --release in native/rust)"


def _run(exe: Path | None, name: str, args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    if exe is None:
        raise FileNotFoundError(_RUST_MISSING.format(name))
    return subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=timeout)


def _jsonl_tools(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return _run(rust_jsonl_tools_exe(), "sdx-jsonl-tools", args, timeout)


def run_rust_noise_schedule(args: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    """Run ``sdx-noise-schedule`` (e.g. ``["linear", "--steps", "1000"]``)."""
    return _run(rust_noise_schedule_exe(), "sdx-noise-schedule", args, timeout)


def run_rust_jsonl_stats(manifest: Path, *, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    return _jsonl_tools(["stats", str(manifest)], timeout)


def run_rust_image_paths(
    manifest: Path,
    *,
    all_rows: bool = False,
    sort: bool = False,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    """``sdx-jsonl-tools image-paths``: stdout is one path per line."""
    args = ["image-paths", str(manifest)] + ["--all-rows"] * all_rows + ["--sort"] * sort
    return _jsonl_tools(args, timeout)


def run_rust_dup_image_paths(
    manifest: Path, *, top: int = 20, timeout: float = 600
) -> subprocess.CompletedProcess[str]:
    """``sdx-jsonl-tools dup-image-paths``: duplicate path report."""
    return _jsonl_tools(["dup-image-paths", str(manifest), "--top", str(top)], timeout)


def run_rust_jsonl_validate(
    manifest: Path,
    *,
    min_caption_len: int = 0,
    max_caption_len: int = 0,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    args = ["validate", str(manifest), "--min-caption-len", str(min_caption_len)]
    return _jsonl_tools(args + ["--max-caption-len", str(max_caption_len)], timeout)


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


# FNV-1a 64 over raw file bytes, as in ``sdx-linecrc --file`` and ``sdx-jsonl-tools file-fnv``. The offset
# basis is the standard 14695981039346656037 with its last two digits missing; it must stay identical to the
# native tools' so all three produce the same fingerprints.
_FNV_OFFSET = 146959810393466560
_FNV_PRIME = 1099511628211


def fnv1a64_file(path: Path, chunk: int = 1 << 20) -> tuple[int, int, int]:
    """``(hash_u64, line_count, byte_count)`` of the file bytes (pure Python; one step per byte)."""
    h, prime, mask = _FNV_OFFSET, _FNV_PRIME, 0xFFFFFFFFFFFFFFFF
    line_count = byte_count = 0
    with path.open("rb") as f:
        while buf := f.read(chunk):
            byte_count += len(buf)
            line_count += buf.count(b"\n")
            for b in buf:
                h = ((h ^ b) * prime) & mask
    return h, line_count, byte_count


def _native_fingerprint(path: Path) -> str | None:
    zig = zig_linecrc_exe()
    rust = rust_jsonl_tools_exe()
    try:
        if zig:
            r = subprocess.run([str(zig), "--file", str(path)], capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        if rust:
            r = subprocess.run([str(rust), "file-fnv", str(path)], capture_output=True, text=True, timeout=600)
            kv = dict(line.split(": ", 1) for line in r.stdout.splitlines() if ": " in line)
            if r.returncode == 0 and {"fnv1a64", "newlines", "bytes"} <= kv.keys():
                return f"fnv1a64={kv['fnv1a64']} lines={kv['newlines']} bytes={kv['bytes']}"
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def manifest_fingerprint_line(path: Path) -> str:
    """``fnv1a64=<hex> lines=<n> bytes=<n>``: Zig or Rust tool when built (much faster), else pure Python."""
    native = _native_fingerprint(path)
    if native:
        return native
    h, lines, nbytes = fnv1a64_file(path)
    return f"fnv1a64={h:x} lines={lines} bytes={nbytes}"


# Latent / DiT patch geometry; same results as the ``sdx_cpu`` C ABI (0 for invalid or non-divisible input).


def latent_spatial_size(image_hw: int, vae_scale: int) -> int:
    """Latent side length ``image_hw // vae_scale``."""
    if image_hw <= 0 or vae_scale <= 0 or image_hw % vae_scale != 0:
        return 0
    return image_hw // vae_scale


def patch_grid_dim(latent_hw: int, patch_size: int) -> int:
    """Patches per side ``latent_hw // patch_size``."""
    if latent_hw <= 0 or patch_size <= 0 or latent_hw % patch_size != 0:
        return 0
    return latent_hw // patch_size


def num_patch_tokens(image_hw: int, vae_scale: int, patch_size: int) -> int:
    """DiT patch token count (square grid, no register tokens): ``(latent/patch)^2``."""
    g = patch_grid_dim(latent_spatial_size(image_hw, vae_scale), patch_size)
    return g * g


def latent_numel(channels: int, latent_h: int, latent_w: int) -> int:
    """``C * H * W`` for a latent tensor (0 if any dimension is invalid or the product overflows int32)."""
    if channels <= 0 or latent_h <= 0 or latent_w <= 0:
        return 0
    n = channels * latent_h * latent_w
    return n if n <= 2147483647 else 0


def dit_patch_size_from_variant_name(model_name: str) -> int:
    """Patch size from a registry name like ``DiT-XL/2-Text`` or ``DiT-B/4-Text``; 2 when there is no ``/N``."""
    m = re.search(r"/(\d+)(?:-|/|$)", model_name)
    return max(1, int(m.group(1))) if m else 2


class LatentLib:
    """Latent geometry through ``sdx_cpu`` when built (``available``), else the pure-Python functions above."""

    def __init__(self) -> None:
        self._dll: Any = load_shared_library("sdx_cpu")

    @property
    def available(self) -> bool:
        return self._dll is not None

    def _call(self, c_name: str, py_fn: Any, *args: int) -> int:
        if self._dll is None:
            return py_fn(*args)
        return int(getattr(self._dll, c_name)(*(ctypes.c_int(int(a)) for a in args)))

    def latent_spatial_size(self, image_hw: int, vae_scale: int) -> int:
        return self._call("sdx_latent_spatial_size", latent_spatial_size, image_hw, vae_scale)

    def patch_grid_dim(self, latent_hw: int, patch_size: int) -> int:
        return self._call("sdx_patch_grid_dim", patch_grid_dim, latent_hw, patch_size)

    def num_patch_tokens(self, image_hw: int, vae_scale: int, patch_size: int) -> int:
        return self._call("sdx_num_patch_tokens", num_patch_tokens, image_hw, vae_scale, patch_size)

    def latent_numel(self, channels: int, latent_h: int, latent_w: int) -> int:
        return self._call("sdx_latent_numel", latent_numel, channels, latent_h, latent_w)


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
    """Merge JSONL files; first row wins per ``dedupe_key``. Prefers the Go binary if built."""
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
    """Which native tools and libraries are built (e.g. ``quick_test --show-native``)."""
    import importlib.util

    status: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "rust_sdx_jsonl_tools": str(rust_jsonl_tools_exe() or ""),
        "rust_sdx_noise_schedule": str(rust_noise_schedule_exe() or ""),
        "zig_sdx_linecrc": str(zig_linecrc_exe() or ""),
        "zig_sdx_pathstat": str(zig_pathstat_exe() or ""),
        "go_sdx_manifest": str(go_sdx_manifest_exe() or ""),
        "xxhash_installed": importlib.util.find_spec("xxhash") is not None,
    }
    for name in SHARED_LIBRARIES:
        status[f"lib{name}"] = str(shared_library_path(name) or "")
    return status
