"""
Optional ``native/`` helpers: discover built CLIs, run JSONL tools, FNV fingerprints,
ctypes access to ``libsdx_latent``, and pure-Python fallbacks (including JSONL stat/prompt-lint).

Nothing here is required for ``train.py`` / ``sample.py``; failures degrade gracefully.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sdx_native.latent_geometry import latent_numel as py_latent_numel
from sdx_native.latent_geometry import latent_spatial_size as py_latent_spatial_size
from sdx_native.latent_geometry import num_patch_tokens as py_num_patch_tokens
from sdx_native.latent_geometry import patch_grid_dim as py_patch_grid_dim

# Repo root: native/python/sdx_native/native_tools.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _release_dir(name: str) -> Path:
    return REPO_ROOT / "native" / name


def rust_jsonl_tools_exe() -> Optional[Path]:
    """``sdx-jsonl-tools`` release binary if ``cargo build --release`` was run."""
    base = _release_dir("rust/sdx-jsonl-tools/target/release")
    for n in ("sdx-jsonl-tools.exe", "sdx-jsonl-tools"):
        p = base / n
        if p.is_file():
            return p
    return None


def rust_noise_schedule_exe() -> Optional[Path]:
    """``sdx-noise-schedule`` release binary if ``cargo build --release`` was run."""
    base = _release_dir("rust/sdx-noise-schedule/target/release")
    for n in ("sdx-noise-schedule.exe", "sdx-noise-schedule"):
        p = base / n
        if p.is_file():
            return p
    return None


def run_rust_noise_schedule(args: List[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    """Run ``sdx-noise-schedule`` (e.g. ``[\"linear\", \"--steps\", \"1000\"]``)."""
    exe = rust_noise_schedule_exe()
    if not exe:
        raise FileNotFoundError("Rust sdx-noise-schedule not built (cargo build --release in native/rust/sdx-noise-schedule)")
    return subprocess.run(
        [str(exe), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def zig_linecrc_exe() -> Optional[Path]:
    """Zig ``sdx-linecrc`` if ``zig build`` was run."""
    base = _release_dir("zig/sdx-linecrc/zig-out/bin")
    for n in ("sdx-linecrc.exe", "sdx-linecrc"):
        p = base / n
        if p.is_file():
            return p
    return None


def zig_pathstat_exe() -> Optional[Path]:
    """Zig ``sdx-pathstat`` — file sizes for a path list (one per line)."""
    base = _release_dir("zig/sdx-pathstat/zig-out/bin")
    for n in ("sdx-pathstat.exe", "sdx-pathstat"):
        p = base / n
        if p.is_file():
            return p
    return None


def go_sdx_manifest_exe() -> Optional[Path]:
    """Go merge helper if built next to source."""
    d = _release_dir("go/sdx-manifest")
    for n in ("sdx-manifest.exe", "sdx-manifest"):
        p = d / n
        if p.is_file():
            return p
    return None


def latent_shared_library_path() -> Optional[Path]:
    """First matching ``libsdx_latent`` build artifact."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_latent.dll",
        cpp / "Debug" / "sdx_latent.dll",
        cpp / "libsdx_latent.so",
        cpp / "libsdx_latent.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def inference_timesteps_shared_library_path() -> Optional[Path]:
    """First matching ``sdx_inference_timesteps`` build artifact (inference path finalization)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_inference_timesteps.dll",
        cpp / "Debug" / "sdx_inference_timesteps.dll",
        cpp / "libsdx_inference_timesteps.so",
        cpp / "libsdx_inference_timesteps.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def beta_schedules_shared_library_path() -> Optional[Path]:
    """First matching ``sdx_beta_schedules`` build artifact (squared-cosine betas)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_beta_schedules.dll",
        cpp / "Debug" / "sdx_beta_schedules.dll",
        cpp / "libsdx_beta_schedules.so",
        cpp / "libsdx_beta_schedules.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def line_stats_shared_library_path() -> Optional[Path]:
    """``sdx_line_stats`` — fast manifest byte + newline count."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_line_stats.dll",
        cpp / "Debug" / "sdx_line_stats.dll",
        cpp / "libsdx_line_stats.so",
        cpp / "libsdx_line_stats.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def cuda_hwc_to_chw_shared_library_path() -> Optional[Path]:
    """Optional CUDA ``sdx_cuda_hwc_to_chw`` (``-DSDX_BUILD_CUDA=ON``)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_cuda_hwc_to_chw.dll",
        cpp / "Debug" / "sdx_cuda_hwc_to_chw.dll",
        cpp / "libsdx_cuda_hwc_to_chw.so",
        cpp / "libsdx_cuda_hwc_to_chw.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def cuda_ml_shared_library_path() -> Optional[Path]:
    """Optional CUDA ``sdx_cuda_ml`` (L2 row normalize; ``-DSDX_BUILD_CUDA=ON``)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_cuda_ml.dll",
        cpp / "Debug" / "sdx_cuda_ml.dll",
        cpp / "libsdx_cuda_ml.so",
        cpp / "libsdx_cuda_ml.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def cuda_flow_matching_shared_library_path() -> Optional[Path]:
    """Optional CUDA ``sdx_cuda_flow_matching`` (velocity residual ``eps - x0``; ``-DSDX_BUILD_CUDA=ON``)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_cuda_flow_matching.dll",
        cpp / "Debug" / "sdx_cuda_flow_matching.dll",
        cpp / "libsdx_cuda_flow_matching.so",
        cpp / "libsdx_cuda_flow_matching.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def cuda_nf4_shared_library_path() -> Optional[Path]:
    """Optional CUDA ``sdx_cuda_nf4`` (NF4 dequant; ``-DSDX_BUILD_CUDA=ON``)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_cuda_nf4.dll",
        cpp / "Debug" / "sdx_cuda_nf4.dll",
        cpp / "libsdx_cuda_nf4.so",
        cpp / "libsdx_cuda_nf4.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def cuda_sdpa_online_shared_library_path() -> Optional[Path]:
    """Optional CUDA ``sdx_cuda_sdpa_online`` (online-softmax SDPA, head_dim=64; ``-DSDX_BUILD_CUDA=ON``)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_cuda_sdpa_online.dll",
        cpp / "Debug" / "sdx_cuda_sdpa_online.dll",
        cpp / "libsdx_cuda_sdpa_online.so",
        cpp / "libsdx_cuda_sdpa_online.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def cuda_rmsnorm_shared_library_path() -> Optional[Path]:
    """Optional CUDA ``sdx_cuda_rmsnorm`` (row-wise RMSNorm; ``-DSDX_BUILD_CUDA=ON``)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_cuda_rmsnorm.dll",
        cpp / "Debug" / "sdx_cuda_rmsnorm.dll",
        cpp / "libsdx_cuda_rmsnorm.so",
        cpp / "libsdx_cuda_rmsnorm.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def cuda_rope_shared_library_path() -> Optional[Path]:
    """Optional CUDA ``sdx_cuda_rope`` (interleaved RoPE on host buffers)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_cuda_rope.dll",
        cpp / "Debug" / "sdx_cuda_rope.dll",
        cpp / "libsdx_cuda_rope.so",
        cpp / "libsdx_cuda_rope.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def cuda_silu_gate_shared_library_path() -> Optional[Path]:
    """Optional CUDA ``sdx_cuda_silu_gate`` (fused SiLU * gate)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_cuda_silu_gate.dll",
        cpp / "Debug" / "sdx_cuda_silu_gate.dll",
        cpp / "libsdx_cuda_silu_gate.so",
        cpp / "libsdx_cuda_silu_gate.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def cuda_gaussian_blur_shared_library_path() -> Optional[Path]:
    """Optional CUDA ``sdx_cuda_gaussian_blur`` (depthwise Gaussian blur on latents)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_cuda_gaussian_blur.dll",
        cpp / "Debug" / "sdx_cuda_gaussian_blur.dll",
        cpp / "libsdx_cuda_gaussian_blur.so",
        cpp / "libsdx_cuda_gaussian_blur.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def cuda_percentile_clamp_shared_library_path() -> Optional[Path]:
    """Optional CUDA ``sdx_cuda_percentile_clamp`` (per-sample percentile clamp)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_cuda_percentile_clamp.dll",
        cpp / "Debug" / "sdx_cuda_percentile_clamp.dll",
        cpp / "libsdx_cuda_percentile_clamp.so",
        cpp / "libsdx_cuda_percentile_clamp.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def mask_ops_shared_library_path() -> Optional[Path]:
    """CPU ``sdx_mask_ops`` (mask → patch weights for part-aware training)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_mask_ops.dll",
        cpp / "Debug" / "sdx_mask_ops.dll",
        cpp / "libsdx_mask_ops.so",
        cpp / "libsdx_mask_ops.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def rust_diffusion_math_shared_library_path() -> Optional[Path]:
    """Rust ``sdx_diffusion_math`` cdylib (alpha_cumprod, SNR, beta schedules)."""
    base = REPO_ROOT / "native" / "rust" / "sdx-diffusion-math" / "target"
    candidates = [
        base / "release" / "sdx_diffusion_math.dll",
        base / "release" / "libsdx_diffusion_math.so",
        base / "release" / "libsdx_diffusion_math.dylib",
        base / "debug" / "sdx_diffusion_math.dll",
        base / "debug" / "libsdx_diffusion_math.so",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def fnv64_file_shared_library_path() -> Optional[Path]:
    """``sdx_fnv64_file`` — streaming FNV-1a 64 + newlines (matches Python ``fnv1a64_file``)."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_fnv64_file.dll",
        cpp / "Debug" / "sdx_fnv64_file.dll",
        cpp / "libsdx_fnv64_file.so",
        cpp / "libsdx_fnv64_file.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def rmsnorm_rows_cpu_shared_library_path() -> Optional[Path]:
    """CPU ``sdx_rmsnorm_rows_cpu`` shared library path, if built."""
    cpp = REPO_ROOT / "native" / "cpp" / "build"
    candidates = [
        cpp / "Release" / "sdx_rmsnorm_rows_cpu.dll",
        cpp / "Debug" / "sdx_rmsnorm_rows_cpu.dll",
        cpp / "libsdx_rmsnorm_rows_cpu.so",
        cpp / "libsdx_rmsnorm_rows_cpu.dylib",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def mojo_cli_path() -> str:
    """Modular ``mojo`` or ``magic`` CLI if on PATH."""
    return (shutil.which("mojo") or shutil.which("magic") or "").strip()


def run_rust_jsonl_stats(manifest: Path, *, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    exe = rust_jsonl_tools_exe()
    if not exe:
        raise FileNotFoundError("Rust sdx-jsonl-tools not built (cargo build --release in native/rust/sdx-jsonl-tools)")
    return subprocess.run(
        [str(exe), "stats", str(manifest)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_rust_image_paths(
    manifest: Path,
    *,
    all_rows: bool = False,
    sort: bool = False,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    """Run `sdx-jsonl-tools image-paths` — stdout is one path per line."""
    exe = rust_jsonl_tools_exe()
    if not exe:
        raise FileNotFoundError("Rust sdx-jsonl-tools not built")
    cmd = [str(exe), "image-paths", str(manifest)]
    if all_rows:
        cmd.append("--all-rows")
    if sort:
        cmd.append("--sort")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def run_rust_dup_image_paths(
    manifest: Path, *, top: int = 20, timeout: float = 600
) -> subprocess.CompletedProcess[str]:
    """Run `sdx-jsonl-tools dup-image-paths` — duplicate path report."""
    exe = rust_jsonl_tools_exe()
    if not exe:
        raise FileNotFoundError("Rust sdx-jsonl-tools not built")
    return subprocess.run(
        [str(exe), "dup-image-paths", str(manifest), "--top", str(top)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_rust_file_fnv(path: Path, *, timeout: float = 3600) -> subprocess.CompletedProcess[str]:
    """Run ``sdx-jsonl-tools file-fnv`` — raw-byte FNV-1a 64 fingerprint."""
    exe = rust_jsonl_tools_exe()
    if not exe:
        raise FileNotFoundError("Rust sdx-jsonl-tools not built")
    return subprocess.run(
        [str(exe), "file-fnv", str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_rust_file_md5(path: Path, *, timeout: float = 3600) -> subprocess.CompletedProcess[str]:
    """Run ``sdx-jsonl-tools file-md5`` — streaming MD5 (matches ``hashlib.md5``)."""
    exe = rust_jsonl_tools_exe()
    if not exe:
        raise FileNotFoundError("Rust sdx-jsonl-tools not built")
    return subprocess.run(
        [str(exe), "file-md5", str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def maybe_rust_file_md5_hex(path: Path, *, timeout: float = 3600) -> Optional[str]:
    """
    Return 32-char lowercase hex MD5 if the Rust tool is built and the path is readable.

    Used for large-image dedup without reading the whole file into Python memory.
    """
    exe = rust_jsonl_tools_exe()
    if exe is None or not path.is_file():
        return None
    try:
        r = subprocess.run(
            [str(exe), "file-md5", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            return None
        hx = (r.stdout or "").strip().lower()
        if len(hx) == 32 and all(c in "0123456789abcdef" for c in hx):
            return hx
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def file_md5_hex(path: Path, *, prefer_native_md5: bool = True) -> str:
    """
    MD5 hex digest of file bytes (streaming). Prefers ``file-md5`` subprocess when the Rust
    tool is built; otherwise ``hashlib`` in 1 MiB chunks. Returns an empty string on failure.
    """
    if prefer_native_md5:
        hx = maybe_rust_file_md5_hex(path)
        if hx:
            return hx
    try:
        h = hashlib.md5()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def run_zig_pathstat_list(pathlist_file: Path, *, timeout: float = 3600) -> subprocess.CompletedProcess[str]:
    exe = zig_pathstat_exe()
    if not exe:
        raise FileNotFoundError("Zig sdx-pathstat not built")
    return subprocess.run(
        [str(exe), "--file", str(pathlist_file)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_rust_jsonl_validate(
    manifest: Path,
    *,
    min_caption_len: int = 0,
    max_caption_len: int = 0,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    exe = rust_jsonl_tools_exe()
    if not exe:
        raise FileNotFoundError("Rust sdx-jsonl-tools not built")
    return subprocess.run(
        [
            str(exe),
            "validate",
            str(manifest),
            "--min-caption-len",
            str(min_caption_len),
            "--max-caption-len",
            str(max_caption_len),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_zig_linecrc_file(manifest: Path, *, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    exe = zig_linecrc_exe()
    if not exe:
        raise FileNotFoundError("Zig sdx-linecrc not built")
    return subprocess.run(
        [str(exe), "--file", str(manifest)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# FNV-1a 64 — matches ``native/zig/sdx-linecrc`` **file** mode (raw bytes, chunk-wise).
_FNV_OFFSET = 146959810393466560
_FNV_PRIME = 1099511628211


def fnv1a64_bytes(data: bytes) -> int:
    h = _FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def fnv1a64_file(path: Path, chunk: int = 65536) -> Tuple[int, int, int]:
    """
    Same fingerprint as ``sdx-linecrc --file`` (streaming over file bytes).

    Returns ``(hash_u64, line_count, byte_count)``.
    """
    h = _FNV_OFFSET
    line_count = 0
    byte_count = 0
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            byte_count += len(buf)
            line_count += buf.count(b"\n")
            for b in buf:
                h ^= b
                h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h, line_count, byte_count


def manifest_fingerprint_line(path: Path) -> str:
    """
    Human-readable fingerprint; uses Zig if built, else Python FNV (same **file** mode).
    """
    zig = zig_linecrc_exe()
    if zig:
        try:
            r = run_zig_linecrc_file(path, timeout=600)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
    h, lines, nbytes = fnv1a64_file(path)
    return f"fnv1a64={h:x} lines={lines} bytes={nbytes}"


class LatentLib:
    """ctypes wrapper for ``libsdx_latent`` with Python fallbacks."""

    def __init__(self) -> None:
        self._dll: Any = None
        p = latent_shared_library_path()
        if p is None:
            return
        try:
            dll = ctypes.CDLL(str(p))
            dll.sdx_latent_spatial_size.argtypes = (ctypes.c_int, ctypes.c_int)
            dll.sdx_latent_spatial_size.restype = ctypes.c_int
            dll.sdx_patch_grid_dim.argtypes = (ctypes.c_int, ctypes.c_int)
            dll.sdx_patch_grid_dim.restype = ctypes.c_int
            dll.sdx_num_patch_tokens.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
            dll.sdx_num_patch_tokens.restype = ctypes.c_int
            dll.sdx_latent_hw.argtypes = (ctypes.c_int, ctypes.c_int)
            dll.sdx_latent_hw.restype = ctypes.c_int
            # Optional: older builds may lack sdx_latent_numel
            if hasattr(dll, "sdx_latent_numel"):
                dll.sdx_latent_numel.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
                dll.sdx_latent_numel.restype = ctypes.c_int
            self._dll = dll
        except OSError:
            self._dll = None

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


_LATENT_SINGLETON: Optional[LatentLib] = None


def get_latent_lib() -> LatentLib:
    global _LATENT_SINGLETON
    if _LATENT_SINGLETON is None:
        _LATENT_SINGLETON = LatentLib()
    return _LATENT_SINGLETON


def merge_jsonl_files(
    inputs: List[Path],
    output: Path,
    *,
    dedupe_key: str = "image_path",
    prefer_go: bool = True,
) -> None:
    """
    Merge JSONL files; first row wins per ``dedupe_key``. Prefers Go binary if present.
    """
    out = output
    go = go_sdx_manifest_exe() if prefer_go else None
    if go is not None:
        cmd = [str(go), "merge", "-o", str(out), "--dedupe-key", dedupe_key]
        cmd.extend(str(p) for p in inputs)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout or "go merge failed")
        return

    seen: set[str] = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fout:
        for inp in inputs:
            with inp.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = None
                    if dedupe_key in obj:
                        key = str(obj[dedupe_key])
                    else:
                        for alt in ("path", "image", "image_path"):
                            if alt in obj:
                                key = str(obj[alt])
                                break
                    if key is None:
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _xxhash_available() -> bool:
    try:
        import xxhash  # type: ignore[import-untyped]

        return True
    except ImportError:
        return False


def native_stack_status() -> Dict[str, Any]:
    """Summary for diagnostics (e.g. ``quick_test --show-native``)."""
    return {
        "repo_root": str(REPO_ROOT),
        "rust_sdx_jsonl_tools": str(rust_jsonl_tools_exe() or ""),
        "rust_sdx_noise_schedule": str(rust_noise_schedule_exe() or ""),
        "rust_file_md5_available": bool(rust_jsonl_tools_exe()),
        "zig_sdx_linecrc": str(zig_linecrc_exe() or ""),
        "zig_sdx_pathstat": str(zig_pathstat_exe() or ""),
        "go_sdx_manifest": str(go_sdx_manifest_exe() or ""),
        "jsonl_manifest_pure": "sdx_native.jsonl_manifest_pure (python -m sdx_native.jsonl_manifest_pure stat|promptlint)",
        "libsdx_latent": str(latent_shared_library_path() or ""),
        "libsdx_inference_timesteps": str(inference_timesteps_shared_library_path() or ""),
        "libsdx_beta_schedules": str(beta_schedules_shared_library_path() or ""),
        "libsdx_line_stats": str(line_stats_shared_library_path() or ""),
        "libsdx_fnv64_file": str(fnv64_file_shared_library_path() or ""),
        "libsdx_rmsnorm_rows_cpu": str(rmsnorm_rows_cpu_shared_library_path() or ""),
        "libsdx_cuda_hwc_to_chw": str(cuda_hwc_to_chw_shared_library_path() or ""),
        "libsdx_cuda_ml": str(cuda_ml_shared_library_path() or ""),
        "libsdx_cuda_flow_matching": str(cuda_flow_matching_shared_library_path() or ""),
        "libsdx_cuda_nf4": str(cuda_nf4_shared_library_path() or ""),
        "libsdx_cuda_sdpa_online": str(cuda_sdpa_online_shared_library_path() or ""),
        "libsdx_cuda_rmsnorm": str(cuda_rmsnorm_shared_library_path() or ""),
        "libsdx_cuda_rope": str(cuda_rope_shared_library_path() or ""),
        "libsdx_cuda_silu_gate": str(cuda_silu_gate_shared_library_path() or ""),
        "libsdx_cuda_gaussian_blur": str(cuda_gaussian_blur_shared_library_path() or ""),
        "libsdx_cuda_percentile_clamp": str(cuda_percentile_clamp_shared_library_path() or ""),
        "libsdx_mask_ops": str(mask_ops_shared_library_path() or ""),
        "libsdx_diffusion_math_rust": str(rust_diffusion_math_shared_library_path() or ""),
        "mojo_or_magic_cli": mojo_cli_path(),
        "latent_lib_ctypes": get_latent_lib().available,
        "caption_text_hygiene": True,
        "xxhash_installed": _xxhash_available(),
    }
