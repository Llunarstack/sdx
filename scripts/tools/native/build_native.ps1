# Build SDX native C++ libraries (optional CUDA). Run from repo root or any cwd.
#   .\scripts\tools\native\build_native.ps1
# Env:
#   $env:SDX_BUILD_CUDA = "0"  force CPU-only C++ (no nvcc targets)
#   $env:SDX_BUILD_CUDA = "1"  force CUDA targets (requires nvcc on PATH)
#   unset: auto - CUDA ON if nvcc exists, else OFF
#
# If the previous configure used a different SDX_BUILD_CUDA, this script removes
# native\cpp\build automatically so CMake does not reuse a stale cache.

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$Cpp = Join-Path $RepoRoot "native\cpp"
$BuildDir = Join-Path $Cpp "build"

$UseCuda = "OFF"
if ($env:SDX_BUILD_CUDA -eq "0") {
    $UseCuda = "OFF"
} elseif ($env:SDX_BUILD_CUDA -eq "1") {
    $UseCuda = "ON"
} elseif (Get-Command nvcc -ErrorAction SilentlyContinue) {
    $UseCuda = "ON"
} else {
    Write-Host "(info) nvcc not on PATH - C++ build without CUDA (sdx_cuda_*). Set SDX_BUILD_CUDA=1 and PATH to nvcc to enable."
}

$CacheFile = Join-Path $BuildDir "CMakeCache.txt"
$CachedCuda = $null
if (Test-Path $CacheFile) {
    $row = Select-String -Path $CacheFile -Pattern '^SDX_BUILD_CUDA:' | Select-Object -First 1
    if ($row) {
        $ln = $row.Line.TrimEnd()
        if ($ln -match '^SDX_BUILD_CUDA:\w+=(ON|OFF)\s*$') {
            $CachedCuda = $Matches[1]
        } else {
            Write-Host "(info) Clearing $BuildDir (invalid SDX_BUILD_CUDA in CMakeCache: $ln)"
            Remove-Item -Recurse -Force $BuildDir
        }
    }
}
if ($null -ne $CachedCuda -and $CachedCuda -ne $UseCuda) {
    Write-Host "(info) CMake cache had SDX_BUILD_CUDA=$CachedCuda but this run uses $UseCuda - removing $BuildDir"
    Remove-Item -Recurse -Force $BuildDir
}

Write-Host "==> CMake configure: $Cpp (SDX_BUILD_CUDA=$UseCuda)"
# Quote -D so PowerShell does not mangle values; avoids corrupt cache like SDX_BUILD_CUDA=$UseCuda literal.
& cmake -S $Cpp -B $BuildDir "-DCMAKE_BUILD_TYPE=Release" "-DSDX_BUILD_CUDA=$UseCuda"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> CMake build: Release"
& cmake --build $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "DLLs under: $BuildDir\Release (MSVC) or $BuildDir (Ninja)"

if (Get-Command cargo -ErrorAction SilentlyContinue) {
    Write-Host "==> cargo build: sdx-jsonl-tools"
    Push-Location (Join-Path $RepoRoot "native\rust\sdx-jsonl-tools")
    try {
        & cargo build --release
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        Pop-Location
    }
    Write-Host "==> cargo build: sdx-noise-schedule"
    Push-Location (Join-Path $RepoRoot "native\rust\sdx-noise-schedule")
    try {
        & cargo build --release
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "(skip) cargo not on PATH - Rust tools optional"
}

Write-Host "Done."
