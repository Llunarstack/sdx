# Build SDX native C++ libraries (optional CUDA). Run from repo root or any cwd.
#   .\scripts\tools\native\build_native.ps1
# Env:
#   $env:SDX_BUILD_CUDA = "0"  to force CPU-only (skip nvcc target)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$Cpp = Join-Path $RepoRoot "native\cpp"
$BuildDir = Join-Path $Cpp "build"

$UseCuda = "ON"
if ($env:SDX_BUILD_CUDA -eq "0") { $UseCuda = "OFF" }

Write-Host "==> CMake configure: $Cpp (SDX_BUILD_CUDA=$UseCuda)"
& cmake -S $Cpp -B $BuildDir -DCMAKE_BUILD_TYPE=Release -DSDX_BUILD_CUDA=$UseCuda
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
} else {
    Write-Host "(skip) cargo not on PATH - Rust tools optional"
}

Write-Host "Done."
