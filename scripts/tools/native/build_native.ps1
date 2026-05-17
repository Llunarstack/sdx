# Build SDX native C++ libraries (optional CUDA). Run from repo root or any cwd.

#   .\scripts\tools\native\build_native.ps1
#   .\scripts\tools\native\clean_native_builds.ps1   # remove cpp/build* and stray Cargo dirs

#

# Env:

#   SDX_BUILD_CUDA = "0" | "1" | unset — unset auto-detects nvcc.

#   SDX_USE_NINJA = "0" — disable Ninja even if ninja.exe is on PATH.

#   SDX_CUDA_ARCHITECTURES — forwarded to CMake (e.g. "89", "native", "62;real").

#   SDX_C_COMPILER_LAUNCHER / SDX_CXX_COMPILER_LAUNCHER / SDX_CUDA_COMPILER_LAUNCHER —

#     override compiler cache (otherwise sccache, else ccache when found).

#   RUSTC_WRAPPER — if unset and sccache exists, cargo uses sccache for Rust.

#

# If the previous configure used a different SDX_BUILD_CUDA or CMake generator

# (Ninja vs Visual Studio), native\cpp\build is removed automatically.



$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")

$Cpp = Join-Path $RepoRoot "native\cpp"

$BuildDir = Join-Path $Cpp "build"



$OldRustWrapper = $env:RUSTC_WRAPPER



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



$UseNinja = $false

if ($env:SDX_USE_NINJA -ne "0") {

    if (Get-Command ninja -ErrorAction SilentlyContinue) {

        $UseNinja = $true

    }

}



$LaunchC = $env:SDX_C_COMPILER_LAUNCHER

$LaunchCXX = $env:SDX_CXX_COMPILER_LAUNCHER

$LaunchCUDA = $env:SDX_CUDA_COMPILER_LAUNCHER

if (-not $LaunchC -and -not $LaunchCXX) {

    if (Get-Command sccache -ErrorAction SilentlyContinue) {

        $LaunchC = "sccache"

        $LaunchCXX = "sccache"

        if ($UseCuda -eq "ON" -and -not $LaunchCUDA) {

            $LaunchCUDA = "sccache"

        }

    } elseif (Get-Command ccache -ErrorAction SilentlyContinue) {

        $LaunchC = "ccache"

        $LaunchCXX = "ccache"

        if ($UseCuda -eq "ON" -and -not $LaunchCUDA) {

            $LaunchCUDA = "ccache"

        }

    }

}



$CacheFile = Join-Path $BuildDir "CMakeCache.txt"

$CachedCuda = $null

$GeneratorStale = $false

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

    $genRow = Select-String -Path $CacheFile -Pattern '^CMAKE_GENERATOR:INTERNAL=' | Select-Object -First 1

    if ($genRow) {

        $gen = ($genRow.Line -replace '^CMAKE_GENERATOR:INTERNAL=', '').Trim()

        $isNinja = ($gen -eq "Ninja")

        $GeneratorStale = ($UseNinja -ne $isNinja)

    }

}

if ($null -ne $CachedCuda -and $CachedCuda -ne $UseCuda) {

    Write-Host "(info) CMake cache had SDX_BUILD_CUDA=$CachedCuda but this run uses $UseCuda - removing $BuildDir"

    Remove-Item -Recurse -Force $BuildDir

} elseif ($GeneratorStale -and (Test-Path $BuildDir)) {

    Write-Host "(info) CMake generator changed (Ninja=$UseNinja) - removing $BuildDir for a clean configure"

    Remove-Item -Recurse -Force $BuildDir

}



$CMakeArgs = @(

    "-S", $Cpp,

    "-B", $BuildDir,

    "-DCMAKE_BUILD_TYPE=Release",

    "-DSDX_BUILD_CUDA=$UseCuda"

)

if ($UseNinja) {

    $CMakeArgs = @("-G", "Ninja") + $CMakeArgs

}

if ($LaunchC) {

    $CMakeArgs += "-DCMAKE_C_COMPILER_LAUNCHER=$LaunchC"

}

if ($LaunchCXX) {

    $CMakeArgs += "-DCMAKE_CXX_COMPILER_LAUNCHER=$LaunchCXX"

}

if ($UseCuda -eq "ON" -and $LaunchCUDA) {

    $CMakeArgs += "-DCMAKE_CUDA_COMPILER_LAUNCHER=$LaunchCUDA"

}

if ($env:SDX_CUDA_ARCHITECTURES) {

    $CMakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$($env:SDX_CUDA_ARCHITECTURES)"

}

# Embedded debug info caches better with sccache + MSVC.

if (($env:OS -match "Windows") -and ($LaunchCXX -eq "sccache")) {

    $CMakeArgs += "-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded"

}



if ($UseNinja) {

    Write-Host "==> CMake configure (Ninja): $Cpp  SDX_BUILD_CUDA=$UseCuda"

} else {

    Write-Host "==> CMake configure (default generator): $Cpp  SDX_BUILD_CUDA=$UseCuda"

}

if ($LaunchCXX) {

    $cudaNote = ""

    if ($UseCuda -eq "ON" -and $LaunchCUDA) { $cudaNote = "  CUDA launcher: $LaunchCUDA" }

    Write-Host ("    CXX launcher: {0}{1}" -f $LaunchCXX, $cudaNote)

}



& cmake @CMakeArgs

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }



Write-Host "==> CMake build: Release"

if ($UseNinja) {

    & cmake --build $BuildDir --parallel

} else {

    & cmake --build $BuildDir --config Release --parallel

}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }



if ($UseNinja) {

    Write-Host "Artifacts: $BuildDir (Ninja single-config Release)"

} else {

    Write-Host "Artifacts: typically $BuildDir\Release *.dll when using Visual Studio generator"

}



if (-not $env:RUSTC_WRAPPER) {

    if (Get-Command sccache -ErrorAction SilentlyContinue) {

        $env:RUSTC_WRAPPER = "sccache"

        Write-Host "(info) RUSTC_WRAPPER=sccache for cargo"

    }

}



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

    Write-Host "==> cargo build: sdx-diffusion-math"

    Push-Location (Join-Path $RepoRoot "native\rust\sdx-diffusion-math")

    try {

        & cargo build --release

        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    } finally {

        Pop-Location

    }

    Write-Host "==> cargo build: sdx-image-metrics"

    Push-Location (Join-Path $RepoRoot "native\rust\sdx-image-metrics")

    try {

        & cargo build --release

        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    } finally {

        Pop-Location

    }

} else {

    Write-Host "(skip) cargo not on PATH - Rust tools optional"

}



if ($null -eq $OldRustWrapper) {

    Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue

} else {

    $env:RUSTC_WRAPPER = $OldRustWrapper

}



Write-Host "Done."

