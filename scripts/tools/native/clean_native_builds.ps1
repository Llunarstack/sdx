# Remove local native build artifacts (safe — does not touch source).
#   .\scripts\tools\native\clean_native_builds.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$Cpp = Join-Path $RepoRoot "native\cpp"

foreach ($name in @("build", "build-compile-db")) {
    $p = Join-Path $Cpp $name
    if (Test-Path $p) {
        Remove-Item -Recurse -Force $p
        Write-Host "Removed $p"
    }
}

Get-ChildItem -Path (Join-Path $RepoRoot "native\rust") -Directory -Recurse -Filter "target*" |
    Where-Object { $_.Name -ne "target" } |
    ForEach-Object {
        Remove-Item -Recurse -Force $_.FullName
        Write-Host "Removed $($_.FullName)"
    }

Write-Host "Done."
