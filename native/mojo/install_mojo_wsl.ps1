# One-shot: Pixi + Mojo inside WSL2 (linux-64). Modular has no win-64 `mojo` conda package.
# Run from repo root or this directory:
#   powershell -File native/mojo/install_mojo_wsl.ps1

$ErrorActionPreference = "Stop"
$Here = Resolve-Path $PSScriptRoot
$UnixHere = (wsl wslpath -a "$Here").Trim()
wsl -e bash -lc "export PATH=`"`$HOME/.pixi/bin:`$PATH`" && cd `"$UnixHere`" && (command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | sh) && export PATH=`"`$HOME/.pixi/bin:`$PATH`" && pixi install && pixi run mojo-run"
