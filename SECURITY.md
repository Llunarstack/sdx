# Security

## Supported versions

Security fixes are applied on the **`main`** branch. Release tags (e.g. `v7.x`) point at snapshots; use **`main`** for the latest fixes.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

- Prefer **private** contact to the repository owner / maintainers (GitHub Security Advisories if enabled, or a direct message path you already use with the project owner).
- Include: affected component (e.g. `sample.py`, native extension), reproduction steps, and impact (RCE, local file read, etc.).

## Scope notes

- SDX runs **local PyTorch** jobs and may download **models and dependencies** from the network. Treat checkpoints and `pip` / `requirements.txt` sources as **trusted supply chain** inputs.
- Native extensions (Rust/C/C++/CUDA) are **compiled locally**; only build artifacts you trust.

## General hardening (operators)

- Run training and sampling in a **dedicated venv**; avoid `sudo pip`.
- Do not pass untrusted **file paths** into tools that read arbitrary paths from captions or manifests without review.