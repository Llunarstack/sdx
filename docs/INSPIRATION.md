# Tools, repos, and inspiration

References and ideas we use to make the model and images better.

## Cloned reference repos (`external/`)

| Repo | What we take from it |
|------|----------------------|
| [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | Base transformer for diffusion; patch embed, timestep embed, adaLN. |
| [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | Structural conditioning (depth, edge, pose); control scale blending. |
| [black-forest-labs/flux](https://github.com/black-forest-labs/flux) | Modern diffusion design; fast inference; img2img/editing. |
| [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) | SD3 / MM-DiT; official stack; architecture and training ideas. |
| [PixArt-alpha/PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) | **PixArt-α:** T5 + DiT, efficient T2I on a fraction of SD-style budget. |
| [PixArt-alpha/PixArt-sigma](https://github.com/PixArt-alpha/PixArt-sigma) | **PixArt-Σ:** 4K T2I, weak-to-strong training, token compression attention. |
| [Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image) | **Z-Image (S3-DiT):** Single-stream DiT; text and image in one sequence; efficient. |
| [willisma/SiT](https://github.com/willisma/SiT) | **SiT:** Flow matching + DiT backbone; interpolant framework; faster convergence. |
| [Alpha-VLLM/Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X) | **Lumina-T2I / Next-DiT:** Next-gen DiT scaling; multi-resolution; Rectified Flow. |

Run `scripts/setup/clone_repos.ps1` (Windows) or `scripts/setup/clone_repos.sh` (Linux/macOS) to clone all into `external/`. SDX does not import them at runtime.

## [PixAI.art](https://pixai.art/en/generator/image) (website — not a repo)

**PixAI.art** is the AI art generator site we take **prompt and tag style** from. It is **not** the same as PixArt-alpha (a separate T5+DiT research repo on GitHub; we don’t clone it and it’s unrelated to PixAI.art).

- **PixAI-style prompts**: Tag-based, `(tag)` / `((tag))` emphasis, `[tag]` de-emphasis, subject-first order.
- **Model lineup**: XL and DiT families; named lines (Haruka, Tsubaki, Hoshino, etc.). See `config/pixai_reference.py`.

## ComfyUI / A1111 ideas

- **CFG rescale / dynamic threshold**: Internal sampling uses fixed behavior; not exposed as CLI options.
- **Post-process**: Optional `--sharpen`, `--contrast` in `sample.py`. See `utils/quality/quality.py`.

## Features that make images look better and match the user

1. **Negative prompt** — Model subtracts negative conditioning so it avoids unwanted content.
2. **Quality tags** — Boost tags like `masterpiece`, `best quality` in data so the model learns to improve quality.
3. **Style + ControlNet + LoRA** — Blended with scales (style_strength, control_scale, per-LoRA scale) so output isn’t messy.
4. **CFG rescale / dynamic threshold** — (Internal only; not exposed in sample CLI.)
5. **Post-process** — Optional sharpen and contrast for a crisper look (see `utils/quality/quality.py`).
6. **Step-based training + save best** — More steps → better checkpoint; no epoch ceiling.
7. **Refinement** — Train on small-t to fix imperfections; optional refinement pass at inference.
8. **Img2img + inpainting** — Training and sample.py `--init-image`, `--mask`.
9. **Aesthetic / sample weighting** — In JSONL, use `weight` or `aesthetic_score` so high-quality samples count more in loss.

## Optional dependencies for quality

- **scipy** — For `--sharpen` (unsharp mask). Install with `pip install scipy` if you use post-process sharpen.

## File map and roadmap

- **[FILES.md](FILES.md)** — Map of all SDX project files and key files in external repos (DiT, ControlNet, flux, generative-models) with links.
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** — Structured improvement ideas (quality, fixes to old techniques, features from other SD/DiT/FLUX models).
