# Model lines and release gates

SDX uses one repository and one shared core. Image, video, and spatial work should
land as focused changes to their modules, with shared changes reviewed separately.
This avoids maintaining three divergent copies of diffusion, conditioning, and
training code.

| Line | Current code | Current status | First independent release gate |
| --- | --- | --- | --- |
| Image | `train.py`, `sample.py`, `diffusion/`, `models/dit_text.py`, `pipelines/image_gen/` | Image training and sampling exist; no trained checkpoint is included in this checkout. | Reproducible training run, held-out quality set, saved checkpoint and inference recipe, latency and VRAM measurements. |
| Video | `pipelines/video/`, `models/video_dit.py` | Compositional scene pipeline and an optional neural video path exist. | State which engine produced each example; validate temporal consistency, clips, audio if claimed, and time/VRAM on published hardware. |
| Spatial / 3D | `frontier/multiview.py` | Research code, without a complete user-facing 3D training and export path. | Add a defined dataset, train/evaluate pipeline, view-consistency tests, mesh or field export, and documented hardware requirements. |

## Branches and releases

- Branch from `main` for each bounded change: `feature/image/<topic>`,
  `feature/video/<topic>`, `feature/spatial/<topic>`, or `fix/core/<topic>`.
  Merge shared core changes first so each line can reuse them. Close branches
  after merge; do not keep separate permanent model-code branches.
- Keep source releases tagged `vMAJOR.MINOR.PATCH`. Release notes should mark
  image, video, and spatial capabilities separately and say which ones have
  trained weights or only implementation code.
- Publish component artifacts under their own versioned names only after their
  gates pass: `sdx-image-vX.Y.Z`, `sdx-video-vX.Y.Z`, and
  `sdx-spatial-vX.Y.Z`. An artifact should include the checkpoint or export,
  config, exact dependency versions, model card, license, evaluation results,
  and a small reproduction command. Do not label research scaffolding as a
  released model.
- CI should run core checks for every pull request and affected-line tests for
  image, video, or spatial changes. Release jobs should run the relevant smoke
  inference and attach measured results before a tag is published.

## Public model references, checked November 15, 2026

These are research leads, not claims that SDX already matches their quality or
speed. Use public behavior and documented techniques; review the license before
using open weights, and do not copy closed weights or proprietary code.

| Model | Verified public information | SDX experiment |
| --- | --- | --- |
| [Seedream 5.5](https://seed.bytedance.com/en/blog/more-realistic-images-more-stable-editing-across-100-rounds-introducing-seedream-5-5) | ByteDance describes repeatable editing, unchanged-region preservation, alpha output, and text rendering. Architecture details are not public. | Measure edit drift over repeated turns, untouched-region error, alpha edges, and text fidelity. |
| [GPT-Image-2.6 Sunburst and Flare](https://openai.com/index/introducing-images-2-6/) | OpenAI documents faster and more detailed variants, plus generation and editing controls in its [API guide](https://developers.openai.com/api/docs/guides/image-prompting?model=gpt-image-2.6). | Compare SDX fast and quality presets on the same prompts, seeds, hardware, latency, VRAM, and quality rubric. |
| [MiniMax H3-Image](https://github.com/MiniMax-AI/MiniMax-H3-Image) | The official repository documents text-to-image/editing, reference images, a large DiT, and a distilled low-step path; its Community License governs weights. | Test reference-image consistency and measure whether fewer steps preserve quality on SDX's own model. |
| [Ideogram 4.0](https://ideogram.ai/blog/ideogram-4.0/) | Ideogram publishes architecture and inference details, including structured captions and quantized checkpoints. | Evaluate structured layout/text prompts and quantization on representative GPU hardware. |

## Performance evidence required

Record a warmed-up baseline with exact commit, config, dataset or prompt pack,
checkpoint, device, batch size, image size, steps, precision, and peak VRAM.
Profile the data loader, VAE/text encoders, denoiser forward/backward, optimizer,
and distributed communication before replacing a hot path with Rust, C++, CUDA,
or Triton. Report throughput and quality together. This checkout has one GPU and
no local training set or checkpoint, so distributed scaling and end-to-end model
speedups cannot yet be measured here.
