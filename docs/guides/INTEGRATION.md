# Integration Guide

Current integration points for multi-sample quality tooling and spatial layout control.

## Batch / quality tooling

Batch-effect prototype APIs under `utils.inference` were removed. For multi-sample quality:

- `python -m scripts.tools superior_generate` / pick-best via Superior Stack
- `python -m scripts.tools image_quality_qc`
- See [SUPERIOR_STACK.md](../SUPERIOR_STACK.md) and [QUALITY.md](../QUALITY.md)

## Spatial layout

Region-level composition goes through `sample.py --box-layout layout.json`
(`utils/generation/regional_box_prompting.py`): each region carries a normalized box and its own prompt,
applied as regional CFG during sampling. See `examples/box_layout.example.json` and
`examples/box_layout_sketch.example.json`.
