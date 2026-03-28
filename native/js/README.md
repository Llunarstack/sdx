# JavaScript tools (removed)

Former `*.mjs` helpers live in pure Python now:

```bash
python -m sdx_native.jsonl_manifest_pure stat path/to/manifest.jsonl
python -m sdx_native.jsonl_manifest_pure promptlint path/to/manifest.jsonl
```

(`native/python` must be on `PYTHONPATH`, or run from repo root so `pyproject.toml` applies.)

For a compiled fast path, build **Rust** `native/rust/sdx-jsonl-tools` instead.
