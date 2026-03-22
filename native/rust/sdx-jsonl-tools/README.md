# Rust — `sdx-jsonl-tools`

Fast JSONL manifest **stats**, **validate**, **prompt-lint**, **`image-paths`**, **`dup-image-paths`** (aligned with `data/t2i_dataset.py` key names).

```bash
cargo build --release
target/release/sdx-jsonl-tools stats path/to/manifest.jsonl
target/release/sdx-jsonl-tools image-paths --sort path/to/manifest.jsonl
target/release/sdx-jsonl-tools dup-image-paths path/to/manifest.jsonl
```

See [../../README.md](../../README.md) for full usage.
