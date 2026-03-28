# Rust — `sdx-jsonl-tools`

Fast JSONL manifest **stats**, **validate**, **prompt-lint**, **`image-paths`**, **`dup-image-paths`**, plus **`file-fnv`** (raw-byte FNV-1a 64 + newline count), **`file-md5`** (streaming MD5 for image dedup; matches Python `hashlib.md5`), and **`caption-len-buckets`** (caption length histogram for trimming datasets).

```bash
cargo build --release
target/release/sdx-jsonl-tools stats path/to/manifest.jsonl
target/release/sdx-jsonl-tools file-fnv path/to/manifest.jsonl
target/release/sdx-jsonl-tools file-md5 path/to/image.png
target/release/sdx-jsonl-tools caption-len-buckets path/to/manifest.jsonl --width 32 --buckets 128
target/release/sdx-jsonl-tools image-paths --sort path/to/manifest.jsonl
target/release/sdx-jsonl-tools dup-image-paths path/to/manifest.jsonl
```

See [../../README.md](../../README.md) for full usage.
