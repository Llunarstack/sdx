# Zig — `sdx-pathstat`

Fast **file size** lookup for a list of paths (one per line). Prints:

`path<TAB>size_bytes<TAB>ok|missing|error:…`

Relative paths use the **current working directory**.

```bash
zig build -Doptimize=ReleaseFast
zig-out/bin/sdx-pathstat --file paths.txt
native/rust/.../target/release/sdx-jsonl-tools image-paths manifest.jsonl | zig-out/bin/sdx-pathstat
```

Use with Rust **`image-paths`** to audit disk usage or find missing files before training.
