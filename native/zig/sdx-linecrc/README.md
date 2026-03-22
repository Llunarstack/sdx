# Zig — `sdx-linecrc`

Streaming **FNV-1a 64** fingerprint over a file (or stdin line-by-line).

```bash
zig build -Doptimize=ReleaseFast
zig-out/bin/sdx-linecrc --file path/to/manifest.jsonl
```

Python mirrors file-mode hashing in `sdx_native.native_tools.fnv1a64_file`.
