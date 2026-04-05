# Go — `sdx-manifest`

JSONL **merge** with dedupe (first row wins per key).

```bash
cd sdx-manifest
go build -o sdx-manifest .
./sdx-manifest merge -o merged.jsonl a.jsonl b.jsonl
```

Output binary is expected next to the source tree for `sdx_native.native_tools` discovery (`sdx-manifest` / `sdx-manifest.exe`).
