# Contributing

Thanks for helping improve SDX.

SDX is a **modular training/sampling codebase** first. Not every PR needs a new benchmark or sample images: docs, small tools, and **tiny reproducible training configs** are valuable too. For what SDX is and is not, see the README sections **[What is SDX?](README.md#what-is-sdx)** and **[What this is *not*](README.md#what-this-is-not)**.

## Before you open a PR

1. **Run from repo root** so imports resolve (`config`, `data`, `models`, …).
2. **Format & lint**
```bash
pip install ruff
ruff format .
ruff check .
# CI also enforces: ruff format --check .
```
3. **Tests** — run the test suite before submitting:
   ```bash
   pytest tests/ -v
   ```
   For a faster check that doesn't need a GPU:
   ```bash
   python -m scripts.tools quick_test
   ```
4. **Manual sanity** (when you touch training/sampling/core utils): run a minimal `python -m py_compile` on changed modules and/or a short `sample.py` / `train.py` invocation with your new flags if applicable.
5. **Docs** — If you add flags or new modules, update `README.md` and/or `docs/FILES.md` when it helps others find the change.
6. **Doc links** (if you edit cross-links in markdown)
   ```bash
   python -m scripts.tools verify_doc_links
   ```


7. **Typecheck spot-check** (matches CI; fail only on `errorCount`, not warnings):

   ```bash
   pip install basedpyright
   OUT=/tmp/basedpyright.json
   python -m basedpyright --outputjson native/_experimental/python/sdx_native/native_tools.py diffusion/schedules.py examples/run_baseline_eval.py utils/runtime.py training/cli.py utils/generation/sample_cli_parser.py > "$OUT" || true
   python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8-sig')); s=d.get('summary') or {}; sys.exit(1 if int(s.get('errorCount',0)) else 0)" "$OUT"
   ```

   Full mirror: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md#mirror-github-ci-locally).

8. **Optional pre-commit** — [`.pre-commit-config.yaml`](.pre-commit-config.yaml): `pip install pre-commit && pre-commit install`.
9. **Commit hook** — keep AI tools off the contributor list: `cp scripts/tools/dev/commit-msg .git/hooks/commit-msg` drops Claude / Cursor `Co-authored-by` and `Made-with` lines from every commit message. GitHub counts those trailers as contributors.

## Style

- **Ruff** is the source of truth (`pyproject.toml`).
- Prefer **clear names** and **small focused functions** over clever one-liners.
- For large features, a short note in `docs/` or an entry in `docs/RESEARCH.md` is welcome.

## Engineering standards

- **Truthful smoke/doc gates** — Do not skip live packages as archived. Smoke tests and doc-link checks must reflect what is actually shipped and importable.
- **Thin CLIs** — Keep entrypoints thin; extract argparse/parsers into focused modules (e.g. `training/cli.py`). Avoid new mega-files that mix parsing, I/O, and domain logic.
- **Type allowlist** — When you touch a module under the basedpyright/CI spot-check set, expand the allowlist so the changed surface stays typechecked.
- **Tools dispatcher** — Prefer `python -m scripts.tools <command>` over adding new root-level CLI scripts.

## License

By contributing, you agree your contributions are under the same **Apache-2.0** license as the project.
