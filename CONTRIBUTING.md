# Contributing

Thanks for helping improve SDX.

For **why contribute**, **ideas for first PRs**, and a **dev quick start**, see the README section **[Contributing & community](README.md#contributing--community)**.

**Context:** SDX is a **modular tr/sampling codebase** first. Not every PR needs a new benchmark or sample images—docs, small tools, and **tiny reproducible training configs** are valuable too. See the README section **Project status, compute, and expectations** for how we frame scope.

## Before you open a PR

1. **Run from repo root** so imports resolve (`config`, `data`, `models`, …).
2. **Format & lint**
   ```bash
   pip install ruff
   ruff format .
   ruff check .
   ```
3. **Tests** — run the test suite before submitting:
   ```bash
   pytest tests/ -v
   ```
   For a faster check that doesn't need a GPU:
   ```bash
   python -m scripts.tools quick_test
   ```
4. **Manual sanity** (when you touch tr/sampling/core utils): run a minimal `python -m py_compile` on changed modules and/or a short `sample.py` / `train.py` invocation with your new flags if applicable.
5. **Docs** — If you add flags or new modules, update `README.md` and/or `docs/FILES.md` when it helps others find the change.
6. **Doc links** (if you edit cross-links in markdown)
   ```bash
   python -m scripts.tools verify_doc_links
   ```


7. **Typecheck spot-check** (matches CI; fail only on `errorCount`, not warnings):

   ```bash
   pip install basedpyright
   OUT=/tmp/basedpyright.json
   python -m basedpyright --outputjson native/python/sdx_native/diffusion_sigma_fast.py utils/generation/run_artifacts.py diffusion/snr_utils.py utils/generation/inference_stages.py utils/generation/eval_prompt_pack.py examples/run_baseline_eval.py > "$OUT"
   python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8-sig')); s=d.get('summary') or {}; sys.exit(1 if int(s.get('errorCount',0)) else 0)" "$OUT"
   ```

   Full mirror: [docs/recipes/local_ci_mirror.md](docs/recipes/local_ci_mirror.md).

8. **Optional pre-commit** — [`.pre-commit-config.yaml`](.pre-commit-config.yaml): `pip install pre-commit && pre-commit install`.

## Style

- **Ruff** is the source of truth (`pyproject.toml`).
- Prefer **clear names** and **small focused functions** over clever one-liners.
- For large features, a short note in `docs/` or an entry in `docs/IMPROVEMENTS.md` is welcome.

## License

By contributing, you agree your contributions are under the same **Apache-2.0** license as the project.
