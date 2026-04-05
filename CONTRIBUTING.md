# Contributing

Thanks for helping improve SDX.

For **why contribute**, **ideas for first PRs**, and a **dev quick start**, see the README section **[Contributing & community](README.md#contributing--community)**.

**Context:** SDX is a **modular training/sampling codebase** first. Not every PR needs a new benchmark or sample images—docs, small tools, and **tiny reproducible training configs** are valuable too. See the README section **Project status, compute, and expectations** for how we frame scope.

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
   python scripts/tools/dev/quick_test.py
   ```
4. **Manual sanity** (when you touch training/sampling/core utils): run a minimal `python -m py_compile` on changed modules and/or a short `sample.py` / `train.py` invocation with your new flags if applicable.
5. **Docs** — If you add flags or new modules, update `README.md` and/or `docs/FILES.md` when it helps others find the change.
6. **Doc links** (if you edit cross-links in markdown)
   ```bash
   python scripts/tools/repo/verify_doc_links.py
   ```

## Style

- **Ruff** is the source of truth (`pyproject.toml`).
- Prefer **clear names** and **small focused functions** over clever one-liners.
- For large features, a short note in `docs/` or an entry in `docs/IMPROVEMENTS.md` is welcome.

## License

By contributing, you agree your contributions are under the same **Apache-2.0** license as the project.
