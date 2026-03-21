# Contributing

Thanks for helping improve SDX.

For **why contribute**, **ideas for first PRs**, and a **dev quick start**, see the README section **[Contributing & community](README.md#contributing--community)**.

**Context:** SDX is a **modular training/sampling codebase** first. Not every PR needs a new benchmark or sample images—docs, tests, small tools, and **tiny reproducible training configs** are valuable too. See the README section **Project status, compute, and expectations** for how we frame scope.

## Before you open a PR

1. **Run from repo root** so imports resolve (`config`, `data`, `models`, …).
2. **Format & lint**
   ```bash
   pip install ruff
   ruff format .
   ruff check .
   ```
3. **Tests** (when you touch training/sampling/core utils)
   ```bash
   pytest tests/ -q
   ```
4. **Docs** — If you add flags or new modules, update `README.md` and/or `docs/FILES.md` when it helps others find the change.
5. **Doc links** (if you edit cross-links in markdown)
   ```bash
   python scripts/tools/verify_doc_links.py
   ```

## Style

- **Ruff** is the source of truth (`pyproject.toml`).
- Prefer **clear names** and **small focused functions** over clever one-liners.
- For large features, a short note in `docs/` or an entry in `docs/IMPROVEMENTS.md` is welcome.

## License

By contributing, you agree your contributions are under the same **Apache-2.0** license as the project.
