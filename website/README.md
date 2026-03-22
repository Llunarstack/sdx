# SDX Codebase Atlas (static site)

**Recent:** multi-view **dashboard** (Overview · Files · Graph · Pipeline), glass UI, **intelligence drawer**, import navigation — regenerate data with `python scripts/tools/generate_codebase_site.py` from repo root.

Polished **read-only browser** for source files: docstring summaries, full detail text, **Python import graph** (imports + reverse “imported by”), role badges, and search. Indexed paths **exclude** `docs/`, `external/`, `model/` (weights), and caches.

## View locally

**Offline-first:** `index.html` loads **`data/files-inline.js`**, which sets `window.__SDX_CODEBASE__`, so the atlas works from **`file://`** or any static host **without** `fetch` to `files.json`.

Optional: serve the folder and use `files.json` only if the inline bundle is missing:

```bash
cd website
python -m http.server 8765
```

Open **http://127.0.0.1:8765/**

## Regenerate

From the repo root:

```bash
python scripts/tools/generate_codebase_site.py
```

This writes:

- **`files.json`** — same payload as the inline bundle (for HTTP `fetch` fallback).
- **`data/files-inline.js`** — embeds the index for offline use (large file; commit or gitignore per your policy).

**Content per file**

- **`summary`** — first line of module docstring (or markdown heading / fallback).
- **`detail`** — full docstring up to ~1.8k chars, or markdown intro paragraphs.
- **`imports`** / **`imported_by`** — resolved **intra-repo** Python imports (stdlib / PyTorch / pip packages are not listed).
- **`atlas_summary`** / **`atlas_tags`** — how each file fits **image gen**, **book/comic**, **training**, and **sampling** (`scripts/tools/atlas_pipeline_meta.py`).

## Design

- **Inter** + **JetBrains Mono**; **dark “Codebase Atlas” dashboard** — navy background, glass panels, cyan/violet accents (aligned with the product mockup).
- **Left nav rail:** **Overview** · **Files** · **Graph** · **Pipeline** · link to **GitHub**.
- **Overview:** hero copy, live stats (indexed files, import edges, generation time), tiles for dataset/config, model core, sampling/pipelines.
- **Files:** search + role filter, **sticky “Jump” mini-TOC** by top-level folder, **role-colored** cards with atlas summary, tags, docstring + import graph.
- **Graph:** inline SVG **architecture map** (data, models/diffusion, sampling, config, utils, pipelines, ViT).
- **Pipeline:** **prompt → image** diagram (image gen vs book/comic lanes + shared merge) and **training timeline** with repo paths per step.
- **Intelligence panel** (right drawer): purpose, dependencies / used-by, docstring snippet — open from a card or follow import links.

## Files

| File | Role |
|------|------|
| `index.html` | Shell, hero, filters, TOC, loads inline bundle + `app.js` |
| `styles.css` | Theme and layout |
| `app.js` | Renders cards, search, TOC anchors, import navigation |
| `files.json` | Generated JSON (do not hand-edit) |
| `data/files-inline.js` | Generated `window.__SDX_CODEBASE__` assignment |
