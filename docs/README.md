# SDX Documentation

**Start here:** [GETTING_STARTED.md](GETTING_STARTED.md) · [../README.md](../README.md) (install + quick demo)

| Doc | What it covers |
|-----|----------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Install, first image, train/sample basics, FAQ, hardware, smoke training, reproducibility, fast-training defaults, local CI mirror |
| [CODEBASE.md](CODEBASE.md) | **Repo map:** layers, modules, train/sample flows, where to edit |
| [HOW_GENERATION_WORKS.md](HOW_GENERATION_WORKS.md) | End-to-end pipeline + checkpoint wiring, Holy Grail per-step scheduling, solvers/schedules, TCIS, agentic stack, visual brain, editing phase |
| [PROMPT_STACK.md](PROMPT_STACK.md) | Text path before T5 (inference + training parity), prompt cookbook, style/artist tags, region captions, hard domains |
| [MODEL_STACK.md](MODEL_STACK.md) | Encoders, DiT, VAE/RAE, block-wise AR, recommended `pretrained/` stack, native + system libs |
| [QUALITY.md](QUALITY.md) | Sampling fixes, failure modes, mitigations, Superior Stack, RSI feedback, eval recipes |
| [GUIDES.md](GUIDES.md) | Captions ↔ pixels, captioning, Danbooru/HF data, dataset shortlist, book/comic, character consistency, VIDEOMAX |
| [GLOSSARY.md](GLOSSARY.md) | Plain-English jargon & acronym reference |
| [FILES.md](FILES.md) | Per-file map of the current tree (generated) + external reference repos |
| [RESEARCH.md](RESEARCH.md) | Roadmaps, industry snapshots, competitive analysis, design notes — not needed day to day |
| [CHANGELOG.md](CHANGELOG.md) | Release notes, newest first (v13 → v0.1) |

Elsewhere: [../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) (generated tree), [../DEPRECATIONS.md](../DEPRECATIONS.md) (removed paths), [../frontier/README.md](../frontier/README.md), [../pipelines/README.md](../pipelines/README.md).

## Maintenance

```bash
python -m scripts.tools smoke_imports
python -m scripts.tools verify_doc_links
python -m scripts.tools update_project_structure
pytest tests/ -q
```
