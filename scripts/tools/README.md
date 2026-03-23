# `scripts/tools/` — utility index

Run from repo root (`python scripts/tools/...` or `python -m scripts.tools...`).

## Canonical grouped entrypoints

| Group | Examples |
|---|---|
| **dev** | `python -m scripts.tools.dev.quick_test`, `...dev.smoke_imports` |
| **data** | `python -m scripts.tools.data.data_quality`, `...data.manifest_paths` |
| **prompt** | `python -m scripts.tools.prompt.prompt_lint`, `...prompt.tag_coverage` |
| **ops** | `python -m scripts.tools.ops.orchestrate_pipeline`, `...ops.op_preflight` |
| **export** | `python -m scripts.tools.export.export_onnx`, `...export.export_safetensors` |
| **repo** | `python -m scripts.tools.repo.update_project_structure`, `...repo.verify_doc_links` |

Legacy flat scripts (`scripts/tools/*.py`) are still supported for compatibility.

## See also

- [scripts/README.md](../README.md)
- [docs/REPOSITORY_STRUCTURE.md](../../docs/REPOSITORY_STRUCTURE.md)

