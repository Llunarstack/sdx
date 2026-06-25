# SDX Version Comparison

## v1 → v12 at a glance

| | **v1** (foundation) | **v12** (current) |
|---|---------------------|-------------------|
| **What it was** | Train + sample DiT on your data | Full research studio: image + video + frontier |
| **Entry points** | `train.py`, `sample.py` | + video CLI, frontier bridge, 15+ tools |
| **Training modes** | Diffusion, flow matching, DPO | + GRPO (6 variants), distillation, agentic loops |
| **Layout control** | None | Regional box prompting, Omost canvas, LAMIC |
| **Video** | None | Scene-graph TI2V, 60+ modules, 25 frontier filmmakers |
| **Quality stack** | Basic loss | ELIQ, artifacts, drift, explainability, TCIS |
| **Research layer** | Monolithic scripts | `innovations/` + `frontier/` (80+ modules) |
| **Tests** | Handful | **802+** (CI: ruff, pytest, smoke, doc links) |
| **Docs** | Scattered READMEs | Structured `docs/` + release notes per version |
| **Transparency** | Readable ~500 LOC entry | Full metadata, provenance, reproducibility guides |

### Capability timeline

| Version | Focus |
|---------|--------|
| **v1 / v0.1** | Core DiT training + sampling framework |
| **v3–v7** | CI, acceleration, benchmarks, data filtering |
| **v8** | Style Genome, PromptStack v2 |
| **v9** | GRPO family, Superior Stack, agentic training |
| **v10** | ELIQ, artifact detection, explainable quality |
| **v11** | Regional box layout, frontier research, package restructure |
| **v12** | **AI film studio video pipeline, frontier horizon, 800+ tests** |

---

## Feature matrix (selected)

| Capability | v1 | v8 | v11 | **v12** |
|---|:---:|:---:|:---:|:---:|
| Text-to-image training | ✓ | ✓ | ✓ | **✓** |
| Flow matching | ◐ | ✓ | ✓ | **✓** |
| DPO / GRPO | ✗ | ◐ | ✓ | **✓** |
| Regional box layout | ✗ | ✗ | ✓ | **✓** |
| Holy Grail / TCIS | ✗ | ✓ | ✓ | **✓** |
| Agentic quality stack | ✗ | ✗ | ◐ | **✓** |
| Frontier research modules | ✗ | ✗ | ◐ | **✓** |
| Video / TI2V pipeline | ✗ | ✗ | ✗ | **✓** |
| Scene graph director | ✗ | ✗ | ✗ | **✓** |
| Continuity validators | ✗ | ✗ | ✗ | **✓** |
| 800+ automated tests | ✗ | ✗ | ◐ | **✓** |

◐ = partial · ✗ = not available · ✓ = built in

---

## Release notes index

| Version | Document |
|---------|----------|
| **v12** | [v12.md](v12.md) · [GitHub release](v12-github-release.md) |
| v11 | [v11.md](v11.md) |
| v10 | [v10.md](v10.md) |
| v9 | [v9.md](v9.md) |
| v8 | [v8.md](v8.md) |
| v1 foundation | [v0.1.0.md](v0.1.0.md) |

---

## Historical comparison (v8 → v10)

See the detailed v8–v10 matrix in git history; v12 supersedes the metrics above.

**Quality trajectory (relative):**

```
v1:   Baseline framework — train your own DiT
v8:   + Style Genome, Holy Grail (~108%)
v9:   + GRPO, Superior Stack (~125%)
v10:  + ELIQ, explainability (~140%)
v11:  + Layout control, frontier (~150%)
v12:  + Video studio, 25 filmmaker modules, continuity (~multimodal)
```
