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
| **Tests** | Handful | **803+** (CI: ruff, pytest, smoke, doc links) |
| **Docs** | Scattered READMEs | Structured `docs/` + release notes per version |
| **Transparency** | Readable ~500 LOC entry | Full metadata, provenance, reproducibility guides |

### Capability timeline

| Version | Tag | Focus |
|---------|-----|--------|
| **v0.1** | `v0.1.0` | Core DiT training + sampling framework |
| **v0.2** | `v0.2.0` | Flow matching, DPO, knowledge distillation |
| **v3** | `v3` | Automated hard-case detection, benchmark training loops |
| **v4** | `v4` | Smart quality filtering, adaptive iteration |
| **v5** | `v5.0.0` | Inference scaling, beam search, data curation |
| **v6** | `v6.0.0` | Native acceleration, book/comic generation pipeline |
| **v7** | `v7.0.0` | CI, reproducibility, security, evaluation benchmarks |
| **v8** | `v8.0.0` | Style Genome, unified PromptStack |
| **v9** | `v9.0.0` | GRPO family, Superior Stack, agentic training |
| **v10** | `v10.0.0` | ELIQ, artifact detection, explainable quality |
| **v11** | `v11.0.0` | Regional box layout, frontier research, package restructure |
| **v12** | `v12.0.0` | **AI film studio video pipeline, frontier horizon, 803+ tests** |

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

| Version | Tag | Document |
|---------|-----|----------|
| **v12** | `v12.0.0` | [v12.md](v12.md) · [GitHub release](v12-github-release.md) |
| v11 | `v11.0.0` | [v11.md](v11.md) · [GitHub release](v11-github-release.md) |
| v10 | `v10.0.0` | [v10.md](v10.md) · [GitHub release](v10-github-release.md) |
| v9 | `v9.0.0` | [v9.md](v9.md) |
| v8 | `v8.0.0` | [v8.md](v8.md) · [GitHub release](v8-github-release.md) |
| v7 | `v7.0.0` | [v7.md](v7.md) |
| v6 | `v6.0.0` | [v6.md](v6.md) |
| v5 | `v5.0.0` | [v5.md](v5.md) |
| v4 | `v4` | [v4.md](v4.md) |
| v3 | `v3` | [v3.md](v3.md) |
| v0.2 | `v0.2.0` | [v0.2.0.md](v0.2.0.md) |
| v0.1 | `v0.1.0` | [v0.1.0.md](v0.1.0.md) |

---

## Historical comparison (v8 → v10)

See the detailed v8–v10 matrix in git history; v12 supersedes the metrics above.

**Quality trajectory (relative):**

```
v0.1: Baseline framework — train your own DiT
v0.2: + Flow matching, DPO, distillation
v3:   + Benchmark loops, hard-case automation
v4:   + Smart quality filtering
v5:   + Inference scaling, beam search
v6:   + Native acceleration, book/comic pipeline
v7:   + CI, reproducibility, security
v8:   + Style Genome, Holy Grail (~108%)
v9:   + GRPO, Superior Stack (~125%)
v10:  + ELIQ, explainability (~140%)
v11:  + Layout control, frontier (~150%)
v12:  + Video studio, 25 filmmaker modules (~multimodal)
```
