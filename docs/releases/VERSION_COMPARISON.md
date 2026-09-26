# SDX Version Comparison

## v1 → v13 at a glance

| | **v1** (foundation) | **v13** (current) |
|---|---------------------|-------------------|
| **What it was** | Train + sample DiT on your data | Image + video + RSI + Invention Lab |
| **Entry points** | `train.py`, `sample.py` | + video CLI, VIDEOMAX, feedback / rsi_loop |
| **Training modes** | Diffusion, flow, DPO | + GRPO, agentic loops, **user-feedback DPO** |
| **Video** | None | Scene-graph TI2V + **VIDEOMAX** |
| **Self-improve** | None | RSI feedback bus → flywheel → DPO |
| **Tests** | Handful | **1700+** |

### Capability timeline

| Version | Tag | Focus |
|---------|-----|--------|
| **v0.1** | `v0.1.0` | Core DiT train + sample |
| **v0.2** | `v0.2.0` | Flow matching, DPO, distillation |
| **v3–v7** | … | Hard-cases, quality, native, CI |
| **v8** | `v8.0.0` | Style Genome, PromptStack |
| **v9** | `v9.0.0` | GRPO, Superior Stack |
| **v10** | `v10.0.0` | ELIQ, explainable quality |
| **v11** | `v11.0.0` | Regional layout, frontier |
| **v12** | `v12.0.0` | AI film studio video pipeline |
| **v13** | `v13.0.0` | **VIDEOMAX, RSI, Invention waves, 1700+ tests** |

## Release notes index

| Version | Tag | Document |
|---------|-----|----------|
| **v13** | `v13.0.0` | [v13.md](v13.md) · [GitHub](v13-github-release.md) |
| v12 | `v12.0.0` | [v12.md](v12.md) · [GitHub](v12-github-release.md) |
| v11 | `v11.0.0` | [v11.md](v11.md) |
| v10 | `v10.0.0` | [v10.md](v10.md) |
| v9 | `v9.0.0` | [v9.md](v9.md) |
| v8 | `v8.0.0` | [v8.md](v8.md) |

Earlier releases: [v7](v7.md) · [v6](v6.md) · [v5](v5.md) · [v4](v4.md) · [v3](v3.md) · [v0.2](v0.2.0.md) · [v0.1](v0.1.0.md)
