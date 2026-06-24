# Agentic Stack

Autonomous **plan → tool → verify → reflect → evolve** layer for SDX image generation.

Built on research from GenEvolve, VisionCreator (UTPC), and VisionCreator-R1 (Act–Reflect–Think–Act).

## Architecture

```
User prompt
    │
    ▼
┌─────────┐   UTPC plan (research/agi_image)
│ Planner │── rag → expand → generate → verify → reflect
└─────────┘
    │
    ▼
┌─────────┐   sample.py + Superior Stack flags
│  Tools  │── composite rank, self-correct, flywheel, benchmark
└─────────┘
    │
    ▼
┌──────────┐   best–worst trajectory diff
│ Experience│── prompt patches → experience_memory.jsonl
└──────────┘
```

## Modules (`utils/agentic/` → `utils/_archive/agentic/`)

Generation orchestration agents (plan → tool → verify → reflect). Implementations live in the archive; import via `utils.agentic` shims.

| Module | Role |
|--------|------|
| `agent.py` | `ImageGenerationAgent` — main loop |
| `tools.py` | Tool registry (RAG, generate, verify, flywheel, …) |
| `planner.py` | UTPC plan builder |
| `reflector.py` | Act–Reflect–Think–Act verdicts |
| `experience.py` | GenEvolve-style trajectory distillation |
| `roles.py` | Designer / Verifier / Reasoner single-pass pipeline |
| `state.py` | Traces + trajectory records |

## CLI

```bash
# Single agentic run (reflect loop)
python -m scripts.tools agentic_generate \
  --ckpt results/best.pt \
  --prompt "portrait of a scientist, studio lighting" \
  --local-rag-jsonl datasets/facts.jsonl \
  --vit-ckpt results/vit/best.pt \
  --out out.png

# Multi-trajectory evolve + experience memory
python -m scripts.tools agentic_evolve \
  --ckpt results/best.pt \
  --prompt "neon alley at night" \
  --variants 3 --work-dir evolve_run

# Full agentic flywheel (evolve → benchmark → align)
python -m scripts.tools agentic_flywheel \
  --base-ckpt results/best.pt \
  --prompt "your test prompt" \
  --local-rag-jsonl datasets/facts.jsonl

# Explicit multi-role pass (no reflect loop)
python -m scripts.tools agentic_roles \
  --ckpt results/best.pt \
  --prompt "mountain lake at dawn" \
  --out roles_out.png
```

PowerShell one-click:

```powershell
.\scripts\tools\ops\run_agentic.ps1 -Ckpt results\best.pt -Prompt "neon alley"
.\scripts\tools\ops\run_agentic.ps1 -Ckpt results\best.pt -Prompt "..." -Mode evolve -Variants 3
```

## Traces

Each run writes `agent_trace.json` under `--work-dir` with trajectories, reflections, and patches.

Experience memory appends to `experience_memory.jsonl` for cross-session learning.

## Relation to Superior Stack

- **Superior Stack** (`utils/superior/` → archive) = tools and quality modules
- **Agentic Stack** (`utils/agentic/` → archive) = autonomous orchestration over those tools
- **Innovations agentic** (`innovations/agentic/`) = quality/adherence systems (ELIQ, artifacts, drift) — see [QUALITY_AGENTS.md](QUALITY_AGENTS.md)
- **Research schemas** (`research/agi_image/`) = portable plan/message types

## Honest scope

This is a **real, local agent loop** — not a frontier MLLM agent. Planning and reflection are heuristic (optionally LLM via `--expand-prompt` / Qwen when configured). Value: structured trajectories, automatic retries, and experience memory without manual CLI chaining.
