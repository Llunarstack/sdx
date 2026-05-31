# Agentic Research Map (2025–2026)

| Paper / trend | Idea | SDX module |
|---------------|------|------------|
| GenEvolve | Tool-orchestrated trajectories + visual experience distillation | `utils/agentic/experience.py`, `agentic_evolve` |
| VisionCreator | UTPC: Understand, Think, Plan, Create | `utils/agentic/planner.py` |
| VisionCreator-R1 | Act–Reflect–Think–Act self-correction | `utils/agentic/reflector.py`, reflect loop in `agent.py` |
| Vision-SR1 | Self-reward GRPO (perception + language) | `utils/training/dense_grpo.py`, `train_flow_grpo` |
| Designer/Verifier/Reasoner | Multi-role pipeline | `utils/generation/orchestration.py` + agent tools |

## Folder layout

```
utils/agentic/          ← operational agent (NEW)
research/agi_image/     ← portable schemas + GenerationPlan DAG
utils/superior/         ← quality tools the agent calls
config/defaults/agentic_stack.py
docs/agentic/AGENTIC_STACK.md
```

## When to use what

| Goal | Command |
|------|---------|
| One-shot best image | `agentic_generate` |
| Learn from multiple trajectories | `agentic_evolve` |
| Full self-improve loop | `agentic_flywheel` or `run_flywheel` |
| Manual control | `sample.py` + Superior flags |
