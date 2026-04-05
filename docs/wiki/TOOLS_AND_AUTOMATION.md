# Tools and Automation

This page covers the tooling system used to evaluate and improve checkpoints.

## Dispatcher

Run tools via:

```bash
python -m scripts.tools <command> [args...]
```

Use `python -m scripts.tools help` to list all commands.

## High-impact tools

- `benchmark_suite`: compare checkpoints with structured prompt packs.
- `mine_preference_pairs`: build DPO win/lose pairs from benchmark results.
- `auto_improve_loop`: benchmark -> mine -> DPO -> re-benchmark; supports `--iterations`.
- `startup_readiness`: environment report before expensive runs.
- `pretrained_status`: local-vs-HF model resolution report.
- `gen_searcher_bridge`: convert external agentic outputs into SDX fact JSONL.

## Continuous improvement loop

`auto_improve_loop.py` orchestrates:

1. Pre-DPO benchmark.
2. Preference mining.
3. DPO training stage.
4. Post-DPO benchmark.
5. Optional checkpoint promotion.

Enhancements included:

- multi-iteration carry-forward (`--iterations N`)
- seed robustness ranking (`--seed-list`, `--robustness-penalty`)
- hard-case export + hardcase-aware preference boost

## Outputs worth tracking

- benchmark: `results.json`, `leaderboard.json`, `leaderboard.csv`
- preferences: mined JSONL
- hardcases: tagged low-score case JSONL
- ops reports: readiness and pretrained status JSON/Markdown
