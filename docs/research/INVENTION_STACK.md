# Invention stack

`sample.py --invention-stack auto|all|off|<comma list>` rewrites the prompt with passes aimed at the
failure modes text-to-image models are known for. Code: `utils/generation/inventions/`.

| Pass | Module | What it does |
|------|--------|--------------|
| ORACLE | `failure_oracle.py` | Scores prompt risks (count, binding, negation, anatomy, glyph, multi-character, plastic, long prompt) and picks passes in `auto` mode. |
| NEGATRON | `negatron.py` | Moves "no X" / "without X" clauses out of the positive and into the negative. |
| BINDLOCK | `bindlock.py` | Binds colours to their nouns and negates attribute swaps. |
| COUNTGATE | `countgate.py` | Exact-count tags plus a grid box layout (regional CFG) for counted subjects. |
| HYDRA-SLOTS | `countgate.py` | One regional-CFG slot per counted instance. |
| ANATOMON | `priors.py` | Joint-chain anatomy language when the prompt shows a person. |
| FRICTION | `priors.py` | Anti-plastic microtexture language. |
| Scene priors | `priors.py` | Composition, depth layers, eye/teeth/contact-shadow/skin cues (people only), filmic highlights and lens look (photographic or unstated medium only). |

Runtime glue (`wire.py`):

- `--invention-adaptive-steps`: step budget from oracle risk (+3 steps per risk).
- Oracle risks fill unset guidance knobs: `--guidance-schedule piecewise` for count/binding/negation/multi-character
  prompts (not with `--holy-grail`), `--apg-parallel-eta 0` for binding/plastic prompts.
- `--invention-self-heal`: writes `self_healing_plan.json` (rewritten prompt, APEX scheduler/solver argv,
  regions to redo when the critic fires).
- `--cast-json` / `--corpus-jsonl`: multi-character cast compiler and training-corpus retrieval.

The OCR repair re-run receives the already-rewritten prompt, so invention, cast and corpus flags are not forwarded.

## CLI

```bash
python -m scripts.tools invention_lab --prompt "four red cubes left of blue spheres, no text, hands" --enable auto --print-argv
python -m scripts.tools invention_lab --prompt "2girls" --self-heal --print-argv
python -m scripts.tools invention_lab --atlas-out outputs/failure_atlas.jsonl   # failure-mode eval prompts
```
