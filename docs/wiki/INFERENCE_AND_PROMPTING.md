# Inference and Prompting

How `sample.py` builds prompts, generates candidates, and chooses outputs.

## Generation command shape

```bash
python sample.py --prompt "your prompt" --num 4 --pick-best auto
```

Core stages:

1. Parse flags and seed/runtime settings.
2. Compose final prompt from base text + optional controls/grounding.
3. Generate candidate images.
4. Score candidates with quality/adherence metrics.
5. Pick and save best outputs.

## Prompt stack

Prompt processing can include:

- content controls and safety controls
- style/medium/domain guidance
- negative prompt add-ons for known failure patterns
- external grounding facts (GenSearcher bridge + fact merge)

See also:

- `utils/prompt/content_controls.py`
- `utils/prompt/rag_prompt.py`
- `pipelines/book_comic/prompt_lexicon.py`

## Pick-best scoring

`utils/quality/test_time_pick.py` provides multiple modes, including:

- text adherence and OCR signals
- people/object count matching
- edge sharpness and exposure balance
- saturation balance to reduce oversaturation artifacts

Use `--pick-best auto` for automatic metric profile selection based on prompt constraints.

## Practical quality flags

- `--num`: more candidates helps hard prompts.
- `--pick-best auto`: dynamic strategy for text/count-sensitive prompts.
- `--auto-constraint-boost`: raises candidate count for prompts with strict constraints.
- `--refine-gate auto`: conditionally refine only when preview quality is below threshold.

## Prompt-grounding integration

For agentic/fact-grounded prompting:

1. Convert external research output via `python -m scripts.tools gen_searcher_bridge`.
2. Provide merged facts input to `sample.py` using `--agentic-facts-json` and related flags.
