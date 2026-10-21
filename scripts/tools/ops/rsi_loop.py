#!/usr/bin/env python3
"""RSI: an SDX checkpoint teaches itself to make better art.

Every round:

1. **Curriculum** builds art prompts (medium x subject x light x framing) from the repo's medium
   library, weighted toward the medium families the champion currently scores worst on.
2. **Paint**: the champion makes ``--candidates`` images per prompt through the normal ``sample.py``
   pipeline (``--jobs-jsonl``, so the checkpoint and text encoders load once per round).
3. **Judge**: every image gets one art score from the HF reward panel (PickScore, CLIP-H, CAFE
   aesthetic; offline, whichever are on disk) + ``art_critic`` (composition, exposure, colour
   harmony) + ``ai_look`` (generated-look tells).
4. **Learn**: best-vs-worst pairs per prompt, plus any human thumbs from the feedback bus,
   train a candidate with ``train_diffusion_dpo``.
5. **Gate**: candidate and champion paint the same held-out prompts with the same seeds. Held-out
   subjects never appear in training prompts. The candidate replaces the champion only if the mean
   score rises by ``--min-gain`` and it wins at least half the head-to-heads. A rejected candidate is
   deleted and the DPO learning rate halves; a promoted one resets it.

State lives in ``<work-dir>/state.json``, so a run resumes where it stopped and ``--rounds`` adds more.

    python -m scripts.tools rsi_loop --base-ckpt results/run/best.pt --rounds 5
    python -m scripts.tools rsi_loop --rounds 3                  # resume outputs/rsi
    python -m scripts.tools rsi_loop --status                    # round history
    python -m scripts.tools rsi_loop --base-ckpt ... --dry-run   # print round 1's prompts and commands

The judge runs with ``HF_HUB_OFFLINE=1`` unless you set it yourself, so an unattended loop never starts
a multi-GB download; that also applies to the sample/DPO subprocesses it launches.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Held-out subjects never appear in training prompts, so the gate measures art skill, not memorised prompts.
TRAIN_SUBJECTS = (
    "a lighthouse on a storm-battered cliff",
    "an old fisherman mending nets",
    "a fox curled up in fresh snow",
    "a crowded night market in the rain",
    "a cathedral interior with stained light",
    "a dancer mid-leap",
    "a still life of pears and a copper jug",
    "a mountain lake at dawn",
    "a dragon coiled around a tower",
    "a cyclist on a coastal road",
    "a cat asleep on a windowsill",
    "a ruined temple overgrown with vines",
    "a jazz trio in a smoky club",
    "a field of sunflowers under a thunderstorm",
    "a portrait of an elderly woman laughing",
    "a sailing ship in heavy seas",
    "a greenhouse full of tropical plants",
    "a knight resting beside a campfire",
    "a koi pond with lily pads",
    "a desert caravan at sunset",
    "a busy train station platform",
    "an owl perched on a mossy branch",
    "a child flying a kite on a hill",
    "a bowl of ramen on a wooden counter",
)
EVAL_SUBJECTS = (
    "a heron wading through reeds",
    "a violinist on a city rooftop",
    "a stone bridge over a misty river",
    "a bakery window at closing time",
    "a wolf howling under the aurora",
    "a lantern-lit alley in an old town",
)
LIGHT = (
    "golden hour light",
    "soft overcast light",
    "dramatic chiaroscuro",
    "cool moonlight",
    "warm candlelight",
    "bright midday sun",
    "neon glow",
    "foggy diffused light",
)
FRAMING = (
    "rule of thirds composition",
    "centered symmetrical composition",
    "wide establishing shot",
    "intimate close-up",
    "low angle view",
    "high angle view",
    "leading lines toward the subject",
    "generous negative space",
)
JUDGE_WEIGHTS = {"reward": 0.45, "art": 0.30, "natural": 0.25}


# ---------------------------------------------------------------- prompts / curriculum


def medium_families() -> dict[str, list[str]]:
    """Medium family -> medium phrases, straight from ``config.defaults.art_mediums``."""
    from config.defaults.art_mediums import MEDIUM_SPECS, detect_medium

    out: dict[str, list[str]] = {}
    for spec in MEDIUM_SPECS:
        phrase = spec.keywords[0]
        out.setdefault(detect_medium(phrase).family, []).append(phrase)
    return out


def art_prompt(rng: random.Random, medium: str, subjects: tuple[str, ...]) -> str:
    return f"{rng.choice(subjects)}, {medium}, {rng.choice(LIGHT)}, {rng.choice(FRAMING)}"


def train_prompts(n: int, weights: dict[str, float], rng: random.Random) -> list[tuple[str, str]]:
    """``n`` (family, prompt) rows, families drawn in proportion to ``weights``."""
    fams = medium_families()
    names = sorted(f for f in fams if weights.get(f, 0) > 0)
    picks = rng.choices(names, [weights[f] for f in names], k=n)
    return [(f, art_prompt(rng, rng.choice(fams[f]), TRAIN_SUBJECTS)) for f in picks]


def eval_prompts(per_family: int) -> list[tuple[str, str]]:
    """Fixed held-out set: same prompts every round, so champion and candidate scores compare."""
    rng = random.Random(1234)
    return [
        (fam, art_prompt(rng, rng.choice(media), EVAL_SUBJECTS))
        for fam, media in sorted(medium_families().items())
        for _ in range(per_family)
    ]


def curriculum_weights(family_scores: dict[str, float]) -> dict[str, float]:
    """Weaker families get more practice; unknown families start at full weight."""
    return {f: max(0.05, 1.05 - family_scores.get(f, 0.0)) for f in medium_families()}


# ---------------------------------------------------------------- judge


class ArtJudge:
    """One art score in [0, 1] per image: reward panel + art critic + generated-look naturalness."""

    def __init__(self, device: str = "cuda") -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from utils.modeling.hf_reward import HFRewardPanel, HFRewardWeights

        zero = {k: 0.0 for k in HFRewardWeights.__dataclass_fields__}
        self.panel = HFRewardPanel(
            HFRewardWeights(**{**zero, "pickscore": 0.45, "clip_h14": 0.20, "cafe_aesthetic": 0.35}),
            device=device,
        )

    def __call__(self, path: str | Path, prompt: str) -> tuple[float, dict[str, float]]:
        import numpy as np
        from PIL import Image
        from utils.quality.ai_look import ai_look_score
        from utils.quality.art_critic import score_art

        rgb = np.asarray(Image.open(path).convert("RGB"))
        reward = self.panel.score_breakdown(rgb, prompt=prompt)
        parts = {"art": score_art(rgb, prompt=prompt).score, "natural": ai_look_score(rgb, prompt=prompt).natural}
        if reward.parts:  # panel falls back to a flat 0.5 when no scorer loaded; leave it out then
            parts["reward"] = reward.composite
        score = sum(JUDGE_WEIGHTS[k] * v for k, v in parts.items()) / sum(JUDGE_WEIGHTS[k] for k in parts)
        return float(score), {**parts, **reward.parts}

    @staticmethod
    def release() -> None:
        """Free scorer VRAM before the DPO subprocess needs it."""
        from utils.modeling import hf_loaders

        hf_loaders._CACHE.store.clear()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------- pairs / gate


def mine_pairs(scored: list[dict[str, Any]], min_margin: float) -> list[dict[str, Any]]:
    """Per prompt, pair best with worst, 2nd best with 2nd worst, ... while the score gap is >= ``min_margin``."""
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_prompt[row["prompt"]].append(row)
    pairs = []
    for prompt, rows in by_prompt.items():
        rows = sorted(rows, key=lambda r: r["score"], reverse=True)
        for i in range(len(rows) // 2):
            win, lose = rows[i], rows[-1 - i]
            margin = win["score"] - lose["score"]
            if margin < min_margin:
                break
            pairs.append(
                {
                    "win_image_path": win["path"],
                    "lose_image_path": lose["path"],
                    "caption": prompt,
                    "source": "rsi_self",
                    "margin": round(margin, 4),
                }
            )
    return pairs


def human_pairs(feedback_log: str | Path) -> list[dict[str, Any]]:
    """Your thumbs/pairs from the feedback bus (image media, files still on disk)."""
    from utils.training.feedback_bus import pairs_from_feedback

    if not Path(feedback_log).is_file():
        return []
    return [
        p
        for p in pairs_from_feedback(feedback_log)
        if Path(p["win_image_path"]).is_file() and Path(p["lose_image_path"]).is_file()
    ]


def gate(champion: dict[str, float], candidate: dict[str, float], min_gain: float) -> dict[str, Any]:
    """Paired comparison on identical (seed, prompt) keys."""
    keys = sorted(champion.keys() & candidate.keys())
    if not keys:
        return {"gain": 0.0, "win_rate": 0.0, "n": 0, "promote": False}
    deltas = [candidate[k] - champion[k] for k in keys]
    gain, win_rate = fmean(deltas), fmean(d > 0 for d in deltas)
    return {
        "gain": round(gain, 5),
        "win_rate": round(win_rate, 3),
        "n": len(keys),
        "promote": gain >= min_gain and win_rate >= 0.5,
    }


def family_means(scored: list[dict[str, Any]]) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        by[r["family"]].append(r["score"])
    return {f: round(fmean(v), 4) for f, v in by.items()}


# ---------------------------------------------------------------- SDX subprocesses


def sdx_generate(ckpt: str, jobs: list[dict[str, Any]], out_dir: Path, opts: argparse.Namespace) -> None:
    """Run every job through ``sample.py`` with one model load. Jobs that fail just leave no file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = out_dir / "jobs.jsonl"
    jobs_path.write_text("".join(json.dumps(j) + "\n" for j in jobs), encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "sample.py"),
        "--ckpt",
        ckpt,
        "--jobs-jsonl",
        str(jobs_path),
        "--num",
        "1",
        "--steps",
        str(opts.steps),
        "--device",
        opts.device,
        "--no-lora-bank",  # train the model itself, not whatever adapters the bank would pick
        *opts.sample_arg,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=False)


def sdx_train(ckpt: str, prefs: Path, out: Path, lr: float, opts: argparse.Namespace) -> bool:
    from utils.superior.auto_loop import run_tool

    argv = [
        "--ckpt",
        ckpt,
        "--preference-jsonl",
        str(prefs),
        "--out",
        str(out),
        "--steps",
        str(opts.dpo_steps),
        "--batch-size",
        "2",
        "--lr",
        str(lr),
        "--dpo-beta",
        str(opts.dpo_beta),
        "--save-every",
        "0",
        "--device",
        opts.device,
    ]
    return run_tool("train_diffusion_dpo", argv, repo_root=ROOT) == 0 and out.is_file()


# ---------------------------------------------------------------- loop


def paint_and_judge(
    ckpt: str,
    rows: list[tuple[str, str]],
    seeds: list[int],
    out_dir: Path,
    *,
    generate: Callable[[str, list[dict[str, Any]], Path], None],
    judge: Callable[[str | Path, str], tuple[float, dict[str, float]]],
) -> list[dict[str, Any]]:
    """Paint every (row, seed) and score what came out."""
    jobs = [
        {"prompt": prompt, "seed": seed, "out": str(out_dir / f"p{i:03d}_s{seed}.png"), "family": fam}
        for i, (fam, prompt) in enumerate(rows)
        for seed in seeds
    ]
    generate(ckpt, [{k: j[k] for k in ("prompt", "seed", "out")} for j in jobs], out_dir)
    scored = []
    for j in jobs:
        if Path(j["out"]).is_file():
            score, parts = judge(j["out"], j["prompt"])
            scored.append({**j, "path": j["out"], "score": round(score, 5), "parts": parts})
    return scored


def eval_scores(scored: list[dict[str, Any]]) -> dict[str, float]:
    return {f"{r['seed']}|{r['prompt']}": r["score"] for r in scored}


def load_state(work: Path) -> dict[str, Any]:
    path = work / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def save_state(work: Path, state: dict[str, Any]) -> None:
    tmp = work / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, work / "state.json")


def run(
    opts: argparse.Namespace,
    *,
    generate: Callable[[str, list[dict[str, Any]], Path], None] | None = None,
    train: Callable[[str, Path, Path, float], bool] | None = None,
    judge: Any = None,
) -> dict[str, Any]:
    """Run ``opts.rounds`` rounds; ``generate`` / ``train`` / ``judge`` are injectable for tests."""
    work = Path(opts.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    state = load_state(work)
    if not state:
        if not opts.base_ckpt:
            raise SystemExit(f"No state in {work}; pass --base-ckpt to start.")
        state = {"base_ckpt": opts.base_ckpt, "champion": opts.base_ckpt, "round": 0, "lr": opts.dpo_lr, "history": []}
    generate = generate or (lambda c, j, d: sdx_generate(c, j, d, opts))
    train = train or (lambda c, p, o, lr: sdx_train(c, p, o, lr, opts))
    judge = judge or ArtJudge(opts.device)
    release = getattr(judge, "release", lambda: None)
    held_out = eval_prompts(opts.eval_per_family)
    eval_seeds = list(range(9000, 9000 + opts.eval_seeds))

    if "champion_eval" not in state:
        print(f"[rsi] baseline: judging champion {state['champion']} on {len(held_out)} held-out prompts")
        scored = paint_and_judge(
            state["champion"], held_out, eval_seeds, work / "round_000" / "eval", generate=generate, judge=judge
        )
        if not scored:
            raise SystemExit("Baseline produced no images; check that sample.py runs with this checkpoint.")
        state["champion_eval"] = eval_scores(scored)
        state["family_scores"] = family_means(scored)
        save_state(work, state)

    for _ in range(opts.rounds):
        rnd = state["round"] + 1
        rdir = work / f"round_{rnd:03d}"
        rng = random.Random(opts.seed + rnd)
        weights = curriculum_weights(state["family_scores"])
        rows = train_prompts(opts.prompts, weights, rng)
        seeds = [rng.randrange(1, 2**31) for _ in range(opts.candidates)]
        print(f"[rsi] round {rnd}: {len(rows)} prompts x {len(seeds)} candidates, weights={weights}")

        scored = paint_and_judge(state["champion"], rows, seeds, rdir / "train", generate=generate, judge=judge)
        pairs = mine_pairs(scored, opts.min_margin) + (human_pairs(opts.from_feedback) if opts.from_feedback else [])
        prefs = rdir / "prefs.jsonl"
        prefs.write_text("".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
        report: dict[str, Any] = {"round": rnd, "lr": state["lr"], "pairs": len(pairs), "images": len(scored)}

        candidate = rdir / "candidate.pt"
        release()
        if len(pairs) < 2:
            report["result"] = "skipped: fewer than 2 pairs cleared --min-margin"
        elif not train(state["champion"], prefs, candidate, state["lr"]):
            report["result"] = "skipped: DPO training failed"
        else:
            cand_scored = paint_and_judge(
                str(candidate), held_out, eval_seeds, rdir / "eval", generate=generate, judge=judge
            )
            verdict = gate(state["champion_eval"], eval_scores(cand_scored), opts.min_gain)
            report.update(verdict)
            if verdict["promote"]:
                champ = work / "champion.pt"
                if champ.is_file():
                    os.replace(champ, work / "champion.prev.pt")
                shutil.move(str(candidate), champ)
                state.update(
                    champion=str(champ),
                    champion_eval=eval_scores(cand_scored),
                    family_scores=family_means(cand_scored),
                    lr=opts.dpo_lr,
                )
                report["result"] = "promoted"
            else:
                candidate.unlink(missing_ok=True)
                state["lr"] = max(state["lr"] * 0.5, opts.dpo_lr / 64)
                report["result"] = "rejected"
        state["round"] = rnd
        state["history"].append(report)
        (rdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        save_state(work, state)
        print(f"[rsi] round {rnd}: {report['result']} {json.dumps({k: report.get(k) for k in ('gain', 'win_rate')})}")
    return state


def print_status(work: Path) -> int:
    state = load_state(work)
    if not state:
        print(f"No RSI state in {work}.")
        return 1
    print(f"champion: {state['champion']}  (base: {state['base_ckpt']})")
    print(f"family scores: {state.get('family_scores')}")
    for r in state["history"]:
        print(f"  round {r['round']:>3}: {r['result']:<10} gain={r.get('gain', '-')} win_rate={r.get('win_rate', '-')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from utils.training.feedback_bus import default_feedback_log

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", default="", help="SDX checkpoint to start from (only needed for a new work dir)")
    ap.add_argument("--work-dir", default="outputs/rsi")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--prompts", type=int, default=16, help="Training prompts per round")
    ap.add_argument("--candidates", type=int, default=4, help="Images per training prompt")
    ap.add_argument("--eval-per-family", type=int, default=3, help="Held-out prompts per medium family")
    ap.add_argument("--eval-seeds", type=int, default=2)
    ap.add_argument("--min-margin", type=float, default=0.03, help="Min judge-score gap for a self pair")
    ap.add_argument("--min-gain", type=float, default=0.005, help="Min held-out mean gain to promote")
    ap.add_argument("--steps", type=int, default=30, help="Sampling steps")
    ap.add_argument("--dpo-steps", type=int, default=200)
    ap.add_argument("--dpo-lr", type=float, default=1e-6)
    ap.add_argument("--dpo-beta", type=float, default=500.0)
    ap.add_argument("--from-feedback", default=str(default_feedback_log()), help="Human feedback log ('' = none)")
    ap.add_argument("--sample-arg", action="append", default=[], help="Extra sample.py token (repeatable)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--status", action="store_true", help="Print round history and exit")
    ap.add_argument("--dry-run", action="store_true", help="Print round 1's prompts and commands, run nothing")
    return ap


def main(argv: list[str] | None = None) -> int:
    opts = build_parser().parse_args(argv)
    if opts.status:
        return print_status(Path(opts.work_dir))
    if opts.dry_run:
        weights = curriculum_weights({})
        for fam, prompt in train_prompts(opts.prompts, weights, random.Random(opts.seed + 1)):
            print(f"[{fam}] {prompt}")
        print(f"held-out: {len(eval_prompts(opts.eval_per_family))} prompts x {opts.eval_seeds} seeds")
        sdx_cmd = f"sample.py --ckpt {opts.base_ckpt or '<champion>'} --jobs-jsonl <round>/jobs.jsonl"
        print(f"generate: {sdx_cmd} --steps {opts.steps}")
        print(f"train:    train_diffusion_dpo --steps {opts.dpo_steps} --lr {opts.dpo_lr} --dpo-beta {opts.dpo_beta}")
        return 0
    run(opts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
