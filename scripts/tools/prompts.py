#!/usr/bin/env python3
"""Prompt tooling: preview, compose, diff, research, style invention, taste and scene planning.

    python -m scripts.tools prompts preview --prompt "1girl, city street, golden hour" --json
    python -m scripts.tools prompts compose "@wlop +character: 1girl, silver hair"
    python -m scripts.tools prompts diff "a red cube" "a blue cube" --json
    python -m scripts.tools prompts research-image path/to/image.png --seed "1girl, fantasy"
    python -m scripts.tools prompts suggest-styles --query "anime 3d game keyart" --limit 8
    python -m scripts.tools prompts explore-styles --prompt "void priest" --mode apocalypse --chaos 1.0
    python -m scripts.tools prompts creative-explore --prompt "..." --mutate 4 --run -- --ckpt path/to.pt
    python -m scripts.tools prompts style-gallery outputs/ --out style_gallery.html
    python -m scripts.tools prompts taste --profile taste.json like img.png --prompt "..."
    python -m scripts.tools prompts character-session --path elena.json init --name Elena
    python -m scripts.tools prompts invention-lab --prompt "2girls" --self-heal --print-argv
    python -m scripts.tools prompts scene-compile --cast cast.json --prompt "..." --print-argv
    python -m scripts.tools prompts copilot --prompt "oil portrait" --dry-run

Old names (``preview_prompt_stack``, ``preview_generation_prompt``, ``prompt_compose``, ``prompt_diff``,
``research_image_prompt``, ``suggest_style_packs``, ``explore_styles``, ``creative_explore``, ``style_gallery``,
``taste_profile``, ``character_session``, ``invention_lab``, ``scene_compile``, ``creative_copilot``) still work.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "native" / "_experimental" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_PROG = "python -m scripts.tools prompts"


def _parser(cmd: str, description: str | None) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=f"{_PROG} {cmd}", description=description)


def _preview_main(argv: list[str]) -> int:
    """Preview the full PromptStack (guidance, controls, clauses, neg filter) without a GPU.

    Set ``SDX_PROMPT_STACK_TRACE=1`` to also print the resolved controls.
    """
    from config import DEFAULT_NEGATIVE_PROMPT
    from utils.prompt.art_direction import apply_art_direction_to_args
    from utils.prompt.stack import PromptContext, StackMode, run_prompt_stack

    p = _parser("preview", "Preview SDX PromptStack output (pos/neg + trace).")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--quality-pack", default="none")
    p.add_argument("--adherence-pack", default="none")
    p.add_argument("--auto-content-fix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prompt-clauses", default="", help="Comma-separated clause names (see utils.prompt.stack.clauses)")
    p.add_argument("--no-prompt-stack-intelligence", action="store_true")
    p.add_argument("--prompt-special-helpers", default="auto")
    p.add_argument("--less-ai", action="store_true")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human text")
    args = p.parse_args(argv)

    auto = "auto" if args.less_ai else "none"
    ns = SimpleNamespace(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        shortcomings_mitigation=auto,
        shortcomings_2d=False,
        art_guidance_mode=auto,
        anatomy_guidance="none",
        style_guidance_mode=auto,
        style_guidance_artists=True,
        no_art_guidance_photography=False,
        auto_photo_realism=True,
        photo_realism_pack="none",
        photo_color_grade="none",
        photo_lighting_technique="none",
        photo_filter="none",
        photo_grain_style="none",
        photo_realism_strength=1.0,
        realism_autopilot=False,
        quality_pack=args.quality_pack,
        adherence_pack=args.adherence_pack,
        auto_content_fix=args.auto_content_fix,
        prompt_clauses=args.prompt_clauses,
        prompt_stack_intelligence=not args.no_prompt_stack_intelligence,
        prompt_special_helpers=args.prompt_special_helpers,
        prompt_stack_auto_quality=True,
        one_shot_boost=True,
        less_ai=args.less_ai,
        lora=[],
        text_in_image=False,
        no_neg_filter=False,
        scene_domain="none",
        view_angle="none",
        pose_mode="none",
        style_mode="none",
        clothing_mode="none",
        composition_mode="none",
        artist_composition="none",
        people_layout="none",
        hand_mode="none",
        lighting_mode="none",
        anti_ai_pack="lite" if args.less_ai else "none",
        human_media_mode="photographic" if args.less_ai else "none",
        lora_scaffold="none",
        lora_scaffold_auto=False,
    )
    ns._user_prompt, ns._user_negative = args.prompt, args.negative_prompt
    apply_art_direction_to_args(ns)
    result = run_prompt_stack(PromptContext(positive=args.prompt, mode=StackMode.PREVIEW, args=ns))

    if args.json:
        payload = {
            "positive": result.positive,
            "negative": result.negative,
            "trace": result.trace,
            "analysis": result.analysis,
            "resolved_controls": result.resolved_controls,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print("=== POSITIVE ===")
    print(result.positive)
    print("\n=== NEGATIVE ===")
    print(result.negative)
    if result.analysis:
        print("\n=== ANALYSIS ===")
        print(f"complexity: {result.analysis.get('complexity')}  domains: {result.analysis.get('domains')}")
    print("\n=== TRACE ===")
    print(" -> ".join(result.trace))
    if os.environ.get("SDX_PROMPT_STACK_TRACE", "").strip() and result.resolved_controls:
        print("\n=== CONTROLS ===")
        for k, v in sorted(result.resolved_controls.items()):
            if v and v != "none":
                print(f"  {k}: {v}")
    return 0


def _compose_main(argv: list[str]) -> int:
    from utils.prompt.text import compose_prompt

    p = _parser("compose", "Compose @artist / +category prompts for SDX.")
    p.add_argument("prompt", help="Prompt with @artist and/or +category blocks.")
    p.add_argument("--artist-strength", type=float, default=1.0)
    p.add_argument("--artist-index", default="", help="artist_index.json path")
    args = p.parse_args(argv)
    cp = compose_prompt(
        args.prompt, artist_strength=args.artist_strength, artist_index=args.artist_index.strip() or None
    )
    print(cp.positive)
    if cp.artists:
        print(f"# artists: {', '.join(cp.artists)}", file=sys.stderr)
    return 0


def diff_prompts(a: str, b: str) -> dict[str, list[str]]:
    """Subject / negation set differences between two prompts (``PromptParser``)."""
    from models.conditioning import PromptParser

    parser = PromptParser()

    def parts(prompt: str) -> tuple[set[str], set[str]]:
        parsed = parser.parse(prompt)
        subj = {getattr(s, "text", str(s)) for s in getattr(parsed, "subjects", []) or []}
        neg = {getattr(n, "text", str(n)) for n in getattr(parsed, "negations", []) or []}
        return subj, neg

    (sa, na), (sb, nb) = parts(a), parts(b)
    return {
        "only_in_a": sorted(sa - sb),
        "only_in_b": sorted(sb - sa),
        "shared_subjects": sorted(sa & sb),
        "negations_only_in_a": sorted(na - nb),
        "negations_only_in_b": sorted(nb - na),
    }


def _diff_main(argv: list[str]) -> int:
    p = _parser("diff", "Semantic diff between two prompts (subjects, negations, layout tokens).")
    p.add_argument("prompt_a", type=str)
    p.add_argument("prompt_b", type=str, nargs="?", default="")
    p.add_argument("--file-b", type=str, default="", help="Read second prompt from file")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    pb = Path(args.file_b).read_text(encoding="utf-8").strip() if args.file_b else args.prompt_b
    result = diff_prompts(args.prompt_a, pb)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    for key, val in result.items():
        if val:
            print(f"{key}:")
            for line in val:
                print(f"  - {line}")
    return 0


def _research_image_main(argv: list[str]) -> int:
    from utils.captioning import research_prompt_for_image

    data = os.environ.get("SDX_DATA", r"D:\Development\sdx-data")
    default_corpus = os.environ.get("SDX_RAG_CORPUS", str(Path(data) / "rag_corpus.jsonl"))

    p = _parser(
        "research-image", "Research diffusion prompt via uncensored VLM + local RAG (recommended over SauceNAO)."
    )
    p.add_argument("image", help="Image path")
    p.add_argument("--rag-corpus", default=default_corpus, help="rag_corpus.jsonl from build_rag_corpus.py")
    p.add_argument("--seed", default="", help="Optional intent seed merged with VLM description")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--creativity", type=float, default=0.45)
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--no-creative-rag", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    result = research_prompt_for_image(
        args.image,
        rag_corpus=args.rag_corpus if Path(args.rag_corpus).is_file() else None,
        seed_prompt=args.seed,
        creativity_level=args.creativity,
        top_k=args.top_k,
        device=args.device,
        use_rag=not args.no_rag,
        use_creative_rag=not args.no_creative_rag,
    )
    if args.json:
        keys = (
            "diffusion_prompt",
            "negative_prompt",
            "image_description",
            "scene_summary",
            "retrieved_facts",
            "reasoning",
            "sources",
            "fallback_used",
        )
        print(json.dumps({k: getattr(result, k) for k in keys}, ensure_ascii=False, indent=2))
        return 0
    print("diffusion_prompt:", result.diffusion_prompt)
    print("negative_prompt:", result.negative_prompt)
    print("image_description:", result.image_description[:500])
    if result.retrieved_facts:
        print("rag_facts:", result.retrieved_facts[:3])
    print("reasoning:", result.reasoning[:400])
    print("sources:", result.sources)
    print("fallback:", result.fallback_used)
    return 0


_STYLE_FLAGS = (
    ("lexicon_style", "--lexicon-style"),
    ("art_medium_pack", "--art-medium-pack"),
    ("art_medium_family", "--art-medium-family"),
    ("art_medium_variant", "--art-medium-variant"),
    ("artist_pack", "--artist-pack"),
    ("color_render_pack", "--color-render-pack"),
)


def _suggest_styles_main(argv: list[str]) -> int:
    p = _parser("suggest-styles", "Suggest style preset packs from free-text.")
    p.add_argument("--query", type=str, default="", help="Free-text style query, e.g. 'anime 3d game keyart'.")
    p.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "digital", "3d", "drawing", "painting"],
        help="Optional category filter.",
    )
    p.add_argument("--limit", type=int, default=10, help="Max rows to return.")
    p.add_argument("--json", action="store_true", help="Emit JSON rows.")
    args = p.parse_args(argv)

    from pipelines.book_comic.prompt_lexicon import suggest_popular_style_presets

    rows = suggest_popular_style_presets(
        str(args.query or ""), category=str(args.category or "all"), limit=max(1, int(args.limit))
    )
    compact: list[dict[str, str]] = []
    for row in rows:
        out = {k: str(row.get(k, d)) for k, d in (("id", ""), ("category", ""), ("score", "0"), ("description", ""))}
        out.update({k: str(row[k]).strip() for k, _ in _STYLE_FLAGS if str(row.get(k, "")).strip()})
        compact.append(out)
    if args.json:
        print(json.dumps(compact, indent=2, ensure_ascii=False))
        return 0
    if not compact:
        print("No style presets matched your query.")
        return 0

    print(f"Query: {args.query!r} | category={args.category} | top={len(compact)}")
    print("")
    for i, row in enumerate(compact, start=1):
        cmd = " ".join(f"{flag} {row[k]}" for k, flag in _STYLE_FLAGS if row.get(k))
        print(f"{i:02d}. {row['id']}  [{row['category']}]  score={row['score']}")
        print(f"    {row['description']}")
        if cmd:
            print(f"    {cmd}")
        print("")
    return 0


def _explore_styles_main(argv: list[str]) -> int:
    p = _parser("explore-styles", "Invent StyleGenomes and build explore manifests.")
    p.add_argument("--prompt", type=str, default="", help="Base scene/subject prompt")
    p.add_argument("--negative", type=str, default="", help="Base negative prompt")
    p.add_argument("--genomes", type=int, default=3, help="Number of invented genomes")
    p.add_argument(
        "--mutations", type=int, default=0, help="Extra prompt mutations per genome (utils.prompt.prompt_mutation)"
    )
    p.add_argument("--creativity", type=float, default=0.75, help="0–1 invention novelty")
    p.add_argument("--chaos", type=float, default=0.0, help="0–1 chaos spice (wild axis injection)")
    p.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=("normal", "insane", "apocalypse", "chimera", "glitch", "eldritch", "cyberpunk"),
        help="Invention mode",
    )
    p.add_argument("--preset", type=str, default="", help="Named insane preset (see --list-presets)")
    p.add_argument("--fusion", action="store_true", help="Add chimera fusion candidates")
    p.add_argument("--hypermutate", type=int, default=0, metavar="N", help="Add N hypermutated siblings per genome")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-qwen", action="store_true", help="Deterministic fallback inventor only")
    p.add_argument(
        "--out",
        type=str,
        default="data/style_genomes/explore_manifest.jsonl",
        help="JSONL manifest (caption, negative, style, genome metadata)",
    )
    p.add_argument("--genomes-json", type=str, default="", help="Optional path to write full genome JSON array")
    p.add_argument(
        "--preview-stack", action="store_true", help="Run PromptStack preview on first candidate (no image generation)"
    )
    p.add_argument("--list-presets", action="store_true", help="Print insane preset ids and exit")
    p.add_argument(
        "--insane",
        action="store_true",
        help="Shorthand: --mode apocalypse --chaos 0.95 --genomes 4 --fusion --hypermutate 1",
    )
    p.add_argument(
        "--native-stats", action="store_true", help="After write, print Go explore-stats (requires built sdx-manifest)."
    )
    p.add_argument(
        "--dedupe-out", type=str, default="", help="Dedupe manifest by --dedupe-key via Go (requires sdx-manifest)."
    )
    p.add_argument("--dedupe-key", type=str, default="style_genome_id", help="Field for --dedupe-out")
    p.add_argument(
        "--show-native", action="store_true", help="Print Rust/CUDA/Go/Mojo style native discovery JSON and exit."
    )
    args = p.parse_args(argv)

    if args.show_native:
        from utils.prompt.style import style_native_status

        print(json.dumps(style_native_status(), indent=2))
        return 0
    if args.list_presets:
        from utils.prompt.style import list_insane_presets

        for pid in list_insane_presets():
            print(pid)
        return 0
    if not (args.prompt or "").strip():
        p.error("--prompt is required (unless --list-presets)")
    if args.insane:
        args.mode = "apocalypse"
        args.chaos = max(args.chaos, 0.95)
        args.genomes = max(args.genomes, 4)
        args.fusion = True
        args.hypermutate = max(args.hypermutate, 1)

    from utils.prompt.style import StyleExplorePlanner

    plan = StyleExplorePlanner(device="cpu", use_qwen=not args.no_qwen).plan(
        args.prompt,
        n_genomes=max(1, args.genomes),
        mutations_per_genome=max(0, args.mutations),
        base_negative=args.negative,
        creativity_level=max(0.0, min(1.0, args.creativity)),
        seed=args.seed,
        invention_mode=args.mode,  # type: ignore[arg-type]
        chaos_level=max(0.0, min(1.0, args.chaos)),
        fusion_pairs=bool(args.fusion),
        hypermutate_siblings=max(0, args.hypermutate),
        preset=args.preset,
    )
    out_path = Path(args.out)
    plan.write_manifest(out_path)
    print(f"Wrote {len(plan.candidates)} candidates -> {out_path} [{args.mode} chaos={args.chaos:.2f}]")

    if args.native_stats:
        from utils.prompt.style import run_go_explore_stats

        stats = run_go_explore_stats(out_path)
        if stats:
            print(stats)
        else:
            print("Go explore-stats unavailable (build native/go/sdx-manifest)", file=sys.stderr)
    if args.dedupe_out:
        from utils.prompt.style import run_go_explore_dedupe

        if run_go_explore_dedupe(out_path, Path(args.dedupe_out), key=args.dedupe_key):
            print(f"Deduped -> {args.dedupe_out}")
        else:
            print("Go explore-dedupe failed or sdx-manifest not built", file=sys.stderr)
    if args.genomes_json:
        plan.write_genomes_json(Path(args.genomes_json))
        print(f"Wrote genomes -> {args.genomes_json}")

    for i, cand in enumerate(plan.candidates[:12]):
        print(
            f"\n[{i}] {cand.genome.name} ({cand.candidate_kind}, overlap={cand.catalog_overlap:.2f}) "
            f"mut={cand.mutation_strategy or '-'}"
        )
        print(f"  style: {cand.style[:110]}")
        print(f"  pos:   {cand.positive[:130]}...")

    if args.preview_stack and plan.candidates:
        from utils.prompt.stack import PromptContext, StackMode, run_prompt_stack

        c0 = plan.candidates[0]
        ns = SimpleNamespace(
            prompt=c0.positive,
            negative_prompt=c0.negative,
            style=c0.style,
            invent_styles=0,
            style_genome_file="",
            style_genome_enabled=True,
            _active_style_genome=c0.genome,
            style_chaos_level=args.chaos,
            shortcomings_mitigation="none",
            art_guidance_mode="none",
            anatomy_guidance="none",
            style_guidance_mode="none",
            auto_content_fix=False,
            prompt_stack_intelligence=True,
            prompt_stack_auto_quality=True,
            prompt_clauses="style.chaos,style.surreal" if args.chaos > 0.5 else "",
            prompt_special_helpers="off",
            one_shot_boost=False,
            text_in_image=False,
            no_neg_filter=False,
        )
        result = run_prompt_stack(
            PromptContext(positive=c0.positive, negative=c0.negative, mode=StackMode.PREVIEW, args=ns)
        )
        print("\nPreview stack trace:", " -> ".join(result.trace))
        print("Preview positive:", result.positive[:220])
    return 0


def _creative_explore_main(argv: list[str]) -> int:
    p = _parser("creative-explore", "Preview creative mutations or run creative auto-refine via sample.py.")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--mutate", type=int, default=6, help="Number of prompt variants to show or run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-constraint", action="store_true")
    p.add_argument("--json", action="store_true", help="Print imagination plan as JSON")
    p.add_argument("--run", action="store_true", help="Run sample.py with creative auto-refine")
    p.add_argument("--out", type=str, default="creative_best.png")
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("sample_args", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)

    from frontier.imagination import analyze_imagination

    plan = analyze_imagination(
        args.prompt,
        mutate_count=args.mutate,
        mutate_seed=args.seed,
        random_constraint_seed=args.seed if args.random_constraint else None,
    )
    if args.json:
        keys = ("augmented_prompt", "serendipity_dial", "cfg_multiplier", "creative_trace", "mutations")
        print(json.dumps({k: getattr(plan, k) for k in keys}, indent=2))
    else:
        print(f"Augmented: {plan.augmented_prompt[:200]}...")
        print(f"Serendipity: {plan.serendipity_dial:.3f}  CFG mult: {plan.cfg_multiplier:.3f}")
        print(f"Trace: {', '.join(plan.creative_trace)}")
        print("Mutations:")
        for i, m in enumerate(plan.mutations):
            print(f"  [{i}] {m}")
    if not args.run:
        return 0

    from utils.generation.sample_prompt import run_creative_refine

    sample_args = list(args.sample_args)
    if sample_args and sample_args[0] == "--":
        sample_args = sample_args[1:]
    cmd = [sys.executable, str(_REPO / "sample.py"), "--frontier-creative", "--prompt", args.prompt, *sample_args]
    res = run_creative_refine(
        sample_cmd=cmd,
        base_prompt=args.prompt,
        num_candidates=args.candidates,
        seed_base=args.seed,
        out_stem="creative_explore",
        mutate_count=args.mutate,
    )
    shutil.copy(res.outputs[res.best_index], args.out)
    print(f"Best {res.best_index} score={res.scores[res.best_index]:.3f} -> {args.out}")
    if res.prompts:
        print(f"Winning prompt: {res.prompts[res.best_index]}")
    return 0


_GALLERY_CSS = """
body { font-family: system-ui, sans-serif; background: #0d0d12; color: #e8e8f0; margin: 0; padding: 24px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }
.card { background: #16161f; border-radius: 12px; overflow: hidden; border: 1px solid #2a2a38; }
.card img { width: 100%; display: block; aspect-ratio: 1; object-fit: cover; }
.prompt { font-size: 13px; padding: 12px; margin: 0; line-height: 1.4; }
.genome { font-size: 11px; padding: 0 12px 12px; margin: 0; color: #9aa0b5; white-space: pre-wrap; }
h1 { font-weight: 600; letter-spacing: -0.02em; }
"""


def render_gallery_html(rows: list[dict[str, Any]], *, title: str = "SDX Style Gallery") -> str:
    cards = "\n".join(
        f'<article class="card"><img src="{html.escape(r["path"])}" loading="lazy" />'
        f'<p class="prompt">{html.escape(r.get("prompt", ""))}</p>'
        f'<pre class="genome">{html.escape(r.get("genome", ""))}</pre></article>'
        for r in rows
    )
    t = html.escape(title)
    return (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8" />\n<title>{t}</title>\n'
        f'<style>{_GALLERY_CSS}</style>\n</head>\n<body>\n<h1>{t}</h1>\n<div class="grid">\n{cards}\n</div>\n'
        "</body>\n</html>"
    )


def _style_gallery_main(argv: list[str]) -> int:
    p = _parser("style-gallery", "Export Style Genome runs as a browsable HTML gallery.")
    p.add_argument("images_dir", type=Path, help="Directory of PNG outputs (optional sidecar JSON)")
    p.add_argument("--out", type=Path, default=Path("style_gallery.html"))
    p.add_argument("--title", type=str, default="SDX Style Gallery")
    args = p.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for img in sorted(args.images_dir.rglob("*.png")):
        meta = img.with_suffix(".json")
        prompt = genome = ""
        if meta.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                prompt = str(data.get("prompt", "") or "")
                genome = str(data.get("style_genome", "") or data.get("genome", "") or "")
            except (OSError, ValueError, AttributeError):
                pass
        rows.append({"path": str(img), "prompt": prompt, "genome": genome})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_gallery_html(rows, title=args.title), encoding="utf-8")
    print(f"Wrote {len(rows)} cards -> {args.out}")
    return 0


def export_taste_pairs(data: dict[str, Any], *, min_prompt_match: bool = True) -> list[dict[str, str]]:
    """Pair each like against each dislike (same prompt unless ``min_prompt_match`` is off)."""
    pairs: list[dict[str, str]] = []
    for win in data.get("likes") or []:
        wp = str(win.get("prompt", "") or "")
        for lose in data.get("dislikes") or []:
            lp = str(lose.get("prompt", "") or "")
            if min_prompt_match and wp and lp and wp != lp:
                continue
            pairs.append(
                {
                    "win_image_path": str(win.get("image", "")),
                    "lose_image_path": str(lose.get("image", "")),
                    "caption": wp or lp,
                }
            )
    return pairs


def _taste_main(argv: list[str]) -> int:
    p = _parser("taste", "Manage taste profiles (likes/dislikes) and export DPO preference pairs.")
    p.add_argument("--profile", type=Path, default=Path("taste_profile.json"))
    sub = p.add_subparsers(dest="cmd", required=True)
    like = sub.add_parser("like", help="Record a liked image")
    like.add_argument("image", type=str)
    like.add_argument("--prompt", type=str, default="")
    like.add_argument("--score", type=float, default=1.0)
    dislike = sub.add_parser("dislike", help="Record a disliked image")
    dislike.add_argument("image", type=str)
    dislike.add_argument("--prompt", type=str, default="")
    sub.add_parser("show", help="Print profile JSON")
    exp = sub.add_parser("export-dpo", help="Write preference pairs JSONL")
    exp.add_argument("--out", type=Path, default=Path("taste_pairs.jsonl"))
    exp.add_argument("--allow-cross-prompt", action="store_true")
    args = p.parse_args(argv)

    prof: dict[str, Any] = {"likes": [], "dislikes": [], "notes": ""}
    if args.profile.is_file():
        prof = json.loads(args.profile.read_text(encoding="utf-8"))
        if not isinstance(prof, dict):
            raise ValueError("profile must be a JSON object")
        prof.setdefault("likes", [])
        prof.setdefault("dislikes", [])

    if args.cmd in ("like", "dislike"):
        row = {"image": args.image, "prompt": args.prompt, "score": float(getattr(args, "score", 1.0))}
        prof[f"{args.cmd}s"].append(row)
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        args.profile.write_text(json.dumps(prof, indent=2), encoding="utf-8")
        try:  # best-effort mirror into the shared preference store
            from utils.training import preference

            getattr(preference, f"record_{args.cmd}")(args.image, prompt=args.prompt)
        except Exception:
            pass
        print(f"Added {args.cmd} -> {args.profile}")
    elif args.cmd == "show":
        print(json.dumps(prof, indent=2))
    else:
        pairs = export_taste_pairs(prof, min_prompt_match=not args.allow_cross_prompt)
        lines = [json.dumps(r) for r in pairs if r["win_image_path"] and r["lose_image_path"]]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"Wrote {len(lines)} pairs -> {args.out}")
    return 0


def _character_session_main(argv: list[str]) -> int:
    from utils.generation.sample_prompt import load_character_session, save_character_session

    p = _parser(
        "character-session", "Create and inspect character session lock JSON for sample.py --character-session."
    )
    p.add_argument("--path", type=Path, default=Path("character_session.json"))
    sub = p.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init", help="Create empty session file")
    init.add_argument("--name", type=str, default="character")
    setp = sub.add_parser("set", help="Set a session field")
    setp.add_argument("field", choices=["prompt_additions", "negative_prompt", "reference_images"])
    setp.add_argument("value", type=str)
    sub.add_parser("show", help="Print session JSON")
    args = p.parse_args(argv)
    path = args.path

    if args.cmd == "init":
        empty = {"name": args.name, "prompt_additions": "", "negative_prompt": "", "reference_images": []}
        save_character_session(path, empty)
        print(f"Created {path}")
        return 0
    data = load_character_session(path) if path.is_file() else {}
    if args.cmd == "set":
        if args.field == "reference_images":
            data[args.field] = [v.strip() for v in args.value.split(",") if v.strip()]
        else:
            data[args.field] = args.value
        save_character_session(path, data)
        print(f"Updated {args.field} in {path}")
    else:
        print(json.dumps(data, indent=2))
    return 0


def _print_argv(argv: list[str]) -> None:
    print("sample argv:")
    print(" ", " ".join(shlex.quote(a) for a in argv))


def _invention_lab_main(argv: list[str]) -> int:
    p = _parser("invention-lab", "SDX Invention Lab: diagnose prompt failure risks and apply the invention passes.")
    p.add_argument("--prompt", default="")
    p.add_argument("--negative", default="")
    p.add_argument("--enable", default="auto", help="auto|all|off|bindlock,negatron,...")
    p.add_argument("--work-dir", default="outputs/inventions")
    p.add_argument("--diagnose-only", action="store_true")
    p.add_argument("--self-heal", action="store_true", help="Write the self-healing plan (oracle -> stack -> APEX)")
    p.add_argument("--atlas-out", default="", help="Write the failure-atlas eval prompts as JSONL")
    p.add_argument("--print-argv", action="store_true")
    args = p.parse_args(argv)

    from utils.generation.inventions import (
        apply_invention_stack,
        diagnose_failures,
        plan_self_healing,
        repair_plan_from_failures,
        save_failure_atlas_jsonl,
        write_self_healing_plan,
    )

    if args.atlas_out:
        print(f"FailureAtlas -> {save_failure_atlas_jsonl(args.atlas_out)}")
        return 0
    if not args.prompt:
        p.error("--prompt is required unless --atlas-out")

    if args.self_heal:
        plan = plan_self_healing(args.prompt, args.negative)
        path = write_self_healing_plan(plan, args.work_dir)
        print(json.dumps(plan.to_dict(), indent=2)[:2000])
        print(f"wrote {path}")
        if args.print_argv:
            _print_argv(plan.sample_argv)
        return 0
    if args.diagnose_only:
        rep = repair_plan_from_failures(diagnose_failures(args.prompt, negative=args.negative))
        print(json.dumps(rep.to_dict(), indent=2))
        return 0

    res = apply_invention_stack(args.prompt, args.negative, enable=args.enable, work_dir=args.work_dir)
    print(f"positive: {res.positive[:240]}...")
    print(f"negative: {res.negative[:200]}...")
    print(f"risks: {list((res.reports.get('oracle') or {}).get('risks', []))}")
    print(f"wrote: {Path(args.work_dir) / 'invention_stack.json'}")
    if args.print_argv:
        argv_list = ["--prompt", res.positive]
        if res.negative:
            argv_list.extend(["--negative-prompt", res.negative])
        if res.box_layout:
            argv_list.extend(["--box-layout", str(Path(args.work_dir) / "invention_box_layout.json"), "--anti-bleed"])
        _print_argv(argv_list)
    return 0


def _scene_compile_main(argv: list[str]) -> int:
    p = _parser("scene-compile", "Compile multi-character cast / Ideogram layout / edit skills to sample argv.")
    p.add_argument("--cast", default="", help="Multi-character cast JSON")
    p.add_argument("--prompt", default="")
    p.add_argument("--ideogram", action="store_true", help="Compile Ideogram-style text/layout")
    p.add_argument("--glyph-text", action="append", default=[])
    p.add_argument("--palette", default="", help="Comma-separated hex colors")
    p.add_argument("--edit-image", default="", help="Init image for photoshop-like edit plan")
    p.add_argument("--fix-region", action="append", default=[])
    p.add_argument("--change-outfit", default="")
    p.add_argument("--work-dir", default="outputs/scene_compile")
    p.add_argument("--print-argv", action="store_true")
    p.add_argument("--json-out", default="")
    args = p.parse_args(argv)

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}

    if args.cast:
        from utils.generation.inventions import compile_cast_scene, load_cast_scene

        scene = load_cast_scene(args.cast)
        compiled = compile_cast_scene(scene, base_prompt=args.prompt)
        out["cast"] = compiled.to_dict()
        box_path = work / "cast_box_layout.json"
        argv_list = ["--prompt", compiled.positive]
        if compiled.negative:
            argv_list.extend(["--negative-prompt", compiled.negative])
        if compiled.box_layout:
            box_path.write_text(json.dumps(compiled.box_layout, indent=2), encoding="utf-8")
            out["box_layout_path"] = str(box_path)
            argv_list.extend(["--box-layout", str(box_path), "--anti-bleed"])
        if compiled.sheet_paths:
            argv_list.extend(["--character-sheet", ",".join(compiled.sheet_paths), "--label-multi-character-sheets"])
        for fr in compiled.face_refs[:1]:
            argv_list.extend(["--reference-image", fr])
        out["sample_argv"] = argv_list
        print(f"cast: {len(scene.members)} characters")
        print(f"prompt: {compiled.positive[:200]}...")

    if args.ideogram or args.glyph_text:
        from utils.generation.editing import plan_ideogram_layout

        ideo = plan_ideogram_layout(
            args.prompt or (out.get("cast") or {}).get("positive", ""),
            texts=list(args.glyph_text or None),
            palette_hex=[c.strip() for c in args.palette.split(",") if c.strip()],
        )
        brief_path = work / "design_brief.json"
        brief_path.write_text(json.dumps(ideo.design_brief, indent=2), encoding="utf-8")
        out["ideogram"] = ideo.to_dict()
        out["design_brief_path"] = str(brief_path)
        base_argv = list(out.get("sample_argv") or ideo.sample_argv)
        if ideo.box_layout and "--box-layout" not in base_argv:
            glyph_box = work / "glyph_box_layout.json"
            glyph_box.write_text(json.dumps(ideo.box_layout, indent=2), encoding="utf-8")
            base_argv.extend(["--box-layout", str(glyph_box)])
        out["sample_argv"] = base_argv

    if args.edit_image:
        from utils.generation.editing import plan_edit_skills

        eplan = plan_edit_skills(
            init_image=args.edit_image,
            prompt=args.prompt,
            fix_regions=list(args.fix_region or []),
            change_outfit=args.change_outfit,
            work_dir=work / "edit",
        )
        out["edit"] = eplan.to_dict()
        if eplan.sample_argv:
            out["sample_argv"] = eplan.sample_argv

    json_path = Path(args.json_out or (work / "compile.json"))
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"plan -> {json_path}")
    if args.print_argv and out.get("sample_argv"):
        _print_argv(out["sample_argv"])
    return 0


def _copilot_main(argv: list[str]) -> int:
    """Creative Co-Pilot — decompose and apply (inspiration → style → control → moodboard)."""
    from utils.generation.planners import CopilotConfig, format_plan_summary, run_creative_copilot

    ap = _parser("copilot", "Creative Co-Pilot: inspiration -> style -> control -> moodboard -> sample.py.")
    ap.add_argument("--prompt", required=True, help="Creative goal / subject prompt.")
    ap.add_argument("--ckpt", default="", help="DiT checkpoint for fusion (sample.py).")
    ap.add_argument("--out", default="copilot_out.png")
    ap.add_argument("--work-dir", default="creative_copilot_run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reference-images", default="", help="Comma-separated local style references (moodboard).")
    ap.add_argument(
        "--structure-image",
        default="",
        help="Optional layout/pose/depth source for ControlNet map (defaults to first moodboard).",
    )
    ap.add_argument("--web-search", action="store_true", help="Search the web for style references.")
    ap.add_argument("--no-web-search", action="store_true")
    ap.add_argument("--no-vlm", action="store_true", help="Skip VLM style captions (heuristic only).")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument(
        "--control-prefer",
        default="canny",
        choices=["canny", "depth", "softedge", "hed"],
        help="Preferred control map (depth needs local Depth-Anything / Marigold).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Plan only; no search download / VLM / sample.")
    ap.add_argument("--execute", action="store_true", help="Run sample.py after planning (requires --ckpt).")
    ap.add_argument(
        "--extra-sample-arg",
        action="append",
        default=[],
        help="Extra argv passed to sample.py (repeatable), e.g. --extra-sample-arg --steps=28",
    )
    args = ap.parse_args(argv)

    cfg = CopilotConfig(
        web_search=bool(args.web_search) and not bool(args.no_web_search),
        use_vlm=not bool(args.no_vlm),
        extract_control=not bool(args.no_control),
        control_prefer=str(args.control_prefer),
        device=str(args.device),
    )
    plan = run_creative_copilot(
        goal=str(args.prompt),
        ckpt=str(args.ckpt or ""),
        work_dir=args.work_dir,
        out=args.out,
        reference_images=[x.strip() for x in str(args.reference_images or "").split(",") if x.strip()],
        structure_image=str(args.structure_image or ""),
        config=cfg,
        dry_run=bool(args.dry_run),
        execute=bool(args.execute),
        extra_sample_args=list(args.extra_sample_arg or []),
    )
    print(format_plan_summary(plan))
    if plan.work_dir and not args.dry_run:
        print(f"plan written: {Path(plan.work_dir) / 'copilot_plan.json'}")
    return 0


COMMANDS = {
    "preview": _preview_main,
    "compose": _compose_main,
    "diff": _diff_main,
    "research-image": _research_image_main,
    "suggest-styles": _suggest_styles_main,
    "explore-styles": _explore_styles_main,
    "creative-explore": _creative_explore_main,
    "style-gallery": _style_gallery_main,
    "taste": _taste_main,
    "character-session": _character_session_main,
    "invention-lab": _invention_lab_main,
    "scene-compile": _scene_compile_main,
    "copilot": _copilot_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("Commands: " + ", ".join(COMMANDS))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return int(COMMANDS[argv[0]](argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
