"""
**VisualBrain** — full understand → plan → generate → edit → verify loop.

1. **Search** — optional web image search (DuckDuckGo / Wikimedia).
2. **Understand** — OCR + VLM + control maps on all references.
3. **Dissect** — parse prompt for parts; segment with GDINO/SAM when available.
4. **Brief** — scene synthesis merging user prompt + facts (never drop user intent).
5. **Generate** — ``sample.py`` with creative RAG, dissection init, ControlNet.
6. **Verify** — composite + CLIP + OCR when text expected.
7. **Edit** — OCR-fix / inpaint retry until coverage threshold or max loops.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from config.defaults.agentic_stack import AgenticStackDefaults
from PIL import Image

from .image_search import download_search_hits, search_reference_images
from .scene_brief import SceneBrief, prompt_coverage_score, synthesize_scene_brief
from .understand import understand_images


@dataclass(slots=True)
class VisualBrainConfig:
    web_search: bool = True
    max_search_images: int = 4
    understand_ocr: bool = True
    understand_vlm: bool = True
    extract_control: bool = True
    dissect_heavy: bool = True
    creative_rag: bool = True
    creative_rag_level: float = 0.65
    max_edit_loops: int = 3
    min_coverage: float = 0.62
    min_clip: float = 0.22
    num_candidates: int = 4
    pick_metric: str = "superior_composite"
    self_correct: bool = True


@dataclass(slots=True)
class VisualBrainResult:
    accepted: bool
    out_path: str
    brief: SceneBrief
    reference_paths: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    trace_path: str = ""


class VisualBrain:
    """Coordinator for visual understanding + iterative generation."""

    def __init__(
        self,
        *,
        config: Optional[VisualBrainConfig] = None,
        defaults: Optional[AgenticStackDefaults] = None,
        repo_root: Optional[Path] = None,
    ) -> None:
        self.config = config or VisualBrainConfig()
        self.defaults = defaults or AgenticStackDefaults()
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2])

    def run(
        self,
        *,
        ckpt: str,
        prompt: str,
        work_dir: str = "visual_brain_run",
        out: str = "brain_out.png",
        device: str = "cuda",
        negative_prompt: str = "",
        reference_images: Optional[Sequence[str]] = None,
        local_rag_jsonl: str = "",
        expected_text: str = "",
        vit_ckpt: str = "",
        dry_run: bool = False,
    ) -> VisualBrainResult:
        cfg = self.config
        work = Path(work_dir)
        if not dry_run:
            work.mkdir(parents=True, exist_ok=True)

        user_prompt = str(prompt).strip()
        trace: Dict[str, Any] = {"goal": user_prompt, "steps": []}

        # --- 1. Gather references (user + web search) ---
        ref_paths: List[str] = [str(p) for p in (reference_images or []) if str(p).strip()]
        search_dir = work / "search"
        if cfg.web_search and user_prompt and not dry_run:
            search_q = user_prompt[:120]
            sr = search_reference_images(search_q, max_results=cfg.max_search_images, allow_web=True)
            trace["steps"].append({"search": {"query": search_q, "hits": len(sr.hits), "notes": sr.notes}})
            if sr.hits:
                downloaded = download_search_hits(sr.hits, search_dir, max_download=cfg.max_search_images)
                for h in downloaded:
                    if h.local_path:
                        ref_paths.append(h.local_path)

        # --- 2. Understand all references ---
        understandings = []
        if ref_paths and not dry_run:
            understandings = understand_images(
                ref_paths,
                user_prompt=user_prompt,
                work_dir=work,
                sources=["user" if i < len(reference_images or []) else "web" for i in range(len(ref_paths))],
                run_ocr=cfg.understand_ocr,
                run_vlm=cfg.understand_vlm,
                run_control=cfg.extract_control,
                device=device,
            )
            trace["steps"].append(
                {
                    "understand": [
                        {"path": u.path, "caption": u.caption[:200], "ocr": u.ocr_text[:80]} for u in understandings
                    ]
                }
            )

        # --- 3. Dissect parts from prompt ---
        dissection_facts: List[str] = []
        init_image = ""
        inpaint_mask = ""
        dissect_refs_arg = ""
        if ref_paths:
            dissect_refs_arg = ",".join(ref_paths)
            try:
                from utils.generation.image_dissection import dissect_images_to_parts

                _reqs, _parts, dissection_facts = dissect_images_to_parts(
                    user_prompt,
                    ref_paths,
                    output_dir=work / "dissection",
                    enable_heavy_models=cfg.dissect_heavy,
                )
                if _parts and not dry_run:
                    from utils.generation.part_compositing import build_init_and_inpaint_mask

                    spec = build_init_and_inpaint_mask(
                        reference_images=ref_paths,
                        parts=_parts,
                        output_dir=work / "composite",
                        target_size=(512, 512),
                    )
                    init_image = spec.init_image_path
                    inpaint_mask = spec.mask_path
                    trace["steps"].append({"dissect": {"parts": len(_parts), "init": init_image}})
            except Exception as exc:
                trace["steps"].append({"dissect_error": str(exc)})

        # --- 4. Local RAG facts ---
        rag_facts: List[str] = []
        if local_rag_jsonl and Path(local_rag_jsonl).is_file():
            try:
                from utils.superior.retrieval import build_tfidf_index_from_jsonl

                idx = build_tfidf_index_from_jsonl(local_rag_jsonl)
                rag_facts = idx.retrieve(user_prompt, top_k=self.defaults.rag_top_k)
            except Exception:
                pass

        # --- 5. Creative RAG enrichment ---
        creative_enriched = ""
        creative_reasoning = ""
        creative_neg = ""
        if cfg.creative_rag and not dry_run:
            try:
                from utils.prompt.creative_rag import CreativeRAGEngine

                engine = CreativeRAGEngine()
                ref_for_creative = ref_paths[0] if ref_paths else None
                cr = engine.enrich(
                    prompt=user_prompt,
                    reference_image_path=ref_for_creative,
                    facts=rag_facts + dissection_facts,
                    creativity_level=float(cfg.creative_rag_level),
                )
                creative_enriched = cr.enriched_prompt
                creative_reasoning = cr.reasoning
                creative_neg = cr.negative_additions
                trace["steps"].append({"creative_rag": {"novelty": cr.novelty_score, "fallback": cr.fallback_used}})
            except Exception as exc:
                trace["steps"].append({"creative_rag_error": str(exc)})

        neg_eff = ", ".join(x for x in (negative_prompt, creative_neg) if x).strip(", ")

        brief = synthesize_scene_brief(
            user_prompt,
            understandings=understandings,
            dissection_facts=dissection_facts,
            rag_facts=rag_facts,
            expected_text=expected_text,
            negative_prompt=neg_eff,
            init_image=init_image,
            inpaint_mask=inpaint_mask,
            creative_enriched=creative_enriched,
            creative_reasoning=creative_reasoning,
        )
        brief.save(work / "scene_brief.json")

        if dry_run:
            return VisualBrainResult(
                accepted=True,
                out_path=str(work / out),
                brief=brief,
                reference_paths=ref_paths,
                metrics={"composite": 0.85, "clip": 0.35, "coverage": 0.9},
                trace_path=str(work / "brain_trace.json"),
            )

        # --- 6. Generate + verify + edit loop ---
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = work / out_path
        gen_prompt = brief.build_generation_prompt()
        metrics: Dict[str, float] = {}
        accepted = False

        for loop in range(max(1, cfg.max_edit_loops)):
            extra = self._sample_args_from_brief(
                brief,
                dissect_refs=dissect_refs_arg,
                init_image=init_image,
                inpaint_mask=inpaint_mask,
                loop=loop,
            )
            suffix = "" if loop == 0 else f"_edit{loop}"
            cmd = self._build_sample_cmd(
                ckpt=ckpt,
                prompt=gen_prompt,
                out_path=out_path,
                suffix=suffix,
                device=device,
                negative_prompt=brief.negative_prompt,
                expected_text=expected_text,
                vit_ckpt=vit_ckpt,
                local_rag_jsonl=local_rag_jsonl,
                extra_args=extra,
            )
            rc = subprocess.run(cmd, cwd=str(self.repo_root), check=False).returncode
            cand = out_path if suffix == "" else out_path.with_name(out_path.stem + suffix + out_path.suffix)
            if rc != 0 or not cand.is_file():
                trace["steps"].append({"generate_fail": {"loop": loop, "rc": rc}})
                continue

            metrics = self._verify_image(
                cand, gen_prompt, expected_text=expected_text, device=device, vit_ckpt=vit_ckpt
            )
            cov = prompt_coverage_score(brief, metrics)
            metrics["coverage"] = cov
            trace["steps"].append({"loop": loop, "metrics": metrics, "out": str(cand)})

            if cov >= cfg.min_coverage and metrics.get("clip", 0.0) >= cfg.min_clip:
                accepted = True
                out_path = cand
                break

            # Edit pass: OCR fix or inpaint refine
            if expected_text and float(metrics.get("ocr_match", 0.0) or 0.0) < 0.65:
                extra_edit = ["--ocr-fix", "--expected-text", expected_text]
                cmd2 = self._build_sample_cmd(
                    ckpt=ckpt,
                    prompt=gen_prompt,
                    out_path=cand,
                    suffix=f"_ocrfix{loop}",
                    device=device,
                    negative_prompt=brief.negative_prompt,
                    expected_text=expected_text,
                    vit_ckpt=vit_ckpt,
                    local_rag_jsonl=local_rag_jsonl,
                    extra_args=extra + extra_edit + ["--init-image", str(cand), "--strength", "0.35"],
                )
                subprocess.run(cmd2, cwd=str(self.repo_root), check=False)
                fix_path = cand.with_name(cand.stem + f"_ocrfix{loop}" + cand.suffix)
                if fix_path.is_file():
                    cand = fix_path
                    metrics = self._verify_image(
                        cand, gen_prompt, expected_text=expected_text, device=device, vit_ckpt=vit_ckpt
                    )
                    metrics["coverage"] = prompt_coverage_score(brief, metrics)
                    if metrics["coverage"] >= cfg.min_coverage:
                        accepted = True
                        out_path = cand
                        break

            # Prompt patch from low CLIP — re-anchor to user prompt
            if float(metrics.get("clip", 0.0) or 0.0) < cfg.min_clip:
                gen_prompt = f"{user_prompt}. {gen_prompt}"[:2000]

        trace_path = work / "brain_trace.json"
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

        return VisualBrainResult(
            accepted=accepted,
            out_path=str(out_path),
            brief=brief,
            reference_paths=ref_paths,
            metrics=metrics,
            iterations=int(trace.get("steps", [{}])[-1].get("loop", 0) + 1 if trace["steps"] else 1),
            trace_path=str(trace_path),
        )

    def _sample_args_from_brief(
        self,
        brief: SceneBrief,
        *,
        dissect_refs: str,
        init_image: str,
        inpaint_mask: str,
        loop: int,
    ) -> List[str]:
        cfg = self.config
        args: List[str] = list(self.defaults.extra_sample_args)
        if cfg.creative_rag:
            args.append("--creative-rag")
            args.extend(["--creative-rag-level", str(cfg.creative_rag_level)])
        if dissect_refs:
            args.extend(["--dissect-refs", dissect_refs])
        if init_image and inpaint_mask and loop == 0:
            args.extend(
                [
                    "--init-image",
                    init_image,
                    "--mask",
                    inpaint_mask,
                    "--auto-init-from-dissection",
                ]
            )
        elif brief.control_image:
            args.extend(["--control-image", brief.control_image, "--control-type", "canny"])
        if cfg.self_correct:
            args.append("--superior-self-correct")
        args.extend(["--human-made", "standard"])
        return args

    def _build_sample_cmd(
        self,
        *,
        ckpt: str,
        prompt: str,
        out_path: Path,
        suffix: str,
        device: str,
        negative_prompt: str,
        expected_text: str,
        vit_ckpt: str,
        local_rag_jsonl: str,
        extra_args: List[str],
    ) -> List[str]:
        cfg = self.config
        out = out_path if not suffix else out_path.with_name(out_path.stem + suffix + out_path.suffix)
        cmd = [
            sys.executable,
            str(self.repo_root / "sample.py"),
            "--ckpt",
            ckpt,
            "--prompt",
            prompt,
            "--out",
            str(out),
            "--device",
            device,
            "--num",
            str(cfg.num_candidates),
            "--pick-best",
            cfg.pick_metric,
            "--preset",
            "superior",
        ]
        if negative_prompt:
            cmd.extend(["--negative-prompt", negative_prompt])
        if expected_text:
            cmd.extend(["--expected-text", expected_text])
        if vit_ckpt:
            cmd.extend(["--pick-vit-ckpt", vit_ckpt])
        if local_rag_jsonl:
            cmd.extend(["--local-rag-jsonl", local_rag_jsonl])
        cmd.extend(extra_args)
        return cmd

    def _verify_image(
        self,
        path: Path,
        prompt: str,
        *,
        expected_text: str,
        device: str,
        vit_ckpt: str,
    ) -> Dict[str, float]:
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        from utils.quality import test_time_pick as ttp
        from utils.superior.composite_ranker import CompositeRanker

        ranker = CompositeRanker()
        scores = ranker.score_images([arr], prompt=prompt, device=device, vit_ckpt=vit_ckpt or "")
        clip = ttp.score_clip_similarity([arr], prompt, device=device)
        metrics: Dict[str, float] = {
            "composite": float(scores[0]),
            "clip": float(clip[0]) if clip else 0.0,
            "sharpness": float(ttp.score_edge_sharpness(arr)),
            "aesthetic": float(ttp.score_aesthetic_proxy(arr)),
        }
        if expected_text:
            ocr_scores = ttp.score_ocr_match([arr], expected_text)
            metrics["ocr_match"] = float(ocr_scores[0]) if ocr_scores else 0.0
        return metrics


__all__ = ["VisualBrain", "VisualBrainConfig", "VisualBrainResult"]
