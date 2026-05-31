"""
**Agent tools** — callable skills the agent orchestrates (GenEvolve-style).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


class AgentTool(str, Enum):
    rag_retrieve = "rag_retrieve"
    web_search = "web_search"
    understand_refs = "understand_refs"
    dissect_refs = "dissect_refs"
    build_scene_brief = "build_scene_brief"
    expand_prompt = "expand_prompt"
    generate = "generate"
    verify = "verify"
    verify_ocr = "verify_ocr"
    inpaint_edit = "inpaint_edit"
    self_correct = "self_correct"
    reflect = "reflect"
    benchmark = "benchmark"
    flywheel = "flywheel"


@dataclass(slots=True)
class ToolResult:
    tool: AgentTool
    ok: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Execute agent tools against ``AgentContext`` + mutable session state."""

    def __init__(
        self,
        ctx: Any,
        *,
        repo_root: Path,
        num_candidates: int = 4,
        pick_metric: str = "superior_composite",
        expand_prompt: bool = True,
        self_correct: bool = True,
        rag_top_k: int = 8,
        extra_sample_args: Optional[List[str]] = None,
    ) -> None:
        self.ctx = ctx
        self.repo_root = repo_root
        self.num_candidates = max(1, int(num_candidates))
        self.pick_metric = str(pick_metric)
        self.expand_prompt = bool(expand_prompt)
        self.self_correct = bool(self_correct)
        self.rag_top_k = int(rag_top_k)
        self.extra_sample_args = list(extra_sample_args or [])
        self.session: Dict[str, Any] = {
            "prompt": ctx.prompt,
            "negative_prompt": ctx.negative_prompt,
            "rag_facts": [],
            "reference_paths": list(getattr(ctx, "reference_images", []) or []),
            "understandings": [],
            "scene_brief": None,
            "last_out": "",
            "metrics": {},
        }

    def execute(self, tool: AgentTool, **kwargs: Any) -> ToolResult:
        fn = {
            AgentTool.rag_retrieve: self._rag_retrieve,
            AgentTool.web_search: self._web_search,
            AgentTool.understand_refs: self._understand_refs,
            AgentTool.dissect_refs: self._dissect_refs,
            AgentTool.build_scene_brief: self._build_scene_brief,
            AgentTool.expand_prompt: self._expand_prompt,
            AgentTool.generate: self._generate,
            AgentTool.verify: self._verify,
            AgentTool.verify_ocr: self._verify_ocr,
            AgentTool.inpaint_edit: self._inpaint_edit,
            AgentTool.self_correct: self._self_correct,
            AgentTool.reflect: self._reflect_stub,
            AgentTool.benchmark: self._benchmark,
            AgentTool.flywheel: self._flywheel,
        }.get(tool)
        if fn is None:
            return ToolResult(tool=tool, ok=False, message=f"unknown tool {tool}")
        return fn(**kwargs)

    def _rag_retrieve(self, **_: Any) -> ToolResult:
        path = str(getattr(self.ctx, "local_rag_jsonl", "") or "").strip()
        if not path:
            return ToolResult(AgentTool.rag_retrieve, ok=True, message="no rag corpus")
        from utils.superior.retrieval import build_tfidf_index_from_jsonl

        idx = build_tfidf_index_from_jsonl(path)
        facts = idx.retrieve(self.session["prompt"], top_k=self.rag_top_k)
        self.session["rag_facts"] = facts
        if facts:
            merged = self.session["prompt"] + ". " + " ".join(facts[:3])
            self.session["prompt"] = merged.strip()
        return ToolResult(AgentTool.rag_retrieve, ok=True, data={"facts": facts, "count": len(facts)})

    def _web_search(self, **_: Any) -> ToolResult:
        if bool(getattr(self.ctx, "dry_run", False)):
            return ToolResult(AgentTool.web_search, ok=True, message="dry_run")
        allow = bool(getattr(self.ctx, "web_search", True))
        if not allow:
            return ToolResult(AgentTool.web_search, ok=True, message="web search disabled")
        from utils.brain.image_search import download_search_hits, search_reference_images

        work = Path(getattr(self.ctx, "work_dir", "agentic_run")) / "search"
        sr = search_reference_images(self.session["prompt"], max_results=4, allow_web=True)
        hits = download_search_hits(sr.hits, work, max_download=4)
        for h in hits:
            if h.local_path:
                self.session["reference_paths"].append(h.local_path)
        return ToolResult(
            AgentTool.web_search,
            ok=True,
            data={"hits": len(hits), "paths": [h.local_path for h in hits if h.local_path], "notes": sr.notes},
        )

    def _understand_refs(self, **_: Any) -> ToolResult:
        paths = list(self.session.get("reference_paths") or [])
        if not paths:
            return ToolResult(AgentTool.understand_refs, ok=True, message="no references")
        if bool(getattr(self.ctx, "dry_run", False)):
            return ToolResult(AgentTool.understand_refs, ok=True, message="dry_run")
        from utils.brain.understand import understand_images

        work = Path(getattr(self.ctx, "work_dir", "agentic_run"))
        unders = understand_images(
            paths,
            user_prompt=self.session["prompt"],
            work_dir=work,
            device=str(getattr(self.ctx, "device", "cuda")),
        )
        self.session["understandings"] = [u.__dict__ for u in unders]
        facts = []
        for i, u in enumerate(unders):
            if u.caption:
                facts.append(f"Reference {i + 1}: {u.caption[:250]}")
            if u.ocr_text:
                facts.append(f"Reference {i + 1} text: {u.ocr_text[:120]}")
        self.session["rag_facts"].extend(facts)
        return ToolResult(AgentTool.understand_refs, ok=True, data={"count": len(unders), "facts": facts})

    def _dissect_refs(self, **_: Any) -> ToolResult:
        paths = list(self.session.get("reference_paths") or [])
        if not paths:
            return ToolResult(AgentTool.dissect_refs, ok=True, message="no references")
        if bool(getattr(self.ctx, "dry_run", False)):
            return ToolResult(AgentTool.dissect_refs, ok=True, message="dry_run")
        from utils.generation.image_dissection import dissect_images_to_parts

        work = Path(getattr(self.ctx, "work_dir", "agentic_run")) / "dissection"
        _reqs, parts, facts = dissect_images_to_parts(
            self.session["prompt"],
            paths,
            output_dir=work,
            enable_heavy_models=True,
        )
        self.session["rag_facts"].extend(facts)
        self.session["dissect_parts"] = len(parts)
        if parts:
            try:
                from utils.generation.part_compositing import build_init_and_inpaint_mask

                spec = build_init_and_inpaint_mask(
                    reference_images=paths,
                    parts=parts,
                    output_dir=work / "composite",
                    target_size=(512, 512),
                )
                self.session["init_image"] = spec.init_image_path
                self.session["inpaint_mask"] = spec.mask_path
            except Exception:
                pass
        return ToolResult(AgentTool.dissect_refs, ok=True, data={"parts": len(parts), "facts": facts})

    def _build_scene_brief(self, **_: Any) -> ToolResult:
        from utils.brain.scene_brief import synthesize_scene_brief
        from utils.brain.understand import ImageUnderstanding

        unders_raw = self.session.get("understandings") or []
        unders = [ImageUnderstanding(**u) if isinstance(u, dict) else u for u in unders_raw]
        brief = synthesize_scene_brief(
            self.session["prompt"],
            understandings=unders,
            dissection_facts=self.session.get("rag_facts") or [],
            expected_text=str(getattr(self.ctx, "expected_text", "") or ""),
            negative_prompt=str(self.session.get("negative_prompt") or ""),
            init_image=str(self.session.get("init_image") or ""),
            inpaint_mask=str(self.session.get("inpaint_mask") or ""),
        )
        self.session["scene_brief"] = brief
        self.session["prompt"] = brief.build_generation_prompt()
        return ToolResult(AgentTool.build_scene_brief, ok=True, data={"elements": len(brief.elements)})

    def _expand_prompt(self, **_: Any) -> ToolResult:
        if not self.expand_prompt:
            return ToolResult(AgentTool.expand_prompt, ok=True, message="skipped")
        from utils.superior.prompt_expand import expand_prompt_heuristic

        before = self.session["prompt"]
        self.session["prompt"] = expand_prompt_heuristic(before)
        return ToolResult(AgentTool.expand_prompt, ok=True, data={"before": before, "after": self.session["prompt"]})

    def _generate(self, *, out_suffix: str = "", extra_args: Optional[List[str]] = None) -> ToolResult:
        out = Path(getattr(self.ctx, "out", "agent_out.png"))
        if out_suffix:
            out = out.with_name(out.stem + out_suffix + out.suffix)
        work = Path(getattr(self.ctx, "work_dir", "agentic_run"))
        if not out.is_absolute():
            out = work / out
        if bool(getattr(self.ctx, "dry_run", False)):
            self.session["last_out"] = str(out)
            return ToolResult(AgentTool.generate, ok=True, message="dry_run", data={"out": str(out)})
        cmd = [
            sys.executable,
            str(self.repo_root / "sample.py"),
            "--ckpt",
            str(self.ctx.ckpt),
            "--prompt",
            self.session["prompt"],
            "--out",
            str(out),
            "--device",
            str(getattr(self.ctx, "device", "cuda")),
            "--num",
            str(self.num_candidates),
            "--pick-best",
            self.pick_metric,
            "--preset",
            "superior",
        ]
        if self.session.get("negative_prompt"):
            cmd.extend(["--negative-prompt", str(self.session["negative_prompt"])])
        if str(getattr(self.ctx, "local_rag_jsonl", "") or "").strip():
            cmd.extend(["--local-rag-jsonl", str(self.ctx.local_rag_jsonl)])
        if self.self_correct:
            cmd.append("--superior-self-correct")
        if self.expand_prompt:
            cmd.append("--expand-prompt")
        vit = str(getattr(self.ctx, "vit_ckpt", "") or "").strip()
        if vit:
            cmd.extend(["--pick-vit-ckpt", vit])
        exp = str(getattr(self.ctx, "expected_text", "") or "").strip()
        if exp:
            cmd.extend(["--expected-text", exp])
        refs = self.session.get("reference_paths") or []
        if refs:
            cmd.extend(["--dissect-refs", ",".join(str(r) for r in refs)])
            cmd.append("--creative-rag")
        init_i = str(self.session.get("init_image") or "").strip()
        mask_i = str(self.session.get("inpaint_mask") or "").strip()
        if init_i and mask_i:
            cmd.extend(["--init-image", init_i, "--mask", mask_i, "--auto-init-from-dissection"])
        cmd.extend(self.extra_sample_args)
        if extra_args:
            cmd.extend(extra_args)
        rc = subprocess.run(cmd, cwd=str(self.repo_root), check=False).returncode
        ok = rc == 0 and out.is_file()
        self.session["last_out"] = str(out)
        return ToolResult(AgentTool.generate, ok=ok, message=f"exit={rc}", data={"out": str(out), "cmd": cmd})

    def _verify(self, **_: Any) -> ToolResult:
        if bool(getattr(self.ctx, "dry_run", False)):
            metrics = {"composite": 0.85, "clip": 0.35, "sharpness": 200.0}
            self.session["metrics"] = metrics
            return ToolResult(AgentTool.verify, ok=True, data=metrics)
        out = self.session.get("last_out", "")
        if not out or not Path(out).is_file():
            return ToolResult(AgentTool.verify, ok=False, message="no image to verify")
        arr = np.array(Image.open(out).convert("RGB"), dtype=np.uint8)
        from utils.superior.composite_ranker import CompositeRanker

        ranker = CompositeRanker()
        vit = str(getattr(self.ctx, "vit_ckpt", "") or "").strip()
        scores = ranker.score_images([arr], prompt=self.session["prompt"], device=str(self.ctx.device), vit_ckpt=vit)
        from utils.quality import test_time_pick as ttp

        clip = ttp.score_clip_similarity([arr], self.session["prompt"], device=str(self.ctx.device))
        metrics = {
            "composite": float(scores[0]),
            "clip": float(clip[0]) if clip else 0.0,
            "sharpness": float(ttp.score_edge_sharpness(arr)),
            "aesthetic": float(ttp.score_aesthetic_proxy(arr)),
        }
        exp = str(getattr(self.ctx, "expected_text", "") or "").strip()
        if exp:
            ocr = ttp.score_ocr_match([arr], exp)
            metrics["ocr_match"] = float(ocr[0]) if ocr else 0.0
        self.session["metrics"] = metrics
        return ToolResult(AgentTool.verify, ok=True, data=metrics)

    def _verify_ocr(self, **_: Any) -> ToolResult:
        v = self._verify()
        exp = str(getattr(self.ctx, "expected_text", "") or "").strip()
        if not exp:
            return ToolResult(AgentTool.verify_ocr, ok=True, message="no expected text", data=v.data)
        ok = float(v.data.get("ocr_match", 0.0) or 0.0) >= 0.65
        return ToolResult(AgentTool.verify_ocr, ok=ok, data=v.data)

    def _inpaint_edit(self, **kwargs: Any) -> ToolResult:
        """Regional edit via sample.py img2img/inpaint."""
        out = self.session.get("last_out", "")
        if not out or not Path(out).is_file():
            return ToolResult(AgentTool.inpaint_edit, ok=False, message="no image")
        region = str(kwargs.get("region", "full") or "full")
        extra = ["--init-image", str(out), "--strength", str(kwargs.get("strength", 0.4))]
        if region != "full":
            try:
                from utils.generation.segmentation_to_mask import build_segmentation_mask_for_edit

                seg = build_segmentation_mask_for_edit(out, region)
                mask_path = Path(getattr(self.ctx, "work_dir", "agentic_run")) / "edit_mask.png"
                seg.mask.save(mask_path)
                extra.extend(["--mask", str(mask_path)])
            except Exception:
                pass
        return self._generate(out_suffix="_inpaint", extra_args=extra)

    def _self_correct(self, **_: Any) -> ToolResult:
        return ToolResult(AgentTool.self_correct, ok=True, message="handled in generate via --superior-self-correct")

    def _reflect_stub(self, **kwargs: Any) -> ToolResult:
        suggestion = kwargs.get("suggestion", "")
        if suggestion:
            self.session["prompt"] = (self.session["prompt"] + ", " + suggestion).strip(", ")
        return ToolResult(AgentTool.reflect, ok=True, data={"prompt": self.session["prompt"]})

    def _benchmark(self, **_: Any) -> ToolResult:
        if bool(getattr(self.ctx, "dry_run", False)):
            return ToolResult(AgentTool.benchmark, ok=True, message="dry_run")
        work = Path(getattr(self.ctx, "work_dir", "agentic_run")) / "benchmark"
        cmd = [
            sys.executable,
            "-m",
            "scripts.tools",
            "benchmark_suite",
            "--ckpt",
            str(self.ctx.ckpt),
            "--out-dir",
            str(work),
            "--preset",
            "superior",
            "--num",
            str(self.num_candidates),
            "--pick-best",
            self.pick_metric,
        ]
        rc = subprocess.run(cmd, cwd=str(self.repo_root), check=False).returncode
        return ToolResult(AgentTool.benchmark, ok=rc == 0, data={"out_dir": str(work)})

    def _flywheel(self, **_: Any) -> ToolResult:
        if bool(getattr(self.ctx, "dry_run", False)):
            return ToolResult(AgentTool.flywheel, ok=True, message="dry_run")
        work = Path(getattr(self.ctx, "work_dir", "agentic_run")) / "flywheel"
        cmd = [
            sys.executable,
            "-m",
            "scripts.tools",
            "run_flywheel",
            "--base-ckpt",
            str(self.ctx.ckpt),
            "--work-dir",
            str(work),
        ]
        rag = str(getattr(self.ctx, "local_rag_jsonl", "") or "").strip()
        if rag:
            cmd.extend(["--local-rag-jsonl", rag])
        vit = str(getattr(self.ctx, "vit_ckpt", "") or "").strip()
        if vit:
            cmd.extend(["--vit-ckpt", vit])
        rc = subprocess.run(cmd, cwd=str(self.repo_root), check=False).returncode
        return ToolResult(AgentTool.flywheel, ok=rc == 0, data={"work_dir": str(work)})


__all__ = ["AgentTool", "ToolRegistry", "ToolResult"]
