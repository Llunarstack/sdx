"""
Invent novel StyleGenome objects from a user prompt (Qwen2.5 or deterministic fallback).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import textwrap
import uuid
from pathlib import Path
from typing import List

from .style_genome import StyleGenome, is_genome_novel_enough
from .style_genome_chaos import InventionMode, apply_chaos_level, invent_insane_batch

_PRETRAINED_ROOT = Path(__file__).resolve().parents[2] / "pretrained"
_QWEN_PATH = _PRETRAINED_ROOT / "Qwen2.5-14B-Instruct"

_FALLBACK_PALETTES = (
    "volcanic glass highlights on desaturated umber base",
    "bioluminescent cyan accents over charcoal neutrals",
    "sun-bleached terracotta and oxidized copper",
    "frosted lavender mist with pearl grey shadows",
    "electric magenta rim light on deep indigo fill",
)
_FALLBACK_LINES = (
    "variable-weight ink strokes with dry-brush breaks",
    "clean vector contours with one weighted accent edge",
    "scratchy gestural linework, incomplete outlines",
    "engraved cross-hatching with selective solid blacks",
)
_FALLBACK_SURFACES = (
    "visible paper tooth, matte gouache layering",
    "subsurface wax resist on rough cotton",
    "anisotropic brushed metal with micro-scratches",
    "chalky pigment bloom at edge transitions",
)
_FALLBACK_CAMERAS = (
    "wide angle with slight dutch tilt",
    "telephoto compression, shallow depth of field",
    "low hero angle, asymmetric negative space",
    "overhead tableau framing, graphic flatness",
)
_FALLBACK_LIGHT = (
    "single hard key with colored bounce fill",
    "overcast skylight, soft wrap, no blown highlights",
    "rim-only silhouette with fog scatter",
    "practical lamp pools in otherwise dim scene",
)
_ANTI_CLONE = (
    "not studio ghibli",
    "not miyazaki",
    "not disney",
    "not pixar",
    "not marvel comic",
    "not generic anime",
)


def _extract_json_array(text: str) -> List[dict]:
    text = (text or "").strip()
    if not text:
        return []
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    return []


def _qwen_invent_genomes(
    prompt: str,
    n: int,
    creativity_level: float,
    *,
    device: str = "cpu",
    max_new_tokens: int = 1024,
    invention_mode: InventionMode = "normal",
) -> List[StyleGenome]:
    if not _QWEN_PATH.exists() or n < 1:
        return []
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return []

    try:
        tok = AutoTokenizer.from_pretrained(str(_QWEN_PATH), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(_QWEN_PATH),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            device_map=device if device != "cpu" else None,
        )
        model.eval()

        insane = invention_mode in ("insane", "apocalypse", "chimera", "glitch", "eldritch", "cyberpunk")
        novelty = (
            "GO UNHINGED: impossible materials, sacred-profane tension, visual danger, synesthetic color. "
            "No famous artist names. Make statisticians weep."
            if insane or creativity_level > 0.85
            else "Maximize originality — avoid naming famous artists or trademarked franchises."
            if creativity_level > 0.55
            else "Balance novelty with clarity; still avoid artist name drops."
        )

        system_msg = textwrap.dedent("""
            You invent NEW visual styles for an AI image model. Output only valid JSON.
            Each style must be a fresh aesthetic identity built from orthogonal axes, not a copy
            of a known artist or franchise.
        """).strip()
        if insane:
            system_msg += (
                " You are allowed to be extreme: glitch sacrament, biolume abyss, rusted sky myth, "
                "porcelain fracture, eldritch taxonomy — but always original phrasing."
            )

        user_msg = textwrap.dedent(f"""
            Scene / subject prompt: "{prompt}"
            Invention mode: {invention_mode}

            Invent exactly {n} distinct style genomes as a JSON array. {novelty}

            Each object must have:
            - "name": short title (2–5 words)
            - "palette", "line", "surface", "camera", "lighting": one vivid phrase each
            - "signature": one sentence tying the look together
            - "positive_fragments": array of 2–4 extra comma-ready tags
            - "negative_fragments": array of 3–6 anti-generic terms
            - "anti_clone": array of 2–4 "not X" phrases blocking famous style clones
            - "reasoning": one sentence

            Do not repeat the same palette or line approach across genomes.
            Respond with ONLY the JSON array, no markdown fences.
        """).strip()

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=creativity_level > 0.25,
                temperature=max(0.2, min(1.1, 0.35 + creativity_level * 0.75)),
                top_p=0.92,
                repetition_penalty=1.08,
                pad_token_id=tok.eos_token_id,
            )

        response = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        raw_items = _extract_json_array(response)
        genomes: List[StyleGenome] = []
        for i, item in enumerate(raw_items[:n]):
            item = dict(item)
            item.setdefault("id", f"qwen_{uuid.uuid4().hex[:8]}")
            item.setdefault("name", f"Invented style {i + 1}")
            g = StyleGenome.from_dict(item)
            if is_genome_novel_enough(g):
                genomes.append(g)
        return genomes
    except Exception:
        return []


def _fallback_invent_genomes(
    prompt: str,
    n: int,
    *,
    seed: int = 42,
    creativity_level: float = 0.7,
) -> List[StyleGenome]:
    h = hashlib.sha256(f"{prompt}|{seed}|{creativity_level:.2f}".encode()).hexdigest()
    rng = random.Random(int(h[:8], 16))
    genomes: List[StyleGenome] = []
    used_sigs: set[str] = set()

    for i in range(n):
        palette = rng.choice(_FALLBACK_PALETTES)
        line = rng.choice(_FALLBACK_LINES)
        surface = rng.choice(_FALLBACK_SURFACES)
        camera = rng.choice(_FALLBACK_CAMERAS)
        lighting = rng.choice(_FALLBACK_LIGHT)
        signature = (
            f"{palette.split(',')[0]} meets {line.split(',')[0]} — "
            f"intentional {rng.choice(('asymmetry', 'tension', 'stillness', 'rhythm'))} for the scene"
        )
        if signature in used_sigs:
            signature += f" (variant {i})"
        used_sigs.add(signature)

        anti = tuple(rng.sample(_ANTI_CLONE, k=min(3, len(_ANTI_CLONE))))
        pos_extra = (
            "original composition grammar",
            "material-accurate rendering",
            "cohesive color script",
        )
        neg_extra = (
            "generic stock photo",
            "template composition",
            "muddy midtones",
            "overprocessed HDR",
        )

        genomes.append(
            StyleGenome(
                id=f"fb_{uuid.uuid4().hex[:8]}",
                name=f"Synthetic genome {i + 1}",
                palette=palette,
                line=line,
                surface=surface,
                camera=camera,
                lighting=lighting,
                signature=signature,
                anti_clone=anti,
                positive_fragments=pos_extra,
                negative_fragments=neg_extra,
                reasoning=f"Deterministic axis blend for prompt intent (creativity={creativity_level:.2f}).",
            )
        )
    return genomes


class StyleInventor:
    """Propose N novel style genomes for a base prompt."""

    def __init__(self, *, device: str = "cpu", use_qwen: bool = True) -> None:
        self.device = device
        self.use_qwen = use_qwen

    def invent(
        self,
        prompt: str,
        n: int = 3,
        *,
        creativity_level: float = 0.75,
        seed: int = 42,
        max_catalog_overlap: float = 0.55,
        bank_boost: bool = True,
        invention_mode: InventionMode = "normal",
        chaos_level: float = 0.0,
    ) -> List[StyleGenome]:
        prompt = (prompt or "").strip()
        if not prompt or n < 1:
            return []

        mode: InventionMode = invention_mode  # type: ignore[assignment]
        chaos = max(0.0, min(1.0, float(chaos_level)))
        insane_modes = ("insane", "apocalypse", "chimera", "glitch", "eldritch", "cyberpunk")
        if mode in insane_modes:
            max_catalog_overlap = max(max_catalog_overlap, 0.92)

        genomes: List[StyleGenome] = []
        if mode in insane_modes:
            genomes = invent_insane_batch(
                prompt,
                n,
                seed=seed,
                mode=mode,
                chaos_level=max(chaos, 0.65 if mode == "insane" else 0.9),
            )

        if self.use_qwen and mode not in insane_modes:
            genomes = _qwen_invent_genomes(
                prompt,
                n,
                creativity_level,
                device=self.device,
                invention_mode=mode,
            )
        elif self.use_qwen and len(genomes) < n:
            extra = _qwen_invent_genomes(
                prompt,
                n - len(genomes),
                max(creativity_level, 0.9),
                device=self.device,
                invention_mode=mode,
            )
            genomes.extend(extra)

        if len(genomes) < n and mode not in insane_modes:
            need = n - len(genomes)
            genomes.extend(_fallback_invent_genomes(prompt, need, seed=seed, creativity_level=creativity_level))
        elif len(genomes) < n:
            genomes.extend(
                invent_insane_batch(
                    prompt,
                    n - len(genomes),
                    seed=seed + 17,
                    mode="insane",
                    chaos_level=max(chaos, 0.8),
                )
            )

        if chaos > 0.01:
            chaos_rng = random.Random(seed + 9001)
            genomes = [apply_chaos_level(g, chaos, rng=chaos_rng) for g in genomes]

        if bank_boost:
            try:
                from .style_memory import StyleGenomeBank

                for g in StyleGenomeBank().top_genomes(k=2):
                    if len(genomes) >= n + 1:
                        break
                    if g.id not in {x.id for x in genomes}:
                        genomes.append(g)
            except Exception:
                pass

        # Filter near-duplicate catalog styles; dedupe by signature
        filtered: List[StyleGenome] = []
        seen: set[str] = set()
        for g in genomes:
            sig = g.signature.lower()[:80]
            if sig in seen:
                continue
            if not is_genome_novel_enough(g, max_catalog_overlap=max_catalog_overlap):
                continue
            seen.add(sig)
            filtered.append(g)
            if len(filtered) >= n:
                break

        if len(filtered) < n:
            for g in genomes:
                if g not in filtered:
                    filtered.append(g)
                if len(filtered) >= n:
                    break

        return filtered[:n]


__all__ = ["StyleInventor"]
