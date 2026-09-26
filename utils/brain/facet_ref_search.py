"""Per-facet reference image search (subject / attire / artist style)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.prompt.facet_decompose import PromptFacets, decompose_prompt_facets

__all__ = ["FacetRefPack", "gather_facet_references"]


@dataclass
class FacetRefPack:
    facets: dict[str, Any]
    queries: dict[str, str] = field(default_factory=dict)
    paths: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    nsfw: bool = False

    def moodboard_paths(self, *, prefer: tuple[str, ...] = ("style", "subject", "attire")) -> list[str]:
        out: list[str] = []
        for key in prefer:
            out.extend(self.paths.get(key) or [])
        # Dedup preserve order
        seen: set[str] = set()
        uniq: list[str] = []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq


def gather_facet_references(
    prompt: str | PromptFacets,
    *,
    work_dir: str | Path,
    max_per_facet: int = 4,
    allow_web: bool = True,
    facets_filter: tuple[str, ...] = ("subject", "attire", "style"),
) -> FacetRefPack:
    """Search and download refs into work_dir/refs/{facet}/."""
    if isinstance(prompt, PromptFacets):
        facets = prompt
    else:
        facets = decompose_prompt_facets(prompt)

    work = Path(work_dir)
    pack = FacetRefPack(facets=facets.to_dict(), nsfw=facets.nsfw)
    pack.queries = dict(facets.search_queries)

    try:
        from utils.brain.image_search import download_search_hits, search_reference_images
    except Exception as exc:
        pack.notes.append(f"image_search unavailable: {exc}")
        return pack

    for facet in facets_filter:
        q = facets.search_queries.get(facet) or ""
        if not q.strip():
            continue
        # NSFW prompts: broader query wording (provider safe-search is not always controllable).
        query = q
        if facets.nsfw and facet in ("subject", "attire"):
            query = f"{q} (reference art)"
        res = search_reference_images(query, max_results=max_per_facet, allow_web=allow_web)
        if not res.hits:
            pack.notes.append(f"{facet}: no hits ({res.notes or 'empty'})")
            continue
        dest = work / "refs" / facet
        try:
            hits = download_search_hits(res.hits, dest, max_download=max_per_facet)
        except Exception as exc:
            pack.notes.append(f"{facet}: download failed: {exc}")
            continue
        pack.paths[facet] = [str(h.local_path) for h in hits if getattr(h, "local_path", None)]
        pack.notes.append(f"{facet}: {len(pack.paths[facet])} files from '{query}'")
    return pack
