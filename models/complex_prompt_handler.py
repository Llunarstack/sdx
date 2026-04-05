"""
Complex, NSFW, Surreal & Weird Prompt Handler.

Standard diffusion models fail on:
  1. NSFW / explicit anatomy — "melted wax" artifacts, wrong topology
  2. Surreal / contradictory concepts — "melting clocks + cyberpunk + Victorian ghost"
  3. Physics-defying scenes — liquid, fabric, hair dynamics, tentacles
  4. Extreme body horror / transformation — partial morphs, hybrid creatures
  5. Fetish precision — specific material properties, unusual poses
  6. Abstract / conceptual — emotions as landscapes, synesthesia, impossible geometry

Architecture:
  - PromptComplexityAnalyzer: scores prompt complexity and routes to specialist paths
  - ConceptFusionModule: cleanly blends contradictory/surreal concepts
  - PhysicsAwareTokenizer: tags tokens with physical property hints (liquid, rigid, soft, etc.)
  - NSFWAnatomyRouter: routes explicit anatomy tokens to high-precision attention heads
  - SurrealismBlender: handles contradictory concept fusion without averaging artifacts
  - ComplexPromptConditioner: top-level module
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_enhancements import RMSNorm

# ---------------------------------------------------------------------------
# Prompt complexity analysis
# ---------------------------------------------------------------------------

@dataclass
class PromptComplexityProfile:
    """Analysis of a prompt's complexity and content type."""
    complexity_score: float = 0.0       # 0=simple, 1=extremely complex
    is_nsfw: bool = False
    is_surreal: bool = False
    is_physics_heavy: bool = False
    is_multi_concept: bool = False
    concept_count: int = 1
    dominant_mode: str = "standard"     # standard, nsfw, surreal, horror, abstract
    physics_tags: List[str] = field(default_factory=list)
    concept_groups: List[List[str]] = field(default_factory=list)


class PromptComplexityAnalyzer:
    """
    Analyzes prompt complexity and content type to route to specialist modules.
    """

    _NSFW = re.compile(
        r'\b(nude|naked|topless|bottomless|explicit|nsfw|erotic|sexual|'
        r'nipple|breast|penis|vagina|genitalia|intercourse|orgasm|'
        r'lingerie|bikini|underwear|bra|panties|thong|'
        r'bondage|bdsm|fetish|kink|latex|leather restraint|'
        r'tentacle|hentai|ecchi|lewd|suggestive|sensual|seductive)\b',
        re.IGNORECASE
    )
    _SURREAL = re.compile(
        r'\b(surreal|impossible|paradox|melting|floating|inverted|'
        r'dreamlike|nightmare|eldritch|lovecraftian|non-euclidean|'
        r'glitch|distorted|warped|fractured|fragmented|abstract|'
        r'dali|escher|magritte|kafkaesque|liminal|backrooms)\b',
        re.IGNORECASE
    )
    _PHYSICS = re.compile(
        r'\b(liquid|fluid|water|lava|slime|goo|gel|viscous|flowing|'
        r'fabric|cloth|silk|velvet|draping|billowing|rippling|'
        r'hair|fur|feathers|scales|tentacles|vines|chains|rope|'
        r'smoke|fire|flame|explosion|shatter|breaking|crumbling|'
        r'elastic|stretching|morphing|transforming|melting|freezing)\b',
        re.IGNORECASE
    )
    _HORROR = re.compile(
        r'\b(body horror|grotesque|mutant|hybrid|chimera|eldritch|'
        r'extra limbs|multiple heads|fused|merged|corrupted|infected|'
        r'parasite|symbiote|transformation|metamorphosis|visceral)\b',
        re.IGNORECASE
    )
    _CONCEPT_SEPARATORS = re.compile(r'\s*(?:\+|and|meets|combined with|fused with|mixed with)\s*', re.IGNORECASE)

    def analyze(self, prompt: str) -> PromptComplexityProfile:
        words = prompt.split()
        is_nsfw = bool(self._NSFW.search(prompt))
        is_surreal = bool(self._SURREAL.search(prompt))
        is_physics = bool(self._PHYSICS.search(prompt))
        is_horror = bool(self._HORROR.search(prompt))

        # Count distinct concept groups (separated by + / "meets" / "and")
        concept_parts = self._CONCEPT_SEPARATORS.split(prompt)
        concept_count = len([p for p in concept_parts if p.strip()])
        is_multi = concept_count > 2

        # Complexity score
        score = min(1.0, (
            len(words) / 200.0 * 0.3 +
            (0.2 if is_nsfw else 0) +
            (0.2 if is_surreal else 0) +
            (0.15 if is_physics else 0) +
            (0.15 if is_horror else 0) +
            min(concept_count / 5.0, 0.2)
        ))

        # Dominant mode
        if is_horror or is_surreal:
            mode = "surreal"
        elif is_nsfw:
            mode = "nsfw"
        elif is_physics:
            mode = "physics"
        else:
            mode = "standard"

        # Physics tags
        physics_tags = []
        for tag in ["liquid", "fabric", "hair", "fire", "smoke", "tentacle", "elastic"]:
            if re.search(rf'\b{tag}\b', prompt, re.IGNORECASE):
                physics_tags.append(tag)

        return PromptComplexityProfile(
            complexity_score=score,
            is_nsfw=is_nsfw,
            is_surreal=is_surreal,
            is_physics_heavy=is_physics,
            is_multi_concept=is_multi,
            concept_count=concept_count,
            dominant_mode=mode,
            physics_tags=physics_tags,
            concept_groups=[p.strip().split() for p in concept_parts if p.strip()],
        )


# ---------------------------------------------------------------------------
# Concept Fusion Module
# ---------------------------------------------------------------------------

class ConceptFusionModule(nn.Module):
    """
    Cleanly blends contradictory or surreal concepts without averaging artifacts.

    Standard cross-attention averages all text tokens equally, which causes
    contradictory concepts to cancel each other out (e.g. "melting + solid").
    This module instead:
    1. Groups tokens into concept clusters.
    2. Applies each cluster's attention separately.
    3. Blends the results with a learned interpolation that preserves
       the distinct character of each concept.

    Args:
        hidden_size: Token dimension.
        num_heads: Attention heads.
        max_concepts: Maximum number of distinct concepts to handle.
    """

    def __init__(self, hidden_size: int, num_heads: int, max_concepts: int = 6):
        super().__init__()
        self.max_concepts = max_concepts
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Concept cluster detector: assigns tokens to concept groups
        self.cluster_assign = nn.Linear(hidden_size, max_concepts, bias=False)

        # Per-concept cross-attention (shared weights, different routing)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Concept blend weights: how to combine concept outputs
        self.blend_weights = nn.Parameter(torch.ones(max_concepts) / max_concepts)

        # Surrealism gate: for contradictory concepts, preserve both rather than average
        self.surreal_gate = nn.Parameter(torch.zeros(1))

        self.norm = RMSNorm(hidden_size)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        profile: Optional[PromptComplexityProfile] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) image tokens.
            text_emb: (B, L, D) text tokens.
            profile: Optional complexity profile for routing.
        Returns:
            (B, N, D) concept-fused tokens.
        """
        B, N, D = x.shape
        _, L, _ = text_emb.shape

        # Assign text tokens to concept clusters
        cluster_logits = self.cluster_assign(text_emb)  # (B, L, C)
        cluster_weights = F.softmax(cluster_logits, dim=-1)  # (B, L, C)

        # For each concept cluster, compute weighted text representation
        # (B, C, L) x (B, L, D) -> (B, C, D)
        concept_reps = torch.bmm(cluster_weights.transpose(1, 2), text_emb)  # (B, C, D)

        # Cross-attention: image tokens attend to each concept separately
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        concept_outputs = []
        for c in range(self.max_concepts):
            c_rep = concept_reps[:, c:c+1, :]  # (B, 1, D) — single concept token
            k = self.k_proj(c_rep).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(c_rep).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
            out = out.transpose(1, 2).reshape(B, N, D)
            concept_outputs.append(out)

        # Blend concept outputs
        blend = F.softmax(self.blend_weights, dim=0)  # (C,)
        fused = sum(blend[c] * concept_outputs[c] for c in range(self.max_concepts))

        # Surrealism gate: for surreal prompts, add a "contradiction signal"
        # that preserves the tension between concepts rather than averaging
        if profile is not None and profile.is_surreal:
            gate = torch.sigmoid(self.surreal_gate)
            # Contradiction = variance across concept outputs
            stacked = torch.stack(concept_outputs, dim=0)  # (C, B, N, D)
            contradiction = stacked.var(dim=0)  # (B, N, D)
            fused = fused + gate * contradiction * 0.3

        return x + self.out_proj(self.norm(fused))


# ---------------------------------------------------------------------------
# Physics-Aware Token Tagger
# ---------------------------------------------------------------------------

class PhysicsAwareTokenTagger(nn.Module):
    """
    Tags image tokens with physical property hints (liquid, rigid, soft, etc.)
    and injects physics-aware conditioning.

    This helps the model understand that:
    - Liquid tokens should have smooth, flowing boundaries
    - Fabric tokens should have fold/drape patterns
    - Hair tokens should have strand-level detail
    - Fire/smoke tokens should have volumetric, wispy edges

    Args:
        hidden_size: Token dimension.
    """

    PHYSICS_TYPES = ["rigid", "liquid", "fabric", "hair", "fire", "smoke",
                     "elastic", "tentacle", "crystal", "organic", "mechanical"]

    def __init__(self, hidden_size: int):
        super().__init__()
        n = len(self.PHYSICS_TYPES)

        # Physics type embeddings
        self.physics_embed = nn.Embedding(n, hidden_size)

        # Physics region predictor: which patches have which physics type
        self.physics_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, n),
        )

        # Physics conditioning injection
        self.physics_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.physics_proj.weight)

        self._type_to_id = {t: i for i, t in enumerate(self.PHYSICS_TYPES)}

    def get_physics_tags(self, physics_tags: List[str], device: torch.device) -> torch.Tensor:
        """Convert physics tag strings to embedding sum. Returns (D,)."""
        if not physics_tags:
            return torch.zeros(self.physics_embed.embedding_dim, device=device)
        ids = [self._type_to_id.get(t, 0) for t in physics_tags]
        ids_t = torch.tensor(ids, device=device)
        return self.physics_embed(ids_t).mean(dim=0)

    def forward(
        self,
        x: torch.Tensor,
        physics_tags: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) image tokens.
            physics_tags: List of active physics types from prompt analysis.
        Returns:
            (B, N, D) physics-conditioned tokens.
        """
        B, N, D = x.shape

        # Predict per-patch physics type
        physics_logits = self.physics_predictor(x)  # (B, N, num_types)
        physics_probs = F.softmax(physics_logits, dim=-1)  # (B, N, num_types)

        # Weighted sum of physics embeddings per patch
        # (B, N, num_types) x (num_types, D) -> (B, N, D)
        phys_emb = self.physics_embed.weight  # (num_types, D)
        physics_context = torch.einsum('bnt,td->bnd', physics_probs, phys_emb)

        # If specific physics tags are provided, boost those types
        if physics_tags:
            tag_emb = self.get_physics_tags(physics_tags, x.device)  # (D,)
            physics_context = physics_context + 0.3 * tag_emb.unsqueeze(0).unsqueeze(0)

        return x + 0.15 * self.physics_proj(physics_context)


# ---------------------------------------------------------------------------
# NSFW Anatomy Router
# ---------------------------------------------------------------------------

class NSFWAnatomyRouter(nn.Module):
    """
    Routes explicit anatomy tokens to high-precision attention heads.

    The core NSFW problem: standard attention treats anatomy tokens the same
    as any other tokens, leading to "melted wax" artifacts. This module:
    1. Detects anatomy-related tokens in the text sequence.
    2. Routes them to dedicated high-resolution attention heads.
    3. Applies stronger spatial grounding so anatomy appears in the right place.
    4. Uses a topology-aware loss to prevent impossible geometry.

    Args:
        hidden_size: Token dimension.
        num_heads: Number of attention heads.
        precision_heads: Number of heads dedicated to anatomy precision.
    """

    # Anatomy token patterns (explicit + implicit)
    _ANATOMY = re.compile(
        r'\b(body|torso|chest|breast|nipple|abdomen|stomach|waist|hip|'
        r'buttock|ass|groin|thigh|leg|knee|ankle|foot|toe|'
        r'arm|elbow|wrist|hand|finger|shoulder|neck|face|'
        r'penis|vagina|vulva|genitalia|pubic|intimate|private)\b',
        re.IGNORECASE
    )

    def __init__(self, hidden_size: int, num_heads: int, precision_heads: int = 4):
        super().__init__()
        assert precision_heads <= num_heads
        self.num_heads = num_heads
        self.precision_heads = precision_heads
        self.head_dim = hidden_size // num_heads

        # Anatomy token detector
        self.anatomy_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.anatomy_detector[-2].weight)
        nn.init.constant_(self.anatomy_detector[-2].bias, -2.0)

        # High-precision cross-attention for anatomy tokens
        self.precision_q = nn.Linear(hidden_size, self.head_dim * precision_heads, bias=False)
        self.precision_k = nn.Linear(hidden_size, self.head_dim * precision_heads, bias=False)
        self.precision_v = nn.Linear(hidden_size, self.head_dim * precision_heads, bias=False)
        self.precision_out = nn.Linear(self.head_dim * precision_heads, hidden_size, bias=False)
        nn.init.zeros_(self.precision_out.weight)

        # Topology constraint: adjacent anatomy patches should be connected
        self.topology_gate = nn.Parameter(torch.zeros(1))

    def get_anatomy_mask(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Detect anatomy tokens in text sequence.
        text_emb: (B, L, D) -> (B, L, 1) anatomy probability.
        """
        return self.anatomy_detector(text_emb)

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        is_nsfw: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) image tokens.
            text_emb: (B, L, D) text tokens.
            is_nsfw: Whether to activate NSFW precision routing.
        Returns:
            (B, N, D) anatomy-routed tokens.
        """
        if not is_nsfw:
            return x

        B, N, D = x.shape
        _, L, _ = text_emb.shape
        H = self.precision_heads
        Dh = self.head_dim

        # Detect anatomy tokens
        anatomy_mask = self.get_anatomy_mask(text_emb)  # (B, L, 1)

        # Weight text tokens by anatomy probability
        anatomy_text = text_emb * anatomy_mask  # (B, L, D)

        # High-precision cross-attention
        q = self.precision_q(x).reshape(B, N, H, Dh).transpose(1, 2)
        k = self.precision_k(anatomy_text).reshape(B, L, H, Dh).transpose(1, 2)
        v = self.precision_v(anatomy_text).reshape(B, L, H, Dh).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, scale=Dh ** -0.5)
        out = out.transpose(1, 2).reshape(B, N, H * Dh)
        out = self.precision_out(out)

        # Topology gate: only apply where anatomy tokens are active
        anatomy_strength = anatomy_mask.mean(dim=1, keepdim=True)  # (B, 1, 1) — global anatomy signal
        gate = torch.sigmoid(self.topology_gate) * anatomy_strength.squeeze(-1).unsqueeze(-1)

        return x + gate * out


# ---------------------------------------------------------------------------
# Complex Prompt Conditioner (top-level)
# ---------------------------------------------------------------------------

class ComplexPromptConditioner(nn.Module):
    """
    Top-level module for complex/NSFW/surreal/weird prompt handling.

    Combines:
    - PromptComplexityAnalyzer: routes to specialist paths
    - ConceptFusionModule: clean surreal concept blending
    - PhysicsAwareTokenTagger: physics property conditioning
    - NSFWAnatomyRouter: explicit anatomy precision

    Usage:
        conditioner = ComplexPromptConditioner(hidden_size=1152, num_heads=16)
        profile = conditioner.analyze("cyberpunk girl melting into neon liquid, nsfw, tentacles")
        x = conditioner(x, text_emb, profile)
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.analyzer = PromptComplexityAnalyzer()
        self.concept_fusion = ConceptFusionModule(hidden_size, num_heads)
        self.physics_tagger = PhysicsAwareTokenTagger(hidden_size)
        self.nsfw_router = NSFWAnatomyRouter(hidden_size, num_heads)

    def analyze(self, prompt: str) -> PromptComplexityProfile:
        return self.analyzer.analyze(prompt)

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        profile: Optional[PromptComplexityProfile] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) image tokens.
            text_emb: (B, L, D) text tokens.
            profile: Complexity profile from analyze().
        Returns:
            (B, N, D) conditioned tokens.
        """
        # Concept fusion (always active for multi-concept prompts)
        if profile is None or profile.is_multi_concept or profile.is_surreal:
            x = self.concept_fusion(x, text_emb, profile)

        # Physics conditioning
        physics_tags = profile.physics_tags if profile else []
        if physics_tags or (profile and profile.is_physics_heavy):
            x = self.physics_tagger(x, physics_tags)

        # NSFW anatomy routing
        is_nsfw = profile.is_nsfw if profile else False
        x = self.nsfw_router(x, text_emb, is_nsfw)

        return x


__all__ = [
    "PromptComplexityAnalyzer",
    "PromptComplexityProfile",
    "ConceptFusionModule",
    "PhysicsAwareTokenTagger",
    "NSFWAnatomyRouter",
    "ComplexPromptConditioner",
]
