"""Token-level Mixture-of-Experts for DiT blocks (DiT-MoE style, conditioning-aware routing).

Every token is routed to its ``top_k`` experts by a linear gate over ``token + context_proj(c)``; the gate also
yields the Switch-style load-balancing loss ``E * sum(importance * load)``. ``MoERouter`` is the bare gate (one
router can be shared by every MoE layer of a block); ``MoEExperts`` adds the experts, and ``MoEFeedForward`` /
``MoEProjection`` build GELU-MLP / linear experts for it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    """Top-k gate: ``(B, N, C)`` tokens (+ optional ``(B, C)`` context) -> ``(top_idx, top_weights, aux_loss)``."""

    def __init__(self, hidden_size: int, *, num_experts: int, top_k: int = 2):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.hidden_size = hidden_size
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.context_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, routing_context: torch.Tensor | None = None):
        """``top_idx`` / ``top_weights`` are ``(B*N, top_k)``; ``aux_loss`` is a scalar."""
        if x.dim() != 3:
            raise ValueError(f"{type(self).__name__} expects x=(B,N,C) but got shape {tuple(x.shape)}")
        B, N, C = x.shape
        T = B * N
        x_flat = x.reshape(T, C)
        if routing_context is not None:
            if routing_context.dim() != 2 or routing_context.shape[0] != B or routing_context.shape[1] != C:
                raise ValueError(f"routing_context must be (B,C) with matching B,C; got {tuple(routing_context.shape)}")
            x_flat = x_flat + self.context_proj(routing_context).unsqueeze(1).expand(B, N, C).reshape(T, C)
        logits = self.gate(x_flat)
        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)
        top_weights = F.softmax(top_vals, dim=-1)
        # Router probabilities are differentiable, the expert load counts are not (Switch Transformer).
        importance = F.softmax(logits, dim=-1).sum(dim=0) / float(T)
        assign = torch.zeros(T, self.num_experts, device=x.device, dtype=importance.dtype)
        load = assign.scatter_(1, top_idx, 1.0).sum(dim=0) / float(T)
        return top_idx, top_weights, self.num_experts * (importance * load).sum()


class MoEExperts(MoERouter):
    """Top-k mixture over ``experts`` (each maps ``(n, C) -> (n, C)``); the last aux loss is in ``last_aux_loss``.

    ``router_override`` routes with a shared ``MoERouter`` instead of this module's own gate.
    """

    def __init__(self, hidden_size: int, experts: nn.ModuleList, *, top_k: int = 2):
        super().__init__(hidden_size, num_experts=len(experts), top_k=top_k)
        self.experts = experts
        self.last_aux_loss = None

    def forward(
        self,
        x: torch.Tensor,
        routing_context: torch.Tensor | None = None,
        router_override: MoERouter | None = None,
        report_aux_loss: bool = True,
    ) -> torch.Tensor:
        router = router_override if router_override is not None else super().forward
        top_idx, top_weights, aux_loss = router(x, routing_context=routing_context)
        self.last_aux_loss = aux_loss if report_aux_loss else None
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)
        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = top_idx == e
            if not mask.any():
                continue
            token_indices, k_indices = mask.nonzero(as_tuple=False).T
            y = self.experts[e](x_flat[token_indices])
            out.index_add_(0, token_indices, y * top_weights[token_indices, k_indices].unsqueeze(-1))
        return out.reshape(B, N, C)


class MoEFeedForward(MoEExperts):
    """MoE MLP: ``num_experts`` independent ``Linear -> act -> Linear`` feed-forwards."""

    def __init__(
        self,
        hidden_size: int,
        hidden_features: int,
        *,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.0,
        act_layer=nn.GELU,
    ):
        experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_features),
                act_layer(),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, hidden_size),
                nn.Dropout(dropout),
            )
            for _ in range(max(0, int(num_experts)))
        )
        super().__init__(hidden_size, experts, top_k=top_k)
        self.hidden_features = hidden_features


class MoEProjection(MoEExperts):
    """Sparse ``C -> C`` projection (attention output) over ``num_experts`` linear experts."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        act_layer=None,
    ):
        experts = []
        for _ in range(max(0, int(num_experts))):
            layers = [nn.Linear(hidden_size, hidden_size, bias=bias)]
            if act_layer is not None:
                layers.append(act_layer())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            experts.append(nn.Sequential(*layers))
        super().__init__(hidden_size, nn.ModuleList(experts), top_k=top_k)
