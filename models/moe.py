import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    """
    Shared expert router producing top-k expert assignments.

    Used to mimic DiT-MoE "shared routing": multiple MoE submodules in a block
    can reuse the same router parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        num_experts: int,
        top_k: int = 2,
    ):
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

    def forward(
        self,
        x: torch.Tensor,
        routing_context: torch.Tensor = None,
    ):
        """
        Args:
            x: (B,N,C)
            routing_context: (B,C)
        Returns:
            top_idx: (T,K)
            top_weights: (T,K)
            aux_loss: scalar tensor
        """
        if x.dim() != 3:
            raise ValueError(f"MoERouter expects x=(B,N,C) but got shape {tuple(x.shape)}")
        B, N, C = x.shape
        if C != self.hidden_size:
            raise ValueError(f"MoERouter hidden size mismatch: got C={C}, expected {self.hidden_size}")
        T = B * N
        x_flat = x.reshape(T, C)

        if routing_context is not None:
            if routing_context.dim() != 2 or routing_context.shape[0] != B or routing_context.shape[1] != C:
                raise ValueError(f"routing_context must be (B,C) with matching B,C; got {tuple(routing_context.shape)}")
            ctx = self.context_proj(routing_context).unsqueeze(1).expand(B, N, C).reshape(T, C)
            x_flat_for_routing = x_flat + ctx
        else:
            x_flat_for_routing = x_flat

        logits = self.gate(x_flat_for_routing)  # (T,E)
        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)  # (T,K)
        top_weights = F.softmax(top_vals, dim=-1)  # (T,K)

        # Aux loss: importance * load (Switch-like)
        probs = F.softmax(logits, dim=-1)  # (T,E)
        importance = probs.sum(dim=0) / float(T)  # (E,)
        assign = torch.zeros(T, self.num_experts, device=x.device, dtype=probs.dtype)
        assign.scatter_(1, top_idx, 1.0)
        load = assign.sum(dim=0) / float(T)  # (E,)
        aux_loss = self.num_experts * (importance * load).sum()
        return top_idx, top_weights, aux_loss


class MoEFeedForward(nn.Module):
    """
    Simple MoE feed-forward (MLP-only MoE).

    - Input/Output: (B, N, C)
    - Each token is routed to `top_k` experts via a gating network.
    - Optional auxiliary load-balancing loss is computed and exposed as `last_aux_loss`.

    Notes:
    - This implementation focuses on correctness + minimal integration effort.
    - Balance loss uses router probabilities (differentiable) and expert "load" counts (non-differentiable),
      which is standard for Switch/Router-style MoE.
    """

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
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")

        self.hidden_size = hidden_size
        self.hidden_features = hidden_features
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)

        # Router: token -> expert logits
        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False)
        # Make routing explicitly conditioning-aware (timestep/class info).
        # DiT provides `c` to blocks; we project it and add to token embeddings for routing.
        self.context_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Experts: independent FFNs
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_features),
                    act_layer(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_features, hidden_size),
                    nn.Dropout(dropout),
                )
                for _ in range(self.num_experts)
            ]
        )

        # Exposed for training loop.
        self.last_aux_loss = None

    def forward(
        self,
        x: torch.Tensor,
        routing_context: torch.Tensor = None,
        router_override: MoERouter = None,
        report_aux_loss: bool = True,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"MoEFeedForward expects (B,N,C) but got shape {tuple(x.shape)}")
        B, N, C = x.shape
        T = B * N
        x_flat = x.reshape(T, C)

        if router_override is not None:
            top_idx, top_weights, aux_loss = router_override(x, routing_context=routing_context)
            self.last_aux_loss = aux_loss if report_aux_loss else None
        else:
            if routing_context is not None:
                if routing_context.dim() != 2 or routing_context.shape[0] != B or routing_context.shape[1] != C:
                    raise ValueError(
                        f"routing_context must be (B,C) with matching B,C; got {tuple(routing_context.shape)}"
                    )
                ctx = self.context_proj(routing_context).unsqueeze(1).expand(B, N, C).reshape(T, C)
                x_flat_for_routing = x_flat + ctx
            else:
                x_flat_for_routing = x_flat

            # Router logits and top-k routing
            logits = self.gate(x_flat_for_routing)  # (T, E)
            top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)  # both (T, K)
            top_weights = F.softmax(top_vals, dim=-1)  # (T, K)

            # Aux loss (optional): importance * load (Switch-like).
            probs = F.softmax(logits, dim=-1)  # (T, E) differentiable
            importance = probs.sum(dim=0) / float(T)  # (E)
            assign = torch.zeros(T, self.num_experts, device=x.device, dtype=probs.dtype)
            assign.scatter_(1, top_idx, 1.0)
            load = assign.sum(dim=0) / float(T)  # (E)
            aux_loss = self.num_experts * (importance * load).sum()
            self.last_aux_loss = aux_loss if report_aux_loss else None

        # Dispatch to experts and combine
        out_flat = torch.zeros(T, C, device=x.device, dtype=x.dtype)
        # For each expert e, gather tokens assigned to it (via top_idx match).
        for e in range(self.num_experts):
            mask = top_idx == e  # (T, K)
            if not mask.any():
                continue
            token_indices, k_indices = mask.nonzero(as_tuple=False).T
            x_sel = x_flat[token_indices]  # (n_sel, C)
            y_sel = self.experts[e](x_sel)  # (n_sel, C)
            w_sel = top_weights[token_indices, k_indices].unsqueeze(-1)  # (n_sel, 1)
            out_flat.index_add_(0, token_indices, y_sel * w_sel)

        return out_flat.reshape(B, N, C)


class MoEExperts(nn.Module):
    """
    Generic MoE wrapper: given a list of expert modules, route tokens to top-k experts.

    Experts must implement `forward(x)` where x is (n_tokens, hidden_size) and output
    is the same shape (n_tokens, hidden_size).
    """

    def __init__(self, hidden_size: int, experts: nn.ModuleList, *, top_k: int = 2):
        super().__init__()
        if len(experts) <= 0:
            raise ValueError("MoEExperts requires at least 1 expert")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if top_k > len(experts):
            raise ValueError("top_k cannot exceed num_experts")

        self.hidden_size = hidden_size
        self.experts = experts
        self.num_experts = len(experts)
        self.top_k = int(top_k)

        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.context_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_aux_loss = None

    def forward(
        self,
        x: torch.Tensor,
        routing_context: torch.Tensor = None,
        router_override: MoERouter = None,
        report_aux_loss: bool = True,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"MoEExperts expects (B,N,C) but got shape {tuple(x.shape)}")
        B, N, C = x.shape
        T = B * N
        x_flat = x.reshape(T, C)

        if router_override is not None:
            top_idx, top_weights, aux_loss = router_override(x, routing_context=routing_context)
            self.last_aux_loss = aux_loss if report_aux_loss else None
        else:
            if routing_context is not None:
                if routing_context.dim() != 2 or routing_context.shape[0] != B or routing_context.shape[1] != C:
                    raise ValueError(
                        f"routing_context must be (B,C) with matching B,C; got {tuple(routing_context.shape)}"
                    )
                ctx = self.context_proj(routing_context).unsqueeze(1).expand(B, N, C).reshape(T, C)
                x_flat_for_routing = x_flat + ctx
            else:
                x_flat_for_routing = x_flat

            logits = self.gate(x_flat_for_routing)  # (T, E)
            top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)  # (T, K)
            top_weights = F.softmax(top_vals, dim=-1)  # (T, K)

            probs = F.softmax(logits, dim=-1)  # (T,E)
            importance = probs.sum(dim=0) / float(T)
            assign = torch.zeros(T, self.num_experts, device=x.device, dtype=probs.dtype)
            assign.scatter_(1, top_idx, 1.0)
            load = assign.sum(dim=0) / float(T)
            aux_loss = self.num_experts * (importance * load).sum()
            self.last_aux_loss = aux_loss if report_aux_loss else None

        out_flat = torch.zeros(T, C, device=x.device, dtype=x.dtype)
        for e in range(self.num_experts):
            mask = top_idx == e
            if not mask.any():
                continue
            token_indices, k_indices = mask.nonzero(as_tuple=False).T
            x_sel = x_flat[token_indices]
            y_sel = self.experts[e](x_sel)
            w_sel = top_weights[token_indices, k_indices].unsqueeze(-1)
            out_flat.index_add_(0, token_indices, y_sel * w_sel)

        return out_flat.reshape(B, N, C)


class MoEProjection(nn.Module):
    """
    MoE sparse projection: each token routes to top-k linear experts for a (C->C) projection.
    This is intended for attention output projections to mimic DiT-MoE style sparse compute.
    """

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

        # Expert projections (optional activation kept minimal for stability).
        experts = []
        for _ in range(self.num_experts):
            layers = [nn.Linear(hidden_size, hidden_size, bias=bias)]
            if act_layer is not None:
                layers.append(act_layer())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            experts.append(nn.Sequential(*layers))
        self.experts = nn.ModuleList(experts)

        self.last_aux_loss = None

    def forward(
        self,
        x: torch.Tensor,
        routing_context: torch.Tensor = None,
        router_override: MoERouter = None,
        report_aux_loss: bool = True,
    ) -> torch.Tensor:
        # x: (B,N,C)
        if x.dim() != 3:
            raise ValueError(f"MoEProjection expects (B,N,C) but got shape {tuple(x.shape)}")
        B, N, C = x.shape
        T = B * N
        x_flat = x.reshape(T, C)

        if router_override is not None:
            top_idx, top_weights, aux_loss = router_override(x, routing_context=routing_context)
            self.last_aux_loss = aux_loss if report_aux_loss else None
        else:
            if routing_context is not None:
                if routing_context.dim() != 2 or routing_context.shape[0] != B or routing_context.shape[1] != C:
                    raise ValueError(
                        f"routing_context must be (B,C) with matching B,C; got {tuple(routing_context.shape)}"
                    )
                ctx = self.context_proj(routing_context).unsqueeze(1).expand(B, N, C).reshape(T, C)
                x_flat_for_routing = x_flat + ctx
            else:
                x_flat_for_routing = x_flat

            logits = self.gate(x_flat_for_routing)  # (T,E)
            top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)  # (T,K)
            top_weights = F.softmax(top_vals, dim=-1)  # (T,K)

            probs = F.softmax(logits, dim=-1)
            importance = probs.sum(dim=0) / float(T)
            assign = torch.zeros(T, self.num_experts, device=x.device, dtype=probs.dtype)
            assign.scatter_(1, top_idx, 1.0)
            load = assign.sum(dim=0) / float(T)
            aux_loss = self.num_experts * (importance * load).sum()
            self.last_aux_loss = aux_loss if report_aux_loss else None

        out_flat = torch.zeros(T, C, device=x.device, dtype=x.dtype)
        for e in range(self.num_experts):
            mask = top_idx == e  # (T,K)
            if not mask.any():
                continue
            token_indices, k_indices = mask.nonzero(as_tuple=False).T
            x_sel = x_flat[token_indices]  # (n_sel,C)
            y_sel = self.experts[e](x_sel)  # (n_sel,C)
            w_sel = top_weights[token_indices, k_indices].unsqueeze(-1)  # (n_sel,1)
            out_flat.index_add_(0, token_indices, y_sel * w_sel)

        return out_flat.reshape(B, N, C)
