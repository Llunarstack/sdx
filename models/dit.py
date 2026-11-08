"""Class-conditional Diffusion Transformer (Peebles & Xie, "Scalable Diffusion Models with Transformers") and the
shared DiT building blocks.

A latent ``(B, C, H, W)`` is patchified into tokens, run through transformer blocks and unpatchified.
Timestep and class label enter through adaLN-Zero: each block predicts per-channel shift/scale/gate from the
conditioning vector, and those layers start at zero so every block begins as the identity, which is what
lets deep DiTs train stably.

Shared pieces: ``timestep_embedding``, ``TimestepEmbedder``, ``DiTBlock``, ``FinalLayer``, ``LayerScale``,
sin/cos position tables, and the token-level Mixture-of-Experts (``MoERouter`` / ``MoEExperts`` /
``MoEFeedForward`` / ``MoEProjection``: DiT-MoE style, conditioning-aware top-k routing with a Switch-style
load-balancing loss). The text-to-image variants live in ``models/dit_text.py``.
"""

import functools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed


def modulate(x, shift, scale):
    """adaLN modulation ``x * (1 + scale) + shift`` with per-sample ``(B, D)`` shift/scale over ``(B, N, D)``.

    ``1 + scale`` makes a zero-initialized modulation network the identity (adaLN-Zero).
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@functools.lru_cache(maxsize=32)
def _sinusoid_freqs(half: int, max_period: float, device: torch.device) -> torch.Tensor:
    return torch.exp(-math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / half)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
    """Sinusoidal embedding of scalars ``t`` (any shape) -> ``(*t.shape, dim)``.

    Each channel is a sinusoid of a different frequency (geometric spacing up to ``max_period``), giving the
    network a smooth, unique code per value; the same table embeds timesteps and PixArt size conditions.
    """
    args = t.float()[..., None] * _sinusoid_freqs(dim // 2, float(max_period), t.device)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    return embedding


def xavier_init_linears(module: nn.Module) -> None:
    """DiT base init: Xavier-uniform weights and zero bias for every ``nn.Linear`` under ``module``."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class LayerScale(nn.Module):
    """Per-channel learnable residual gain ``gamma`` (CaiT, Touvron et al. 2021).

    ``init_value <= 0`` means off: ``gamma`` starts at 1. It must never start at 0 inside an adaLN-Zero block,
    where a zero gate times a zero ``gamma`` leaves both with zero gradient and the branch never trains.
    """

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        init = float(init_value) if float(init_value) > 0.0 else 1.0
        self.gamma = nn.Parameter(torch.full((int(dim),), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.to(dtype=x.dtype)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer MLP -> ``(B, hidden_size)``."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """Class-label embedding with label dropout for classifier-free guidance.

    With ``dropout_prob > 0`` the table has one extra "null" row (index ``num_classes``); training replaces
    labels with it at that rate, and passing it at inference gives the unconditional prediction.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + (1 if dropout_prob > 0 else 0), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, force_drop_ids=None):
        if force_drop_ids is not None:
            drop = force_drop_ids.bool()
        elif train and self.dropout_prob > 0:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            return self.embedding_table(labels)
        return self.embedding_table(torch.where(drop, torch.full_like(labels, self.num_classes), labels))


class DiTBlock(nn.Module):
    """Pre-norm block whose non-affine LayerNorms get shift/scale and a residual gate from ``c`` (adaLN-Zero).

    ``adaLN_modulation`` emits six vectors: shift/scale/gate for the attention and for the MLP (an MoE
    feed-forward when ``moe_num_experts > 0``).
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        *,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if moe_num_experts > 0:
            self.mlp = MoEFeedForward(
                hidden_size, mlp_hidden_dim, num_experts=int(moe_num_experts), top_k=int(moe_top_k), act_layer=nn.GELU
            )
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        if isinstance(self.mlp, MoEFeedForward):
            mlp_out = self.mlp(mlp_in, routing_context=c)
        else:
            mlp_out = self.mlp(mlp_in)
        return x + gate_mlp.unsqueeze(1) * mlp_out


class FinalLayer(nn.Module):
    """adaLN shift/scale (no gate), then a linear map of each token to ``patch_size**2 * out_channels``."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def get_2d_sincos_pos_embed(embed_dim, grid_size, base_size=None):
    """Fixed 2D sin/cos positional embeddings (MAE) for a ``grid_size`` (int or ``(h, w)``) patch grid.

    Returns ``(h*w, embed_dim)`` in raster order; half the channels encode the column and half the row. ``base_size`` applies PixArt-alpha
    position interpolation (Chen et al. 2023): each axis is rescaled to span ``base_size`` positions, so
    any grid stays inside the coordinate range the model was trained on. At ``h == w == base_size`` it is
    the plain table.
    """
    gh, gw = (grid_size, grid_size) if isinstance(grid_size, int) else (int(grid_size[0]), int(grid_size[1]))
    grid_h = np.arange(gh, dtype=np.float32)
    grid_w = np.arange(gw, dtype=np.float32)
    if base_size:
        grid_h *= np.float32(base_size / gh)
        grid_w *= np.float32(base_size / gw)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, gh, gw])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


@functools.lru_cache(maxsize=16)
def sincos_pos_table(embed_dim: int, gh: int, gw: int, base_size: int, device, dtype) -> torch.Tensor:
    """Cached ``(1, gh*gw, embed_dim)`` 2D sin/cos table (``base_size`` 0 = no PixArt interpolation)."""
    pe = get_2d_sincos_pos_embed(embed_dim, (gh, gw), base_size=base_size)
    return torch.from_numpy(pe).to(device=device, dtype=dtype).unsqueeze(0)


def grid_pos_embed(pos_embed: torch.Tensor, base_size: int, gh: int, gw: int) -> torch.Tensor:
    """Positional embedding ``(1, gh*gw, D)`` for a ``gh x gw`` patch grid.

    The native ``base_size`` grid uses the stored ``pos_embed`` buffer (so old checkpoints behave exactly
    as before); any other grid gets interpolated sin/cos, built once per shape and device.
    """
    if gh == base_size and gw == base_size:
        return pos_embed
    return sincos_pos_table(pos_embed.shape[-1], gh, gw, base_size, pos_embed.device, pos_embed.dtype)


def unpatchify(x: torch.Tensor, patch_size: int, channels: int, h: int, w: int) -> torch.Tensor:
    """Patch tokens ``(B, h*w, p*p*C)`` -> latent ``(B, C, h*p, w*p)`` (inverse of ``PatchEmbed``)."""
    p = int(patch_size)
    x = x.reshape(x.shape[0], h, w, p, p, channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(x.shape[0], channels, h * p, w * p)


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sin/cos embedding for a flat list of positions -> ``(len(pos), embed_dim)``."""
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class DiT(nn.Module):
    """Class-conditional DiT denoiser: patch tokens + sin/cos positions, ``c = timestep + label``, ``depth``
    ``DiTBlock``s, ``FinalLayer``, unpatchify.

    ``input_size`` is the native latent side the stored position table is built for; any ``H, W`` divisible
    by ``patch_size`` works (other grids use interpolated positions). ``learn_sigma`` doubles the output
    channels (mean, then variance); ``moe_num_experts > 0`` makes the MLPs Mixture-of-Experts.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        moe_num_experts: int = 0,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self._moe_num_experts = int(moe_num_experts) if moe_num_experts is not None else 0
        self._moe_top_k = int(moe_top_k) if moe_top_k is not None else 2
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    moe_num_experts=self._moe_num_experts,
                    moe_top_k=self._moe_top_k,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self._moe_layers = [m for m in self.modules() if hasattr(m, "last_aux_loss")]
        self.moe_aux_loss = None

    def initialize_weights(self):
        """DiT init: Xavier linears, sin/cos positions, and zeroed adaLN / output layers (adaLN-Zero)."""
        xavier_init_linears(self)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.view([self.x_embedder.proj.weight.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y):
        """``x`` ``(B, in_channels, H, W)``, ``t`` and ``y`` ``(B,)`` -> ``(B, out_channels, H, W)``."""
        if self._moe_layers:
            self.moe_aux_loss = None
            for m in self._moe_layers:
                m.last_aux_loss = None
        hp, wp = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.x_embedder(x) + grid_pos_embed(self.pos_embed, self.x_embedder.grid_size[0], hp, wp)
        c = self.t_embedder(t) + self.y_embedder(y, self.training)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        if self._moe_layers:
            aux_vals = [m.last_aux_loss for m in self._moe_layers if m.last_aux_loss is not None]
            self.moe_aux_loss = sum(aux_vals) if aux_vals else None
        return unpatchify(x, self.patch_size, self.out_channels, hp, wp)


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
}


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
        # Group the (token, slot) assignments by expert with one host sync (the per-expert counts); the stable
        # sort keeps each expert's rows in the same (token, slot) order as a per-expert nonzero() would.
        flat = top_idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        token_ids, slots = order // top_idx.shape[1], order % top_idx.shape[1]
        start = 0
        for e, count in enumerate(torch.bincount(flat, minlength=self.num_experts).tolist()):
            if count:
                tok, k = token_ids[start : start + count], slots[start : start + count]
                y = self.experts[e](x_flat[tok])
                out.index_add_(0, tok, y * top_weights[tok, k].unsqueeze(-1))
            start += count
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
