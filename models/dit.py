"""Base Diffusion Transformer (DiT) — class-conditional.

This is the original DiT architecture (Peebles & Xie, "Scalable Diffusion Models
with Transformers"). It operates on **latent patches**: an image latent of shape
``(B, C, H, W)`` is split into patches, flattened to a token sequence, processed
by a stack of transformer blocks, and reassembled back into a latent.

Conditioning (the diffusion timestep, and optionally a class label) is injected
through **adaLN-Zero**: instead of cross-attention, each block predicts per-channel
shift/scale/gate values from the conditioning vector and modulates its own
normalized activations. The "-Zero" part — initializing those modulation layers to
zero so each block starts as an identity function — is the key trick that makes
deep DiTs train stably.

This module is the class-conditional baseline and the source of reusable building
blocks (``TimestepEmbedder``, ``DiTBlock``, ``FinalLayer``). The text-to-image
variants live in ``models/dit_text.py``.
"""

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

from .moe import MoEFeedForward


def modulate(x, shift, scale):
    """Apply adaLN feature-wise affine modulation: ``x * (1 + scale) + shift``.

    ``shift``/``scale`` are per-sample, per-channel vectors ``(B, D)`` produced from
    the conditioning signal; ``unsqueeze(1)`` broadcasts them across the token axis.
    Using ``(1 + scale)`` means a zero-initialized modulation network leaves ``x``
    untouched — the foundation of the adaLN-Zero scheme.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds a scalar diffusion timestep into a hidden-size conditioning vector.

    Builds a sinusoidal embedding of the timestep (same idea as Transformer
    positional encodings) and passes it through a small MLP. The result is added to
    the label/text embedding to form the conditioning ``c`` consumed by every block.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Sinusoidal embedding of timesteps ``t`` -> ``(len(t), dim)``.

        Each dimension is a sinusoid of a different frequency (geometric spacing up
        to ``max_period``), giving the network a smooth, unique code per timestep.
        """
        half = dim // 2
        device = t.device if isinstance(t, torch.Tensor) else torch.device("cpu")
        freqs = torch.exp(-torch.arange(half, device=device, dtype=torch.float32) * (float(np.log(max_period)) / half))
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:  # odd dim: pad the last column so shapes line up
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels, with built-in label dropout for classifier-free guidance.

    For CFG the model must also learn an *unconditional* path. We reserve one extra
    embedding row as a "null" class and, during training, randomly replace real
    labels with it (probability ``dropout_prob``). At inference, passing the null
    label yields the unconditional prediction used in the CFG formula.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg = dropout_prob > 0
        # +1 embedding row for the "null"/unconditional class when CFG is enabled.
        self.embedding_table = nn.Embedding(num_classes + (1 if use_cfg else 0), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, force_drop_ids=None):
        if force_drop_ids is not None:
            # Explicitly force-drop the specified samples (e.g. all-ones mask for unconditional CFG).
            labels = torch.where(
                force_drop_ids.bool(),
                torch.full_like(labels, self.num_classes),
                labels,
            )
        elif train and self.dropout_prob > 0:
            # Random label dropout during training so the null class is well-trained.
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    """One DiT transformer block: adaLN-Zero modulated self-attention + MLP/MoE.

    Standard pre-norm transformer block, except the LayerNorms are *non-affine* and
    their shift/scale — plus a residual ``gate`` — come from the conditioning vector
    via ``adaLN_modulation`` (which outputs 6 vectors: shift/scale/gate for both the
    attention and MLP sub-layers). The MLP may optionally be a Mixture-of-Experts
    feed-forward when ``moe_num_experts > 0``.
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
        # Non-affine norms: the affine shift/scale are supplied per-step by adaLN below.
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if moe_num_experts and int(moe_num_experts) > 0:
            self.mlp = MoEFeedForward(
                hidden_size,
                mlp_hidden_dim,
                num_experts=int(moe_num_experts),
                top_k=int(moe_top_k),
                dropout=0.0,
                act_layer=nn.GELU,
            )
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=nn.GELU,
                drop=0,
            )
        # Produces 6 * hidden_size: shift/scale/gate for attention and for the MLP.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        # Split the conditioning projection into the six modulation vectors.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # Gated residual attention on conditionally-modulated, normalized tokens.
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        if isinstance(self.mlp, MoEFeedForward):
            # MoE routes each token to experts using the conditioning as context.
            x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in, routing_context=c)
        else:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class FinalLayer(nn.Module):
    """Projects tokens back to patch pixels after a last adaLN modulation.

    Maps each token from ``hidden_size`` to ``patch_size**2 * out_channels`` so the
    sequence can be unpatchified into a latent image.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),  # only shift/scale here (no gate)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """Build fixed 2D sin/cos positional embeddings for a ``grid_size x grid_size`` patch grid.

    Returns ``(grid_size**2, embed_dim)`` (optionally prefixed with zero rows for
    cls/extra tokens). Half the channels encode the row position and half the column,
    so each patch gets a unique, non-learned spatial code.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token or extra_tokens > 0:
        emb = np.concatenate([np.zeros([extra_tokens, embed_dim]), emb], axis=0)
    return emb


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sin/cos embedding for a flat list of positions -> ``(len(pos), embed_dim)``."""
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class DiT(nn.Module):
    """Class-conditional Diffusion Transformer denoiser.

    Pipeline (see ``forward``): patch-embed the noisy latent and add fixed 2D
    positional embeddings, build a conditioning vector ``c = timestep + label``,
    run the token sequence through ``depth`` ``DiTBlock``s, project with
    ``FinalLayer``, and unpatchify back to a latent.

    Args:
        input_size: Spatial size (H=W) of the input latent.
        patch_size: Side length of each square patch; smaller -> more tokens, more compute.
        in_channels: Latent channel count (e.g. 4 for a typical VAE latent).
        hidden_size: Transformer width.
        depth: Number of ``DiTBlock``s.
        num_heads: Attention heads per block.
        mlp_ratio: MLP hidden expansion factor.
        class_dropout_prob: Label-dropout probability for classifier-free guidance.
        num_classes: Number of conditioning classes.
        learn_sigma: If True, also predict per-channel variance, doubling out_channels.
        moe_num_experts / moe_top_k: Enable Mixture-of-Experts FFNs when > 0.
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
        # learn_sigma doubles outputs: first half is the mean prediction, second the variance.
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Positional embeddings are fixed (requires_grad=False); filled in initialize_weights().
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
        # Collect all MoE modules that expose `last_aux_loss` (FFN MoE + projection MoE).
        self._moe_layers = [m for m in self.modules() if hasattr(m, "last_aux_loss")]
        self.moe_aux_loss = None

    def initialize_weights(self):
        """Apply DiT's prescribed init, including the critical adaLN/output zero-init.

        Xavier-inits linears, loads the fixed sin/cos positional embeddings, then
        zeros every block's final adaLN layer and the output projection. Zeroing
        adaLN makes each block start as an identity map (adaLN-Zero), which is what
        lets very deep DiTs train without diverging.
        """

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.view([self.x_embedder.proj.weight.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # adaLN-Zero: start each block (and the final layer) as identity / no-op.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """Inverse of patch embedding: ``(B, num_patches, p*p*C)`` -> ``(B, C, H, W)``."""
        c, p = self.out_channels, self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape((x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y):
        """Denoise latent ``x`` at timestep ``t`` conditioned on class label ``y``.

        Shapes: ``x`` is ``(B, in_channels, H, W)``, ``t`` is ``(B,)``, ``y`` is
        ``(B,)``. Returns ``(B, out_channels, H, W)``.
        """
        # Clear aux loss each forward to avoid stale values.
        if self._moe_layers:
            self.moe_aux_loss = None
            for m in self._moe_layers:
                m.last_aux_loss = None
        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = t_emb + y_emb  # combined conditioning shared by every block
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        # Sum MoE load-balancing losses so the trainer can add them to the main loss.
        if self._moe_layers:
            aux_vals = [m.last_aux_loss for m in self._moe_layers if m.last_aux_loss is not None]
            self.moe_aux_loss = sum(aux_vals) if aux_vals else None
        return self.unpatchify(x)


# --- Preset constructors -------------------------------------------------------
# "DiT-XL/2" == the XL config with patch_size 2. Patch 2 = more tokens/detail but
# heavier; patch 4 = cheaper. Registered in DiT_models for lookup by name.


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
}
