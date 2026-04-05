from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .model_enhancements import RMSNorm, TokenFiLM


def concat_padding_masks(
    batch_size: int,
    segment_masks: List[Optional[torch.Tensor]],
    segment_lengths: List[int],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> Optional[torch.Tensor]:
    """
    Build ``src_key_padding_mask`` (B, S) for ``TransformerEncoder``: True = ignore position.

    *segment_masks[i]* is (B, segment_lengths[i]) or None (no padding in that segment).
    """
    if all(m is None for m in segment_masks):
        return None
    cols: List[torch.Tensor] = []
    for m, n in zip(segment_masks, segment_lengths):
        if m is None:
            cols.append(torch.zeros(batch_size, n, device=device, dtype=dtype))
        else:
            if m.shape != (batch_size, n):
                raise ValueError(f"padding mask expected ({batch_size}, {n}), got {tuple(m.shape)}")
            cols.append(m)
    return torch.cat(cols, dim=1)


class _VisionTextCrossAttention(nn.Module):
    """Vision tokens attend to text (+ optional extra) memory; pre-norm + residual."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if int(model_dim) % int(num_heads) != 0:
            raise ValueError("model_dim must be divisible by num_heads for cross-attention")
        self.q_norm = nn.LayerNorm(int(model_dim))
        self.kv_norm = nn.LayerNorm(int(model_dim))
        self.mha = nn.MultiheadAttention(
            int(model_dim), int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.drop = nn.Dropout(float(dropout))

    def forward(
        self,
        vision: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.q_norm(vision)
        k = self.kv_norm(memory)
        try:
            attn_out, _ = self.mha(q, k, k, key_padding_mask=memory_key_padding_mask, need_weights=False)
        except TypeError:
            attn_out, _ = self.mha(q, k, k, key_padding_mask=memory_key_padding_mask)
        return vision + self.drop(attn_out)


class NativeMultimodalTransformer(nn.Module):
    """
    Lightweight native multimodal token fusion block for SDX experiments.

    Concatenates projected vision + text (+ optional extra) tokens and runs a
    ``TransformerEncoder`` (pre-norm, GELU FFN). Optional **learnable modality
    embeddings** help the model separate streams without hand-built masks.

    Optional **cross-attention** lets vision query text (+ extra) before joint self-attention.

    Inputs:
      - vision_tokens: (B, Nv, Dv)
      - text_tokens: (B, Nt, Dt)
      - optional extra_tokens: (B, Ne, De)
    Output dict:
      - fused_vision_tokens: (B, Nv, D_model)
      - fused_text_tokens: (B, Nt, D_model)
      - fused_all_tokens: (B, Nv+Nt+Ne, D_model)
      - optional fused_extra_tokens when extra branch exists
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        model_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        proj_dropout: float = 0.0,
        extra_dim: int = 0,
        use_modality_embeddings: bool = True,
        cross_attn_heads: int = 0,
        output_norm: str = "layernorm",
        film_cond_dim: int = 0,
    ):
        super().__init__()
        if int(model_dim) % int(num_heads) != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})")
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.use_modality_embeddings = bool(use_modality_embeddings)
        self.extra_dim = int(extra_dim)
        self.cross_attn_heads = int(cross_attn_heads)
        self.film_cond_dim = int(film_cond_dim)

        self.vision_proj = nn.Linear(int(vision_dim), self.model_dim)
        self.text_proj = nn.Linear(int(text_dim), self.model_dim)
        self.extra_proj = nn.Linear(int(extra_dim), self.model_dim) if int(extra_dim) > 0 else None

        self.proj_drop = nn.Dropout(float(proj_dropout)) if float(proj_dropout) > 0 else nn.Identity()

        if self.use_modality_embeddings:
            self.modality_embed = nn.Parameter(torch.empty(3, self.model_dim))
            nn.init.normal_(self.modality_embed, std=0.02)
        else:
            self.register_parameter("modality_embed", None)

        if self.cross_attn_heads > 0:
            self.cross_attn = _VisionTextCrossAttention(self.model_dim, self.cross_attn_heads, dropout=dropout)
        else:
            self.cross_attn = None

        onorm = (output_norm or "layernorm").lower()
        if onorm not in ("layernorm", "rmsnorm"):
            raise ValueError("output_norm must be 'layernorm' or 'rmsnorm'")
        self._output_norm_type = onorm

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.model_dim * float(mlp_ratio)),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=int(num_layers), enable_nested_tensor=False
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

        if onorm == "rmsnorm":
            self.out_norm = RMSNorm(self.model_dim)
        else:
            self.out_norm = nn.LayerNorm(self.model_dim)

        if self.film_cond_dim > 0:
            self.vision_film = TokenFiLM(self.model_dim, self.film_cond_dim)
        else:
            self.vision_film = None

        self._reset_proj_weights()

    def _reset_proj_weights(self) -> None:
        for m in (self.vision_proj, self.text_proj):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.extra_proj is not None:
            nn.init.xavier_uniform_(self.extra_proj.weight)
            if self.extra_proj.bias is not None:
                nn.init.zeros_(self.extra_proj.bias)

    def _add_modality(self, x: torch.Tensor, modality_idx: int) -> torch.Tensor:
        if self.modality_embed is None:
            return x
        return x + self.modality_embed[modality_idx : modality_idx + 1, :].view(1, 1, self.model_dim)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        *,
        extra_tokens: Optional[torch.Tensor] = None,
        vision_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
        extra_padding_mask: Optional[torch.Tensor] = None,
        film_cond: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vision_padding_mask: (B, Nv) bool, True = padded vision token (ignored in self-attn).
            text_padding_mask: (B, Nt) bool.
            extra_padding_mask: (B, Ne) bool when ``extra_tokens`` is set.
            film_cond: (B, film_cond_dim) when ``film_cond_dim > 0``; modulates **vision** output.
        """
        if self.vision_film is not None:
            if film_cond is None:
                raise ValueError("film_cond is required when film_cond_dim > 0")
            if film_cond.shape[-1] != self.film_cond_dim:
                raise ValueError(f"film_cond dim {film_cond.shape[-1]} != film_cond_dim {self.film_cond_dim}")

        b, nv, _ = vision_tokens.shape
        _, nt, _ = text_tokens.shape

        v = self.proj_drop(self.vision_proj(vision_tokens))
        t = self.proj_drop(self.text_proj(text_tokens))
        v = self._add_modality(v, 0)
        t = self._add_modality(t, 1)

        ne = 0
        e: Optional[torch.Tensor] = None
        if extra_tokens is not None:
            if self.extra_proj is None:
                raise ValueError("extra_tokens provided but extra_dim was 0 at init")
            ne = extra_tokens.shape[1]
            e = self.proj_drop(self.extra_proj(extra_tokens))
            e = self._add_modality(e, 2)

        if self.cross_attn is not None:
            mem_parts: List[torch.Tensor] = [t]
            mem_masks: List[Optional[torch.Tensor]] = [text_padding_mask]
            mem_lens: List[int] = [nt]
            if e is not None:
                mem_parts.append(e)
                mem_masks.append(extra_padding_mask)
                mem_lens.append(ne)
            memory = torch.cat(mem_parts, dim=1)
            mem_pad = concat_padding_masks(b, mem_masks, mem_lens, device=v.device, dtype=torch.bool)
            v = self.cross_attn(v, memory, memory_key_padding_mask=mem_pad)

        seq_list: List[torch.Tensor] = [v, t]
        if e is not None:
            seq_list.append(e)

        x = torch.cat(seq_list, dim=1)

        pad_parts: List[Optional[torch.Tensor]] = [vision_padding_mask, text_padding_mask]
        len_parts: List[int] = [nv, nt]
        if e is not None:
            pad_parts.append(extra_padding_mask)
            len_parts.append(ne)

        key_padding = concat_padding_masks(b, pad_parts, len_parts, device=x.device, dtype=torch.bool)

        y = self.encoder(x, src_key_padding_mask=key_padding)
        y = self.out_norm(y)

        fv = y[:, :nv, :]
        ft = y[:, nv : nv + nt, :]
        fe = y[:, nv + nt :, :] if e is not None else None
        if self.vision_film is not None:
            fv = self.vision_film(fv, film_cond)

        if fe is not None:
            y_all = torch.cat([fv, ft, fe], dim=1)
        else:
            y_all = torch.cat([fv, ft], dim=1)

        out: Dict[str, torch.Tensor] = {
            "fused_vision_tokens": fv,
            "fused_text_tokens": ft,
            "fused_all_tokens": y_all,
        }
        if fe is not None:
            out["fused_extra_tokens"] = fe
        return out


__all__ = ["NativeMultimodalTransformer", "concat_padding_masks"]
