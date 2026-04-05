"""
Register Tokens for Vision Transformers (arxiv 2309.16588 + 2602.22394).

Problem: ViTs develop "artifact" tokens — high-norm outlier patches that
accumulate global information and corrupt local attention maps.

Solution: Add a small set of learnable "register" tokens that act as
global memory slots. The model routes global/background information into
these registers, leaving patch tokens to focus on local structure.

Benefits:
  - Cleaner, more interpretable attention maps
  - Better local feature quality
  - Improved generation of fine details
  - No extra inference cost (registers are discarded after the final block)

Usage:
    reg = RegisterTokens(hidden_size=1152, num_registers=8)
    x_with_regs = reg.prepend(x)          # (B, N+R, D)
    x_with_regs = transformer_blocks(x_with_regs)
    x_clean = reg.strip(x_with_regs)      # (B, N, D)  — registers removed
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RegisterTokens(nn.Module):
    """
    Learnable register tokens prepended to the patch token sequence.

    Args:
        hidden_size: Transformer hidden dimension.
        num_registers: Number of register tokens (4–16 is typical).
        init_std: Initialization std for register embeddings.
    """

    def __init__(self, hidden_size: int, num_registers: int = 8, init_std: float = 0.02):
        super().__init__()
        self.num_registers = int(num_registers)
        self.registers = nn.Parameter(
            torch.randn(1, self.num_registers, int(hidden_size)) * float(init_std)
        )

    def prepend(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend register tokens to patch sequence.
        x: (B, N, D) -> (B, R+N, D)
        """
        B = x.shape[0]
        regs = self.registers.expand(B, -1, -1)
        return torch.cat([regs, x], dim=1)

    def strip(self, x: torch.Tensor) -> torch.Tensor:
        """Remove register tokens from the front of the sequence.
        x: (B, R+N, D) -> (B, N, D)
        """
        return x[:, self.num_registers:, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: prepend registers (call strip() after transformer blocks)."""
        return self.prepend(x)


class JumboToken(nn.Module):
    """
    A single wide "Jumbo" global token (arxiv 2502.15021).

    Wider than patch tokens, processed by its own wider FFN to increase
    model capacity without scaling all patch tokens. Acts as a global
    summary that attends to all patches.

    Args:
        hidden_size: Patch token dimension.
        jumbo_size: Jumbo token dimension (typically 2x–4x hidden_size).
        ffn_expansion: FFN expansion ratio for the jumbo-specific FFN.
    """

    def __init__(
        self,
        hidden_size: int,
        jumbo_size: int = 0,
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.jumbo_size = int(jumbo_size) if jumbo_size > 0 else int(hidden_size) * 2

        # Project patch dim <-> jumbo dim
        self.up_proj = nn.Linear(self.hidden_size, self.jumbo_size, bias=False)
        self.down_proj = nn.Linear(self.jumbo_size, self.hidden_size, bias=False)

        # Jumbo-specific FFN (wider capacity)
        h = self.jumbo_size * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(self.jumbo_size, h, bias=False),
            nn.GELU(),
            nn.Linear(h, self.jumbo_size, bias=False),
        )
        self.norm = nn.LayerNorm(self.jumbo_size)

        # Learnable initial embedding
        self.token = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)

        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.weight)

    def get_token(self, batch_size: int) -> torch.Tensor:
        """Return (B, 1, D) initial jumbo token in patch space."""
        return self.token.expand(batch_size, -1, -1)

    def process(self, jumbo: torch.Tensor) -> torch.Tensor:
        """
        Run the jumbo-specific FFN on the jumbo token after attention.
        jumbo: (B, 1, D) in patch space -> (B, 1, D)
        """
        z = self.up_proj(jumbo)          # (B, 1, J)
        z = self.norm(z + self.ffn(z))   # residual in jumbo space
        return self.down_proj(z)         # (B, 1, D)


__all__ = ["RegisterTokens", "JumboToken"]
