"""World Model Backbone: transformer with canvas-structured attention.

The backbone is a standard transformer that operates on the canvas tensor.
The canvas layout determines which positions attend to which other positions
via the topology-derived attention mask. The model itself doesn't know about
modalities — it just sees positions with embeddings and an attention mask.

Two modes:
1. From scratch: train a small transformer on the canvas
2. Grafted: take a pretrained diffusion transformer (CogVideoX, etc.)
   and graft looped attention blocks onto it

For the world model, we typically train from scratch because:
- The data is heterogeneous time series, not video
- We want to control the architecture precisely
- The canvas positions represent fundamentally different things than video patches

But the grafting path exists for when we want to leverage video priors.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for 1D sequences."""

    def __init__(self, d_model: int, max_len: int = 65536):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class WorldModelBlock(nn.Module):
    """Single transformer block with pre-norm and optional masked attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out

        # Pre-norm feedforward
        x = x + self.ff(self.norm2(x))
        return x


class WorldModelBackbone(nn.Module):
    """Transformer backbone for world model training.

    Takes a canvas tensor (B, N, d_model) and produces output of same shape.
    The attention mask from the canvas topology constrains which positions
    can attend to which.

    Args:
        d_model: Latent dimension per position.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        d_ff: Feedforward hidden dimension. Default: 4 * d_model.
        n_positions: Total canvas positions (for positional encoding).
        dropout: Dropout rate.
        n_loops: Number of weight-sharing loops per block (from canvas-engineering).
            1 = standard transformer, 3 = optimal looped attention.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int | None = None,
        n_positions: int = 8192,
        dropout: float = 0.1,
        n_loops: int = 1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=n_positions)

        self.blocks = nn.ModuleList([
            WorldModelBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.n_loops = n_loops
        if n_loops > 1:
            # Learned loop embeddings (zero-init for safety)
            self.loop_embeddings = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, d_model))
                for _ in range(n_loops)
            ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) canvas tensor.
            mask: (N, N) additive attention mask from topology.

        Returns:
            (B, N, d_model) output tensor.
        """
        x = self.pos_enc(x)

        if self.n_loops > 1:
            for loop_idx in range(self.n_loops):
                loop_emb = self.loop_embeddings[loop_idx]
                x_loop = x + loop_emb
                for block in self.blocks:
                    x_loop = block(x_loop, mask=mask)
                x = x_loop
        else:
            for block in self.blocks:
                x = block(x, mask=mask)

        return self.final_norm(x)


def build_world_model(
    bound_schema,
    n_layers: int = 6,
    n_heads: int = 4,
    d_ff: int | None = None,
    n_loops: int = 3,
    dropout: float = 0.1,
) -> WorldModelBackbone:
    """Build a WorldModelBackbone sized for a compiled schema.

    Args:
        bound_schema: BoundSchema from compile_schema / project().
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        d_ff: Feedforward dim. Default: 4 * d_model.
        n_loops: Looped attention iterations (3 is optimal).
        dropout: Dropout rate.

    Returns:
        WorldModelBackbone ready for training.
    """
    return WorldModelBackbone(
        d_model=bound_schema.layout.d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        n_positions=bound_schema.layout.num_positions,
        dropout=dropout,
        n_loops=n_loops,
    )
