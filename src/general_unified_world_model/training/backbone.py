"""World Model Backbone: transformer with canvas-structured attention.

The backbone operates on the canvas tensor (B, N, d_model) and produces
output of the same shape. Three attention modes:

1. Dispatched attention (default when topology provided): per-connection
   attention dispatch using canvas-engineering's AttentionDispatcher.
   Each connection can use a different attention type (cross, linear,
   gated, pooling, etc.) as declared in the topology.

2. Masked attention (fallback): standard transformer with a single (N, N)
   additive mask from the topology. All connections use the same attention.

3. CogVideoX grafting: load a pretrained CogVideoX diffusion transformer
   and graft per-block loop embeddings + projection layers. Only ~0.1% of
   parameters are trainable. Uses full attention through frozen blocks.
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
    """Single transformer block with pre-norm and optional masked or dispatched attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 dispatcher=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dispatcher = dispatcher
        if dispatcher is None:
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
        # Pre-norm attention (dispatched or masked)
        normed = self.norm1(x)
        if self.dispatcher is not None:
            attn_out = self.dispatcher(normed)
        else:
            attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out

        # Pre-norm feedforward
        x = x + self.ff(self.norm2(x))
        return x


class WorldModelBackbone(nn.Module):
    """Transformer backbone for world model training.

    Takes a canvas tensor (B, N, d_model) and produces output of same shape.

    When topology and layout are provided, uses per-connection attention
    dispatch — each connection can use a different attention function type
    (cross_attention, linear_attention, gated, pooling, etc.).

    When only a mask is provided, falls back to standard masked attention.

    Args:
        d_model: Latent dimension per position.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        d_ff: Feedforward hidden dimension. Default: 4 * d_model.
        n_positions: Total canvas positions (for positional encoding).
        dropout: Dropout rate.
        n_loops: Number of weight-sharing loops per block.
            1 = standard transformer, 3 = optimal looped attention.
        topology: CanvasTopology for per-connection attention dispatch.
        layout: CanvasLayout for position index computation.
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
        topology=None,
        layout=None,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=n_positions)
        self.use_dispatch = topology is not None and layout is not None

        if self.use_dispatch:
            from canvas_engineering.dispatch import AttentionDispatcher
            # Each block gets its own dispatcher (separate attention params)
            dispatchers = [
                AttentionDispatcher(topology, layout, d_model, n_heads, dropout)
                for _ in range(n_layers)
            ]
            self.blocks = nn.ModuleList([
                WorldModelBlock(d_model, n_heads, d_ff, dropout, dispatcher=dispatchers[i])
                for i in range(n_layers)
            ])
        else:
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
                Ignored when using dispatched attention.

        Returns:
            (B, N, d_model) output tensor.
        """
        x = self.pos_enc(x)

        # When using dispatch, mask is ignored (dispatch handles connectivity)
        effective_mask = None if self.use_dispatch else mask

        if self.n_loops > 1:
            for loop_idx in range(self.n_loops):
                loop_emb = self.loop_embeddings[loop_idx]
                x_loop = x + loop_emb
                for block in self.blocks:
                    x_loop = block(x_loop, mask=effective_mask)
                x = x_loop
        else:
            for block in self.blocks:
                x = block(x, mask=effective_mask)

        return self.final_norm(x)


def build_world_model(
    bound_schema,
    n_layers: int = 6,
    n_heads: int = 4,
    d_ff: int | None = None,
    n_loops: int = 3,
    dropout: float = 0.1,
    use_dispatch: bool = True,
) -> WorldModelBackbone:
    """Build a WorldModelBackbone sized for a compiled schema.

    Args:
        bound_schema: BoundSchema from compile_schema / project().
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        d_ff: Feedforward dim. Default: 4 * d_model.
        n_loops: Looped attention iterations (3 is optimal).
        dropout: Dropout rate.
        use_dispatch: Whether to use per-connection attention dispatch.
            When True and the schema has a topology, each connection uses
            its declared attention function. When False, falls back to
            standard masked attention.

    Returns:
        WorldModelBackbone ready for training.
    """
    topology = None
    layout = None
    if use_dispatch and hasattr(bound_schema, 'topology') and bound_schema.topology is not None:
        topology = bound_schema.topology
        layout = bound_schema.layout

    return WorldModelBackbone(
        d_model=bound_schema.layout.d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        n_positions=bound_schema.layout.num_positions,
        dropout=dropout,
        n_loops=n_loops,
        topology=topology,
        layout=layout,
    )


class CogVideoXBackbone(nn.Module):
    """World model backbone grafted onto pretrained CogVideoX transformer.

    Loads pretrained CogVideoX transformer blocks and adds:
    - Projection layers (canvas d_model <-> CogVideoX inner_dim)
    - Per-block loop embeddings (zero-init for safe grafting)
    - Per-block gated residuals
    - Single learned conditioning token for encoder stream

    The frozen blocks provide spatiotemporal priors from video pretraining.
    Only loop parameters and projections are trainable (~0.1% of total).

    Interface matches WorldModelBackbone: (B, N, d_model) -> (B, N, d_model).

    Note:
        CogVideoX frozen blocks use full attention internally. Per-connection
        attention dispatch (Connection.fn / RegionSpec.default_attn) is NOT
        applied to frozen blocks — it requires a custom backbone that iterates
        over topology.attention_ops(). Use WorldModelBackbone with
        use_dispatch=True for per-connection attention dispatch from scratch,
        or use canvas_engineering.AttentionDispatcher directly in a custom
        backbone.
    """

    def __init__(
        self,
        transformer: nn.Module,
        d_model: int = 64,
        n_positions: int = 8192,
        n_loops: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        from canvas_engineering.cogvideox import detect_inner_dim
        self.inner_dim = detect_inner_dim(transformer)

        # Store blocks as plain list — NOT nn.Module children.
        # This keeps state_dict() clean (only trainable params) and avoids
        # moving shared frozen blocks when individual backbones move to CPU.
        self._frozen_blocks = list(transformer.transformer_blocks)
        n_blocks = len(self._frozen_blocks)

        for block in self._frozen_blocks:
            for p in block.parameters():
                p.requires_grad = False

        # Precompute temb from timestep 0 (neutral adaptive-norm conditioning)
        try:
            with torch.no_grad():
                device = next(transformer.parameters()).device
                dtype = next(transformer.parameters()).dtype
                t = torch.zeros(1, dtype=torch.long, device=device)
                t_proj = transformer.time_proj(t)
                base_temb = transformer.time_embedding(t_proj.to(dtype))
            self.register_buffer('_base_temb', base_temb.cpu())
        except (AttributeError, RuntimeError):
            block = self._frozen_blocks[0]
            if hasattr(block, 'scale_shift_table'):
                temb_dim = block.scale_shift_table.numel()
            else:
                temb_dim = 6 * self.inner_dim
            self.register_buffer('_base_temb', torch.zeros(1, temb_dim))

        # Canvas positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=n_positions)

        # Project canvas d_model <-> CogVideoX inner_dim
        self.proj_in = nn.Sequential(
            nn.Linear(d_model, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )
        self.proj_out = nn.Linear(self.inner_dim, d_model)
        self.final_norm = nn.LayerNorm(d_model)

        # Single learned conditioning token for the encoder stream.
        # CogVideoXBlock does joint attention over [encoder; hidden] and splits back,
        # so this token acts as a learnable global context for all canvas positions.
        self.encoder_cond = nn.Parameter(torch.randn(1, 1, self.inner_dim) * 0.02)

        # Per-block, per-loop trainable parameters (zero-init = identity at start)
        self.n_loops = n_loops
        self.loop_embs = nn.ParameterList([
            nn.Parameter(torch.zeros(n_loops, self.inner_dim))
            for _ in range(n_blocks)
        ])
        self.loop_embs_enc = nn.ParameterList([
            nn.Parameter(torch.zeros(n_loops, self.inner_dim))
            for _ in range(n_blocks)
        ])
        self.loop_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(n_loops, 1))
            for _ in range(n_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) canvas tensor.
            mask: Unused — CogVideoX blocks use full attention.
                Accepted for API compatibility with WorldModelBackbone.

        Returns:
            (B, N, d_model) output tensor.
        """
        B, N, _ = x.shape

        x = self.pos_enc(x)
        h = self.proj_in(x)  # (B, N, inner_dim)
        e = self.encoder_cond.expand(B, -1, -1)  # (B, 1, inner_dim)
        temb = self._base_temb.expand(B, -1)

        # Determine block dtype for mixed-precision casting
        block_dtype = next(self._frozen_blocks[0].parameters()).dtype

        for i, block in enumerate(self._frozen_blocks):
            h_in, e_in = h, e

            for l in range(self.n_loops):
                h_input = (h_in + self.loop_embs[i][l]).to(block_dtype)
                e_input = (e_in + self.loop_embs_enc[i][l]).to(block_dtype)
                h_out, e_out = block(
                    h_input, e_input, temb.to(block_dtype),
                )
                h_out, e_out = h_out.float(), e_out.float()

                gate = torch.sigmoid(self.loop_gates[i][l])
                h_in = gate * h_out + (1 - gate) * h_in
                e_in = gate * e_out + (1 - gate) * e_in

            h, e = h_in, e_in

        return self.final_norm(self.proj_out(h))

    def frozen_param_count(self) -> int:
        return sum(p.numel() for b in self._frozen_blocks for p in b.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_cogvideox_world_model(
    transformer: nn.Module,
    bound_schema,
    n_loops: int = 3,
    dropout: float = 0.1,
) -> CogVideoXBackbone:
    """Build a CogVideoXBackbone sized for a compiled schema.

    Args:
        transformer: Pretrained CogVideoX transformer (kept on its current device).
        bound_schema: BoundSchema from compile_schema / project().
        n_loops: Looped attention iterations.
        dropout: Dropout rate.

    Returns:
        CogVideoXBackbone with trainable loop params and projections.
    """
    return CogVideoXBackbone(
        transformer=transformer,
        d_model=bound_schema.layout.d_model,
        n_positions=bound_schema.layout.num_positions,
        n_loops=n_loops,
        dropout=dropout,
    )
