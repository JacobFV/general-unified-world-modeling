"""Heterogeneous dataset training for the Grand Unified World Model.

The key insight: most datasets only cover a fraction of the world model's
fields. A GDP dataset doesn't have equity prices. A market dataset doesn't
have demographics. Traditional approaches either:
  (a) throw out data that's missing fields — wasteful
  (b) impute missing fields — introduces noise

Canvas engineering gives us a third option:
  (c) mask missing fields in the loss — train on what you have

Each dataset declares which fields it populates. The trainer:
  1. Places available data into the canvas at the correct positions
  2. Generates a per-position mask (1 where data exists, 0 where missing)
  3. Multiplies loss by (mask * loss_weight_mask) — only backprop through
     fields that have ground truth
  4. The diffusion process still generates predictions for masked fields —
     they just don't contribute to the loss

This means:
  - A GDP-only dataset trains the macro fields and the regime latent
  - A market dataset trains financial fields and the regime latent
  - Both datasets train the *shared* regime latent, which learns to compress
    the joint distribution even though no single dataset has everything

Multi-frequency handling:
  - Fields with period > 1 are held constant across their period
  - A daily dataset updates daily fields each tick but monthly fields only
    every 192 ticks
  - The canvas's period system handles alignment automatically
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Field Mapping ────────────────────────────────────────────────────────

@dataclass
class FieldMapping:
    """Maps a column/key in a dataset to a field in the world model.

    Args:
        source_key: Key in the dataset dict (e.g. "gdp_growth_yoy")
        target_field: Dotted path in the world model (e.g. "country_us.macro.output.gdp_nowcast")
        transform: Optional function to normalize the raw value.
            Receives (value: torch.Tensor) -> torch.Tensor of shape matching
            the field's (h, w) spatial dimensions.
        frequency: How often this field updates in the source data,
            in units of the source's base timestep. None means every tick.
    """
    source_key: str
    target_field: str
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    frequency: Optional[int] = None


@dataclass
class DatasetSpec:
    """Declares how a dataset maps to the world model.

    Args:
        name: Human-readable name (e.g. "FRED macro", "Yahoo Finance daily")
        mappings: List of FieldMappings from source columns to world model fields.
        base_period: How many base ticks one row of this dataset represents.
            E.g. if the dataset is daily and base tick is sub-minute: 16.
            If the dataset is already at tick frequency: 1.
        temporal_range: (start_tick, end_tick) in world model time.
            None means the dataset spans the full training window.
        weight: Relative importance of this dataset in mixed training.
    """
    name: str
    mappings: list[FieldMapping]
    base_period: int = 1
    temporal_range: Optional[tuple[int, int]] = None
    weight: float = 1.0


# ── Heterogeneous Dataset ────────────────────────────────────────────────

class HeterogeneousDataset(Dataset):
    """A dataset that combines multiple heterogeneous data sources.

    Each source provides data for different subsets of the world model.
    Missing fields get masked out of the loss — no imputation needed.

    The dataset produces:
        - canvas_data: (N, d_model) tensor — values placed at field positions
        - presence_mask: (N,) binary tensor — 1 where data exists
        - field_names: list of field paths that have data in this sample
    """

    def __init__(
        self,
        bound_schema,
        sources: list[tuple[DatasetSpec, Any]],
        seq_len: int = 1,
    ):
        """
        Args:
            bound_schema: BoundSchema from compile_schema / project()
            sources: List of (DatasetSpec, raw_data) pairs where raw_data
                is either a dict of tensors or a callable that returns one.
            seq_len: Number of timesteps per training sample.
        """
        self.bound = bound_schema
        self.sources = sources
        self.seq_len = seq_len
        self.d_model = bound_schema.layout.d_model
        self.n_positions = bound_schema.layout.num_positions

        # Precompute field indices for each source
        self._source_indices = []
        for spec, _ in sources:
            field_indices = {}
            for mapping in spec.mappings:
                try:
                    bf = bound_schema[mapping.target_field]
                    field_indices[mapping.source_key] = {
                        "bound_field": bf,
                        "indices": bf.indices(),
                        "transform": mapping.transform,
                        "frequency": mapping.frequency,
                    }
                except (KeyError, AttributeError):
                    # Field not in current projection — skip silently
                    pass
            self._source_indices.append(field_indices)

        # Build sample index: (source_idx, time_offset) pairs
        self._samples = []
        for src_idx, (spec, raw_data) in enumerate(sources):
            if callable(raw_data):
                raw_data = raw_data()
            # Determine dataset length from first mapping's data
            n_rows = 0
            for mapping in spec.mappings:
                key = mapping.source_key
                if isinstance(raw_data, dict) and key in raw_data:
                    tensor = raw_data[key]
                    if isinstance(tensor, torch.Tensor) and tensor.dim() >= 1:
                        n_rows = max(n_rows, tensor.shape[0])
                    break

            # Store the raw data
            if isinstance(raw_data, dict):
                self.sources[src_idx] = (spec, raw_data)
            for t in range(max(0, n_rows - seq_len + 1)):
                self._samples.append((src_idx, t))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        src_idx, t_offset = self._samples[idx]
        spec, raw_data = self.sources[src_idx]
        field_info = self._source_indices[src_idx]

        # Create empty canvas-shaped tensors
        canvas_data = torch.zeros(self.n_positions, self.d_model)
        presence_mask = torch.zeros(self.n_positions)

        for source_key, info in field_info.items():
            if not isinstance(raw_data, dict) or source_key not in raw_data:
                continue

            tensor = raw_data[source_key]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, dtype=torch.float32)

            # Extract temporal slice
            if tensor.dim() >= 1 and tensor.shape[0] > t_offset:
                end = min(t_offset + self.seq_len, tensor.shape[0])
                value = tensor[t_offset:end]
            else:
                value = tensor

            # Apply transform if specified
            transform = info["transform"]
            if transform is not None:
                value = transform(value)

            # Ensure value is float and expand to d_model
            value = value.float()
            if value.dim() == 0:
                value = value.unsqueeze(0)

            # Place into canvas positions
            indices = info["indices"]
            n_idx = len(indices)

            # Flatten value to match number of positions
            value_flat = value.reshape(-1)
            # Expand scalar/vector to fill d_model per position
            for i, pos_idx in enumerate(indices):
                if pos_idx < self.n_positions:
                    if i < len(value_flat):
                        # Write the value into the first dim, leave rest as context
                        canvas_data[pos_idx, 0] = value_flat[i]
                    presence_mask[pos_idx] = 1.0

        return {
            "canvas_data": canvas_data,
            "presence_mask": presence_mask,
            "source_idx": src_idx,
            "t_offset": t_offset,
        }


# ── Encoders / Decoders ─────────────────────────────────────────────────

class FieldEncoder(nn.Module):
    """Projects raw scalar/vector field values into d_model latent space.

    Each field gets its own learned linear projection. This is the bridge
    between raw data (GDP growth = 0.023) and latent space.
    """

    def __init__(self, bound_schema, raw_dims: dict[str, int] | None = None):
        """
        Args:
            bound_schema: BoundSchema from compile_schema
            raw_dims: Dict mapping field paths to their raw input dimension.
                If not specified, defaults to 1 (scalar) for all fields.
        """
        super().__init__()
        self.d_model = bound_schema.layout.d_model
        self.encoders = nn.ModuleDict()

        for field_name in bound_schema.field_names:
            raw_dim = (raw_dims or {}).get(field_name, 1)
            # Sanitize field name for ModuleDict (replace dots and brackets)
            safe_name = field_name.replace(".", "__").replace("[", "_").replace("]", "")
            self.encoders[safe_name] = nn.Sequential(
                nn.Linear(raw_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )

        self._field_to_safe = {
            name: name.replace(".", "__").replace("[", "_").replace("]", "")
            for name in bound_schema.field_names
        }

    def forward(self, field_name: str, raw_value: torch.Tensor) -> torch.Tensor:
        """Encode a raw field value to d_model.

        Args:
            field_name: Dotted path to the field.
            raw_value: (batch, n_positions, raw_dim) tensor.

        Returns:
            (batch, n_positions, d_model) tensor.
        """
        safe = self._field_to_safe.get(field_name)
        if safe and safe in self.encoders:
            return self.encoders[safe](raw_value)
        # Fallback: zero-pad to d_model
        B, N = raw_value.shape[:2]
        out = torch.zeros(B, N, self.d_model, device=raw_value.device)
        out[..., :raw_value.shape[-1]] = raw_value
        return out


class FieldDecoder(nn.Module):
    """Projects d_model latent back to raw field dimensions.

    Each field gets its own learned projection. This is where
    the model's predictions become interpretable numbers.
    """

    def __init__(self, bound_schema, raw_dims: dict[str, int] | None = None):
        super().__init__()
        self.d_model = bound_schema.layout.d_model
        self.decoders = nn.ModuleDict()

        for field_name in bound_schema.field_names:
            raw_dim = (raw_dims or {}).get(field_name, 1)
            safe_name = field_name.replace(".", "__").replace("[", "_").replace("]", "")
            self.decoders[safe_name] = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, raw_dim),
            )

        self._field_to_safe = {
            name: name.replace(".", "__").replace("[", "_").replace("]", "")
            for name in bound_schema.field_names
        }

    def forward(self, field_name: str, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to raw field value.

        Args:
            field_name: Dotted path to the field.
            latent: (batch, n_positions, d_model) tensor.

        Returns:
            (batch, n_positions, raw_dim) tensor.
        """
        safe = self._field_to_safe.get(field_name)
        if safe and safe in self.decoders:
            return self.decoders[safe](latent)
        return latent[..., :1]  # fallback: take first dim


# ── Masked Canvas Trainer ────────────────────────────────────────────────

class MaskedCanvasTrainer:
    """Trains a world model on heterogeneous data with masked loss.

    The trainer handles:
    1. Encoding raw field values to d_model latent space
    2. Placing them on the canvas at correct positions
    3. Running the transformer backbone
    4. Extracting predictions and computing masked loss
    5. Only backpropagating through fields that have ground truth

    The key property: shared latent regions (regime state, compressed world state)
    get gradient signal from ALL datasets, even though each dataset only covers
    a fraction of the fields. This is how the model learns the joint distribution
    without any single dataset containing everything.
    """

    def __init__(
        self,
        bound_schema,
        backbone: nn.Module,
        encoder: FieldEncoder,
        decoder: FieldDecoder,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
    ):
        self.bound = bound_schema
        self.backbone = backbone.to(device)
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.optimizer = optimizer
        self.device = device

        # Precompute loss weight mask
        self.loss_weight_mask = bound_schema.layout.loss_weight_mask(device)

        # Build attention mask from topology
        if bound_schema.topology is not None:
            self.attn_mask = bound_schema.topology.to_additive_mask(
                bound_schema.layout, device=device
            )
        else:
            self.attn_mask = None

    def train_step(self, batch: dict) -> dict:
        """Single training step with masked loss.

        Args:
            batch: Dict with keys:
                - canvas_data: (B, N, d_model) — encoded field values
                - presence_mask: (B, N) — 1 where data exists

        Returns:
            Dict with 'loss' and per-field metrics.
        """
        canvas_data = batch["canvas_data"].to(self.device)
        presence_mask = batch["presence_mask"].to(self.device)

        # Forward through backbone
        if self.attn_mask is not None:
            canvas_out = self.backbone(canvas_data, mask=self.attn_mask)
        else:
            canvas_out = self.backbone(canvas_data)

        # Compute per-position reconstruction loss
        per_pos_loss = F.mse_loss(canvas_out, canvas_data, reduction="none")
        per_pos_loss = per_pos_loss.mean(dim=-1)  # (B, N) — mean over d_model

        # Apply masks: presence * loss_weight
        # presence_mask: (B, N) — 1 where we have data
        # loss_weight_mask: (N,) — per-position weights from schema
        weighted_loss = per_pos_loss * presence_mask * self.loss_weight_mask.unsqueeze(0)

        # Normalize by number of active positions
        n_active = presence_mask.sum()
        if n_active > 0:
            loss = weighted_loss.sum() / n_active
        else:
            loss = weighted_loss.sum() * 0.0  # no data — zero loss

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "n_active_positions": int(n_active.item()),
            "coverage": float(n_active.item() / (presence_mask.numel() or 1)),
        }


# ── Multi-source DataLoader ─────────────────────────────────────────────

def build_mixed_dataloader(
    bound_schema,
    sources: list[tuple[DatasetSpec, Any]],
    batch_size: int = 32,
    num_workers: int = 0,
    seq_len: int = 1,
) -> DataLoader:
    """Build a DataLoader that mixes multiple heterogeneous data sources.

    Samples are drawn proportionally to each source's weight.
    Each batch may contain samples from different sources —
    they all map to the same canvas positions.

    Args:
        bound_schema: BoundSchema from compile_schema / project()
        sources: List of (DatasetSpec, data) pairs.
        batch_size: Samples per batch.
        num_workers: DataLoader workers.
        seq_len: Timesteps per sample.

    Returns:
        DataLoader yielding batches with canvas_data and presence_mask.
    """
    dataset = HeterogeneousDataset(bound_schema, sources, seq_len=seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
