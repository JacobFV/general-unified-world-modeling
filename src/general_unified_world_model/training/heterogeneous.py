"""Heterogeneous dataset training for the General Unified World Model.

The key insight: most datasets only cover a fraction of the world model's
fields. A GDP dataset doesn't have equity prices. A market dataset doesn't
have demographics. Traditional approaches either:
  (a) throw out data that's missing fields — wasteful
  (b) impute missing fields — introduces noise

Canvas engineering solves this via coarse-graining: you declare exactly
which nodes you want to model. Non-included sub-types simply don't exist
on the canvas — no positions, no attention, no loss. Their parent's
coarse-grained field still participates in attention, learning compressed
dynamics from whatever data IS available.

Each dataset maps its columns to fields on the canvas. The trainer places
data at the correct positions and computes loss only on positions that
have ground truth. Fields without data in the current batch get no
gradient — the topology handles what connects to what.

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

@dataclass
class InputSpec:
    """Declares an input modality that a dataset provides to the canvas.

    Args:
        key: Column/key name in the raw data dict.
        semantic_type: Natural language description of this modality,
            used for semantic conditioning (e.g. "10-year US Treasury yield, daily").
        field_path: Dotted path in the world model schema
            (e.g. "country_us.macro.output.gdp_nowcast").
        dtype: Data type hint for selecting default encoder.
        encoder: Custom encoder module. If None, uses default for dtype.
        region_size: Override canvas region size (number of positions).
        transform: Optional pre-processing transform on raw values.
        frequency: How often this field updates. None means every tick.
    """
    key: str
    semantic_type: str
    field_path: str
    dtype: str = "float32"
    encoder: Any = None
    region_size: Optional[int] = None
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    frequency: Optional[int] = None


@dataclass
class OutputSpec:
    """Declares an output modality — where the model predicts and receives gradient.

    Args:
        key: Column/key name for ground truth data.
        semantic_type: Natural language description of this modality.
        field_path: Dotted path in the world model schema.
        dtype: Data type hint for selecting default decoder.
        decoder: Custom decoder module. If None, uses default for dtype.
        loss_fn: Loss function name or callable. Default "mse".
        loss_weight: Relative weight for this output's loss.
        transform: Optional inverse transform on predictions.
        frequency: How often this field updates.
    """
    key: str
    semantic_type: str
    field_path: str
    dtype: str = "float32"
    decoder: Any = None
    loss_fn: str | Callable = "mse"
    loss_weight: float = 1.0
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    frequency: Optional[int] = None


# ── Default registries ──────────────────────────────────────────────────

DEFAULT_ENCODERS: dict[str, Callable[[int], nn.Module]] = {
    "float32": lambda d: nn.Sequential(nn.Linear(1, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)),
    "float64": lambda d: nn.Sequential(nn.Linear(1, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)),
    "int64": lambda d: nn.Sequential(nn.Linear(1, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)),
    "embedding": lambda d, dim=768: nn.Sequential(nn.Linear(dim, d), nn.LayerNorm(d)),
}

DEFAULT_DECODERS: dict[str, Callable[[int], nn.Module]] = {
    "float32": lambda d: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)),
    "float64": lambda d: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)),
    "int64": lambda d: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)),
}

DEFAULT_LOSS_FNS: dict[str, Callable] = {
    "mse": F.mse_loss,
    "l1": F.l1_loss,
    "huber": F.huber_loss,
}


def _infer_semantic_type(field_path: str) -> str:
    """Derive a human-readable semantic type from a dotted field path."""
    return field_path.replace("_", " ").replace(".", " > ")


@dataclass
class DatasetSpec:
    """Declares how a dataset maps onto the canvas.

    Provide input_specs and output_specs to define the mapping.
    """
    name: str
    description: str = ""
    input_specs: list[InputSpec] = dc_field(default_factory=list)
    output_specs: list[OutputSpec] = dc_field(default_factory=list)
    base_period: int = 1
    temporal_range: Optional[tuple[int, int]] = None
    weight: float = 1.0

    @property
    def all_field_paths(self) -> list[str]:
        """All unique field paths from both input and output specs."""
        paths = set()
        for s in self.input_specs:
            paths.add(s.field_path)
        for s in self.output_specs:
            paths.add(s.field_path)
        return sorted(paths)


# ── DataSource ────────────────────────────────────────────────────────────

@dataclass
class DataSource:
    """A dataset bound to its data: the spec (mapping) + actual tensor values.

    This is the standard exchange type for data throughout the pipeline.
    Adapters return DataSource, trainers consume DataSource.

    Usage::

        source = fred_adapter(...)
        source.spec    # the DatasetSpec
        source.data    # the tensor dict
    """
    spec: DatasetSpec
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def field_paths(self) -> list[str]:
        """All unique field paths this source covers."""
        return self.spec.all_field_paths

    def covers(self, bound_schema) -> "CoverageReport":
        """Check how well this source covers a BoundSchema's fields."""
        return check_coverage([self], bound_schema)


@dataclass
class CoverageReport:
    """Report on how well data sources cover a schema projection."""
    covered: set[str]
    missing: set[str]
    extra: set[str]
    coverage_ratio: float
    is_complete: bool

    def __str__(self) -> str:
        lines = [f"Coverage: {self.coverage_ratio:.0%} ({len(self.covered)}/{len(self.covered) + len(self.missing)} fields)"]
        if self.missing:
            lines.append(f"  Missing: {', '.join(sorted(self.missing)[:10])}")
            if len(self.missing) > 10:
                lines.append(f"    ... and {len(self.missing) - 10} more")
        return "\n".join(lines)


def check_coverage(
    data_sources: list[DataSource],
    bound_schema,
) -> CoverageReport:
    """Check how well data sources cover a BoundSchema's fields.

    Args:
        data_sources: List of DataSource objects.
        bound_schema: BoundSchema from project() / compile_schema().

    Returns:
        CoverageReport with covered/missing/extra field sets.
    """
    sources = data_sources

    # Gather all field paths declared by data sources
    source_paths = set()
    for ds in sources:
        for ispec in ds.spec.input_specs:
            source_paths.add(ispec.field_path)

    schema_paths = set(bound_schema.field_names)

    covered = source_paths & schema_paths
    missing = schema_paths - source_paths
    extra = source_paths - schema_paths

    total = len(schema_paths)
    ratio = len(covered) / total if total > 0 else 0.0

    return CoverageReport(
        covered=covered,
        missing=missing,
        extra=extra,
        coverage_ratio=ratio,
        is_complete=(len(missing) == 0),
    )


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
        sources: list[DataSource],
        seq_len: int = 1,
    ):
        """
        Args:
            bound_schema: BoundSchema from compile_schema / project()
            sources: List of DataSource objects.
            seq_len: Number of timesteps per training sample.
        """
        self.bound = bound_schema
        self.sources = list(sources)
        self.seq_len = seq_len
        self.d_model = bound_schema.layout.d_model
        self.n_positions = bound_schema.layout.num_positions

        # Set of field names for fast lookup during coarse-grained routing
        self._field_name_set = set(bound_schema.field_names)

        # Build dot→compiled name mapping for entity-prefixed fields.
        # compile_schema uses "__" for entity prefixes (e.g. "financial__yield_curves.ten_year")
        # but users write dotted paths (e.g. "financial.yield_curves.ten_year").
        self._dot_to_compiled: dict[str, str] = {}
        for name in bound_schema.field_names:
            dot_name = name.replace("__", ".")
            if dot_name != name:
                self._dot_to_compiled[dot_name] = name

        # Precompute field indices for each source
        self._source_indices = []
        for ds in sources:
            field_indices = {}
            for ispec in ds.spec.input_specs:
                target = ispec.field_path
                # Normalize: try compiled name if dot-path doesn't match directly
                if target in self._dot_to_compiled:
                    target = self._dot_to_compiled[target]
                try:
                    bf = bound_schema[target]
                    field_indices[ispec.key] = {
                        "bound_field": bf,
                        "indices": bf.indices(),
                        "transform": ispec.transform,
                        "frequency": ispec.frequency,
                        "is_coarse": False,
                    }
                except (KeyError, AttributeError):
                    # Field not in projection — try routing to a
                    # coarse-grained parent (e.g. "country_us.politics"
                    # covers "country_us.politics.executive_stability")
                    coarse = self._find_coarse_parent(target)
                    if coarse is not None:
                        try:
                            bf = bound_schema[coarse]
                            field_indices[ispec.key] = {
                                "bound_field": bf,
                                "indices": bf.indices(),
                                "transform": ispec.transform,
                                "frequency": ispec.frequency,
                                "is_coarse": True,
                            }
                        except (KeyError, AttributeError):
                            pass
            self._source_indices.append(field_indices)

        # Build sample index: (source_idx, time_offset) pairs
        self._samples = []
        for src_idx, ds in enumerate(self.sources):
            raw_data = ds.data
            if callable(raw_data):
                raw_data = raw_data()
                self.sources[src_idx] = DataSource(spec=ds.spec, data=raw_data)
            # Determine dataset length from first input spec's data
            n_rows = 0
            for ispec in ds.spec.input_specs:
                key = ispec.key
                if isinstance(raw_data, dict) and key in raw_data:
                    tensor = raw_data[key]
                    if isinstance(tensor, torch.Tensor) and tensor.dim() >= 1:
                        n_rows = max(n_rows, tensor.shape[0])
                    break

            for t in range(max(0, n_rows - seq_len + 1)):
                self._samples.append((src_idx, t))

    def _find_coarse_parent(self, target_field: str) -> str | None:
        """Walk up the field path to find a coarse-grained parent.

        If target_field is "country_us.politics.executive_stability"
        and the schema has "country_us.politics" (or "country_us__politics")
        as a 1×1 coarse-grained field, returns the compiled name.
        """
        parts = target_field.rsplit(".", 1)
        while len(parts) == 2:
            parent = parts[0]
            # Try direct match
            if parent in self._field_name_set:
                return parent
            # Try compiled name (dot → __)
            compiled = self._dot_to_compiled.get(parent)
            if compiled and compiled in self._field_name_set:
                return compiled
            parts = parent.rsplit(".", 1)
        return None

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        src_idx, t_offset = self._samples[idx]
        ds = self.sources[src_idx]
        raw_data = ds.data
        field_info = self._source_indices[src_idx]

        # Create empty canvas-shaped tensors
        canvas_data = torch.zeros(self.n_positions, self.d_model)
        presence_mask = torch.zeros(self.n_positions)
        # Multiple source fields may route to the same coarse-grained
        # position — accumulate values and average at the end.
        coarse_accum: dict[int, list[float]] = {}

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

            if info.get("is_coarse", False):
                # Coarse-grained: accumulate and average later
                value_flat = value.reshape(-1)
                for pos_idx in indices:
                    if pos_idx < self.n_positions:
                        mean_val = value_flat.mean().item()
                        coarse_accum.setdefault(pos_idx, []).append(mean_val)
            else:
                # Regular field: place directly
                value_flat = value.reshape(-1)
                for i, pos_idx in enumerate(indices):
                    if pos_idx < self.n_positions:
                        if i < len(value_flat):
                            canvas_data[pos_idx, 0] = value_flat[i]
                        presence_mask[pos_idx] = 1.0

        # Finalize coarse-grained positions: average all routed values
        for pos_idx, values in coarse_accum.items():
            canvas_data[pos_idx, 0] = sum(values) / len(values)
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
        conditioner: nn.Module | None = None,
    ):
        self.bound = bound_schema
        self.backbone = backbone.to(device)
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.optimizer = optimizer
        self.device = device
        self.conditioner = conditioner.to(device) if conditioner is not None else None

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

        # Apply semantic conditioning before backbone
        if self.conditioner is not None:
            canvas_data = self.conditioner.condition_canvas(canvas_data, self.bound.layout)

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
        clip_params = (
            list(self.backbone.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )
        if self.conditioner is not None:
            clip_params += list(self.conditioner.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "n_active_positions": int(n_active.item()),
            "coverage": float(n_active.item() / (presence_mask.numel() or 1)),
        }


# ── Multi-source DataLoader ─────────────────────────────────────────────

def build_mixed_dataloader(
    bound_schema,
    sources: list[DataSource],
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
        sources: List of DataSource objects.
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
