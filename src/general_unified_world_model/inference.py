"""Inference: general-purpose world model with dynamic canvas management.

WorldModel is the general base class for any canvas-based world model.
It holds runtime state (canvas tensor, backbone, encoder/decoder) and
supports dynamic layout and topology changes at runtime.

GeneralUnifiedWorldModel is a convenience subclass that bakes in the
857-field World() schema with built-in dataset configurations.

Usage:
    # General: any schema
    from canvas_engineering import Field, compile_schema
    bound = compile_schema(my_schema, T=1, d_model=64)
    model = WorldModel(bound)
    model.observe("sensor.temperature", 42.0)
    preds = model.predict()

    # Convenience: the 857-field world model
    model = GeneralUnifiedWorldModel(
        include=["financial", "country_us.macro", "regime", "forecasts"],
    )
    model.observe("financial.yield_curves.ten_year", 4.25)
    preds = model.predict()
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from general_unified_world_model.training.backbone import (
    WorldModelBackbone, build_world_model,
)
from general_unified_world_model.training.heterogeneous import (
    FieldEncoder, FieldDecoder, DatasetSpec, DataSource, InputSpec, OutputSpec,
    check_coverage, CoverageReport,
)
from general_unified_world_model.training.diffusion import (
    DiffusionWorldModelTrainer, CosineNoiseSchedule,
)
from general_unified_world_model.projection.subset import project


class WorldModel:
    """General-purpose world model over any canvas schema.

    Holds runtime state (canvas tensor, backbone, encoder/decoder).
    Supports dynamic layout changes, topology swaps, region management,
    and data ingestion from dataset specs.

    Can be constructed three ways:
        1. From a pre-compiled BoundSchema (power users)
        2. From a schema root + include/exclude (convenience)
        3. Via WorldModel.load() from a checkpoint

    The canvas tensor tracks the current world state. Observations
    write into it, predictions read from it.
    """

    def __init__(
        self,
        bound_schema,
        backbone: WorldModelBackbone | None = None,
        encoder: FieldEncoder | None = None,
        decoder: FieldDecoder | None = None,
        device: str = "cpu",
        *,
        n_layers: int = 6,
        n_heads: int = 4,
        n_loops: int = 3,
        dropout: float = 0.1,
        dataset_specs: list[DatasetSpec] | None = None,
    ):
        """Initialize a WorldModel.

        Args:
            bound_schema: Compiled BoundSchema from compile_schema/project().
            backbone: Pre-built backbone. If None, auto-built from schema.
            encoder: Pre-built field encoder. If None, auto-built.
            decoder: Pre-built field decoder. If None, auto-built.
            device: Device for tensors and model.
            n_layers: Backbone depth (used when auto-building).
            n_heads: Attention heads (used when auto-building).
            n_loops: Looped attention iterations (used when auto-building).
            dropout: Dropout rate.
            dataset_specs: Optional list of DatasetSpecs for ingest().
        """
        self.bound = bound_schema
        self.device = device

        self.n_positions = bound_schema.layout.num_positions
        self.d_model = bound_schema.layout.d_model

        # Store construction params for resize/rebuild
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._n_loops = n_loops
        self._dropout = dropout

        # Auto-build components if not provided
        if backbone is None:
            backbone = build_world_model(
                bound_schema, n_layers=n_layers, n_heads=n_heads,
                n_loops=n_loops, dropout=dropout,
            )
        if encoder is None:
            encoder = FieldEncoder(bound_schema)
        if decoder is None:
            decoder = FieldDecoder(bound_schema)

        self.backbone = backbone.to(device).eval()
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()

        # Diffusion infrastructure
        self.schedule = CosineNoiseSchedule()
        self.diffusion_trainer = DiffusionWorldModelTrainer(
            bound_schema, backbone, self.schedule, device=device,
        )

        # Runtime state: the live canvas tensor (1, N, d_model)
        self._canvas = torch.zeros(
            1, self.n_positions, self.d_model, device=device
        )

        # Current observations (field_path -> value)
        self._observations: dict[str, torch.Tensor] = {}

        # Data sources and dataset specs for ingest() / training
        self._data_sources: list[DataSource] = []
        self._dataset_specs: dict[str, DatasetSpec] = {}
        if dataset_specs:
            for spec in dataset_specs:
                self._dataset_specs[spec.name] = spec

    # ── Canvas access ────────────────────────────────────────────────

    def get_canvas(self, t: int | None = None) -> torch.Tensor:
        """Get the canvas state tensor.

        Args:
            t: Timestep index. If the canvas has T > 1, returns positions
                at timestep t. If None, returns the full canvas.

        Returns:
            (1, N_t, d_model) tensor of canvas positions.
        """
        if t is not None:
            layout = self.bound.layout
            H, W = layout.H, layout.W
            start = t * H * W
            end = start + H * W
            return self._canvas[:, start:end].clone()
        return self._canvas.clone()

    def set_canvas(self, canvas: torch.Tensor) -> None:
        """Set the canvas state tensor directly.

        Args:
            canvas: (1, N, d_model) or (N, d_model) tensor.
        """
        if canvas.dim() == 2:
            canvas = canvas.unsqueeze(0)
        self._canvas = canvas.to(self.device)

    # ── Observation / prediction ──────────────────────────────────────

    def observe(self, field_path: str, value: float | list | torch.Tensor,
                t: int | None = None):
        """Set an observed value for a field.

        Writes into both the observations dict (for diffusion conditioning)
        and the canvas tensor (for direct state access).

        Args:
            field_path: Dotted path to field.
            value: Observed value (scalar, list, or tensor).
            t: Timestep to write to. None = all timesteps.
        """
        if isinstance(value, (int, float)):
            value = torch.tensor([value], dtype=torch.float32)
        elif isinstance(value, list):
            value = torch.tensor(value, dtype=torch.float32)
        self._observations[field_path] = value

        # Also write into the canvas tensor
        try:
            bf = self.bound[field_path]
            indices = bf.indices()
            value_dev = value.to(self.device).float()
            if value_dev.dim() == 1:
                value_dev = value_dev.unsqueeze(0).unsqueeze(0)
            encoded = self.encoder(field_path, value_dev)
            for i, idx in enumerate(indices):
                if idx < self.n_positions and i < encoded.shape[1]:
                    self._canvas[0, idx] = encoded[0, i]
        except (KeyError, AttributeError):
            pass

    def clear_observations(self):
        """Clear all observations and reset canvas to zeros."""
        self._observations.clear()
        self._canvas.zero_()

    def _build_conditioning(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build conditioning canvas and mask from current observations."""
        x_cond = torch.zeros(1, self.n_positions, self.d_model, device=self.device)
        cond_mask = torch.zeros(1, self.n_positions, device=self.device)

        for field_path, value in self._observations.items():
            try:
                bf = self.bound[field_path]
                indices = bf.indices()
                value_dev = value.to(self.device).float()
                if value_dev.dim() == 1:
                    value_dev = value_dev.unsqueeze(0).unsqueeze(0)
                encoded = self.encoder(field_path, value_dev)
                for i, idx in enumerate(indices):
                    if idx < self.n_positions:
                        if i < encoded.shape[1]:
                            x_cond[0, idx] = encoded[0, i]
                        cond_mask[0, idx] = 1.0
            except (KeyError, AttributeError):
                continue

        return x_cond, cond_mask

    @torch.no_grad()
    def predict(self, n_steps: int = 50) -> dict[str, torch.Tensor]:
        """Generate predictions for all unobserved fields.

        Uses reverse diffusion conditioned on observations.

        Args:
            n_steps: Number of denoising steps (more = better quality).

        Returns:
            Dict mapping field paths to predicted values.
        """
        x_cond, cond_mask = self._build_conditioning()
        x_pred = self.diffusion_trainer.sample(x_cond, cond_mask, n_steps=n_steps)

        predictions = {}
        for field_name in self.bound.field_names:
            try:
                bf = self.bound[field_name]
                indices = bf.indices()
                field_latent = x_pred[0, indices]
                decoded = self.decoder(field_name, field_latent.unsqueeze(0))
                predictions[field_name] = decoded.squeeze(0).cpu()
            except (KeyError, AttributeError):
                continue

        return predictions

    @torch.no_grad()
    def predict_field(self, field_path: str, n_steps: int = 50) -> torch.Tensor:
        """Predict a single field."""
        predictions = self.predict(n_steps=n_steps)
        if field_path in predictions:
            return predictions[field_path]
        raise KeyError(f"Field '{field_path}' not found in predictions")

    # ── Dynamic layout / topology ─────────────────────────────────────

    def resize_layout(
        self,
        H: int | None = None,
        W: int | None = None,
        T: int | None = None,
        d_model: int | None = None,
    ) -> None:
        """Change canvas dimensions. Zero-initializes new positions.

        Recompiles the schema with new dimensions. Existing field data is
        preserved where possible by matching field names between old and
        new layouts.

        Args:
            H, W: New spatial dimensions. None = keep current.
            T: New temporal extent. None = keep current.
            d_model: New latent dim. None = keep current.
        """
        old_layout = self.bound.layout
        new_H = H if H is not None else old_layout.H
        new_W = W if W is not None else old_layout.W
        new_T = T if T is not None else old_layout.T
        new_d = d_model if d_model is not None else old_layout.d_model

        # Save old canvas data by field name
        old_data = {}
        for name in self.bound.field_names:
            try:
                bf = self.bound[name]
                indices = bf.indices()
                old_data[name] = self._canvas[0, indices].clone()
            except (KeyError, AttributeError):
                continue

        # Recompile with new dimensions
        # We need to rebuild the projected dataclass — store the topology
        old_topology = self.bound.topology

        # Rebuild by re-running compile_schema on the same projected root
        # but with new dimensions. Since BoundSchema doesn't store the
        # original root, we construct a new schema from the existing
        # field layout (same regions, new bounds).
        from canvas_engineering import CanvasLayout, compile_schema
        new_bound = self._recompile(new_T, new_H, new_W, new_d)
        if new_bound is None:
            return

        # Build new canvas and transfer data
        new_n = new_bound.layout.num_positions
        new_canvas = torch.zeros(1, new_n, new_d, device=self.device)

        for name in new_bound.field_names:
            if name in old_data:
                try:
                    new_bf = new_bound[name]
                    new_indices = new_bf.indices()
                    old_vals = old_data[name]
                    # Transfer min(old_size, new_size) positions
                    n_transfer = min(len(new_indices), old_vals.shape[0])
                    if new_d == old_layout.d_model:
                        new_canvas[0, new_indices[:n_transfer]] = old_vals[:n_transfer]
                    # If d_model changed, zero-init (can't transfer different dims)
                except (KeyError, AttributeError):
                    continue

        self.bound = new_bound
        self.n_positions = new_n
        self.d_model = new_d
        self._canvas = new_canvas

        # Rebuild model components
        self.backbone = build_world_model(
            new_bound, n_layers=self._n_layers, n_heads=self._n_heads,
            n_loops=self._n_loops, dropout=self._dropout,
        ).to(self.device).eval()
        self.encoder = FieldEncoder(new_bound).to(self.device).eval()
        self.decoder = FieldDecoder(new_bound).to(self.device).eval()
        self.diffusion_trainer = DiffusionWorldModelTrainer(
            new_bound, self.backbone, self.schedule, device=self.device,
        )

    def _recompile(self, T, H, W, d_model):
        """Recompile the schema with new dimensions. Override in subclass."""
        # Base class can't recompile without the original schema root.
        # Subclasses (GeneralUnifiedWorldModel) store the root and can.
        return None

    def set_topology(self, topology) -> None:
        """Change the attention topology and rebuild backbone dispatchers.

        The layout stays the same — only the connectivity graph changes.
        This rebuilds attention dispatchers in each backbone block.

        Args:
            topology: New CanvasTopology.
        """
        from canvas_engineering.dispatch import AttentionDispatcher

        # Update the bound schema's topology
        self.bound.schema.topology = topology
        # Also update the BoundSchema's topology reference if it caches it
        if hasattr(self.bound, '_topology'):
            self.bound._topology = topology

        # Rebuild dispatchers in backbone blocks
        if hasattr(self.backbone, 'blocks'):
            layout = self.bound.layout
            for block in self.backbone.blocks:
                if hasattr(block, 'dispatcher') and block.dispatcher is not None:
                    new_dispatcher = AttentionDispatcher(
                        topology, layout, self.d_model,
                        self._n_heads, self._dropout,
                    )
                    block.dispatcher = new_dispatcher

    def add_region(self, name: str, spec) -> None:
        """Add a new region to the canvas. Zero-initializes positions.

        Args:
            name: Region name.
            spec: RegionSpec for the new region.
        """
        layout = self.bound.layout
        if name in layout.regions:
            return  # Already exists

        # Add to layout regions dict
        layout.regions[name] = spec

        # Expand canvas if needed
        indices = layout.region_indices(name)
        max_idx = max(indices) if indices else 0
        if max_idx >= self._canvas.shape[1]:
            # Extend canvas tensor
            extra = max_idx - self._canvas.shape[1] + 1
            padding = torch.zeros(1, extra, self.d_model, device=self.device)
            self._canvas = torch.cat([self._canvas, padding], dim=1)
            self.n_positions = self._canvas.shape[1]

    def remove_region(self, name: str) -> None:
        """Remove a region from the canvas.

        The positions are zeroed out but the canvas size doesn't change
        (other regions may share the spatial extent).

        Args:
            name: Region name to remove.
        """
        layout = self.bound.layout
        if name not in layout.regions:
            return

        # Zero out positions
        indices = layout.region_indices(name)
        idx = torch.tensor(indices, device=self.device, dtype=torch.long)
        self._canvas[0, idx] = 0.0

        # Remove from layout
        del layout.regions[name]

    # ── Data ingestion ────────────────────────────────────────────────

    def ingest(self, data: dict[str, Any] | DataSource,
               spec: DatasetSpec | str | None = None):
        """Populate canvas from a data dict or DataSource.

        Args:
            data: Dict mapping column names to values, or a DataSource.
            spec: DatasetSpec, or name of a registered spec, or None to
                auto-match keys to field paths. Ignored if data is a DataSource.
        """
        if isinstance(data, DataSource):
            spec = data.spec
            data = data.data

        if isinstance(spec, str):
            spec = self._dataset_specs.get(spec)
        if spec is None:
            # Try to auto-match data keys to field paths
            for key, value in data.items():
                self.observe(key, value)
            return

        # Use input specs to map data to canvas
        for input_spec in spec.input_specs:
            if input_spec.key in data:
                value = data[input_spec.key]
                if input_spec.transform is not None:
                    if isinstance(value, torch.Tensor):
                        value = input_spec.transform(value)
                    else:
                        value = input_spec.transform(torch.tensor(value, dtype=torch.float32))
                self.observe(input_spec.field_path, value)

    def add_data(self, source: DataSource):
        """Add a data source for training and ingestion.

        Args:
            source: A DataSource object.
        """
        self._data_sources.append(source)
        self._dataset_specs[source.name] = source.spec

    def register_dataset(self, spec: DatasetSpec):
        """Register a DatasetSpec for use with ingest()."""
        self._dataset_specs[spec.name] = spec

    def check_coverage(self) -> CoverageReport:
        """Check how well registered data sources cover this model's fields."""
        return check_coverage(self._data_sources, self.bound)

    # ── Projection ────────────────────────────────────────────────────

    def project_subset(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> "WorldModel":
        """Create a new WorldModel from a subset of this model's fields.

        Copies relevant weights from self where field names match.

        Args:
            include: Dotted paths to include.
            exclude: Dotted paths to exclude.

        Returns:
            New WorldModel with the subset projection.
        """
        # This requires the original schema root, which only
        # GeneralUnifiedWorldModel stores. Base class raises.
        raise NotImplementedError(
            "project_subset() requires the original schema root. "
            "Use GeneralUnifiedWorldModel or construct a new BoundSchema manually."
        )

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str | Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "backbone": self.backbone.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "canvas": self._canvas.cpu(),
            "observations": {k: v.cpu() for k, v in self._observations.items()},
        }, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        bound_schema=None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        entities: dict[str, Any] | None = None,
        connectivity=None,
        T: int = 1,
        H: int = 64,
        W: int = 64,
        d_model: int = 64,
        n_layers: int = 6,
        n_loops: int = 3,
        device: str = "cpu",
    ) -> "WorldModel":
        """Load a trained world model from checkpoint.

        Args:
            path: Path to checkpoint file.
            bound_schema: Pre-compiled BoundSchema. Takes precedence.
            include: Dotted paths to include (used if bound_schema is None).
            exclude: Dotted paths to exclude.
            entities: Dynamic entity dict.
            connectivity: Override connectivity policy.
            T, H, W, d_model: Canvas dimensions (must match training).
            n_layers, n_loops: Backbone architecture (must match training).
            device: Device.

        Returns:
            WorldModel ready for inference.
        """
        if bound_schema is None:
            if include is None:
                raise ValueError("Provide either bound_schema or include paths")
            bound_schema = project(
                include=include, exclude=exclude, entities=entities,
                connectivity=connectivity, T=T, H=H, W=W, d_model=d_model,
            )

        backbone = build_world_model(
            bound_schema, n_layers=n_layers, n_loops=n_loops,
        )
        encoder = FieldEncoder(bound_schema)
        decoder = FieldDecoder(bound_schema)

        checkpoint = torch.load(path, map_location=device, weights_only=True)
        backbone.load_state_dict(checkpoint["backbone"])
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])

        model = cls(bound_schema, backbone, encoder, decoder, device=device)

        # Restore canvas and observations if saved
        if "canvas" in checkpoint:
            model._canvas = checkpoint["canvas"].to(device)
        if "observations" in checkpoint:
            model._observations = {
                k: v.to(device) for k, v in checkpoint["observations"].items()
            }

        return model

    @classmethod
    def from_schema(
        cls,
        schema_root,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        entities: dict[str, Any] | None = None,
        T: int = 1,
        H: int | None = None,
        W: int | None = None,
        d_model: int = 64,
        device: str = "cpu",
        **kwargs,
    ) -> "WorldModel":
        """Create a WorldModel from any schema root.

        Args:
            schema_root: Any dataclass with Field attributes.
            include: Dotted paths to include. Default: all.
            exclude: Dotted paths to exclude.
            entities: Dynamic entity dict.
            T, H, W, d_model: Canvas dimensions.
            device: Device.
            **kwargs: Passed to WorldModel constructor.

        Returns:
            WorldModel ready for use.
        """
        bound = project(
            schema_root,
            include=include,
            exclude=exclude,
            entities=entities,
            T=T, H=H, W=W, d_model=d_model,
        )
        return cls(bound, device=device, **kwargs)


class GeneralUnifiedWorldModel(WorldModel):
    """The 857-field world model with built-in schema and dataset support.

    Convenience subclass of WorldModel that bakes in the World() schema.
    Just pass include/exclude to select which domains to model.

    Usage:
        model = GeneralUnifiedWorldModel(
            include=["financial", "country_us.macro", "regime", "forecasts"],
            d_model=64,
        )
        model.observe("financial.yield_curves.ten_year", 4.25)
        predictions = model.predict()
    """

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        entities: dict[str, Any] | None = None,
        T: int = 1,
        H: int | None = None,
        W: int | None = None,
        d_model: int = 64,
        device: str = "cpu",
        data_sources: list[DataSource] | None = None,
        dataset_specs: list[DatasetSpec] | None = None,
        connectivity: "ConnectivityPolicy | None" = None,
        **kwargs,
    ):
        """Initialize the General Unified World Model.

        Args:
            include: Which schema paths to include. Default: all 857 fields.
            exclude: Paths to exclude.
            entities: Dynamic entities (e.g. {"firm_AAPL": Business()}).
            T: Temporal extent.
            H, W: Canvas size. None = auto-sized.
            d_model: Latent dimension.
            device: Device.
            data_sources: Data sources for training/ingestion.
            dataset_specs: Additional dataset specs (legacy, prefer data_sources).
            connectivity: Override connectivity policy.
            **kwargs: Passed to WorldModel (n_layers, n_heads, etc.).
        """
        from general_unified_world_model.schema.world import World

        self._schema_root = World()
        self._include = include or ["*"]
        self._exclude = exclude or []
        self._entities = entities or {}
        self._connectivity = connectivity

        bound = project(
            self._schema_root,
            include=self._include,
            exclude=self._exclude,
            entities=self._entities,
            T=T, H=H, W=W, d_model=d_model,
            connectivity=connectivity,
        )

        super().__init__(
            bound,
            device=device,
            dataset_specs=dataset_specs,
            **kwargs,
        )

        if data_sources:
            for ds in data_sources:
                self.add_data(ds)

    def _recompile(self, T, H, W, d_model):
        """Recompile with the stored World schema and projection params."""
        return project(
            self._schema_root,
            include=self._include,
            exclude=self._exclude,
            entities=self._entities,
            T=T, H=H, W=W, d_model=d_model,
            connectivity=self._connectivity,
        )

    def project_subset(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        entities: dict[str, Any] | None = None,
    ) -> "GeneralUnifiedWorldModel":
        """Create a new GeneralUnifiedWorldModel from a subset.

        Args:
            include: Override include paths. Default: use current.
            exclude: Override exclude paths.
            entities: Override entities.

        Returns:
            New GeneralUnifiedWorldModel with the subset projection.
        """
        return GeneralUnifiedWorldModel(
            include=include or self._include,
            exclude=exclude or self._exclude,
            entities=entities or self._entities,
            T=self.bound.layout.T,
            d_model=self.d_model,
            device=self.device,
            n_layers=self._n_layers,
            n_heads=self._n_heads,
            n_loops=self._n_loops,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        entities: dict[str, Any] | None = None,
        T: int = 1,
        H: int | None = None,
        W: int | None = None,
        d_model: int = 64,
        n_layers: int = 6,
        n_loops: int = 3,
        device: str = "cpu",
        **kwargs,
    ) -> "GeneralUnifiedWorldModel":
        """Load a trained GeneralUnifiedWorldModel from checkpoint.

        Args:
            path: Path to checkpoint file.
            include: Schema paths to include.
            exclude: Paths to exclude.
            entities: Dynamic entities.
            T, H, W, d_model: Canvas dimensions (must match training).
            n_layers, n_loops: Architecture (must match training).
            device: Device.

        Returns:
            GeneralUnifiedWorldModel ready for inference.
        """
        model = cls(
            include=include, exclude=exclude, entities=entities,
            T=T, H=H, W=W, d_model=d_model, device=device,
            n_layers=n_layers, n_loops=n_loops, **kwargs,
        )

        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.backbone.load_state_dict(checkpoint["backbone"])
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.decoder.load_state_dict(checkpoint["decoder"])

        if "canvas" in checkpoint:
            model._canvas = checkpoint["canvas"].to(device)
        if "observations" in checkpoint:
            model._observations = {
                k: v.to(device) for k, v in checkpoint["observations"].items()
            }

        return model
