"""Inference: generate world model predictions.

Given a trained world model and partial observations, fill in the gaps.
This is the primary user-facing API for making predictions.

Usage:
    model = WorldModel.load("checkpoint.pt", projection)
    model.observe("financial.yield_curves.ten_year", 4.25)
    model.observe("country_us.macro.inflation.headline_cpi", 3.1)
    predictions = model.predict()

    # Access specific predictions
    recession_prob = predictions["forecasts.macro.recession_prob_3m"]
    regime = predictions["regime.growth_regime"]
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from general_unified_world_model.training.backbone import WorldModelBackbone, build_world_model
from general_unified_world_model.training.heterogeneous import FieldEncoder, FieldDecoder
from general_unified_world_model.training.diffusion import DiffusionWorldModelTrainer, CosineNoiseSchedule
from general_unified_world_model.projection.subset import WorldProjection, project


class WorldModel:
    """High-level interface for world model inference.

    Observe some fields, predict the rest. The model uses diffusion
    to generate coherent predictions conditioned on observations.
    """

    def __init__(
        self,
        bound_schema,
        backbone: WorldModelBackbone,
        encoder: FieldEncoder,
        decoder: FieldDecoder,
        device: str = "cuda",
    ):
        self.bound = bound_schema
        self.backbone = backbone.to(device).eval()
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.device = device

        self.n_positions = bound_schema.layout.num_positions
        self.d_model = bound_schema.layout.d_model

        # Diffusion infrastructure
        self.schedule = CosineNoiseSchedule()
        self.diffusion_trainer = DiffusionWorldModelTrainer(
            bound_schema, backbone, self.schedule, device=device,
        )

        # Current observations
        self._observations: dict[str, torch.Tensor] = {}

    def observe(self, field_path: str, value: float | list | torch.Tensor):
        """Set an observed value for a field.

        Args:
            field_path: Dotted path to field (e.g. "financial.yield_curves.ten_year").
            value: Observed value (scalar, list, or tensor).
        """
        if isinstance(value, (int, float)):
            value = torch.tensor([value], dtype=torch.float32)
        elif isinstance(value, list):
            value = torch.tensor(value, dtype=torch.float32)
        self._observations[field_path] = value

    def clear_observations(self):
        """Clear all observations."""
        self._observations.clear()

    def _build_conditioning(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build conditioning canvas and mask from current observations.

        Returns:
            (x_cond, cond_mask) — (1, N, d_model) and (1, N) tensors.
        """
        x_cond = torch.zeros(1, self.n_positions, self.d_model, device=self.device)
        cond_mask = torch.zeros(1, self.n_positions, device=self.device)

        for field_path, value in self._observations.items():
            try:
                bf = self.bound[field_path]
                indices = bf.indices()

                # Encode the value
                value = value.to(self.device).float()
                if value.dim() == 1:
                    value = value.unsqueeze(0).unsqueeze(0)  # (1, 1, raw_dim)
                encoded = self.encoder(field_path, value)  # (1, 1, d_model)

                # Place into canvas
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

        # Run reverse diffusion
        x_pred = self.diffusion_trainer.sample(x_cond, cond_mask, n_steps=n_steps)

        # Extract predictions for each field
        predictions = {}
        for field_name in self.bound.field_names:
            try:
                bf = self.bound[field_name]
                indices = bf.indices()
                field_latent = x_pred[0, indices]  # (n_pos, d_model)
                # Decode back to raw space
                decoded = self.decoder(field_name, field_latent.unsqueeze(0))
                predictions[field_name] = decoded.squeeze(0).cpu()
            except (KeyError, AttributeError):
                continue

        return predictions

    @torch.no_grad()
    def predict_field(self, field_path: str, n_steps: int = 50) -> torch.Tensor:
        """Predict a single field.

        Args:
            field_path: Dotted path to the target field.
            n_steps: Denoising steps.

        Returns:
            Predicted tensor for the field.
        """
        predictions = self.predict(n_steps=n_steps)
        if field_path in predictions:
            return predictions[field_path]
        raise KeyError(f"Field '{field_path}' not found in predictions")

    def save(self, path: str | Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "backbone": self.backbone.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        projection: WorldProjection,
        T: int = 1,
        H: int = 64,
        W: int = 64,
        d_model: int = 64,
        n_layers: int = 6,
        n_loops: int = 3,
        device: str = "cuda",
    ) -> WorldModel:
        """Load a trained world model from checkpoint.

        Args:
            path: Path to checkpoint file.
            projection: WorldProjection defining the schema.
            T, H, W, d_model: Canvas dimensions (must match training).
            n_layers, n_loops: Backbone architecture (must match training).
            device: Device.

        Returns:
            WorldModel ready for inference.
        """
        bound = project(projection, T=T, H=H, W=W, d_model=d_model)
        backbone = build_world_model(bound, n_layers=n_layers, n_loops=n_loops)
        encoder = FieldEncoder(bound)
        decoder = FieldDecoder(bound)

        checkpoint = torch.load(path, map_location=device, weights_only=True)
        backbone.load_state_dict(checkpoint["backbone"])
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])

        return cls(bound, backbone, encoder, decoder, device=device)
